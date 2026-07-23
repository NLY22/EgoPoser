"""
EgoPoser → Unity 桥接 (FK位置驱动)
发送22个关节位置, Unity做可视化验证
"""
import os, sys, time, struct, socket
import numpy as np
import torch
from collections import deque

EGOPOSER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EGOPOSER_DIR)

from utils import utils_option as option
from models.select_model import define_Model, define_bm
from utils import utils_transform
from human_body_prior.tools.rotation_tools import aa2matrot, local2global_pose

# ===== 网络配置 =====
# 本地测试: HOST = '127.0.0.1'
# 局域网/Vision Pro: HOST = '0.0.0.0' (监听所有网卡)
HOST = '0.0.0.0'
PORT = 8888
YAML_PATH = os.path.join(EGOPOSER_DIR, 'options/test_egoposer.yaml')
# 选择测试数据文件
DATA_FILE = 'support_data/github_data/dmpl_sample.npz'    # DMPL 样本 (有动作)
# DATA_FILE = 'support_data/github_data/amass_sample.npz'   # AMASS 样本 (走路)
WINDOW_SIZE = 80

# ===== 模式开关 =====
# True  = 直接输出 AMASS 原始动作 (ground truth)，不经过 EgoPoser 推理
# False = EgoPoser 模型推理 (默认)
RAW_MODE = False


def load_model(yaml_path=YAML_PATH):
    opt = option.parse(yaml_path, is_train=True)
    opt['path']['pretrained'] = opt['pretrained_model']
    opt = option.dict_to_nonedict(opt)
    model = define_Model(opt)
    model.load(test=True)
    model.net.eval()
    return model


def load_amass_sample():
    data = np.load(DATA_FILE, allow_pickle=True)
    poses = torch.tensor(data['poses'])
    trans = torch.tensor(data['trans'])
    return poses, trans


def process_amass_to_input(poses, trans, bm, device):
    poses = poses.to(device).float()
    trans = trans.to(device).float()
    n_frames = min(600, poses.shape[0])
    poses = poses[:n_frames]
    trans = trans[:n_frames]

    pose_aa = poses[:, :66].reshape(-1, 3)
    pose_6d = utils_transform.aa2sixd(pose_aa).reshape(n_frames, -1)
    pose_matrot = aa2matrot(poses.reshape(-1, 3)).reshape(n_frames, -1, 9)
    rot_global = local2global_pose(pose_matrot, bm.kintree_table[0].long())

    head_rot = rot_global[:, [15], :, :]
    lhand_rot = rot_global[:, [20], :, :]
    rhand_rot = rot_global[:, [21], :, :]

    head_6d = utils_transform.matrot2sixd(head_rot.reshape(-1, 3, 3)).reshape(n_frames, 6)
    lhand_6d = utils_transform.matrot2sixd(lhand_rot.reshape(-1, 3, 3)).reshape(n_frames, 6)
    rhand_6d = utils_transform.matrot2sixd(rhand_rot.reshape(-1, 3, 3)).reshape(n_frames, 6)

    body = bm(**{'pose_body': poses[:, 3:66], 'root_orient': poses[:, :3], 'trans': trans})
    jpos = body.Jtr[:, :22, :]
    head_pos = jpos[:, 15, :]
    lhand_pos = jpos[:, 20, :]
    rhand_pos = jpos[:, 21, :]

    head_vel = torch.cat([torch.zeros(1, 6, device=device), head_6d[1:] - head_6d[:-1]], dim=0)
    lhand_vel = torch.cat([torch.zeros(1, 6, device=device), lhand_6d[1:] - lhand_6d[:-1]], dim=0)
    rhand_vel = torch.cat([torch.zeros(1, 6, device=device), rhand_6d[1:] - rhand_6d[:-1]], dim=0)

    zero3 = torch.zeros(n_frames, 3, device=device)
    return torch.cat([
        head_6d, lhand_6d, rhand_6d,
        head_vel, lhand_vel, rhand_vel,
        head_pos, lhand_pos, rhand_pos,
        zero3, zero3, zero3,
    ], dim=-1).float()


def main():
    print("=" * 50)
    print("  EgoPoser → Unity (FK位置驱动)")
    print("=" * 50)

    model = load_model(YAML_PATH)
    opt = option.parse(YAML_PATH, is_train=True)
    opt = option.dict_to_nonedict(opt)
    bm_dict = define_bm(opt)
    bm = bm_dict['male']

    poses, trans = load_amass_sample()
    input_data = process_amass_to_input(poses, trans, bm, model.device)
    print(f"[AMASS] 输入: {input_data.shape}")

    # 提取原始 AMASS 关节位置 (GT, 用于 RAW_MODE)
    poses_gpu = poses.to(model.device).float()
    trans_gpu = trans.to(model.device).float()
    n_frames = min(600, poses_gpu.shape[0])
    body_gt = bm(**{'pose_body': poses_gpu[:n_frames, 3:66],
                    'root_orient': poses_gpu[:n_frames, :3],
                    'trans': trans_gpu[:n_frames]})
    gt_joints = body_gt.Jtr[:, :22, :].cpu().numpy()  # [N, 22, 3], SMPL 空间

    # 坐标转换辅助函数
    def smpl_to_unity(jpos):
        jp = np.zeros_like(jpos)
        jp[:, 0] = jpos[:, 0]
        jp[:, 1] = jpos[:, 2]
        jp[:, 2] = -jpos[:, 1]
        jp[:, 0] += 0.2
        jp[:, 1] -= 0.1
        return jp

    mode_name = "RAW (原始AMASS动作)" if RAW_MODE else "EgoPoser 推理"
    print(f"[模式] {mode_name}")

    if not RAW_MODE:
        infer = EgoPoserInference(model)
        infer.prefill(input_data[0].cpu().numpy())
    # 提取输入数据中的头部位置 (54维输入的索引36:39)
    input_head_pos = input_data[:, 36:39].cpu().numpy()  # [600, 3], SMPL空间

    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    print(f"[网络] 等待 Unity 连接 {HOST}:{PORT} ...")
    conn, addr = server.accept()
    print(f"[网络] ✅ Unity 已连接: {addr}")

    frame_idx = 0
    frame_count = 0
    t_start = time.time()

    try:
        while True:
            if frame_idx >= len(input_data):
                frame_idx = 0

            if RAW_MODE:
                # 直接输出原始 AMASS 关节位置
                joint_pos = smpl_to_unity(gt_joints[frame_idx])
                frame_idx += 1
            else:
                # EgoPoser 推理
                sparse_frame = input_data[frame_idx].cpu().numpy()
                frame_idx += 1
                result = infer.step_pos(sparse_frame, input_head_pos[frame_idx - 1])
                if result is None:
                    time.sleep(1.0 / 60)
                    continue
                joint_pos = result

            # 发送 22个关节位置 (264字节)
            data = joint_pos.astype(np.float32).tobytes()
            conn.sendall(struct.pack('I', len(data)) + data)

            frame_count += 1
            if frame_count == 1:
                print(f"[调试] 第1帧 Pelvis: {joint_pos[0]}")
                print(f"[调试] 第1帧 Head:   {joint_pos[15]}")

            if frame_count % 100 == 0:
                print(f"[发送] {frame_count} 帧")

            sleep_time = 1.0/60 - (time.time() - t_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
            t_start = time.time()

    except (BrokenPipeError, ConnectionResetError) as e:
        print(f"[网络] ❌ {e}")
    except KeyboardInterrupt:
        print("\n[系统] 用户中断")
    finally:
        conn.close(); server.close()
        print(f"[系统] 共发送 {frame_count} 帧")


class EgoPoserInference:
    def __init__(self, model):
        self.model = model
        self.device = model.device
        self.window_size = WINDOW_SIZE
        self.sparse_buffer = deque(maxlen=WINDOW_SIZE)
        self.fov_l_buffer = deque(maxlen=WINDOW_SIZE)
        self.fov_r_buffer = deque(maxlen=WINDOW_SIZE)

    def prefill(self, frame):
        for _ in range(WINDOW_SIZE):
            self.sparse_buffer.append(frame.copy())
            self.fov_l_buffer.append(True)
            self.fov_r_buffer.append(True)

    def step_pos(self, sparse_frame, input_head_pos=None, fov_l=True, fov_r=True):
        self.sparse_buffer.append(sparse_frame)
        self.fov_l_buffer.append(fov_l)
        self.fov_r_buffer.append(fov_r)
        if len(self.sparse_buffer) < self.window_size:
            return None

        sparse = torch.FloatTensor(np.array(self.sparse_buffer)).unsqueeze(0).to(self.device)
        fov_l_t = torch.BoolTensor(np.array(self.fov_l_buffer)).unsqueeze(0)
        fov_r_t = torch.BoolTensor(np.array(self.fov_r_buffer)).unsqueeze(0)

        x = {'sparse_input': sparse, 'fov_l': fov_l_t, 'fov_r': fov_r_t}
        with torch.no_grad():
            output = self.model.net(x)

            root_orient_6d = output['root_orient']  # [1, 6]
            pose_body_6d = output['pose_body']      # [1, 126]

            # 6D → 轴角 (与原项目 test() 一致)
            root_orient_aa = utils_transform.sixd2aa(root_orient_6d.reshape(-1,6)).reshape(-1,3).float()
            pose_body_aa = utils_transform.sixd2aa(pose_body_6d.reshape(-1,6)).reshape(-1,63).float()

            # 1. 无位移身体, 获取头部相对于骨盆的位置
            body_local = self.model.bm(**{'pose_body': pose_body_aa, 'root_orient': root_orient_aa})
            t_head2root = body_local.Jtr[0, 15].cpu().numpy()

            # 2. 计算骨盆位移
            if input_head_pos is not None:
                t_root2world = -t_head2root + input_head_pos
            else:
                t_root2world = np.zeros(3)

            # 3. 带位移的完整身体
            t_tensor = torch.tensor(t_root2world, dtype=torch.float32, device=self.device).unsqueeze(0)
            body_pose = self.model.bm(**{
                'pose_body': pose_body_aa, 'root_orient': root_orient_aa,
                'trans': t_tensor, 'betas': output.get('betas', None)
            })
            joint_pos = body_pose.Jtr[0, :22].cpu().numpy()  # [22, 3]

        # 坐标转换
        jp_unity = np.zeros_like(joint_pos)
        jp_unity[:, 0] = joint_pos[:, 0]   # SMPL X → Unity X
        jp_unity[:, 1] = joint_pos[:, 2]   # SMPL Z → Unity Y
        jp_unity[:, 2] = -joint_pos[:, 1]  # -SMPL Y → Unity Z
        jp_unity[:, 0] += 0.2              # X偏移
        jp_unity[:, 1] -= 0.1              # Y偏移
        return jp_unity  # (22, 3)


if __name__ == '__main__':
    main()