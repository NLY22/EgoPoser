"""
EgoPoser WebSocket 服务器
接收 Unity 端的 trackers 数据 (head + 双手腕)，实时 EgoPoser 推理，返回 body_pose

协议: WebSocket + JSON
端口: 8888

输入 (Unity -> Server, ~60Hz):
{
    "type": "trackers",
    "sequence": 123,
    "timestamp": 12.345678,
    "head":         {"position": [x,y,z], "rotation": [x,y,z,w], "tracked": true},
    "left_wrist":   {"position": [x,y,z], "rotation": [x,y,z,w], "tracked": true},
    "right_wrist":  {"position": [x,y,z], "rotation": [x,y,z,w], "tracked": true}
}

输出 (Server -> Unity):
{
    "type": "body_pose",
    "sequence": 123,
    "inference_ms": 6.8,
    "root_position": [x,y,z],
    "root_rotation": [x,y,z,w],
    "joints_world": [[x,y,z], ...22个]
}

暖机阶段 (80 帧未满):
{
    "type": "warming_up",
    "sequence": 123,
    "frames_collected": M,
    "frames_needed": 80
}

内部 54 维 sparse_input 布局 (与 EgoPoser 原项目 process_amass_to_input 完全一致):
    [0:6]   head_6d   (旋转矩阵前两列)
    [6:12]  lhand_6d
    [12:18] rhand_6d
    [18:24] head_vel  (6D 直接差分: head_6d[t] - head_6d[t-1])
    [24:30] lhand_vel (同上)
    [30:36] rhand_vel (同上)
    [36:39] head_pos  (当前位置)
    [39:42] lhand_pos
    [42:45] rhand_pos
    [45:48] zero      (原项目固定 0)
    [48:51] zero
    [51:54] zero

窗口:
    sparse_input: Float32 [1, 80, 54]
    fov_l: Bool [1, 80]
    fov_r: Bool [1, 80]
"""

import os
import sys
import json
import time
import asyncio
import numpy as np
import torch
from collections import deque

EGOPOSER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EGOPOSER_DIR)

import websockets
from utils import utils_option as option
from models.select_model import define_Model, define_bm
from utils import utils_transform
from human_body_prior.tools.rotation_tools import aa2matrot, local2global_pose

# ===== 配置 =====
HOST = '0.0.0.0'
PORT = 8888
YAML_PATH = os.path.join(EGOPOSER_DIR, 'options/test_egoposer.yaml')
WINDOW_SIZE = 80

# ===== 数学工具 =====

def quat_xyzw_to_rotation_matrix(qxyzw):
    """
    Unity 四元数 [x, y, z, w] -> 3x3 旋转矩阵 (numpy)
    """
    x, y, z, w = qxyzw
    # 归一化
    n = np.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x/n, y/n, z/n, w/n
    # 旋转矩阵
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),       2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),       2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w),   1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return R


def rotation_matrix_to_6d(R):
    """
    3x3 旋转矩阵 -> 6D 表示 (前两列展平)
    rotation_6D = [R[:,0], R[:,1]]  -> shape (6,)
    """
    return np.concatenate([R[:, 0], R[:, 1]], axis=0).astype(np.float32)


def rotation_matrix_to_quat_xyzw(R):
    """
    3x3 旋转矩阵 -> Unity 四元数 [x, y, z, w]
    """
    m = R.astype(np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    # 归一化
    q = np.array([x, y, z, w], dtype=np.float64)
    n = np.linalg.norm(q)
    if n > 1e-12:
        q = q / n
    return q.astype(np.float32)


def sixd_to_quat_xyzw(sixd):
    """
    6D 旋转 -> Unity 四元数 [x,y,z,w]
    流程: 6D -> 3x3 矩阵 (Gram-Schmidt) -> 四元数
    """
    a1 = sixd[:3].astype(np.float64)
    a2 = sixd[3:6].astype(np.float64)
    b1 = a1 / (np.linalg.norm(a1) + 1e-12)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-12)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=-1)  # 3x3
    return rotation_matrix_to_quat_xyzw(R)


# ===== EgoPoser 推理封装 =====

class EgoPoserInference:
    """
    维护 80 帧滑动窗口，每帧做一次 EgoPoser 推理
    输出: 22 个 SMPL 关节位置 [22, 3] (SMPL 空间，不做坐标变换)
    """
    def __init__(self, model, bm):
        self.model = model
        self.bm = bm
        self.device = model.device
        self.window_size = WINDOW_SIZE

        # 滑动窗口 (每帧一个 54 维向量)
        self.sparse_buffer = deque(maxlen=WINDOW_SIZE)
        self.fov_l_buffer = deque(maxlen=WINDOW_SIZE)
        self.fov_r_buffer = deque(maxlen=WINDOW_SIZE)

        # 上一帧 6D (用于 6D 直接差分, 与 EgoPoser 原项目一致)
        self.prev_6d = None  # list of 6D arrays: [head_6d, lhand_6d, rhand_6d]

        # 是否已收集到足够帧
        self.warmed_up = False

    def reset(self):
        self.sparse_buffer.clear()
        self.fov_l_buffer.clear()
        self.fov_r_buffer.clear()
        self.prev_6d = None
        self.warmed_up = False

    def prefill(self, frame_54):
        """用第一帧填满 80 帧窗口 (暖机)"""
        for _ in range(self.window_size):
            self.sparse_buffer.append(frame_54.copy())
            self.fov_l_buffer.append(True)
            self.fov_r_buffer.append(True)

    def process_trackers(self, trackers_msg):
        """
        解析 trackers JSON -> 更新内部状态 -> 返回 54 维 sparse_input

        布局严格对齐 EgoPoser 原项目 process_amass_to_input:
            [0:6]   head_6d   (旋转矩阵前两列)
            [6:12]  lhand_6d
            [12:18] rhand_6d
            [18:24] head_vel  (6D 直接差分: cur_6d - prev_6d)
            [24:30] lhand_vel (同上)
            [30:36] rhand_vel (同上)
            [36:39] head_pos
            [39:42] lhand_pos
            [42:45] rhand_pos
            [45:48] zero
            [48:51] zero
            [51:54] zero

        trackers_msg: dict, 格式见文件头
        返回: (sparse_frame np.array[54], fov_l bool, fov_r bool)
        """
        head = trackers_msg.get('head', {})
        lw = trackers_msg.get('left_wrist', {})
        rw = trackers_msg.get('right_wrist', {})

        # 解析位置和旋转
        def parse_node(node):
            pos = np.array(node.get('position', [0, 0, 0]), dtype=np.float32)
            rot = node.get('rotation', [0, 0, 0, 1])
            tracked = bool(node.get('tracked', False))
            return pos, rot, tracked

        head_pos, head_rot, head_tracked = parse_node(head)
        lw_pos, lw_rot, lw_tracked = parse_node(lw)
        rw_pos, rw_rot, rw_tracked = parse_node(rw)

        # 四元数 (xyzw) -> 旋转矩阵
        R_head = quat_xyzw_to_rotation_matrix(head_rot)
        R_lw = quat_xyzw_to_rotation_matrix(lw_rot)
        R_rw = quat_xyzw_to_rotation_matrix(rw_rot)

        # [0:18] 当前全局旋转 6D (旋转矩阵前两列)
        head_6d = rotation_matrix_to_6d(R_head)
        lw_6d = rotation_matrix_to_6d(R_lw)
        rw_6d = rotation_matrix_to_6d(R_rw)

        # [18:36] 6D 直接差分 (与原项目 head_6d[1:] - head_6d[:-1] 一致)
        if self.prev_6d is not None:
            head_vel = head_6d - self.prev_6d[0]
            lw_vel = lw_6d - self.prev_6d[1]
            rw_vel = rw_6d - self.prev_6d[2]
        else:
            # 第一帧 vel = 0 (原项目用 torch.cat([zeros, diff]) 做对齐)
            head_vel = np.zeros(6, dtype=np.float32)
            lw_vel = np.zeros(6, dtype=np.float32)
            rw_vel = np.zeros(6, dtype=np.float32)

        # 组装 54 维 (顺序与原项目 process_amass_to_input 完全一致)
        sparse_frame = np.concatenate([
            head_6d, lw_6d, rw_6d,   # [0:18]  当前旋转 6D
            head_vel, lw_vel, rw_vel, # [18:36] 6D 直接差分
            head_pos, lw_pos, rw_pos, # [36:45] 当前位置
            np.zeros(9, dtype=np.float32),  # [45:54] 全零 (原项目固定 0)
        ], axis=0).astype(np.float32)

        # 更新上一帧状态
        self.prev_6d = [head_6d, lw_6d, rw_6d]

        fov_l = lw_tracked
        fov_r = rw_tracked

        return sparse_frame, fov_l, fov_r

    def step(self, sparse_frame, fov_l, fov_r, input_head_pos=None):
        """
        推理一帧
        返回: joint_pos [22, 3] (SMPL 空间，不做坐标变换), 或 None (暖机中)
        """
        self.sparse_buffer.append(sparse_frame)
        self.fov_l_buffer.append(fov_l)
        self.fov_r_buffer.append(fov_r)

        if len(self.sparse_buffer) < self.window_size:
            return None

        # 满帧，开始推理
        self.warmed_up = True

        sparse = torch.FloatTensor(np.array(self.sparse_buffer)).unsqueeze(0).to(self.device)
        fov_l_t = torch.BoolTensor(np.array(self.fov_l_buffer)).unsqueeze(0)
        fov_r_t = torch.BoolTensor(np.array(self.fov_r_buffer)).unsqueeze(0)

        x = {'sparse_input': sparse, 'fov_l': fov_l_t, 'fov_r': fov_r_t}

        with torch.no_grad():
            output = self.model.net(x)
            root_orient_6d = output['root_orient']   # [1, 6]
            pose_body_6d = output['pose_body']       # [1, 126]

            # 6D -> 轴角
            root_orient_aa = utils_transform.sixd2aa(root_orient_6d.reshape(-1, 6)).reshape(-1, 3).float()
            pose_body_aa = utils_transform.sixd2aa(pose_body_6d.reshape(-1, 6)).reshape(-1, 63).float()

            # 1. 无位移身体，得到 head 相对 pelvis 的位置
            body_local = self.model.bm(**{
                'pose_body': pose_body_aa,
                'root_orient': root_orient_aa,
            })
            t_head2root = body_local.Jtr[0, 15].cpu().numpy()

            # 2. 计算 pelvis 世界位移 (让 head 落在 input_head_pos)
            if input_head_pos is not None:
                t_root2world = -t_head2root + input_head_pos
            else:
                t_root2world = np.zeros(3)

            # 3. 带位移的完整身体
            t_tensor = torch.tensor(t_root2world, dtype=torch.float32, device=self.device).unsqueeze(0)
            body_pose = self.model.bm(**{
                'pose_body': pose_body_aa,
                'root_orient': root_orient_aa,
                'trans': t_tensor,
                'betas': output.get('betas', None),
            })
            joint_pos = body_pose.Jtr[0, :22].cpu().numpy()  # [22, 3] SMPL 空间

        return joint_pos


# ===== WebSocket 服务器 =====

def load_model(yaml_path=YAML_PATH):
    opt = option.parse(yaml_path, is_train=True)
    opt['path']['pretrained'] = opt['pretrained_model']
    opt = option.dict_to_nonedict(opt)
    model = define_Model(opt)
    model.load(test=True)
    model.net.eval()
    return model, opt


class EgoPoserServer:
    def __init__(self):
        print("=" * 60)
        print("  EgoPoser WebSocket Server")
        print(f"  Listen: ws://{HOST}:{PORT}")
        print("=" * 60)

        # 加载模型
        print("[Init] 加载 EgoPoser 模型 ...")
        self.model, opt = load_model(YAML_PATH)
        bm_dict = define_bm(opt)
        self.bm = bm_dict['male']
        print("[Init] ✅ 模型加载完成")

        # 推理器
        self.infer = EgoPoserInference(self.model, self.bm)

        # 统计
        self.recv_count = 0
        self.infer_count = 0
        self.last_log_time = time.time()
        self.last_seq = -1
        self.latest_trackers = None  # 积压时只保留最新

    async def handle_connection(self, websocket):
        """处理一个 Unity 连接"""
        peer = websocket.remote_address if hasattr(websocket, 'remote_address') else "?"
        print(f"[WS] ✅ Unity 已连接: {peer}")

        try:
            async for raw_msg in websocket:
                try:
                    msg = json.loads(raw_msg)
                except json.JSONDecodeError as e:
                    print(f"[WARN] JSON 解析失败: {e}")
                    continue

                if msg.get('type') != 'trackers':
                    continue

                seq = msg.get('sequence', -1)
                self.latest_trackers = msg
                self.recv_count += 1

                # 处理最新帧 (积压时旧帧被覆盖丢弃)
                response = self.process_one_frame(self.latest_trackers)
                self.latest_trackers = None

                if response is not None:
                    try:
                        await websocket.send(json.dumps(response))
                    except Exception as e:
                        print(f"[WS] 发送失败: {e}")
                        break

                # 周期日志
                now = time.time()
                if now - self.last_log_time > 5.0:
                    fps = self.recv_count / (now - self.last_log_time)
                    win_len = len(self.infer.sparse_buffer)
                    print(f"[STAT] recv_fps={fps:.1f} | seq={seq} | "
                          f"window={win_len}/{WINDOW_SIZE} | "
                          f"infer_total={self.infer_count}")
                    self.recv_count = 0
                    self.last_log_time = now

        except websockets.exceptions.ConnectionClosed as e:
            print(f"[WS] 连接关闭: {e}")
        except Exception as e:
            print(f"[WS] 异常: {e}")
        finally:
            print(f"[WS] Unity 断开: {peer}")
            # 重置推理器，等下一次连接重新暖机
            self.infer.reset()

    def process_one_frame(self, trackers_msg):
        """
        处理一帧 trackers，返回响应 dict (body_pose 或 warming_up)
        """
        seq = trackers_msg.get('sequence', -1)
        self.last_seq = seq

        # 1. 解析 trackers -> 54 维 sparse_frame
        sparse_frame, fov_l, fov_r = self.infer.process_trackers(trackers_msg)

        # 2. 提取 input_head_pos (当前位置 [36:39])
        input_head_pos = sparse_frame[36:39].copy()

        # 3. 推理
        t0 = time.time()
        joint_pos = self.infer.step(sparse_frame, fov_l, fov_r, input_head_pos)
        t_infer = (time.time() - t0) * 1000  # ms

        # 4. 构造响应
        if joint_pos is None:
            # 暖机中
            frames_collected = len(self.infer.sparse_buffer)
            return {
                "type": "warming_up",
                "sequence": seq,
                "frames_collected": frames_collected,
                "frames_needed": WINDOW_SIZE,
            }

        # 5. NaN/Inf 检查
        if not np.all(np.isfinite(joint_pos)):
            print(f"[WARN] 推理结果含 NaN/Inf, seq={seq}")
            return None

        self.infer_count += 1

        # joints_world: 22 个 [x,y,z] (SMPL 原始空间，不做坐标变换 - B 方案)
        joints_world = joint_pos.tolist()

        # root_position / root_rotation: Pelvis = joints[0]
        root_pos = joint_pos[0].tolist()

        # root_rotation: 从 EgoPoser 输出获取 (这里简化用单位四元数, 真实场景可从模型输出取)
        # EgoPoser 输出的是 root_orient_6d, 我们可以转成四元数
        # 但为了简单和稳定，先用单位四元数
        root_rot = [0.0, 0.0, 0.0, 1.0]  # 单位四元数

        return {
            "type": "body_pose",
            "sequence": seq,
            "inference_ms": round(t_infer, 2),
            "root_position": root_pos,
            "root_rotation": root_rot,
            "joints_world": joints_world,
        }

    async def run(self):
        """启动 WebSocket 服务器"""
        print(f"[Server] 启动 WebSocket 服务器 ws://{HOST}:{PORT} ...")

        # 使用 websockets.serve 启动服务器
        async with websockets.serve(self.handle_connection, HOST, PORT,
                                     max_size=1024 * 1024,  # 1MB max message
                                     ping_interval=20,
                                     ping_timeout=60):
            print(f"[Server] ✅ 等待 Unity 连接 ws://{HOST}:{PORT}")
            print(f"[Server] 按 Ctrl+C 停止")
            # 永久运行
            await asyncio.Future()


def main():
    server = EgoPoserServer()
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\n[系统] 用户中断，关闭服务器")
    except Exception as e:
        print(f"[ERROR] 服务器异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
