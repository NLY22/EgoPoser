"""
Vision Pro 头显追踪数据接收器
从 Unity 接收头部 + 双手追踪数据，实时送入 EgoPoser 推理

数据格式 (TCP, 长度前缀):
  [4字节长度] + [108字节数据]
  108字节 = 27个float32:
    [0..5]   head_6d      (6D旋转)
    [6..11]  lhand_6d     (左手6D旋转)
    [12..17] rhand_6d     (右手6D旋转)
    [18..20] head_pos     (头部位置)
    [21..23] lhand_pos    (左手位置)
    [24..26] rhand_pos    (右手位置)
"""

import socket, struct, time, os
import numpy as np
import torch
from collections import deque

# ===== 配置 =====
HOST = '0.0.0.0'
PORT = 8889          # 与 Unity EgoPoserTracker.cs 的 sendPort 一致
RECORD_TO_NPZ = True  # 同时记录到 npz 文件
NPZ_SAVE_PATH = 'support_data/github_data/visionpro_track.npz'


def sixd_to_rotation_matrix(d6):
    """6D 旋转表示 → 3x3 旋转矩阵 (Gram-Schmidt 正交化)"""
    d6 = d6.reshape(-1, 2, 3)
    b1 = d6[:, 0]
    b2 = d6[:, 1] - np.sum(d6[:, 1] * b1, axis=1, keepdims=True) * b1
    b1 = b1 / np.linalg.norm(b1, axis=1, keepdims=True)
    b2 = b2 / np.linalg.norm(b2, axis=1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def rotation_matrix_to_axis_angle(R):
    """3x3 旋转矩阵 → 轴角 (Rodrigues)"""
    # 从旋转矩阵提取轴角
    angle = np.arccos(np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1))
    rx = R[:, 2, 1] - R[:, 1, 2]
    ry = R[:, 0, 2] - R[:, 2, 0]
    rz = R[:, 1, 0] - R[:, 0, 1]
    axis = np.stack([rx, ry, rz], axis=1)
    sin_angle = np.sin(angle)
    # 处理角度为0的情况
    mask = np.abs(sin_angle) > 1e-8
    axis = np.where(mask[:, None], axis / sin_angle[:, None], axis)
    return axis * angle[:, None]  # 轴角 = 单位轴 * 角度


def unity_to_smpl_coords(pos_unity):
    """
    Unity 空间 (X右, Y上, Z前) → SMPL 空间 (X前, Y左右, Z上)
    即逆变换: SMPL→Unity 的逆
    """
    pos_smpl = np.zeros_like(pos_unity)
    pos_smpl[:, 0] = pos_unity[:, 0]   # Unity X → SMPL X (不变)
    pos_smpl[:, 1] = -pos_unity[:, 2]  # -Unity Z → SMPL Y
    pos_smpl[:, 2] = pos_unity[:, 1]   # Unity Y → SMPL Z
    return pos_smpl


def main():
    print("=" * 50)
    print("  Vision Pro 追踪数据接收器")
    print(f"  监听端口: {HOST}:{PORT}")
    print("=" * 50)

    # 存储接收到的数据 (用于记录到 npz)
    all_head_pos = []
    all_head_rot_6d = []
    all_lhand_pos = []
    all_lhand_rot_6d = []
    all_rhand_pos = []
    all_rhand_rot_6d = []
    timestamps = []

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    print(f"[网络] 等待 Unity 连接 {HOST}:{PORT} ...")
    conn, addr = server.accept()
    print(f"[网络] ✅ Unity 已连接: {addr}")

    frame_count = 0
    t_start = time.time()

    try:
        while True:
            # 读 4 字节长度头
            header = conn.recv(4)
            if not header:
                print("[网络] 连接关闭")
                break
            data_len = struct.unpack('I', header)[0]

            # 读数据体
            data = b''
            while len(data) < data_len:
                chunk = conn.recv(data_len - len(data))
                if not chunk:
                    break
                data += chunk

            if len(data) < data_len:
                print("[网络] 数据不完整，跳过")
                continue

            # 解析 27 个 float32
            values = np.frombuffer(data, dtype=np.float32)  # [27]
            head_6d = values[0:6]
            lhand_6d = values[6:12]
            rhand_6d = values[12:18]
            head_pos = values[18:21]   # Unity 空间
            lhand_pos = values[21:24]  # Unity 空间
            rhand_pos = values[24:27]  # Unity 空间

            # 记录
            all_head_pos.append(head_pos.copy())
            all_head_rot_6d.append(head_6d.copy())
            all_lhand_pos.append(lhand_pos.copy())
            all_lhand_rot_6d.append(lhand_6d.copy())
            all_rhand_pos.append(rhand_pos.copy())
            all_rhand_rot_6d.append(rhand_6d.copy())
            timestamps.append(time.time() - t_start)

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"[接收] {frame_count} 帧 | "
                      f"头: ({head_pos[0]:.2f}, {head_pos[1]:.2f}, {head_pos[2]:.2f}) | "
                      f"左手: ({lhand_pos[0]:.2f}, {lhand_pos[1]:.2f}, {lhand_pos[2]:.2f})")

    except (BrokenPipeError, ConnectionResetError) as e:
        print(f"[网络] ❌ {e}")
    except KeyboardInterrupt:
        print("\n[系统] 用户中断")
    finally:
        conn.close()
        server.close()
        print(f"[系统] 共接收 {frame_count} 帧")

        # 保存到 npz
        if RECORD_TO_NPZ and len(all_head_pos) > 0:
            # 转换为 numpy 数组
            np_head_pos = np.array(all_head_pos)         # [N, 3]
            np_head_6d = np.array(all_head_rot_6d)       # [N, 6]
            np_lhand_pos = np.array(all_lhand_pos)       # [N, 3]
            np_lhand_6d = np.array(all_lhand_rot_6d)     # [N, 6]
            np_rhand_pos = np.array(all_rhand_pos)       # [N, 3]
            np_rhand_6d = np.array(all_rhand_rot_6d)     # [N, 6]

            # 保存
            save_dir = os.path.dirname(NPZ_SAVE_PATH)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            np.savez(NPZ_SAVE_PATH,
                     head_pos=np_head_pos,
                     head_rot_6d=np_head_6d,
                     lhand_pos=np_lhand_pos,
                     lhand_rot_6d=np_lhand_6d,
                     rhand_pos=np_rhand_pos,
                     rhand_rot_6d=np_rhand_6d,
                     timestamps=np.array(timestamps))
            print(f"[保存] ✅ {NPZ_SAVE_PATH} ({len(all_head_pos)} 帧)")


if __name__ == '__main__':
    main()