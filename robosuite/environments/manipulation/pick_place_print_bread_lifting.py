#!/usr/bin/env python3
import time
import sys
import numpy as np
import robosuite as suite

OBJECT_NAMES = ["milk", "bread", "cereal", "can"]

# ------------------- 기본 헬퍼 -------------------
def _get_eef_pos_from_obs(obs):
    for key in ["robot0_eef_pos", "eef_pos"]:
        if key in obs:
            return np.array(obs[key])
    return None

def _get_eef_pose(env):
    # Return (pos, rot_mat) of EE in world
    for site_name in ("gripper0_right_grip_site", "gripper0_grip_site"):
        try:
            sid = env.sim.model.site_name2id(site_name)
            pos = np.array(env.sim.data.site_xpos[sid])
            rot = np.array(env.sim.data.site_xmat[sid]).reshape(3, 3).T
            return pos, rot
        except Exception:
            continue
    return None, np.eye(3)

def _get_body_pos(env, body_name: str):
    try:
        return np.array(env.sim.data.get_body_xpos(body_name))
    except Exception:
        try:
            bid = env.sim.model.body_name2id(body_name)
            return np.array(env.sim.data.body_xpos[bid])
        except Exception:
            return None

# ------------------- Bread pose resolver -------------------
def _find_body_id_by_token(env, token: str):
    token = token.lower()
    try:
        for i in range(env.sim.model.nbody):
            try:
                name_i = env.sim.model.body(i).name
            except Exception:
                name_i = None
            if not name_i:
                continue
            if token in name_i.lower():
                return i
    except Exception:
        pass
    return None

def _get_bread_pos(env):
    # Try via env.objects root_body when available
    try:
        objects = getattr(env, "objects", None)
    except Exception:
        objects = None
    if objects:
        for obj in objects:
            if "bread" in obj.name.lower():
                candidate = getattr(obj, "root_body", None) or obj.name
                try:
                    bid = env.sim.model.body_name2id(candidate)
                    return np.array(env.sim.data.body_xpos[bid])
                except Exception:
                    pass
    # Fallback to direct name
    p = _get_body_pos(env, "bread")
    if p is not None:
        return p
    # Token fallback across all bodies
    bid = _find_body_id_by_token(env, "bread")
    if bid is not None:
        try:
            return np.array(env.sim.data.body_xpos[bid])
        except Exception:
            return None
    return None

def _status_write(line: str):
    sys.stdout.write("\r\x1b[2K" + line)
    sys.stdout.flush()

# ------------------- Orientation -------------------
def _rotmat_to_euler_zyx(R: np.ndarray):
    sy = -R[2, 0]
    pitch = np.arcsin(np.clip(sy, -1.0, 1.0))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])
    return yaw, pitch, roll

# ------------------- Status print -------------------
def _print_status(env):
    eef_pos, R_eef = _get_eef_pose(env)
    yaw_e, pitch_e, roll_e = _rotmat_to_euler_zyx(R_eef)
    yaw_ed, pitch_ed, roll_ed = np.degrees([yaw_e, pitch_e, roll_e])
    bread_pos = _get_bread_pos(env)
    parts = [
        f"EE[{eef_pos[0]:.3f},{eef_pos[1]:.3f},{eef_pos[2]:.3f}]",
        f"EE(rpy°)[{roll_ed:.1f},{pitch_ed:.1f},{yaw_ed:.1f}]",
    ]
    if bread_pos is not None:
        parts.append(f"B[{bread_pos[0]:.3f},{bread_pos[1]:.3f},{bread_pos[2]:.3f}]")
    _status_write(" | ".join(parts)[:160])

# ------------------- Linear planner/tracker -------------------
def _linspace_path(p0: np.ndarray, p1: np.ndarray, num: int):
    alphas = np.linspace(0.0, 1.0, max(2, int(num)))
    return np.stack([(1 - a) * p0 + a * p1 for a in alphas], axis=0)

def _track_linear_path(env, path_points: np.ndarray, gripper_dof: int,
                       pos_gain: float = 4.0, max_dpos: float = 0.05):
    for p_tgt in path_points:
        eef_pos, R_eef = _get_eef_pose(env)
        dpos = (p_tgt - eef_pos) * pos_gain
        dist = np.linalg.norm(dpos)
        if dist > max_dpos:
            dpos = dpos * (max_dpos / (dist + 1e-9))
        dpos = dpos.astype(np.float32)
        drot = np.zeros(3, dtype=np.float32)
        g = np.repeat(-1.0, gripper_dof) if gripper_dof > 1 else np.array([ -1.0 ], dtype=np.float32)
        action = np.concatenate([dpos, drot, g]).astype(np.float32)
        env.step(action)
        env.render()
        _print_status(env)

def _track_to_target(env, target_pos: np.ndarray, gripper_dof: int,
                     dist_thresh: float = 0.02, max_steps: int = 2000,
                     pos_gain: float = 5.0, max_dpos: float = 0.06,
                     freeze_axes: tuple = (False, False, False),
                     info_prefix: str = "",
                     grip_val: float = -1.0):
    steps = 0
    while steps < max_steps:
        steps += 1
        eef_pos, _ = _get_eef_pose(env)
        if eef_pos is None:
            break
        d = target_pos - eef_pos
        # freeze selected axes
        mask = np.array([not freeze_axes[0], not freeze_axes[1], not freeze_axes[2]], dtype=np.float32)
        d = d * mask
        dist = np.linalg.norm(d)
        if dist < dist_thresh:
            if info_prefix:
                print(f"[INFO] {info_prefix} 도달: dist={dist:.3f} < {dist_thresh}")
            return True
        dpos = (d * pos_gain)
        nrm = np.linalg.norm(dpos)
        if nrm > max_dpos:
            dpos = dpos * (max_dpos / (nrm + 1e-9))
        dpos = dpos.astype(np.float32)
        drot = np.zeros(3, dtype=np.float32)
        g = np.repeat(grip_val, gripper_dof) if gripper_dof > 1 else np.array([ grip_val ], dtype=np.float32)
        action = np.concatenate([dpos, drot, g]).astype(np.float32)
        env.step(action); env.render(); _print_status(env)
        if steps % 50 == 0 and info_prefix:
            print(f"[INFO] {info_prefix} 진행: dist={dist:.3f}< {dist_thresh}")
    if info_prefix:
        print(f"[WARN] {info_prefix} 타임아웃: 마지막 dist={dist:.3f}< {dist_thresh}, steps={steps}")
    return False

# ------------------- main -------------------
def main():
    env = suite.make("PickPlaceBread", robots="Panda",
                     has_renderer=True, use_camera_obs=False,
                     use_object_obs=True, ignore_done=True,
                     horizon=2000, control_freq=20, reward_shaping=True)
    obs = env.reset(); env.viewer.set_camera(camera_id=0)
    print("[INFO] 직선 경로로 빵 위치까지 트래킹을 시작합니다.")

    # gripper dof
    try:
        robot = env.robots[0]
        arm_name = robot.arms[0] if hasattr(robot, "arms") else "right"
        gripper_dof = robot.gripper[arm_name].dof if robot.gripper[arm_name] else 1
    except:
        gripper_dof = 1

    # plan linear path from current EE to bread
    eef_pos, _ = _get_eef_pose(env)
    bread_pos = _get_bread_pos(env)
    if eef_pos is None or bread_pos is None:
        print("[WARN] pose 읽기 실패")
    else:
        # 목표1: (bread_x, bread_y, 현재 z)까지 XY 정렬, 목표2: z 하강 (bread_z-0.05)
        p_goal_xy = np.array([bread_pos[0], bread_pos[1], eef_pos[2]], dtype=np.float64)
        p_goal_z = np.array([bread_pos[0], bread_pos[1], bread_pos[2] - 0.05], dtype=np.float64)
        _track_to_target(env, p_goal_xy, gripper_dof,
                         dist_thresh=0.02, max_steps=1500,
                         pos_gain=5.0, max_dpos=0.06,
                         freeze_axes=(False, False, True),
                         info_prefix="XY 정렬",
                         grip_val=-1.0)
        _track_to_target(env, p_goal_z, gripper_dof,
                         dist_thresh=0.06, max_steps=1500,
                         pos_gain=5.0, max_dpos=0.06,
                         freeze_axes=(False, False, False),
                         info_prefix="Z 하강",
                         grip_val=-1.0)
        # close gripper
        for _ in range(40):
            action = np.concatenate([np.zeros(6, dtype=np.float32), np.array([1.0], dtype=np.float32)]).astype(np.float32)
            env.step(action); env.render()
            _print_status(env)

        # lift up by +0.2m in z from current pose
        eef_pos, _ = _get_eef_pose(env)
        if eef_pos is not None:
            p_lift = np.array([eef_pos[0], eef_pos[1], eef_pos[2] + 0.2], dtype=np.float64)
            _track_to_target(env, p_lift, gripper_dof,
                             dist_thresh=0.02, max_steps=1500,
                             pos_gain=5.0, max_dpos=0.06,
                             freeze_axes=(False, False, False),
                             info_prefix="상승",
                             grip_val=1.0)

    print("[INFO] 완료. Ctrl+C 종료.")
    while True:
        env.render(); _print_status(env)

if __name__ == "__main__":
    main()
