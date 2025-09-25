#!/usr/bin/env python3
import time
import sys
import numpy as np

import robosuite as suite


OBJECT_NAMES = ["milk", "bread", "cereal", "can"]


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
            # MuJoCo columns are local axes in world
            rot = np.array(env.sim.data.site_xmat[sid]).reshape(3, 3).T
            return pos, rot
        except Exception:
            continue
    return None, np.eye(3)


def _get_eef_pos_from_sim(env):
    candidate_site_names = [
        "gripper0_right_grip_site",
        "gripper0_grip_site",
    ]
    for site_name in candidate_site_names:
        try:
            site_id = env.sim.model.site_name2id(site_name)
            return np.array(env.sim.data.site_xpos[site_id])
        except Exception:
            continue
    return None


def _find_body_pos_by_token(env, token: str):
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
                try:
                    return np.array(env.sim.data.body(i).xpos)
                except Exception:
                    pass
    except Exception:
        pass
    return None


def _get_body_pos(env, body_name: str):
    try:
        return np.array(env.sim.data.get_body_xpos(body_name))
    except Exception:
        try:
            bid = env.sim.model.body_name2id(body_name)
            return np.array(env.sim.data.body_xpos[bid])
        except Exception:
            # Fallback: try capitalized and token search
            try:
                bid2 = env.sim.model.body_name2id(body_name.capitalize())
                return np.array(env.sim.data.body_xpos[bid2])
            except Exception:
                pos = _find_body_pos_by_token(env, body_name)
                return pos


def _get_object_positions(env, obs):
    obj_pos = {}
    try:
        objects = getattr(env, "objects", None)
    except Exception:
        objects = None

    if objects:
        for obj in objects:
            label_key = None
            for name in OBJECT_NAMES:
                if name in obj.name.lower():
                    label_key = name
                    break
            if label_key is None:
                continue
            body_name = getattr(obj, "root_body", None) or obj.name
            pos = _get_body_pos(env, body_name)
            if pos is None:
                pos = _find_body_pos_by_token(env, label_key)
            obj_pos[label_key] = pos

    for name in OBJECT_NAMES:
        if name in obj_pos and obj_pos[name] is not None:
            continue
        for key in (f"{name}_pos", f"{name}_position", f"{name}_xyz"):
            if key in obs:
                val = np.array(obs[key]).reshape(-1)
                if val.size >= 3:
                    obj_pos[name] = val[:3]
                    break
        if name in obj_pos and obj_pos[name] is not None:
            continue
        pos = _get_body_pos(env, name)
        if pos is None:
            pos = _get_body_pos(env, name.capitalize())
        if pos is None:
            pos = _find_body_pos_by_token(env, name)
        obj_pos[name] = pos

    return obj_pos


def _status_write(line: str):
    sys.stdout.write("\r\x1b[2K" + line)
    sys.stdout.flush()


# === New helpers for orientation ===

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


def _get_body_rot_mat(env, body_name: str):
    # Returns 3x3 rotation matrix (local->world)
    try:
        bid = env.sim.model.body_name2id(body_name)
    except Exception:
        bid = None
    if bid is None:
        # Try capitalized and token search
        try:
            bid = env.sim.model.body_name2id(body_name.capitalize())
        except Exception:
            bid = None
    if bid is None:
        bid = _find_body_id_by_token(env, body_name)
    if bid is None:
        return None
    try:
        # MuJoCo stores row-major; transpose to get columns as local axes in world
        return np.array(env.sim.data.body_xmat[bid]).reshape(3, 3).T
    except Exception:
        return None


def _rotmat_to_euler_zyx(R: np.ndarray):
    # Returns yaw(Z), pitch(Y), roll(X) in radians
    # Assumes R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    sy = -R[2, 0]
    pitch = np.arcsin(np.clip(sy, -1.0, 1.0))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])
    return yaw, pitch, roll


def _euler_zyx_to_rotmat(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll), np.sin(roll)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return Rz @ Ry @ Rx


def _rotmat_to_axis_angle(R: np.ndarray) -> np.ndarray:
    # Convert small rotation R to axis-angle vector w (so that exp([w]x) ~ R)
    # For small angles, w ~ 0.5 * [R32 - R23, R13 - R31, R21 - R12]
    w = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) * 0.5
    return w

def main():
    env = suite.make(
        "PickPlace",
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        ignore_done=True,
        horizon=100000,
        control_freq=20,
        reward_shaping=True,
    )

    obs = env.reset()
    env.viewer.set_camera(camera_id=0)

    print("[INFO] 빵의 orientation(회전 ZYX 오일러각)을 출력합니다. Ctrl+C로 종료하세요.")

    # Determine action dimension for no-op stepping
    try:
        action_dim = env.action_dim
    except Exception:
        try:
            low, high = env.action_spec
            action_dim = low.shape[0]
        except Exception:
            action_dim = 7

    # TTY detection
    is_tty = sys.stdout.isatty()
    if is_tty:
        sys.stdout.write("\x1b[?25l")
        sys.stdout.flush()

    # Resolve bread body
    bread_body_id = None
    bread_body_name = None
    try:
        objects = getattr(env, "objects", None)
    except Exception:
        objects = None
    if objects:
        for obj in objects:
            if "bread" in obj.name.lower():
                candidate = getattr(obj, "root_body", None) or obj.name
                try:
                    bread_body_id = env.sim.model.body_name2id(candidate)
                    bread_body_name = candidate
                    break
                except Exception:
                    pass
    if bread_body_id is None:
        bread_body_id = _find_body_id_by_token(env, "bread")
        if bread_body_id is not None:
            try:
                bread_body_name = env.sim.model.body(bread_body_id).name
            except Exception:
                bread_body_name = None

    if bread_body_id is not None:
        R0_bread = np.array(env.sim.data.body_xmat[bread_body_id]).reshape(3, 3).T
        p0_bread = np.array(env.sim.data.body_xpos[bread_body_id])
    else:
        R0_bread = None
        p0_bread = None

    target_fps = 20
    frame_dt = 1.0 / target_fps

    try:
        while True:
            start = time.time()

                        # default: no positional motion
            dpos = np.zeros(3, dtype=np.float32)

            # EE pos
            eef_pos = _get_eef_pos_from_obs(obs)
            if eef_pos is None:
                eef_pos = _get_eef_pos_from_sim(env)
            # EE rot (world)
            _, R_eef = _get_eef_pose(env)

            # Bread pos & rot
            if bread_body_id is not None:
                bread_pos = np.array(env.sim.data.body_xpos[bread_body_id])
                R_bread = np.array(env.sim.data.body_xmat[bread_body_id]).reshape(3, 3).T
            else:
                bread_pos = _get_body_pos(env, "bread")
                R_bread = _get_body_rot_mat(env, "bread")

            # Control: match EE yaw/pitch to Bread yaw/pitch (keep EE roll)
            drot = np.zeros(3, dtype=np.float32)
            if R_eef is not None and R_bread is not None:
                # Extract angles
                yaw_e, pitch_e, roll_e = _rotmat_to_euler_zyx(R_eef)
                yaw_b, pitch_b, roll_b = _rotmat_to_euler_zyx(R_bread)
                # Target yaw: match absolute value of bread yaw with sign closest to current yaw
                target_abs = abs(yaw_b)
                cand1 = target_abs  # positive
                cand2 = -target_abs # negative
                # Choose target yaw minimizing angular distance to current yaw
                def ang_diff(a, b):
                    d = a - b
                    return (d + np.pi) % (2*np.pi) - np.pi
                if abs(ang_diff(yaw_e, cand1)) <= abs(ang_diff(yaw_e, cand2)):
                    yaw_t = cand1
                else:
                    yaw_t = cand2
                # Keep EE pitch and roll unchanged
                R_target = _euler_zyx_to_rotmat(yaw_t, pitch_e, roll_e)
                # Small yaw-only correction in world frame via rotation error
                R_err = R_target @ R_eef.T
                w = _rotmat_to_axis_angle(R_err)
                # Limit magnitude for stability, emphasize z (yaw) component
                max_step = 0.03
                w = np.clip(w, -max_step, max_step)
                # Zero-out x,y components to avoid affecting roll/pitch
                w[0] = 0.0
                w[1] = 0.0
                drot = w.astype(np.float32)

            # Compose action: [dpos(3), drot(3), grip]
            try:
                robot = env.robots[0]
                arm_name = robot.arms[0] if hasattr(robot, "arms") else "right"
                gripper_dof = robot.gripper[arm_name].dof if robot.gripper[arm_name] is not None else 0
            except Exception:
                gripper_dof = 1
            grip_cmd = np.zeros(max(1, gripper_dof), dtype=np.float32)
            action = np.concatenate([dpos, drot, grip_cmd])

            # Apply controlled step
            obs, _, done, _ = env.step(action)
            if done:
                obs = env.reset()

            # Build log line
            parts = []
            if eef_pos is not None:
                parts.append(f"EE[{eef_pos[0]:.3f},{eef_pos[1]:.3f},{eef_pos[2]:.3f}]")
            if R_eef is not None:
                yaw_e, pitch_e, roll_e = _rotmat_to_euler_zyx(R_eef)
                yaw_ed, pitch_ed, roll_ed = np.degrees([yaw_e, pitch_e, roll_e])
                parts.append(f"EE(rpy°)[{roll_ed:.1f},{pitch_ed:.1f},{yaw_ed:.1f}]")
            if bread_pos is not None:
                parts.append(f"B[{bread_pos[0]:.3f},{bread_pos[1]:.3f},{bread_pos[2]:.3f}]")
            if R_bread is not None:
                yaw, pitch, roll = _rotmat_to_euler_zyx(R_bread)
                yaw_d, pitch_d, roll_d = np.degrees([yaw, pitch, roll])
                parts.append(f"B(rpy°)[{roll_d:.1f},{pitch_d:.1f},{yaw_d:.1f}]")

            line = " | ".join(parts)
            _status_write(line[:120]) 

            env.render()

            elapsed = time.time() - start
            if frame_dt - elapsed > 0:
                time.sleep(frame_dt - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        if is_tty:
            sys.stdout.write("\x1b[?25h\n")
            sys.stdout.flush()
        env.close()

if __name__ == "__main__":
    main()
