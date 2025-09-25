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
            return None


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


# Helper: rotmat -> Euler ZYX (yaw, pitch, roll)
def _rotmat_to_euler_zyx(R: np.ndarray):
    sy = -R[2, 0]
    pitch = np.arcsin(np.clip(sy, -1.0, 1.0))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])
    return yaw, pitch, roll


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

    print("[INFO] Keyboard teleop enabled. Printing EE & object world coordinates. Press Ctrl+C to stop.")

    from robosuite.devices.keyboard import Keyboard

    device = Keyboard(env=env, pos_sensitivity=1.0, rot_sensitivity=1.0)
    device.start_control()
    try:
        env.viewer.add_keypress_callback(device.on_press)
    except Exception:
        pass

    try:
        robot = env.robots[0]
        arm_name = robot.arms[0] if hasattr(robot, "arms") else "right"
        gripper_dof = robot.gripper[arm_name].dof if robot.gripper[arm_name] is not None else 0
    except Exception:
        gripper_dof = 1

    target_fps = 20
    frame_dt = 1.0 / target_fps

    sys.stdout.write("\x1b[?25l")
    sys.stdout.flush()

    try:
        while True:
            start = time.time()

            state = device.get_controller_state()
            dpos_local = state["dpos"]
            drot_local = state["raw_drotation"]
            grasp_toggle = state["grasp"]

            # Update mapping per request:
            # x_new (FB) = -y_old  
            # y_new (LR) = -x_old  
            # z_new (UD) = +z_old (unchanged)
            dpos_local_swapped = np.array([ -dpos_local[1], -dpos_local[0], dpos_local[2] ])
            drot_local_adj = drot_local.copy()

            _, R_eef = _get_eef_pose(env)
            dpos_world = R_eef @ dpos_local_swapped
            drot_world = R_eef @ drot_local_adj

            grip_val = 1.0 if grasp_toggle else -1.0
            grip_cmd = np.repeat(grip_val, gripper_dof) if gripper_dof > 1 else np.array([grip_val])

            action = np.concatenate([dpos_world, drot_world, grip_cmd])
            obs, _, done, _ = env.step(action)
            if done:
                obs = env.reset()

            eef_pos = _get_eef_pos_from_obs(obs)
            if eef_pos is None:
                eef_pos = _get_eef_pos_from_sim(env)

            obj_pos = _get_object_positions(env, obs)

            parts = []
            if eef_pos is not None:
                parts.append(f"EE: {eef_pos[0]: .3f} {eef_pos[1]: .3f} {eef_pos[2]: .3f}")
        
            for name in OBJECT_NAMES:
                p = obj_pos.get(name)
                if p is not None:
                    label = "Cn" if name == "can" else (name[0].upper())
                    parts.append(f"{label}: {p[0]: .3f} {p[1]: .3f} {p[2]: .3f}")
            line = " | ".join(parts) if parts else "(no positions)"
            _status_write(line[:120])

            env.render()

            elapsed = time.time() - start
            if frame_dt - elapsed > 0:
                time.sleep(frame_dt - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\x1b[?25h\n")
        sys.stdout.flush()
        env.close()


if __name__ == "__main__":
    main() 