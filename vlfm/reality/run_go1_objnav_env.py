# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import time

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
import sys

from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from vlfm.policy.reality_policies import RealityConfig, RealityITMPolicyV2
from vlfm.reality.objectnav_go1_env import ObjectNavEnv
from vlfm.reality.robots.go1_robot import Go1Robot
from vlfm.reality.robots.go1_zed_wrapper.go1_zed import Go1Zed


@hydra.main(version_base=None, config_path="../../config/", config_name="experiments/reality_go1")
def main(cfg: RealityConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    policy = RealityITMPolicyV2.from_config(cfg)

    # Initialize Go1 robot with ZED camera
    go1_zed = Go1Zed()
    try:
        go1_zed.stand()
        print("Go1 robot commanded to stand")
    except Exception as e:
        print(f"Warning: Could not send stand command: {e}")
        print("Continuing anyway - make sure robot is already standing or standing manually")
    
    # Create robot wrapper
    robot = Go1Robot(go1_zed)
    
    # Go1 doesn't have gripper or arm, so we skip those setup steps
    
    # Create environment
    env = ObjectNavEnv(
        robot=robot,
        max_body_cam_depth=cfg.env.max_body_cam_depth,
        max_gripper_cam_depth=cfg.env.max_gripper_cam_depth,  # Not used for Go1, but kept for API compatibility
        max_lin_dist=cfg.env.max_lin_dist,
        max_ang_dist=cfg.env.max_ang_dist,
        time_step=cfg.env.time_step,
    )
    goal = cfg.env.goal
    run_env(env, policy, goal)


def run_env(env: ObjectNavEnv, policy: RealityITMPolicyV2, goal: str) -> None:
    observations = env.reset(goal)
    done = False
    mask = torch.zeros(1, 1, device="cuda", dtype=torch.bool)
    st = time.time()
    action = policy.get_action(observations, mask)
    print(f"get_action took {time.time() - st:.2f} seconds")
    while not done:
        observations, _, done, info = env.step(action)
        st = time.time()
        action = policy.get_action(observations, mask, deterministic=True)
        print(f"get_action took {time.time() - st:.2f} seconds")
        mask = torch.ones_like(mask)
        if done:
            print("Episode finished because done is True")
            break


if __name__ == "__main__":
    main()
