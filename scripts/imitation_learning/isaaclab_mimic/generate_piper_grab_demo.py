#!/usr/bin/env python3
"""
Script to generate demonstration data for Piper robot pick-and-place task using IsaacLab.
The script runs trajectory planning to collect visual, proprioceptive, and action data in HDF5 format.
"""

import os
import torch
import numpy as np
import h5py
from typing import Dict, List, Tuple
import gymnasium as gym

from isaaclab_tasks.manager_based.piper_grab.grab_vision_env_cfg import PiperGrabVisuomotorEnvCfg


def save_demo_data_to_hdf5(data_dict: Dict, file_path: str) -> None:
    """
    Save collected demonstration data to HDF5 file.
    
    Args:
        data_dict: Dictionary containing all collected data
        file_path: Path to save the HDF5 file
    """
    with h5py.File(file_path, 'w') as f:
        def write_recursive(group, data_dict):
            for key, value in data_dict.items():
                if isinstance(value, dict):
                    subgroup = f.create_group(key)
                    write_recursive(subgroup, value)
                else:
                    if isinstance(value, torch.Tensor):
                        value = value.cpu().numpy()
                    f.create_dataset(key, data=value, compression='gzip')
        
        write_recursive(f, data_dict)


def collect_demonstration_data(num_episodes: int = 10, max_steps: int = 500, output_dir: str = "./demonstrations"):
    """
    Collect demonstration data using trajectory planning for Piper robot pick-and-place task.
    
    Args:
        num_episodes: Number of episodes to collect
        max_steps: Maximum steps per episode
        output_dir: Directory to save the demonstration data
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment
    env = gym.make("Isaac-Grab-Piper-v0", cfg=PiperGrabVisuomotorEnvCfg())
    
    print(f"[INFO] Created environment with {env.unwrapped.num_envs} environments")
    
    # Initialize data storage
    all_episode_data = []
    
    for episode_idx in range(num_episodes):
        print(f"[INFO] Collecting episode {episode_idx + 1}/{num_episodes}")
        
        # Reset environment
        obs_dict = env.reset()
        episode_data = {
            'actions': [],
            'obs': {
                'actions': [],
                'cube_positions': [],
                'cube_orientations': [],
                'eef_pos': [],
                'eef_quat': [],
                'gripper_pos': [],
                'joint_pos': [],
                'joint_vel': [],
                'object': [],
                'table_cam': [],
                'wrist_cam': []
            },
            'states': {
                'articulation': {
                    'robot': {
                        'joint_position': [],
                        'joint_velocity': [],
                        'root_pose': [],
                        'root_velocity': []
                    }
                },
                'rigid_object': {
                    'object_1': {
                        'root_pose': [],
                        'root_velocity': []
                    },
                    'box': {
                        'root_pose': [],
                        'root_velocity': []
                    }
                }
            },
            'initial_state': {
                'articulation': {
                    'robot': {
                        'joint_position': None,
                        'joint_velocity': None,
                        'root_pose': None,
                        'root_velocity': None
                    }
                },
                'rigid_object': {
                    'object_1': {
                        'root_pose': None,
                        'root_velocity': None
                    },
                    'box': {
                        'root_pose': None,
                        'root_velocity': None
                    }
                }
            }
        }
        
        # Store initial state
        state_dict = env.get_state()
        for obj_type in ['articulation', 'rigid_object']:
            for entity in state_dict[obj_type]:
                for prop in state_dict[obj_type][entity]:
                    episode_data['initial_state'][obj_type][entity][prop] = state_dict[obj_type][entity][prop][0].cpu().numpy()
        
        # Initialize episode variables
        step_count = 0
        
        while step_count < max_steps:
            # Generate demonstration action using simple trajectory planning logic
            # In a real scenario, this would come from an expert controller or motion planner
            current_joints = obs_dict['policy']['joint_pos'][0]
            
            # Simple trajectory planning: move to pre-grasp, grasp, lift, move to target, place
            episode_progress = step_count / max_steps
            
            if episode_progress < 0.2:  # Move to pre-grasp position
                target_pos = torch.tensor([0.3, 0.0, 0.2], device=env.unwrapped.device, dtype=torch.float32)
                target_orn = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.unwrapped.device, dtype=torch.float32)
                gripper_cmd = torch.tensor([0.04, 0.04], device=env.unwrapped.device, dtype=torch.float32)  # Open
            elif episode_progress < 0.3:  # Descend to grasp
                target_pos = torch.tensor([0.3, 0.0, 0.05], device=env.unwrapped.device, dtype=torch.float32)
                target_orn = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.unwrapped.device, dtype=torch.float32)
                gripper_cmd = torch.tensor([0.04, 0.04], device=env.unwrapped.device, dtype=torch.float32)  # Open
            elif episode_progress < 0.4:  # Close gripper
                target_pos = torch.tensor([0.3, 0.0, 0.05], device=env.unwrapped.device, dtype=torch.float32)
                target_orn = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.unwrapped.device, dtype=torch.float32)
                gripper_cmd = torch.tensor([-0.04, -0.04], device=env.unwrapped.device, dtype=torch.float32)  # Close
            elif episode_progress < 0.6:  # Lift and move to target
                target_pos = torch.tensor([0.1, 0.3, 0.2], device=env.unwrapped.device, dtype=torch.float32)
                target_orn = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.unwrapped.device, dtype=torch.float32)
                gripper_cmd = torch.tensor([-0.04, -0.04], device=env.unwrapped.device, dtype=torch.float32)  # Closed
            elif episode_progress < 0.7:  # Descend to place
                target_pos = torch.tensor([0.1, 0.3, 0.05], device=env.unwrapped.device, dtype=torch.float32)
                target_orn = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.unwrapped.device, dtype=torch.float32)
                gripper_cmd = torch.tensor([-0.04, -0.04], device=env.unwrapped.device, dtype=torch.float32)  # Closed
            else:  # Open gripper to place
                target_pos = torch.tensor([0.1, 0.3, 0.05], device=env.unwrapped.device, dtype=torch.float32)
                target_orn = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.unwrapped.device, dtype=torch.float32)
                gripper_cmd = torch.tensor([0.04, 0.04], device=env.unwrapped.device, dtype=torch.float32)  # Open
                
            # Combine position, orientation and gripper command into action
            action = torch.cat([target_pos, target_orn[0:3], gripper_cmd], dim=-1).unsqueeze(0)  # Shape: [1, 8]
            
            # Perform step
            obs_dict, rew, terminated, truncated, info = env.step(action)
            
            # Store data
            episode_data['actions'].append(action[0].cpu().numpy())
            for key, value in obs_dict['policy'].items():
                if key in episode_data['obs']:
                    episode_data['obs'][key].append(value[0].cpu().numpy())
            
            # Store states
            state_dict = env.get_state()
            for obj_type in ['articulation', 'rigid_object']:
                for entity in state_dict[obj_type]:
                    for prop in state_dict[obj_type][entity]:
                        episode_data['states'][obj_type][entity][prop].append(state_dict[obj_type][entity][prop][0].cpu().numpy())
            
            step_count += 1
            
            # Check termination
            if any(terminated) or any(truncated):
                break
        
        # Convert lists to numpy arrays
        episode_data['actions'] = np.array(episode_data['actions'])
        for key, value in episode_data['obs'].items():
            if len(value) > 0:
                episode_data['obs'][key] = np.array(value)
        
        for obj_type in ['articulation', 'rigid_object']:
            for entity in episode_data['states'][obj_type]:
                for prop in episode_data['states'][obj_type][entity]:
                    episode_data['states'][obj_type][entity][prop] = np.array(episode_data['states'][obj_type][entity][prop])
        
        all_episode_data.append(episode_data)
        
        # Save individual episode
        episode_file_path = os.path.join(output_dir, f"demo_{episode_idx}.h5")
        save_demo_data_to_hdf5({f'data/demo_{episode_idx}': episode_data}, episode_file_path)
        print(f"[INFO] Saved episode {episode_idx} to {episode_file_path}")
    
    # Close environment
    env.close()
    print(f"[INFO] Completed collecting {num_episodes} episodes of demonstration data")


def parse_app_launch_args():
    """Simple argument parser since the original function might not be available."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate demonstration data for Piper robot.")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to collect")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum steps per episode")
    parser.add_argument("--output_dir", type=str, default="./demonstrations", help="Output directory for demonstrations")
    
    return parser.parse_args()


def main():
    """Main function to run the demonstration data collection."""
    # Parse arguments
    args_cli = parse_app_launch_args()
    
    # Create logs directory
    log_dir = os.path.join("logs", "demonstration_generation")
    os.makedirs(log_dir, exist_ok=True)
    
    print("[INFO] Starting demonstration data collection for Piper robot...")
    
    # Collect demonstration data
    collect_demonstration_data(
        num_episodes=args_cli.num_episodes,
        max_steps=args_cli.max_steps,
        output_dir=args_cli.output_dir
    )
    
    print("[INFO] Demonstration data collection completed!")


if __name__ == "__main__":
    main()