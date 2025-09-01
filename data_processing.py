import numpy as np
import yaml
from typing import Dict, Any, Tuple
import pickle
from pathlib import Path

def load_config(config_path = "config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config




def load_npz_data(data_dir="data"):
    data_dir = Path(data_dir)
    npz_files = sorted(data_dir.glob("*.npz"))
    all_states, all_inputs = [], []
    timestamps = None

    for file_path in npz_files:
        data = np.load(file_path)
        all_states.append(data['states'])
        all_inputs.append(data['inputs'])
        if timestamps is None:
            timestamps = data['timestamps']

    states = np.concatenate(all_states, axis=1)
    inputs = np.concatenate(all_inputs, axis=1)
    return states, inputs, timestamps




def validate_data(states, inputs, config):
    if np.any(np.isnan(states)) or np.any(np.isnan(inputs)):
        raise ValueError("Data contains NaN values")
    if np.any(np.isinf(states)) or np.any(np.isinf(inputs)):
        raise ValueError("Data contains infinite values")
    if config.get('verbose', True):
        print("Data validation passed")




def convert_to_relative_pos(states):
    for robot_idx in range(states.shape[1]):
        x_initial = states[0, robot_idx, 0]
        y_initial = states[0, robot_idx, 1]
        states[:, robot_idx, 0] -= x_initial
        states[:, robot_idx, 1] -= y_initial
    return states




def split_data(states, inputs, timestamps, config):
    train_ratio = config["data"]["train_ratio"]
    val_ratio = config["data"]["val_ratio"]
    test_ratio = config["data"]["test_ratio"]

    total_robots  = states.shape[1]
    train_count = int(total_robots * train_ratio)
    val_count = int(total_robots * val_ratio)
    test_count = total_robots - train_count - val_count

    states_relative = convert_to_relative_pos(states)

    np.random.seed(42)
    all_robot_ids = np.arange(total_robots)
    np.random.shuffle(all_robot_ids)

    test_robots = all_robot_ids[:test_count].tolist()
    val_robots = all_robot_ids[test_count:test_count + val_count].tolist()
    train_robots = all_robot_ids[test_count + val_count:].tolist()

    train_states = states_relative[:, train_robots, :]
    train_inputs = inputs[:, train_robots, :]
    train_timestamps = timestamps

    val_states = states_relative[:, val_robots, :]
    val_inputs = inputs[:, val_robots, :]
    val_timestamps = timestamps

    test_states = states_relative[:, test_robots, :]
    test_inputs = inputs[:, test_robots, :]
    test_timestamps = timestamps

    return test_states, test_inputs, test_timestamps, \
           train_states, train_inputs, train_timestamps, \
           val_states, val_inputs, val_timestamps, \
           train_robots, val_robots, test_robots    




def create_multistep_samples(states, inputs, timestamps, config):
    n_steps = config["data"]["num_multi_step_predictions"]
    num_timesteps, num_robots, _ = states.shape
    samples = []
    for r in range(num_robots):
        for t in range(num_timesteps - n_steps + 1):
            window_states = states[t:t+n_steps, r, :]
            window_inputs = inputs[t:t+n_steps, r, :]
            window = np.concatenate([window_states, window_inputs], axis=1).T
            samples.append(window)
    samples_array = np.stack(samples, axis=0)
    return samples_array

def process_data(config_path="config.yaml"):


    config = load_config(config_path)
    input_dir = Path(config["data"]["input_dir"])
    base_name = input_dir.name
    output_dir = Path("processed_data") / base_name
    output_dir.mkdir(parents=True, exist_ok=True)
    # Load raw data
    states, inputs, timestamps = load_npz_data(config["data"]["input_dir"])
    dt = float(timestamps[1] - timestamps[0])
    print(f"Calculated dt from timestamps: {dt}")
    config['data']['dt'] = dt
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    validate_data(states, inputs, config)
    test_states, test_inputs, test_timestamps, \
    train_states, train_inputs, train_timestamps, \
    val_states, val_inputs, val_timestamps, \
    train_robots, val_robots, test_robots = split_data(states, inputs, timestamps, config)


    train_samples = create_multistep_samples(train_states, train_inputs, train_timestamps, config)
    val_samples = create_multistep_samples(val_states, val_inputs, val_timestamps, config)
    test_samples = create_multistep_samples(test_states, test_inputs, test_timestamps, config)

    np.savez(output_dir / "train_data.npz", samples=train_samples)
    np.savez(output_dir / "val_data.npz", samples=val_samples)
    np.savez(output_dir / "test_data.npz", samples=test_samples)
    print("Data processing complete. Processed files saved to:", output_dir)

if __name__ == "__main__":
    process_data("config.yaml")
    # Print normalization parameters for yaw to verify
    # normalization_params = np.load("processed_data/train_data.npz")
    # with open("processed_data/normalization_params.json", "r") as f:
    #     norm = json.load(f)
    # print("Yaw normalization check:")
    # print("Yaw mean:", norm['state_mean'][2])
    # print("Yaw std:", norm['state_std'][2])
    # print("Full state mean:", norm['state_mean'])
    # print("Full state std:", norm['state_std'])
  





