import numpy as np
import yaml
from typing import Dict, Any, Tuple
import json
import pickle
from pathlib import Path

def load_config(config_path = "config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config





def load_npz_data(data_dir="data"):
    """
    Loads and concatenates all NPZ files in the given directory.
    Returns:
        states: ndarray of shape (timesteps, total_robots, 7)
            - Each robot trajectory is kept separate.
            - State variables: [x_pos, y_pos, yaw, steering, velocity, side_slip, yaw_rate]
        inputs: ndarray of shape (timesteps, total_robots, 2)
            - Input variables: [acceleration, steering_rate]
        timestamps: ndarray of shape (timesteps,)
            - Time values for each timestep.
    """
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
    """
    Converts absolute positions (x_pos, y_pos) to relative positions (Δx, Δy) for each robot.
    Args:
        states: ndarray (timesteps, num_robots, 7)
    Returns:
        states: ndarray (timesteps, num_robots, 7) with Δx, Δy
    """
    for robot_idx in range(states.shape[1]):
        x_initial = states[0, robot_idx, 0]  #x_pos
        y_initial = states[0, robot_idx, 1]  #y_pos
        
        states[:, robot_idx, 0] -= x_initial
        states[:, robot_idx, 1] -= y_initial

    return states


def split_data(states, inputs, timestamps, config):
    """
    Splits the data by robot trajectories into train/val/test sets.
    Args:
        states: ndarray (timesteps, total_robots, 7)
        inputs: ndarray (timesteps, total_robots, 2)
        timestamps: ndarray (timesteps,)
        config: configuration dictionary
    Returns:
        train_states: (timesteps, train_count, 7)
        train_inputs: (timesteps, train_count, 2)
        train_timestamps: (timesteps,)
        val_states: (timesteps, val_count, 7)
        val_inputs: (timesteps, val_count, 2)
        val_timestamps: (timesteps,)
        test_states: (timesteps, test_count, 7)
        test_inputs: (timesteps, test_count, 2)
        test_timestamps: (timesteps,)
        train_robots, val_robots, test_robots: lists of robot indices for each split
    """
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

           


def compute_normalization_params(train_states, train_inputs):
    """
    Compute mean and std for z-score normalization from training data only.
    Args:
        train_states: ndarray (timesteps, train_count, 7)
        train_inputs: ndarray (timesteps, train_count, 2)
    Returns:
        Dictionary with normalization parameters
        - 'state_mean': ndarray (7,)
        - 'state_std': ndarray (7,)
        - 'input_mean': ndarray (2,)
        - 'input_std': ndarray (2,)
    """
    all_states = train_states.reshape(-1, train_states.shape[-1])  # (timesteps*train_count, 7)
    all_inputs = train_inputs.reshape(-1, train_inputs.shape[-1])  # (timesteps*train_count, 2)
    state_mean = np.mean(all_states, axis=0)
    state_std = np.std(all_states, axis=0)
    # Do not normalize yaw (index 2)
    state_mean[2] = 0.0
    state_std[2] = 1.0
    return {
        'state_mean': state_mean,
        'state_std': state_std,
        'input_mean': np.mean(all_inputs, axis=0),
        'input_std': np.std(all_inputs, axis=0)
    }



def apply_normalization(data, mean, std):
    return (data - mean) / std


def normalize_dataset(train_states, train_inputs, val_states, val_inputs, test_states, test_inputs, normalization_params, config):
    """
    Normalize the dataset using precomputed mean and std.
    Args:
        train_states: ndarray (timesteps, train_count, 7)
        train_inputs: ndarray (timesteps, train_count, 2)
        val_states: ndarray (timesteps, val_count, 7)
        val_inputs: ndarray (timesteps, val_count, 2)
        test_states: ndarray (timesteps, test_count, 7)
        test_inputs: ndarray (timesteps, test_count, 2)
        normalization_params: dict with 'state_mean', 'state_std', 'input_mean', 'input_std'
    Returns:
        Normalized datasets
    """ 


    train_states_norm = apply_normalization(train_states, normalization_params['state_mean'], normalization_params['state_std'])
    train_inputs_norm = apply_normalization(train_inputs, normalization_params['input_mean'], normalization_params['input_std'])

    val_states_norm = apply_normalization(val_states, normalization_params['state_mean'], normalization_params['state_std'])
    val_inputs_norm = apply_normalization(val_inputs, normalization_params['input_mean'], normalization_params['input_std'])

    test_states_norm = apply_normalization(test_states, normalization_params['state_mean'], normalization_params['state_std'])
    test_inputs_norm = apply_normalization(test_inputs, normalization_params['input_mean'], normalization_params['input_std'])

    noise_std = config["data"]["noise_std"]

    if noise_std > 0:
        train_states_norm += np.random.normal(0, noise_std, train_states_norm.shape)
        train_inputs_norm += np.random.normal(0, noise_std, train_inputs_norm.shape)

    return train_states_norm, train_inputs_norm, val_states_norm, val_inputs_norm, test_states_norm, test_inputs_norm




def create_multistep_samples(states, inputs, timestamps, config):
    """
    Create a 3D array for multi-step prediction:
    Shape: (num_samples, num_states + num_control_inputs, n_steps)
    For each robot, slide a window of n_steps over its trajectory.
    Each sample contains states and inputs for n_steps timesteps, concatenated as [state, input] per timestep.
    Args:
        states: ndarray (timesteps, num_robots, 7)
        inputs: ndarray (timesteps, num_robots, 2)
        timestamps: ndarray (timesteps,)
        n_steps: int, window size
    Returns:
        samples_array: ndarray (num_samples, 9, n_steps)
    """

    n_steps = config["data"]["num_multi_step_predictions"]
    num_timesteps, num_robots, _ = states.shape
    samples = []
    for r in range(num_robots):
        for t in range(num_timesteps - n_steps + 1):
            # Collect states and inputs for n_steps window
            window_states = states[t:t+n_steps, r, :]  # (n_steps, 7)
            window_inputs = inputs[t:t+n_steps, r, :]  # (n_steps, 2)
            # Concatenate state and input for each timestep
            window = np.concatenate([window_states, window_inputs], axis=1).T  # (9, n_steps)
            samples.append(window)
    samples_array = np.stack(samples, axis=0)  # (num_samples, 9, n_steps)
    return samples_array

def process_data(config_path="config.yaml"):
    """
    Main function to process the dataset.
    Loads data, splits into train/val/test, normalizes, and creates multi-step samples.
    Saves processed data to disk.
    """
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
    
    # Validate data
    validate_data(states, inputs, config)
    
    # Split data
    test_states, test_inputs, test_timestamps, \
    train_states, train_inputs, train_timestamps, \
    val_states, val_inputs, val_timestamps, \
    train_robots, val_robots, test_robots = split_data(states, inputs, timestamps, config)
    
    # Compute normalization parameters
    normalization_params = compute_normalization_params(train_states, train_inputs)
    
    # Normalize datasets
    train_states_norm, train_inputs_norm, val_states_norm, val_inputs_norm, test_states_norm, test_inputs_norm = \
        normalize_dataset(train_states, train_inputs,
                          val_states, val_inputs,
                          test_states, test_inputs,
                          normalization_params,
                          config)
    
    # Create multi-step samples
    # train_samples = create_multistep_samples(train_states_norm, train_inputs_norm, train_timestamps, config)
    # val_samples = create_multistep_samples(val_states_norm, val_inputs_norm, val_timestamps, config)
    # test_samples = create_multistep_samples(test_states_norm, test_inputs_norm, test_timestamps, config)


    # Create multi-step samples unormalized
    train_samples = create_multistep_samples(train_states, train_inputs, train_timestamps, config)
    val_samples = create_multistep_samples(val_states, val_inputs, val_timestamps, config)
    test_samples = create_multistep_samples(test_states, test_inputs, test_timestamps, config)

    # Save processed arrays in processed_data directory
    np.savez(output_dir / "train_data.npz", samples=train_samples)
    np.savez(output_dir / "val_data.npz", samples=val_samples)
    np.savez(output_dir / "test_data.npz", samples=test_samples)

    # Convert numpy arrays in normalization_params to lists for JSON serialization
    normalization_params_serializable = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in normalization_params.items()}
    with open(output_dir / "normalization_params.json", 'w') as f:
        json.dump(normalization_params_serializable, f)

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
  





