import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pickle
import yaml
from pathlib import Path
from models import HybridODE

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_test_data(processed_dir="processed_data"):
    proc = Path(processed_dir)
    test_samples = np.load(proc / "test_data.npz")["samples"]
    return test_samples

def load_trained_params(params_path="results/model_params.pkl"):
    with open(params_path, "rb") as f:
        return pickle.load(f)

STATE_NAMES = ["delta_x", "delta_y", "yaw", "steering", "velocity", "side_slip", "yaw_rate"]

def plot_trajectory(pred_traj, true_traj, inputs_seq, dt, sample_idx):
    """Plot a single trajectory comparison"""
    n_steps = pred_traj.shape[0]
    t_vec = np.arange(n_steps) * dt
    
    # Create subplot for states and controls
    fig, axs = plt.subplots(9, 1, figsize=(12, 16), sharex=True)
    
    # Plot all 7 states
    for i, name in enumerate(STATE_NAMES):
        axs[i].plot(t_vec, true_traj[:, i], label=f"True {name}", color='blue')
        axs[i].plot(t_vec, pred_traj[:, i], label=f"Pred {name}", color='red', linestyle='--')
        axs[i].set_ylabel(name)
        axs[i].legend()
        axs[i].grid(True, alpha=0.3)
    
    # Plot control inputs
    axs[7].plot(t_vec, inputs_seq[:, 0], color='orange', label="Acceleration")
    axs[7].set_ylabel("Acceleration")
    axs[7].legend()
    axs[7].grid(True, alpha=0.3)
    
    axs[8].plot(t_vec, inputs_seq[:, 1], color='green', label="Steering Rate")
    axs[8].set_ylabel("Steering Rate")
    axs[8].legend()
    axs[8].grid(True, alpha=0.3)
    
    axs[-1].set_xlabel("Time [s]")
    plt.suptitle(f"Trajectory Prediction - Sample {sample_idx}")
    plt.tight_layout()
    
    # Create 2D trajectory plot
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    ax2.plot(true_traj[:, 0], true_traj[:, 1], label="True Trajectory", color='blue', linewidth=2)
    ax2.plot(pred_traj[:, 0], pred_traj[:, 1], label="Predicted Trajectory", color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel("delta_x [m]")
    ax2.set_ylabel("delta_y [m]")
    ax2.set_title(f"2D Trajectory - Sample {sample_idx}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis("equal")
    
    plt.show()

def main():
    # Load configuration and data
    config = load_config()
    test_samples = load_test_data()
    params = load_trained_params()
    
    # Initialize model
    model = HybridODE(config)
    dt = config['data']['dt']
    
    print(f"Loaded test data with shape: {test_samples.shape}")
    print(f"Using dt = {dt}")
    
    # Select 5 random samples
    np.random.seed(42)  # For reproducibility
    n_samples = test_samples.shape[0]
    selected_indices = np.random.choice(n_samples, size=5, replace=False)
    
    print(f"Selected sample indices: {selected_indices}")
    
    for i, sample_idx in enumerate(selected_indices):
        print(f"\nProcessing sample {i+1}/5 (index {sample_idx})...")
        
        # Extract sample data
        sample = test_samples[sample_idx]  # shape: (9, n_steps)
        state_dim = 7
        
        # Extract initial state and input sequence
        initial_state = sample[:state_dim, 0]  # shape: (7,)
        inputs_sequence = sample[state_dim:, :].T  # shape: (n_steps, 2)
        true_trajectory = sample[:state_dim, :].T  # shape: (n_steps, 7)
        
        # Convert to JAX arrays
        initial_state = jnp.array(initial_state)
        inputs_sequence = jnp.array(inputs_sequence)
        
        # Predict trajectory
        pred_trajectory = model.predict_trajectory(params, initial_state, inputs_sequence, dt)
        
        # Convert back to numpy for plotting
        pred_trajectory = np.array(pred_trajectory)
        
        print(f"Prediction shape: {pred_trajectory.shape}")
        print(f"True trajectory shape: {true_trajectory.shape}")
        
        # Calculate and print some basic metrics
        mse = np.mean((pred_trajectory - true_trajectory) ** 2)
        pos_error = np.sqrt((pred_trajectory[:, 0] - true_trajectory[:, 0])**2 + 
                           (pred_trajectory[:, 1] - true_trajectory[:, 1])**2)
        final_pos_error = pos_error[-1]
        
        print(f"MSE: {mse:.6f}")
        print(f"Final position error: {final_pos_error:.4f} m")
        
        # Plot the trajectory
        plot_trajectory(pred_trajectory, true_trajectory, inputs_sequence, dt, sample_idx)

if __name__ == "__main__":
    main()
