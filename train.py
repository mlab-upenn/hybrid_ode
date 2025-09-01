import jax
import jax.numpy as jnp
import numpy as np
import yaml
import pickle
import wandb
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
from functools import partial
import optax
from flax.training import train_state
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from models import HybridODE, create_train_state, Node


# print(jax.devices())
def load_data(processed_dir="processed_data"):
    """
    Loads processed train/val/test samples and normalization parameters as JAX arrays.
    Returns:
        train_samples: jnp.ndarray (num_samples, 9, n_steps)
        val_samples: jnp.ndarray (num_samples, 9, n_steps)
        test_samples: jnp.ndarray (num_samples, 9, n_steps)
        normalization_params: dict with 'state_mean', 'state_std', 'input_mean', 'input_std' as jnp.ndarray
    """
    processed_dir = Path(processed_dir)
    train_data = np.load(processed_dir / "train_data.npz")
    val_data = np.load(processed_dir / "val_data.npz")
    test_data = np.load(processed_dir / "test_data.npz")

    train_samples = jnp.array(train_data["samples"])
    val_samples = jnp.array(val_data["samples"])
    test_samples = jnp.array(test_data["samples"])


    return train_samples, val_samples, test_samples





def create_minibatches(samples, batch_size, shuffle=True, key=None):
    """
    Generator for minibatches from a JAX array.
    Args:
        samples: jnp.ndarray of shape (num_samples, 9, n_steps)
        batch_size: int
        shuffle: bool, whether to shuffle samples
        key: jax.random.PRNGKey, for reproducible shuffling (optional)
    Yields:
        batch: jnp.ndarray of shape (batch_size, 9, n_steps)
    """
    num_samples = samples.shape[0]
    indices = jnp.arange(num_samples)
    if shuffle:
        if key is None:
            key = jax.random.PRNGKey(np.random.randint(1e6))
        indices = jax.random.permutation(key, indices)
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_indices = indices[start:end]
        yield samples[batch_indices]




def loss_function(pred_traj, true_traj):
    """
    Computes mean squared error only for the last 3 states (velocity, side_slip, yaw_rate).
    Args:
        pred_traj: jnp.ndarray, shape (batch_size, n_steps, state_dim)
        true_traj: jnp.ndarray, shape (batch_size, n_steps, state_dim)
    Returns:
        loss: scalar, mean loss over batch, time, and selected states
    """
    # Select last 3 states: indices 4, 5, 6
    pred_last3 = pred_traj[..., 4:7]
    true_last3 = true_traj[..., 4:7]
    squared_error = (pred_last3 - true_last3) ** 2
    return jnp.mean(squared_error)









def train_step(train_state, batch, model, dt):
    """
    Performs a single training step: predicts trajectory, computes loss, gradients, and updates parameters.
    Args:
        train_state: Flax TrainState
        batch: jnp.ndarray of shape (batch_size, 9, n_steps)
        model: HybridODE instance
        dt: float, time step
    Returns:
        new_train_state: updated TrainState
        loss: scalar
    """
    state_dim = 7
    initial_state = batch[:, :state_dim, 0]
    inputs_sequence = batch[:, state_dim:, :].transpose(0, 2, 1)
    true_traj = batch[:, :state_dim, :].transpose(0, 2, 1)

    def loss_fn(params):
        pred_traj = model.predict_batch_trajectories(params, initial_state, inputs_sequence, dt)
        return loss_function(pred_traj, true_traj)

    loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
    # Gradient clipping
    grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
    new_train_state = train_state.apply_gradients(grads=grads)
    return new_train_state, loss

train_step = jax.jit(train_step, static_argnames=["model"])


def validate(train_state, val_samples, model, dt, batch_size):
    """
    Computes average validation loss over all validation samples.
    Args:
        train_state: Flax TrainState
        val_samples: jnp.ndarray (num_samples, 9, n_steps)
        model: HybridODE instance
        dt: float, time step
        batch_size: int
    Returns:
        avg_loss: float
    """
    losses = []
    for batch in create_minibatches(val_samples, batch_size, shuffle=False):
        state_dim = 7
        initial_state = batch[:, :state_dim, 0]
        inputs_sequence = batch[:, state_dim:, :].transpose(0, 2, 1)
        true_traj = batch[:, :state_dim, :].transpose(0, 2, 1)
        pred_traj = model.predict_batch_trajectories(train_state.params, initial_state, inputs_sequence, dt)
        loss = loss_function(pred_traj, true_traj)
        losses.append(loss)
    return float(jnp.mean(jnp.array(losses)))


def run_training_loop(train_samples, val_samples, model, train_state, dt, epochs, batch_size, validation_interval=10, early_stopping_patience=20, lr_schedule=None, learning_rate=None):
    wandb.init(project="hybrid_ode_training", config={
        "epochs": epochs,
        "batch_size": batch_size,
        "validation_interval": validation_interval,
        "early_stopping_patience": early_stopping_patience,
        "dt": dt
    })
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    # Path to save model parameters
    params_dir = Path("results") / base_name
    params_dir.mkdir(parents=True, exist_ok=True)
    params_path = params_dir / "model_params.pkl"

    for epoch in range(epochs):
        epoch_losses = []
        batch_iter = tqdm(create_minibatches(train_samples, batch_size, shuffle=True),
                         desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch_idx, batch in enumerate(batch_iter):
            train_state, loss = train_step(train_state, batch, model, dt)
            epoch_losses.append(float(loss))
            batch_iter.set_postfix({"batch_loss": float(loss)})

        avg_train_loss = np.mean(epoch_losses)
        train_losses.append(avg_train_loss)
     
        current_step = int(train_state.step)
        if lr_schedule is not None:
            current_lr = float(lr_schedule(current_step))
        elif learning_rate is not None:
            current_lr = float(learning_rate)
        else:
            current_lr = None  
        wandb.log({
            "train_loss": avg_train_loss,
            "epoch": epoch+1,
            "learning_rate": current_lr
        })
        if (epoch+1) % validation_interval == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}")
            val_loss = validate(train_state, val_samples, model, dt, batch_size)
            val_losses.append(val_loss)
            wandb.log({"val_loss": val_loss, "epoch": epoch+1})
            print(f"Validation Loss: {val_loss:.6f}")

            # Autosave model parameters
            with open(params_path, "wb") as f:
                pickle.dump(train_state.params, f)
            print(f"Autosaved model parameters to {params_path}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    wandb.finish()
    return train_state, train_losses, val_losses


def get_current_lr(train_state):

    opt_state = train_state.opt_state

    if hasattr(opt_state, 'inner_state'):  
        step = opt_state.inner_state[0].count
    else:
        step = opt_state[0].count
    
    return step  


if __name__ == "__main__":
    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    dt = config['data']['dt']
    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    validation_interval = config['training']['validation_interval']
    early_stopping_patience = config['training']['early_stopping_patience']
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    key = jax.random.PRNGKey(config['random_seed'])

    input_dir = Path(config["data"]["input_dir"])
    base_name = input_dir.name
    output_dir = Path("processed_data") / base_name
    train_samples, val_samples, test_samples = load_data(processed_dir=output_dir)
    
    # Print device info for arrays and params
    print(f"train_samples device: {train_samples.device}")    

    # Initialize model and train state
    if config['model_type'] == 'HybridODE':
        model = HybridODE(config)
        print("Using HybridODE model")
    elif config['model_type'] == 'Node':
        model = Node(config)
        print("Using Node model")
    else:
        raise ValueError(f"Unknown model type HybridODE or Node: {config['model_type']}")

    # Prepare learning rate schedule if using weight decay (AdamW)
    lr_schedule = None
    if weight_decay > 0:
        lr_schedule = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=100,
            decay_rate=0.99,
            staircase=True
        )
        train_state = create_train_state(model, learning_rate, key, weight_decay)
    else:
        train_state = create_train_state(model, learning_rate, key, weight_decay)

    print(f"model params device: {jax.tree_util.tree_leaves(train_state.params)[0].device}")
    # Run training loop
    train_state, train_losses, val_losses = run_training_loop(
        train_samples, val_samples, model, train_state, dt,
        epochs=epochs, batch_size=batch_size,
        validation_interval=validation_interval,
        early_stopping_patience=early_stopping_patience,
        lr_schedule=lr_schedule,
        learning_rate=learning_rate
    )
    print("Training complete.")
    print(f"Best validation loss: {min(val_losses) if val_losses else 'N/A'}")

    # Save trained model parameters to disk
    results_dir = Path(config['data'].get('output_dir', 'results'))
    results_dir.mkdir(exist_ok=True)
    params_path = results_dir / "model_params.pkl"
    with open(params_path, "wb") as f:
        pickle.dump(train_state.params, f)
    print(f"Saved trained model parameters to {params_path}")
