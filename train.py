import jax
import jax.numpy as jnp
import numpy as np
import yaml
import pickle
import wandb
from pathlib import Path
from functools import partial
import optax
from flax.training import train_state
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import HybridODE, create_train_state

class PlateauScheduler:
    """Learning rate scheduler that reduces LR when validation loss plateaus."""
    def __init__(self, initial_lr, factor=0.5, patience=10, min_lr=1e-7, threshold=1e-4):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.threshold = threshold
        self.best_loss = float('inf')
        self.wait = 0
        
    def step(self, val_loss):
        """Update learning rate based on validation loss."""
        if val_loss < self.best_loss - self.threshold:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            old_lr = self.current_lr
            self.current_lr = max(self.current_lr * self.factor, self.min_lr)
            self.wait = 0
            if self.current_lr < old_lr:
                print(f"Reducing learning rate from {old_lr:.6f} to {self.current_lr:.6f}")
                return True  # LR was reduced
        return False  # LR was not reduced
    
    def get_lr(self):
        return self.current_lr

print(jax.devices())



def load_data(processed_dir="processed_data"):
    """Loads processed train/val/test samples as JAX arrays."""
    processed_dir = Path(processed_dir)
    train_data = np.load(processed_dir / "train_data.npz")
    val_data = np.load(processed_dir / "val_data.npz")
    test_data = np.load(processed_dir / "test_data.npz")
    train_samples = jnp.array(train_data["samples"])
    val_samples = jnp.array(val_data["samples"])
    test_samples = jnp.array(test_data["samples"])
    return train_samples, val_samples, test_samples




def create_minibatches(samples, batch_size, shuffle=True, key=None):
    """Generator for minibatches from a JAX array."""
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




def yaw_error(pred, true):
    """Shortest-path angular error for yaw angle."""
    return ((pred - true + jnp.pi) % (2 * jnp.pi)) - jnp.pi



# def side_slip_error(pred, true):
#     """Shortest-path angular error for side slip angle."""
#     return ((pred - true + jnp.pi/2) % (2 * jnp.pi)) - jnp.pi/2

def loss_function(pred_traj, true_traj):
    """Mean squared error for all states with proper yaw angle handling."""
    # Handle yaw angle (index 2) with circular distance
    yaw_pred = pred_traj[..., 2]
    yaw_true = true_traj[..., 2]
    yaw_loss = jnp.mean(yaw_error(yaw_pred, yaw_true) ** 2)


    # side_slip_pred = pred_traj[..., 5]
    # side_slip_true = true_traj[..., 5]
    # side_slip_loss = jnp.mean(side_slip_error(side_slip_pred, side_slip_true) ** 2)
    
    # Handle all other states with regular MSE
    other_indices = jnp.array([0, 1, 3, 4, 6])  # all except yaw (index 2)
    other_pred = pred_traj[..., other_indices]
    other_true = true_traj[..., other_indices]
    other_loss = jnp.mean((other_pred - other_true) ** 2)
    
    # Combine losses
    total_loss = other_loss + yaw_loss
    # total_loss += side_slip_loss
    return total_loss

# def loss_function(pred_traj, true_traj):
#     #loss over the last 3 states (velocity, side slip, yaw rate)
#     # This is a simplified version, you can adjust it based on your requirements
#     pred = pred_traj[..., -3:]  # last 3 states
#     true = true_traj[..., -3:]  # last 3 states
#     return jnp.mean((pred - true) ** 2)
    





def train_step(train_state, batch, model, dt):
    """Single training step: predict, compute loss, update params."""
    state_dim = 7
    initial_state = batch[:, :state_dim, 0]
    inputs_sequence = batch[:, state_dim:, :].transpose(0, 2, 1)
    true_traj = batch[:, :state_dim, :].transpose(0, 2, 1)

    def loss_fn(params):
        pred_traj = model.predict_batch_trajectories(
            params, initial_state, inputs_sequence, dt, training=True  # Add training=True
        )
        return loss_function(pred_traj, true_traj)

    loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
    grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
    new_train_state = train_state.apply_gradients(grads=grads)
    return new_train_state, loss




train_step = jax.jit(train_step, static_argnames=["model"])




def validate(train_state, val_samples, model, dt, batch_size):
    """Average validation loss over all validation samples."""
    losses = []
    for batch in create_minibatches(val_samples, batch_size, shuffle=False):
        state_dim = 7
        initial_state = batch[:, :state_dim, 0]
        inputs_sequence = batch[:, state_dim:, :].transpose(0, 2, 1)
        true_traj = batch[:, :state_dim, :].transpose(0, 2, 1)
        pred_traj = model.predict_batch_trajectories(
            train_state.params, initial_state, inputs_sequence, dt, training=False  # Add training=False
        )
        loss = loss_function(pred_traj, true_traj)
        losses.append(loss)
    return float(jnp.mean(jnp.array(losses)))




def run_training_loop(train_samples, val_samples, model, train_state, dt, epochs, batch_size, validation_interval=10, early_stopping_patience=20, lr_scheduler=None, learning_rate=None, weight_decay=0.0):
    wandb.init(project="hybrid_ode_training", config={
        "epochs": epochs,
        "batch_size": batch_size,
        "validation_interval": validation_interval,
        "early_stopping_patience": early_stopping_patience,
        "dt": dt,
        "initial_learning_rate": learning_rate
    })
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
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
        
        # Get current learning rate
        current_lr = lr_scheduler.get_lr() if lr_scheduler is not None else learning_rate
        
        wandb.log({
            "train_loss": avg_train_loss,
            "epoch": epoch+1,
            "learning_rate": current_lr
        })
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f} - LR: {current_lr:.6f}")
        
        if (epoch+1) % validation_interval == 0:
            val_loss = validate(train_state, val_samples, model, dt, batch_size)
            val_losses.append(val_loss)
            wandb.log({"val_loss": val_loss, "epoch": epoch+1})
            print(f"Validation Loss: {val_loss:.6f}")
            
            # Update learning rate scheduler if using plateau scheduling
            if lr_scheduler is not None:
                lr_reduced = lr_scheduler.step(val_loss)
                if lr_reduced:
                    # Create new optimizer with updated learning rate
                    new_lr = lr_scheduler.get_lr()
                    print(f"Creating new optimizer with learning rate: {new_lr:.6f}")
                    
                    # Determine which optimizer to use based on weight decay
                    if weight_decay > 0:
                        new_optimizer = optax.adamw(learning_rate=new_lr, weight_decay=weight_decay)
                    else:
                        new_optimizer = optax.adam(learning_rate=new_lr)
                    
                    # Create new train state with updated optimizer
                    train_state = train_state.replace(tx=new_optimizer)
            
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



    

if __name__ == "__main__":
    train_samples, val_samples, test_samples = load_data()
    print(f"train_samples device: {train_samples.device}")
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

    # Initialize model and train state
    model = HybridODE(config)
    print("Using HybridODE model")

    # Initialize plateau-based learning rate scheduler
    plateau_config = config['training'].get('plateau_scheduler', {})
    lr_scheduler = PlateauScheduler(
        initial_lr=learning_rate,
        factor=float(plateau_config.get('factor', 0.5)),
        patience=int(plateau_config.get('patience', 5)),
        min_lr=float(plateau_config.get('min_lr', 1e-7)),
        threshold=float(plateau_config.get('threshold', 1e-4))
    )
    print(f"Using plateau-based learning rate scheduling:")
    print(f"  Initial LR: {learning_rate}")
    print(f"  Factor: {lr_scheduler.factor} (LR will be multiplied by this when plateau detected)")
    print(f"  Patience: {lr_scheduler.patience} (validation intervals to wait before reducing)")
    print(f"  Min LR: {lr_scheduler.min_lr}")
    print(f"  Threshold: {lr_scheduler.threshold} (minimum improvement to reset patience)")
    
    train_state = create_train_state(model, learning_rate, key, weight_decay)

    print(f"model params device: {jax.tree_util.tree_leaves(train_state.params)[0].device}")
    train_state, train_losses, val_losses = run_training_loop(
        train_samples, val_samples, model, train_state, dt,
        epochs=epochs, batch_size=batch_size,
        validation_interval=validation_interval,
        early_stopping_patience=early_stopping_patience,
        lr_scheduler=lr_scheduler,
        learning_rate=learning_rate,
        weight_decay=weight_decay
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
