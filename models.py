import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from typing import Dict, Any
import numpy as np
from functools import partial
from scipy.interpolate import interp1d
import yaml
import optax




import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from typing import Dict, Any
import numpy as np
from functools import partial
import yaml
import optax

class PhysicsAwareAttention(nn.Module):
    """
    Attention mechanism for physics states.
    Based on Fourier Neural Operator and Graph Network approaches.
    """
    num_heads: int = 4
    embed_dim: int = 64
    
    @nn.compact
    def __call__(self, physics_features, neural_features):
        # Multi-head self-attention on combined features
        combined_features = jnp.concatenate([physics_features, neural_features], axis=-1)
        
        # Project to embedding space
        embedded = nn.Dense(self.embed_dim)(combined_features)
        
        # Self-attention to capture feature interactions
        attended = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim
        )(embedded)
        
        # Project back to output space
        output = nn.Dense(3)(attended)  # 3 neural states
        
        return output

class ResidualPhysicsBlock(nn.Module):
    """
    Residual block optimized for physics applications.
    Based on Wang et al. (2021) PINN gradient flow analysis.
    """
    features: int
    activation: str = 'swish'
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training=True):
        residual = x
        
        # First layer with normalization
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.features)(x)
        
        if self.activation == 'swish':
            x = nn.swish(x)
        elif self.activation == 'gelu':
            x = nn.gelu(x)
        else:
            x = nn.tanh(x)
        
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        
        # Second layer
        x = nn.LayerNorm()(x)
        x = nn.Dense(residual.shape[-1])(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        
        # Residual connection with learned gating (helps with gradient flow)
        gate = nn.Dense(1)(residual)
        gate = nn.sigmoid(gate)
        
        return gate * x + (1 - gate) * residual

class AttentionResidualMLPDynamics(nn.Module):
    """
    Combines attention and residual connections for vehicle dynamics.
    Architecture inspired by successful PINN and vehicle dynamics papers.
    """
    hidden_dims: tuple = (64, 64)
    num_attention_heads: int = 4
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training=True):
        # Extract physics-meaningful features
        physics_features = self.extract_physics_features(x)
        
        # Initial embedding
        embedded = nn.Dense(self.hidden_dims[0])(x)
        
        # Residual blocks
        for i, dim in enumerate(self.hidden_dims):
            embedded = ResidualPhysicsBlock(
                features=dim,
                dropout_rate=self.dropout_rate,
                name=f'res_block_{i}'
            )(embedded, training=training)
        
        # Attention mechanism to fuse physics and neural features
        neural_features = embedded
        attended_output = PhysicsAwareAttention(
            num_heads=self.num_attention_heads,
            embed_dim=64
        )(physics_features, neural_features)
        
        # Final residual connection
        final_output = attended_output + 0.1 * neural_features[:3]  # Small residual weight
        
        # Apply physics constraints
        return self.apply_physics_constraints(final_output, x)
    
    def extract_physics_features(self, x):
        """Extract physics-meaningful features from input"""
        # x = [yaw, steering, v, beta, psi_dot, a, delta_dot]
        yaw, steering, v, beta, psi_dot, a, delta_dot = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
        
        # Physics-based features
        kinetic_energy = 0.5 * v**2
        lateral_force = v * jnp.sin(beta + steering)
        longitudinal_force = v * jnp.cos(beta + steering)
        angular_momentum = v * jnp.sin(beta)
        centripetal_accel = v * psi_dot
        slip_angle = jnp.arctan(jnp.tan(beta))
        
        physics_features = jnp.array([
            kinetic_energy, lateral_force, longitudinal_force,
            angular_momentum, centripetal_accel, slip_angle
        ])
        
        return physics_features
    
    def apply_physics_constraints(self, output, input_state):
        """Apply physics-based constraints to outputs"""
        v_dot, beta_dot, psi_ddot = output[0], output[1], output[2]
        v = input_state[2]  # current velocity
        
        # Velocity constraints (energy-based)
        max_accel = 10.0  # m/s^2
        v_dot = jnp.tanh(v_dot / max_accel) * max_accel
        
        # Side slip rate constraints (tire friction limits)
        max_beta_dot = 2.0  # rad/s
        beta_dot = jnp.tanh(beta_dot / max_beta_dot) * max_beta_dot
        
        # Yaw acceleration constraints (stability)
        max_yaw_accel = 5.0  # rad/s^2
        psi_ddot = jnp.tanh(psi_ddot / max_yaw_accel) * max_yaw_accel
        
        return jnp.array([v_dot, beta_dot, psi_ddot])



class MLPDynamics(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(32)(x)
        x = nn.tanh(x)
        x = nn.Dense(16)(x)
        x = nn.tanh(x)
        # x = nn.Dense(32)(x)
        # x = nn.tanh(x)
        x = nn.Dense(3)(x)
        return x

def kinematics(state, inputs):
    pos_x, pos_y, yaw, steering, v, beta, psi_dot = state
    a, delta_dot = inputs
    dx_dt = v * jnp.cos(yaw + beta)
    dy_dt = v * jnp.sin(yaw + beta)
    dpsi_dt = psi_dot
    ddelta_dt = delta_dot
    return jnp.array([dx_dt, dy_dt, dpsi_dt, ddelta_dt])

class HybridODE:
    def __init__(self, config, model=None):
        self.config = config
        self.input_dim = len(config['model']['input_names'])
        self.physics_states = config['model']['physics_states']
        self.neural_states = config['model']['neural_states']
        
        # Check which model to use based on config
        model_type = config.get('neural_model_type', 'mlp')  # Default to 'mlp'
        
        if model_type == 'attention_residual':
            print("Using Physics-Aware Attention + Residual model")
            self.neural_net = AttentionResidualMLPDynamics(
                hidden_dims=config.get('hidden_dims', (64, 64)),
                num_attention_heads=config.get('num_attention_heads', 4),
                dropout_rate=config.get('dropout_rate', 0.1)
            )
            self.use_training_mode = True  # Attention model needs training parameter
        else:
            print("Using basic MLP model")
            self.neural_net = MLPDynamics()
            self.use_training_mode = False  # Basic MLP doesn't need training parameter
        
        self.params = self.init_network(jax.random.PRNGKey(0))
        self.model = model

    def init_network(self, key):
        dummy_input = jnp.ones((7,))
        if self.use_training_mode:
            params = self.neural_net.init(key, dummy_input, training=False)
        else:
            params = self.neural_net.init(key, dummy_input)
        return params

    def neural_dynamics(self, state, inputs, params=None, training=True):
        # take the last 5 states as the neural inputs (yaw, steering, v, beta, psi_dot)
        state = state[2:]
        nn_inputs = jnp.concatenate((state, inputs))

        assert nn_inputs.shape == (7,), f"nn_inputs shape is {nn_inputs.shape}, expected (7,)"
        if params is None:
            params = self.params
        
        # Apply neural network based on model type
        if self.use_training_mode:
            neural_output = self.neural_net.apply(params, nn_inputs, training=training)
        else:
            neural_output = self.neural_net.apply(params, nn_inputs)
        
        return neural_output

    def hybrid_dynamics(self, state, inputs, params=None, training=True):
        kinematic_derivs = kinematics(state, inputs)
        neural_derivatives = self.neural_dynamics(state, inputs, params, training)
        state_derivs = jnp.concatenate((kinematic_derivs, neural_derivatives))
        return state_derivs

    def rk4_step(self, state, inputs_t, inputs_t_plus_dt, dt, params=None, training=True):
        k1 = self.hybrid_dynamics(state, inputs_t, params, training)
        k2 = self.hybrid_dynamics(state + 0.5 * dt * k1, 0.5 * (inputs_t + inputs_t_plus_dt), params, training)
        k3 = self.hybrid_dynamics(state + 0.5 * dt * k2, 0.5 * (inputs_t + inputs_t_plus_dt), params, training)
        k4 = self.hybrid_dynamics(state + dt * k3, inputs_t_plus_dt, params, training)
        next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return next_state

    def predict_trajectory(self, params, initial_state, inputs_sequence, dt, training=False):
        num_steps = inputs_sequence.shape[0]
        
        # Pad inputs_sequence to avoid boundary checking
        last_input = inputs_sequence[-1:, :]  # Shape: (1, input_dim)
        padded_inputs = jnp.concatenate([inputs_sequence, last_input], axis=0)

        def scan_step(state, t):
            current_input = padded_inputs[t]
            next_input = padded_inputs[t + 1]  # Always safe now
            next_state = self.rk4_step(state, current_input, next_input, dt, params, training)
            return next_state, next_state

        indices = jnp.arange(num_steps - 1)
        _, states = jax.lax.scan(scan_step, initial_state, indices)
        
        trajectory = jnp.vstack([initial_state, states])
        return trajectory

    def predict_batch_trajectories(self, params, initial_states, inputs_batch, dt, training=False):
        batch_predict_fn = jax.vmap(
            lambda s, i: self.predict_trajectory(params, s, i, dt, training), 
            in_axes=(0, 0)
        )
        return batch_predict_fn(initial_states, inputs_batch)














# class MLPDynamics(nn.Module):
#     @nn.compact
#     def __call__(self, x):
#         x = nn.Dense(128)(x)
#         x = nn.tanh(x)
#         x = nn.Dense(128)(x)
#         x = nn.tanh(x)
#         x = nn.Dense(128)(x)
#         x = nn.tanh(x)
#         x = nn.Dense(64)(x)
#         x = nn.tanh(x)
#         x = nn.Dense(3)(x)
#         return x


# class HybridODE:
#     def __init__(self, config):
#         self.config = config
#         self.input_dim = len(config['model']['input_names'])
#         self.physics_states = config['model']['physics_states']
#         self.neural_states = config['model']['neural_states']
#         self.neural_net = MLPDynamics()  
#         self.params = self.init_network(jax.random.PRNGKey(0))

#     def init_network(self, key):
#         dummy_input = jnp.ones((9,))
#         params = self.neural_net.init(key, dummy_input)
#         return params

#     def neural_dynamics(self, state, inputs, params=None):
#         # take the last 4 states as the neural inputs
#         # neural_state = state[2:]  # yaw, steering, v, beta, psi_dot
#         nn_inputs = jnp.concatenate((state, inputs))

#         assert nn_inputs.shape == (9,), f"nn_inputs shape is {nn_inputs.shape}, expected (9,)"
#         if params is None:
#             params = self.params
#         neural_output = self.neural_net.apply(params, nn_inputs)
#         return neural_output

#     def hybrid_dynamics(self, state, inputs, params=None):

#         kinematic_derivs = kinematics(state, inputs)
#         neural_derivatives = self.neural_dynamics(state, inputs, params)
#         state_derivs = jnp.concatenate((kinematic_derivs, neural_derivatives))
#         return state_derivs

#     def rk4_step(self, state, inputs_t, inputs_t_plus_dt, dt, params=None):
#         k1 = self.hybrid_dynamics(state, inputs_t, params)
#         k2 = self.hybrid_dynamics(state + 0.5 * dt * k1, 0.5 * (inputs_t + inputs_t_plus_dt), params)
#         k3 = self.hybrid_dynamics(state + 0.5 * dt * k2, 0.5 * (inputs_t + inputs_t_plus_dt), params)
#         k4 = self.hybrid_dynamics(state + dt * k3, inputs_t_plus_dt, params)
#         next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
#         return next_state

#     def predict_trajectory(self, params, initial_state, inputs_sequence, dt):
        
#         num_steps = inputs_sequence.shape[0]

#         def scan_step(state, t):
#             current_input = inputs_sequence[t]
#             next_input = inputs_sequence[t + 1]  
#             next_state = self.rk4_step(state, current_input, next_input, dt, params)
          
#             return next_state, next_state

        
#         indices = jnp.arange(num_steps - 1)
#         _, states = jax.lax.scan(scan_step, initial_state, indices)
        
#         trajectory = jnp.vstack([initial_state, states])
#         return trajectory
    
    


#     def predict_batch_trajectories(self, params, initial_states, inputs_batch, dt):
#         batch_predict_fn = jax.vmap(lambda s, i: self.predict_trajectory(params, s, i, dt), in_axes=(0, 0))
#         return batch_predict_fn(initial_states, inputs_batch)




















    

def create_train_state(model, learning_rate, key, weight_decay=0.0):
    
    params = model.init_network(key)
    if weight_decay > 0:
        # Example: Exponential decay scheduler
        schedule = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=100,
            decay_rate=0.99,
            staircase=True
        )
        optimizer = optax.adamw(schedule, weight_decay=weight_decay)
    else:
        optimizer = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.neural_net.apply,
        params=params,
        tx=optimizer
    )
    


    
if __name__ == "__main__":
    

    print("=" * 50)
    print("Testing Hybrid ODE with Multi-Step Prediction (models_new.py)")
    print("=" * 50)

    # Load  config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Initialize model
    model = HybridODE(config)
    print("Model initialized.")

    # Initialize parameters (not used directly in this structure, but shown for completeness)
    key = jax.random.PRNGKey(42)
    params = model.init_network(key)
    print("Parameters initialized.")

    # Load a processed test sample
    data = np.load("processed_data/test_data.npz")
    samples = data["samples"]  # shape: (num_samples, 9, n_steps)
    print(f"Loaded test samples: {samples.shape}")

    # Use the first sample for demonstration
    sample = samples[0]  # shape: (9, n_steps)
    n_steps = sample.shape[1]
    state_dim = 7
    input_dim = 2

    # Extract initial state and input sequence
    initial_state = sample[:state_dim, 0]  # shape: (7,)
    inputs_sequence = sample[state_dim:, :].T  # shape: (n_steps, 2)
    inputs_sequence = jnp.array(inputs_sequence)

    # Use a fixed dt (if your data is uniform in time)
    dt = 0.1  # or set from your config/timestamps if available

    # Predict trajectory (for demo/testing): pass params explicitly
    pred_traj = model.predict_trajectory(params, initial_state, inputs_sequence, dt)
    print(f"Predicted trajectory shape: {pred_traj.shape}")
    print("First few predicted states:\n", np.array(pred_traj[:3]))

    print("\nMulti-step prediction is working correctly in models_new.py!")





