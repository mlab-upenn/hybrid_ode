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

class MLPDynamics(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.tanh(x)
        x = nn.Dense(128)(x)
        x = nn.tanh(x)
        x = nn.Dense(128)(x)
        x = nn.tanh(x)
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
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
    def __init__(self, config):
        self.config = config
        self.input_dim = len(config['model']['input_names'])
        self.physics_states = config['model']['physics_states']
        self.neural_states = config['model']['neural_states']
        self.neural_net = MLPDynamics()  
        self.params = self.init_network(jax.random.PRNGKey(0))

    def init_network(self, key):
        dummy_input = jnp.ones((9,))
        params = self.neural_net.init(key, dummy_input)
        return params

    def neural_dynamics(self, state, inputs, params=None):
        # take the last 4 states as the neural inputs
        # neural_state = state[2:]  # yaw, steering, v, beta, psi_dot
        nn_inputs = jnp.concatenate((state, inputs))

        assert nn_inputs.shape == (9,), f"nn_inputs shape is {nn_inputs.shape}, expected (9,)"
        if params is None:
            params = self.params
        neural_output = self.neural_net.apply(params, nn_inputs)
        return neural_output

    def hybrid_dynamics(self, state, inputs, params=None):

        kinematic_derivs = kinematics(state, inputs)
        neural_derivatives = self.neural_dynamics(state, inputs, params)
        state_derivs = jnp.concatenate((kinematic_derivs, neural_derivatives))
        return state_derivs

    def rk4_step(self, state, inputs_t, inputs_t_plus_dt, dt, params=None):
        k1 = self.hybrid_dynamics(state, inputs_t, params)
        k2 = self.hybrid_dynamics(state + 0.5 * dt * k1, 0.5 * (inputs_t + inputs_t_plus_dt), params)
        k3 = self.hybrid_dynamics(state + 0.5 * dt * k2, 0.5 * (inputs_t + inputs_t_plus_dt), params)
        k4 = self.hybrid_dynamics(state + dt * k3, inputs_t_plus_dt, params)
        next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return next_state

    def predict_trajectory(self, params, initial_state, inputs_sequence, dt):
        
        num_steps = inputs_sequence.shape[0]

        def scan_step(state, t):
            current_input = inputs_sequence[t]
            next_input = inputs_sequence[t + 1]  
            next_state = self.rk4_step(state, current_input, next_input, dt, params)
          
            return next_state, next_state

        
        indices = jnp.arange(num_steps - 1)
        _, states = jax.lax.scan(scan_step, initial_state, indices)
        
        trajectory = jnp.vstack([initial_state, states])
        return trajectory
    
    


    def predict_batch_trajectories(self, params, initial_states, inputs_batch, dt):
        batch_predict_fn = jax.vmap(lambda s, i: self.predict_trajectory(params, s, i, dt), in_axes=(0, 0))
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





