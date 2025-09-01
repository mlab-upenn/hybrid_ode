import json
import pickle
from pathlib import Path
from typing import Dict, Generator, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import HybridODE, Node, KinematicBicycle, DynamicBicycle  # your model definition

# ----------------------------------------------------------------------------- #
# Utility helpers                                                               #
# ----------------------------------------------------------------------------- #
def load_config(path: str = "config.yaml") -> Dict:
    with open(path, "r") as fp:
        return yaml.safe_load(fp)


def load_test_data(processed_dir: str = "processed_data"
                   ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    proc = Path(processed_dir)
    test_samples = jnp.array(np.load(proc / "val_data.npz")["samples"])
    with open(proc / "normalization_params.json") as fp:
        params = json.load(fp)
    norm = {k: jnp.array(v) for k, v in params.items()}
    return test_samples, norm


def denorm(x: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
    return x 


def batch_iter(data: jnp.ndarray, batch: int
               ) -> Generator[Tuple[jnp.ndarray, int], None, None]:
    n = data.shape[0]
    for i in range(0, n, batch):
        yield data[i:i + batch], i // batch

# ----------------------------------------------------------------------------- #
# Metric computation                                                            #
# ----------------------------------------------------------------------------- #
STATE_NAMES = ["delta_x", "delta_y", "yaw", "steering",
               "velocity", "side_slip", "yaw_rate"]


def yaw_err(pred, true):
    """Shortest-path angular error."""
    return ((pred - true + jnp.pi) % (2 * jnp.pi)) - jnp.pi


def trajectory_metrics(pred: jnp.ndarray, true: jnp.ndarray) -> Dict:
    mse_tot = float(jnp.mean((pred - true) ** 2))

    mse_state = {}
    for idx, name in enumerate(STATE_NAMES):
        if name == "yaw":
            mse_state[name] = float(jnp.mean(yaw_err(pred[..., idx],
                                                     true[..., idx]) ** 2))
        else:
            mse_state[name] = float(jnp.mean((pred[..., idx] -
                                              true[..., idx]) ** 2))

    pos_err = jnp.sqrt((pred[..., 0] - true[..., 0]) ** 2 +
                       (pred[..., 1] - true[..., 1]) ** 2)
    metrics = {
        "mse_total": mse_tot,
        "mse_per_state": mse_state,
        "position_rmse": float(jnp.sqrt(jnp.mean(pos_err ** 2))),
        "position_mae": float(jnp.mean(pos_err)),
        "final_position_rmse": float(jnp.sqrt(jnp.mean(pos_err[:, -1] ** 2))),
        "final_position_mae": float(jnp.mean(pos_err[:, -1])),
        "velocity_rmse": float(jnp.sqrt(
            jnp.mean((pred[..., 4] - true[..., 4]) ** 2))),
        "velocity_mae": float(jnp.mean(jnp.abs(pred[..., 4] - true[..., 4])))
    }
    return metrics

# ----------------------------------------------------------------------------- #
# Fast CSV writer (vectorised)                                                  #
# ----------------------------------------------------------------------------- #
def save_batch(batch_id: int,
               pred: jnp.ndarray,
               true: jnp.ndarray,
               metrics: Dict,
               t_vec: np.ndarray,
               outdir: Path,
               state_dim: int,
               state_names: list) -> None:
    """
    Writes a single CSV + JSON for this batch using bulk device→host copy
    and vectorised DataFrame construction.
    """
    outdir.mkdir(exist_ok=True)

    pred_np = np.asarray(pred)        # shape (B, T, state_dim)
    true_np = np.asarray(true)

    B, T, _ = pred_np.shape
    traj_ids = np.repeat(np.arange(B), T)
    step_ids = np.tile(np.arange(T), B)
    times    = np.tile(t_vec, B)

    flat_pred = pred_np.reshape(-1, state_dim)
    flat_true = true_np.reshape(-1, state_dim)
    flat_err  = flat_pred - flat_true

    if "yaw" in state_names:
        yaw_idx = state_names.index("yaw")
        flat_err[:, yaw_idx] = ((flat_err[:, yaw_idx] + np.pi) % (2*np.pi)) - np.pi

    df_dict = {
        "batch": batch_id,
        "traj":  traj_ids,
        "step":  step_ids,
        "time":  times,
    }
    for i, name in enumerate(state_names):
        df_dict[f"pred_{name}"] = flat_pred[:, i]
        df_dict[f"true_{name}"] = flat_true[:, i]
        df_dict[f"err_{name}"]  = flat_err[:, i]

    pd.DataFrame(df_dict).to_csv(outdir / f"batch_{batch_id}.csv",
                                 index=False)

    with open(outdir / f"batch_{batch_id}_metrics.json", "w") as fp:
        json.dump(metrics, fp, indent=2)

# ----------------------------------------------------------------------------- #
# Main routine                                                                  #
# ----------------------------------------------------------------------------- #
def main(cfg_path: str = "config.yaml") -> None:
    config = load_config(cfg_path)
    bs  = config["training"]["batch_size"]
    dt  = config["data"]["dt"]                     # adjust if needed

    indir = Path(config["data"]["input_dir"])
    base_name = indir.name
    processed_dir = Path("processed_data") / base_name
    outdir = Path("test_results") / base_name

    # --------------------------------------------------------------------- #
    print("Loading test data …")
    test_samples, norm = load_test_data(processed_dir=processed_dir)
    n_steps = test_samples.shape[2]
    t_vec   = np.arange(n_steps) * dt

    # --------------------------------------------------------------------- #
    print("Loading trained parameters …")
    params_path = Path("results") / base_name / "model_params.pkl"
   
    with open(params_path, "rb") as fp:
        params = pickle.load(fp)

    # --------------------------------------------------------------------- #
    if config['model_type'] == 'HybridODE':
        model = HybridODE(config)
        print("Using HybridODE model")
    elif config['model_type'] == 'Node':
        model = Node(config)
        print("Using Node model")
    elif config['model_type'] == 'KinematicBicycle':
        model = KinematicBicycle(config)
        print("Using KinematicBicycle model")
    elif config['model_type'] == 'DynamicBicycle':
        model = DynamicBicycle(config)
        print("Using DynamicBicycle model")
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")

    print(f"Running inference on {len(jax.devices())} device(s)…")
    overall = []

    last_batch_pred = None
    last_batch_true = None

    for batch, bid in tqdm(batch_iter(test_samples, bs),
                       total=(test_samples.shape[0] + bs - 1) // bs,
                       desc="Batches"):
        if config['model_type'] == 'KinematicBicycle':
            st_dim = 5
            state_names = STATE_NAMES[:5]
        elif config['model_type'] == 'DynamicBicycle':
            st_dim = 7
            state_names = STATE_NAMES
        else:
            st_dim = 7
            state_names = STATE_NAMES

        s0_norm  = batch[:, :st_dim, 0]
        u_norm   = batch[:, st_dim:, :].transpose(0, 2, 1)
        gt_norm  = batch[:, :st_dim, :].transpose(0, 2, 1)

        # Forward pass
        pred_norm = model.predict_batch_trajectories(params, s0_norm, u_norm, dt)

        # Denormalise only for models with 7D output
        if config['model_type'] in ['HybridODE', 'Node']:
            pred = denorm(pred_norm, norm["state_mean"], norm["state_std"])
            gt   = denorm(gt_norm,   norm["state_mean"], norm["state_std"])
        else:
            pred = pred_norm
            gt = gt_norm

        # Metrics + save
        m = trajectory_metrics(pred, gt)
        overall.append(m)
        save_batch(bid, pred, gt, m, t_vec, outdir, st_dim, state_names)

        print(f"[Batch {bid}]  total MSE={m['mse_total']:.6f} "
              f"Pos RMSE={m['position_rmse']:.4f}")

        # Plot and save for first 3 trajectories in this batch
        vis_dir = Path("visualizations")
        vis_dir.mkdir(exist_ok=True)
        num_traj = min(3, pred.shape[0])  # change 3 to any number you want

        for traj_idx in range(num_traj):
            # Plot all states + control inputs
            fig, axs = plt.subplots(9, 1, figsize=(10, 22), sharex=True)  # 7 states + 2 controls
            for i, name in enumerate(STATE_NAMES):
                axs[i].plot(t_vec, gt[traj_idx, :, i], label=f"True {name}")
                axs[i].plot(t_vec, pred[traj_idx, :, i], label=f"Pred {name}", linestyle='--')
                axs[i].set_ylabel(name)
                axs[i].legend()
            # Acceleration control input (assumed index 0)
            axs[7].plot(t_vec, u_norm[traj_idx, :, 0], color='tab:orange', label="Acceleration (u0)")
            axs[7].set_ylabel("Acceleration")
            axs[7].legend()
            # Steering rate control input (assumed index 1)
            axs[8].plot(t_vec, u_norm[traj_idx, :, 1], color='tab:green', label="Steering Rate (u1)")
            axs[8].set_ylabel("Steering Rate")
            axs[8].legend()

            axs[-1].set_xlabel("Time [s]")
            plt.suptitle(f"True vs Predicted States + Controls (Trajectory {traj_idx} in Batch {bid})")
            plt.tight_layout()
            plt.savefig(vis_dir / f"states_controls_batch{bid}_traj{traj_idx}.png")
            plt.close(fig)

            # 2D trajectory plot (delta_x vs delta_y)
            fig2 = plt.figure(figsize=(8, 6))
            plt.plot(gt[traj_idx, :, 0], gt[traj_idx, :, 1], label="True Trajectory")
            plt.plot(pred[traj_idx, :, 0], pred[traj_idx, :, 1], label="Predicted Trajectory", linestyle='--')
            plt.xlabel("delta_x")
            plt.ylabel("delta_y")
            plt.title(f"2D Trajectory (Trajectory {traj_idx} in Batch {bid})")
            plt.legend()
            plt.axis("equal")
            plt.tight_layout()
            plt.savefig(vis_dir / f"2d_batch{bid}_traj{traj_idx}.png")
            plt.close(fig2)

        # Save last batch for visualization
        last_batch_pred = pred
        last_batch_true = gt

    agg = {}
    for key in overall[0]:
        if key == "mse_per_state":
            agg[key] = {n: float(np.mean([o[key][n] for o in overall]))
                        for n in STATE_NAMES}
        else:
            agg[key] = float(np.mean([o[key] for o in overall]))

    with open(outdir / "overall_statistics.json", "w") as fp:
        json.dump(agg, fp, indent=2)

    print("\nFinished!  Results stored in →", outdir.resolve())

if __name__ == "__main__":
    main()
