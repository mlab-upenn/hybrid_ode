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

from models import HybridODE


def load_config(path: str = "config.yaml") -> Dict:
    with open(path, "r") as fp:
        return yaml.safe_load(fp)


def load_test_data(processed_dir: str = "processed_data") -> jnp.ndarray:
    proc = Path(processed_dir)
    test_samples = jnp.array(np.load(proc / "test_data.npz")["samples"])
    return test_samples


def batch_iter(data: jnp.ndarray, batch: int) -> Generator[Tuple[jnp.ndarray, int], None, None]:
    n = data.shape[0]
    for i in range(0, n, batch):
        yield data[i:i + batch], i // batch



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
            mse_state[name] = float(jnp.mean(yaw_err(pred[..., idx], true[..., idx]) ** 2))
        else:
            mse_state[name] = float(jnp.mean((pred[..., idx] - true[..., idx]) ** 2))
    pos_err = jnp.sqrt((pred[..., 0] - true[..., 0]) ** 2 + (pred[..., 1] - true[..., 1]) ** 2)
    metrics = {
        "mse_total": mse_tot,
        "mse_per_state": mse_state,
        "position_rmse": float(jnp.sqrt(jnp.mean(pos_err ** 2))),
        "position_mae": float(jnp.mean(pos_err)),
        "final_position_rmse": float(jnp.sqrt(jnp.mean(pos_err[:, -1] ** 2))),
        "final_position_mae": float(jnp.mean(pos_err[:, -1])),
        "velocity_rmse": float(jnp.sqrt(jnp.mean((pred[..., 4] - true[..., 4]) ** 2))),
        "velocity_mae": float(jnp.mean(jnp.abs(pred[..., 4] - true[..., 4])))
    }
    return metrics


def main(cfg_path: str = "config.yaml") -> None:
    config = load_config(cfg_path)
    bs  = config["training"]["batch_size"]
    dt  = config["data"]["dt"]

    indir = Path(config["data"]["input_dir"])
    base_name = indir.name
    processed_dir = Path("processed_data") / base_name
    outdir = Path("test_results") / base_name

    # --------------------------------------------------------------------- #
    print("Loading test data …")
    test_samples = load_test_data(processed_dir=processed_dir)
    n_steps = test_samples.shape[2]
    t_vec   = np.arange(n_steps) * dt

    # --------------------------------------------------------------------- #
    print("Loading trained parameters …")
    params_path = Path("results") / base_name / "model_params.pkl"
   
    with open(params_path, "rb") as fp:
        params = pickle.load(fp)

    model = HybridODE(config)
    print("Using HybridODE model")
    overall = []

    for batch, bid in tqdm(batch_iter(test_samples, bs),
                       total=(test_samples.shape[0] + bs - 1) // bs,
                       desc="Batches"):
        st_dim = 7
        state_names = STATE_NAMES
        s0 = batch[:, :st_dim, 0]
        u = batch[:, st_dim:, :].transpose(0, 2, 1)
        gt = batch[:, :st_dim, :].transpose(0, 2, 1)
        pred = model.predict_batch_trajectories(params, s0, u, dt)
        m = trajectory_metrics(pred, gt)
        overall.append(m)

        print(f"[Batch {bid}]  total MSE={m['mse_total']:.6f} Pos RMSE={m['position_rmse']:.4f}")

        # Plot and save for first 3 trajectories in this batch
        vis_dir = Path("visualizations")
        vis_dir.mkdir(exist_ok=True)
        num_traj = min(3, pred.shape[0])
        for i in range(num_traj):
            traj_idx = pred.shape[0] - num_traj + i
            fig, axs = plt.subplots(9, 1, figsize=(10, 22), sharex=True)
            for i, name in enumerate(STATE_NAMES):
                axs[i].plot(t_vec, gt[traj_idx, :, i], label=f"True {name}")
                axs[i].plot(t_vec, pred[traj_idx, :, i], label=f"Pred {name}", linestyle='--')
                axs[i].set_ylabel(name)
                axs[i].legend()
            axs[7].plot(t_vec, u[traj_idx, :, 0], color='tab:orange', label="Acceleration (u0)")
            axs[7].set_ylabel("Acceleration")
            axs[7].legend()
            axs[8].plot(t_vec, u[traj_idx, :, 1], color='tab:green', label="Steering Rate (u1)")
            axs[8].set_ylabel("Steering Rate")
            axs[8].legend()
            axs[-1].set_xlabel("Time [s]")
            plt.suptitle(f"True vs Predicted States + Controls (Trajectory {traj_idx} in Batch {bid})")
            plt.tight_layout()
            plt.savefig(vis_dir / f"states_controls_batch{bid}_traj{traj_idx}.png")
            plt.close(fig)
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

    agg = {}
    for key in overall[0]:
        if key == "mse_per_state":
            agg[key] = {n: float(np.mean([o[key][n] for o in overall]))
                        for n in STATE_NAMES}
        else:
            agg[key] = float(np.mean([o[key] for o in overall]))

    with open(outdir / "overall_statistics.json", "w") as fp:
        import json
        json.dump(agg, fp, indent=2)

    print("\nFinished!  Results stored in →", outdir.resolve())

if __name__ == "__main__":
    main()
