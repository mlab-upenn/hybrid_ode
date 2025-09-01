import json
import pickle
from pathlib import Path
from typing import Dict, Generator, Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import HybridODE, Node

STATE_NAMES = ["delta_x", "delta_y", "yaw", "steering",
               "velocity", "side_slip", "yaw_rate"]

def load_config(path: str = "config.yaml") -> Dict:
    with open(path, "r") as fp:
        return yaml.safe_load(fp)

def load_test_data(processed_dir: str = "processed_data") -> jnp.ndarray:
    proc = Path(processed_dir)
    return jnp.array(np.load(proc / "test_data.npz")["samples"])

def batch_iter(data: jnp.ndarray, batch: int) -> Generator[Tuple[jnp.ndarray, int], None, None]:
    n = data.shape[0]
    for i in range(0, n, batch):
        yield data[i:i + batch], i // batch

def yaw_err(pred, true):
    """Shortest-path angular error."""
    return ((pred - true + jnp.pi) % (2 * jnp.pi)) - jnp.pi

def trajectory_metrics(pred: jnp.ndarray, true: jnp.ndarray) -> Dict:
    mse_tot = float(jnp.mean((pred - true) ** 2))
    mse_state = {}
    for idx, name in enumerate(STATE_NAMES):
        if name == "yaw":
            err = yaw_err(pred[..., idx], true[..., idx])
            mse_state[name] = float(jnp.mean(err ** 2))
        else:
            mse_state[name] = float(jnp.mean((pred[..., idx] - true[..., idx]) ** 2))
    pos_err = jnp.sqrt((pred[..., 0] - true[..., 0]) ** 2 +
                       (pred[..., 1] - true[..., 1]) ** 2)
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
    # Load configuration
    config = load_config(cfg_path)
    bs     = config["training"]["batch_size"]
    dt     = config["data"]["dt"]
    outdir = Path("test_results")
    outdir.mkdir(exist_ok=True)

    print("Loading test data …")
    test_samples = load_test_data()
    n_steps = test_samples.shape[2]
    t_vec   = np.arange(n_steps) * dt

    print("Loading trained parameters …")
    params_path = config.get("model_params_path", "results/model_params.pkl")
    with open(params_path, "rb") as fp:
        params = pickle.load(fp)

    model = Node(config)
    print("Using Node model")

    overall = []

    # Evaluate in batches
    for batch, bid in tqdm(batch_iter(test_samples, bs),
                           total=(test_samples.shape[0] + bs - 1) // bs,
                           desc="Batches"):
        st_dim = 7
        s0 = batch[:, :st_dim, 0]
        u  = batch[:, st_dim:, :].transpose(0, 2, 1)
        gt = batch[:, :st_dim, :].transpose(0, 2, 1)

        pred = model.predict_batch_trajectories(params, s0, u, dt)
        m = trajectory_metrics(pred, gt)
        overall.append(m)

        print(f"[Batch {bid}] total MSE={m['mse_total']:.6f} "
              f"Pos RMSE={m['position_rmse']:.4f}")

        # Visualization for up to 3 trajectories
        vis_dir = Path("visualizations"); vis_dir.mkdir(exist_ok=True)
        num_vis = min(3, pred.shape[0])
        for i in range(num_vis):
            idx = pred.shape[0] - num_vis + i

            # States + controls
            fig, axs = plt.subplots(9, 1, figsize=(10, 22), sharex=True)
            for si, name in enumerate(STATE_NAMES):
                axs[si].plot(t_vec, gt[idx, :, si], label=f"True {name}")
                axs[si].plot(t_vec, pred[idx, :, si], '--', label=f"Pred {name}")
                axs[si].set_ylabel(name); axs[si].legend()
            axs[7].plot(t_vec, u[idx, :, 0], color='orange', label="Acceleration")
            axs[7].set_ylabel("Acceleration"); axs[7].legend()
            axs[8].plot(t_vec, u[idx, :, 1], color='green',  label="Steer Rate")
            axs[8].set_ylabel("Steer Rate"); axs[8].legend()
            axs[-1].set_xlabel("Time [s]")
            plt.suptitle(f"Batch {bid}, Traj {idx}: States+Controls")
            plt.tight_layout()
            fig.savefig(vis_dir / f"states_ctrl_batch{bid}_traj{idx}.png")
            plt.close(fig)

            # 2D trajectory
            fig2 = plt.figure(figsize=(8, 6))
            plt.plot(gt[idx, :, 0], gt[idx, :, 1], label="True")
            plt.plot(pred[idx, :, 0], pred[idx, :, 1], '--', label="Pred")
            plt.axis('equal'); plt.xlabel("delta_x"); plt.ylabel("delta_y")
            plt.title(f"Batch {bid}, Traj {idx}: 2D")
            plt.legend(); plt.tight_layout()
            fig2.savefig(vis_dir / f"2d_batch{bid}_traj{idx}.png")
            plt.close(fig2)

    # Aggregate metrics
    agg = {}
    for key in overall[0]:
        if key == "mse_per_state":
            agg[key] = {n: float(np.mean([o[key][n] for o in overall]))
                        for n in STATE_NAMES}
        else:
            agg[key] = float(np.mean([o[key] for o in overall]))

    # Save JSON
    with open(outdir / "overall_statistics.json", "w") as fp:
        json.dump(agg, fp, indent=2)

    # Prepare CSV of final-step MSE per state and position
    final_mse = agg["mse_per_state"]
    # compute final position MSE from final_position_rmse
    final_mse["position_xy"] = agg["final_position_rmse"] ** 2

    # Build DataFrame
    df = pd.DataFrame({
        "state": STATE_NAMES + ["position_xy"],
        "mse":   [final_mse[n] for n in STATE_NAMES] + [final_mse["position_xy"]]
    })

    # Map units
    unit_map = {
        "delta_x": "m²", "delta_y": "m²", "position_xy": "m²",
        "yaw": "rad²", "steering": "rad²",
        "velocity": "(m/s)²", "side_slip": "rad²", "yaw_rate": "(rad/s)²"
    }
    df["units"] = df["state"].map(unit_map)

    # Save CSV
    csv_path = outdir / "state_mse.csv"
    df.to_csv(csv_path, index=False)

    print(f"\nSaved overall statistics → {outdir.resolve()}")
    print(f"Saved state MSE CSV        → {csv_path.resolve()}")

if __name__ == "__main__":
    main()
