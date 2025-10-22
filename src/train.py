import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
import wandb

from src.model import MultiLayerPerceptron
from src.preprocess import get_dataloaders

# -----------------------------------------------------------------------------
# Global constant – absolute path to the configuration directory
# -----------------------------------------------------------------------------
CONFIG_PATH = str((Path(__file__).resolve().parent.parent / "config").absolute())

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

# -----------------------------------------------------------------------------
# Training of one hyper-parameter setting (inner objective)
# -----------------------------------------------------------------------------

def train_single_model(*,
                       evaluation_index: int,
                       lr: float,
                       hidden_units: int,
                       cfg: DictConfig,
                       loaders: Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader],
                       device: torch.device) -> Tuple[Dict, float]:
    """Train a 2-layer MLP and return metrics & wall-clock time."""

    start_time = time.perf_counter()

    train_loader, val_loader = loaders
    model = MultiLayerPerceptron(input_dim=cfg.model.input_dim,
                                 hidden_units=hidden_units,
                                 output_dim=cfg.model.output_dim,
                                 activation=cfg.model.activation).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.training.weight_decay)

    best_val_acc: float = 0.0
    cm_best: np.ndarray | None = None

    global_step = 0
    for epoch in range(cfg.training.epochs):
        # -------------------------- training loop --------------------------
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if cfg.trial_mode and batch_idx > 1:
                break  # quick pass in trial_mode
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(1)
            running_correct += preds.eq(targets).sum().item()
            running_total += targets.size(0)

            if cfg.wandb.mode != "disabled":
                wandb.log({
                    "train/loss": loss.item(),
                    "epoch": epoch,
                    "batch": batch_idx,
                    "evaluation_index": evaluation_index,
                }, commit=False)
            global_step += 1

        epoch_train_loss = running_loss / running_total
        epoch_train_acc = running_correct / running_total

        # ---------------------------- validation ---------------------------
        model.eval()
        val_correct, val_total = 0, 0
        all_true: List[int] = []
        all_pred: List[int] = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                if cfg.trial_mode and batch_idx > 1:
                    break
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(1)
                val_correct += preds.eq(targets).sum().item()
                val_total += targets.size(0)
                all_true.extend(targets.cpu().tolist())
                all_pred.extend(preds.cpu().tolist())
        val_acc = val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            cm_best = confusion_matrix(all_true, all_pred, labels=list(range(cfg.model.output_dim)))

        if cfg.wandb.mode != "disabled":
            wandb.log({
                "train/epoch_loss": epoch_train_loss,
                "train/epoch_acc": epoch_train_acc,
                "val/acc": val_acc,
                "evaluation_index": evaluation_index,
                "epoch": epoch,
            }, commit=True)

    runtime = time.perf_counter() - start_time

    if cm_best is None:
        cm_best = confusion_matrix(all_true, all_pred, labels=list(range(cfg.model.output_dim)))

    metrics = {
        "best_val_accuracy": best_val_acc,
        "confusion_matrix": cm_best.tolist(),
    }
    return metrics, runtime

# -----------------------------------------------------------------------------
# Math helpers
# -----------------------------------------------------------------------------

def expected_improvement(mu: np.ndarray, sigma: np.ndarray, y_best: float) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-9)
    z = (mu - y_best) / sigma
    ei = (mu - y_best) * norm.cdf(z) + sigma * norm.pdf(z)
    return ei


def auc(xs: np.ndarray, ys: np.ndarray) -> float:
    """Trapezoidal AUC."""
    return np.trapz(ys, xs)


def first_time(ys: List[float], ts: List[float], thresh: float = 0.85) -> float:
    """First timestamp where y >= thresh, `inf` if never reached."""
    for y, t in zip(ys, ts):
        if y >= thresh:
            return t
    return float("inf")

# -----------------------------------------------------------------------------
# BOIL optimiser (baseline)
# -----------------------------------------------------------------------------

def run_boil(cfg: DictConfig, loaders, device):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    from sklearn.linear_model import LinearRegression

    bounds = np.array([
        [np.log10(cfg.optuna.search_space.learning_rate.low), np.log10(cfg.optuna.search_space.learning_rate.high)],
        [cfg.optuna.search_space.hidden_units.low, cfg.optuna.search_space.hidden_units.high],
    ])

    def sample(n: int) -> np.ndarray:
        lr = 10 ** np.random.uniform(bounds[0, 0], bounds[0, 1], n)
        hu = np.random.randint(bounds[1, 0], bounds[1, 1] + 1, n)
        return np.stack([lr, hu], 1)

    n_init = cfg.algorithm.n_init_points
    total_evals = 1 if cfg.trial_mode else cfg.algorithm.total_evaluations

    X, y_u, y_c = [], [], []
    cum_t, cum_ts = 0.0, []
    best_so_far = 0.0

    gp_u = GaussianProcessRegressor(kernel=Matern() + WhiteKernel(1e-5))
    cost_lin = LinearRegression()

    for ei in range(total_evals):
        if ei < n_init or len(X) < 2:
            x_next = sample(1)[0]
        else:
            gp_u.fit(np.array(X), np.array(y_u))
            cost_lin.fit(np.array(X), np.array(y_c))
            cand = sample(4000)
            mu_u, sigma_u = gp_u.predict(cand, return_std=True)
            mu_c = cost_lin.predict(cand)
            acq = np.log(expected_improvement(mu_u, sigma_u, max(y_u)) + 1e-12) - np.log(mu_c + 1e-12)
            x_next = cand[np.argmax(acq)]

        lr, hu = float(x_next[0]), int(round(x_next[1]))
        metrics, wall = train_single_model(evaluation_index=ei, lr=lr, hidden_units=hu,
                                           cfg=cfg, loaders=loaders, device=device)
        val_acc = metrics["best_val_accuracy"]

        cum_t += wall
        cum_ts.append(cum_t)
        X.append(x_next)
        y_u.append(val_acc)
        y_c.append(np.log(wall))
        best_so_far = max(best_so_far, val_acc)

        if cfg.wandb.mode != "disabled":
            wandb.log({
                "evaluation": ei,
                "lr": lr,
                "hidden_units": hu,
                "val_acc": val_acc,
                "best_so_far": best_so_far,
                "cumulative_time": cum_t,
                "wall_clock": wall,
            })

    if cfg.wandb.mode != "disabled":
        wandb.summary["final_best_val_acc"] = best_so_far
        wandb.summary["auc_accuracy"] = auc(np.arange(len(y_u)), np.asarray(y_u))
        wandb.summary["time_to_85"] = first_time(y_u, cum_ts, 0.85)
        wandb.summary["confusion_matrix"] = metrics["confusion_matrix"]

# -----------------------------------------------------------------------------
# CA-BOIL optimiser (proposed)
# -----------------------------------------------------------------------------

def run_ca_boil(cfg: DictConfig, loaders, device):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel

    bounds = np.array([
        [np.log10(cfg.optuna.search_space.learning_rate.low), np.log10(cfg.optuna.search_space.learning_rate.high)],
        [cfg.optuna.search_space.hidden_units.low, cfg.optuna.search_space.hidden_units.high],
    ])

    def sample(n: int) -> np.ndarray:
        lr = 10 ** np.random.uniform(bounds[0, 0], bounds[0, 1], n)
        hu = np.random.randint(bounds[1, 0], bounds[1, 1] + 1, n)
        return np.stack([lr, hu], 1)

    n_init = cfg.algorithm.n_init_points
    total_evals = 1 if cfg.trial_mode else cfg.algorithm.total_evaluations
    kappa = cfg.algorithm.acquisition.kappa

    X, y_u, y_c = [], [], []
    cum_t, cum_ts = 0.0, []
    best_so_far = 0.0

    gp_u = GaussianProcessRegressor(kernel=Matern() + WhiteKernel(1e-5))
    gp_c = GaussianProcessRegressor(kernel=Matern() + WhiteKernel(1e-5))

    for ei in range(total_evals):
        if ei < n_init or len(X) < 3:
            x_next = sample(1)[0]
        else:
            gp_u.fit(np.array(X), np.array(y_u))
            gp_c.fit(np.array(X), np.array(y_c))
            cand = sample(4000)
            mu_u, sigma_u = gp_u.predict(cand, return_std=True)
            mu_c, sigma_c = gp_c.predict(cand, return_std=True)
            ei_vals = expected_improvement(mu_u, sigma_u, max(y_u))
            acq = ei_vals / (mu_c + kappa * sigma_c + 1e-12)
            x_next = cand[np.argmax(acq)]

        lr, hu = float(x_next[0]), int(round(x_next[1]))
        metrics, wall = train_single_model(evaluation_index=ei, lr=lr, hidden_units=hu,
                                           cfg=cfg, loaders=loaders, device=device)
        val_acc = metrics["best_val_accuracy"]

        cum_t += wall
        cum_ts.append(cum_t)
        X.append(x_next)
        y_u.append(val_acc)
        y_c.append(np.log(wall))
        best_so_far = max(best_so_far, val_acc)

        if cfg.wandb.mode != "disabled":
            wandb.log({
                "evaluation": ei,
                "lr": lr,
                "hidden_units": hu,
                "val_acc": val_acc,
                "best_so_far": best_so_far,
                "cumulative_time": cum_t,
                "wall_clock": wall,
            })

    if cfg.wandb.mode != "disabled":
        wandb.summary["final_best_val_acc"] = best_so_far
        wandb.summary["auc_accuracy"] = auc(np.arange(len(y_u)), np.asarray(y_u))
        wandb.summary["time_to_85"] = first_time(y_u, cum_ts, 0.85)
        wandb.summary["confusion_matrix"] = metrics["confusion_matrix"]

# -----------------------------------------------------------------------------
# Main entry-point for a single experiment run
# -----------------------------------------------------------------------------

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig):
    # ------------------------- trial-mode tweaks -------------------------
    if cfg.trial_mode:
        cfg.wandb.mode = "disabled"
        cfg.training.epochs = 1
        cfg.algorithm.total_evaluations = 1
        cfg.optuna.n_trials = 0

    results_root = ensure_dir(Path(cfg.results_dir).expanduser())
    run_dir = ensure_dir(results_root / cfg.run.run_id)
    set_seed(42)

    # --------------------- WandB initialisation -------------------------
    if cfg.wandb.mode != "disabled":
        wandb.init(entity=cfg.wandb.entity,
                   project=cfg.wandb.project,
                   id=cfg.run.run_id,
                   resume="allow",
                   mode=cfg.wandb.mode,
                   dir=str(run_dir),
                   config=OmegaConf.to_container(cfg, resolve=True))
        print("WandB URL:", wandb.run.get_url(), flush=True)
    else:
        os.environ["WANDB_MODE"] = "disabled"

    # --------------------------- data ------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.training.device == "cuda" else "cpu")
    train_loader, val_loader, _ = get_dataloaders(cfg)

    # --------------------- launch optimiser ------------------------
    algo_name = cfg.algorithm.name.lower()
    if algo_name == "boil":
        run_boil(cfg, (train_loader, val_loader), device)
    elif algo_name in {"ca-boil", "caboil", "ca_boil"}:
        run_ca_boil(cfg, (train_loader, val_loader), device)
    else:
        raise ValueError(f"Unsupported algorithm {cfg.algorithm.name}")

    # ------------- persist WandB creds for evaluator --------------
    cred_path = results_root / "config.yaml"
    if not cred_path.exists():
        with open(cred_path, "w", encoding="utf-8") as fp:
            fp.write(OmegaConf.to_yaml(cfg.wandb))

    # --------------------------- finish ---------------------------
    if cfg.wandb.mode != "disabled":
        wandb.finish()


if __name__ == "__main__":
    main()