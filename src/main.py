import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# -----------------------------------------------------------------------------
# Absolute path to configuration directory (one level up from this file)
# -----------------------------------------------------------------------------
from pathlib import Path as _P
CONFIG_PATH = str((_P(__file__).resolve().parent.parent / "config").absolute())


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig):
    results_dir = Path(cfg.results_dir).expanduser()
    results_dir.mkdir(parents=True, exist_ok=True)

    # persist WandB credentials (once per results_dir)
    wandb_cfg_path = results_dir / "config.yaml"
    if not wandb_cfg_path.exists():
        with open(wandb_cfg_path, "w", encoding="utf-8") as fp:
            fp.write(OmegaConf.to_yaml(cfg.wandb))

    # build command for the actual training script
    cmd = [sys.executable, "-u", "-m", "src.train", f"run={cfg.run.run_id}", f"results_dir={cfg.results_dir}"]
    if cfg.trial_mode:
        cmd.append("trial_mode=true")
        cmd.append("wandb.mode=disabled")

    print("[main] Executing:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd)
    proc.communicate()
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()