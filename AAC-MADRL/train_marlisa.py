#!/usr/bin/env python3
from pathlib import Path
import os
import argparse
import sys

# --- Optional wandb import (can be disabled) ---
try:
    import wandb
except Exception:
    wandb = None

from citylearn.agents.marlisa import MARLISA as RLAgent
from citylearn.citylearn import CityLearnEnv


def parse_args():
    p = argparse.ArgumentParser(description="Train MARLISA on CityLearn in a portable way.")
    p.add_argument("--dataset-name", default="TX_10", type=str, help="CityLearn dataset name.")
    p.add_argument("--central-agent", action="store_true", help="Use a central agent.")
    p.add_argument("--episodes", default=12, type=int, help="Training episodes.")
    p.add_argument("--lr", default=3e-4, type=float, help="Learning rate.")
    p.add_argument("--beta", default=0.0, type=float, help="Custom beta tag for run naming/paths.")

    # Output root; everything is saved under here
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path.cwd() / "outputs" / "TX_10",
        help="Root directory for results/models (created if missing).",
    )

    # CityLearn data dir (optional)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.getenv("CITYLEARN_DATA_DIR", Path.cwd() / "citylearn" / "data")),
        help="Directory for CityLearn datasets (will be created if needed).",
    )

    # W&B controls
    p.add_argument("--wandb", choices=["on", "off"], default=os.getenv("WANDB", "on"),
                   help="Enable or disable Weights & Biases logging.")
    p.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "TX_100Building"))
    p.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY", None))
    p.add_argument("--wandb-run-name", default=None,
                   help="Custom run name. If omitted, a sensible default is used.")
    p.add_argument("--wandb-tags", nargs="*", default=None, help="Optional list of tags.")
    return p.parse_args()


def maybe_init_wandb(args, config):
    """Initialize W&B safely across users/machines."""
    if args.wandb == "off" or wandb is None:
        return None

    try:
        run_name = args.wandb_run_name or f"beta={args.beta}_marlisa_lr={args.lr}"
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,            # can be None
            name=run_name,
            config=config,
            reinit=True,
            anonymous="allow",                   # works even if not logged in
        )
        if args.wandb_tags:
            wandb.run.tags = list(set((wandb.run.tags or []) + args.wandb_tags))
        return run
    except Exception as e:
        print(f"[W&B] init failed ({e}). Continuing without W&B.", file=sys.stderr)
        return None


def main():
    args = parse_args()

    # Prepare directories (portable)
    output_root: Path = args.output_root
    models_dir = output_root / "save_models" / "marlisa" / f"beta={args.beta}" / f"lr={args.lr}"
    for d in [output_root, models_dir, args.data_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Minimal config for logging
    config = {
        "dataset_name": args.dataset_name,
        "central_agent": args.central_agent,
        "episodes": args.episodes,
        "lr": args.lr,
        "beta": args.beta,
        "output_root": str(output_root),
        "data_dir": str(args.data_dir),
    }

    # Init W&B (optional)
    run = maybe_init_wandb(args, config)

    # If CityLearn allows pointing to data dir via env var, set it (harmless if unused)
    os.environ.setdefault("CITYLEARN_DATA_DIR", str(args.data_dir))

    # --- Env & Agent ---
    env = CityLearnEnv(args.dataset_name, central_agent=args.central_agent)
    model = RLAgent(env, lr=args.lr)

    # --- Train ---
    model.learn(episodes=args.episodes)

    # --- Save models ---
    model.save_models(directory=str(models_dir))

    print("[OK] Training finished, models saved in:", models_dir)

    # W&B artifact/log (optional)
    if run is not None and wandb is not None:
        try:
            wandb.log({"episodes": args.episodes, "lr": args.lr})
        except Exception as e:
            print(f"[W&B] log failed ({e})", file=sys.stderr)
        finally:
            run.finish()


if __name__ == "__main__":
    main()
