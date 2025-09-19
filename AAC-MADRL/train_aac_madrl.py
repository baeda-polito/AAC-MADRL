#!/usr/bin/env python3
from pathlib import Path
import os
import argparse
import sys
import time


# --- Optional wandb import (can be disabled) ---
try:
    import wandb
except Exception:
    wandb = None

from citylearn.agents.aac_madrl import AAC_MADRL as RLAgent
from citylearn.citylearn import CityLearnEnv


def parse_args():
    p = argparse.ArgumentParser(description="Train AAC_MADRL (MAAC) on CityLearn - portable, no KPI export.")
    p.add_argument("--dataset-name", default="TX_10", type=str, help="CityLearn dataset name.")
    p.add_argument("--central-agent", action="store_true", help="Use a central agent.")
    p.add_argument("--episodes", default=12, type=int, help="Training episodes.")
    p.add_argument("--lr", default=1e-3, type=float, help="Learning rate.")
    p.add_argument("--beta", default=0.0, type=float, help="Tag for run naming/paths.")
    p.add_argument("--attend-heads", default=1, type=int, help="Attention heads (if supported).")
    p.add_argument("--sample", action="store_true", help="Enable stochastic sampling in policy.")

    # REQUIRED classes (configurabili da CLI)
    p.add_argument("--dhw-storage", default=11, type=int, help="Number of DHW storage units/class size.")
    p.add_argument("--electrical-storage", default=11, type=int, help="Number of electrical storage units/class size.")

    # Output root
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path.cwd() / "outputs" / "TX_10",
        help="Root directory for model saves (created if missing).",
    )

    # CityLearn data dir (optional)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.getenv("CITYLEARN_DATA_DIR", Path.cwd() / "citylearn" / "data")),
        help="Directory for CityLearn datasets (created if needed).",
    )

    # W&B
    p.add_argument("--wandb", choices=["on", "off"], default=os.getenv("WANDB", "on"),
                   help="Enable or disable Weights & Biases logging.")
    p.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "TX_100Building"))
    p.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY", None))
    p.add_argument("--wandb-run-name", default=None, help="Custom run name.")
    p.add_argument("--wandb-tags", nargs="*", default=None, help="Optional list of tags.")
    return p.parse_args()


def maybe_disable_wandb_globally(args):
    """If wandb=off, make wandb a no-op to protect any internal wandb.log calls."""
    if args.wandb == "off":
        os.environ.setdefault("WANDB_MODE", "disabled")
        if wandb is not None:
            wandb.init = lambda *a, **k: None
            wandb.log = lambda *a, **k: None
            wandb.save = lambda *a, **k: None
            wandb.finish = lambda *a, **k: None
        return None
    return True


def maybe_init_wandb(args, config):
    if args.wandb == "off" or wandb is None:
        return None
    try:
        run_name = args.wandb_run_name or f"beta={args.beta}_maac_lr={args.lr}"
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=config,
            reinit=True,
            anonymous="allow",
        )
        if args.wandb_tags:
            wandb.run.tags = list(set((wandb.run.tags or []) + args.wandb_tags))
        return run
    except Exception as e:
        print(f"[W&B] init failed ({e}). Continuing without W&B.", file=sys.stderr)
        return None


def main():
    args = parse_args()
    maybe_disable_wandb_globally(args)

    # Directories
    output_root: Path = args.output_root
    models_dir = output_root / "save_models" / "aac_madrl" / f"beta={args.beta}" / f"lr={args.lr}"
    (models_dir / "mean_std").mkdir(parents=True, exist_ok=True)  # alcune impl. usano questa sottocartella
    args.data_dir.mkdir(parents=True, exist_ok=True)

    # Required classes (sempre presenti)
    classes = {
        "dhw_storage": args.dhw_storage,
        "electrical_storage": args.electrical_storage,
    }

    # Config
    config = {
        "agent": "AAC_MADRL",
        "dataset_name": args.dataset_name,
        "central_agent": args.central_agent,
        "episodes": args.episodes,
        "lr": args.lr,
        "beta": args.beta,
        "attend_heads": args.attend_heads,
        "sample": args.sample,
        "classes": classes,
        "output_root": str(output_root),
        "data_dir": str(args.data_dir),
    }
    run = maybe_init_wandb(args, config)

    # Point CityLearn to data dir
    os.environ.setdefault("CITYLEARN_DATA_DIR", str(args.data_dir))

    # Env & Agent
    env = CityLearnEnv(args.dataset_name, central_agent=args.central_agent)
    model = RLAgent(env, classes=classes, attend_heads=args.attend_heads, lr=args.lr, sample=args.sample)

    # Train
    start = time.time()
    model.learn(episodes=args.episodes)
    end = time.time()

    # Save
    model.save_models(directory=str(models_dir))
    print(f"[OK] Training finished in {end - start:.1f}s. Models saved to: {models_dir}")

    # W&B tidy up
    if run is not None and wandb is not None:
        try:
            wandb.log({"episodes": args.episodes, "lr": args.lr, "train_seconds": end - start})
        except Exception as e:
            print(f"[W&B] log failed ({e})", file=sys.stderr)
        finally:
            run.finish()


if __name__ == "__main__":
    main()

