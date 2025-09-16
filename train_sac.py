#!/usr/bin/env python3

from pathlib import Path
import os
import argparse
import sys

# --- Optional wandb import (può essere disabilitato) ---
try:
    import wandb
except Exception:
    wandb = None

from citylearn.agents.sac import SAC as RLAgent
from citylearn.citylearn import CityLearnEnv


def resolve_dataset_arg(raw, data_dir: Path):
    """Nome → ./data/<nome>/schema.json → <data_dir>/<nome>/schema.json → altrimenti dataset ufficiale."""
    p = Path(raw)
    if p.exists():
        if p.is_dir():
            schema = p / "schema.json"
            if schema.exists():
                return str(schema.resolve()), "path"
        return str(p.resolve()), "path"

    candidates = [
        Path.cwd() / "data" / raw / "schema.json",
        data_dir / raw / "schema.json",
    ]
    env_dd = os.getenv("CITYLEARN_DATA_DIR")
    if env_dd:
        candidates.append(Path(env_dd) / raw / "schema.json")

    for c in candidates:
        if c.exists():
            return str(c.resolve()), "local"

    return raw, "official"


def parse_args():
    p = argparse.ArgumentParser(description="Train SAC on CityLearn (portable, no hardcoded paths).")
    p.add_argument(
        "--dataset-name",
        default=str(Path("data") / "TX_10_dynamics" / "schema.json"),
        type=str,
        help=("Nome dataset CityLearn oppure path/cartella o solo NOME locale (risolto a .../schema.json). "
              "Default: data/schema.json."),
    )
    p.add_argument("--central-agent", action="store_true", help="Usa un agente centrale.")
    p.add_argument("--episodes", default=12, type=int, help="Episodi di training.")
    p.add_argument("--lr", default=3e-4, type=float, help="Learning rate.")
    p.add_argument("--beta", default=0.0, type=float, help="Valore beta (tenuto nei config/W&B).")

    p.add_argument(
        "--output-root",
        type=Path,
        default=Path.cwd() / "outputs",
        help="Directory radice per risultati (creata se non esiste).",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.getenv("CITYLEARN_DATA_DIR", Path.home() / ".citylearn" / "data")),
        help="Directory locale per cache/dataset CityLearn.",
    )

    # Simulazione (opzionale)
    p.add_argument("--sim-start", type=int, default=3624, help="simulation_start_time_step (opzionale).")
    p.add_argument("--sim-end", type=int, default=4343, help="simulation_end_time_step (opzionale).")

    # W&B controls
    p.add_argument("--wandb", choices=["on", "off"], default=os.getenv("WANDB", "off"),
                   help="Abilita o disabilita Weights & Biases.")
    p.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "CityLearn-SAC"))
    p.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY", None))
    p.add_argument("--wandb-run-name", default=None, help="Nome run personalizzato.")
    p.add_argument("--wandb-tags", nargs="*", default=None, help="Lista di tag (opzionale).")

    # Dtype pesi (opzionale)
    p.add_argument("--weights-dtype", choices=["none", "fp16", "bf16", "fp32"], default="none",
                   help="Cast dei tensori floating prima del salvataggio per ridurre dimensione.")

    return p.parse_args()


def maybe_init_wandb(args, config):
    if args.wandb == "off" or wandb is None:
        return None
    try:
        run_name = args.wandb_run_name or f"sac_lr={args.lr}_beta={args.beta}"
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

    # Risolvi dataset
    dataset_arg, mode = resolve_dataset_arg(args.dataset_name, args.data_dir)
    print(f"[dataset] modalità={mode} → {dataset_arg}")

    output_root = args.output_root.resolve()

    # Path richiesto ESATTO: outputs/save_models/sac/beta/lr/sac.zip
    save_dir = output_root / args.dataset_name / "save_models" / "sac" / "beta" / "lr"
    save_dir.mkdir(parents=True, exist_ok=True)
    zip_file = save_dir / "sac.zip"

    # Config per logging/W&B
    config = {
        "dataset": dataset_arg,
        "dataset_mode": mode,
        "central_agent": args.central_agent,
        "episodes": args.episodes,
        "lr": args.lr,
        "beta": args.beta,
        "output_root": str(output_root),
        "zip_path": str(zip_file),
        "data_dir": str(args.data_dir),
        "sim_start": args.sim_start,
        "sim_end": args.sim_end,
    }

    run = maybe_init_wandb(args, config)

    # Hint per CityLearn cache
    os.environ.setdefault("CITYLEARN_DATA_DIR", str(args.data_dir))

    # --- Env & Agent ---
    env_kwargs = {"central_agent": args.central_agent}
    if args.sim_start is not None:
        env_kwargs["simulation_start_time_step"] = args.sim_start
    if args.sim_end is not None:
        env_kwargs["simulation_end_time_step"] = args.sim_end

    env = CityLearnEnv(dataset_arg, **env_kwargs)
    model = RLAgent(env, lr=args.lr)

    # --- Train ---
    model.learn(episodes=args.episodes)

    # --- Save: SOLO ZIP → outputs/save_models/sac/beta/lr/sac.zip ---
    dtype_arg = None if args.weights_dtype == "none" else args.weights_dtype
    model.save_models(zip_path=str(zip_file), dtype=dtype_arg)
    print("[OK] Zipped models at:", zip_file)

    if run is not None and wandb is not None:
        try:
            wandb.log({"episodes": args.episodes, "lr": args.lr, "beta": args.beta})
        except Exception as e:
            print(f"[W&B] log failed ({e})", file=sys.stderr)
        finally:
            run.finish()


if __name__ == "__main__":
    main()


