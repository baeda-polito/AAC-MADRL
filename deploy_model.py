#!/usr/bin/env python3
from pathlib import Path
import argparse
import os
import zipfile
import tempfile
import json
from typing import Optional
import re

import pandas as pd
import numpy as np
import torch

from citylearn.citylearn import CityLearnEnv
from citylearn.agents.aac_madrl import AAC_MADRL
from citylearn.agents.sac import SAC
from citylearn.agents.marlisa import MARLISA
from citylearn.agents.rbc import OptimizedRBC as RBC


# ----------------------------- helpers -----------------------------

def resolve_dataset_arg(raw: str, data_dir: Path) -> str:
    """Nome → ./data/<nome>/schema.json → <data_dir>/<nome>/schema.json → altrimenti lascia 'raw' (ufficiale)."""
    p = Path(raw)
    if p.exists():
        if p.is_dir():
            sch = p / "schema.json"
            return str(sch.resolve()) if sch.exists() else str(p.resolve())
        return str(p.resolve())

    candidates = [
        Path.cwd() / "data" / raw / "schema.json",
        data_dir / raw / "schema.json",
    ]
    env_dd = os.getenv("CITYLEARN_DATA_DIR")
    if env_dd:
        candidates.append(Path(env_dd) / raw / "schema.json")

    for c in candidates:
        if c.exists():
            return str(c.resolve())

    # dataset ufficiale CityLearn (nome)
    return raw


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _torch_cast_state_dict(sd, target_dtype):
    if target_dtype is None:
        return sd
    return {k: (v.to(target_dtype) if torch.is_floating_point(v) else v) for k, v in sd.items()}


def _load_norm_from_extracted(model, tmp_root: Path):
    cfg = tmp_root / "mean_std" / "config.json"
    if cfg.exists():
        params = json.loads(cfg.read_text())
        model.norm_mean = [np.asarray(x) for x in params.get("mean", [])]
        model.norm_std  = [np.asarray(x) for x in params.get("std", [])]


def _generic_load_from_extracted(model, tmp_root: Path, map_location=None, cast_to=None, strict=True):
    """Best-effort: carica i file .pt noti (SAC/MARLISA naming) se il modello non espone load_models(zip_path=...)."""
    _load_norm_from_extracted(model, tmp_root)

    def _indices(pattern):
        rx = re.compile(pattern)
        idx = []
        for p in tmp_root.rglob("*"):
            m = rx.fullmatch(p.name)
            if m:
                idx.append(int(m.group(1)))
        return sorted(set(idx))

    indices = _indices(r"policy_net_(\d+)\.pt") or _indices(r"soft_q_net1_(\d+)\.pt")

    if cast_to:
        cast_to = cast_to.lower()
    tgt_dtype = {"fp16": torch.float16, "float16": torch.float16,
                 "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
                 "fp32": torch.float32, "float32": torch.float32}.get(cast_to)

    def _load_into(module, path: Path):
        sd = torch.load(path, map_location=map_location)
        if tgt_dtype is None:
            try:
                target_dtype = next(module.parameters()).dtype
                sd = _torch_cast_state_dict(sd, target_dtype)
            except StopIteration:
                pass
        else:
            sd = _torch_cast_state_dict(sd, tgt_dtype)
        module.load_state_dict(sd, strict=strict)

    soft_q_net1 = getattr(model, "soft_q_net1", None)
    soft_q_net2 = getattr(model, "soft_q_net2", None)
    target_soft_q_net1 = getattr(model, "target_soft_q_net1", None)
    target_soft_q_net2 = getattr(model, "target_soft_q_net2", None)
    policy_net = getattr(model, "policy_net", None)

    for i in indices:
        p = tmp_root / f"soft_q_net1_{i}.pt"
        if soft_q_net1 is not None and len(soft_q_net1) > i and p.exists():
            _load_into(soft_q_net1[i], p)
        p = tmp_root / f"soft_q_net2_{i}.pt"
        if soft_q_net2 is not None and len(soft_q_net2) > i and p.exists():
            _load_into(soft_q_net2[i], p)
        p = tmp_root / f"target_soft_q_net1_{i}.pt"
        if target_soft_q_net1 is not None and len(target_soft_q_net1) > i and p.exists():
            _load_into(target_soft_q_net1[i], p)
        p = tmp_root / f"target_soft_q_net2_{i}.pt"
        if target_soft_q_net2 is not None and len(target_soft_q_net2) > i and p.exists():
            _load_into(target_soft_q_net2[i], p)
        p = tmp_root / f"policy_net_{i}.pt"
        if policy_net is not None and len(policy_net) > i and p.exists():
            _load_into(policy_net[i], p)

    if hasattr(model, "critic") and (tmp_root / "critic.pt").exists():
        _load_into(model.critic, tmp_root / "critic.pt")
    if hasattr(model, "target_critic") and (tmp_root / "target_critic.pt").exists():
        _load_into(model.target_critic, tmp_root / "target_critic.pt")


def load_from_zip(model, zip_path: Path, map_location: Optional[str] = None, cast_to: Optional[str] = None, strict: bool = True):
    """Carica pesi e normalizzazioni da uno zip. Prova model.load_models(zip_path=...), poi fallback generico."""
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip non trovato: {zip_path}")

    try:
        return model.load_models(zip_path=str(zip_path), map_location=map_location, cast_to=cast_to, strict=strict)
    except TypeError:
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(td)
            try:
                return model.load_models(directory=td)
            except Exception:
                _generic_load_from_extracted(model, Path(td), map_location=map_location, cast_to=cast_to, strict=strict)


def derive_weights_zip(weights_root: Path, model_type: str, beta: float, lr: float) -> Path:
    m = model_type.lower()
    zip_name = {"sac": "sac.zip",
                "sac_centralized": "sac.zip",
                "marlisa": "marlisa.zip",
                "aac_madrl": "aac_madrl.zip"}.get(m, f"{m}.zip")

    p_new = weights_root / "save_models" / m / "beta" / str(beta) / "lr" / str(lr) / zip_name
    if p_new.exists():
        return p_new

    p_old = weights_root / "save_models" / m / f"beta={beta}" / f"lr={lr}" / zip_name
    return p_old


# --------------------------- main routine --------------------------

def run_model_and_save_obs(
    dataset_name: str,
    model_type: str,
    obs_dir: Path,
    weights_zip: Optional[Path],
    lr: float,
    beta: float,
    building: int = 0,
    per_building: bool = False,
    classes: Optional[dict] = None,
    sim_start: Optional[int] = None,
    sim_end: Optional[int] = None,
) -> None:

    # --- Env & Model ---
    central = (model_type == "SAC_CENTRALIZED")
    env_kwargs = {"central_agent": central}
    if sim_start is not None:
        env_kwargs["simulation_start_time_step"] = sim_start
    if sim_end is not None:
        env_kwargs["simulation_end_time_step"] = sim_end
    env = CityLearnEnv(dataset_name, **env_kwargs)

    if model_type == "AAC_MADRL":
        model = AAC_MADRL(env, classes=classes, attend_heads=1, lr=lr, sample=False)
    elif model_type in ("SAC", "SAC_CENTRALIZED"):
        model = SAC(env, lr=lr)
    elif model_type == "MARLISA":
        model = MARLISA(env, lr=lr)
    elif model_type == "RBC":
        model = RBC(env)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    # --- Load weights (non-RBC) ---
    if model_type != "RBC":
        if weights_zip is None or not weights_zip.exists():
            raise FileNotFoundError(
                "Model weights ZIP not found.\n"
                f"  Expected here: {weights_zip}\n"
            )
        print(f"[INFO] Using weights from: {weights_zip}")
        load_from_zip(model, weights_zip, map_location="cpu", cast_to=None, strict=True)

    # --- Deterministic rollout ---
    model.learn(deterministic=True)

    # --- Save observations ---
    ensure_dir(obs_dir)

    if per_building:
        b = model.env.buildings[building]

        pos_el = np.where(b.electrical_storage_electricity_consumption > 0,
                          b.electrical_storage_electricity_consumption, 0)
        neg_el = np.where(b.electrical_storage_electricity_consumption < 0,
                          b.electrical_storage_electricity_consumption, 0)

        pos_dhw = np.where(b.dhw_storage_electricity_consumption > 0,
                           b.dhw_storage_electricity_consumption, 0)
        neg_dhw = np.where(b.dhw_storage_electricity_consumption < 0,
                           b.dhw_storage_electricity_consumption, 0)

        nec = np.array(b.net_electricity_consumption)
        pos_nec = np.where(nec > 0, nec, 0)
        neg_nec = np.where(nec < 0, nec, 0)

        df_obs = pd.DataFrame({
            "cooling demand": b.cooling_demand,
            "cooling electricity consumption": b.cooling_electricity_consumption,
            "heating electricity consumption": b.heating_electricity_consumption,
            "dhw demand": b.dhw_demand,
            "dhw electricity consumption": b.dhw_electricity_consumption,
            "dhw storage electricity consumption": b.dhw_storage_electricity_consumption,
            "positive dhw storage electricity consumption": pos_dhw,
            "negative dhw storage electricity consumption": neg_dhw,
            "dhw soc storage": b.dhw_storage.soc if hasattr(b, "dhw_storage") else None,
            "dhw capacity": b.dhw_storage.capacity if hasattr(b, "dhw_storage") else None,
            "dhw stored energy (?)": (b.dhw_storage.soc * b.dhw_storage.capacity) if hasattr(b, "dhw_storage") else None,
            "electrical storage electricity consumption": b.electrical_storage_electricity_consumption,
            "positive electrical storage electricity consumption": pos_el,
            "negative electrical storage electricity consumption": neg_el,
            "electrical storage soc": b.electrical_storage.soc if hasattr(b, "electrical_storage") else None,
            "electrical capacity history": b.electrical_storage.capacity_history if hasattr(b, "electrical_storage") else None,
            "stored electrical energy": (b.electrical_storage.soc * b.electrical_storage.capacity_history) if hasattr(b, "electrical_storage") else None,
            "energy from cooling device": b.energy_from_cooling_device,
            "energy from dhw device": b.energy_from_dhw_storage,
            "energy from dhw device to dhw storage": b.energy_from_dhw_device_to_dhw_storage,
            "energy from dhw storage": b.energy_from_dhw_storage,
            "energy from electrical storage": b.energy_from_electrical_storage,
            "energy from heating device": b.energy_from_heating_device,
            "energy to electrical storage": b.energy_to_electrical_storage,
            "energy to non shiftable load": b.energy_to_non_shiftable_load,
            "net electricity consumption": b.net_electricity_consumption,
            "positive net electricity consumption": pos_nec,
            "negative net electricity consumption": neg_nec,
            "net electricity consumption without storage": b.net_electricity_consumption_without_storage,
            "net electricity consumption without storage and pv": b.net_electricity_consumption_without_storage_and_pv,
            "non shiftable load": b.non_shiftable_load,
            "solar generation": b.solar_generation,
            "indoor_temperature": b.indoor_dry_bulb_temperature,
            "heating_sp": b.indoor_dry_bulb_temperature_heating_set_point,
            "cooling_sp": b.indoor_dry_bulb_temperature_cooling_set_point,
            "comfort_band": b.comfort_band
        })
        df_obs.to_csv(obs_dir / f"obs_building_{building}.csv", index=False)

        if model_type != "SAC_CENTRALIZED" and hasattr(model, "actions") and hasattr(b, "active_actions"):
            try:
                actions_name = b.active_actions
                actions_value = model.actions[building]
                dictionary = {actions_name[act]: [row[act] for row in actions_value[:-1]]
                              for act in range(len(actions_name))}
                pd.DataFrame(dictionary).to_csv(obs_dir / f"action_building_{building}.csv", index=False)
            except Exception:
                pass

    else:
        pos_el = np.where(model.env.electrical_storage_electricity_consumption > 0,
                          model.env.electrical_storage_electricity_consumption, 0)
        neg_el = np.where(model.env.electrical_storage_electricity_consumption < 0,
                          model.env.electrical_storage_electricity_consumption, 0)

        pos_dhw = np.where(model.env.dhw_storage_electricity_consumption > 0,
                           model.env.dhw_storage_electricity_consumption, 0)
        neg_dhw = np.where(model.env.dhw_storage_electricity_consumption < 0,
                           model.env.dhw_storage_electricity_consumption, 0)

        nec = np.array(model.env.net_electricity_consumption)
        pos_nec = np.where(nec > 0, nec, 0)
        neg_nec = np.where(nec < 0, nec, 0)

        df_obs = pd.DataFrame({
            "cooling demand": model.env.cooling_demand,
            "cooling electricity consumption": model.env.cooling_electricity_consumption,
            "heating electricity consumption": model.env.heating_electricity_consumption,
            "dhw demand": model.env.dhw_demand,
            "dhw electricity consumption": model.env.dhw_electricity_consumption,
            "dhw storage electricity consumption": model.env.dhw_storage_electricity_consumption,
            "positive dhw storage electricity consumption": pos_dhw,
            "negative dhw storage electricity consumption": neg_dhw,
            "electrical storage electricity consumption": model.env.electrical_storage_electricity_consumption,
            "positive electrical storage electricity consumption": pos_el,
            "negative electrical storage electricity consumption": neg_el,
            "energy from cooling device": model.env.energy_from_cooling_device,
            "energy from dhw device": model.env.energy_from_dhw_storage,
            "energy from dhw device to dhw storage": model.env.energy_from_dhw_device_to_dhw_storage,
            "energy from dhw storage": model.env.energy_from_dhw_storage,
            "energy from electrical storage": model.env.energy_from_electrical_storage,
            "energy from heating device": model.env.energy_from_heating_device,
            "energy to electrical storage": model.env.energy_to_electrical_storage,
            "energy to non shiftable load": model.env.energy_to_non_shiftable_load,
            "net electricity consumption": model.env.net_electricity_consumption,
            "positive net electricity consumption": pos_nec,
            "negative net electricity consumption": neg_nec,
            "net electricity consumption without storage": model.env.net_electricity_consumption_without_storage,
            "net electricity consumption without storage and pv": model.env.net_electricity_consumption_without_storage_and_pv,
            "net electricity consumption without storage and partial load": model.env.net_electricity_consumption_without_storage_and_partial_load,
            "net electricity consumption without storage and partial load and pv": model.env.net_electricity_consumption_without_storage_and_partial_load_and_pv,
            "non shiftable load": model.env.non_shiftable_load,
            "solar generation": model.env.solar_generation,
        })
        df_obs.to_csv(obs_dir / "district_obs.csv", index=False)


def parse_args():
    p = argparse.ArgumentParser(
        description="Deploy CityLearn agents: salva osservazioni (e azioni, se disponibili)."
    )
    p.add_argument(
        "--dataset-name",
        default=str(Path("data") / "TX_10_dynamics" / "schema.json"),
        type=str,
        help=("Dataset su cui fare il deploy: nome CityLearn, cartella, o path a schema.json. "
              "Default: data/TX_10_dynamics/schema.json."),
    )
    p.add_argument(
        "--weights-dataset-name",
        type=str,
        default=None,
        help="Dataset da cui caricare i pesi (default: uguale a --dataset-name).",
    )
    p.add_argument("--model-type", choices=["AAC_MADRL", "MARLISA", "SAC", "SAC_CENTRALIZED", "RBC"], required=True)
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate usato in training.")
    p.add_argument("--beta", type=float, default=0.0, help="Beta usato in training.")
    p.add_argument("--n-buildings", type=int, default=1, help="Quanti edifici salvare (per-building).")
    p.add_argument("--output-root", type=Path, default=Path.cwd() / "outputs", help="Root di lavoro.")
    p.add_argument("--data-dir", type=Path, default=Path.home() / ".citylearn" / "data", help="Dir dati locale.")
    # NEW: controlli di simulazione
    p.add_argument("--sim-start", type=int, default=None, help="simulation_start_time_step (opzionale).")
    p.add_argument("--sim-end", type=int, default=None, help="simulation_end_time_step (opzionale).")
    return p.parse_args()


def main():
    args = parse_args()

    dataset_arg = resolve_dataset_arg(args.dataset_name, args.data_dir)

    dataset_root = args.output_root / (Path(dataset_arg).parent.name if Path(dataset_arg).exists() else args.dataset_name)
    if args.model_type != "RBC":
        obs_dir = dataset_root / "obs" / args.model_type.lower() / "beta" / str(args.beta) / "lr" / str(args.lr)
    else:
        obs_dir = dataset_root / "obs" / args.model_type.lower()

    weights_dataset = args.weights_dataset_name or args.dataset_name
    weights_dataset_arg = resolve_dataset_arg(weights_dataset, args.data_dir)
    weights_key = Path(weights_dataset_arg).parent.name if Path(weights_dataset_arg).exists() else weights_dataset
    weights_root = args.output_root / weights_key
    weights_zip = derive_weights_zip(weights_root, args.model_type, args.beta, args.lr)

    print("[INFO] Deploy dataset:      ", args.dataset_name, "→", dataset_arg)
    print("[INFO] Weights dataset:     ", weights_dataset)
    print("[INFO] Sim range:           ", f"start={args.sim_start}, end={args.sim_end}")
    print("[INFO] Output (obs/actions):", obs_dir)
    if args.model_type != "RBC":
        print("[INFO] Expected weights at: ", weights_zip)

    classes = {"dhw_storage": 21, "electrical_storage": 21, "cooling_or_heating_device": 21}

    # DISTRICT
    run_model_and_save_obs(
        dataset_name=dataset_arg,
        model_type=args.model_type,
        obs_dir=obs_dir,
        weights_zip=weights_zip if args.model_type != "RBC" else None,
        lr=args.lr,
        beta=args.beta,
        building=0,
        per_building=False,
        classes=classes,
        sim_start=args.sim_start,
        sim_end=args.sim_end,
    )

    # PER-BUILDING
    for i in range(args.n_buildings):
        run_model_and_save_obs(
            dataset_name=dataset_arg,
            model_type=args.model_type,
            obs_dir=obs_dir,
            weights_zip=weights_zip if args.model_type != "RBC" else None,
            lr=args.lr,
            beta=args.beta,
            building=i,
            per_building=True,
            classes=classes,
            sim_start=args.sim_start,
            sim_end=args.sim_end,
        )


if __name__ == "__main__":
    main()



