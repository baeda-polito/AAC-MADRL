#!/usr/bin/env python3
from pathlib import Path
import argparse
import importlib.util
from typing import Optional

import pandas as pd
import numpy as np

from citylearn.citylearn import CityLearnEnv
from citylearn.agents.aac_madrl import AAC_MADRL
from citylearn.agents.sac import SAC
from citylearn.agents.marlisa import MARLISA
from citylearn.agents.rbc import BasicBatteryRBC


# ----------------------------- helpers -----------------------------

def load_norm_params(model, directory: Path) -> None:
    """Carica mean/std dal file mean_std/config.py se presente."""
    cfg = directory / "mean_std" / "config.py"
    if not cfg.exists():
        return
    spec = importlib.util.spec_from_file_location("config", str(cfg))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model.norm_mean = [np.array(lst) for lst in mod.params["mean"]]
    model.norm_std = [np.array(lst) for lst in mod.params["std"]]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# --------------------------- main routine --------------------------

def run_model_and_save_obs(
    dataset_name: str,
    model_type: str,
    obs_dir: Path,
    weights_dir: Path,
    lr: float,
    beta: float,
    building: int = 0,
    per_building: bool = False,
    classes: Optional[dict] = None,
) -> None:

    # --- Init env + model ---
    if model_type == "AAC_MADRL":
        env = CityLearnEnv(dataset_name, central_agent=False, simulation_end_time_step=743)
        model = AAC_MADRL(env, classes=classes, attend_heads=1, lr=lr, sample=False)
    elif model_type == "MARLISA":
        env = CityLearnEnv(dataset_name, central_agent=False, simulation_end_time_step=743)
        model = MARLISA(env, lr=lr)
    elif model_type == "SAC":
        env = CityLearnEnv(dataset_name, central_agent=False, simulation_end_time_step=743)
        model = SAC(env, lr=lr)
    elif model_type == "SAC_CENTRALIZED":
        env = CityLearnEnv(dataset_name, central_agent=True, simulation_end_time_step=743)
        model = SAC(env, lr=lr)
    elif model_type == "RBC":
        env = CityLearnEnv(dataset_name, central_agent=False, simulation_end_time_step=743)
        model = BasicBatteryRBC(env)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    # --- Load NN weights if needed (non-RBC) ---
    if model_type != "RBC":
        expected_any = weights_dir / "policy_net_0.pt"
        if not expected_any.exists():
            raise FileNotFoundError(
                "Model weights not found.\n"
                f"  Expected here: {weights_dir}\n"
                "Suggerimenti:\n"
                f"  • Controlla che beta e lr combacino con l'allenamento.\n"
                f"  • Usa --weights-dataset-name per puntare al dataset dei pesi (es. TX_10).\n"
            )
        print(f"[INFO] Using weights from: {weights_dir}")
        model.load_models(directory=str(weights_dir))
        load_norm_params(model, weights_dir)

    # --- Deterministic rollout ---
    model.learn(deterministic=True)

    # --- Save observations (e azioni se disponibili) ---
    ensure_dir(obs_dir)

    if per_building:
        b = model.env.buildings[building]

        # split positivi/negativi
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

        # osservazioni complete come richiesto
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
            "energy from dhw device": b.energy_from_dhw_storage,  # lasciato come nel tuo snippet
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
        })
        df_obs.to_csv(obs_dir / f"obs_building_{building}.csv", index=False)

        # Azioni (solo agenti decentralizzati e se disponibili)
        if model_type != "SAC_CENTRALIZED" and hasattr(model, "actions") and hasattr(b, "active_actions"):
            try:
                actions_name = b.active_actions
                actions_value = model.actions[building]
                dictionary = {
                    actions_name[act]: [row[act] for row in actions_value[:-1]]
                    for act in range(len(actions_name))
                }
                pd.DataFrame(dictionary).to_csv(obs_dir / f"action_building_{building}.csv", index=False)
            except Exception:
                # best-effort: non bloccare se la struttura differisce
                pass

    else:
        # DISTRICT-LEVEL
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
            "energy from dhw device to dhw storaga": model.env.energy_from_dhw_device_to_dhw_storage,  # typo voluto per coerenza con il tuo header
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


def main():
    parser = argparse.ArgumentParser(
        description="Deploy CityLearn agents: salva osservazioni (e azioni, se disponibili). Niente KPI."
    )
    parser.add_argument("--dataset-name", required=True, help="Dataset su cui fai il deploy (es. TX_10_test).")
    parser.add_argument(
        "--weights-dataset-name",
        type=str,
        default=None,
        help="Dataset da cui caricare i pesi (default: uguale a --dataset-name, es. TX_10).",
    )
    parser.add_argument("--model-type", choices=["AAC_MADRL", "MARLISA", "SAC", "SAC_CENTRALIZED", "RBC"], required=True)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate usato in training (per trovare la cartella).")
    parser.add_argument("--beta", type=float, default=0.0, help="Beta usato in training (per trovare la cartella).")
    parser.add_argument("--n-buildings", type=int, default=1, help="Quanti edifici salvare (per-building).")
    parser.add_argument("--output-root", type=Path, default=Path.cwd() / "outputs", help="Root di lavoro.")
    args = parser.parse_args()

    # Dove SALVARE osservazioni/azioni del deploy corrente
    dataset_root = args.output_root / args.dataset_name
    if args.model_type != "RBC":
        obs_dir = dataset_root / "obs" / args.model_type.lower() / f"beta={args.beta}" / f"lr={args.lr}"
    else:
        obs_dir = dataset_root / "obs" / args.model_type.lower()

    # Dove CERCARE i pesi (dataset dei pesi può essere diverso dal dataset del deploy)
    weights_dataset = args.weights_dataset_name or args.dataset_name
    weights_root = args.output_root / weights_dataset
    weights_dir = weights_root / "save_models" / args.model_type.lower() / f"beta={args.beta}" / f"lr={args.lr}"

    print("[INFO] Deploy dataset:      ", args.dataset_name)
    print("[INFO] Weights dataset:     ", weights_dataset)
    print("[INFO] Output (obs/actions):", obs_dir)
    if args.model_type != "RBC":
        print("[INFO] Expected weights at: ", weights_dir)

    # classi (per AAC_MADRL; adatta se servono altre)
    classes = {"dhw_storage": 11, "electrical_storage": 11}

    # Distretto
    run_model_and_save_obs(
        dataset_name=args.dataset_name,
        model_type=args.model_type,
        obs_dir=obs_dir,
        weights_dir=weights_dir,
        lr=args.lr,
        beta=args.beta,
        building=0,
        per_building=False,
        classes=classes,
    )

    # Per-edificio
    for i in range(args.n_buildings):
        run_model_and_save_obs(
            dataset_name=args.dataset_name,
            model_type=args.model_type,
            obs_dir=obs_dir,
            weights_dir=weights_dir,
            lr=args.lr,
            beta=args.beta,
            building=i,
            per_building=True,
            classes=classes,
        )


if __name__ == "__main__":
    main()

