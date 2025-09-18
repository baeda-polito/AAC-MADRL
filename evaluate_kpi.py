from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Optional

# ---------- path helpers ----------

ALGO_DIR = {
    "AAC-MADRL": "aac_madrl",
    "SAC": "sac",
    "RBC": "rbc",
}

def find_obs_csv(outputs_root: Path, dataset_key: str, algorithm: str, beta: float, lr: float) -> Path:
    """Trova district_obs.csv nel layout nuovo, con fallback al layout vecchio (beta=.../lr=...)."""
    algo = algorithm.upper()
    algo_dir = ALGO_DIR.get(algo, algo.lower())
    base = outputs_root / dataset_key / "obs" / algo_dir
    if algo == "RBC":
        candidates = [base / "district_obs.csv"]
    else:
        candidates = [
            base / "beta" / str(beta) / "lr" / str(lr) / "district_obs.csv",   # nuovo
            base / f"beta={beta}" / f"lr={lr}" / "district_obs.csv",            # vecchio
        ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"district_obs.csv non trovato per {algorithm}. Cercati: {', '.join(map(str, candidates))}")

def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def load_or_init_kpi_csv(kpi_path: Path) -> pd.DataFrame:
    """Apre il KPI CSV se esiste, altrimenti crea un dataframe vuoto con indice 'cost_function'."""
    if kpi_path.exists():
        try:
            return pd.read_csv(kpi_path, index_col="cost_function")
        except Exception:
            # fallback: prova prima colonna come indice
            return pd.read_csv(kpi_path, index_col=0)

    idx = [
        "Positive Net El.onsumption",
        "Variance",
        "District Export",
        "Self-sufficiency",
        "1-Self Sufficiency",
        "1-Self Sufficiency2",
        "Self-Sufficiency2",
        "Self-consumption",
        "1-Self Consumption",
        "Daily Peak Average",
        "Monthly Average PAR",
        "Monthly Average PAR2",
        # la riga nuova può non essere presente: verrà aggiunta alla prima assegnazione
        # "Avg Comfort Violation (°C)",
    ]
    df = pd.DataFrame(index=pd.Index(idx, name="cost_function"))
    df["District"] = np.nan
    return df

# ---------- comfort helpers ----------

def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _mean_violation_for_building_csv(csv_path: Path) -> Optional[float]:
    """Media (in °C) della violazione della banda di comfort per un file obs_building_*.csv."""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    assert df['heating_sp'].equals(df['cooling_sp']), "Set point di riscaldamento e raffreddamento devono essere uguali"
    SP = pd.to_numeric(df['heating_sp'], errors="coerce")
    T  = pd.to_numeric(df["indoor_temperature"], errors="coerce")
    B  = pd.to_numeric(df['comfort_band'], errors="coerce")

    low  = SP - B
    high = SP + B

    below = (low - T).clip(lower=0)
    above = (T - high).clip(lower=0)
    viol = (below + above)
    number_of_violations = (viol > 0).sum()
    m = float(viol.mean(skipna=True)) if len(viol) else None
    return m

def mean_comfort_violation_for_folder(obs_folder: Path) -> Optional[float]:
    """Scansiona obs_building_*.csv nella cartella e ritorna la media delle medie per edificio (°C)."""
    vals = []
    for f in sorted(obs_folder.glob("obs_building_*.csv")):
        v = _mean_violation_for_building_csv(f)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return None
    return float(np.mean(vals))

# ---------- KPI core ----------

def process_kpi(outputs_root: Path, dataset_key: str, algorithm: str, kpi_dir: Path, lr: float, beta: float, cooling: bool):
    """Legge district_obs.csv dal layout nuovo e aggiorna/salva il file KPI dell'algoritmo (CSV)."""
    obs_csv = find_obs_csv(outputs_root, dataset_key, algorithm, beta, lr)
    df_obs = pd.read_csv(obs_csv)

    # KPI file path (versione CSV)
    algo = algorithm.upper()
    if algo == "RBC":
        kpi_file = kpi_dir / f"{algo.lower()}.csv"
    else:
        kpi_file = kpi_dir / f"{algo.lower()}_lr={lr}.csv"
    ensure_parent(kpi_file)
    df_kpi = load_or_init_kpi_csv(kpi_file)

    # --- metriche base ---
    total_consumption = float(np.sum(df_obs["positive net electricity consumption"]))
    df_kpi.loc["Positive Net El.onsumption", "District"] = total_consumption
    df_kpi.loc["Variance", "District"] = float(np.var(df_obs["net electricity consumption"]))
    df_kpi.loc["District Export", "District"] = float(-1 * np.sum(df_obs["negative net electricity consumption"]))

    # --- carichi e PV ---
    load = (
        df_obs["dhw electricity consumption"]
        + df_obs["non shiftable load"]
        + df_obs["positive electrical storage electricity consumption"]
    )
    load += df_obs["cooling electricity consumption"] if cooling else df_obs["heating electricity consumption"]

    # nelle obs nuove "solar generation" è positiva
    pv = df_obs["solar generation"].clip(lower=0)

    # Self-sufficiency / consumption
    load_sum = float(load.sum()) if load.sum() != 0 else 1.0
    pv_sum = float(pv.sum()) if pv.sum() != 0 else 1.0

    match = np.minimum(load.to_numpy(), pv.to_numpy())
    self_suff = float(np.sum(match) / load_sum)
    self_cons = float(np.sum(match) / pv_sum)

    df_kpi.loc["Self-sufficiency", "District"] = self_suff
    df_kpi.loc["1-Self Sufficiency", "District"] = 1 - self_suff

    # versioni elementwise (gestione load==0)
    load_safe = load.replace(0, np.nan)
    ratio = (np.minimum(load_safe, pv) / load_safe).fillna(0.0)
    df_kpi.loc["Self-Sufficiency2", "District"] = float(np.sum(ratio))
    df_kpi.loc["1-Self Sufficiency2", "District"] = float(1 - np.sum(ratio))

    df_kpi.loc["Self-consumption", "District"] = self_cons
    df_kpi.loc["1-Self Consumption", "District"] = 1 - self_cons

    # --- NUOVO KPI: media violazione comfort per edificio e media distrettuale ---
    obs_folder = obs_csv.parent
    mean_violation = mean_comfort_violation_for_folder(obs_folder)
    if mean_violation is not None and np.isfinite(mean_violation):
        df_kpi.loc["Avg Comfort Violation (°C)", "District"] = float(mean_violation)
    else:
        # se mancano i file o le colonne, lascia NaN
        df_kpi.loc["Avg Comfort Violation (°C)", "District"] = np.nan

    # salva KPI aggiornato in CSV
    df_kpi.to_csv(kpi_file, index=True, index_label="cost_function")
    return df_obs, df_kpi, kpi_file

# ---------- driver ----------

if __name__ == "__main__":
    # Config
    building_counts = [10]
    learning_rate = 0.001
    control_algorithms = ["RBC", "SAC", "AAC-MADRL"]
    beta = 0.5
    dataset = "TX"                         # per comporre dataset_key
    start_date = "2017-06-01 00:00:00"

    outputs_root = Path.cwd() / "outputs"

    for n in building_counts:
        # es. "TX_10_dynamics"
        dataset_key = f"{dataset}_{n}_dynamics"

        # KPI dir
        kpi_dir = outputs_root / dataset_key / "kpi" / "test" / f"beta={beta}"
        kpi_dir.mkdir(parents=True, exist_ok=True)

        # Processa ogni algoritmo
        df_rbc, df_rbc_kpi, _ = process_kpi(outputs_root, dataset_key, "RBC",        kpi_dir, learning_rate, beta, cooling=True)
        df_aac_madrl, df_aac_madrl_kpi, _ = process_kpi(outputs_root, dataset_key, "AAC-MADRL", kpi_dir, learning_rate, beta, cooling=True)
        df_sac,  df_sac_kpi,  _ = process_kpi(outputs_root, dataset_key, "SAC",      kpi_dir, learning_rate, beta, cooling=True)

        # allinea index a datetime (se ti serve in seguito)
        num_hours = len(df_rbc)
        date_range = pd.date_range(start=start_date, periods=num_hours, freq="h")
        for df in [df_rbc, df_aac_madrl, df_sac]:
            df.index = date_range

        # daily aggregates (non modificati)
        daily_dfs = {
            "RBC": df_rbc.resample("D").agg({
                "net electricity consumption": ["max", "mean"],
                "positive net electricity consumption": ["max", "mean"],
            }),
            "AAC-MADRL": df_aac_madrl.resample("D").agg({
                "net electricity consumption": ["max", "mean"],
                "positive net electricity consumption": ["max", "mean"],
            }),
            "SAC": df_sac.resample("D").agg({
                "net electricity consumption": ["max", "mean"],
                "positive net electricity consumption": ["max", "mean"],
            })
        }

        for algorithm in ["RBC", "AAC-MADRL", "SAC"]:
            daily_df = daily_dfs[algorithm]
            df_kpi = {
                "RBC": df_rbc_kpi,
                "AAC-MADRL": df_aac_madrl_kpi,
                "SAC": df_sac_kpi,
            }[algorithm]

            average_daily_peak = float(daily_df["net electricity consumption"]["max"].mean())
            daily_par  = (np.abs(daily_df["net electricity consumption"]["max"]) /
                          np.abs(daily_df["positive net electricity consumption"]["mean"]).replace(0, np.nan)).fillna(0).mean()
            daily_par2 = (np.abs(daily_df["net electricity consumption"]["max"]) /
                          np.abs(daily_df["net electricity consumption"]["mean"]).replace(0, np.nan)).fillna(0).mean()

            df_kpi.loc["Daily Peak Average", "District"] = average_daily_peak
            df_kpi.loc["Monthly Average PAR", "District"] = float(daily_par)
            df_kpi.loc["Monthly Average PAR2", "District"] = float(daily_par2)

            # salva ogni volta sul file CSV corretto
            if algorithm == "RBC":
                kpi_path = kpi_dir / f"{algorithm.lower()}.csv"
            else:
                lr_used = learning_rate
                kpi_path = kpi_dir / f"{algorithm.lower()}_lr={lr_used}.csv"
            df_kpi.to_csv(kpi_path, index=True, index_label="cost_function")


