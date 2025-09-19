from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple

# ---------- path helpers ----------

ALGO_DIR = {
    "AAC-MADRL": "aac_madrl",
    "SAC": "sac",
    "RBC": "rbc",
}

def find_obs_csv(outputs_root: Path, dataset_key: str, algorithm: str, beta: float, lr: float, gamma: float) -> Path:
    """
    Trova district_obs.csv nel layout nuovo (beta=..._gamma=.../lr=...) con fallback al layout vecchio (beta=.../lr=...).
    Struttura base: <outputs_root>/<dataset_key>/schema.json/obs/<algo_dir>/...
    """
    algo = algorithm.upper()
    algo_dir = ALGO_DIR.get(algo, algo.lower())
    base = outputs_root / dataset_key / "schema.json" / "obs" / algo_dir

    if algo == "RBC":
        candidates = [base / "district_obs.csv"]
    else:
        candidates = [
            base / f"beta={beta}_gamma={gamma}" / f"lr={lr}" / "district_obs.csv",  # nuovo
            base / f"beta={beta}"              / f"lr={lr}" / "district_obs.csv",   # vecchio (senza gamma)
        ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"district_obs.csv non trovato per {algorithm}. Cercati: {', '.join(map(str, candidates))}"
    )

def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def load_or_init_kpi_csv(kpi_path: Path) -> pd.DataFrame:
    """Apre il KPI CSV se esiste, altrimenti crea un dataframe vuoto con indice 'cost_function'."""
    if kpi_path.exists():
        try:
            return pd.read_csv(kpi_path, index_col="cost_function")
        except Exception:
            return pd.read_csv(kpi_path, index_col=0)

    idx = [
        "Import",
        "Variance",
        "Daily Peak Average",
        "Avg Num Comfort Violations",
        "Avg Comfort Violation Above (°C)",
        "Avg Comfort Violation Below (°C)",
        "Avg Num Comfort Violations Above",
        "Avg Num Comfort Violations Below",
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

def _mean_violation_for_building_csv(csv_path: Path) -> Optional[Tuple[float, float, float, float, float]]:
    """
    Media violazioni per un edificio.
    Ritorna: (num_violazioni_tot, mean_above, mean_below, num_viol_above, num_viol_below)
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    required = {"indoor_temperature", "heating_sp", "cooling_sp", "comfort_band"}
    if not required.issubset(df.columns):
        return None

    # come richiesto: heating_sp == cooling_sp
    if not df["heating_sp"].equals(df["cooling_sp"]):
        raise AssertionError("Set point di riscaldamento e raffreddamento devono essere uguali")

    SP = pd.to_numeric(df["heating_sp"],        errors="coerce")
    T  = pd.to_numeric(df["indoor_temperature"], errors="coerce")
    B  = pd.to_numeric(df["comfort_band"],       errors="coerce").abs()

    low, high = SP - B, SP + B

    above = (T - high).clip(lower=0)
    below = (low - T).clip(lower=0)

    mean_above = float(above[above > 0].mean()) if (above > 0).any() else 0.0
    mean_below = float(below[below > 0].mean()) if (below > 0).any() else 0.0

    num_above = int((above > 0).sum())
    num_below = int((below > 0).sum())
    num_total = num_above + num_below

    return num_total, mean_above, mean_below, num_above, num_below

def mean_comfort_violation_for_folder(obs_folder: Path) -> Tuple[float, float, float, float, float]:
    """
    Scansiona obs_building_*.csv e ritorna le medie (tra edifici) di:
      (num_viol_tot, mean_above, mean_below, num_viol_above, num_viol_below)
    """
    num_violations, vas, vbs, nvas, nvbs = [], [], [], [], []

    for f in sorted(obs_folder.glob("obs_building_*.csv")):
        res = _mean_violation_for_building_csv(f)
        if res is None:
            continue
        num_v, va, vb, nva, nvb = res

        num_violations.append(num_v)
        vas.append(va)
        vbs.append(vb)
        nvas.append(nva)
        nvbs.append(nvb)

    def _safe_mean(lst: List[float]) -> float:
        return float(np.mean(lst)) if len(lst) > 0 else 0.0

    return (
        _safe_mean(num_violations),
        _safe_mean(vas),
        _safe_mean(vbs),
        _safe_mean(nvas),
        _safe_mean(nvbs),
    )

# ---------- KPI core ----------

def process_kpi(outputs_root: Path, dataset_key: str, algorithm: str, kpi_dir: Path, lr: float, beta: float, gamma: float):
    """Legge district_obs.csv e aggiorna/salva il file KPI dell'algoritmo (CSV)."""
    obs_csv = find_obs_csv(outputs_root, dataset_key, algorithm, beta, lr, gamma)
    df_obs = pd.read_csv(obs_csv)

    # KPI file path (CSV)
    algo = algorithm.upper()
    if algo == "RBC":
        kpi_file = kpi_dir / f"{algo.lower()}.csv"
    else:
        kpi_file = kpi_dir / f"{algo.lower()}_lr={lr}.csv"

    ensure_parent(kpi_file)
    df_kpi = load_or_init_kpi_csv(kpi_file)

    # --- metriche base minime ---
    total_consumption = float(np.sum(df_obs["positive net electricity consumption"]))
    df_kpi.loc["Import", "District"] = total_consumption
    df_kpi.loc["Variance", "District"] = float(np.var(df_obs["net electricity consumption"]))

    # --- KPI comfort: medie su tutti gli edifici della cartella ---
    obs_folder = obs_csv.parent
    num_violations, va, vb, nva, nvb = mean_comfort_violation_for_folder(obs_folder)

    df_kpi.loc["Avg Num Comfort Violations", "District"] = float(num_violations)
    df_kpi.loc["Avg Comfort Violation Above (°C)", "District"] = float(va)
    df_kpi.loc["Avg Comfort Violation Below (°C)", "District"] = float(vb)
    df_kpi.loc["Avg Num Comfort Violations Above", "District"] = float(nva)
    df_kpi.loc["Avg Num Comfort Violations Below", "District"] = float(nvb)

    # salva KPI aggiornato in CSV
    df_kpi.to_csv(kpi_file, index=True, index_label="cost_function")
    return df_obs, df_kpi, kpi_file

# ---------- driver ----------

if __name__ == "__main__":
    # Config
    building_counts = [10]
    learning_rate = 0.001
    control_algorithms = ["RBC", "SAC", "AAC-MADRL"]
    beta = 0.0
    gamma = 2.8
    dataset = "CA"                         # per comporre dataset_key
    start_date = "2017-02-01 00:00:00"

    outputs_root = Path.cwd() / "outputs" / "data"

    for n in building_counts:
        # es. "TX_10_dynamics"
        dataset_key = f"{dataset}_{n}_dynamics"

        # KPI dir (coerente con layout nuovo, dentro schema.json)
        kpi_dir = outputs_root / dataset_key / "schema.json" / "kpi" / "test" / f"beta={beta}_gamma={gamma}" / f"lr={learning_rate}"
        kpi_dir.mkdir(parents=True, exist_ok=True)

        # Processa ogni algoritmo
        df_rbc, df_rbc_kpi, _ = process_kpi(outputs_root, dataset_key, "RBC",        kpi_dir, learning_rate, beta, gamma)
        df_aac, df_aac_kpi,   _ = process_kpi(outputs_root, dataset_key, "AAC-MADRL", kpi_dir, learning_rate, beta, gamma)
        df_sac, df_sac_kpi,   _ = process_kpi(outputs_root, dataset_key, "SAC",       kpi_dir, learning_rate, beta, gamma)

        # allinea index a datetime (se serve in seguito)
        num_hours = len(df_rbc)
        date_range = pd.date_range(start=start_date, periods=num_hours, freq="h")
        for df in [df_rbc, df_aac, df_sac]:
            df.index = date_range

        # daily aggregates (se servono)
        daily_dfs = {
            "RBC": df_rbc.resample("D").agg({
                "net electricity consumption": ["max", "mean"],
                "positive net electricity consumption": ["max", "mean"],
            }),
            "AAC-MADRL": df_aac.resample("D").agg({
                "net electricity consumption": ["max", "mean"],
                "positive net electricity consumption": ["max", "mean"],
            }),
            "SAC": df_sac.resample("D").agg({
                "net electricity consumption": ["max", "mean"],
                "positive net electricity consumption": ["max", "mean"],
            }),
        }

        for algorithm in ["RBC", "AAC-MADRL", "SAC"]:
            daily_df = daily_dfs[algorithm]
            df_kpi = {
                "RBC": df_rbc_kpi,
                "AAC-MADRL": df_aac_kpi,
                "SAC": df_sac_kpi,
            }[algorithm]

            average_daily_peak = float(daily_df["net electricity consumption"]["max"].mean())
            df_kpi.loc["Daily Peak Average", "District"] = average_daily_peak

            # salva ogni volta sul file CSV corretto
            if algorithm == "RBC":
                kpi_path = kpi_dir / f"{algorithm.lower()}.csv"
            else:
                kpi_path = kpi_dir / f"{algorithm.lower()}_lr={learning_rate}.csv"
            df_kpi.to_csv(kpi_path, index=True, index_label="cost_function")



