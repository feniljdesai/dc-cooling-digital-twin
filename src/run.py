import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from model_aircooled import simulate_aircooled
from kpis import compute_kpis

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_profiles(cfg):
    dt = cfg["sim"]["dt_s"]
    duration_min = cfg["sim"]["duration_min"]
    n = int(duration_min * 60 / dt)
    t_s = np.arange(n) * dt

    t_min = t_s / 60.0
    T_amb = cfg["ambient"]["base_C"] + 3.0 * np.sin(2.0 * np.pi * (t_min / (24 * 60)))
    T_amb += cfg["ambient"]["heat_island_delta_C"]

    Q_it = np.ones(n) * (cfg["it_load"]["base_MW"] * 1e6)
    spike_MW = cfg["it_load"]["spike_MW"]
    spike_dur = cfg["it_load"]["spike_duration_min"]

    for start_min in cfg["it_load"]["spike_minutes"]:
        start = int(start_min * 60 / dt)
        end = start + int(spike_dur * 60 / dt)
        Q_it[start:end] += spike_MW * 1e6

    return t_s, T_amb, Q_it

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    t_s, T_amb, Q_it = build_profiles(cfg)

    out = simulate_aircooled(
        Q_it_w=Q_it,
        T_amb_c=T_amb,
        dt_s=cfg["sim"]["dt_s"],
        plant=cfg["plant"],
        chiller=cfg["chiller"],
    )

    kpis = compute_kpis(
        t_s=t_s,
        T_supply_C=out["T_supply_C"],
        P_chiller_W=out["P_chiller_W"],
        Q_it_W=Q_it,
        T_set_C=cfg["plant"]["T_supply_set_C"],
    )

    df = pd.DataFrame({
        "time_s": t_s,
        "T_amb_C": T_amb,
        "Q_it_MW": Q_it / 1e6,
        "Q_cool_MW": out["Q_cool_W"] / 1e6,
        "P_chiller_MW": out["P_chiller_W"] / 1e6,
        "T_supply_C": out["T_supply_C"],
    })
    csv_path = results_dir / "aircooled_timeseries.csv"
    df.to_csv(csv_path, index=False)

    t_min = t_s / 60.0

    plt.figure()
    plt.plot(t_min, df["Q_it_MW"], label="IT Load (MW)")
    plt.plot(t_min, df["Q_cool_MW"], label="Cooling (MW)")
    plt.xlabel("Time (min)")
    plt.ylabel("MW")
    plt.legend()
    plt.title("Load vs Cooling")
    plt.tight_layout()
    plt.savefig(results_dir / "load_vs_cooling.png", dpi=200)

    plt.figure()
    plt.plot(t_min, df["T_supply_C"], label="CHW Supply (C)")
    plt.axhline(cfg["plant"]["T_supply_set_C"], linestyle="--", label="Setpoint")
    plt.xlabel("Time (min)")
    plt.ylabel("C")
    plt.legend()
    plt.title("Supply Temperature Response")
    plt.tight_layout()
    plt.savefig(results_dir / "supply_temp.png", dpi=200)

    plt.figure()
    plt.plot(t_min, df["P_chiller_MW"], label="Chiller Power (MW)")
    plt.xlabel("Time (min)")
    plt.ylabel("MW")
    plt.legend()
    plt.title("Chiller Electric Power")
    plt.tight_layout()
    plt.savefig(results_dir / "chiller_power.png", dpi=200)

    print("\nKPIs")
    for k, v in kpis.items():
        print(f"  {k}: {v}")

    print(f"\nSaved: {csv_path}")
    print("Saved plots to /results")

if __name__ == "__main__":
    main()