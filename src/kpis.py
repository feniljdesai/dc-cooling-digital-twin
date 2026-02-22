import numpy as np

def compute_kpis(t_s, T_supply_C, P_chiller_W, Q_it_W, T_set_C):
    max_excursion_C = float(np.max(T_supply_C - T_set_C))
    time_above_set_min = float(np.sum(T_supply_C > T_set_C) * (t_s[1] - t_s[0]) / 60.0)

   energy_MWh = float(np.trapezoid(P_chiller_W, t_s) / 3.6e9)
    peak_MW = float(np.max(P_chiller_W) / 1e6)

    avg_it_MW = float(np.mean(Q_it_W) / 1e6)
    avg_chiller_MW = float(np.mean(P_chiller_W) / 1e6)
    kw_per_mw_it = float((avg_chiller_MW * 1000.0) / max(avg_it_MW, 1e-9))

    return {
        "max_supply_temp_excursion_C": max_excursion_C,
        "time_above_setpoint_min": time_above_set_min,
        "chiller_energy_MWh": energy_MWh,
        "chiller_peak_MW": peak_MW,
        "avg_kW_per_MW_IT": kw_per_mw_it,
    }