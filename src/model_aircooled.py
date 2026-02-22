import numpy as np

def cop_from_ambient(T_amb, cop_ref=6.0, T_ref=25.0, k_per_C=0.08, cop_min=2.5):
    cop = cop_ref - k_per_C * (T_amb - T_ref)
    return max(cop, cop_min)

def simulate_aircooled(Q_it_w, T_amb_c, dt_s, plant, chiller):
    n = len(Q_it_w)
    T_supply = np.zeros(n)
    Q_cool = np.zeros(n)
    P_chiller = np.zeros(n)

    T_supply[0] = plant["T_supply_set_C"]

    for i in range(1, n):
        e = T_supply[i - 1] - plant["T_supply_set_C"]
        u = np.clip(plant["Kp"] * e, 0.0, 1.0)

        Q_cool[i] = u * (plant["Q_cool_max_MW"] * 1e6)

        dT = (Q_it_w[i] - Q_cool[i]) * dt_s / plant["C_th_J_per_K"]
        T_supply[i] = T_supply[i - 1] + dT

        cop = cop_from_ambient(
            T_amb_c[i],
            cop_ref=chiller["cop_ref"],
            T_ref=chiller["T_ref_C"],
            k_per_C=chiller["k_per_C"],
            cop_min=chiller["cop_min"],
        )
        P_chiller[i] = Q_cool[i] / cop

    return {
        "T_supply_C": T_supply,
        "Q_cool_W": Q_cool,
        "P_chiller_W": P_chiller,
    }