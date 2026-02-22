import numpy as np
import matplotlib.pyplot as plt
from openpyxl import Workbook

def cop_from_ambient(T_amb_C):
    COP_ref = 6.0
    T_ref = 25.0
    k = 0.08
    COP_min = 2.5
    cop = COP_ref - k * (T_amb_C - T_ref)
    return max(cop, COP_min)

def simulate():
    dt_s = 10
    duration_min = 120
    n = int(duration_min * 60 / dt_s)

    t_s = np.arange(n) * dt_s
    t_min = t_s / 60.0

    # Plant parameters
    T_set_C = 7.0
    C_th_J_per_K = 250_000_000.0
    Q_cool_max_W = 15e6
    Kp = 0.15

    # Ambient
    T_amb_C = np.ones(n) * (28.0 + 4.0)  # 28C + 4C heat island

    # IT load with spikes
    Q_it_W = np.ones(n) * 8e6
    spike_starts_min = [20, 50, 80, 95]
    spike_duration_min = 6
    spike_add_W = 4e6

    for start_min in spike_starts_min:
        start = int(start_min * 60 / dt_s)
        end = start + int(spike_duration_min * 60 / dt_s)
        Q_it_W[start:end] += spike_add_W

    # States / outputs
    T_supply_C = np.zeros(n)
    T_supply_C[0] = T_set_C
    Q_cool_W = np.zeros(n)
    P_chiller_W = np.zeros(n)

    # Sim loop
    for i in range(1, n):
        error = T_supply_C[i - 1] - T_set_C
        u = Kp * error
        if u < 0.0:
            u = 0.0
        if u > 1.0:
            u = 1.0

        Q_cool_W[i] = u * Q_cool_max_W

        dT = (Q_it_W[i] - Q_cool_W[i]) * dt_s / C_th_J_per_K
        T_supply_C[i] = T_supply_C[i - 1] + dT

        cop = cop_from_ambient(T_amb_C[i])
        P_chiller_W[i] = Q_cool_W[i] / cop

    # KPIs (manual trapezoid integration)
    max_excursion_C = float(np.max(T_supply_C - T_set_C))
    peak_power_MW = float(np.max(P_chiller_W) / 1e6)

    energy_Wh = 0.0
    for i in range(1, n):
        energy_Wh += 0.5 * (P_chiller_W[i] + P_chiller_W[i - 1]) * dt_s / 3600.0
    energy_MWh = float(energy_Wh / 1e6)

    # Save to Excel .xlsx
    wb = Workbook()
    ws = wb.active
    ws.title = "timeseries"

    ws.append(["time_min", "T_amb_C", "Q_it_MW", "Q_cool_MW", "P_chiller_MW", "T_supply_C"])
    for i in range(n):
        ws.append([
            float(t_min[i]),
            float(T_amb_C[i]),
            float(Q_it_W[i] / 1e6),
            float(Q_cool_W[i] / 1e6),
            float(P_chiller_W[i] / 1e6),
            float(T_supply_C[i]),
        ])

    ws2 = wb.create_sheet("kpis")
    ws2.append(["metric", "value"])
    ws2.append(["max_supply_temp_excursion_C", float(max_excursion_C)])
    ws2.append(["peak_chiller_power_MW", float(peak_power_MW)])
    ws2.append(["chiller_energy_MWh_2hr", float(energy_MWh)])

    wb.save("results_timeseries.xlsx")

    print("RESULTS")
    print(f"Max supply temp excursion above setpoint: {max_excursion_C:.3f} C")
    print(f"Peak chiller electric power: {peak_power_MW:.2f} MW")
    print(f"Chiller energy (2 hours): {energy_MWh:.2f} MWh")
    print("Saved: results_timeseries.xlsx")

    # Plot 1
    plt.figure()
    plt.plot(t_min, Q_it_W / 1e6, label="IT Load (MW)")
    plt.plot(t_min, Q_cool_W / 1e6, label="Cooling Provided (MW)")
    plt.xlabel("Time (min)")
    plt.ylabel("MW")
    plt.title("IT Load vs Cooling")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results_load_vs_cooling.png", dpi=200)
    plt.show()

    # Plot 2
    plt.figure()
    plt.plot(t_min, T_supply_C, label="CHW Supply Temp (C)")
    plt.axhline(T_set_C, linestyle="--", label="Setpoint")
    plt.xlabel("Time (min)")
    plt.ylabel("C")
    plt.title("Supply Temperature Response")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results_supply_temp.png", dpi=200)
    plt.show()

    # Plot 3
    plt.figure()
    plt.plot(t_min, P_chiller_W / 1e6, label="Chiller Electric Power (MW)")
    plt.xlabel("Time (min)")
    plt.ylabel("MW")
    plt.title("Chiller Power")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results_chiller_power.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    simulate()