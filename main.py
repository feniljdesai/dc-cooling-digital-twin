sim:
  dt_s: 10
  duration_min: 120

ambient:
  base_C: 28
  heat_island_delta_C: 4

it_load:
  base_MW: 8
  spike_MW: 4
  spike_minutes: [20, 50, 80, 95]
  spike_duration_min: 6

plant:
  T_supply_set_C: 7.0
  C_th_J_per_K: 2.5e8
  Q_cool_max_MW: 15
  Kp: 0.15

chiller:
  cop_ref: 6.0
  T_ref_C: 25.0
  k_per_C: 0.08
  cop_min: 2.5