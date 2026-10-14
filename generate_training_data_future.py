"""
Enhanced Training Dataset Generator – produces training_data_future_5min.csv
with genuine 5-minute-ahead prediction targets.

OUTPUT COLUMNS
--------------
Input features (20):
  hour, minute, day_of_week,
  floor_req_f1..f10,  total_requests,
  elev1_floor, elev2_floor, elev3_floor,
  elev1_occ, elev2_occ, elev3_occ

5-min-ahead targets (11):
  congestion_t5          – congestion_level at t+5 minutes
  demand_f1_t5 .. demand_f10_t5 – per-floor request counts at t+5 minutes

Also retained for diagnostics:
  timestamp, traffic_scenario, congestion_level (current), avg_wait_time

NOTE: This file is SYNTHETIC DATA.  Floor-demand and congestion patterns are
designed to be time-correlated (morning peak, evening peak, lunch rush, night
idle, sudden surge) so the LSTM has a learnable temporal signal.  Predictions
from this model must NOT be presented as measured real-building performance.
"""

import csv
import datetime
import math
import random
import os

# ── generation parameters ───────────────────────────────────────────────────
NUM_ROWS      = 52000  # generate extra rows so 5-min shift still leaves 50 000
NUM_FLOORS    = 10
NUM_ELEVATORS = 3
CAPACITY      = 8
START_TIME    = datetime.datetime(2024, 1, 1, 0, 0, 0)
BASE_WAIT     = 15.0
RANDOM_SEED   = 42          # reproducible dataset


# ── scenario helpers ─────────────────────────────────────────────────────────

def get_scenario_and_base_congestion(hour: int, minute: int) -> tuple[str, float]:
    """Returns (scenario_name, base_congestion) for the given wall-clock time."""
    # 2 % chance of sudden surge regardless of hour
    if random.random() < 0.02:
        congestion = random.uniform(0.80, 1.00)
        return "sudden_surge", congestion

    if hour >= 22 or hour < 6:
        return "night_idle", random.uniform(0.00, 0.08)

    # Ramp up into morning peak (6-8 ramp, 8-10 peak)
    if 6 <= hour < 8:
        t = (hour - 6) * 60 + minute
        ramp = t / 120.0
        return "morning_ramp", random.uniform(0.10 + 0.30 * ramp, 0.20 + 0.50 * ramp)

    if 8 <= hour < 10:
        return "morning_peak", random.uniform(0.70, 0.95)

    if 10 <= hour < 12:
        t = (hour - 10) * 60 + minute
        decay = t / 120.0
        return "mid_morning", random.uniform(0.25 - 0.10 * decay, 0.45 - 0.10 * decay)

    if 12 <= hour < 14:
        return "lunch_rush", random.uniform(0.40, 0.65)

    if 14 <= hour < 17:
        return "afternoon", random.uniform(0.15, 0.35)

    if 17 <= hour < 20:
        return "evening_peak", random.uniform(0.60, 0.90)

    if 20 <= hour < 22:
        t = (hour - 20) * 60 + minute
        decay = t / 120.0
        return "evening_wind", random.uniform(0.20 - 0.10 * decay, 0.55 - 0.15 * decay)

    return "normal", random.uniform(0.10, 0.30)


def generate_floor_requests(scenario: str, congestion: float) -> tuple[list, int]:
    """Returns (floor_reqs[10], total) with realistic floor-distribution by scenario."""
    # Scale total request count by scenario + congestion
    if scenario == "night_idle":
        total = random.randint(0, 2)
    elif scenario in ("morning_peak",):
        total = int(random.uniform(25, 48) * (congestion / 0.825))
    elif scenario == "morning_ramp":
        total = int(random.uniform(5, 25) * (congestion / 0.40))
    elif scenario == "evening_peak":
        total = int(random.uniform(20, 42) * (congestion / 0.75))
    elif scenario == "lunch_rush":
        total = int(random.uniform(15, 32) * (congestion / 0.525))
    elif scenario == "sudden_surge":
        total = int(random.uniform(38, 58) * (congestion / 0.90))
    elif scenario == "mid_morning":
        total = int(random.uniform(8, 20) * (congestion / 0.35))
    elif scenario == "afternoon":
        total = int(random.uniform(5, 18) * (congestion / 0.25))
    elif scenario == "evening_wind":
        total = int(random.uniform(8, 28) * (congestion / 0.375))
    else:
        total = int(random.uniform(4, 15) * (congestion / 0.20))

    total = max(0, int(total))
    floor_reqs = [0] * NUM_FLOORS
    if total == 0:
        return floor_reqs, 0

    if scenario == "morning_peak":
        # 60-80 % of arrivals at lobby (Floor 1)
        lobby = int(total * random.uniform(0.60, 0.80))
        floor_reqs[0] = lobby
        for _ in range(total - lobby):
            floor_reqs[random.randint(1, NUM_FLOORS - 1)] += 1

    elif scenario == "morning_ramp":
        lobby = int(total * random.uniform(0.40, 0.65))
        floor_reqs[0] = lobby
        for _ in range(total - lobby):
            floor_reqs[random.randint(1, NUM_FLOORS - 1)] += 1

    elif scenario == "evening_peak":
        # 60-80 % from upper floors (7-10) heading down
        upper = int(total * random.uniform(0.60, 0.80))
        for _ in range(upper):
            floor_reqs[random.choice([6, 7, 8, 9])] += 1
        for _ in range(total - upper):
            floor_reqs[random.randint(0, 6)] += 1

    elif scenario == "evening_wind":
        upper = int(total * random.uniform(0.40, 0.65))
        for _ in range(upper):
            floor_reqs[random.choice([5, 6, 7, 8, 9])] += 1
        for _ in range(total - upper):
            floor_reqs[random.randint(0, 5)] += 1

    elif scenario == "lunch_rush":
        # Floors 1 and 5 peak
        peak = int(total * random.uniform(0.50, 0.70))
        for _ in range(peak):
            floor_reqs[random.choice([0, 4])] += 1
        for _ in range(total - peak):
            floor_reqs[random.randint(0, NUM_FLOORS - 1)] += 1

    elif scenario == "sudden_surge":
        # Two adjacent random floors get 70-90 %
        start = random.randint(0, NUM_FLOORS - 2)
        cluster = [start, start + 1]
        surge = int(total * random.uniform(0.70, 0.90))
        for _ in range(surge):
            floor_reqs[random.choice(cluster)] += 1
        for _ in range(total - surge):
            floor_reqs[random.randint(0, NUM_FLOORS - 1)] += 1

    else:
        for _ in range(total):
            floor_reqs[random.randint(0, NUM_FLOORS - 1)] += 1

    return floor_reqs, sum(floor_reqs)


# ── main generator ───────────────────────────────────────────────────────────

def generate_dataset(
    output_path: str = "backend/training_data_future_5min.csv",
    num_rows: int = NUM_ROWS,
    seed: int = RANDOM_SEED,
):
    """
    Generates the enhanced dataset.

    Strategy
    --------
    1. First, generate *num_rows* raw timestep records (no target shift yet).
    2. Then, for each row i, attach congestion_t5  = congestion[i+5]
                                    demand_f*_t5  = floor_reqs[i+5]
    Rows without a valid t+5 pair (last 5 rows) are discarded.
    This prevents any future-data leakage into the input features.
    """
    random.seed(seed)
    current_time = START_TIME
    elev_floors  = [1, 5, 10]
    elev_dirs    = [1,  1, -1]
    elev_occs    = [0,  0,  0]

    # --- Pass 1: generate all raw rows -----------------------------------------
    raw: list[dict] = []
    for _ in range(num_rows):
        hour       = current_time.hour
        minute     = current_time.minute
        day_of_week = current_time.weekday()

        scenario, base_cong = get_scenario_and_base_congestion(hour, minute)
        noisy_cong = base_cong + random.gauss(0, 0.04)
        cong = round(max(0.0, min(1.0, noisy_cong)), 4)

        floor_reqs, total_req = generate_floor_requests(scenario, cong)

        # Advance simulated elevator positions
        for i in range(NUM_ELEVATORS):
            step = random.choice([0, 1, 1, 2])
            elev_floors[i] += step * elev_dirs[i]
            if elev_floors[i] >= NUM_FLOORS:
                elev_floors[i] = NUM_FLOORS
                elev_dirs[i]   = -1
            elif elev_floors[i] <= 1:
                elev_floors[i] = 1
                elev_dirs[i]   = 1
            target_occ = int(round(cong * CAPACITY + random.uniform(-1.5, 1.5)))
            elev_occs[i] = max(0, min(CAPACITY, target_occ))

        wait_noise  = random.gauss(0, 2.0)
        avg_wait    = round(max(2.0, BASE_WAIT * (1.0 + cong * 4.0) + wait_noise), 2)

        raw.append({
            "timestamp":      current_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hour":           hour,
            "minute":         minute,
            "day_of_week":    day_of_week,
            "floor_reqs":     list(floor_reqs),
            "total_requests": total_req,
            "elev_floors":    list(elev_floors),
            "elev_occs":      list(elev_occs),
            "avg_wait_time":  avg_wait,
            "traffic_scenario": scenario,
            "congestion_level": cong,
        })
        current_time += datetime.timedelta(minutes=1)

    # --- Pass 2: attach 5-min-ahead targets and write CSV --------------------
    HORIZON = 5   # minutes ahead
    valid_rows = num_rows - HORIZON   # last HORIZON rows have no future target

    headers = [
        # ── input features ────────────────────────────────────────────────
        "timestamp", "hour", "minute", "day_of_week",
        "floor_req_f1",  "floor_req_f2",  "floor_req_f3",  "floor_req_f4",
        "floor_req_f5",  "floor_req_f6",  "floor_req_f7",  "floor_req_f8",
        "floor_req_f9",  "floor_req_f10",
        "total_requests",
        "elev1_floor", "elev2_floor", "elev3_floor",
        "elev1_occ",   "elev2_occ",   "elev3_occ",
        # ── current diagnostics (NOT input to model) ──────────────────────
        "avg_wait_time", "traffic_scenario", "congestion_level",
        # ── 5-minute-ahead prediction targets ────────────────────────────
        "congestion_t5",
        "demand_f1_t5",  "demand_f2_t5",  "demand_f3_t5",  "demand_f4_t5",
        "demand_f5_t5",  "demand_f6_t5",  "demand_f7_t5",  "demand_f8_t5",
        "demand_f9_t5",  "demand_f10_t5",
    ]

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i in range(valid_rows):
            r     = raw[i]
            r_t5  = raw[i + HORIZON]
            row = [
                r["timestamp"], r["hour"], r["minute"], r["day_of_week"],
                *r["floor_reqs"],
                r["total_requests"],
                *r["elev_floors"],
                *r["elev_occs"],
                r["avg_wait_time"], r["traffic_scenario"], r["congestion_level"],
                # targets
                r_t5["congestion_level"],
                *r_t5["floor_reqs"],
            ]
            writer.writerow(row)

    print(f"[DataGen] Generated {valid_rows} rows -> {output_path}")
    print(f"[DataGen] Input features: 20 | Congestion target: t+{HORIZON}min | Per-floor targets: {NUM_FLOORS} x t+{HORIZON}min")
    print(f"[DataGen] Scenarios: night_idle, morning_ramp, morning_peak, mid_morning, lunch_rush, afternoon, evening_peak, evening_wind, sudden_surge")
    print(f"[DataGen] This is SYNTHETIC DATA — do not represent predictions as real building measurements.")
    return output_path


if __name__ == "__main__":
    import sys
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "backend", "training_data_future_5min.csv")
    if len(sys.argv) > 1:
        out = sys.argv[1]
    generate_dataset(output_path=out)
