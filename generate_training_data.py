"""
Elevator Traffic Training Dataset Generator
Generates realistic elevator simulation data for LSTM traffic & congestion forecasting.
"""

import csv
import datetime
import math
import random
import os

NUM_ROWS = 50000
NUM_FLOORS = 10
NUM_ELEVATORS = 3
CAPACITY = 8
START_TIME = datetime.datetime(2024, 1, 1, 0, 0, 0)
BASE_WAIT = 15.0

def get_scenario_and_congestion(hour, minute):
    # 2% chance of sudden surge regardless of hour
    if random.random() < 0.02:
        congestion = random.uniform(0.8, 1.0)
        return "sudden_surge", congestion

    # Night idle: 22:00 to 06:00
    if hour >= 22 or hour < 6:
        congestion = random.uniform(0.0, 0.08)
        return "night_idle", congestion

    # Morning peak: 08:00 to 10:00
    if 8 <= hour < 10:
        congestion = random.uniform(0.7, 0.95)
        return "morning_peak", congestion

    # Lunch rush: 12:00 to 14:00
    if 12 <= hour < 14:
        congestion = random.uniform(0.4, 0.65)
        return "lunch_rush", congestion

    # Evening peak: 17:00 to 20:00
    if 17 <= hour < 20:
        congestion = random.uniform(0.6, 0.9)
        return "evening_peak", congestion

    # Normal daytime
    congestion = random.uniform(0.1, 0.3)
    return "normal", congestion

def generate_floor_requests(scenario, congestion):
    # Determine total requests scaled by congestion
    if scenario == "night_idle":
        total_requests = random.randint(0, 2)
    elif scenario == "morning_peak":
        total_requests = int(random.uniform(25, 45) * (congestion / 0.8))
    elif scenario == "evening_peak":
        total_requests = int(random.uniform(20, 40) * (congestion / 0.75))
    elif scenario == "lunch_rush":
        total_requests = int(random.uniform(15, 30) * (congestion / 0.5))
    elif scenario == "sudden_surge":
        total_requests = int(random.uniform(35, 55) * (congestion / 0.9))
    else:  # normal
        total_requests = int(random.uniform(5, 15) * (congestion / 0.2))

    total_requests = max(0, total_requests)
    floor_reqs = [0] * NUM_FLOORS

    if total_requests == 0:
        return floor_reqs, 0

    if scenario == "morning_peak":
        # Floor 1 gets 60-80% of requests (people arriving at lobby)
        lobby_ratio = random.uniform(0.6, 0.8)
        lobby_count = int(total_requests * lobby_ratio)
        floor_reqs[0] = lobby_count
        remaining = total_requests - lobby_count
        for _ in range(remaining):
            idx = random.randint(1, NUM_FLOORS - 1)
            floor_reqs[idx] += 1

    elif scenario == "evening_peak":
        # Floors 8-10 get 60-80% of requests (people leaving upper floors)
        upper_ratio = random.uniform(0.6, 0.8)
        upper_count = int(total_requests * upper_ratio)
        remaining = total_requests - upper_count
        for _ in range(upper_count):
            idx = random.choice([7, 8, 9])  # Floors 8, 9, 10 (0-indexed 7, 8, 9)
            floor_reqs[idx] += 1
        for _ in range(remaining):
            idx = random.randint(0, 6)
            floor_reqs[idx] += 1

    elif scenario == "lunch_rush":
        # Floors 1 and 5 peak
        peak_ratio = random.uniform(0.5, 0.7)
        peak_count = int(total_requests * peak_ratio)
        remaining = total_requests - peak_count
        for _ in range(peak_count):
            idx = random.choice([0, 4])  # Floor 1 or Floor 5
            floor_reqs[idx] += 1
        for _ in range(remaining):
            idx = random.randint(0, NUM_FLOORS - 1)
            floor_reqs[idx] += 1

    elif scenario == "sudden_surge":
        # Random cluster of 2 adjacent floors gets 70-90%
        cluster_start = random.randint(0, NUM_FLOORS - 2)
        cluster_floors = [cluster_start, cluster_start + 1]
        cluster_ratio = random.uniform(0.7, 0.9)
        cluster_count = int(total_requests * cluster_ratio)
        remaining = total_requests - cluster_count
        for _ in range(cluster_count):
            idx = random.choice(cluster_floors)
            floor_reqs[idx] += 1
        for _ in range(remaining):
            idx = random.randint(0, NUM_FLOORS - 1)
            floor_reqs[idx] += 1

    else:
        # Normal uniform distribution
        for _ in range(total_requests):
            idx = random.randint(0, NUM_FLOORS - 1)
            floor_reqs[idx] += 1

    actual_total = sum(floor_reqs)
    return floor_reqs, actual_total

def generate_dataset(output_paths=[
    "training_data.csv",
    os.path.join("backend", "training_data.csv")
], num_rows=NUM_ROWS):
    current_time = START_TIME
    elev_floors = [1, 5, 10]
    elev_dirs = [1, 1, -1]  # 1 for up, -1 for down
    elev_occs = [0, 0, 0]

    headers = [
        "timestamp", "hour", "minute", "day_of_week",
        "floor_req_f1", "floor_req_f2", "floor_req_f3", "floor_req_f4", "floor_req_f5",
        "floor_req_f6", "floor_req_f7", "floor_req_f8", "floor_req_f9", "floor_req_f10",
        "total_requests",
        "elev1_floor", "elev2_floor", "elev3_floor",
        "elev1_occ", "elev2_occ", "elev3_occ",
        "avg_wait_time", "traffic_scenario", "congestion_level"
    ]

    rows = []

    for _ in range(num_rows):
        hour = current_time.hour
        minute = current_time.minute
        day_of_week = current_time.weekday()  # 0 to 6

        scenario, base_congestion = get_scenario_and_congestion(hour, minute)

        # Add gaussian noise std=0.05 to congestion
        noisy_congestion = base_congestion + random.gauss(0, 0.05)
        congestion_level = max(0.0, min(1.0, round(noisy_congestion, 4)))

        floor_reqs, total_requests = generate_floor_requests(scenario, congestion_level)

        # Realistic elevator updates per minute
        for i in range(NUM_ELEVATORS):
            # Move elevator 0-2 floors towards direction
            step = random.choice([0, 1, 1, 2])
            elev_floors[i] += step * elev_dirs[i]
            if elev_floors[i] >= NUM_FLOORS:
                elev_floors[i] = NUM_FLOORS
                elev_dirs[i] = -1
            elif elev_floors[i] <= 1:
                elev_floors[i] = 1
                elev_dirs[i] = 1

            # Update occupancy based on congestion
            target_occ = int(round(congestion_level * CAPACITY + random.uniform(-1.5, 1.5)))
            target_occ = max(0, min(CAPACITY, target_occ))
            elev_occs[i] = target_occ

        # Calculate avg_wait_time: base_wait * (1 + congestion_level * 4) + noise
        wait_noise = random.gauss(0, 2.0)
        avg_wait_time = max(2.0, round(BASE_WAIT * (1.0 + congestion_level * 4.0) + wait_noise, 2))

        iso_timestamp = current_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        row = [
            iso_timestamp, hour, minute, day_of_week,
            *floor_reqs,
            total_requests,
            elev_floors[0], elev_floors[1], elev_floors[2],
            elev_occs[0], elev_occs[1], elev_occs[2],
            avg_wait_time, scenario, congestion_level
        ]
        rows.append(row)
        current_time += datetime.timedelta(minutes=1)

    for path in output_paths:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

    print(f"Dataset generated: {num_rows} rows")

if __name__ == "__main__":
    generate_dataset()
