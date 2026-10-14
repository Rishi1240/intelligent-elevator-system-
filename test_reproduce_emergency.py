import backend.app as app

env = app.env
emg = app.emergency_dispatcher
env.reset()
emg.reset()
env.elevator_positions = [2, 2, 2]

print("=== INITIAL STATE ===")
print("Positions:", env.elevator_positions)

res = emg.dispatch_emergency(9)
print("\n=== DISPATCH RESULT ===")
print("Status:", res["status"])
print("Emergency:", res["emergency"])
print("Stops:", env.assigned_stops)
print("Targets:", env.committed_targets)

print("\n=== SIMULATION STEPS ===")
for step in range(1, 12):
    next_state, reward, info, has_work = app.run_rl_dispatch_step()
    active_emg = emg.get_all_active()
    statuses = [e["status"] for e in active_emg]
    print(f"Step {step:2d}: pos={env.elevator_positions}, targets={env.committed_targets}, stops={env.assigned_stops}, emg_statuses={statuses}, has_work={has_work}")
