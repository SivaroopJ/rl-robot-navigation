"""Export animated rollout data for the reproduction viewer.

Read-only w.r.t. the original implementation: it imports RobotNavEnv and the trained
curriculum model and replays the SAME protocol as evaluation/evaluate.py
(reset(seed=episode_index), deterministic=True), additionally recording per-step obstacle
positions and LiDAR returns, which evaluate() does not store.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.getcwd())
from stable_baselines3 import PPO
from robot_env.robot_nav_env import RobotNavEnv

MODEL = "models/ppo_robot_nav_curriculum_shaping"
CONFIGS = [
    ("e1", 3,  1.0, "trained"),
    ("e1", 6,  1.0, "trained"),
    ("e1", 10, 1.0, "trained"),
    ("e2", 6,  0.5, "trained"),
    ("e2", 6,  1.5, "trained"),
    ("e3", 5,  0.8, "unseen"),
    ("e3", 8,  1.2, "unseen"),
    ("e3", 10, 1.5, "unseen"),
    ("e3", 12, 2.0, "unseen"),
    ("e3", 15, 0.3, "unseen"),
]
N_EP = 30
r3 = lambda a: [round(float(x), 3) for x in np.asarray(a).ravel()]

model = PPO.load(MODEL, device="cpu")
out = {"world": None, "static": None, "configs": []}

for exp, n_obs, spd, kind in CONFIGS:
    env = RobotNavEnv(config_path="config.json", n_dynamic_obstacles=n_obs, obstacle_speed=spd)
    if out["world"] is None:
        out["world"] = {"size": env.WORLD_SIZE, "agent_r": env.AGENT_RADIUS,
                        "obs_r": env.OBSTACLE_RADIUS, "goal_r": env.TARGET_RADIUS,
                        "lidar_range": env.LIDAR_RANGE, "n_rays": env.N_LIDAR_RAYS}
        out["static"] = [list(map(float, s)) for s in env.static_obstacles]
    eps = []
    for ep in range(N_EP):
        obs, _ = env.reset(seed=ep)
        agent, obstacles, lidar, rews = [], [], [], []
        agent.append(r3(env.agent_position))
        obstacles.append([r3(p) for p in env.obstacle_positions])
        lidar.append(r3(env._cast_lidar_rays()))
        goal = r3(env.target_position)
        term = trunc = False
        total = 0.0
        while not (term or trunc):
            a, _ = model.predict(obs, deterministic=True)
            obs, rew, term, trunc, info = env.step(a)
            total += float(rew)
            agent.append(r3(env.agent_position))
            obstacles.append([r3(p) for p in env.obstacle_positions])
            lidar.append(r3(env._cast_lidar_rays()))
            rews.append(round(float(rew), 3))
        outcome = "success" if info["success"] else ("timeout" if trunc else "collision")
        eps.append({"seed": ep, "goal": goal, "agent": agent, "obs": obstacles,
                    "lidar": lidar, "rew": rews, "ret": round(total, 2),
                    "outcome": outcome, "steps": len(rews)})
    env.close()
    sr = sum(1 for e in eps if e["outcome"] == "success") / len(eps)
    out["configs"].append({"id": f"obs{n_obs}_spd{spd}", "exp": exp, "n": n_obs,
                           "spd": spd, "kind": kind, "sr30": round(sr, 4), "episodes": eps})
    print(f"{exp} N={n_obs:<3} spd={spd:<4} -> {sr*100:5.1f}% over {N_EP} eps")

dst = "reproduction_results/plots/rollouts.json"
with open(dst, "w") as f:
    json.dump(out, f, separators=(",", ":"))
print("wrote", dst, os.path.getsize(dst) // 1024, "KB")
