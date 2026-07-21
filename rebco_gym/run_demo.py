#!/usr/bin/env python3
"""
Example: run both REBCO gym environments with simple heuristic agents.

Usage:
    python examples/run_demo.py
"""

import numpy as np
import gymnasium as gym
import rebco_gym  # registers the envs


def run_thermal_runaway_demo():
    """Demo the ThermalRunawayEnv with a threshold-based agent."""
    print("=" * 70)
    print("  ThermalRunawayEnv-v0 Demo")
    print("  Action: 0=HOLD, 1=RAMP_DOWN")
    print("  Agent: ramp down when margin > 0.85 or dT/dt > 1.0 K/s")
    print("=" * 70)

    env = gym.make("ThermalRunawayEnv-v0", render_mode="ansi",
                    disturbance_prob=0.08, max_steps=300)

    n_episodes = 5
    results = {"quenched": 0, "safe_shutdown": 0, "timeout": 0}

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        total_reward = 0.0
        done = False

        while not done:
            # Observation: [T, T0, I_op, Ic(T), dTdt, V_qa, margin]
            T, T0, I_op, Ic_T, dTdt, V_qa, margin = obs

            # Simple heuristic policy
            if margin > 0.85 or dTdt > 1.0 or V_qa > 0.5:
                action = 1  # RAMP DOWN
            else:
                action = 0  # HOLD

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            if info["step"] % 50 == 0:
                rendered = env.render()
                if rendered:
                    print(rendered)

        # Classify outcome
        if terminated:
            results["safe_shutdown"] += 1
            outcome = "SAFE SHUTDOWN"
        elif info.get("dT", 0) >= 12.0:
            results["quenched"] += 1
            outcome = "QUENCH!"
        else:
            results["timeout"] += 1
            outcome = "TIMEOUT"

        print(f"\n  Episode {ep+1}: {outcome} | "
              f"Steps={info['step']} | Reward={total_reward:.1f} | "
              f"Final T={info['T']:.1f}K\n")

    env.close()
    print(f"\nResults over {n_episodes} episodes:")
    print(f"  Safe shutdowns: {results['safe_shutdown']}")
    print(f"  Quenches:       {results['quenched']}")
    print(f"  Timeouts:       {results['timeout']}")
    return results


def run_current_redist_demo():
    """Demo the CurrentRedistEnv with a proportional controller."""
    print("\n" + "=" * 70)
    print("  CurrentRedistEnv-v0 Demo")
    print("  Action: continuous [0,1] current fraction")
    print("  Agent: reduce current proportional to max temperature rise")
    print("=" * 70)

    env = gym.make("CurrentRedistEnv-v0", render_mode="ansi",
                    disturbance_prob=0.05, max_steps=500)

    n_episodes = 3

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep + 100)
        total_reward = 0.0
        done = False
        nx = env.unwrapped.nx

        while not done:
            # Extract key quantities from observation
            # obs = [Ic(nx), E(nx), Jc(nx), T(nx), V_qa, I_total, T_max]
            T_rebco = obs[3*nx : 4*nx]
            V_qa = obs[4*nx]
            T_max = obs[4*nx + 2]
            T0 = info["T0"]

            # Proportional controller: reduce current if T rises
            dT = T_max - T0
            dT_crit = env.unwrapped.dT_crit
            if dT > dT_crit * 0.3:
                # Proportionally reduce
                frac = max(1.0 - (dT / dT_crit), 0.1)
            else:
                frac = 1.0

            action = np.array([frac], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            if info["step"] % 100 == 0:
                rendered = env.render()
                if rendered:
                    print(rendered)

        print(f"\n  Episode {ep+1}: Steps={info['step']} | "
              f"Reward={total_reward:.1f} | T_max={info['T_max']:.1f}K\n")

    env.close()


def run_full_env_demo():
    """Demo REBCOFullEnv-v0 with a margin-aware throttle-back controller."""
    print("\n" + "=" * 70)
    print("  REBCOFullEnv-v0 Demo")
    print("  Action: continuous [0,1] fraction of the requested ramp rate")
    print("  Agent: throttle back when local margin or dT looks risky")
    print("=" * 70)

    env = gym.make("REBCOFullEnv-v0", render_mode="ansi", max_steps=1000)

    n_episodes = 5
    results = {"reached_target": 0, "quenched": 0, "timeout": 0}

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        total_reward = 0.0
        done = False

        while not done:
            # obs = [I_op, I_req, T0, T_max, V, V_qa, dTdt, margin, B_self, B_ext]
            I_op, I_req, T0, T_max, V, V_qa, dTdt, margin, B_self, B_ext = obs

            if (T_max - T0) > 2.0 or margin > 3.0 or dTdt > 200.0:
                frac = 0.1
            else:
                frac = 1.0

            action = np.array([frac], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            if info["step"] % 200 == 0:
                rendered = env.render()
                if rendered:
                    print(rendered)

        dT = info["T_max"] - info["T0"]
        if terminated:
            results["reached_target"] += 1
            outcome = "TARGET REACHED"
        elif dT >= env.unwrapped.dT_crit:
            results["quenched"] += 1
            outcome = "QUENCH!"
        else:
            results["timeout"] += 1
            outcome = "TIMEOUT"

        print(f"\n  Episode {ep+1}: {outcome} | Steps={info['step']} | "
              f"Reward={total_reward:.1f} | I_op={info['I_op']:.0f}/"
              f"{info['I_target_final']:.0f}A | Tmax={info['T_max']:.1f}K\n")

    env.close()
    print(f"\nResults over {n_episodes} episodes:")
    print(f"  Target reached: {results['reached_target']}")
    print(f"  Quenches:       {results['quenched']}")
    print(f"  Timeouts:       {results['timeout']}")
    return results


def check_env_api():
    """Verify both envs pass Gymnasium API checks."""
    from gymnasium.utils.env_checker import check_env
    print("\n" + "=" * 70)
    print("  Gymnasium API Compliance Check")
    print("=" * 70)

    for env_id in ["ThermalRunawayEnv-v0", "CurrentRedistEnv-v0", "REBCOFullEnv-v0"]:
        print(f"\n  Checking {env_id}...")
        env = gym.make(env_id)
        try:
            check_env(env.unwrapped, skip_render_check=True)
            print(f"  ✓ {env_id} passes Gymnasium API check")
        except Exception as e:
            print(f"  ✗ {env_id} failed: {e}")
        env.close()


if __name__ == "__main__":
    check_env_api()
    run_thermal_runaway_demo()
    run_current_redist_demo()
    run_full_env_demo()
