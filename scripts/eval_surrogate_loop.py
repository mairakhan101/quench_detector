"""
eval_surrogate_loop.py

The practical payoff of wiring env.set_surrogate(): compare the oracle
controller (dumps on the environment's ground-truth margin) against the
surrogate controller (dumps on the GRU sequence surrogate's margin_pred, from
telemetry alone) over many episodes of QuenchDetectorEnv.

Usage
-----
    python scripts/eval_surrogate_loop.py --episodes 300 --checkpoint models/seq_gru.pt
"""

import _bootstrap  # noqa: F401

import argparse

from hts_tape_env.controllers import (
    evaluate,
    oracle_controller,
    surrogate_controller,
    surrogate_prob_controller,
)
from hts_tape_env.env import QuenchDetectorEnv
from hts_tape_env.seq_surrogates import SequenceSurrogateAdapter, load_seq_surrogate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--dump-threshold", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--checkpoint", default="models/seq_gru.pt")
    args = ap.parse_args()

    result = load_seq_surrogate(args.checkpoint)

    env_oracle = QuenchDetectorEnv(dump_threshold=args.dump_threshold)
    oracle = evaluate(env_oracle, oracle_controller(args.dump_threshold),
                       n_episodes=args.episodes, seed0=args.seed)

    env_surr = QuenchDetectorEnv(dump_threshold=args.dump_threshold)
    env_surr.set_surrogate(SequenceSurrogateAdapter(result))
    surrogate = evaluate(env_surr, surrogate_controller(args.dump_threshold),
                          n_episodes=args.episodes, seed0=args.seed)

    env_prob = QuenchDetectorEnv(dump_threshold=args.dump_threshold)
    env_prob.set_surrogate(SequenceSurrogateAdapter(result))
    surrogate_prob = evaluate(env_prob, surrogate_prob_controller(0.5),
                               n_episodes=args.episodes, seed0=args.seed)

    print(f"{'metric':<20}{'oracle (true margin)':<24}{'surrogate (margin_pred)':<26}{'surrogate (breach_prob)'}")
    for k in ("mean_reward", "hit_rate", "miss_rate", "false_alarm_rate", "mean_final_margin"):
        print(f"{k:<20}{oracle[k]:<24.4f}{surrogate[k]:<26.4f}{surrogate_prob[k]:.4f}")


if __name__ == "__main__":
    main()
