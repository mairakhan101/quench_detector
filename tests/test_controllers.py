"""
test_controllers.py -- rule-based controllers (controllers.py).
"""

from hts_tape_env.controllers import evaluate, oracle_controller
from hts_tape_env.env import QuenchDetectorEnv


def test_oracle_beats_always_continue():
    """Oracle (perfect knowledge of I*_true, dumps in the danger band)
    should never miss a quench, unlike an always-continue baseline which
    always does."""
    env = QuenchDetectorEnv()
    oracle = evaluate(env, oracle_controller(env.dump_threshold), n_episodes=100, seed0=0)
    assert oracle["miss_rate"] == 0.0
    assert oracle["hit_rate"] > 0.8


def test_evaluate_result_keys():
    env = QuenchDetectorEnv()
    result = evaluate(env, oracle_controller(env.dump_threshold), n_episodes=10, seed0=0)
    for key in ("controller", "mean_reward", "hit_rate", "miss_rate",
                "false_alarm_rate", "mean_final_margin"):
        assert key in result
