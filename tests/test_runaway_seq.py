"""
test_runaway_seq.py -- time-to-runaway / runaway-probability models
(runaway_seq.py) on a small synthetic sweep shaped like load_runaway_sweep().
"""

import numpy as np
import pandas as pd
import pytest
import torch

from hts_tape_env.runaway_seq import (
    ARCHS,
    TTR_OFFSET_S,
    RunawayEpisodeDataset,
    add_model_features,
    collate_runaway,
    evaluate_runaway_model,
    load_runaway_model,
    predict_episodes,
    runaway_loss,
    split_by_ratio_blocks,
    train_runaway_model,
)

DT = 0.01
H = 0.2


def _synthetic_sweep(ratios=np.round(np.arange(0.90, 1.40, 0.005), 3)) -> pd.DataFrame:
    frames = []
    for episode_id, ratio in enumerate(ratios):
        runaway = ratio >= 1.2
        t_end = 1.0 - 0.5 * (ratio - 1.2) if runaway else 1.0
        t = DT * np.arange(int(round(t_end / DT)) + 1)
        heat = np.exp(np.clip(ratio - 1.0, 0, None) * 4 * t)
        frames.append(pd.DataFrame(dict(
            episode_id=episode_id, ratio=ratio, step=np.arange(len(t)), t_s=t,
            I_A=311.0 * min(ratio, 1.0 + 20.0 * t.max() / 311.0),
            voltage_V=1e-6 * heat ** 3, T_surface_K=77.0 + (heat - 1.0),
            runaway=int(runaway), t_end_s=t_end,
            time_to_runaway_s=(t_end - t) if runaway else np.nan,
        )))
    return add_model_features(pd.concat(frames, ignore_index=True))


@pytest.fixture(scope="module")
def df():
    return _synthetic_sweep()


@pytest.fixture(scope="module")
def ds(df):
    return RunawayEpisodeDataset(df, horizon_s=H)


def test_runaway_episode_labels(ds):
    e = next(e for e in ds.episodes if not e["censored"])
    assert np.array_equal(e["y"], (e["ttr_s"] <= H).astype(np.float32))
    assert e["y_valid"].all()
    assert np.allclose(e["ttr_log"], np.log(e["ttr_s"] + TTR_OFFSET_S))


def test_censored_episode_labels_masked_near_window_end(ds):
    e = next(e for e in ds.episodes if e["censored"])
    assert not e["y"].any()
    assert np.allclose(e["ttr_s"], 1.0 - e["t_s"], atol=1e-6)
    away_from_edge = np.abs(e["ttr_s"] - H) > 1e-6
    assert np.array_equal((e["y_valid"] > 0)[away_from_edge], (e["ttr_s"] >= H)[away_from_edge])
    assert not e["y_valid"][-1]


def test_censored_loss_only_penalizes_predictions_below_bound(ds):
    batch = collate_runaway([e for e in ds.episodes if e["censored"]][:2])
    no_prob = dict(runaway_logit=torch.full_like(batch["y"], -1e4))
    above = runaway_loss({**no_prob, "ttr_log": batch["ttr_log"] + 1.0}, batch)
    below = runaway_loss({**no_prob, "ttr_log": batch["ttr_log"] - 1.0}, batch)
    assert above.item() == pytest.approx(0.0, abs=1e-6)
    assert below.item() > 0.5


def test_observed_loss_penalizes_both_directions(ds):
    batch = collate_runaway([e for e in ds.episodes if not e["censored"]][:2])
    no_prob = dict(runaway_logit=torch.where(batch["y"] > 0, 1e4, -1e4))
    for shift in (1.0, -1.0):
        loss = runaway_loss({**no_prob, "ttr_log": batch["ttr_log"] + shift}, batch)
        assert loss.item() > 0.5


def test_ratio_block_split_is_disjoint_and_contiguous(df):
    splits = split_by_ratio_blocks(df, block_width=0.02)
    ids = [set(s["episode_id"]) for s in splits.values()]
    assert set.union(*ids) == set(df["episode_id"])
    assert sum(len(s) for s in ids) == df["episode_id"].nunique()
    test_ratios = np.sort(splits["test"]["ratio"].unique())
    train_ratios = splits["train"]["ratio"].unique()
    # nearest training run to any test run is at least one 0.005 grid step away
    assert all(np.min(np.abs(train_ratios - r)) >= 0.005 - 1e-9 for r in test_ratios)


def test_false_alarm_rate_ignores_censored_runs_shorter_than_horizon(df):
    splits = split_by_ratio_blocks(df, block_width=0.02)
    result = train_runaway_model(splits["train"], splits["val"], arch="gru", horizon_s=H,
                                 epochs=1, patience=0, verbose=False, model_kwargs={"hidden": 8})
    m = evaluate_runaway_model(result, splits["test"])["metrics"]
    assert m["n_censored_evaluable"] == m["n_censored_episodes"]

    # Synthetic censored windows are 1.0 s, so a 1.5 s horizon leaves no known-safe step.
    result.meta["horizon_s"] = 1.5
    m = evaluate_runaway_model(result, splits["test"])["metrics"]
    assert m["n_censored_evaluable"] == 0
    assert np.isnan(m["false_alarm_rate"])


@pytest.mark.parametrize("arch", ARCHS)
def test_train_evaluate_save_load(df, tmp_path, arch):
    splits = split_by_ratio_blocks(df, block_width=0.02)
    result = train_runaway_model(splits["train"], splits["val"], arch=arch, horizon_s=H,
                                 epochs=2, patience=0, verbose=False,
                                 model_kwargs={"hidden": 16} if arch != "transformer" else {"d_model": 16})
    report = evaluate_runaway_model(result, splits["test"])
    m = report["metrics"]
    assert 0.0 <= m["prob_auroc"] <= 1.0
    assert 0.0 <= m["false_alarm_rate"] <= 1.0
    assert len(report["per_episode"]) == splits["test"]["episode_id"].nunique()

    loaded = load_runaway_model(result.save(tmp_path / f"{arch}.pt"))
    a = predict_episodes(result, splits["test"])
    b = predict_episodes(loaded, splits["test"])
    assert all(np.allclose(x["prob"], y["prob"]) and np.allclose(x["ttr_pred_s"], y["ttr_pred_s"])
               for x, y in zip(a, b))
