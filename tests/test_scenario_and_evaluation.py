"""Szenario (wortgleich aus der Hill-Climbing-Demo, eingefrorene Werte) und Auswertung (Kennzahlen, Urteil, Parallel
Tempering gegen kalibriertes Simulated Annealing und gegen Hill-Climbing-Neustarts, Sweeps, Streuung)."""

from dataclasses import replace

import numpy as np
import pytest

import pt_constants as C
import pt_evaluation as ev
import pt_scenario as S
import pt_tour as T


# --- Szenario ---------------------------------------------------------------------------------------------------------------------------------


def test_instance_shape_depot_and_area():
    inst = S.generate(60, 0, 3)
    assert inst.xy.shape == (61, 2) and inst.n == 60 and inst.n_nodes == 61
    assert inst.xy[0].tolist() == [50.0, 50.0]
    assert inst.xy.min() >= 0.0 and inst.xy.max() <= C.AREA


def test_instance_is_deterministic_seed_dependent_and_matches_the_frozen_bound():
    a, b, c = S.generate(40, 25, 5), S.generate(40, 25, 5), S.generate(40, 25, 6)
    assert np.array_equal(a.xy, b.xy) and not np.array_equal(a.xy, c.xy)
    inst, D = ev.instance(60, 0, C.DEFAULT_SEED)
    assert inst.xy[0].tolist() == [50.0, 50.0]
    assert float(inst.xy[1:].sum()) == pytest.approx(float(S.generate(60, 0, C.DEFAULT_SEED).xy[1:].sum()))


def test_grouped_stops_lie_closer_together_than_uniform_ones():
    def mean_nn(share):
        vals = []
        for seed in range(10):
            xy = S.generate(80, share, seed).xy[1:]
            d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
            np.fill_diagonal(d, np.inf)
            vals.append(d.min(axis=1).mean())
        return float(np.mean(vals))
    assert mean_nn(100) < 0.7 * mean_nn(0)


def test_unit_of_matches_bound_over_node_count():
    inst, D = ev.instance(60, 0, 35)
    bound = ev.reference_bound(60, 0, 35)
    assert ev.unit_of(60, 0, 35) == pytest.approx(bound / len(D)) == pytest.approx(bound / 61)


# --- Analyse ------------------------------------------------------------------------------------------------------------------------------------


def test_analysis_fields_are_consistent():
    a = ev.analyse(ev.Settings(budget=20000))
    assert a.bound > 0 and a.unit > 0
    assert a.gap == pytest.approx(a.gap_of(a.run.best_length))
    assert a.sa_gap >= 0.0 and a.hcr_gap >= 0.0 and a.coldest_gap >= 0.0
    assert 0.0 <= a.swap_accept_rate <= 1.0
    assert a.hcr_starts >= 1


def test_analysis_is_deterministic_and_chain_seed_matters():
    s = ev.Settings(n=30, budget=20000)
    a, b, c = ev.analyse(s), ev.analyse(s), ev.analyse(replace(s, chain_seed=1))
    assert np.array_equal(a.run.best_tour, b.run.best_tour) and a.gap == b.gap
    assert a.gap != c.gap or not np.array_equal(a.run.best_tour, c.run.best_tour)


def test_without_baselines_the_comparison_fields_are_empty():
    a = ev.analyse(ev.Settings(n=20, budget=2000), with_baselines=False)
    assert a.sa is None and a.hcr_tour is None


def test_hc_restarts_uses_at_least_one_descent_and_stays_near_the_budget():
    inst, D = ev.instance(40, 0, 100000)
    best, starts, used = ev.hc_restarts(D, 1000, 0)
    assert starts >= 1 and used >= 1000
    best2, starts2, used2 = ev.hc_restarts(D, 100000, 0)
    assert starts2 >= 2


# --- Urteil -------------------------------------------------------------------------------------------------------------------------------------


def _fake(gap, sa_gap):
    class F:
        pass
    f = F()
    f.gap, f.sa_gap = gap, sa_gap
    return f


def test_verdict_codes():
    assert ev.verdict(_fake(1.0, 1.0 + ev.WIN_MARGIN + 0.1)) == "beats_sa"
    assert ev.verdict(_fake(1.0 + ev.LOSE_MARGIN + 0.1, 1.0)) == "sa_wins"
    assert ev.verdict(_fake(1.0, 1.05)) == "comparable"


def test_verdict_of_a_real_run_at_the_default_settings():
    assert ev.verdict(ev.analyse(ev.Settings())) in ("beats_sa", "comparable", "sa_wins")


# --- Sweeps und Tabellen -----------------------------------------------------------------------------------------------------------------------


def test_run_config_counts_runs_and_aggregates():
    r = ev.run_config(ev.Settings(n=20, budget=5000))
    assert r["n_runs"] == len(C.SWEEP_SEEDS) * C.SWEEP_CHAINS
    assert r["gap_min"] <= r["gap"] <= r["gap_max"] and r["gap_sd"] >= 0 and r["seconds"] > 0


def test_run_config_ignores_the_seeds_of_the_base_settings():
    a = ev.run_config(ev.Settings(n=15, budget=3000, seed=1, chain_seed=5))
    b = ev.run_config(ev.Settings(n=15, budget=3000, seed=999, chain_seed=0))
    assert all(a[k] == b[k] for k in a if not k.endswith("seconds"))


def test_sweep_values_labels_and_ordering():
    assert set(ev.SWEEP_VALUES) == set(ev.SWEEP_LABELS)
    rows = ev.sweep("budget", ev.Settings(n=20), (2000, 20000))
    assert rows[1]["gap"] <= rows[0]["gap"] + 5.0


def test_n_replicas_sweep_values_match_constants_range():
    assert ev.SWEEP_VALUES["n_replicas"] == tuple(range(C.R_MIN, C.R_MAX + 1))


def test_scaling_table_structure(monkeypatch):
    monkeypatch.setattr(C, "SCALING_N", (10, 20))
    tab = ev.scaling_table(ev.Settings(budget=3000))
    assert len(tab) == 2
    assert all([r["value"] for r in blk["rows"]] == [10, 20] for blk in tab)
    assert tab[0]["label"] != tab[1]["label"]


def test_chain_spread_returns_one_value_per_chain_and_is_deterministic():
    a = ev.chain_spread(ev.Settings(n=15, budget=3000), 5)
    b = ev.chain_spread(ev.Settings(n=15, budget=3000), 5)
    assert len(a["pt"]) == len(a["sa"]) == 5
    assert np.array_equal(a["pt"], b["pt"]) and np.array_equal(a["sa"], b["sa"])
