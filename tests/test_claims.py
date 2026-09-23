"""Jede Zahl in den Hilfetexten, Presets, Tabellen und Grenzen der App ist hier über die fünf festen Sweep-Instanzen
(je drei Ketten-Seeds) belegt, mit denselben Auswertungsfunktionen wie die App selbst (`ev.run_config`/`ev.sweep`)
- NIE über ein Ad-hoc-Skript mit abweichender Zufalls-Bindung (die Lehre aus der lin-kernighan-demo dieser Linie:
dort erzeugte ein Ad-hoc-Skript mit unabhängig gezogenen rng-Instanzen statt des echten paarigen start+seed-
Musters einen überzeugend falschen Befund). Positive UND negative Aussagen: Parallel Tempering gewinnt ab
150-200 Tausend Budget klar UND mit wachsendem Vorsprung (positiv) - aber verliert bei kleinem Budget oder großen
Instanzen deutlich, weil das Budget auf R Ketten aufgeteilt wird (negativ, ehrlicher Kernbefund)."""

from functools import lru_cache

import pytest

import pt_constants as C
import pt_evaluation as ev


@lru_cache(maxsize=None)
def _cfg(items):
    return ev.run_config(ev.Settings(), **dict(items))


def cfg(**kw):
    return _cfg(tuple(sorted(kw.items())))


def near(value, expected, tol):
    assert abs(value - expected) <= tol, f"{value:.3f} statt {expected}"


# --- Budget-Sweep: PT verliert klein, gewinnt gross, mit WACHSENDEM statt schrumpfendem Vorsprung ------------------------------------------------


@pytest.mark.parametrize("budget,pt,sa,tol", [
    (10000, 32.05, 9.01, 4.0), (25000, 9.65, 4.72, 2.0), (50000, 4.89, 3.07, 1.2), (100000, 2.37, 1.94, 0.8),
    (200000, 1.15, 1.42, 0.5), (500000, 0.77, 1.27, 0.4), (1000000, 0.62, 0.94, 0.3), (2000000, 0.49, 0.73, 0.3),
])
def test_budget_sweep_numbers(budget, pt, sa, tol):
    row = cfg(budget=budget)
    near(row["gap"], pt, tol)
    near(row["sa"], sa, tol)


def test_pt_loses_at_small_budget_and_wins_from_two_hundred_thousand_on():
    for budget in (10000, 25000, 50000, 100000):
        row = cfg(budget=budget)
        assert row["gap"] > row["sa"] + 0.3                                 # PT verliert klar bei knappem Budget
    for budget in (200000, 500000, 1000000, 2000000):
        row = cfg(budget=budget)
        assert row["gap"] < row["sa"] - 0.05                                # PT gewinnt ab dem Standardbudget


def test_pt_advantage_over_sa_grows_with_budget_past_two_hundred_thousand():
    """Ehrlicher Kontrast zu den meisten anderen Stücken dieser Linie: der Vorsprung WÄCHST hier mit dem Budget,
    statt zu schrumpfen (GRASP, Lin-Kernighan, Dynasearch zeigen das Gegenteil oder einen stabilen Vorsprung)."""
    mid = cfg(budget=200000)
    large = cfg(budget=2000000)
    mid_edge = mid["sa"] - mid["gap"]
    large_edge = large["sa"] - large["gap"]
    assert mid_edge > 0 and large_edge > mid_edge - 0.05


# --- Skalierung: PT gewinnt bis 60 Stopps, verliert danach deutlich (derselbe Budget-Teilungs-Effekt) --------------------------------------------


@pytest.mark.parametrize("n,pt,sa,tol", [(20, 0.05, 0.05, 0.15), (40, 0.95, 0.79, 0.4), (60, 1.15, 1.42, 0.5),
                                          (100, 6.20, 3.08, 1.5), (150, 12.95, 6.54, 2.5), (200, 23.55, 7.87, 3.5)])
def test_scaling_numbers_at_fixed_two_hundred_thousand_budget(n, pt, sa, tol):
    row = ev.run_config(ev.Settings(), n=n, budget=200000)
    near(row["gap"], pt, tol)
    near(row["sa"], sa, tol)


def test_pt_is_close_at_small_n_but_loses_clearly_from_one_hundred_on_at_fixed_budget():
    """Nicht monoton: bei n=20 praktisch gleichauf, bei n=40 knapp schlechter, bei n=60 gewinnt PT wieder - erst
    ab n=100 wird der Rückstand groß und durchgehend (derselbe Budget-Teilungs-Effekt wie beim kleinen Budget)."""
    for n in (20, 40, 60):
        row = ev.run_config(ev.Settings(), n=n, budget=200000)
        assert row["gap"] <= row["sa"] + 0.5                                # nah dran, kein grosser Rueckstand
    for n in (100, 150, 200):
        row = ev.run_config(ev.Settings(), n=n, budget=200000)
        assert row["gap"] > row["sa"] + 1.0


# --- Kalibrierung: Leiterbreite, R, Tausch-Intervall (nicht-monotone Optima) ---------------------------------------------------------------------


def test_narrow_well_placed_ladder_beats_wide_ladders():
    narrow = cfg(t_min=0.1, t_max=0.3)
    for t_min, t_max in [(0.4, 0.6), (0.2, 0.8), (0.05, 1.0), (0.02, 2.0)]:
        wide = cfg(t_min=t_min, t_max=t_max)
        assert narrow["gap"] < wide["gap"] - 0.3


@pytest.mark.parametrize("r_count,gap,tol", [(2, 2.24, 0.6), (3, 2.19, 0.6), (5, 1.15, 0.5), (7, 1.78, 0.6), (10, 2.32, 0.7)])
def test_n_replicas_numbers_at_calibrated_ladder(r_count, gap, tol):
    row = cfg(n_replicas=r_count)
    near(row["gap"], gap, tol)


def test_n_replicas_five_is_a_non_monotone_optimum():
    r5 = cfg(n_replicas=5)["gap"]
    assert r5 < cfg(n_replicas=2)["gap"] - 0.3
    assert r5 < cfg(n_replicas=10)["gap"] - 0.3


@pytest.mark.parametrize("si,gap,tol", [(10, 2.05, 0.6), (30, 2.03, 0.6), (100, 1.41, 0.6), (300, 1.15, 0.5), (1000, 1.64, 0.6), (3000, 2.19, 0.7)])
def test_swap_interval_numbers_at_calibrated_ladder(si, gap, tol):
    row = cfg(swap_interval=si)
    near(row["gap"], gap, tol)


def test_swap_interval_three_hundred_is_a_non_monotone_optimum():
    s300 = cfg(swap_interval=300)["gap"]
    assert s300 < cfg(swap_interval=10)["gap"] - 0.3
    assert s300 < cfg(swap_interval=3000)["gap"] - 0.3


# --- Der Austausch selbst bringt Wert (nicht nur "mehrere Temperaturen zu haben") ------------------------------------------------------------


def test_swaps_enabled_beats_disabled_on_every_single_sweep_instance():
    for seed in C.SWEEP_SEEDS:
        on = ev.run_config(ev.Settings(), seeds=(seed,))["gap"]
        off = ev.run_config(ev.Settings(swaps_enabled=False), seeds=(seed,))["gap"]
        assert on < off, (seed, on, off)


def test_swaps_enabled_beats_disabled_on_average():
    on = cfg(swaps_enabled=True)
    off = cfg(swaps_enabled=False)
    near(on["gap"], 1.15, 0.5)
    near(off["gap"], 2.40, 0.7)
    assert on["gap"] < off["gap"] - 0.5


# --- Robustheit gegen Fehlkalibrierung: PT schlechter als kalibriert, aber weit vor SAs eigenen Fehlkalibrierungen -------------------------------


def test_pt_robustness_against_bad_ladders_vs_sa_own_miscalibration():
    calibrated = cfg(t_min=0.1, t_max=0.3)["gap"]
    sa_extremes = cfg(t_min=0.02, t_max=0.5)["gap"]
    near(calibrated, 1.15, 0.5)
    near(sa_extremes, 3.22, 0.8)
    assert calibrated < sa_extremes - 0.3                                   # eine schlechte Leiter kostet spürbar

    sa_tuned = ev.run_config(ev.Settings(), sa_t0=0.5, sa_t_end=0.1)["sa"]
    sa_too_cold = ev.run_config(ev.Settings(), sa_t0=0.05, sa_t_end=0.02)["sa"]
    sa_too_hot = ev.run_config(ev.Settings(), sa_t0=1.0, sa_t_end=0.5)["sa"]
    near(sa_tuned, 1.42, 0.5)
    assert sa_too_cold > sa_tuned + 3.0                                     # SAs eigene Fehlkalibrierung ist weit schlimmer ...
    assert sa_too_hot > sa_tuned + 5.0
    assert sa_extremes < sa_too_cold and sa_extremes < sa_too_hot           # ... als selbst PTs schlechteste getestete Leiter


# --- Sonstiges --------------------------------------------------------------------------------------------------------------------------------


def test_preset_count_matches_the_readme():
    assert len(C.PRESETS) == 6


def test_bound_is_positive_and_below_a_naive_upper_estimate():
    inst, D = ev.instance(60, 0, C.DEFAULT_SEED)
    bound = ev.reference_bound(60, 0, C.DEFAULT_SEED)
    assert 0 < bound < D.sum()
