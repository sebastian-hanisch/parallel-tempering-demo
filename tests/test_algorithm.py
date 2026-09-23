"""Parallel Tempering: Tausch-Formel (Grenzfälle, Handrechnung), Regressionen gegen die vertraute `_chain_segment`-
Bausteinlogik (Einzel-Aufruf-Fälle, Ketten-Unabhängigkeit bei ausgeschaltetem Tausch - siehe die Chunking-Anmerkung
im Modul-Docstring von pt_algorithm.py für die Grenzen dieser Vergleiche), die zentrale, PT-eigene Korrektheits-
eigenschaft (der Tausch erhält die GEMEINSAME Boltzmann-Gleichgewichtsverteilung - jede Kette bleibt bei ihrer
EIGENEN Temperatur im Gleichgewicht, auch während sie mit anderen Ketten tauscht), Budget-Buchführung, Monotonie
des besten Ergebnisses, gültige Permutationen."""

import itertools
import math

import numpy as np
import pytest

import pt_algorithm as PT
import pt_tour as T


def _instance(n_nodes, seed):
    rng = np.random.default_rng(seed)
    xy = rng.random((n_nodes, 2)) * 100
    return xy, T.dist_matrix(xy)


# --- Tausch-Formel ---------------------------------------------------------------------------------------------------------------------------


def test_swap_probability_is_always_one_at_equal_temperature():
    for l_i, l_j in [(50.0, 80.0), (80.0, 50.0), (10.0, 10.0), (0.0, 1000.0)]:
        assert PT.swap_probability(0.3, 0.3, l_i, l_j) == 1.0


def test_swap_probability_matches_hand_computed_values():
    # x = (1/T_i - 1/T_j) * (L_i - L_j)
    assert PT.swap_probability(0.1, 0.2, 50.0, 80.0) == pytest.approx(math.exp((1 / 0.1 - 1 / 0.2) * (50 - 80)))
    assert PT.swap_probability(0.1, 0.2, 80.0, 50.0) == 1.0                  # kältere Kette hat die schlechtere Tour -> immer tauschen
    assert PT.swap_probability(0.2, 0.1, 50.0, 80.0) == 1.0                  # (symmetrischer Fall, T_i/T_j vertauscht)


def test_swap_probability_never_exceeds_one_or_underflows_below_zero():
    rng = np.random.default_rng(0)
    for _ in range(200):
        t_i, t_j = rng.uniform(0.01, 5.0, 2)
        l_i, l_j = rng.uniform(0, 2000, 2)
        p = PT.swap_probability(t_i, t_j, l_i, l_j)
        assert 0.0 <= p <= 1.0


# --- Regression gegen _chain_segment (nur bei identischem Aufruf-Zuschnitt, siehe Modul-Docstring) -------------------------------------------


def test_single_replica_single_round_matches_one_direct_chain_segment_call():
    """swap_interval >= budget => genau EIN `_chain_segment`-Aufruf je Kette, sowohl innerhalb von
    `parallel_tempering` als auch im direkten Vergleich - hier ist ein bytegleicher Vergleich sinnvoll (siehe
    Docstring-Hinweis in pt_algorithm.py)."""
    xy, D = _instance(60, 0)
    start = T.random_tour(60, np.random.default_rng(1))
    budget = 20000
    r = PT.parallel_tempering(D, start, [0.15], budget=budget, swap_interval=budget, seed=7)
    direct_t, direct_len, direct_ev, _, direct_best_t, direct_best_len = PT._chain_segment(start, T.tour_length(start, D), D, 0.15, budget, np.random.default_rng(7))
    assert np.array_equal(r.best_tour, direct_best_t) and r.best_length == pytest.approx(direct_best_len)
    assert np.array_equal(r.coldest_tour, direct_t) and r.evaluations == direct_ev == budget


def test_independent_replicas_single_round_each_match_their_own_direct_chain_segment_call():
    xy, D = _instance(60, 0)
    start = T.random_tour(60, np.random.default_rng(1))
    temps = [0.1, 0.3, 0.6]
    budget = 9000
    r = PT.parallel_tempering(D, start, temps, budget=budget, swap_interval=budget, seed=3, swaps_enabled=False)
    base = budget // len(temps)
    per_replica = [base, base, base + (budget - base * len(temps))]
    for i, t_i in enumerate(temps):
        _, direct_len, direct_ev, _, _, _ = PT._chain_segment(start, T.tour_length(start, D), D, t_i, per_replica[i], np.random.default_rng(3 + i))
        assert r.replica_lengths[-1][i] == pytest.approx(direct_len)
    assert r.evaluations == budget


def test_swaps_disabled_replicas_evolve_independently_of_how_many_other_replicas_exist():
    """Die eigentliche 'R unabhängige Ketten'-Eigenschaft: Replikat 0 muss bei IDENTISCHEM eigenen Budget/Tausch-
    Intervall/Seed dasselbe Ergebnis liefern, egal ob es ALLEIN läuft (R=1) oder NEBEN anderen Ketten (R=3) -
    unabhängig vom internen Chunking, da beide Fälle denselben Runden-Zuschnitt für Replikat 0 durchlaufen
    (`base = budget // R` ist in beiden Konfigurationen gleich, wenn das GESAMTBUDGET entsprechend skaliert wird)."""
    xy, D = _instance(60, 0)
    start = T.random_tour(60, np.random.default_rng(1))
    per_replica_budget = 3000
    r_alone = PT.parallel_tempering(D, start, [0.2], budget=per_replica_budget, swap_interval=100, seed=5, swaps_enabled=False)
    r_with_others = PT.parallel_tempering(D, start, [0.2, 0.5, 0.9], budget=per_replica_budget * 3, swap_interval=100, seed=5, swaps_enabled=False)
    assert r_alone.replica_lengths[-1][0] == pytest.approx(r_with_others.replica_lengths[-1][0])


def test_swaps_disabled_two_configs_with_different_swap_interval_still_agree_when_rounds_match():
    """Kontrollprobe zur vorigen Eigenschaft mit einer zweiten, unabhängig gewählten Rundenzahl."""
    xy, D = _instance(30, 2)
    start = T.random_tour(30, np.random.default_rng(4))
    per_replica_budget = 4000
    r_alone = PT.parallel_tempering(D, start, [0.4], budget=per_replica_budget, swap_interval=250, seed=9, swaps_enabled=False)
    r_with_others = PT.parallel_tempering(D, start, [0.4, 1.1], budget=per_replica_budget * 2, swap_interval=250, seed=9, swaps_enabled=False)
    assert r_alone.replica_lengths[-1][0] == pytest.approx(r_with_others.replica_lengths[-1][0])


# --- Zentrale Korrektheitseigenschaft: der Tausch erhält die Boltzmann-Gleichgewichtsverteilung JEDER Kette ------------------------------------


def test_swaps_preserve_each_replicas_own_boltzmann_equilibrium():
    """Kern-Eigenschaft von Parallel Tempering: obwohl die beiden Ketten ständig Zustände tauschen, bleibt die
    STATIONÄRE Verteilung JEDER Kette exakt ihre eigene Boltzmann-Verteilung bei ihrer eigenen Temperatur (die
    Tausch-Formel ist genau dafür konstruiert). Geprüft auf einer vollständig aufzählbaren 6-Knoten-Instanz wie im
    Boltzmann-Test der Simulated-Annealing-Demo, hier über ZWEI gekoppelte Ketten statt einer einzelnen."""
    n = 6
    xy, D = _instance(n, 11)
    mean_edge = float(np.mean(D[D > 0]))
    temps = [0.25 * mean_edge, 0.6 * mean_edge]
    tours = {}
    for perm in itertools.permutations(range(1, n)):
        c = tuple(T.canonical(np.array([0, *perm])).tolist())
        tours[c] = T.tour_length(c, D)
    assert len(tours) == 60
    l_min = min(tours.values())

    def boltzmann(temp):
        w = {c: math.exp(-(L - l_min) / temp) for c, L in tours.items()}
        z = sum(w.values())
        return {c: v / z for c, v in w.items()}

    p = [boltzmann(temps[0]), boltzmann(temps[1])]
    r = PT.parallel_tempering(D, np.arange(n), temps, budget=960000, swap_interval=12, seed=3, keep_snapshots=True)
    assert r.swap_accepts[0] > 0 and r.swap_attempts[0] > 0                  # der Tausch wird tatsächlich genutzt, nicht nur angeboten

    for replica in (0, 1):
        counts = {}
        for snap in r.snapshots[2000:]:
            c = tuple(T.canonical(snap[replica]).tolist())
            counts[c] = counts.get(c, 0) + 1
        total = sum(counts.values())
        tv = 0.5 * sum(abs(counts.get(c, 0) / total - p[replica][c]) for c in p[replica])
        assert tv < 0.06, (replica, tv)                                     # gemessen ~0.01-0.02; Stichprobenfehler bei ~38 000 Zuständen etwa 0.02-0.03


def test_disabling_swaps_still_leaves_each_chain_at_its_own_boltzmann_equilibrium():
    """Kontrollprobe: die Marginalverteilung einer einzelnen Metropolis-Kette bei fester Temperatur ist unabhängig
    davon, ob Tausch aktiv ist (Tausch ändert nur die MISCHUNG, nicht die Gleichgewichtsverteilung selbst) -
    dieselbe Prüfung wie oben, aber mit `swaps_enabled=False` auf einer einzelnen Kette."""
    n = 6
    xy, D = _instance(n, 11)
    mean_edge = float(np.mean(D[D > 0]))
    temp = 0.35 * mean_edge
    tours = {}
    for perm in itertools.permutations(range(1, n)):
        c = tuple(T.canonical(np.array([0, *perm])).tolist())
        tours[c] = T.tour_length(c, D)
    l_min = min(tours.values())
    w = {c: math.exp(-(L - l_min) / temp) for c, L in tours.items()}
    z = sum(w.values())
    p = {c: v / z for c, v in w.items()}

    r = PT.parallel_tempering(D, np.arange(n), [temp], budget=480000, swap_interval=12, seed=5, keep_snapshots=True, swaps_enabled=False)
    counts = {}
    for snap in r.snapshots[2000:]:
        c = tuple(T.canonical(snap[0]).tolist())
        counts[c] = counts.get(c, 0) + 1
    total = sum(counts.values())
    tv = 0.5 * sum(abs(counts.get(c, 0) / total - p[c]) for c in p)
    assert tv < 0.06, tv


# --- Buchführung, Monotonie, gültige Permutationen --------------------------------------------------------------------------------------------


def test_budget_accounting_sums_exactly_across_replicas():
    xy, D = _instance(40, 6)
    start = T.random_tour(40, np.random.default_rng(2))
    for budget in (997, 5000, 50000):
        for r_count in (1, 2, 5):
            temps = np.geomspace(0.1, 1.0, r_count).tolist()
            r = PT.parallel_tempering(D, start, temps, budget=budget, swap_interval=37, seed=1, keep_snapshots=False)
            assert r.evaluations == budget


def test_best_length_is_monotone_non_increasing_across_rounds():
    xy, D = _instance(40, 6)
    start = T.random_tour(40, np.random.default_rng(2))
    r = PT.parallel_tempering(D, start, [0.1, 0.3, 0.8], budget=40000, swap_interval=200, seed=1, keep_snapshots=True)
    assert (np.diff(r.replica_best) <= 1e-9).all()
    assert r.replica_best[-1] == pytest.approx(r.best_length)


def test_temperature_ladder_is_geometric_and_ascending():
    ladder = PT.temperature_ladder(0.05, 1.0, 5)
    assert len(ladder) == 5 and ladder[0] == pytest.approx(0.05) and ladder[-1] == pytest.approx(1.0)
    assert (np.diff(ladder) > 0).all()
    ratios = ladder[1:] / ladder[:-1]
    assert np.ptp(ratios) < 1e-9                                            # geometrisch: konstantes Verhältnis zwischen benachbarten Stufen


def test_temperature_ladder_single_replica_is_just_t_min():
    assert PT.temperature_ladder(0.3, 0.3, 1).tolist() == [0.3]
    assert PT.temperature_ladder(0.1, 0.9, 1).tolist() == [0.1]


def test_sa_baseline_budget_accounting_and_monotone_best():
    xy, D = _instance(40, 6)
    start = T.random_tour(40, np.random.default_rng(2))
    r = PT.sa_baseline(D, start, 0.5, 0.1, budget=20000, levels=50, seed=1)
    assert r.evaluations == 20000
    assert r.best_length <= T.tour_length(start, D) + 1e-6
    assert sorted(r.best_tour.tolist()) == list(range(40)) and sorted(r.final_tour.tolist()) == list(range(40))


def test_sa_baseline_single_level_matches_one_direct_chain_segment_call():
    xy, D = _instance(40, 6)
    start = T.random_tour(40, np.random.default_rng(2))
    r = PT.sa_baseline(D, start, 0.3, 0.3, budget=15000, levels=1, seed=4)
    direct_t, direct_len, direct_ev, _, direct_best_t, direct_best_len = PT._chain_segment(start, T.tour_length(start, D), D, 0.3, 15000, np.random.default_rng(4))
    assert np.array_equal(r.best_tour, direct_best_t) and r.best_length == pytest.approx(direct_best_len) and r.evaluations == direct_ev


def test_all_replica_tours_stay_valid_permutations():
    xy, D = _instance(25, 3)
    start = T.random_tour(25, np.random.default_rng(9))
    r = PT.parallel_tempering(D, start, [0.05, 0.2, 0.5, 1.0], budget=30000, swap_interval=150, seed=2, keep_snapshots=True)
    for snap in r.snapshots:
        for tour in snap:
            assert sorted(tour.tolist()) == list(range(25)) and tour[0] == 0
    assert sorted(r.best_tour.tolist()) == list(range(25))
    assert sorted(r.coldest_tour.tolist()) == list(range(25))


def test_swap_attempts_and_accepts_never_exceed_each_other_and_use_checkerboard_pairs():
    xy, D = _instance(30, 4)
    start = T.random_tour(30, np.random.default_rng(1))
    r = PT.parallel_tempering(D, start, [0.1, 0.2, 0.3, 0.4, 0.5], budget=50000, swap_interval=100, seed=1)
    assert (r.swap_accepts <= r.swap_attempts).all()
    assert r.swap_attempts.sum() > 0
    # Schachbrettschema: Paar i wird nur in geraden (i gerade) oder ungeraden (i ungerade) Runden versucht,
    # also hoechstens ceil(rounds/2) mal - eine grobe, aber robuste obere Schranke je Paar.
    assert (r.swap_attempts <= -(-r.rounds // 2)).all()


def test_single_replica_never_attempts_a_swap():
    xy, D = _instance(20, 5)
    start = T.random_tour(20, np.random.default_rng(1))
    r = PT.parallel_tempering(D, start, [0.3], budget=10000, swap_interval=100, seed=1)
    assert len(r.swap_attempts) == 0 and len(r.swap_accepts) == 0


