"""Unabhängiges Orakel für Parallel Tempering: (1) Tausch-Wahrscheinlichkeit min(1, exp((1/T_i - 1/T_j)(L_i - L_j))) von Hand und Detailed Balance per Aufzählung aller
Zustandspaare eines Mini-TSP (24 Touren); (2) ein Metropolis-Schritt mit gefälschtem Zufallsstrom (ungültiges Paar überspringen, Annahmegrenze, Tourumkehr, Länge);
(3) die Metropolis-Segmente sind im Gleichgewicht Boltzmann-verteilt (Mini-Zustandsraum, Chi-Quadrat); (4) `parallel_tempering` mit festem Zufallsstrom nachsimuliert
(Budgetteilung, Rest an die heißeste Kette, Schachbrett-Paare, Verlauf, Zähler)."""

import itertools
import math

import numpy as np
import pytest

import pt_algorithm as PT
import pt_tour as T


def _states():
    return [[0, *p] for p in itertools.permutations(range(1, 5))]


def test_swap_probability_by_hand_and_detailed_balance_over_all_state_pairs():
    assert PT.swap_probability(1, 2, 10, 4) == 1.0                                  # (1 - 1/2) * (10 - 4) = 3 > 0: die kalte Kette hat die längere Tour
    assert PT.swap_probability(1, 2, 4, 10) == pytest.approx(math.exp(-3))
    assert PT.swap_probability(5.0, 5.0, 7.0, 1.0) == 1.0                           # gleiche Temperatur: immer tauschen
    rng = np.random.default_rng(3)
    D = T.dist_matrix(rng.random((5, 2)) * 100)
    states = _states()
    lengths = [T.tour_length(s, D) for s in states]
    t_i, t_j = 6.0, 25.0
    for x, lx in enumerate(lengths):
        for y, ly in enumerate(lengths):
            pi_xy, pi_yx = math.exp(-lx / t_i - ly / t_j), math.exp(-ly / t_i - lx / t_j)
            a_xy, a_yx = PT.swap_probability(t_i, t_j, lx, ly), PT.swap_probability(t_i, t_j, ly, lx)
            assert a_xy == pytest.approx(min(1.0, pi_yx / pi_xy), rel=1e-12)
            assert pi_xy * a_xy == pytest.approx(pi_yx * a_yx, rel=1e-12)


class _FakeRng:
    """Liefert vorgegebene Paare und eine Zufallszahl: das erste Paar ist ungültig und muss übersprungen werden."""

    def __init__(self, a, b, u):
        self.a, self.b, self.u, self.calls = a, b, u, 0

    def integers(self, lo, hi, size):
        self.calls += 1
        src = self.a if self.calls == 1 else self.b
        return np.array((src + [src[-1]] * size)[:size])

    def random(self, size):
        return np.array((self.u + [self.u[-1]] * size)[:size])


def test_one_metropolis_step_skips_invalid_pairs_and_accepts_exactly_below_the_boltzmann_threshold():
    rng = np.random.default_rng(5)
    for it in range(60):
        n = int(rng.integers(5, 9))
        D = T.dist_matrix(rng.random((n, 2)) * 100)
        t = T.random_tour(n, rng)
        temp = float(rng.choice([0.5, 4.0, 30.0]))
        i, j = sorted(rng.choice(n, 2, replace=False).tolist())
        while j < i + 2 or (i == 0 and j == n - 1):
            i, j = sorted(rng.choice(n, 2, replace=False).tolist())
        delta = D[t[i], t[j]] + D[t[i + 1], t[(j + 1) % n]] - D[t[i], t[i + 1]] - D[t[j], t[(j + 1) % n]]
        p = 1.0 if delta <= 0 else math.exp(-delta / temp)
        for u in (p * 0.999, min(0.99999, p * 1.001 + 1e-6)):
            out, length, done, acc, best_t, best_len = PT._chain_segment(t, T.tour_length(t, D), D, temp, 1, _FakeRng([3, i], [3, j], [0.0, u]))
            accept = delta <= 0 or u < p
            expect = t.copy()
            if accept:
                expect[i + 1:j + 1] = t[i + 1:j + 1][::-1]
            assert done == 1 and acc == int(accept) and np.array_equal(out, expect)
            assert length == pytest.approx(T.tour_length(expect, D), abs=1e-9)
            assert best_len == pytest.approx(min(T.tour_length(t, D), T.tour_length(expect, D)), abs=1e-9)


def test_metropolis_segments_reach_the_boltzmann_distribution_on_a_mini_state_space():
    scipy_stats = pytest.importorskip("scipy.stats")
    D = T.dist_matrix(np.random.default_rng(11).random((5, 2)) * 100)
    states = _states()
    index = {tuple(s): k for k, s in enumerate(states)}
    lengths = np.array([T.tour_length(s, D) for s in states])
    temp = float(np.median(lengths) / 6)
    w = np.exp(-(lengths - lengths.min()) / temp)
    w /= w.sum()
    n_runs = 2500
    counts = np.zeros(len(states))
    for k in range(n_runs):
        s0 = states[k % len(states)]
        out = PT._chain_segment(np.array(s0), T.tour_length(s0, D), D, temp, 100, np.random.default_rng(k))[0]
        counts[index[tuple(out.tolist())]] += 1
    assert scipy_stats.chisquare(counts, w * n_runs).pvalue > 1e-3


def _reference(D, start, temps, budget, interval, seed, swaps):
    R = len(temps)
    rngs = [np.random.default_rng(seed + r) for r in range(R)]
    swap_rng = np.random.default_rng((seed, R)) if swaps else None
    tours = [np.array(start, dtype=np.int64) for _ in range(R)]
    lens = [T.tour_length(start, D)] * R
    base = budget // R
    bud = [base] * R
    bud[-1] += budget - base * R
    used, best_len = [0] * R, lens[0]
    hist = [list(lens)]
    att = np.zeros(max(R - 1, 0), dtype=int)
    acc = np.zeros(max(R - 1, 0), dtype=int)
    rounds = math.ceil(base / interval)
    for rnd in range(rounds):
        for r in range(R):
            st = min(interval, bud[r] - used[r])
            if st > 0:
                tours[r], lens[r], done, _, _, bl = PT._chain_segment(tours[r], lens[r], D, float(temps[r]), st, rngs[r])
                used[r] += done
                best_len = min(best_len, bl)
        if swaps and R > 1:
            for i in range(0 if rnd % 2 == 0 else 1, R - 1, 2):
                att[i] += 1
                x = (1 / temps[i] - 1 / temps[i + 1]) * (lens[i] - lens[i + 1])
                if swap_rng.random() < min(1.0, math.exp(min(x, 0.0))):
                    acc[i] += 1
                    tours[i], tours[i + 1] = tours[i + 1], tours[i]
                    lens[i], lens[i + 1] = lens[i + 1], lens[i]
        hist.append(list(lens))
    tail = False
    for r in range(R):
        if bud[r] - used[r] > 0:
            tail = True
            tours[r], lens[r], done, _, _, bl = PT._chain_segment(tours[r], lens[r], D, float(temps[r]), bud[r] - used[r], rngs[r])
            used[r] += done
            best_len = min(best_len, bl)
    if tail:
        hist.append(list(lens))
    return sum(used), rounds, best_len, np.array(hist), att, acc, tours[0]


def test_parallel_tempering_matches_a_resimulation_with_the_same_random_stream():
    rng = np.random.default_rng(8)
    for it in range(25):
        n = int(rng.integers(6, 14))
        D = T.dist_matrix(rng.random((n, 2)) * 100)
        R = int(rng.integers(1, 7))
        temps = PT.temperature_ladder(0.1, 0.6, R) * 4.0
        budget, interval = int(rng.integers(1, 2500)), int(rng.choice([1, 7, 30, 300]))
        start = T.random_tour(n, rng)
        swaps = it % 4 != 0
        got = PT.parallel_tempering(D, start, temps, budget, interval, it, swaps_enabled=swaps, keep_snapshots=False)
        ev, rounds, best_len, hist, att, acc, cold = _reference(D, start, temps, budget, interval, it, swaps)
        assert got.evaluations == ev == budget and got.rounds == rounds
        assert got.best_length == pytest.approx(best_len, abs=1e-9) and np.allclose(got.replica_lengths, hist)
        assert np.array_equal(got.swap_attempts, att) and np.array_equal(got.swap_accepts, acc) and np.array_equal(got.coldest_tour, cold)
        assert got.best_length == pytest.approx(T.tour_length(got.best_tour, D), abs=1e-7)


def test_temperature_ladder_is_geometric_between_the_endpoints():
    ladder = PT.temperature_ladder(0.1, 0.3, 5)
    assert ladder[0] == pytest.approx(0.1) and ladder[-1] == pytest.approx(0.3)
    assert ladder[1:] / ladder[:-1] == pytest.approx([3 ** 0.25] * 4)
