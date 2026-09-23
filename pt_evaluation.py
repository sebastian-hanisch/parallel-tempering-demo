"""Auswertung der Parallel-Tempering-Demo: eine PT-Leiter gegen kalibriertes Simulated Annealing (derselbe Standard-
Lauf wie die Simulated-Annealing-Demo selbst) und gegen Hill Climbing mit Neustarts, bei gleichem Bewertungsbudget.

Der Abstand zur Schranke ist der Abstand zu einer *unteren* Schranke der kürzesten Tour (1-Baum, Held-Karp). Ein
Vorschlag ist ein bewerteter Nachbar - dieselbe Einheit für PT, SA und Hill Climbing (ein Tausch-Versuch zählt
NICHT als Vorschlag, siehe pt_algorithm.py)."""

import time
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

import pt_algorithm as PT
import pt_constants as C
import pt_scenario as S
import pt_tour as T


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    cluster_share: int = C.DEFAULT_BALLUNG
    seed: int = C.DEFAULT_SEED
    n_replicas: int = C.DEFAULT_R
    t_min: float = C.DEFAULT_T_MIN
    t_max: float = C.DEFAULT_T_MAX
    swap_interval: int = C.DEFAULT_SWAP_INTERVAL
    swaps_enabled: bool = C.DEFAULT_SWAPS_ENABLED
    budget: int = C.DEFAULT_BUDGET
    chain_seed: int = C.DEFAULT_CHAIN_SEED
    sa_t0: float = C.SA_T0
    sa_t_end: float = C.SA_T_END


@lru_cache(maxsize=256)
def instance(n, cluster_share, seed):
    inst = S.generate(n, cluster_share, seed)
    return inst, T.dist_matrix(inst.xy)


@lru_cache(maxsize=256)
def reference_bound(n, cluster_share, seed):
    inst, D = instance(n, cluster_share, seed)
    ref = T.descend(D, T.nearest_neighbor_tour(D), "2opt+oropt", "best", keep_steps=False)
    return T.held_karp_bound(D, ref.length, C.BOUND_ITERATIONS)


def unit_of(n, cluster_share, seed):
    """Temperatureinheit (km): mittlere Kantenlänge einer guten Tour = untere Schranke / Knotenzahl (wie in der Simulated-Annealing-Demo)."""
    return reference_bound(n, cluster_share, seed) / (n + 1)


def hc_restarts(D, budget, seed):
    """Hill Climbing mit Neustarts bei gleichem Bewertungsbudget: der erste Abstieg läuft immer zu Ende, weitere nur mit dem Rest. Gibt (beste Tour, Zahl der Starts, verbrauchte Bewertungen) zurück."""
    rng = np.random.default_rng(seed)
    used, starts, best = 0, 0, None
    while used < budget or best is None:
        cap = None if best is None else budget - used
        r = T.descend(D, T.random_tour(len(D), rng), "2opt", "first", keep_steps=False, max_evaluations=cap)
        used += r.evaluations
        starts += 1
        if best is None or r.length < best.length:
            best = r
    return best.tour, starts, used


@dataclass
class Analysis:
    settings: Settings
    inst: object
    D: np.ndarray
    bound: float
    unit: float
    start_tour: np.ndarray
    run: object                     # PTRun
    seconds: float
    sa: object                      # SABaselineRun (kalibrierte Simulated-Annealing-Vergleichsgröße), None ohne with_baselines
    sa_seconds: float
    hcr_tour: np.ndarray
    hcr_starts: int
    hcr_seconds: float
    crossings_end: int

    def gap_of(self, length):
        return 100.0 * (length - self.bound) / self.bound

    @property
    def gap(self):
        return self.gap_of(self.run.best_length)

    @property
    def coldest_gap(self):
        return self.gap_of(self.run.coldest_length)

    @property
    def sa_gap(self):
        return self.gap_of(self.sa.best_length)

    @property
    def sa_final_gap(self):
        return self.gap_of(self.sa.final_length)

    @property
    def hcr_gap(self):
        return self.gap_of(T.tour_length(self.hcr_tour, self.D))

    @property
    def start_gap(self):
        return self.gap_of(T.tour_length(self.start_tour, self.D))

    @property
    def swap_accept_rate(self):
        return float(self.run.swap_accepts.sum() / max(self.run.swap_attempts.sum(), 1))


def analyse(settings, with_baselines=True, keep_snapshots=False):
    inst, D = instance(settings.n, settings.cluster_share, settings.seed)
    bound = reference_bound(settings.n, settings.cluster_share, settings.seed)
    unit = unit_of(settings.n, settings.cluster_share, settings.seed)
    start = T.random_tour(len(D), np.random.default_rng(settings.chain_seed))
    temps = PT.temperature_ladder(settings.t_min, settings.t_max, settings.n_replicas) * unit
    t0 = time.perf_counter()
    run = PT.parallel_tempering(D, start, temps, settings.budget, settings.swap_interval, settings.chain_seed,
                                 swaps_enabled=settings.swaps_enabled, keep_snapshots=keep_snapshots)
    seconds = time.perf_counter() - t0
    sa = hcr_tour = None
    sa_seconds = hcr_seconds = 0.0
    hcr_starts = 0
    if with_baselines:
        t0 = time.perf_counter()
        sa = PT.sa_baseline(D, start, settings.sa_t0 * unit, settings.sa_t_end * unit, settings.budget, C.SA_LEVELS, settings.chain_seed)
        sa_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        hcr_tour, hcr_starts, _ = hc_restarts(D, settings.budget, settings.chain_seed)
        hcr_seconds = time.perf_counter() - t0
    return Analysis(settings, inst, D, bound, unit, start, run, seconds, sa, sa_seconds, hcr_tour, hcr_starts, hcr_seconds, T.count_crossings(inst.xy, run.best_tour))


# --- Urteil (PT gegen KALIBRIERTES SA - die zentrale Frage dieses Stücks) ------------------------------------------------------------------------

WIN_MARGIN = 0.3
LOSE_MARGIN = 0.3


def verdict(a):
    """Code: beats_sa (PT klar besser als kalibriertes SA, gleiches Budget), sa_wins (SA besser), comparable. Der
    Vergleich ist die zentrale Frage dieses Stücks: braucht man PTs Temperaturleiter überhaupt, wenn man SA sorgfältig
    tunen kann? Gilt für diesen einen Lauf - die Ketten streuen."""
    if a.gap <= a.sa_gap - WIN_MARGIN:
        return "beats_sa"
    if a.sa_gap <= a.gap - LOSE_MARGIN:
        return "sa_wins"
    return "comparable"


# --- Sweeps und Tabellen -----------------------------------------------------------------------------------------------------------------------


def _mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def run_config(base, seeds=C.SWEEP_SEEDS, chains=C.SWEEP_CHAINS, **changes):
    """Mittel über die festen Instanzen und je `chains` Ketten-Seeds für die Einstellungen `base` mit `changes`."""
    s0 = replace(base, **changes)
    rows = []
    for seed in seeds:
        for ch in range(chains):
            a = analyse(replace(s0, seed=seed, chain_seed=ch))
            rows.append({"gap": a.gap, "coldest": a.coldest_gap, "sa": a.sa_gap, "sa_final": a.sa_final_gap, "hcr": a.hcr_gap,
                         "starts": a.hcr_starts, "seconds": a.seconds, "sa_seconds": a.sa_seconds, "swap_accept_rate": a.swap_accept_rate,
                         "crossings": a.crossings_end})
    out = {k: _mean(rows, k) for k in rows[0]}
    out.update({"gap_sd": float(np.std([r["gap"] for r in rows])), "gap_min": float(np.min([r["gap"] for r in rows])),
                "gap_max": float(np.max([r["gap"] for r in rows])), "n_runs": len(rows)})
    return out


SWEEP_VALUES = {"budget": C.BUDGETS, "n_replicas": tuple(range(C.R_MIN, C.R_MAX + 1)), "swap_interval": C.SWAP_INTERVAL_OPTIONS,
                "n": (10, 20, 40, 60, 100, 150, 200), "cluster_share": (0, 25, 50, 75, 100)}
SWEEP_LABELS = {"budget": "Budget (bewertete Nachbarn)", "n_replicas": "Anzahl Replikate", "swap_interval": "Tausch-Intervall",
                "n": "Stopps", "cluster_share": "Anteil in Gruppen (%)"}


def sweep(param, base=Settings(), values=None):
    values = SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(base, **{param: v})} for v in values]


SCALING_POLICIES = (("Budget 200 Tausend", lambda n: 200000), ("Budget 5 000 · Stopps", lambda n: 5000 * n))


def scaling_table(base=Settings()):
    return [{"label": label, "rows": [{"value": n, **run_config(base, n=n, budget=fn(n))} for n in C.SCALING_N]} for label, fn in SCALING_POLICIES]


def chain_spread(settings, k=C.SPREAD_CHAINS):
    """k Ketten-Seeds auf derselben Instanz: PT (beste Tour) gegen kalibriertes SA, gleiches Budget."""
    pt_gaps, sa_gaps = [], []
    for ch in range(k):
        a = analyse(replace(settings, chain_seed=ch))
        pt_gaps.append(a.gap)
        sa_gaps.append(a.sa_gap)
    return {"pt": np.array(pt_gaps), "sa": np.array(sa_gaps)}
