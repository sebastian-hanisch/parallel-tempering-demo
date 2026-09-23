"""Parallel Tempering / Replica Exchange (Swendsen & Wang 1986; Geyer 1991) für eine Rundtour (TSP).

Simulated Annealing braucht EINE Kette mit einem sorgfältig getroffenen Abkühlplan (Anfangs-/Endtemperatur) - trifft
man die Endtemperatur falsch, kommt die Kette entweder nicht zur Ruhe (zu heiß) oder verhält sich wie ein reiner
Abstieg (zu kalt). Parallel Tempering umgeht die Planung: mehrere Ketten ("Replikate") laufen GLEICHZEITIG bei
FESTEN Temperaturen T_1 < T_2 < ... < T_R (eine Leiter). Jede Kette macht normale Metropolis-Schritte bei ihrer
eigenen Temperatur - kein Plan, keine Stufen. In regelmäßigen Abständen wird ein TAUSCH zwischen benachbarten
Ketten vorgeschlagen: ihre aktuellen Touren werden vertauscht, angenommen mit

    p_swap = min(1, exp((1/T_i - 1/T_j) * (L_i - L_j)))

Diese Formel erhält die GEMEINSAME Boltzmann-Verteilung über alle Ketten hinweg exakt (Detailed Balance auf
Ketten-Paar-Ebene) - eine heiße Kette entkommt leicht lokalen Optima, eine kalte Kette verfeinert die besten
Touren, und der Austausch lässt gute Touren zu den kalten Ketten "durchsickern", ohne dass irgendeine Kette selbst
abkühlen muss. Ein Tausch bewertet keinen neuen Nachbarn und zählt deshalb NICHT gegen das Bewertungsbudget -
dieselbe Konvention wie ein ILS-Kick.

Nur Metropolis (nicht die anderen drei Annahmeregeln der Simulated-Annealing-Demo): die Tausch-Formel setzt eine
wohldefinierte Boltzmann-Gleichgewichtsverteilung ~exp(-L/T) je Kette voraus - das gilt nur für Metropolis.
Threshold Accepting, Great Deluge und Late Acceptance Hill Climbing haben keine bekannte Gleichgewichtsverteilung,
die Tausch-Formel wäre für sie nicht herleitbar (literaturtreue, keine willkürliche Einschränkung).

`_chain_segment` ist eine PORTIERTE Teilmenge von `simulated-annealing-demo/sa_algorithm.py`s `anneal()`-Funktion
(nur der Metropolis-Zweig, feste Temperatur, nur 2-opt) - kein eigenständig hergeleitetes Risiko im Kern-Innenloop.

Hinweis zu den Regressionstests (tests/test_algorithm.py): ein in mehrere `_chain_segment`-Aufrufe aufgeteilter
Lauf (wie `parallel_tempering` ihn pro Tausch-Runde macht) konsumiert den Zufallsstrom NICHT bytegleich zu einem
einzigen Aufruf über dieselbe Gesamtzahl an Vorschlägen - die interne Losgröße `m` hängt von den je Aufruf NOCH
AUSSTEHENDEN Vorschlägen ab, ein einzelner großer Aufruf zieht also andere Losgrößen als mehrere kleine. Das ist
keine Ungenauigkeit, nur eine Eigenschaft des Chunking (dieselbe Technik wie in `anneal()`, dort aber nie über
mehrere Aufrufe hinweg fortgesetzt). Die Regressionstests vergleichen deshalb NUR Situationen, in denen exakt
derselbe Aufruf-Zuschnitt vorliegt (ein Tausch-Intervall über dem gesamten Budget = genau ein Aufruf je Kette;
oder: dieselbe Kette mit identischem Budget/Tausch-Intervall/Seed, aber unterschiedlicher Zahl anderer Ketten
daneben) - nicht einen fein zerlegten Lauf gegen einen groben."""

import math
from dataclasses import dataclass, field

import numpy as np

import pt_tour as T

CHUNK = 8192


def swap_probability(t_i, t_j, l_i, l_j):
    """Wahrscheinlichkeit, die Touren zweier Ketten bei Temperatur t_i, t_j (Längen l_i, l_j) zu tauschen.
    t_i == t_j ergibt IMMER 1.0 (Tausch zwischen gleich-temperierten Ketten ändert die gemeinsame Verteilung nie
    und muss immer angenommen werden) - hier kein Sonderfall, sondern eine Folge der Formel: der Exponent ist dann
    0, exp(0) = 1."""
    x = (1.0 / t_i - 1.0 / t_j) * (l_i - l_j)
    if x >= 0.0:
        return 1.0
    return math.exp(x)


def _chain_segment(t, length, D, T, steps, rng):
    """`steps` Metropolis-Vorschläge (2-opt) bei fester Temperatur `T` - direkt aus `sa_algorithm.anneal()`s
    Metropolis-Zweig portiert (kein Plan, keine Stufen, keine andere Nachbarschaft). `steps` ist die Zahl der
    GÜLTIGEN Vorschläge (ungültige Paare werden verworfen und neu gezogen, zählen nicht). Gibt (neue Tour, neue
    Länge, Bewertungen [== steps], angenommene Vorschläge, BESTE während des Segments erreichte Tour, deren Länge)
    zurück - Metropolis nimmt auch Verschlechterungen an, die Länge schwankt also INNERHALB eines Segments; die
    beste Tour muss deshalb laufend mitverfolgt werden, nicht nur am Ende des Segments abgelesen werden (ein realer
    Fehler in einer früheren Fassung: `parallel_tempering`/`sa_baseline` prüften nur den End-Zustand jedes
    Segments auf ein neues Bestergebnis und übersahen dadurch bessere Zwischenstände - siehe tests/test_algorithm.py)."""
    n = len(D)
    Dl = D.tolist()
    t = [int(x) for x in t]
    length = float(length)
    best_t, best_length = list(t), length
    done = accepted = 0
    exp = math.exp
    while done < steps:
        m = min(CHUNK, (steps - done) + (steps - done) // 8 + 16)
        r_a = rng.integers(0, n, size=m).tolist()
        r_b = rng.integers(0, n, size=m).tolist()
        r_u = rng.random(size=m).tolist()
        for q in range(m):
            if done >= steps:
                break
            i, j = r_a[q], r_b[q]
            if i > j:
                i, j = j, i
            if j < i + 2 or (i == 0 and j == n - 1):
                continue                                          # ungültiges Paar: neu ziehen (zählt nicht als Vorschlag)
            a, b, c, d = t[i], t[i + 1], t[j], t[(j + 1) % n]
            delta = Dl[a][c] + Dl[b][d] - Dl[a][b] - Dl[c][d]
            done += 1
            if delta <= 0.0 or r_u[q] < exp(-delta / T):
                accepted += 1
                length += delta
                t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                if length < best_length - 1e-9:
                    best_length = length
                    best_t = list(t)
    return np.array(t, dtype=np.int64), length, done, accepted, np.array(best_t, dtype=np.int64), best_length


@dataclass
class PTRun:
    best_tour: np.ndarray
    best_length: float
    coldest_tour: np.ndarray             # finale Tour der kältesten Kette (Temperaturen[0]) - der SA-`final_tour` am nächsten
    coldest_length: float
    temperatures: np.ndarray             # R Werte, aufsteigend
    replica_lengths: np.ndarray          # (Runden+1, R): Länge je Kette am Ende jeder Runde (Runde 0 = Start)
    replica_best: np.ndarray             # (Runden+1,): bestes bislang gefundenes Ergebnis über alle Ketten
    swap_attempts: np.ndarray            # (R-1,) Zahl versuchter Tausche je Nachbarpaar (i, i+1)
    swap_accepts: np.ndarray             # (R-1,) Zahl angenommener Tausche je Nachbarpaar
    evaluations: int = 0
    rounds: int = 0
    snapshots: list = field(default_factory=list)   # je Runde: Liste der R Touren am Ende dieser Runde (falls keep_snapshots)


def parallel_tempering(D, start, temperatures, budget, swap_interval, seed, swaps_enabled=True, keep_snapshots=True):
    """Parallel Tempering mit R = len(temperatures) Ketten (aufsteigend sortiert erwartet), gemeinsamem
    Bewertungsbudget `budget` (gleichmäßig auf die Ketten aufgeteilt, Rest an die letzte/heißeste Kette - dieselbe
    Konvention wie die letzte Stufe in `sa_algorithm.anneal()`), Tausch-Versuchen alle `swap_interval` Vorschläge
    je Kette im Schachbrett-Schema (gerade Runden: Paare (0,1),(2,3),...; ungerade Runden: (1,2),(3,4),... - sorgt
    dafür, dass sich über die Zeit alle Nachbarpaare mischen). Alle Ketten starten von DERSELBEN Starttour `start`.
    `swaps_enabled=False` läuft R komplett unabhängige Ketten (eigener RNG-Strom `seed + replica_index` je Kette) -
    die Regressionsbasis gegen `_chain_segment` allein (siehe tests/test_algorithm.py)."""
    temps = np.asarray(temperatures, dtype=float)
    R = len(temps)
    D = np.asarray(D)
    start_length = T.tour_length(start, D)
    tours = [np.asarray(start, dtype=np.int64).copy() for _ in range(R)]
    lengths = [start_length] * R
    rngs = [np.random.default_rng(seed + r) for r in range(R)]
    swap_rng = np.random.default_rng((seed, R)) if swaps_enabled else None

    base = budget // R
    replica_budget = [base] * R
    replica_budget[R - 1] += budget - base * R
    used = [0] * R
    evaluations = 0

    best_length, best_tour = lengths[0], tours[0].copy()
    replica_len_hist = [list(lengths)]
    replica_best_hist = [best_length]
    swap_attempts = np.zeros(max(R - 1, 0), dtype=np.int64)
    swap_accepts = np.zeros(max(R - 1, 0), dtype=np.int64)
    snapshots = [[t.copy() for t in tours]] if keep_snapshots else []

    rounds = -(-base // swap_interval) if swap_interval > 0 else 1
    for rnd in range(rounds):
        for r in range(R):
            steps_here = min(swap_interval, replica_budget[r] - used[r])
            if steps_here <= 0:
                continue
            tours[r], lengths[r], done, _acc, seg_best_t, seg_best_len = _chain_segment(tours[r], lengths[r], D, float(temps[r]), steps_here, rngs[r])
            used[r] += done
            evaluations += done
            if seg_best_len < best_length - 1e-9:
                best_length, best_tour = seg_best_len, seg_best_t
        if swaps_enabled and R > 1:
            pairs = range(0, R - 1, 2) if rnd % 2 == 0 else range(1, R - 1, 2)
            for i in pairs:
                j = i + 1
                swap_attempts[i] += 1
                p = swap_probability(float(temps[i]), float(temps[j]), lengths[i], lengths[j])
                if swap_rng.random() < p:
                    swap_accepts[i] += 1
                    tours[i], tours[j] = tours[j], tours[i]
                    lengths[i], lengths[j] = lengths[j], lengths[i]
        replica_len_hist.append(list(lengths))
        replica_best_hist.append(best_length)
        if keep_snapshots:
            snapshots.append([t.copy() for t in tours])

    tail_used = False
    for r in range(R):                                            # Rest-Budget dieser Kette (aus der Aufteilung oben) noch verbrauchen
        remaining = replica_budget[r] - used[r]
        if remaining > 0:
            tail_used = True
            tours[r], lengths[r], done, _acc, seg_best_t, seg_best_len = _chain_segment(tours[r], lengths[r], D, float(temps[r]), remaining, rngs[r])
            used[r] += done
            evaluations += done
            if seg_best_len < best_length - 1e-9:
                best_length, best_tour = seg_best_len, seg_best_t
    if tail_used:                                                  # Historie um den Rest-Schritt ergänzen, damit sie immer mit den Endwerten übereinstimmt
        replica_len_hist.append(list(lengths))
        replica_best_hist.append(best_length)
        if keep_snapshots:
            snapshots.append([t.copy() for t in tours])

    coldest_tour, coldest_length = tours[0], lengths[0]
    return PTRun(best_tour, best_length, coldest_tour, coldest_length, temps,
                 np.array(replica_len_hist), np.array(replica_best_hist), swap_attempts, swap_accepts,
                 evaluations, rounds, snapshots)


def temperature_ladder(t_min, t_max, r):
    """R Temperaturen, geometrisch gestuft von `t_min` bis `t_max` (dieselbe Stufung wie der geometrische
    Abkühlplan der Simulated-Annealing-Demo, hier aber als FESTE Leiter statt eines zeitlichen Plans)."""
    if r == 1:
        return np.array([float(t_min)])
    k = np.arange(r, dtype=float)
    return float(t_min) * (float(t_max) / float(t_min)) ** (k / (r - 1))


@dataclass
class SABaselineRun:
    best_tour: np.ndarray
    best_length: float
    final_tour: np.ndarray
    final_length: float
    evaluations: int = 0


def sa_baseline(D, start, t0, t_end, budget, levels, seed):
    """Simulated Annealing mit geometrischem Abkühlplan, als Vergleichsgröße - dieselbe Metropolis-Bausteinlogik
    wie eine einzelne Parallel-Tempering-Kette (`_chain_segment`), aber mit sich ändernder statt fester Temperatur
    je Stufe (kein eigenständig hergeleitetes Risiko: derselbe Baustein, nur mit einer anderen Temperaturfolge
    aufgerufen). Nur der geometrische Plan (die kalibrierte Voreinstellung der Simulated-Annealing-Demo) - linear/
    logarithmisch bewusst nicht übernommen, siehe Grenzen."""
    D = np.asarray(D)
    levels = max(1, min(int(levels), int(budget)))
    temps = temperature_ladder(t0, t_end, levels) if levels > 1 else np.array([float(t0)])
    level_len = budget // levels
    rng = np.random.default_rng(seed)
    t = np.asarray(start, dtype=np.int64).copy()
    length = T.tour_length(t, D)
    best_length, best_tour = length, t.copy()
    evaluations = 0
    for lvl in range(levels):
        steps = level_len if lvl < levels - 1 else budget - level_len * (levels - 1)
        if steps <= 0:
            continue
        t, length, done, _acc, seg_best_t, seg_best_len = _chain_segment(t, length, D, float(temps[lvl]), steps, rng)
        evaluations += done
        if seg_best_len < best_length - 1e-9:
            best_length, best_tour = seg_best_len, seg_best_t
    return SABaselineRun(best_tour, best_length, t, length, evaluations)
