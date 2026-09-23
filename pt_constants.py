"""Konstanten der Parallel-Tempering-Demo: Szenario (wortgleich zur Hill-Climbing-Demo), Regler, Beschriftungen (Presets folgen nach den Messungen)."""

AREA = 100.0                     # Kantenlänge des Gebiets in km
N_CLUSTERS = 5
CLUSTER_SIGMA = 6.0              # Streuung einer Gruppe in km
CLUSTER_MARGIN = 12.0            # Gruppenmittelpunkte liegen mindestens so weit vom Rand entfernt
SWEEP_SEEDS = tuple(range(100000, 100005))
SWEEP_CHAINS = 3                 # Ketten-Seeds je Instanz in Sweeps und Vergleichstabellen
BOUND_ITERATIONS = 300

N_MIN, N_MAX, DEFAULT_N, N_STEP = 10, 200, 60, 5
BALLUNG_MIN, BALLUNG_MAX, DEFAULT_BALLUNG, BALLUNG_STEP = 0, 100, 0, 25
SEED_MAX = 999999
DEFAULT_SEED = 35
DEFAULT_CHAIN_SEED = 0
DEFAULT_BUDGET = 200000
BUDGETS = (10000, 25000, 50000, 100000, 200000, 500000, 1000000, 2000000)
SCALING_N = (20, 40, 60, 100, 150, 200)
SPREAD_CHAINS = 20

# --- Parallel-Tempering-eigene Regler ------------------------------------------------------------------------------------------------------
# Temperaturen sind wie bei der Simulated-Annealing-Demo Vielfache der mittleren Kantenlänge einer guten Tour (= untere Schranke / Knotenzahl).
# Kalibriert 2026-09-23 (60 Stopps, Standardbudget 200 Tausend, 5 Instanzen x 3 Ketten, siehe tests/test_claims.py):
#   LEITERBREITE (R=5, Tausch-Intervall=100 während der Kalibrierung): (0.4-0.6) 6.79 %, (0.2-0.8) 2.15 %,
#   (0.05-1.0) 3.16 %, (0.02-2.0) 4.54 %, (0.1-0.3) 1.41 % - EINE ENGE, GUT PLATZIERTE Leiter schlägt eine BREITE
#   (die SA-Extreme abdeckende) klar; die Vorab-Vermutung "eine breite Leiter braucht keine Feinabstimmung" stimmt
#   NICHT ungeprüft (siehe README).
#   REPLIKAT-ZAHL R (bei 0.1-0.3, Tausch-Intervall=300, dem kalibrierten Endwert): R=2 2.24 %, R=3 2.19 %,
#   R=5 1.15 %, R=7 1.78 %, R=10 2.32 % - nicht-monotones Optimum bei R=5 (mehr Replikate verteilen dasselbe
#   Budget dünner; zu wenige Replikate lassen die Leiter zu grob).
#   TAUSCH-INTERVALL (bei 0.1-0.3, R=5): 10 2.05 %, 30 2.03 %, 100 1.41 %, 300 1.15 %, 1000 1.64 %, 3000 2.19 % -
#   ebenfalls ein nicht-monotones Optimum (zu häufiger Tausch verschwendet Bewertungen auf Tausch-Diagnostik statt
#   Suche, zu seltener lässt die Ketten wieder isoliert laufen).
#   TAUSCH AN/AUS (kalibrierte Leiter, sonst Standard): an 1.15 %, aus 2.40 % - der Austausch selbst bringt echten
#   Wert (nicht nur "mehrere Ketten bei verschiedenen Temperaturen zu haben" reicht); geprüft über ALLE 5
#   Sweep-Instanzen einzeln, nicht nur im Mittel (an < aus bei jeder einzelnen).
#   ROBUSTHEIT gegen schlecht gewählte Leitern (kalibrierte Tausch-Einstellungen, nur Leiter variiert): kalibriert
#   (0.1-0.3) 1.15 %, SA-Extreme abgedeckt (0.02-0.5) 3.22 %, eng zu heiß (0.4-0.6) 6.62 %, eng zu kalt (0.02-0.1)
#   4.48 % - eine schlecht gewählte Leiter KOSTET spürbar (3-6x schlechter als kalibriert), PT ist also NICHT
#   tuningfrei. Aber selbst die schlechteste getestete Leiter (6.62 %) bleibt weit vor SA's eigenen
#   Fehlkalibrierungen (SA zu kalt 8.66 %, SA zu heiß 14.96 % beste / 31.57 % letzte Tour) - PT ist deutlich
#   ROBUSTER gegen eine ungefähr richtige statt exakt richtige Temperaturwahl, aber nicht tuningfrei.
R_MIN, R_MAX, DEFAULT_R, R_STEP = 2, 10, 5, 1
T_MIN_MIN, T_MIN_MAX, T_MIN_STEP, DEFAULT_T_MIN = 0.01, 1.0, 0.01, 0.1
T_MAX_MIN, T_MAX_MAX, T_MAX_STEP, DEFAULT_T_MAX = 0.1, 4.0, 0.05, 0.3
SWAP_INTERVAL_OPTIONS = (10, 30, 100, 300, 1000, 3000)
DEFAULT_SWAP_INTERVAL = 300
DEFAULT_SWAPS_ENABLED = True

# SA-Vergleichsgröße: die KALIBRIERTEN Standardwerte der Simulated-Annealing-Demo (t0=0.5, t_end=0.1, geometrisch, 100 Stufen) - derselbe Lauf, den die SA-Demo selbst als "Standardfall" zeigt.
SA_T0, SA_T_END, SA_LEVELS = 0.5, 0.1, 100

# BUDGET-SWEEP (60 Stopps, kalibrierte Leiter, PT gegen SA-getunt gegen Hill-Climbing-Neustarts), 10T/25T/50T/100T/
# 200T/500T/1M/2M: PT 32.05/9.65/4.89/2.37/1.15/0.77/0.62/0.49 %; SA 9.01/4.72/3.07/1.94/1.42/1.27/0.94/0.73 %;
# HCR 7.88/7.88/7.88/7.88/4.88/2.93/2.54/1.87 %. PT verliert bei KLEINEM Budget deutlich (das auf R=5 Ketten
# aufgeteilte Budget lässt jeder Kette zu wenig, um überhaupt anzukommen) - der Umschlagpunkt liegt zwischen
# 100 Tausend und 200 Tausend; AB da gewinnt PT, mit wachsendem Vorsprung (2M: 0.49 gegen 0.73 %, ein knappes
# Drittel besser).
# SKALIERUNG bei festem 200-Tausend-Budget, 20/40/60/100/150/200 Stopps: PT 0.05/0.95/1.15/6.20/12.95/23.55 %;
# SA 0.05/0.79/1.42/3.08/6.54/7.87 % - NICHT monoton: bei n=20 praktisch gleichauf, bei n=40 knapp schlechter
# als SA, bei n=60 wieder vorn - erst AB n=100 wird der Rückstand groß und durchgehend (dieselbe
# Budget-Teilungs-Ursache wie beim Budget-Sweep: eine größere Instanz braucht mehr Bewertungen je Kette, das
# geteilte Budget reicht dafür immer schlechter). Bei wachsendem Budget (5000 Stopps): 20/40/60/100/150/200:
# PT 0.05/0.95/0.89/4.16/5.95/7.20 %; SA 0.05/0.79/1.32/3.00/3.06/3.78 % - der Rückstand schrumpft, bleibt aber
# bestehen: SA braucht KEINEN Ketten-Split und skaliert deshalb strukturell günstiger zu großen Instanzen.


def _preset(t_min=DEFAULT_T_MIN, t_max=DEFAULT_T_MAX, n_replicas=DEFAULT_R, swap_interval=DEFAULT_SWAP_INTERVAL,
            swaps_enabled=DEFAULT_SWAPS_ENABLED, budget=DEFAULT_BUDGET, n=DEFAULT_N):
    return {"n": n, "ballung": DEFAULT_BALLUNG, "seed": DEFAULT_SEED, "n_replicas": n_replicas, "t_min": t_min, "t_max": t_max,
            "swap_interval": swap_interval, "swaps_enabled": swaps_enabled, "budget": budget, "chain_seed": DEFAULT_CHAIN_SEED}


PRESETS = {
    "Standardfall (Voreinstellung)": _preset(),
    "Ohne Tausch (Kontrolle)": _preset(swaps_enabled=False),
    "Weite Leiter (deckt SA-Extreme ab)": _preset(t_min=0.02, t_max=0.5),
    "Kleines Budget (10 Tausend)": _preset(budget=10000),
    "Großes Budget (1 Million)": _preset(budget=1000000),
    "Große Instanz (200 Stopps)": _preset(n=200),
}
# Mittel über die fünf festen Sweep-Instanzen (Seeds 100000-100004, je drei Ketten-Seeds), Abstand zur Schranke; "sa" = kalibriertes Simulated Annealing bei gleichem Budget
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "60 Stopps, R=5 Replikate, Leiter 0.1-0.3, Tausch alle 300 Vorschläge, 200 Tausend Vorschläge: die beste Tour liegt im Mittel 1.15 % über der Schranke - kalibriertes Simulated Annealing bei gleichem Budget 1.42 %, Hill Climbing mit Neustarts 4.88 %.",
    "Ohne Tausch (Kontrolle)": "Dieselben 5 Ketten, aber OHNE Austausch (jede läuft isoliert bei ihrer eigenen Temperatur): 2.40 % statt 1.15 % - der Austausch selbst bringt den Vorteil, nicht nur \"mehrere Temperaturen zu haben\".",
    "Weite Leiter (deckt SA-Extreme ab)": "Leiter von 0.02 bis 0.5 - deckt genau die Temperaturen ab, bei denen Simulated Annealing in der Schwester-Demo katastrophal scheitert (zu kalt: 8.66 %, zu heiß: 14.96 % beste / 31.57 % letzte Tour): Parallel Tempering kommt mit 3.22 % deutlich besser weg als SA an JEDEM dieser Extreme - aber auch klar schlechter als die kalibrierte, engere Leiter (1.15 %). Robuster als SA, aber nicht tuningfrei.",
    "Kleines Budget (10 Tausend)": "Nur 10 Tausend Vorschläge, auf 5 Ketten verteilt (2 Tausend je Kette): 32.05 % über der Schranke - jede Kette kommt kaum vom Fleck. Kalibriertes SA (ein Lauf, kein Split) liegt bei 9.01 %.",
    "Großes Budget (1 Million)": "1 Million Vorschläge: 0.62 % über der Schranke gegen 0.94 % für kalibriertes SA - der Vorsprung von Parallel Tempering WÄCHST mit dem Budget, statt zu schrumpfen wie bei den meisten anderen Stücken dieser Linie.",
    "Große Instanz (200 Stopps)": "200 Stopps, 200 Tausend Vorschläge: Parallel Tempering liegt bei 23.55 % gegen 7.87 % für kalibriertes SA - derselbe Budget-Teilungs-Effekt wie beim kleinen Budget, hier durch die Instanzgröße statt das Budget ausgelöst. Ehrlicher Negativbefund.",
}
# Urteile, die bei diesem Preset über verschiedene Instanzen und Ketten-Seeds vorkommen (jedes Preset wird über mehrere Instanzen x 2 Ketten gemessen)
PRESET_EXPECTED_BANDS = {
    "Standardfall (Voreinstellung)": {"beats_sa", "comparable", "sa_wins"},
    "Ohne Tausch (Kontrolle)": {"beats_sa", "comparable", "sa_wins"},
    "Weite Leiter (deckt SA-Extreme ab)": {"beats_sa", "comparable", "sa_wins"},
    "Kleines Budget (10 Tausend)": {"sa_wins"},
    "Großes Budget (1 Million)": {"beats_sa", "comparable", "sa_wins"},
    "Große Instanz (200 Stopps)": {"sa_wins"},
}
