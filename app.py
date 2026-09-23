"""Parallel Tempering - mehrere Ketten bei festen Temperaturen tauschen Zustände statt einen Abkühlplan zu fahren - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Neuntes Stück der "Konzepte"-Reihe der Trajektorien-Metaheuristiken-Linie, Fortsetzung der Simulated-Annealing-Demo
(Kind der Wurzel Hill Climbing, wie SA selbst - kein Kind des Nachbarschafts-Zweigs). Simulated Annealing braucht
EINE Kette mit einem sorgfältig getroffenen Abkühlplan; trifft man die Endtemperatur falsch, kommt die Kette
entweder nicht zur Ruhe (zu heiß) oder verhält sich wie ein reiner Abstieg (zu kalt). Parallel Tempering hält
stattdessen R Ketten bei FESTEN Temperaturen parallel und lässt sie periodisch Zustände TAUSCHEN - kein Abkühlplan
nötig. Löst das das Tuning-Problem, oder kostet die Aufteilung des Budgets auf mehrere Ketten mehr, als der
Austausch einbringt? Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time
from dataclasses import replace

import numpy as np
import streamlit as st

import pt_algorithm as PT
import pt_constants as C
import pt_evaluation as EV
import pt_tour as T
from pt_evaluation import SWEEP_LABELS, Settings, analyse, chain_spread, scaling_table, sweep, verdict
from pt_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_chain_seed,
    randomize_seed,
    sync_query_params,
)
from pt_visualization import build_budget, build_instance, build_ladder_heatmap, build_scaling, build_spread, build_sweep, build_swap_bar, build_tour

st.set_page_config(page_title="Parallel Tempering – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _run_with_snapshots(settings):
    inst, D = EV.instance(settings.n, settings.cluster_share, settings.seed)
    bound = EV.reference_bound(settings.n, settings.cluster_share, settings.seed)
    unit = EV.unit_of(settings.n, settings.cluster_share, settings.seed)
    start = T.random_tour(len(D), np.random.default_rng(settings.chain_seed))
    temps = PT.temperature_ladder(settings.t_min, settings.t_max, settings.n_replicas) * unit
    run = PT.parallel_tempering(D, start, temps, settings.budget, settings.swap_interval, settings.chain_seed,
                                 swaps_enabled=settings.swaps_enabled, keep_snapshots=True)
    return inst, D, bound, temps, run


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base)


@st.cache_data(show_spinner=False)
def _spread(base):
    return chain_spread(base)


@st.cache_data(show_spinner=False)
def _scaling(base):
    return scaling_table(base)


def _fmt_int(x):
    return f"{int(round(x)):,}".replace(",", ".")


st.title("🌡️ Parallel Tempering – mehrere Ketten tauschen Zustände statt einen Abkühlplan zu fahren")
st.markdown(
    """
**Simulated Annealing** braucht eine sorgfältig getroffene Endtemperatur - zu heiß, und die Kette kommt nie zur
Ruhe; zu kalt, und sie verhält sich wie ein reiner Abstieg. **Parallel Tempering** (Swendsen & Wang 1986;
Geyer 1991) umgeht die Planung: **R Ketten** laufen gleichzeitig bei **festen** Temperaturen T₁ < T₂ < ... < T_R
(eine Leiter) und **tauschen** periodisch ihre aktuellen Touren - angenommen mit einer Wahrscheinlichkeit, die
die gemeinsame Gleichgewichtsverteilung über alle Ketten hinweg exakt erhält. Eine heiße Kette entkommt leicht
lokalen Optima, eine kalte verfeinert die besten Touren, und der Austausch lässt gute Touren nach unten sickern.
Löst das Simulated Annealings Tuning-Problem - oder kostet die Aufteilung des Budgets auf mehrere Ketten mehr,
als der Austausch einbringt?
"""
)
st.caption(
    "Neuntes Stück der Trajektorien-Metaheuristiken-Linie der \"Konzepte\"-Reihe, Fortsetzung der "
    "[simulated-annealing-demo](https://sebastianhanisch-simulated-annealing-demo.streamlit.app/) - dieselbe "
    "Rundtour wie in der [hill-climbing-demo](https://sebastianhanisch-hill-climbing-demo.streamlit.app/), der "
    "[iterated-local-search-demo](https://github.com/sebastian-hanisch/iterated-local-search-demo), der "
    "[tabu-search-demo](https://github.com/sebastian-hanisch/tabu-search-demo) und weiterer Geschwister - ein "
    "Depot in der Mitte, n Kundenstopps in einem 100 × 100-km-Gebiet, euklidische Entfernungen."
)

with st.expander("So funktioniert Parallel Tempering", expanded=True):
    st.markdown(
        r"""
1. **Leiter aufstellen.** R Temperaturen T₁ < ... < T_R, geometrisch gestuft zwischen T_min und T_max (dieselbe
   Einheit wie bei Simulated Annealing: Vielfache der mittleren Kantenlänge einer guten Tour).
2. **Jede Kette macht normale Metropolis-Schritte** bei ihrer EIGENEN, FESTEN Temperatur - kein Plan, keine Stufen.
3. **Alle `Tausch-Intervall` Vorschläge**: ein Tausch zwischen benachbarten Ketten (i, i+1) wird vorgeschlagen -
   ihre aktuellen Touren werden vertauscht, angenommen mit
   $$p_{\text{tausch}} = \min\!\left(1,\ \exp\!\left[\left(\tfrac{1}{T_i}-\tfrac{1}{T_j}\right)(L_i-L_j)\right]\right)$$
4. **Diese Formel erhält die gemeinsame Gleichgewichtsverteilung exakt** - jede Kette bleibt bei ihrer EIGENEN
   Temperatur im Gleichgewicht, auch während sie ständig mit anderen tauscht (geprüft auf einer vollständig
   aufzählbaren Instanz, siehe Verifikation im README).
5. **Nur Metropolis**, nicht Simulated Annealings andere drei Annahmeregeln: die Tausch-Formel setzt eine
   wohldefinierte Gleichgewichtsverteilung je Kette voraus - die gibt es nur bei Metropolis.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(C.PRESETS.keys())
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name], key=f"preset_{name}")

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_stops = st.slider(
        "Stopps", *bounds("n_slider"), key="n_slider", step=C.N_STEP,
        help="Anzahl der Kundenstopps (das Depot kommt dazu). Bei fest 200 Tausend Vorschlägen bleibt Parallel Tempering bis 60 Stopps nah an kalibriertem SA dran (bei n=60 sogar leicht vorn), verliert aber ab 100 Stopps deutlich - dieselbe Budget-Teilungs-Ursache wie bei kleinem Budget.",
    )
    cluster_share = st.slider(
        "Anteil der Stopps in Gruppen [%]", *bounds("ballung_slider"), key="ballung_slider", step=C.BALLUNG_STEP,
        help="Wie viele Stopps in fünf Gruppen (Städten) liegen statt gleichverteilt im Gebiet.",
    )
    n_replicas = st.slider(
        "Anzahl Replikate (Ketten)", *bounds("r_slider"), key="r_slider", step=C.R_STEP,
        help="Mehr Ketten decken die Temperaturleiter feiner ab, bekommen aber weniger Budget je Kette - nicht-monotones Optimum bei R=5 (1.15 % gegen 1.58 % bei R=2 und 2.32 % bei R=10, Standardfall).",
    )
    t_min = st.slider(
        "Kälteste Temperatur T_min", *bounds("t_min_slider"), key="t_min_slider", step=C.T_MIN_STEP, format="%.2f",
        help="Vielfache der mittleren Kantenlänge einer guten Tour, wie SAs Endtemperatur.",
    )
    t_max = st.slider(
        "Heißeste Temperatur T_max", *bounds("t_max_slider"), key="t_max_slider", step=C.T_MAX_STEP, format="%.2f",
        help="Eine ENGE, gut platzierte Leiter (0.1-0.3) schlägt eine BREITE, die SAs eigene Fehlkalibrierungen abdeckt (0.02-0.5) klar (1.15 % gegen 3.22 %) - Parallel Tempering ist robuster als SA gegen ungefähr richtige Werte, aber nicht tuningfrei.",
    )
    swap_interval = st.select_slider(
        "Tausch-Intervall (Vorschläge je Kette)", options=list(C.SWAP_INTERVAL_OPTIONS), key="swap_interval_select",
        help="Zu häufiger Tausch verschwendet Bewertungen auf Diagnostik statt Suche, zu seltener lässt die Ketten isoliert laufen - nicht-monotones Optimum bei 300 (Standardfall).",
    )
    swaps_enabled = st.toggle(
        "Austausch aktiv", key="swaps_toggle",
        help="Ausschalten simuliert R komplett unabhängige Ketten bei verschiedenen festen Temperaturen, ohne Austausch - der Austausch selbst bringt echten Wert (1.15 % mit gegen 2.40 % ohne, Standardfall, über jede der 5 Sweep-Instanzen einzeln geprüft).",
    )
    budget = st.select_slider(
        "Budget (bewertete Nachbarn)", options=list(C.BUDGETS), key="budget_select", format_func=_fmt_int,
        help="Das Budget wird auf alle R Ketten aufgeteilt - bei kleinem Budget bekommt jede Kette zu wenig (10 Tausend: 32.05 % gegen 9.01 % für SA); ab etwa 150-200 Tausend gewinnt Parallel Tempering, mit wachsendem Vorsprung (2 Millionen: 0.49 % gegen 0.73 %).",
    )
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Seed für die Lage der Stopps.")
    chain_seed = st.number_input(
        "Zufalls-Seed der Kette", *bounds("chain_seed_input"), key="chain_seed_input", step=1,
        help="Steuert die zufällige gemeinsame Startlösung aller Ketten - der Kern ist sonst deterministisch (Metropolis-Zufall wird vom Seed abgeleitet).",
    )
    st.button("🎲 Neue Kette würfeln", width="stretch", on_click=randomize_chain_seed, help="Würfelt einen neuen Seed für dieselbe Instanz.")

sync_query_params({
    "n_slider": int(n_stops), "ballung_slider": int(cluster_share), "seed_input": int(seed), "r_slider": int(n_replicas),
    "t_min_slider": float(t_min), "t_max_slider": float(t_max), "swap_interval_select": int(swap_interval),
    "swaps_toggle": bool(swaps_enabled), "budget_select": int(budget), "chain_seed_input": int(chain_seed),
})

t_min, t_max = min(t_min, t_max), max(t_min, t_max)
settings = Settings(int(n_stops), int(cluster_share), int(seed), int(n_replicas), float(t_min), float(t_max),
                     int(swap_interval), bool(swaps_enabled), int(budget), int(chain_seed))
with st.spinner("Rechne..."):
    a = _analysis(settings)
    inst, D, bound, temps, run = _run_with_snapshots(settings)
xy = inst.xy
code = verdict(a)

# --- Parallel Tempering in Aktion -----------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Parallel Tempering in Aktion")
STEP_LABELS = {1: "1 · Instanz", 2: "2 · Leiter im Austausch", 3: "3 · Ergebnis"}
step = st.select_slider("Schritt", options=list(STEP_LABELS), key="pt_step", format_func=lambda s: STEP_LABELS[s])

MAX_DISPLAY_ROUNDS = 150
n_rounds = len(run.replica_lengths)
if n_rounds > MAX_DISPLAY_ROUNDS:
    idx = np.linspace(0, n_rounds - 1, MAX_DISPLAY_ROUNDS).astype(int)
else:
    idx = np.arange(n_rounds)

if step == 1:
    st.markdown(f"**{inst.n} Kundenstopps und das Depot (Stern)** – {inst.cluster_share} % der Stopps in Gruppen")
    st.plotly_chart(build_instance(xy), width="stretch", key="s1_map")
    st.caption(f"{inst.n} Stopps; die untere Schranke der kürzesten Rundtour liegt bei {bound:,.0f} km (1-Baum-Schranke, Held-Karp). Temperaturleiter: {', '.join(f'{t:.3f}' for t in temps)} km.".replace(",", "."))
elif step == 2:
    st.markdown(f"**Abstand zur Schranke je Kette über die Runden** ({'mit' if settings.swaps_enabled else 'ohne'} Austausch)")
    st.plotly_chart(build_ladder_heatmap(run.replica_lengths[idx], temps, bound, x=idx.tolist()), width="stretch", key="s2_heatmap")
    if settings.swaps_enabled and settings.n_replicas > 1:
        st.markdown("**Tausch-Annahmequote je Nachbarpaar**")
        st.plotly_chart(build_swap_bar(run.swap_attempts, run.swap_accepts, temps), width="stretch", key="s2_swap")
        st.caption(f"{run.swap_accepts.sum()} von {run.swap_attempts.sum()} versuchten Tauschen angenommen ({100 * run.swap_accepts.sum() / max(run.swap_attempts.sum(), 1):.1f} %).")
    else:
        st.info("Austausch ist ausgeschaltet - die Ketten laufen komplett unabhängig, keine Tausch-Versuche.")
else:
    c1, c2 = st.columns(2)
    c1.markdown(f"**Beste Tour über alle Ketten** – {a.gap:.1f} % über der Schranke")
    c1.plotly_chart(build_tour(xy, a.run.best_tour), width="stretch", key="s3_pt")
    c2.markdown(f"**Kalibriertes Simulated Annealing** (gleiches Budget) – {a.sa_gap:.1f} % über der Schranke")
    c2.plotly_chart(build_tour(xy, a.sa.best_tour), width="stretch", key="s3_sa")

st.markdown("---")

# --- Ergebnis --------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was die Suche gefunden hat")
st.caption(
    "**Abstand zur Schranke:** Länge der Tour gegenüber einer unteren Schranke der kürzesten Rundtour (1-Baum, Held-Karp) in Prozent. "
    "Ein Lauf ist eine Ziehung (nur die gemeinsame Startlösung und der Metropolis-Zufall streuen): Vergleiche gelten für diesen Lauf."
)
m1, m2, m3 = st.columns(3)
m1.metric("Parallel Tempering: beste Tour", f"{a.gap:.1f} %", delta=f"R={settings.n_replicas} Ketten", delta_color="off", help="Abstand zur Schranke der besten je gefundenen Tour über alle Ketten und Runden.")
m2.metric("Simulated Annealing (getunt)", f"{a.sa_gap:.1f} %", delta="gleiches Budget", delta_color="off", help="Kalibrierter SA-Standardlauf (T0=0.5, T_end=0.1) bei gleichem Bewertungsbudget.")
m3.metric("Hill Climbing mit Neustarts", f"{a.hcr_gap:.1f} %", delta=f"{a.hcr_starts} Abstiege", delta_color="off", help="Zum Vergleich: unabhängige Neustarts bei gleichem Budget.")

if code == "beats_sa":
    st.success(f"✅ Parallel Tempering gewinnt: {a.gap:.1f} % über der Schranke gegen {a.sa_gap:.1f} % für kalibriertes Simulated Annealing (gleiches Budget). Andere Ketten streuen um dieses Ergebnis.")
elif code == "comparable":
    st.info(f"ℹ️ Gleichauf: Parallel Tempering {a.gap:.1f} %, Simulated Annealing {a.sa_gap:.1f} % über der Schranke. Eine andere Kette kann das Bild drehen.")
else:
    st.warning(f"⚠️ Kalibriertes Simulated Annealing ist hier besser: {a.sa_gap:.1f} % gegen {a.gap:.1f} % über der Schranke bei gleichem Budget. Bei kleinem Budget oder großen Instanzen ist das der Regelfall - siehe README 'Was nicht funktioniert hat'.")

d1, d2 = st.columns(2)
with d1:
    st.markdown("**Kennzahlen im Detail**")
    unit_time = lambda sec: f"{sec * 1000:.0f} ms"  # noqa: E731
    st.table({"": ["Länge (km)", "Abstand zur Schranke", "Bewertete Nachbarn", "Rechenzeit"],
              "Parallel Tempering": [f"{T.tour_length(a.run.best_tour, a.D):.1f}", f"{a.gap:.2f} %", _fmt_int(settings.budget), unit_time(a.seconds)],
              "Simulated Annealing": [f"{T.tour_length(a.sa.best_tour, a.D):.1f}", f"{a.sa_gap:.2f} %", _fmt_int(settings.budget), unit_time(a.sa_seconds)]})
with d2:
    st.markdown("**Was gerechnet wurde**")
    st.table({"": ["Temperaturleiter", "Tausch-Intervall", "Tausch-Annahmequote", "Kreuzungen der besten Tour"],
              "Einstellung": [f"{settings.t_min:.2f} – {settings.t_max:.2f}", _fmt_int(settings.swap_interval), f"{100 * a.swap_accept_rate:.1f} %", f"{a.crossings_end}"]})
    st.caption("Ein Vorschlag ist ein bewerteter Nachbar - dieselbe Einheit für Parallel Tempering, Simulated Annealing und Hill Climbing. Ein Tausch-Versuch zählt NICHT als Vorschlag. Rechenzeiten hängen vom Rechner ab, nur die Größenordnung zählt.")

st.markdown("---")

# --- Sweeps ------------------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt das Ergebnis von Budget und Instanz ab?")
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(SWEEP_LABELS), format_func=lambda k: SWEEP_LABELS[k], key="sweep_select")
base_sweep = replace(settings, seed=0, chain_seed=0)
if st.button("Sweep über 5 feste Instanzen berechnen (dauert etwa 10 bis 60 Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {(sweep_param, base_sweep)}
if (sweep_param, base_sweep) in st.session_state.get("sweep_done", set()):
    with st.spinner("Rechne den Sweep über 5 feste Instanzen × 3 Ketten..."):
        rows_sweep = _sweep(sweep_param, base_sweep)
    st.plotly_chart(build_sweep(rows_sweep, SWEEP_LABELS[sweep_param]), width="stretch", key="sweep_chart")
    st.caption("Mittel und Streuung (Band) über 5 feste Instanzen (Seeds 100000–100004, getrennt vom Seed oben) mit je drei Ketten; alle anderen Regler wie in der Seitenleiste. "
               "Gepunktet: kalibriertes Simulated Annealing bei gleichem Budget.")

st.markdown("---")

# --- Experimente --------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Budget: bleibt der Vorsprung bestehen?")
if st.button("Budget von 10 Tausend bis 2 Millionen durchfahren (dauert etwa 60 Sekunden)", key="budget_start"):
    st.session_state["budget_on"] = True
if st.session_state.get("budget_on"):
    with st.spinner("Rechne 8 Budgets × 5 Instanzen × 3 Ketten..."):
        rows_b = _sweep("budget", base_sweep)
    st.plotly_chart(build_budget(rows_b), width="stretch", key="budget_chart")
    st.table({"Budget": [_fmt_int(r["value"]) for r in rows_b], "Parallel Tempering (%)": [f"{r['gap']:.2f}" for r in rows_b], "Simulated Annealing (%)": [f"{r['sa']:.2f}" for r in rows_b]})
    st.caption("Mittel über 5 feste Instanzen × 3 Ketten (60 Stopps, kalibrierte Leiter). Bei kleinem Budget verliert Parallel Tempering deutlich (jede der R Ketten bekommt zu wenig) - "
               "ab etwa 150-200 Tausend dreht sich das Bild, und der Vorsprung WÄCHST mit dem Budget (2 Millionen: 0.49 % gegen 0.73 %), statt wie bei den meisten anderen Stücken dieser Linie zu schrumpfen.")

st.markdown("---")

st.subheader("🔬 Streuung: wie verlässlich ist eine Kette?")
if st.button("20 Ketten auf dieser Instanz berechnen (dauert etwa 15 Sekunden)", key="spread_start"):
    st.session_state["spread_on"] = True
if st.session_state.get("spread_on"):
    with st.spinner("Rechne 20 Ketten..."):
        sp = _spread(replace(settings, chain_seed=0))
    st.plotly_chart(build_spread(sp["pt"], sp["sa"]), width="stretch", key="spread_chart")
    s1, s2 = st.columns(2)
    s1.metric("Parallel Tempering: Mittel ± Streuung", f"{sp['pt'].mean():.2f} ± {sp['pt'].std():.2f} %", help="Mittel und Standardabweichung des Abstands der besten Tour über 20 Ketten.")
    s2.metric("Simulated Annealing: Mittel ± Streuung", f"{sp['sa'].mean():.2f} ± {sp['sa'].std():.2f} %", help="Dieselben 20 Ketten, kalibriertes SA statt Parallel Tempering.")
    st.caption("Dieselbe Instanz, 20 verschiedene Ketten-Seeds (steuern nur die gemeinsame Startlösung).")

st.markdown("---")

st.subheader("🔬 Skalierung: wie viel Budget braucht ein größeres Problem?")
if st.button("Stopps von 20 bis 200 durchfahren (dauert etwa 90 Sekunden)", key="scaling_start"):
    st.session_state["scaling_on"] = True
if st.session_state.get("scaling_on"):
    with st.spinner("Rechne 6 Größen × 2 Budgetregeln × 5 Instanzen × 3 Ketten..."):
        sc = _scaling(replace(base_sweep, n=C.DEFAULT_N))
    st.plotly_chart(build_scaling(sc), width="stretch", key="scaling_chart")
    st.caption("Mittel über 5 feste Instanzen × 3 Ketten (Einstellungen wie in der Seitenleiste außer Stopps und Budget). Bei festem 200-Tausend-Budget bleibt Parallel Tempering bis 60 Stopps nah an SA dran (nicht monoton: bei n=40 knapp schlechter, bei n=60 wieder vorn), "
               "verliert aber ab 100 Stopps DEUTLICH (6.20 % gegen 3.08 % bei n=100) - derselbe Budget-Teilungs-Effekt wie bei kleinem Budget, hier durch die Instanzgröße ausgelöst. Simulated Annealing braucht keinen Ketten-Split und skaliert strukturell günstiger.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Genug Budget für alle R Ketten** | Das Budget wird durch R geteilt - bei kleinem Budget (10 Tausend: 32.05 % gegen 9.01 % für SA) oder großen Instanzen (200 Stopps: 23.55 % gegen 7.87 %) bekommt jede Kette zu wenig, um überhaupt anzukommen. | Weniger Replikate R bei knappem Budget (eigener Regler) |
| **Eine ungefähr richtig platzierte Leiter** | Eine schlecht zentrierte oder zu breite Leiter kostet spürbar (3-6x schlechter als kalibriert) - Parallel Tempering ist NICHT tuningfrei. Aber selbst die schlechtesten getesteten Leitern bleiben weit vor Simulated Annealings eigenen Fehlkalibrierungen (14.96 % / 31.57 % bei zu heiß). | Robuster, aber nicht automatisch - eine grobe Voreinschätzung der Temperaturspanne bleibt nötig |
| **Nur Metropolis-Annahme** | Die Tausch-Formel setzt eine wohldefinierte Gleichgewichtsverteilung je Kette voraus - Threshold Accepting, Great Deluge und Late Acceptance Hill Climbing (die anderen drei Regeln der Simulated-Annealing-Demo) haben keine, die Formel wäre für sie nicht herleitbar. | (kein Nachfolger nötig - literaturtreue Einschränkung) |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Problem.** Kürzeste Rundtour über $N = n+1$ Knoten mit euklidischen Entfernungen $d_{ij}$; $L(\pi)$ ist die Länge einer Tour $\pi$.

**Metropolis bei fester Temperatur $T$.** Ein Vorschlag (2-opt) mit Längenänderung $\Delta$ wird angenommen mit
$p_{\text{accept}} = \min(1, \exp(-\Delta/T))$ - dieselbe Regel wie bei Simulated Annealing, hier aber bei
KONSTANTEM $T$ statt einem Abkühlplan. Die stationäre Verteilung dieser Kette ist die Boltzmann-Verteilung
$\pi_T(\text{Tour}) \propto \exp(-L(\text{Tour})/T)$.

**Tausch zweier Ketten.** Für Ketten bei $T_i, T_j$ mit aktuellen Längen $L_i, L_j$ wird der Tausch ihrer Touren
angenommen mit
$$p_{\text{tausch}} = \min\!\left(1,\ \exp\!\left[\left(\tfrac{1}{T_i}-\tfrac{1}{T_j}\right)(L_i-L_j)\right]\right)$$
Diese Formel ist die Metropolis-Hastings-Akzeptanzrate für den Tausch-Zug im PRODUKT-Zustandsraum
$(\text{Tour}_1, ..., \text{Tour}_R)$, dessen Ziel-Verteilung $\prod_i \pi_{T_i}$ ist - sie erhält also die
gemeinsame Verteilung exakt (Detailed Balance auf Ketten-Paar-Ebene), und jede Kette bleibt bei ihrer EIGENEN
Temperatur im Gleichgewicht, unabhängig davon, wie oft sie tauscht.

**Kennzahl.** Abstand zur Schranke $= 100 \cdot (L - w)/w$ mit der 1-Baum-Schranke $w$. Vergleichsgröße bei
gleichem Budget: kalibriertes Simulated Annealing (geometrischer Plan, T0=0.5, T_end=0.1) und Hill Climbing mit
Neustarts.

**Grenzen.** (1) Nur Metropolis (siehe oben). (2) Das Budget wird durch R geteilt - ein struktureller Nachteil bei
kleinem Budget oder großen Instanzen. (3) Eine grobe Voreinschätzung der Temperaturspanne bleibt nötig.

**Literatur.** Swendsen, R. H., & Wang, J. S. (1986). *Replica Monte Carlo Simulation of Spin-Glasses.* Physical
Review Letters, 57(21), 2607 (Ursprung: Teilaustausch). Geyer, C. J. (1991). *Markov Chain Monte Carlo Maximum
Likelihood.* Computing Science and Statistics: Proceedings of the 23rd Symposium on the Interface (vollständiger
Austausch, gilt als Ursprung von Parallel Tempering in der statistischen Physik).

Implementiert in `pt_algorithm.py` (der Metropolis-Kern `_chain_segment`, die Leiter `temperature_ladder`, die
Tausch-Schleife `parallel_tempering`, die SA-Vergleichsgröße `sa_baseline` - alle über denselben Baustein),
`pt_tour.py` (Nachbarschaften, Abstieg, Schranke - aus der Hill-Climbing-Demo), `pt_scenario.py` (Instanzen),
`pt_evaluation.py` (Kennzahlen, Sweeps, Experimente, Urteil).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
