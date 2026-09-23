"""Plotly-Abbildungen der Parallel-Tempering-Demo: Karten, Verlauf, die Leiter-Heatmap (PT-eigen), Tausch-
Annahmequote je Nachbarpaar, Budget-/Größen-Vergleich, Sweeps, Streuung, Skalierung.
Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen."""

import numpy as np
import plotly.graph_objects as go

import pt_constants as C

TOUR_COLOR = "#4c78a8"
PT_COLOR = "#54a24b"
SA_COLOR = "#f58518"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.1), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _map_layout(fig, height=430):
    fig.update_xaxes(range=[-3, C.AREA + 3], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(range=[-3, C.AREA + 3], showgrid=False, zeroline=False, showticklabels=False)
    return _base(fig, height)


def _line_trace(xy, edges, color, name, dash=None, width=2.5, showlegend=True):
    x, y = [], []
    for a, b in edges:
        x += [xy[a, 0], xy[b, 0], None]
        y += [xy[a, 1], xy[b, 1], None]
    return go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=width, dash=dash), name=name, hoverinfo="skip", showlegend=showlegend)


def _tour_edges_list(tour):
    t = np.asarray(tour)
    return list(zip(t.tolist(), np.roll(t, -1).tolist()))


def build_instance(xy):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xy[1:, 0], y=xy[1:, 1], mode="markers", marker=dict(size=7, color=TOUR_COLOR, line=dict(width=1, color="white")), name="Stopps", hovertemplate="Stopp %{customdata}<extra></extra>", customdata=np.arange(1, len(xy))))
    fig.add_trace(go.Scatter(x=[xy[0, 0]], y=[xy[0, 1]], mode="markers", marker=dict(size=14, symbol="star", color="#f58518", line=dict(width=1, color="white")), name="Depot"))
    return _map_layout(fig)


def build_tour(xy, tour, ghost=None):
    fig = go.Figure()
    if ghost is not None:
        fig.add_trace(_line_trace(xy, _tour_edges_list(ghost), "#c9d6e6", "vorige Tour", width=6))
    fig.add_trace(_line_trace(xy, _tour_edges_list(tour), TOUR_COLOR, "aktuelle Tour"))
    fig.add_trace(go.Scatter(x=xy[1:, 0], y=xy[1:, 1], mode="markers", marker=dict(size=6, color="white", line=dict(width=1.5, color=TOUR_COLOR)), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[xy[0, 0]], y=[xy[0, 1]], mode="markers", marker=dict(size=14, symbol="star", color="#f58518", line=dict(width=1, color="white")), name="Depot"))
    return _map_layout(fig)


def build_trace(rounds, lengths, bound, label, color):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rounds, y=lengths, mode="lines+markers", line=dict(color=color, width=2.5), marker=dict(size=5), name=label))
    fig.add_hline(y=bound, line=dict(color="#7f7f7f", dash="dot"), annotation_text="untere Schranke", annotation_position="bottom right")
    fig.update_xaxes(title_text="Runde (Vorschläge je Kette zwischen Tausch-Versuchen)")
    fig.update_yaxes(title_text="Länge (km)")
    return _base(fig, 320)


def build_ladder_heatmap(replica_lengths, temperatures, bound, x=None):
    """Die PT-eigene Ansicht: x = Runde, y = Replikat (nach Temperatur sortiert, kälteste unten), Farbe = Abstand
    zur Schranke der aktuellen Tour dieser Kette. Zeigt sichtbar, wie gute Touren zwischen den Temperaturen
    "nach unten sickern", wenn der Austausch aktiv ist. `x`: die tatsächlichen Rundennummern, falls die Zeilen
    von `replica_lengths` vorher auf eine handhabbare Zahl ausgedünnt wurden (siehe app.py). Die Farbskala ist
    bei 30 % gedeckelt (`zmax`): die Startrunde liegt oft bei mehreren hundert Prozent (zufällige Starttour) und
    würde sonst die gesamte interessante Konvergenz darunter in derselben Farbe verstecken."""
    gaps = 100 * (replica_lengths - bound) / bound                          # (angezeigte Runden, R)
    labels = [f"T={t:.3f}" for t in temperatures]
    xs = list(range(gaps.shape[0])) if x is None else list(x)
    fig = go.Figure(go.Heatmap(z=gaps.T, x=xs, y=labels, colorscale="YlGnBu_r", zmin=0, zmax=30,
                                colorbar=dict(title="Abstand (%, gedeckelt bei 30)"), hovertemplate="Runde %{x}<br>%{y}<br>%{z:.2f} %<extra></extra>"))
    fig.update_xaxes(title_text="Runde")
    fig.update_yaxes(title_text="Kette (kälteste unten)")
    return _base(fig, 340)


def build_swap_bar(swap_attempts, swap_accepts, temperatures):
    labels = [f"{temperatures[i]:.3f}↔{temperatures[i + 1]:.3f}" for i in range(len(temperatures) - 1)]
    rate = np.divide(swap_accepts, swap_attempts, out=np.zeros_like(swap_accepts, dtype=float), where=swap_attempts > 0)
    fig = go.Figure(go.Bar(x=labels, y=rate * 100, marker_color=PT_COLOR))
    fig.update_xaxes(title_text="Nachbarpaar (Temperaturen)")
    fig.update_yaxes(title_text="Tausch-Annahmequote (%)", range=[0, 100])
    return _base(fig, 280)


def build_budget(rows):
    xs = [r["value"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["gap"] for r in rows], mode="lines+markers", line=dict(color=PT_COLOR, width=2.5), name="Parallel Tempering"))
    fig.add_trace(go.Scatter(x=xs, y=[r["sa"] for r in rows], mode="lines+markers", line=dict(color=SA_COLOR, width=2.5), name="Simulated Annealing (getunt)"))
    fig.update_xaxes(title_text="Budget (bewertete Nachbarn)", type="log")
    fig.update_yaxes(title_text="Abstand zur Schranke (%)")
    fig.update_layout(legend=dict(orientation="h", y=-0.3))
    return _base(fig, 380)


def build_sweep(rows, param_label):
    xs = [r["value"] for r in rows]
    fig = go.Figure()
    upper = [r["gap"] + r["gap_sd"] for r in rows]
    lower = [max(0.0, r["gap"] - r["gap_sd"]) for r in rows]
    fig.add_trace(go.Scatter(x=xs + xs[::-1], y=upper + lower[::-1], fill="toself", fillcolor="rgba(84,162,75,0.15)", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["gap"] for r in rows], mode="lines+markers", line=dict(color=PT_COLOR, width=2.5), name="Parallel Tempering"))
    fig.add_trace(go.Scatter(x=xs, y=[r["sa"] for r in rows], mode="lines+markers", line=dict(color=SA_COLOR, width=2, dash="dot"), name="Simulated Annealing (getunt)"))
    fig.update_xaxes(title_text=param_label)
    fig.update_yaxes(title_text="Abstand zur Schranke (%)")
    fig.update_layout(legend=dict(orientation="h", y=-0.3))
    return _base(fig, 380)


def build_spread(pt, sa, pt_label="Parallel Tempering", sa_label="Simulated Annealing (getunt)"):
    fig = go.Figure()
    fig.add_trace(go.Box(y=pt, name=pt_label, marker_color=PT_COLOR, boxmean=True))
    fig.add_trace(go.Box(y=sa, name=sa_label, marker_color=SA_COLOR, boxmean=True))
    fig.update_yaxes(title_text="Abstand zur Schranke (%)")
    return _base(fig, 360)


def build_scaling(blocks):
    """`blocks`: [{"label": ..., "rows": [...]}, ...] (siehe pt_evaluation.scaling_table)."""
    fig = go.Figure()
    colors = (PT_COLOR, SA_COLOR)
    for block, color in zip(blocks, colors):
        label, rows = block["label"], block["rows"]
        xs = [r["value"] for r in rows]
        fig.add_trace(go.Scatter(x=xs, y=[r["gap"] for r in rows], mode="lines+markers", line=dict(color=color, width=2.5), name=label))
    fig.update_xaxes(title_text="Stopps")
    fig.update_yaxes(title_text="Abstand zur Schranke (%)")
    fig.update_layout(legend=dict(orientation="h", y=-0.3))
    return _base(fig, 360)
