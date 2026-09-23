"""AppTest-Rauchtests: Voreinstellung, jedes Preset, jeder Schritt, Randwerte, Würfel-Knöpfe, Permalink-Grenzen, Experimente auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import pt_constants as C
import pt_evaluation as ev

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(pt_step=1, **state):
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    if pt_step != 1:
        at.select_slider(key="pt_step").set_value(pt_step).run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_default_run_has_no_exception_and_shows_the_measured_default():
    at = _run()
    _ok(at)
    assert _metric(at, "Parallel Tempering: beste Tour") == "1.3 %"
    assert _metric(at, "Simulated Annealing (getunt)") == "1.8 %"
    assert _metric(at, "Hill Climbing mit Neustarts") == "3.9 %"
    assert any("gewinnt" in s.value for s in at.success)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    assert at.session_state["r_slider"] == p["n_replicas"] and at.session_state["budget_select"] == p["budget"] and at.session_state["n_slider"] == p["n"]
    assert at.session_state["swaps_toggle"] == p["swaps_enabled"]
    assert at.metric


@pytest.mark.parametrize("step", [1, 2, 3])
def test_every_step_runs(step):
    at = _run(n_slider=20, budget_select=25000, pt_step=step)
    _ok(at)
    assert at.get("plotly_chart") and at.session_state["pt_step"] == step


def test_step_two_without_swaps_shows_no_swap_bar_but_still_runs():
    at = _run(n_slider=20, budget_select=25000, pt_step=2, swaps_toggle=False)
    _ok(at)
    assert any("ausgeschaltet" in s.value for s in at.info)


def test_dice_buttons_change_the_seeds():
    at = _run(budget_select=10000)
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Instanz generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old
    old_c = at.session_state["chain_seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Kette würfeln").click().run()
    _ok(at)
    assert at.session_state["chain_seed_input"] != old_c


@pytest.mark.parametrize("kw", [dict(n_slider=200, budget_select=50000), dict(n_slider=10, ballung_slider=100, budget_select=10000),
                                 dict(r_slider=C.R_MIN, budget_select=25000), dict(r_slider=C.R_MAX, budget_select=25000),
                                 dict(t_min_slider=C.T_MIN_MAX, t_max_slider=C.T_MIN_MAX, budget_select=25000)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def test_permalink_values_are_clamped_and_snapped():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["n"] = "9999"
    at.query_params["ballung"] = "40"
    at.query_params["r"] = "9999"
    at.query_params["budget"] = "12345"
    at.query_params["swaps"] = "false"
    at.run()
    _ok(at)
    assert at.session_state["n_slider"] == C.N_MAX and at.session_state["ballung_slider"] == 50
    assert at.session_state["r_slider"] == C.R_MAX and at.session_state["budget_select"] == C.DEFAULT_BUDGET
    assert at.session_state["swaps_toggle"] is False


def test_sweeps_run_on_demand():
    at = _run(n_slider=10, budget_select=10000)
    at.selectbox(key="sweep_select").set_value("n").run()
    next(b for b in at.button if b.key == "sweep_start").click().run()
    _ok(at)
    assert at.get("plotly_chart")


def test_experiments_run_on_demand(monkeypatch):
    monkeypatch.setitem(ev.SWEEP_VALUES, "budget", (2000, 5000))
    monkeypatch.setattr(C, "SCALING_N", (10, 20))
    monkeypatch.setattr(ev, "SCALING_POLICIES", (("Budget 4 Tausend", lambda n: 4000), ("Budget 300 · Stopps", lambda n: 300 * n)))
    at = _run(n_slider=10, budget_select=10000)
    for key, flag in (("budget_start", "budget_on"), ("spread_start", "spread_on"), ("scaling_start", "scaling_on")):
        next(b for b in at.button if b.key == key).click().run()
        _ok(at)
        assert at.session_state[flag]


def test_footer_and_grenzen_are_present():
    at = _run(budget_select=10000)
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    assert any("Genug Budget für alle R Ketten" in m.value for m in at.markdown)
