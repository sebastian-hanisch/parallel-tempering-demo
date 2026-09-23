"""Presets: Vollständigkeit, gültige Werte, Urteile über mehrere Instanzen und Ketten (Bänder), Permalink-Konstanten."""

import pytest

import pt_constants as C
import pt_evaluation as ev
import pt_presets as P


def _settings(p, seed=None, chain_seed=None):
    return ev.Settings(n=p["n"], cluster_share=p["ballung"], seed=p["seed"] if seed is None else seed, n_replicas=p["n_replicas"],
                        t_min=p["t_min"], t_max=p["t_max"], swap_interval=p["swap_interval"], swaps_enabled=p["swaps_enabled"],
                        budget=p["budget"], chain_seed=p["chain_seed"] if chain_seed is None else chain_seed)


def test_every_preset_has_help_bands_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_EXPECTED_BANDS) and len(C.PRESETS) == 6
    for name, p in C.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and C.PRESET_HELP[name]


def test_preset_values_are_valid_and_match_the_setting_specs():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and (p["n"] - C.N_MIN) % C.N_STEP == 0
        assert C.BALLUNG_MIN <= p["ballung"] <= C.BALLUNG_MAX and p["ballung"] % C.BALLUNG_STEP == 0
        assert C.R_MIN <= p["n_replicas"] <= C.R_MAX
        assert C.T_MIN_MIN <= p["t_min"] <= C.T_MIN_MAX and C.T_MAX_MIN <= p["t_max"] <= C.T_MAX_MAX
        assert p["t_min"] < p["t_max"]
        assert p["swap_interval"] in C.SWAP_INTERVAL_OPTIONS and p["budget"] in C.BUDGETS
        for key, state_key in P.PRESET_KEYS.items():
            P.SETTING_SPECS[state_key].caster(p[key])


def test_default_preset_equals_the_default_settings():
    assert _settings(C.PRESETS["Standardfall (Voreinstellung)"]) == ev.Settings()


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_verdicts_stay_in_their_bands_over_instances_and_chains(name):
    p = C.PRESETS[name]
    seeds = range(2) if p["n"] >= 150 else range(5)
    seen = {ev.verdict(ev.analyse(_settings(p, seed=seed, chain_seed=ch))) for seed in seeds for ch in (0, 1)}
    assert seen <= C.PRESET_EXPECTED_BANDS[name], seen
    assert ev.verdict(ev.analyse(_settings(p))) in C.PRESET_EXPECTED_BANDS[name]


def test_no_swap_preset_disables_swaps_and_others_keep_them_on():
    assert C.PRESETS["Ohne Tausch (Kontrolle)"]["swaps_enabled"] is False
    assert all(p["swaps_enabled"] for name, p in C.PRESETS.items() if name != "Ohne Tausch (Kontrolle)")


def test_wide_ladder_preset_spans_the_sa_failure_extremes():
    p = C.PRESETS["Weite Leiter (deckt SA-Extreme ab)"]
    assert p["t_min"] <= 0.05 and p["t_max"] >= 0.4                          # deckt SAs eigene "zu kalt"/"zu heiss"-Presets ab


def test_bounds_and_snapping_constants():
    assert P.bounds("n_slider") == (C.N_MIN, C.N_MAX) and P.bounds("seed_input") == (0, C.SEED_MAX)
    assert P.STEPS == {"n_slider": C.N_STEP, "ballung_slider": C.BALLUNG_STEP}
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)
    assert C.DEFAULT_BUDGET in C.BUDGETS and C.DEFAULT_SWAP_INTERVAL in C.SWAP_INTERVAL_OPTIONS


def test_bool_caster_accepts_common_spellings():
    assert P._bool("true") is True and P._bool("False") is False
    assert P._bool("1") is True and P._bool("0") is False
    assert P._bool(True) is True and P._bool(False) is False
