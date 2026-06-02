"""Hand-checked tests for the DCF model."""
import numpy as np
import pytest

from neuromorphic.domains.investment_banking.models.dcf import DCFModel, DCFInputs


def _flat_inputs(**over):
    """A zero-growth, no-capex/nwc/D&A company → constant FCF, easy to verify."""
    base = dict(
        ebitda=20e6, revenue=100e6, revenue_growth=0.0, ebitda_margin=0.20,
        wacc=0.10, terminal_growth=0.02, tax_rate=0.25,
        capex_pct=0.0, nwc_pct=0.0, depreciation_pct=0.0,
        projection_years=5, net_debt=0.0,
    )
    base.update(over)
    return DCFInputs(**base)


def test_flat_company_matches_closed_form():
    inp = _flat_inputs()
    r = DCFModel().compute(inp)

    # FCF = revenue*margin*(1-tax), constant every year
    fcf = 100e6 * 0.20 * (1 - 0.25)
    assert r.projected_fcf[0] == pytest.approx(fcf)
    assert all(f == pytest.approx(fcf) for f in r.projected_fcf)

    # Independent closed-form EV
    w, g, n = 0.10, 0.02, 5
    pv_fcfs = sum(fcf / (1 + w) ** t for t in range(1, n + 1))
    tv = fcf * (1 + g) / (w - g)
    pv_tv = tv / (1 + w) ** n
    expected_ev = pv_fcfs + pv_tv
    assert r.enterprise_value == pytest.approx(expected_ev, rel=1e-9)
    assert r.terminal_value == pytest.approx(tv, rel=1e-9)


def test_equity_bridge():
    # Debt-free company: equity value == enterprise value
    r0 = DCFModel().compute(_flat_inputs(net_debt=0.0))
    assert r0.equity_value == pytest.approx(r0.enterprise_value)

    # With net debt the bridge subtracts it exactly
    r1 = DCFModel().compute(_flat_inputs(net_debt=40e6))
    assert r1.equity_value == pytest.approx(r1.enterprise_value - 40e6)


def test_per_share_only_when_shares_given():
    assert DCFModel().compute(_flat_inputs()).price_per_share is None
    r = DCFModel().compute(_flat_inputs(net_debt=0.0, shares_outstanding=10e6))
    assert r.price_per_share == pytest.approx(r.equity_value / 10e6)


def test_mid_year_raises_value_vs_full_year():
    full = DCFModel().compute(_flat_inputs(mid_year_convention=False))
    mid = DCFModel().compute(_flat_inputs(mid_year_convention=True))
    # Discounting half a year less always increases PV
    assert mid.enterprise_value > full.enterprise_value


def test_exit_multiple_terminal_value():
    inp = _flat_inputs(terminal_method="exit_multiple", terminal_exit_multiple=9.0)
    r = DCFModel().compute(inp)
    terminal_ebitda = r.projected_ebitda[-1]
    assert r.terminal_value == pytest.approx(9.0 * terminal_ebitda)
    assert r.tv_exit_multiple == pytest.approx(9.0 * terminal_ebitda)


def test_implied_exit_multiple_consistent():
    r = DCFModel().compute(_flat_inputs())
    terminal_ebitda = r.projected_ebitda[-1]
    assert r.implied_exit_multiple == pytest.approx(r.tv_perpetuity / terminal_ebitda)


@pytest.mark.parametrize("bad", [
    dict(terminal_growth=0.12, wacc=0.10),   # g >= wacc
    dict(wacc=-0.01),                        # negative discount rate
    dict(tax_rate=1.0),                      # tax rate out of range
    dict(projection_years=0),                # no projection
])
def test_validation_rejects_nonsense(bad):
    with pytest.raises(ValueError):
        DCFModel().compute(_flat_inputs(**bad))


def test_sensitivity_table_shape_and_finite():
    r = DCFModel().compute(_flat_inputs())
    assert r.sensitivity_table.shape == (5, 5)
    # Centre cell (no perturbation) equals the headline EV
    assert r.sensitivity_table[2, 2] == pytest.approx(r.enterprise_value, rel=1e-5)
