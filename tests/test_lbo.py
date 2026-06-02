"""Hand-checked tests for the LBO model."""
import pytest

from neuromorphic.domains.investment_banking.models.lbo import LBOModel, LBOInputs


def _inputs(**over):
    base = dict(
        ebitda=100e6, revenue=400e6, entry_ev_ebitda=10.0, revenue_growth=0.05,
        ebitda_margin_exit=0.25, leverage_ratio=5.0, senior_debt_rate=0.07,
        hold_period=5, exit_multiple=10.0, tax_rate=0.25, capex_pct=0.04,
        nwc_pct=0.02, depreciation_pct=0.04, debt_paydown_rate=1.0,
    )
    base.update(over)
    return LBOInputs(**base)


def test_entry_structure():
    r = LBOModel().compute(_inputs())
    assert r.entry_enterprise_value == pytest.approx(100e6 * 10.0)   # EV = EBITDA * entry x
    assert r.total_debt == pytest.approx(100e6 * 5.0)                # debt = EBITDA * leverage
    assert r.equity_contribution == pytest.approx(r.entry_enterprise_value - r.total_debt)


def test_moic_definition():
    r = LBOModel().compute(_inputs())
    assert r.moic == pytest.approx(r.exit_equity_value / r.equity_contribution)


def test_irr_matches_moic_identity():
    # With a single entry outflow and single exit inflow (no interim cash),
    # IRR must equal MOIC^(1/years) - 1.
    inp = _inputs()
    r = LBOModel().compute(inp)
    expected = r.moic ** (1.0 / inp.hold_period) - 1.0
    assert r.irr == pytest.approx(expected, rel=1e-3)


def test_zero_leverage_is_all_equity():
    r = LBOModel().compute(_inputs(leverage_ratio=0.0))
    assert r.total_debt == 0.0
    assert all(d == 0.0 for d in r.debt_schedule)
    assert r.equity_contribution == pytest.approx(r.entry_enterprise_value)


def test_leverage_capped_at_enterprise_value():
    # Asking for more debt than the purchase price is capped (no negative equity).
    r = LBOModel().compute(_inputs(leverage_ratio=50.0))
    assert r.total_debt <= r.entry_enterprise_value + 1.0
    assert r.equity_contribution >= 0.0


def test_debt_paydown_reduces_balance():
    r = LBOModel().compute(_inputs(debt_paydown_rate=1.0))
    assert r.debt_at_exit <= r.total_debt


@pytest.mark.parametrize("bad", [
    dict(hold_period=0),
    dict(ebitda=0.0),
    dict(debt_paydown_rate=1.5),
    dict(exit_multiple=0.0),
])
def test_validation_rejects_nonsense(bad):
    with pytest.raises(ValueError):
        LBOModel().compute(_inputs(**bad))
