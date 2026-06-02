"""Hand-checked tests for the credit / leverage model."""
import pytest

from neuromorphic.domains.investment_banking.models.credit import CreditModel, CreditInputs


def _inputs(**over):
    base = dict(
        ebitda=100.0, revenue=400.0, total_debt=400.0, cash=50.0,
        interest_expense=28.0, capex=20.0, tax_rate=0.25, net_income=40.0,
    )
    base.update(over)
    return CreditInputs(**base)


def test_leverage_and_coverage_exact():
    r = CreditModel().compute(_inputs())
    assert r.leverage_total == pytest.approx(4.0)             # 400 / 100
    assert r.leverage_net == pytest.approx(3.5)               # (400-50) / 100
    assert r.interest_coverage == pytest.approx(100.0 / 28.0)
    assert r.net_debt == pytest.approx(350.0)


def test_implied_rating_monotonic():
    low = CreditModel().compute(_inputs(total_debt=150.0))    # 1.5x
    high = CreditModel().compute(_inputs(total_debt=700.0))   # 7.0x
    ratings = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "D"]
    assert ratings.index(low.implied_rating) < ratings.index(high.implied_rating)


def test_covenant_breach_flagged():
    ok = CreditModel().compute(_inputs(total_debt=400.0))     # 4.0x < 6x
    breach = CreditModel().compute(_inputs(total_debt=800.0))  # 8.0x > 6x
    assert ok.covenants_ok is True
    assert breach.covenants_ok is False


def test_maturity_schedule_changes_dscr():
    base = CreditModel().compute(_inputs())
    with_sched = CreditModel().compute(
        _inputs(debt_maturity_schedule={1: 100.0, 2: 50.0}))
    # A larger near-term repayment than the 5% default lowers DSCR.
    assert with_sched.dscr < base.dscr


@pytest.mark.parametrize("bad", [dict(ebitda=0.0), dict(total_debt=-1.0)])
def test_validation_rejects_nonsense(bad):
    with pytest.raises(ValueError):
        CreditModel().compute(_inputs(**bad))
