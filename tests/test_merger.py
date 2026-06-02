"""Hand-checked tests for the merger / accretion-dilution model."""
import pytest

from neuromorphic.domains.investment_banking.models.merger import MergerModel, MergerInputs


def _inputs(**over):
    base = dict(
        acq_net_income=100e6, acq_shares=100e6, acq_share_price=20.0, acq_pe=20.0,
        tgt_net_income=20e6, tgt_share_price=10.0, tgt_shares=10e6,
        premium_pct=0.0, cash_pct=0.0, stock_pct=1.0,
        annual_synergies=0.0, one_time_costs=0.0, tax_rate=0.25,
    )
    base.update(over)
    return MergerInputs(**base)


def test_all_stock_accretion_closed_form():
    # Acquirer P/E 20 buying a target at deal P/E 5 in stock → accretive.
    r = MergerModel().compute(_inputs())
    # offer value = 10 * 10M = 100M, shares issued = 100M/20 = 5M
    assert r.shares_issued == pytest.approx(5e6)
    assert r.combined_shares == pytest.approx(105e6)
    # combined NI = 120M (no debt, no synergies) → EPS = 120/105
    assert r.pro_forma_eps_yr1 == pytest.approx(120e6 / 105e6)
    assert r.acq_standalone_eps == pytest.approx(1.0)
    assert r.accretion_dilution_pct > 0          # accretive
    assert r.accretion_dilution_pct == pytest.approx((120e6 / 105e6 - 1.0))


def test_premium_stored_and_offer_price():
    r = MergerModel().compute(_inputs(premium_pct=0.30))
    assert r.premium_pct == pytest.approx(0.30)
    assert r.offer_price == pytest.approx(10.0 * 1.30)


def test_exchange_ratio_none_for_all_cash():
    r = MergerModel().compute(_inputs(cash_pct=1.0, stock_pct=0.0))
    assert r.exchange_ratio is None
    # __str__ must still render fully (regression: it used to blank out)
    assert "Offer price" in str(r)


def test_str_renders_for_stock_deal():
    r = MergerModel().compute(_inputs())
    assert r.exchange_ratio is not None
    assert "Exchange ratio" in str(r)


def test_synergies_improve_accretion():
    no_syn = MergerModel().compute(_inputs(cash_pct=1.0, stock_pct=0.0))
    with_syn = MergerModel().compute(
        _inputs(cash_pct=1.0, stock_pct=0.0, annual_synergies=20e6))
    assert (with_syn.accretion_dilution_with_synergies_pct
            > no_syn.accretion_dilution_with_synergies_pct)


def test_consideration_split_must_sum_to_one():
    with pytest.raises(ValueError):
        MergerModel().compute(_inputs(cash_pct=0.6, stock_pct=0.6))
