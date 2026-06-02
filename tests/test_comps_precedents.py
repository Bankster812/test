"""Hand-checked tests for comps and precedent-transaction analysis."""
import pytest

from neuromorphic.domains.investment_banking.models.comps import (
    CompsModel, CompsInputs, CompanyData,
)
from neuromorphic.domains.investment_banking.models.precedents import (
    PrecedentsModel, PrecedentsInputs, TransactionData,
)


# --- Comps -------------------------------------------------------------------

def _peers():
    # EV/EBITDA multiples are exactly 10x, 12x, 14x → median 12x, mean 12x
    return [
        CompanyData("A", ev=1000.0, revenue=500.0, ebitda=100.0, ebit=80.0, net_income=50.0),
        CompanyData("B", ev=1200.0, revenue=600.0, ebitda=100.0, ebit=80.0, net_income=50.0),
        CompanyData("C", ev=1400.0, revenue=700.0, ebitda=100.0, ebit=80.0, net_income=50.0),
    ]


def test_comps_multiples_stats_exact():
    target = CompanyData("T", ev=0, revenue=550.0, ebitda=100.0, ebit=80.0, net_income=50.0)
    r = CompsModel().compute(CompsInputs(target=target, comparables=_peers()))
    s = r.stats["ev_ebitda"]
    assert s["median"] == pytest.approx(12.0)
    assert s["mean"] == pytest.approx(12.0)


def test_comps_implied_ev_nonnegative():
    target = CompanyData("T", ev=0, revenue=550.0, ebitda=100.0, ebit=80.0, net_income=50.0)
    r = CompsModel().compute(CompsInputs(target=target, comparables=_peers()))
    lo, hi = r.implied_ev_range
    assert lo >= 0.0 and hi >= lo


def test_comps_requires_comparables():
    target = CompanyData("T", ev=0, revenue=550.0, ebitda=100.0, ebit=80.0, net_income=50.0)
    with pytest.raises(ValueError):
        CompsModel().compute(CompsInputs(target=target, comparables=[]))


# --- Precedents --------------------------------------------------------------

def _txns():
    return [
        TransactionData("D1", 2022, "tech", ev=1000.0, revenue=500.0, ebitda=100.0, premium_pct=0.30),
        TransactionData("D2", 2021, "tech", ev=1200.0, revenue=600.0, ebitda=100.0, premium_pct=0.40),
        TransactionData("D3", 2019, "tech", ev=1400.0, revenue=700.0, ebitda=100.0, premium_pct=0.20),
    ]


def test_precedents_median_multiple_exact():
    r = PrecedentsModel().compute(PrecedentsInputs(
        target_ebitda=100.0, target_revenue=550.0, target_sector="tech",
        transactions=_txns(), year_filter=2010))
    assert r.multiples["ev_ebitda"]["median"] == pytest.approx(12.0)


def test_precedents_year_filter_applies():
    # Only deals from 2021+ → D1 (30%) and D2 (40%); mean premium 35%.
    r = PrecedentsModel().compute(PrecedentsInputs(
        target_ebitda=100.0, target_revenue=550.0, target_sector="tech",
        transactions=_txns(), year_filter=2021))
    assert len(r.filtered_transactions) == 2
    assert r.mean_premium == pytest.approx(0.35)


def test_precedents_requires_transactions():
    with pytest.raises(ValueError):
        PrecedentsModel().compute(PrecedentsInputs(
            target_ebitda=100.0, target_revenue=550.0, target_sector="tech",
            transactions=[]))
