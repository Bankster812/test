"""Smoke tests: the package and every financial model import and instantiate."""


def test_package_imports():
    import neuromorphic.domains.investment_banking as ib
    assert hasattr(ib, "IBBrain")


def test_all_models_instantiate():
    from neuromorphic.domains.investment_banking.models import (
        DCFModel, LBOModel, MergerModel, CompsModel, PrecedentsModel, CreditModel,
    )
    for cls in (DCFModel, LBOModel, MergerModel, CompsModel, PrecedentsModel, CreditModel):
        assert cls() is not None
