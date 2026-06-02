"""End-to-end tests for the Excel auditor / reader / writer on real workbooks.

openpyxl does not evaluate formulas, so these tests use hardcoded values for
the checks that rely on computed numbers (balance-sheet totals, EBITDA, error
strings) — exactly what the auditor sees when it loads a workbook with
``data_only=True``.
"""
import pytest

openpyxl = pytest.importorskip("openpyxl")

from neuromorphic.domains.investment_banking.excel.auditor import ExcelAuditor
from neuromorphic.domains.investment_banking.excel.reader import ExcelReader
from neuromorphic.domains.investment_banking.models.dcf import DCFModel, DCFInputs


def _flawed_workbook(path):
    wb = openpyxl.Workbook()

    # Balance sheet that does not balance
    bs = wb.active
    bs.title = "Balance Sheet"
    bs["A1"] = "Total Assets";                  bs["B1"] = 1000.0
    bs["A2"] = "Total Liabilities and Equity";  bs["B2"] = 950.0   # 50 off

    # Projections with a broken formula and a negative EBITDA
    proj = wb.create_sheet("Projections")
    proj["A1"] = "EBITDA";   proj["B1"] = 100.0;  proj["C1"] = -20.0   # negative
    proj["A2"] = "Margin";   proj["B2"] = "#DIV/0!"                    # error string

    wb.save(path)


def test_auditor_detects_known_errors(tmp_path):
    path = tmp_path / "model.xlsx"
    _flawed_workbook(path)

    report = ExcelAuditor().audit(str(path))
    cats = {i.category for i in report.issues}

    assert "balance_sheet" in cats           # 1000 != 950
    assert "formula_error" in cats           # #DIV/0!
    assert any("ebitda" in i.description.lower() and "negative" in i.description.lower()
               for i in report.issues)
    assert report.n_critical >= 1


def test_auditor_auto_corrects_div0(tmp_path):
    path = tmp_path / "model.xlsx"
    out = tmp_path / "model_corrected.xlsx"
    _flawed_workbook(path)

    report = ExcelAuditor().audit_and_correct(str(path), str(out))
    assert report.n_auto_fixed >= 1
    assert report.corrected_path == str(out)

    wb = openpyxl.load_workbook(str(out))
    # The #DIV/0! cell should now hold a numeric 0
    assert wb["Projections"]["B2"].value == 0


def test_reader_extracts_assumptions(tmp_path):
    path = tmp_path / "assumptions.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Assumptions"
    ws["A1"] = "WACC";            ws["B1"] = 0.10
    ws["A2"] = "Terminal Growth"; ws["B2"] = 0.025
    wb.save(path)

    got = ExcelReader().extract_assumptions(str(path))
    assert got.get("wacc") == pytest.approx(0.10)
    assert got.get("terminal_growth") == pytest.approx(0.025)


def test_writer_roundtrips_dcf(tmp_path):
    from neuromorphic.domains.investment_banking.excel.writer import ExcelWriter
    path = tmp_path / "dcf.xlsx"
    result = DCFModel().compute(DCFInputs(
        ebitda=20e6, revenue=100e6, revenue_growth=0.05, ebitda_margin=0.20,
        wacc=0.10, terminal_growth=0.02, tax_rate=0.25, capex_pct=0.04,
        nwc_pct=0.02, depreciation_pct=0.04, projection_years=5))
    ExcelWriter().write_dcf(result, str(path))

    assert path.exists()
    wb = openpyxl.load_workbook(str(path))
    assert "DCF" in wb.sheetnames
