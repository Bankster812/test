"""
IBBrain — Investment Banking Neuromorphic Orchestrator
======================================================
Subclasses Brain and adds the full IB capability layer:

  query(text)               → IBResponse  (natural language Q&A)
  ingest_document(path)     → learn from PDF / Excel / text
  ingest_folder(folder)     → batch-ingest all docs in a folder
  build_model(type, inputs) → run DCF / LBO / Merger / Comps / Precedents / Credit
  correct_excel(in, out)    → overnight Excel audit + correction
  start_continuous_learning()→ launch 24/7 web + YouTube daemon
  status()                  → health report dict
"""

from __future__ import annotations

import os
import time
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from neuromorphic.brain import Brain
from neuromorphic.safety.kernel import SafetyKernel
from neuromorphic.safety.constraints import MotorConstraints
from neuromorphic.safety.reflexes import ReflexLibrary

from .encoders.financial_encoder import (
    FinancialEncoder,
    FinancialChunk,
    QueryVector as EncoderQueryVector,
)
from .encoders.document_ingestion import DocumentIngestion
from .decoders.financial_decoder import FinancialDecoder
from .decoders.response_synthesizer import ResponseSynthesizer, IBResponse
from .knowledge.deal_memory import DealMemory
from .knowledge.risk_engine import RiskEngine
from .knowledge.knowledge_base import KnowledgeBase
from .safety.financial_constraints import FinancialSafetyKernel
from .query.query_engine import QueryEngine, QueryVector
from .query.response_formatter import ResponseFormatter
from .models.dcf import DCFModel
from .models.lbo import LBOModel
from .models.merger import MergerModel
from .models.comps import CompsModel
from .models.precedents import PrecedentsModel
from .models.credit import CreditModel
from .excel.auditor import ExcelAuditor
from .learning.continuous_daemon import ContinuousLearningDaemon

logger = logging.getLogger("ib_brain")


# ── Spike-history ring buffer ─────────────────────────────────────────────────

class _SpikeHistory:
    """Fixed-length ring buffer of M1 spike arrays for stable decoding."""

    def __init__(self, n_neurons: int, window: int = 50):
        self._buf    = np.zeros((window, n_neurons), dtype=np.float32)
        self._ptr    = 0
        self._window = window

    def push(self, spikes: np.ndarray) -> None:
        self._buf[self._ptr % self._window] = spikes.astype(np.float32)
        self._ptr += 1

    def mean(self) -> np.ndarray:
        if self._ptr == 0:
            return self._buf[0]
        n = min(self._ptr, self._window)
        return self._buf[:n].mean(axis=0)


# ── IBBrain ───────────────────────────────────────────────────────────────────

class IBBrain(Brain):
    """
    Full IB-specialised neuromorphic brain.

    Parameters
    ----------
    config : module
        Pass ib_config (or neuromorphic.config).  Defaults to ib_config.
    safety_kernel : SafetyKernel | None
    seed : int
    verbose : bool
    """

    def __init__(
        self,
        config=None,
        safety_kernel: SafetyKernel | None = None,
        seed: int = 42,
        verbose: bool = True,
    ):
        from . import ib_config as _ib_cfg
        cfg = config if config is not None else _ib_cfg

        # Safety layer BEFORE Brain.__init__ (architectural invariant)
        # Use N_DOF from config so constraint shapes match the decoder output (32 for IB)
        sk = safety_kernel or SafetyKernel(
            MotorConstraints.default(n_dof=cfg.N_DOF), ReflexLibrary(n_dof=cfg.N_DOF)
        )
        super().__init__(config=cfg, safety_kernel=sk, seed=seed, verbose=verbose)

        n_m1 = self.regions["M1"].end - self.regions["M1"].start

        # ── IB components ─────────────────────────────────────────────
        self.financial_encoder  = FinancialEncoder(cfg, self.rng)
        self.financial_decoder  = FinancialDecoder(n_m1, cfg, seed)
        self.financial_safety   = FinancialSafetyKernel()
        self.synthesizer        = ResponseSynthesizer(cfg)
        self.query_engine       = QueryEngine()
        self.formatter          = ResponseFormatter()
        self.risk_engine        = RiskEngine()
        self.knowledge_base     = KnowledgeBase()
        self.deal_memory        = DealMemory()
        self.doc_ingestion      = DocumentIngestion(cfg)
        self.auditor            = ExcelAuditor()

        # M1 spike history ring buffer (50-step window for stable decoding)
        self._m1_history = _SpikeHistory(n_m1, window=50)

        # Continuous learning daemon (not started until requested)
        self._daemon: ContinuousLearningDaemon | None = None
        self._lock = threading.Lock()

        if verbose:
            print("[IBBrain] IB layer ready.")
            print(f"[IBBrain] KnowledgeBase: {len(self.knowledge_base)} concepts")
            print(f"[IBBrain] DealMemory:    {len(self.deal_memory)} seeded deals")

    # ------------------------------------------------------------------
    # Query — natural language Q&A
    # ------------------------------------------------------------------

    def query(
        self,
        text: str,
        user_inputs: dict | None = None,
        verbose: bool = False,
    ) -> IBResponse:
        """
        Answer an IB question using the spiking network.

        1. Parse → QueryVector (intent, entities, numerics)
        2. Knowledge-base shortcut for definitional questions
        3. Encode → V1/A1/S1 spike trains
        4. Run QUERY_STEPS simulation steps
        5. Decode M1 → FinancialParams
        6. Risk-check, safety-clamp, synthesise IBResponse
        """
        from .ib_config import QUERY_STEPS, PARAM_SLOTS

        user_inputs = user_inputs or {}
        t0 = time.time()

        # ── 1. Parse ──────────────────────────────────────────────────
        qv = self.query_engine.parse(text)

        # ── 2. Knowledge base shortcut ─────────────────────────────────
        if qv.intent == "knowledge" and qv.entities:
            for entity in qv.entities:
                kb_entry = self.knowledge_base.lookup(entity)
                if kb_entry:
                    answer = str(kb_entry)
                    return IBResponse(
                        answer_text  = answer,
                        parameters   = self._zero_params(),
                        confidence   = 0.95,
                    )

        # Merge numerics: query > user_inputs (user_inputs are overrides)
        merged_inputs = {**qv.numerical_values, **user_inputs}

        # ── 3. Build EncoderQueryVector ────────────────────────────────
        encoder_qv = EncoderQueryVector(
            raw_text         = text,
            query_type       = qv.intent,
            concepts         = qv.entities,
            numerical_values = merged_inputs,
            target_sector    = qv.sector,
            target_model     = qv.model_type,
            constraints      = {},
        )

        n_v1 = self.regions["V1"].end - self.regions["V1"].start
        n_a1 = self.regions["A1"].end - self.regions["A1"].start
        n_s1 = self.regions["S1"].end - self.regions["S1"].start
        n_m1 = self.regions["M1"].end - self.regions["M1"].start

        visual, auditory_list, soma = self.financial_encoder.encode_query(
            encoder_qv, n_v1, n_a1, n_s1, self.cfg.DT
        )

        # ── 4. Reset M1 history & run QUERY_STEPS ─────────────────────
        self._m1_history = _SpikeHistory(n_m1, window=50)
        n_steps   = QUERY_STEPS
        reward    = 0.5 + 0.3 * qv.confidence
        aud_steps = max(len(auditory_list), 1)

        for step_i in range(n_steps):
            aud = auditory_list[step_i % aud_steps]
            self.step(
                visual   = visual,
                auditory = aud,
                soma     = soma,
                reward   = reward if step_i < n_steps // 2 else 0.0,
            )
            m1_spikes = self.neurons.spikes[
                self.regions["M1"].start : self.regions["M1"].end
            ].copy()
            self._m1_history.push(m1_spikes)

            if verbose and step_i % 50 == 0:
                fr = m1_spikes.mean() * 1000 / self.cfg.DT
                print(f"  step {step_i}/{n_steps}  M1={fr:.1f}Hz")

        # ── 5. Decode M1 → FinancialParams ────────────────────────────
        avg_m1 = self._m1_history.mean()
        params = self.financial_decoder.decode(avg_m1)

        # Safety clamp
        param_dict = params.as_dict()
        clamped, violations = self.financial_safety.check_params(param_dict)

        # ── 6. Risk analysis ──────────────────────────────────────────
        risk_report = self.risk_engine.analyse({**clamped, **merged_inputs})
        risk_flags  = [str(f) for f in risk_report.flags]

        # ── 7. Run sub-model if intent calls for it ───────────────────
        model_result = None
        if qv.model_type or qv.intent in ("model_build", "lbo", "valuation", "credit"):
            mtype = qv.model_type or self._infer_model(qv)
            try:
                model_result = self._run_model(mtype, {**clamped, **merged_inputs})
            except Exception as e:
                logger.warning(f"Sub-model {mtype} failed: {e}")

        # ── 8. Synthesise IBResponse ──────────────────────────────────
        response = self.synthesizer.synthesize(
            params       = params,
            query        = encoder_qv,
            brain_state  = {"sim_time": self.t, "step": self.step_count},
            model_result = model_result,
            risk_flags   = risk_flags,
        )

        logger.info(
            f"query() {time.time()-t0:.2f}s — "
            f"intent={qv.intent} model={qv.model_type} risk={risk_report.level.value}"
        )
        return response

    # ------------------------------------------------------------------
    # Model execution
    # ------------------------------------------------------------------

    def build_model(
        self,
        model_type: str,
        inputs: dict | None = None,
        verbose: bool = True,
    ) -> dict:
        """
        Run a financial model directly (no neural settling needed).

        Parameters
        ----------
        model_type : "dcf" | "lbo" | "merger" | "comps" | "precedents" | "credit"
        inputs : dict
        """
        inputs = inputs or {}
        result = self._run_model(model_type, inputs) or {}
        if verbose:
            print(f"\n[IBBrain] {model_type.upper()} Result:")
            self._pretty_print_dict(result)
        return result

    def _run_model(self, model_type: str, inputs: dict) -> dict | None:
        from .models.dcf import DCFInputs, DCFModel
        from .models.lbo import LBOInputs, LBOModel
        from .models.merger import MergerInputs, MergerModel
        from .models.comps import CompsInputs, CompsModel, CompanyData
        from .models.precedents import PrecedentsInputs, PrecedentsModel, TransactionData
        from .models.credit import CreditInputs, CreditModel

        mt = (model_type or "").lower()
        g = inputs.get
        M = 1e6   # convert _m inputs (millions) to absolute dollars for model

        try:
            if mt == "dcf":
                rev_m = g("revenue_m", 500.0)
                em    = g("ebitda_margin", 0.25)
                inp = DCFInputs(
                    ebitda           = g("ebitda_m", rev_m * em) * M,
                    revenue          = rev_m * M,
                    revenue_growth   = g("revenue_growth", 0.08),
                    ebitda_margin    = em,
                    wacc             = g("wacc", 0.10),
                    terminal_growth  = g("terminal_growth", 0.025),
                    tax_rate         = g("tax_rate", 0.25),
                    capex_pct        = g("capex_pct_rev", 0.05),
                    nwc_pct          = g("nwc_pct_rev", 0.03),
                    depreciation_pct = g("da_pct_rev", 0.04),
                    projection_years = int(g("projection_years", 5)),
                    net_debt         = g("net_debt_m", 0.0) * M,
                )
                r = DCFModel().compute(inp)
                ebitda_yr1 = r.projected_ebitda[0] if r.projected_ebitda else M
                sens = (r.sensitivity_table / M).tolist() if hasattr(r.sensitivity_table, "tolist") else []
                return {
                    "enterprise_value_m": r.enterprise_value / M,
                    "equity_value_m":     (r.equity_value or r.enterprise_value) / M,
                    "implied_ev_ebitda":  r.enterprise_value / max(ebitda_yr1, 1.0),
                    "terminal_value_pct": r.pv_terminal / max(r.enterprise_value, 1.0) * 100,
                    "sensitivity_m":      sens,
                    "sensitivity_waccs":  r.sensitivity_waccs,
                    "sensitivity_tgs":    r.sensitivity_tgs,
                }

            elif mt == "lbo":
                rev_m = g("revenue_m", 500.0)
                em    = g("ebitda_margin", 0.25)
                ebt_m = g("ebitda_m", rev_m * em)
                inp = LBOInputs(
                    ebitda             = ebt_m * M,
                    revenue            = rev_m * M,
                    entry_ev_ebitda    = g("entry_multiple", 10.0),
                    revenue_growth     = g("revenue_growth", 0.08),
                    ebitda_margin_exit = g("ebitda_margin", em),
                    leverage_ratio     = g("leverage", 6.0),
                    senior_debt_rate   = g("interest_rate", 0.07),
                    hold_period        = int(g("hold_years", 5)),
                    exit_multiple      = g("exit_multiple", 12.0),
                    tax_rate           = g("tax_rate", 0.25),
                    capex_pct          = g("capex_pct_rev", 0.04),
                    nwc_pct            = g("nwc_pct_rev", 0.03),
                    depreciation_pct   = g("da_pct_rev", 0.03),
                )
                r = LBOModel().compute(inp)
                return {
                    "entry_ev_m":     r.entry_enterprise_value / M,
                    "exit_ev_m":      r.exit_enterprise_value / M,
                    "entry_equity_m": r.equity_contribution / M,
                    "exit_equity_m":  r.exit_equity_value / M,
                    "irr":            r.irr,
                    "moic":           r.moic,
                    "sensitivity":    r.sensitivity_irr.tolist() if hasattr(r.sensitivity_irr, "tolist") else [],
                }

            elif mt == "merger":
                inp = MergerInputs(
                    acq_net_income   = g("acq_net_income_m", 200.0) * M,
                    acq_shares       = g("acq_shares_m", 100.0) * M,
                    acq_share_price  = g("acq_share_price", 40.0),
                    acq_pe           = g("acq_pe", 20.0),
                    tgt_net_income   = g("tgt_net_income_m", 50.0) * M,
                    tgt_share_price  = g("tgt_share_price", 20.0),
                    tgt_shares       = g("tgt_shares_m", 50.0) * M,
                    premium_pct      = g("premium_pct", 30.0) / 100.0,
                    cash_pct         = g("cash_pct", 0.5),
                    stock_pct        = g("stock_pct", 0.5),
                    annual_synergies = g("synergies_m", 0.0) * M,
                    one_time_costs   = g("one_time_costs_m", 0.0) * M,
                    tax_rate         = g("tax_rate", 0.25),
                )
                r = MergerModel().compute(inp)
                return {
                    "accretion_dilution_pct": r.accretion_dilution_pct,
                    "pro_forma_eps":          r.pro_forma_eps_yr1,
                    "exchange_ratio":         r.exchange_ratio,
                    "offer_value_m":          r.offer_value / M,
                    "breakeven_synergies_m":  r.breakeven_synergies / M,
                }

            elif mt == "comps":
                peers = g("peers", [
                    CompanyData("Peer A", ev=1200*M, revenue=500*M, ebitda=100*M, ebit=85*M, net_income=60*M),
                    CompanyData("Peer B", ev=2000*M, revenue=800*M, ebitda=160*M, ebit=135*M, net_income=90*M),
                    CompanyData("Peer C", ev=900*M,  revenue=400*M, ebitda=75*M,  ebit=63*M,  net_income=45*M),
                ])
                rev_m = g("revenue_m", 500.0)
                em    = g("ebitda_margin", 0.25)
                target = CompanyData(
                    name="Target",
                    ev         = g("target_ev_m", rev_m * em * 10.0) * M,
                    revenue    = rev_m * M,
                    ebitda     = g("ebitda_m", rev_m * em) * M,
                    ebit       = g("ebit_m", rev_m * em * 0.85) * M,
                    net_income = g("net_income_m", rev_m * em * 0.5) * M,
                )
                inp = CompsInputs(target=target, comparables=peers)
                r = CompsModel().compute(inp)
                ev_eb_stats = r.stats.get("ev_ebitda", {})
                lo, hi = r.implied_ev_range
                return {
                    "median_ev_ebitda":  ev_eb_stats.get("median", 0.0),
                    "mean_ev_ebitda":    ev_eb_stats.get("mean", 0.0),
                    "implied_ev_low_m":  lo / M,
                    "implied_ev_high_m": hi / M,
                    "stats":             r.stats,
                }

            elif mt == "precedents":
                txns = g("transactions", [
                    TransactionData("Deal A", 2022, "technology", 1500*M, 600*M, 120*M, premium_pct=0.32),
                    TransactionData("Deal B", 2021, "technology", 1800*M, 700*M, 140*M, premium_pct=0.38),
                    TransactionData("Deal C", 2020, "technology", 1200*M, 500*M, 100*M, premium_pct=0.28),
                ])
                rev_m = g("revenue_m", 500.0)
                em    = g("ebitda_margin", 0.25)
                inp = PrecedentsInputs(
                    target_ebitda  = g("ebitda_m", rev_m * em) * M,
                    target_revenue = rev_m * M,
                    target_sector  = g("sector", "technology"),
                    transactions   = txns,
                    year_filter    = int(g("min_year", 2015)),
                )
                r = PrecedentsModel().compute(inp)
                lo, hi = r.implied_ev_range
                ev_stats = r.multiples.get("ev_ebitda", {})
                return {
                    "mean_premium_pct":  r.mean_premium * 100,
                    "median_ev_ebitda":  ev_stats.get("median", 0.0),
                    "implied_ev_low_m":  lo / M,
                    "implied_ev_high_m": hi / M,
                    "n_transactions":    len(r.filtered_transactions),
                }

            elif mt == "credit":
                rev_m = g("revenue_m", 500.0)
                em    = g("ebitda_margin", 0.25)
                ebt_m = g("ebitda_m", rev_m * em)
                lev   = g("leverage", 5.0)
                inp = CreditInputs(
                    ebitda           = ebt_m * M,
                    revenue          = rev_m * M,
                    total_debt       = g("total_debt_m", ebt_m * lev) * M,
                    cash             = g("cash_m", 50.0) * M,
                    interest_expense = g("interest_m", ebt_m * lev * 0.07) * M,
                    capex            = g("capex_m", rev_m * 0.05) * M,
                    tax_rate         = g("tax_rate", 0.25),
                    net_income       = g("net_income_m", ebt_m * 0.5) * M,
                )
                r = CreditModel().compute(inp)
                return {
                    "total_leverage":    r.leverage_total,
                    "net_leverage":      r.leverage_net,
                    "interest_coverage": r.interest_coverage,
                    "dscr":              r.dscr,
                    "implied_rating":    r.implied_rating,
                    "risk_level":        r.risk_level,
                    "covenant_headroom": r.covenant_headroom_pct,
                }

            else:
                logger.warning(f"Unknown model type: {mt}")
                return None

        except Exception as e:
            logger.error(f"Model {mt} failed: {e}", exc_info=True)
            return {"error": str(e)}

    def _infer_model(self, qv: QueryVector) -> str:
        return {
            "valuation":   "dcf",
            "lbo":         "lbo",
            "structuring": "merger",
            "credit":      "credit",
        }.get(qv.intent, "dcf")

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------

    def ingest_document(self, path: str, reward: float = 0.7) -> int:
        """Learn from a PDF, Excel, or text file."""
        path = str(Path(path).resolve())
        logger.info(f"Ingesting: {path}")
        chunks = self.doc_ingestion.extract(path)
        self._ingest_chunks(chunks, reward=reward)
        n = len(chunks)
        logger.info(f"Ingested {n} chunks from {os.path.basename(path)}")
        return n

    def ingest_folder(self, folder: str, extensions: list | None = None) -> int:
        """Batch-ingest all documents in a folder."""
        exts = set(extensions or [".pdf", ".xlsx", ".xls", ".txt", ".csv", ".html"])
        total = 0
        for fpath in sorted(Path(folder).resolve().rglob("*")):
            if fpath.suffix.lower() in exts and fpath.is_file():
                try:
                    total += self.ingest_document(str(fpath))
                except Exception as e:
                    logger.warning(f"Failed to ingest {fpath.name}: {e}")
        logger.info(f"Folder ingestion complete: {total} chunks")
        return total

    def _ingest_chunks(self, chunks: list, reward: float = 0.6) -> None:
        """
        Called by ContinuousLearningDaemon.
        Encodes each chunk and runs a short STDP burst to reinforce weights.
        """
        if not chunks:
            return

        n_v1 = self.regions["V1"].end - self.regions["V1"].start
        n_a1 = self.regions["A1"].end - self.regions["A1"].start
        n_s1 = self.regions["S1"].end - self.regions["S1"].start

        for chunk in chunks:
            visual, auditory_list, soma = self.financial_encoder.encode_chunk(
                chunk, n_v1, n_a1, n_s1, self.cfg.DT
            )
            aud_steps = max(len(auditory_list), 1)
            for step_i in range(20):      # short 20-step learning burst
                aud = auditory_list[step_i % aud_steps]
                self.step(visual=visual, auditory=aud, soma=soma, reward=reward)

    # ------------------------------------------------------------------
    # Excel overnight correction
    # ------------------------------------------------------------------

    def correct_excel(self, input_path: str, output_path: str | None = None) -> dict:
        """
        Audit and auto-correct an Excel financial model.

        Returns dict with keys: issues_found, issues_fixed, output_path, report.
        """
        if output_path is None:
            p = Path(input_path)
            output_path = str(p.parent / (p.stem + "_corrected" + p.suffix))

        report = self.auditor.audit_and_correct(input_path, output_path)
        n_fixed = getattr(report, "n_auto_fixed", 0)
        n_total = getattr(report, "n_issues", 0)
        print(f"[IBBrain] Audit: {n_total} issues, {n_fixed} auto-fixed → {output_path}")
        return {
            "issues_found": n_total,
            "issues_fixed": n_fixed,
            "output_path":  output_path,
            "report":       report,
        }

    # ------------------------------------------------------------------
    # Continuous learning daemon
    # ------------------------------------------------------------------

    def start_continuous_learning(
        self,
        interval_minutes: float = 60.0,
        topics: list | None = None,
    ) -> None:
        """Start the 24/7 web + YouTube learning daemon."""
        with self._lock:
            if self._daemon and self._daemon.running:
                print("[IBBrain] Continuous learning already running.")
                return
            self._daemon = ContinuousLearningDaemon(
                brain            = self,
                interval_minutes = interval_minutes,
                topics           = topics,
            )
            self._daemon.start()
            print(f"[IBBrain] Continuous learning started (every {interval_minutes:.0f} min)")

    def stop_continuous_learning(self) -> None:
        with self._lock:
            if self._daemon:
                self._daemon.stop()
                self._daemon = None
                print("[IBBrain] Continuous learning stopped.")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        rates = {
            name: float(
                self.neurons.spikes[region.start:region.end].mean()
                / self.cfg.DT
            )
            for name, region in self.regions.items()
        }
        daemon_running = False
        daemon_str = "stopped"
        if self._daemon:
            daemon_str = self._daemon.status()      # returns a string
            daemon_running = "running" in daemon_str
        return {
            "step_count":       self.step_count,
            "sim_time_s":       float(self.t),
            "total_neurons":    len(self.neurons.v),
            "total_synapses":   sum(p.n_synapses for p in self.pools.values()),
            "firing_rates_hz":  rates,
            "deal_memory_size": len(self.deal_memory),
            "kb_concepts":      len(self.knowledge_base),
            "learning_daemon":  {"running": daemon_running, "status": daemon_str},
        }

    def print_status(self) -> None:
        s = self.status()
        print("\n" + "=" * 60)
        print("  NEUROMORPHIC IB BRAIN STATUS")
        print("=" * 60)
        print(f"  Sim steps  : {s['step_count']:,}")
        print(f"  Sim time   : {s['sim_time_s']:.3f}s")
        print(f"  Neurons    : {s['total_neurons']:,}")
        print(f"  Synapses   : {s['total_synapses']:,}")
        print(f"  Deal memory: {s['deal_memory_size']} deals")
        print(f"  KB concepts: {s['kb_concepts']}")
        print(f"\n  Firing rates (Hz):")
        for region, rate in s["firing_rates_hz"].items():
            bar = "█" * min(int(rate / 5), 20)
            print(f"    {region:5s}: {rate:6.1f}  {bar}")
        d = s["learning_daemon"]
        print(f"\n  Daemon: {'RUNNING' if d.get('running') else 'stopped'}")
        if d.get("status"):
            print(f"    {d['status']}")
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _zero_params(self):
        """Return empty FinancialParams (for knowledge-base shortcut responses)."""
        from .decoders.financial_decoder import FinancialParams
        from .ib_config import PARAM_NAMES, PARAM_RANGES
        n = len(PARAM_NAMES)
        return FinancialParams(
            raw_values  = np.zeros(n, dtype=np.float32),
            confidence  = np.zeros(n, dtype=np.float32),
            param_names = list(PARAM_NAMES),
            param_ranges = PARAM_RANGES,
        )

    @staticmethod
    def _pretty_print_dict(d: dict, indent: int = 2) -> None:
        prefix = " " * indent
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"{prefix}{k}:")
                IBBrain._pretty_print_dict(v, indent + 2)
            elif isinstance(v, float):
                print(f"{prefix}{k}: {v:.4f}")
            elif isinstance(v, list):
                print(f"{prefix}{k}: [{len(v)} items]")
            else:
                print(f"{prefix}{k}: {v}")
