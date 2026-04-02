"""
Neuromorphic Investment Banking Brain
======================================
World-class M&A specialist that:
  - Learns 24/7 from the internet (web + YouTube)
  - Accesses and corrects your Excel models overnight
  - Answers technical and strategic IB questions
  - Runs full DCF, LBO, merger, comps, precedents, credit models
  - Gets smarter with every deal, every document, every interaction

Quick start
-----------
    from neuromorphic.domains.investment_banking import IBBrain

    brain = IBBrain(scale=0.01)          # demo scale
    brain.ingest_document("pitch_deck.pdf")
    response = brain.query("What WACC for mid-market SaaS?")
    print(response.answer_text)

    # Overnight correction
    brain.correct_excel("my_model.xlsx", output="my_model_corrected.xlsx")

    # Start 24/7 learning daemon
    brain.start_continuous_learning(topics=["M&A", "LBO", "DCF", "valuation"])
"""

from neuromorphic.domains.investment_banking.ib_brain import IBBrain

__all__ = ["IBBrain"]
