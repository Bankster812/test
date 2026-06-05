# Compliant lead sourcing — the legitimate alternative to scraping Zillow

You asked to "use Zillow and other platforms to find real owners." Here's the
honest engineering + legal reality, and the compliant path that gets you the
same data without the liability.

## Why not scrape Zillow / Redfin
- Their **Terms of Service prohibit scraping**; automated extraction can lead to
  IP bans and legal action.
- They **don't publish owner contact info** anyway. Owner identity comes from
  county records; phone/email comes from **skip-tracing**, which is regulated
  (FCRA permissible-purpose rules; DPPA for DMV-derived data).
- Building your pipeline on a ToS violation makes every downstream contract
  tainted.

## Legitimate distressed-lead sources (these are what the pros actually use)

| Source | What you get | Access |
|---|---|---|
| **County records** (recorder / clerk / sheriff) | **Lis pendens, Notices of Default (NOD), tax-delinquent, probate** filings — the real pre-foreclosure signal | Public records; many counties have online portals or bulk data |
| **Auction.com, Xome, Hubzu** | Pre-foreclosure & REO listings | Public listings / their APIs/terms |
| **ATTOM Data, CoreLogic** | Nationwide property + pre-foreclosure/owner data via **API** | Paid, license-clean |
| **PropStream, BatchLeads, DealMachine** | Distressed lists + **compliant skip-trace** | Paid SaaS, TCPA-aware |
| **Direct mail / "driving for dollars"** | Vacant/distressed property spotting | Fully compliant, slow |

## The contact rules you must wire in (the platform enforces a gate)
- **TCPA**: no autodialed/pre-recorded calls or texts without consent; scrub
  against the **National Do-Not-Call** registry. Penalties run $500–$1,500 *per
  message*.
- **Pre-foreclosure / equity-purchaser laws**: many states impose mandatory
  written contracts, specific disclosures, and a homeowner **right of
  rescission**. Some require these in a specific font/format.
- **State licensing**: several states now require a real-estate license to
  wholesale, or restrict it. Confirm per state.

## How this maps to the codebase
- `wholesale/data/market.py` is the **lead-source seam** — replace the synthetic
  feed with an adapter to ATTOM/county data above (same `Property`/`Seller`
  shape).
- `wholesale/compliance/gate.py` **blocks live outreach** per state until you
  attest (entity, licensing, attorney, contracts reviewed, foreclosure-law
  cleared, TCPA process, compliant data). Default posture is BLOCKED.
- `wholesale/integrations/` adapters stay **dry-run** until armed *and* wired.

Net: the platform will happily do sourcing → underwriting → buyer-matching →
**draft** contracts all day. The one thing it will not do on its own is fire
real messages at real distressed homeowners — that step is gated behind your
licensing and counsel, by design.
