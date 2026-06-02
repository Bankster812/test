# Disposition targets — sell the contract *first*

Your instinct is right: **line up the buyer before you tie up the seller.** A
wholesale deal only works if you already know who you're assigning to and at
what number. This is a researched starting list (public info, June 2026). It is
*not* an endorsement and these parties' criteria change — verify current
buy-boxes directly.

> Reality check on "hedge funds": the big institutional single-family-rental
> (SFR) funds mostly buy **new-build in bulk from homebuilders or in portfolio
> trades**, and through 2025–2026 they have been **net sellers** of homes, not
> buyers. They rarely take one-off assigned contracts from a new wholesaler.
> Your realistic assignment buyers are **regional cash buyers, flippers, local
> rental operators, and buyer marketplaces** — with institutions as occasional
> portfolio buyers once you have volume.

## 1. Assignment / disposition marketplaces (most realistic first buyers)

| Platform | What it is | Notes |
|---|---|---|
| **InvestorLift** | Largest US wholesaling marketplace (~5.5M cash buyers, $30B+ transacted). AI "Autopilot" blasts SMS/email to matched buyers. | Enterprise pricing; built for volume. Strong once you have deal flow. |
| **OfferMarket** | Free marketplace to list to verified buyers; also off-market sourcing + private lending. | Good low-cost starting point. |
| **DispoBridge** | Performance-based (no upfront), 500+ verified buyers across 800+ cities, all 50 states. | Newer (founded 2025); pay-on-success model. |
| **DealFlow AI** | Pulls cash buyers from county deed records, qualifies them, generates state-specific assignment contracts w/ e-sign. | Lower-cost InvestorLift alternative. |
| **New Western** | Marketplace/brokerage moving large volumes of investment property to a vetted buyer base. | They source and sell; understand their model before relying on it. |
| **Sundae** | Marketplace connecting distressed-property sellers to investor buyers. | More seller-side, but a buyer pool exists. |

## 2. Build your *own* verified cash-buyer list (highest leverage)

The durable asset is your own buyer list — you keep the spread. Build it from:
- **County deed records**: pull recent **cash purchases** (no mortgage/lien) in
  your zips → those buyers are active, verifiable, and public record.
- **Auction.com / Xome / Hubzu** active bidders.
- **Local REIA meetings** and BiggerPockets marketplace.
- **PropStream / BatchLeads** "cash buyer" filters (licensed data, not scraping).

Capture for each buyer: metros, property types, max price, target spread,
proof-of-funds, and how fast they close. That's exactly the `Buyer` buy-box the
platform already models in `wholesale/data/buyers.py` — swap the synthetic book
for your real one.

## 3. Institutional SFR buyers (portfolio buyers, once you have volume)

Public, well-known operators (most buy in **bulk**, not one-off assignments):
Invitation Homes, Progress Residential (Pretium), American Homes 4 Rent (AMH),
Amherst / Main Street Renewal, Tricon Residential, FirstKey Homes, Home Partners
of America (Blackstone). Approach their **acquisitions teams** only when you can
deliver portfolios that fit a published buy-box — not as a first dispo channel.

## Sources
- [St. Louis Fed — Role of Single-Family Rentals (Oct 2025)](https://www.stlouisfed.org/on-the-economy/2025/oct/role-single-family-rentals-us-housing-market)
- [ResiClub — largest institutional homeowner buying slump](https://www.resiclubanalytics.com/p/housing-market-largest-institutional-homeowner-section-8-market)
- [CNBC — big investors fleeing for-sale market (Mar 2026)](https://www.cnbc.com/2026/03/04/institutional-investors-housing-market.html)
- [OfferMarket — best places to wholesale](https://www.offermarket.us/blog/best-places-to-wholesale-real-estate)
- [REIkit — 34 ways to find cash buyers](https://www.reikit.com/wholesaling-houses/marketing/34-ways-to-grow-cash-buyers-list)
- [PropStream — how to find cash buyers](https://www.propstream.com/real-estate-investor-blog/how-to-find-cash-buyers-for-wholesaling-real-estate)
