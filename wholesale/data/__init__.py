"""Market data: synthetic lead/property generation and the cash-buyer book.

The generators here are the seam where real data sources plug in. Swap
`MarketFeed.next_leads` for a list-source / MLS / skip-trace adapter and
the rest of the company is unchanged.
"""
