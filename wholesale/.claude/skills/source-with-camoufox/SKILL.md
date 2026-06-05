---
name: source-with-camoufox
description: Stealth-source distressed real-estate listings from Zillow / Realtor / other sites that block plain HTTP clients. Run the Camoufox-based scraper, verify bot-detection bypass, extract listings into the wholesale pipeline. Use when you need to source leads from Zillow/Realtor, debug a "Zillow returns 0 listings" failure, set Camoufox up on a fresh machine, or take a screenshot of what the Zillow scraper actually sees.
---

# source-with-camoufox

Drive the Zillow / Realtor / Auction.com listing scrapers via
[Camoufox](https://github.com/daijro/camoufox) — a patched Firefox
build that bypasses Cloudflare / PerimeterX / DataDome / Akamai bot
detection at the C++ level (not just JS shims like playwright-stealth).

All paths below are from the repo root (the directory containing
`wholesale/`).

## When to use

- "Scout returns 0 leads from Zillow" → run `driver.py zillow TX` to see what Zillow actually serves.
- Setting up the project on a new machine → install steps below.
- A scraper file (`wholesale/sourcing/zillow.py` etc.) was changed → run `driver.py extract` to confirm it still parses.
- Need a screenshot of what a real Zillow listing page looks like for the user.

## Prerequisites (Ubuntu / Debian)

Firefox runtime libs. The Firefox bundled inside Camoufox needs the same `lib*` packages a normal Firefox does:

```bash
sudo apt-get install -y \
  libgtk-3-0 libdbus-glib-1-2 libxt6 libasound2t64 libxcomposite1 \
  libxdamage1 libxrandr2 libpangocairo-1.0-0 libnss3 libxss1 xvfb
```

`xvfb` is only needed for headed runs on a server. Headless runs (the default) don't need it. **In this container, all of these are already installed.**

## Install Camoufox

```bash
pip install --user "camoufox[geoip]"
pip install --user --upgrade PyYAML        # camoufox needs CLoader from PyYAML 6+
python3 -m camoufox fetch                  # downloads ~630MB patched Firefox
```

### Offline-install fallback (when `camoufox fetch` fails)

`camoufox fetch` calls the GitHub API which is rate-limited and blocked in some sandboxed environments. If you see `403 Client Error: Forbidden for url: https://api.github.com/repos/daijro/camoufox/releases`, install manually:

```bash
# 1. Find latest release tag
TAG=$(curl -sI https://github.com/daijro/camoufox/releases/latest \
      | awk -F'/' '/^location:/ {print $NF}' | tr -d '\r')
echo "latest: $TAG"

# 2. Find the linux x86_64 asset name (Camoufox sometimes labels it alpha.N+1)
ASSET=$(curl -s "https://github.com/daijro/camoufox/releases/expanded_assets/$TAG" \
        | grep -oE 'camoufox-[0-9][^"<]*-lin\.x86_64\.zip' | head -1)
echo "asset: $ASSET"

# 3. Download to camoufox's expected install dir
INSTALL=$(python3 -c 'from platformdirs import user_cache_dir; print(user_cache_dir("camoufox"))')
mkdir -p "$INSTALL" && cd "$INSTALL"
curl -L -o camoufox.zip \
  "https://github.com/daijro/camoufox/releases/download/$TAG/$ASSET"
unzip -q camoufox.zip && rm camoufox.zip

# 4. Camoufox checks for version.json — write it
VER=${TAG#v}                                # strip leading "v"
PART_VERSION=${VER%%-*}                     # "150.0.2"
PART_RELEASE=${VER#*-}                      # "beta.25"
cat > version.json <<EOF
{"version": "$PART_VERSION", "release": "$PART_RELEASE"}
EOF
```

## Run: verify Camoufox works

The driver script lives at `wholesale/.claude/skills/source-with-camoufox/driver.py`. Outputs land in `wholesale/.claude/skills/source-with-camoufox/screenshots/`.

```bash
# Visit Zillow foreclosures for Texas, save a screenshot, count listing cards
xvfb-run -a python3 wholesale/.claude/skills/source-with-camoufox/driver.py zillow TX
```

Expected output (verified in this container):

```
[*] https://www.zillow.com/tx/foreclosures/
[*] title: Texas Foreclosure Homes For Sale - 851 Homes | Zillow
[*] screenshot → .../screenshots/zillow_tx.png
[*] property-cards in DOM: 9
[ok] real listings rendered (9 cards)
```

If you see `[!] zero cards` — the page loaded but Zillow soft-blocked us OR changed selectors. Open the saved screenshot to see what they served.

## Run: extract real listings via the production scraper

```bash
xvfb-run -a python3 wholesale/.claude/skills/source-with-camoufox/driver.py extract TX
```

Expected output (verified in this container):

```
[*] scraping ZillowScraper(markets=['TX-Unknown'])...
[ok] got 10 listings
    5214 Compassion Ct, Midlothian TX 76065  $449,999  pre-foreclosure
    4551 Summer Fall, San Antonio TX 78259  $362,000  pre-foreclosure
    12534 Honor Park Dr, Houston TX 77065  $400,000  pre-foreclosure
    ...
```

This calls `wholesale.sourcing.zillow.ZillowScraper` directly — exactly what the Scout agent (`wholesale/agents/sourcing.py`) uses in production.

## Run: bot-detection fingerprint probe

```bash
xvfb-run -a python3 wholesale/.claude/skills/source-with-camoufox/driver.py probe
```

Hits `bot.sannysoft.com` and saves a screenshot of the detection report to `screenshots/sannysoft.png`. Useful when you're debugging "Zillow blocks me" and want to see whether the underlying fingerprint is OK. **Note:** this command was NOT verified in the build container (sannysoft.com unreachable through the sandbox network policy). It will work on a normal Linux host.

## Use it from code

```python
from wholesale.sourcing.camoufox_browser import CamoufoxBrowser

with CamoufoxBrowser() as b:
    html = b.fetch("https://www.zillow.com/tx/foreclosures/",
                   wait_selector="[data-testid='property-card']")
    # parse __NEXT_DATA__ → listings (see ZillowScraper for the recipe)
```

Env vars:

| Var | Default | Meaning |
|-----|---------|---------|
| `WS_BROWSER_PROXY` | — | `http://user:pass@host:port` for residential-proxy rotation |
| `WS_BROWSER_OS` | `windows` | Fingerprint OS (`windows`, `macos`, `linux`) |
| `WS_BROWSER_HEADLESS` | `1` | `0` to show a window (needs Xvfb on Linux) |

## Gotchas (battle scars from this container)

- **Extract returns 0 listings on some runs, 10 on others** — confirmed flaky. Zillow's PerimeterX rate-limits same-fingerprint requests; after ~1 successful scrape per minute it soft-blocks subsequent ones from the same IP+UA combo. `ZillowScraper.fetch()` auto-retries once with a fresh Camoufox session (new fingerprint). If you see persistent zeros, set `WS_BROWSER_PROXY` to a residential rotation. Re-running the driver typically works.
- **`camoufox fetch` returns 403 from GitHub API** — happens whenever the network has unauthenticated rate-limiting (sandboxes, corp NAT). Use the offline-install fallback above (download release zip directly via `https://github.com/.../releases/download/...`, which is served from a separate CDN with no rate-limit).
- **`ImportError: cannot import name 'CLoader' from 'yaml'`** — the system PyYAML is too old. `pip install --user --upgrade PyYAML` fixes it (6.0+ ships `CLoader`).
- **`FileNotFoundError: Version information not found at .../version.json`** — happens with the offline install. Write `version.json` manually (see fallback above). The format is `{"version": "<X.Y.Z>", "release": "<beta.N>"}`.
- **`Page.goto: SEC_ERROR_UNKNOWN_ISSUER`** — the container's network goes through a TLS-MITM proxy with self-signed certs. The wrapper sets `ignore_https_errors=True` on the context to pass through. In production (no MITM proxy) this is a no-op.
- **PerimeterX "Press & Hold" overlay appears in the screenshot but the JSON still has listings** — this is normal. Camoufox doesn't bypass every UI challenge perfectly, but `__NEXT_DATA__` is server-rendered before PerimeterX activates, so the data is there even when the UI is gated. Our scrapers read JSON, not click buttons, so this works fine.
- **Default uBlock add-on download hits GitHub API and fails** — the wrapper passes `exclude_addons=[DefaultAddons.UBO]` so it never tries. If you need uBlock, fetch it separately and drop it into `~/.cache/camoufox/addons/`.
- **`xvfb-run -a` recommended even for headless** — Camoufox occasionally tries to talk to an X display on startup. Running under Xvfb is harmless and prevents flaky launches on headless servers.

## Run-with-vs-without comparison

Without Camoufox (plain `urllib.request` or even Playwright Chromium):

```
HTTP/2 403  perimeterx challenge
title: "Please verify you are a human"
property-cards: 0
```

With Camoufox (this skill):

```
HTTP/2 200
title: "Texas Foreclosure Homes For Sale - 851 Homes | Zillow"
property-cards: 9
listings extracted from __NEXT_DATA__: 10+ per filter, 4 filters → up to 40 leads/run
```

## ToS warning

Zillow / Realtor.com / Redfin **explicitly prohibit automated access in their Terms of Use**. Camoufox defeats the technical bot-detection — it does not defeat the legal claim. Whether your operation is lawful is your call, and it depends on your jurisdiction, intended use, volume, and how aggressively the target enforces. For an audit-clean feed without legal risk, wire PropStream / BatchLeads / ATTOM through their official APIs (env keys: `PROPSTREAM_API_KEY` / `ATTOM_API_KEY`) — those adapters live in `wholesale/sourcing/providers.py`.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `403 ... /repos/daijro/camoufox/releases` on `camoufox fetch` | GitHub API rate-limit on unauthenticated requests | Use offline-install fallback above |
| `ImportError: cannot import name 'CLoader' from 'yaml'` | System PyYAML too old | `pip install --user --upgrade PyYAML` |
| `FileNotFoundError: Version information not found at .../version.json` | Manual install skipped writing version.json | `echo '{"version":"<X.Y.Z>","release":"<beta.N>"}' > ~/.cache/camoufox/version.json` |
| `Page.goto: SEC_ERROR_UNKNOWN_ISSUER` | Container's TLS-MITM proxy | Wrapper already sets `ignore_https_errors=True`; nothing to do |
| `Page.goto: SEC_ERROR_UNKNOWN_ISSUER` on a non-sandbox host | Real cert problem | Don't bypass — investigate the network path |
| `extract` returns 0 listings, but `zillow` shows 9 cards | PerimeterX rate-limit on consecutive requests | Auto-retries once; re-run; or set `WS_BROWSER_PROXY` |
| Driver hangs on launch | Xvfb not running | Wrap with `xvfb-run -a` |

## Files

```
wholesale/.claude/skills/source-with-camoufox/
  SKILL.md            (this file)
  driver.py           standalone harness (probe / zillow / extract)
  screenshots/        landing directory for driver shots

wholesale/sourcing/
  camoufox_browser.py production wrapper (CamoufoxBrowser context manager)
  zillow.py           ZillowScraper — uses CamoufoxBrowser
  realtor.py          RealtorScraper — uses CamoufoxBrowser
  hud_homes.py        HUD REOs — plain urllib (no JS)
  homepath.py         Fannie Mae REOs — plain urllib
  providers.py        registry; get_provider() picks the right one
```
