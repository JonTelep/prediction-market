# Live validation cases

Ground-truth runs of the Phase-2 backtest stack against real, publicly
documented insider-trading cases. Each section records exactly what the live
APIs returned, how the case was labeled, and what the detectors saw on
point-in-time replay. Wallet findings here are **pattern observations on
public trading data** — never identification of, or accusation against, any
person.

---

## Case 1: Maduro capture — `maduro-out-by-january-31-2026-318`

**Session:** 2026-07-20, live network, human present.
**Market:** "Maduro out by January 31, 2026?" (Gamma id 916440, condition
`0x580adc…6f993`), $10.97M reported volume, closed 2026-01-03 12:14:07 UTC —
within hours of the publicly reported Maduro capture.
**Public record:** DOJ charge against a U.S. soldier (account created
Dec 26 2025; ~13 bets ≈ $33K placed Dec 27–Jan 2; ≈ $400K profit on the
Jan 3 2026 capture).
https://www.justice.gov/opa/pr/us-soldier-charged-using-classified-information-profit-prediction-market-bets

### Market resolution

The exact slug is not in the public DOJ record. The operator supplied the
Polymarket event `venezuela-leader-end-of-2026`, whose child markets are all
still active (they resolve Dec 31 2026) — the wrong market family for a
Jan 3 resolution. A Gamma `public-search` for "Maduro" surfaced the event
`maduro-out-in-2025`, whose child `maduro-out-by-january-31-2026-318`
resolved YES at the capture with $11M volume; confirmed with the operator
before archiving. Sibling candidates (`maduro-out-in-2025-411`,
`us-operation-to-capture-maduro-in-2025`, `maduro-in-us-custody-by-december-31`)
all closed on or before Jan 1 2026, i.e. resolved **before** the capture, so
the Jan-31 market is the only one whose resolution could have paid the
described bets.

### Data retrieved (with gaps)

| Artifact | Result |
|---|---|
| Gamma market metadata | ✔ full record (only reachable with `closed=true` — see mismatches below) |
| CLOB price history | ✔ 538 points, 2025-12-12 → 2026-01-03 12:00 UTC, ~hourly (only via `startTs`/`endTs` in ≤14-day chunks — see mismatches) |
| Data-API trade tape | ✔ 9,367 unique trades, 2025-12-12 → 2026-01-03 12:13 UTC, 3,533 distinct wallets; page cap **not** hit |
| Scheduled events | none archived (calendar sources are keyless-config live-mode features; not part of this archive) |

**Tape-completeness caveat:** the tape's total notional is ≈ $4.5M against
Gamma's reported $11M market volume. Whether Gamma double-counts sides or
the Data API serves a bounded history for resolved markets is not
determinable from public responses; treat the tape as *possibly partial*.
Phase-3 on-chain reconstruction is the definitive fix.

**Price trajectory (YES):** drifted 0.135 → 0.055–0.095 across Dec 24–Jan 2
(the market *drifted away from* the outcome while the DOJ window was open),
then 0.075 → 1.0 on Jan 3. The capture was a genuine surprise to the
aggregate market.

### Live-API mismatches found (each verified live, fixed, committed)

The Phase-2 client code was written from public API documentation with an
explicit charter to verify against reality in this session. Six mismatches
were found; every one is now covered by a regression test:

1. `c95bfb2` — Gamma `/markets?slug=` silently excludes closed markets
   unless `closed=true` is sent; resolved markets were unreachable.
2. `e2ab632` — Data API serves `size`/`price` as JSON numbers; the Trade
   model assumed strings and rejected every live record.
3. `7153c48` — live `/trades` records key the market as `conditionId`, the
   token as `asset`, the time as an epoch number under `timestamp`, and
   carry **no `id`**; the `condition_id` filter param is silently ignored
   (returning the global cross-market feed), and cursor pagination is not
   honored (the same page repeats — an early archive contained 500 copies
   of the global feed's first page). Filter is `market=<conditionId>`;
   pagination is `offset`-based; ids are now synthesized deterministically.
4. `c302bed` — CLOB `/prices-history` requires the token id under `market`
   (`token_id` → 400 "the 'market' (asset id) is mandatory"). This also
   means Phase 1's live backfill price-history path never returned data.
5. `2a7a9ab` — `/prices-history` `interval`-form queries return empty for
   resolved markets; `startTs`/`endTs` ranges work but spans over ~15 days
   are rejected (15d → 200, 21d → 400). The archiver now walks the market
   lifetime in 14-day chunks.
6. `1064269` — `Trade.match_datetime` did not parse epoch-seconds
   timestamps, silently producing an all-zero cumulative-volume spine.

### Label (hand-written, from the public record)

`window_start = 2025-12-27 00:00:00`, `window_end = event_time =
2026-01-03 00:00:00` UTC (DOJ bet window through the capture date; the DOJ
release does not state an announcement hour, so the date boundary is used).

### Replay evaluation (default thresholds)

```
detected:          true
first_hit_time:    2025-12-29 17:00:20   (lead time ≈ 4.3 days / 6,180 min)
hits in window:    6
false alarms:      18   (of 24 total reports; includes the Jan 3 resolution surge)
null control:      20 runs, fp_path_rate = 1.0, mean 2.95 reports/path
```

First in-window report: volume z ≈ 4.3 (combined 4.50, severity medium),
with wallet corroboration attached (top window wallet at 35% share,
profiler score 0.74).

### Wallet evidence vs the DOJ pattern

The DOJ describes ~13 bets totaling ≈ $33K placed Dec 27–Jan 2 from a
fresh account. Searching the archived tape for that shape (≥5 YES buys in
the window, grouped by wallet):

- **No wallet in the retrievable tape matches it.** The largest fresh
  one-sided YES accumulators in the window total ≈ $3.8K and ≈ $3.7K
  (6 and 5 buys at avg 8–10¢) — an order of magnitude short.
- The profiler's top-scoring window wallets are fresh single-trade
  accounts (largest: one $19K NO buy at 0.95 — the *opposite* side,
  consistent with late hedging/market-making, not the described pattern).

Honest verdict on the wallet layer: **the DOJ-described pattern is not
present in the tape we can retrieve.** Given the tape-completeness caveat
above, absence from this tape is not evidence the pattern didn't exist —
only that public REST data (possibly partial, possibly one-sided) does not
surface it. No pattern match is claimed.

### Verdict

**Flagged with lead time — on aggregate signals, with poor specificity.**
The detector raised in-window alerts from Dec 29, 4.3 days before the
event, triggered primarily by volume z-spikes during the DOJ bet window.
But the null control is damning for precision: all 20 volatility-matched
synthetic nulls also produced reports (fp_path_rate 1.0, vs 1/3 on the
Phase-2 minimal fixture's nulls). At default thresholds, an alert on a
market with this volatility profile carries little evidential weight by
itself; the lead-time result is real but must be read against that base
rate. The wallet-corroboration layer attached evidence but could not
reproduce the DOJ pattern from the retrievable tape.

### Follow-ups implied

1. **Threshold/amplifier calibration against null base rates** — the
   event-proximity amplifier and volume z-path both fire freely on
   matched nulls (also observed at seed 42 of the flagship sweep and in
   the Phase-2 FP test). Calibration study is Phase-3 work; the `backtest
   --null-runs` machinery now exists to drive it.
2. **VolumeAnalyzer warm-up quirk** — null-path logs show
   `z=4.02 vol=8453.18 mean=8453.18` (a z-score of 4 when the observation
   equals its own baseline mean, on the first observations of a constant
   volume series). Suspected warm-up artifact; needs a reviewed look.
3. **On-chain tape reconstruction (Phase 3)** — resolves the $4.5M-vs-$11M
   completeness question and the missing DOJ-pattern wallet definitively.
4. **Sibling-market archives** — the insider window overlaps
   `maduro-out-in-2025-411` ($34.6M) and other resolved siblings;
   archiving them would test whether the described flow appears there.
5. **Iran fallback cases** (runbook step 6) — not yet run this session.
