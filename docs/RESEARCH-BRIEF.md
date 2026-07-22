# Research Brief — Prediction-Market Surveillance Prior Art (compiled 2026-07-15)

Context for the Phase-1 prompt series (`docs/prompts/PHASE1-END-TO-END.md`) and future phases. Items marked ⚠️ rest on a single source or could not be independently cross-checked.

## 1. Prior art

**Academic:**
- **Mitts & Ofir (2026)**, "Insider Trading at Scale: Polymarket Transaction-Level Evidence" — screened ~93,000 markets / ~50,000 wallets (Feb 2024–Feb 2026) with a composite of five signals: cross-sectional bet size, within-trader bet size, profitability, pre-event timing, directional concentration. Flagged wallet-market pairs showed a 69.9% win rate (>60σ above chance), ~$143M estimated anomalous profit (conservative). https://corpgov.law.harvard.edu/2026/03/25/from-iran-to-taylor-swift-informed-trading-in-prediction-markets/
- **ForesightFlow** (arXiv:2605.00493) — information-leakage score framework using Polymarket price/volume + GDELT event timing + Bayesian belief tracking. Direct design template for this project.
- arXiv:2605.02286 (leakage on documented Polymarket insider cases) and arXiv:2605.00459 (population-scale evaluation) — ⚠️ titles/abstracts only, full text not verified.
- Intrade 2012: a single trader accounted for >1/3 of Romney "Yes" volume (⚠️ primary citation unconfirmed). PredictIt 2016: campaign staffers bought their own candidate to manufacture narrative.

**Open-source (young projects; read for scoring ideas, not maturity):**
- `pselamy/polymarket-insider-tracker` — WebSocket + Polygon RPC wallet profiling: fresh-wallet scoring (<48h, ≤5 txns, >$1000), size anomalies (>2% of 24h volume or >5% of book depth), DBSCAN "sniper" clustering; multi-signal bonuses; alert threshold 0.6.
- `suislanchez/polymarket-insider-detector` — binomial p-value on win rates (p<0.001, >80%), last-minute-trade ratio, whale tiers, USDC funding-trace Sybil clustering.
- `NYTEMODEONLY/polyterm`, `warproxxx/poly_data` (Envio HyperSync on-chain ingestion).
- Historical L2 order-book archives (Aug 2025+): polymarketbacktesting.com, polymarketdata.co — ⚠️ commercial claims, unverified.

## 2. Transferable TradFi/crypto techniques

- **Wash trading**: same-controller trade pairs, zero net position change with high volume, volume spikes without price impact; funding-graph inference (Cong et al., Yale Cowles: https://cowles.yale.edu/sites/default/files/2022-11/cryptowashtrading040521-crypto-wash-trading.pdf).
- **Spoofing/layering**: requires full order-event history (adds/cancels), not trade prints — Polymarket's public API does not provide this historically; only live WS capture does.
- **CUSUM / Bayesian change-point detection**: materially faster than rolling z-scores at flagging regime shifts (one comparison: 3 samples vs 15); nonparametric variants preferred since pre/post distributions are unknown (arXiv:1509.01570). Candidate Phase-2 upgrade to the z-score core.
- **Lead-lag across correlated markets** (same event, different venues/criteria): a move unmatched in a correlated market is itself a leakage signal — natural future use for the currently-unwired `analysis/correlation.py`.

## 3. Polymarket data reality (as of 2026)

- **Data API** `/trades` includes `proxyWallet` and `transactionHash` — trades are wallet-attributable. `/positions`, `/activity`, `/value` exist per-wallet.
- **`/holders` is hard-capped at 20 holders per token** — full concentration/Gini analysis requires on-chain reconstruction of CTF token transfers via Polygon RPC, not this endpoint. The current HHI-from-top-20 is a biased-but-usable proxy; document it as such.
- **Historical order-book depth is not reliably queryable** — one nautilus_trader issue (#3635) reports `/orderbook-history` no longer returns data. Depth history exists only if we capture it ourselves (which the `orderbook_snapshots` table does going forward).
- **CLOB V2 hard cutover on 2026-04-28**: new Exchange contracts, new order struct, pUSD collateral replacing USDC.e, no V1 compatibility. Read-only endpoints used by this repo (`/book`, `/price`, `/midpoint`, `/prices-history`) should be re-verified live before major reliance. ⚠️ Not verified from this environment.

**Ground-truth validation cases (for a future backtest harness):**
1. **Van Dyke / Maduro capture (Dec 2025–Jan 2026)** — DOJ-charged: account created Dec 26, ~13 bets ~$33K Dec 27–Jan 2, ~$400K profit on Jan 3 capture; post-resolution account-deletion attempt. https://www.justice.gov/opa/pr/us-soldier-charged-using-classified-information-profit-prediction-market-bets
2. **Iran war bets (Feb–Jun 2026)** — Bubblemaps: 9 connected accounts, $2.4M profit, 98% win rate over 80+ wagers (CBS 60 Minutes); Mitts & Ofir: 6 wallets ~$1.2M buying Yes at 10¢ vs 17% implied, one trading 71 min pre-announcement (~$553K).
3. **NYT investigation (May 2026)** — 80+ users with suspicious timing across ~30 topics since 2024 (weaker-confidence labels).
4. **Google Year-in-Search / OpenAI browser launch (late 2025)** — "soft insider" cases; useful for false-positive discrimination testing.

Framing note: Robin Hanson's argument that informed trading *is* what makes prediction markets accurate suggests scoring "price informativeness" separately from "policy-relevant insider flag" rather than conflating them.

## 4. Methodological pitfalls (drives Prompt 5 of the Phase-1 series)

- **Naive z-scores on [0,1] prices are miscalibrated**: variance of price changes shrinks mechanically near 0/1 (a 98¢ market cannot move up 2¢+) and depends on time-to-resolution. Fixed-window z-scores under-flag near 50¢ and over-flag near boundaries (or vice versa).
- **Fix: log-odds (logit) space.** `logit(p) = ln(p/(1-p))` maps (0,1)→ℝ; logit-returns are additive and are the natural space for Gaussian noise models, z-scores, and CUSUM. See arXiv:2510.15205 ("Toward Black-Scholes for Prediction Markets"), arXiv:2607.08199 (structural volatility model). Conditioning volatility on (price level, time-to-resolution) is the full fix; logit transform is the 80% first step.
- **False positives**: benchmark alerts against reference distributions matched on price level / time-to-resolution / liquidity tier; require multi-signal corroboration; separate "anomalous move" from "insider flag".
- **Backtesting without labels**: strict point-in-time discipline (no lookahead — never normalize volatility with a window spanning the event itself); use the documented cases above as a small validation set (precision/recall), never as training data; inter-detector agreement as a confidence proxy (arXiv:2410.14579).

## Deferred-phase candidates (in rough priority order)

1. Wallet-level analysis via Data API `/trades` `proxyWallet` (fresh-wallet, size-anomaly, timing features) — Phase 2.
2. Backtest harness replaying the Maduro/Iran cases point-in-time — Phase 2/3.
3. CUSUM/BOCPD change-point detector alongside logit z-scores — Phase 2.
4. WebSocket live capture (the existing `ws_market.py`/`ws_rtds.py` code) for order-event and depth history — Phase 3.
5. Lead-lag correlation wiring for `analysis/correlation.py` — Phase 3.
6. On-chain Polygon holder reconstruction (fixes the /holders 20-cap) — Phase 3.
