"""Point-in-time replay harness for archived market surveillance cases.

A *case* is a directory (see :mod:`prediction_market.backtest.case_format`)
capturing one archived market's snapshots, trades, and scheduled events.
:func:`prediction_market.backtest.replay.replay_case` drives the real
:class:`~prediction_market.agents.info_leak_detector.InfoLeakDetector`
through that case's snapshots strictly in time order, against a throwaway
SQLite database, so every result is reproducible and free of lookahead.
"""

from __future__ import annotations
