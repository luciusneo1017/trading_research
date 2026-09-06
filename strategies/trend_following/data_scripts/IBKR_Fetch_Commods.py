"""
IBKR TWS/Gateway historical data fetcher.

Pulls daily bars for a fixed ETF universe and caches to parquet.
Requires TWS or IB Gateway running with the API enabled.

    pip install ib_async pandas pyarrow
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
from ib_async import IB, Stock, util

log = logging.getLogger(__name__)

# --- config ---------------------------------------------------------------

HOST = "127.0.0.1"
PORT = 7497          # 7497 TWS paper | 7496 TWS live | 4002 GW paper | 4001 GW live
CLIENT_ID = 11       # must be unique per simultaneous connection

CACHE = Path("data/bars")

# IBKR allows 60 historical requests per 10 minutes. 11s spacing keeps us
# comfortably under it without needing a token bucket.
PACING_SECONDS = 11.0

UNIVERSE = {
    # ticker: primaryExchange
    "GLD": "ARCA",
    "SLV": "ARCA",
    "CPER": "ARCA",
    "USL": "ARCA",
    "UGA": "ARCA",
    "UNL": "ARCA",
    "CORN": "ARCA",
    "WEAT": "ARCA",
    "SOYB": "ARCA",
    "CANE": "ARCA",
    "KRBN": "ARCA",
}


# --- fetch ----------------------------------------------------------------


def fetch_one(
    ib: IB,
    symbol: str,
    exchange: str,
    duration: str = "15 Y",
    what: str = "ADJUSTED_LAST",
) -> pd.DataFrame:
    """Fetch daily bars for one symbol.

    whatToShow='ADJUSTED_LAST' returns a series adjusted for splits AND
    dividends -- which is what you want for signal generation. Note it only
    works with endDateTime='' (i.e. up to now); you cannot chunk historical
    windows with it. If you need chunked history, use 'TRADES' and apply
    your own adjustments from reqDividends / an external corporate-actions
    source.
    """
    contract = Stock(symbol, "SMART", "USD", primaryExchange=exchange)

    # Resolves ambiguity and confirms the contract actually exists.
    (qualified,) = ib.qualifyContracts(contract)

    bars = ib.reqHistoricalData(
        qualified,
        endDateTime="",
        durationStr=duration,
        barSizeSetting="1 day",
        whatToShow=what,
        useRTH=True,
        formatDate=1,
    )

    if not bars:
        raise RuntimeError(f"{symbol}: no bars returned")

    df = util.df(bars)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df[["open", "high", "low", "close", "volume"]]
    df.columns = pd.MultiIndex.from_product([[symbol], df.columns])
    return df


def fetch_universe(
    universe: dict[str, str] = UNIVERSE,
    cache: Path = CACHE,
    refresh: bool = False,
) -> pd.DataFrame:
    cache.mkdir(parents=True, exist_ok=True)

    ib = IB()
    ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=15)
    log.info("connected to %s:%s", HOST, PORT)

    frames: list[pd.DataFrame] = []
    try:
        for i, (symbol, exchange) in enumerate(universe.items()):
            path = cache / f"{symbol}.parquet"

            if path.exists() and not refresh:
                log.info("%s: cached", symbol)
                frames.append(pd.read_parquet(path))
                continue

            if i:
                time.sleep(PACING_SECONDS)

            try:
                df = fetch_one(ib, symbol, exchange)
            except Exception as exc:  # noqa: BLE001
                log.error("%s: %s", symbol, exc)
                continue

            df.to_parquet(path)
            log.info("%s: %d bars from %s", symbol, len(df), df.index[0].date())
            frames.append(df)
    finally:
        ib.disconnect()

    if not frames:
        raise RuntimeError("no data fetched")

    return pd.concat(frames, axis=1).sort_index()


# --- entry point ----------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    panel = fetch_universe()
    closes = panel.xs("close", axis=1, level=1)

    print(closes.tail())
    print()
    print("first valid observation per instrument:")
    print(closes.apply(lambda s: s.first_valid_index()).sort_values())