"""
Tiingo end-of-day fetcher.

Location:  repos/trading_research/data_scripts/fetch_tiingo.py
Config:    repos/trading_research/.env
Output:    repos/data/tiingo/<TICKER>.parquet
           repos/data/tiingo/panel.parquet

The data store lives outside the repo so that several strategy packages can
read the same panel without either duplicating it or committing a few hundred
megabytes of parquet to git history. Its location is read from TIINGO_DATA_DIR
and falls back to a sibling `data/tiingo` directory next to the repo.

Stores raw OHLCV alongside adjusted OHLCV and the corporate-action columns
(divCash, splitFactor) so adjustments stay reproducible: you can always
rebuild an adjusted series from the raw one, but not the reverse.

Setup:
    uv add requests pandas pyarrow python-dotenv

    # repos/trading_research/.env
    TIINGO_API_KEY=your_key_here
    TIINGO_DATA_DIR=C:/Users/luciu/repos/data/tiingo/commods

Usage:
    uv run python data_scripts/fetch_tiingo.py             # incremental update
    uv run python data_scripts/fetch_tiingo.py --full      # force full refetch
    uv run python data_scripts/fetch_tiingo.py --tickers GLD SLV
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

log = logging.getLogger("tiingo")

# --- paths ----------------------------------------------------------------
# this file:  repos/trading_research/data_scripts/fetch_tiingo.py
#   parents[0] = repos/trading_research/data_scripts
#   parents[1] = repos/trading_research      <- repo root, holds .env
#   parents[2] = repos                       <- data/ sits alongside the repo
REPO_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_DIR / ".env"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "tiingo" / "commods"

# --- config ---------------------------------------------------------------

BASE_URL = "https://api.tiingo.com/tiingo/daily"
START_DATE = "2004-01-01"          # before the earliest inception in the universe
PACING_SECONDS = 1.0               # free tier: 50 req/hr, 1000/day. Nowhere near.
TIMEOUT = 30

UNIVERSE = [
    "GLD",    # gold, physical
    "SLV",    # silver, physical
    "CPER",   # copper
    "USL",    # WTI, 12-month strip
    "UGA",    # gasoline
    "UNL",    # natural gas, 12-month strip
    "CORN",
    "WEAT",
    "SOYB",
    "CANE",   # sugar
    "KRBN",   # carbon allowances
]

COLUMNS = [
    "open", "high", "low", "close", "volume",
    "adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume",
    "divCash", "splitFactor",
]


# --- environment ----------------------------------------------------------


def load_env() -> None:
    """Load the repo-root .env.

    load_dotenv is given an explicit path rather than relying on its upward
    search, which starts from the current working directory and therefore
    breaks whenever the script is invoked from somewhere other than the root.
    Variables already exported in the shell win, which is the sensible default.
    """
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
        log.debug("loaded %s", ENV_PATH)
    else:
        log.warning("no .env found at %s", ENV_PATH)


def resolve_token() -> str:
    token = os.environ.get("TIINGO_API_KEY")
    if not token:
        raise SystemExit(
            f"TIINGO_API_KEY not set.\n"
            f"Add it to {ENV_PATH} as:\n"
            f"    TIINGO_API_KEY=your_key_here"
        )
    return token


def resolve_data_dir(override: str | None = None) -> Path:
    """Data store location: CLI flag, then TIINGO_DATA_DIR, then the default."""
    raw = override or os.environ.get("TIINGO_DATA_DIR")
    return Path(raw).expanduser().resolve() if raw else DEFAULT_DATA_DIR


# --- fetch ----------------------------------------------------------------


def _session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "Content-Type": "application/json",
        "Authorization": f"Token {token}",
    })
    return s


def fetch(
    session: requests.Session,
    ticker: str,
    start: str = START_DATE,
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch daily bars for one ticker."""
    params = {"startDate": start, "format": "json"}
    if end:
        params["endDate"] = end

    resp = session.get(f"{BASE_URL}/{ticker}/prices", params=params, timeout=TIMEOUT)

    if resp.status_code == 404:
        raise RuntimeError(f"{ticker}: not found on Tiingo")
    if resp.status_code == 429:
        raise RuntimeError("rate limited -- back off and retry")
    resp.raise_for_status()

    rows = resp.json()
    if not rows:
        return pd.DataFrame(columns=COLUMNS)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
    df = df.set_index("date").sort_index()

    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"{ticker}: response missing {missing}")

    return df[COLUMNS]


def update_ticker(
    session: requests.Session,
    ticker: str,
    data_dir: Path,
    full: bool = False,
) -> pd.DataFrame:
    """Fetch or incrementally extend one ticker's history.

    Tiingo back-adjusts historical prices when a dividend or split occurs,
    so if the incremental slice contains a corporate action the cached
    adjusted series is stale and the whole history has to be refetched.
    """
    path = data_dir / f"{ticker}.parquet"

    if full or not path.exists():
        df = fetch(session, ticker)
        df.to_parquet(path)
        log.info("%-5s full   %5d bars  %s -> %s",
                 ticker, len(df), df.index[0].date(), df.index[-1].date())
        return df

    cached = pd.read_parquet(path)
    resume = (cached.index[-1] + timedelta(days=1)).strftime("%Y-%m-%d")

    if resume > date.today().isoformat():
        log.info("%-5s up to date (%s)", ticker, cached.index[-1].date())
        return cached

    delta = fetch(session, ticker, start=resume)
    if delta.empty:
        log.info("%-5s no new bars", ticker)
        return cached

    corporate_action = (delta["divCash"] > 0).any() or (delta["splitFactor"] != 1).any()
    if corporate_action:
        log.info("%-5s corporate action in delta -- refetching full history", ticker)
        df = fetch(session, ticker)
        df.to_parquet(path)
        return df

    df = pd.concat([cached, delta])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_parquet(path)
    log.info("%-5s +%-4d bars  through %s", ticker, len(delta), df.index[-1].date())
    return df




# --- entry point ----------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch EOD data from Tiingo.")
    parser.add_argument("--full", action="store_true",
                        help="refetch complete history, ignoring cache")
    parser.add_argument("--tickers", nargs="*", default=UNIVERSE,
                        help="override the default universe")
    parser.add_argument("--data-dir", default=None,
                        help="override TIINGO_DATA_DIR for this run")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    load_env()
    token = resolve_token()
    data_dir = resolve_data_dir(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    log.info("data store: %s", data_dir)

    session = _session(token)

    frames: dict[str, pd.DataFrame] = {}
    for i, ticker in enumerate(args.tickers):
        if i:
            time.sleep(PACING_SECONDS)
        try:
            frames[ticker] = update_ticker(session, ticker, data_dir, full=args.full)
        except Exception as exc:  # noqa: BLE001
            log.error("%-5s FAILED: %s", ticker, exc)

    if not frames:
        raise SystemExit("nothing fetched")




if __name__ == "__main__":
    main()
