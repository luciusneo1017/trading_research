"""
Tiingo end-of-day fetcher -- equity index ETFs.

Location:  repos/trading_research/libs/data_scripts/fetch_tiingo_equity.py
Config:    repos/trading_research/.env
Output:    repos/data/tiingo/equity/<TICKER>.parquet
           repos/data/tiingo/equity/panel.parquet

Usage:
    uv run python libs/data_scripts/fetch_tiingo_equity.py
    uv run python libs/data_scripts/fetch_tiingo_equity.py --full
    uv run python libs/data_scripts/fetch_tiingo_equity.py --tickers SPY QQQ
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
# this file:  repos/trading_research/libs/data_scripts/fetch_tiingo_equity.py
#   parents[0] = repos/trading_research/libs/data_scripts
#   parents[1] = repos/trading_research/libs
#   parents[2] = repos/trading_research      <- repo root, holds .env
#   parents[3] = repos                       <- data/ sits alongside the repo
REPO_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_DIR / ".env"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "tiingo" / "equity"

# --- config ---------------------------------------------------------------

BASE_URL = "https://api.tiingo.com/tiingo/daily"
START_DATE = "1993-01-01"          # SPY inception is Jan 1993, the earliest here
PACING_SECONDS = 1.0
TIMEOUT = 30
PANEL_NAME = "panel.parquet"

UNIVERSE = [
    "SPY",    # US large cap
    "QQQ",    # US tech / growth
    "IWM",    # US small cap
    "EFA",    # developed ex-US
    "EWJ",    # Japan
    "EEM",    # emerging markets
]

COLUMNS = [
    "open", "high", "low", "close", "volume",
    "adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume",
    "divCash", "splitFactor",
]


# --- environment ----------------------------------------------------------


def load_env() -> None:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
    else:
        log.warning("no .env found at %s", ENV_PATH)


def resolve_token() -> str:
    token = os.environ.get("TIINGO_API_KEY")
    if not token:
        raise SystemExit(f"TIINGO_API_KEY not set. Add it to {ENV_PATH}")
    return token


def resolve_data_dir(override: str | None = None) -> Path:
    raw = override or os.environ.get("TIINGO_EQUITY_DIR")
    return Path(raw).expanduser().resolve() if raw else DEFAULT_DATA_DIR


# --- fetch ----------------------------------------------------------------


def _session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "Content-Type": "application/json",
        "Authorization": f"Token {token}",
    })
    return s


def fetch(session: requests.Session, ticker: str,
          start: str = START_DATE, end: str | None = None) -> pd.DataFrame:
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
    # each .dt call consumes the accessor and returns a plain Series, so
    # chaining a second datetime method needs .dt again
    df["date"] = (
        pd.to_datetime(df["date"], utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )
    df = df.set_index("date").sort_index()

    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"{ticker}: response missing {missing}")

    return df[COLUMNS]


def update_ticker(session: requests.Session, ticker: str,
                  data_dir: Path, full: bool = False) -> None:
    """Fetch or incrementally extend one ticker.

    Tiingo back-adjusts history on a dividend or split, so a delta containing
    a corporate action means the cached adjusted series is stale and the whole
    history has to be refetched. Equity ETFs pay quarterly, so this fires
    roughly four times a year per ticker.
    """
    path = data_dir / f"{ticker}.parquet"

    if full or not path.exists():
        df = fetch(session, ticker)
        df.to_parquet(path)
        log.info("%-5s full   %5d bars  %s -> %s",
                 ticker, len(df), df.index[0].date(), df.index[-1].date())
        return

    cached = pd.read_parquet(path)
    resume = (cached.index[-1] + timedelta(days=1)).strftime("%Y-%m-%d")

    if resume > date.today().isoformat():
        log.info("%-5s up to date (%s)", ticker, cached.index[-1].date())
        return

    delta = fetch(session, ticker, start=resume)
    if delta.empty:
        log.info("%-5s no new bars", ticker)
        return

    if (delta["divCash"] > 0).any() or (delta["splitFactor"] != 1).any():
        log.info("%-5s corporate action -- refetching full history", ticker)
        fetch(session, ticker).to_parquet(path)
        return

    df = pd.concat([cached, delta])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_parquet(path)
    log.info("%-5s +%-4d bars  through %s", ticker, len(delta), df.index[-1].date())


def build_panel(data_dir: Path) -> pd.DataFrame:
    """Rebuild the panel from every ticker file on disk.

    Reads the store rather than the current run, so fetching a subset with
    --tickers can never truncate the panel.
    """
    paths = sorted(p for p in data_dir.glob("*.parquet") if p.name != PANEL_NAME)
    if not paths:
        raise SystemExit(f"no ticker files in {data_dir}")

    frames = {p.stem: pd.read_parquet(p) for p in paths}
    frames = {k: v for k, v in frames.items() if not v.empty}

    panel = pd.concat(frames.values(), axis=1, keys=frames.keys()).sort_index()
    panel.columns.names = ["ticker", "field"]
    panel.index.name = "date"
    panel.to_parquet(data_dir / PANEL_NAME)
    return panel


# --- entry point ----------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch equity ETF EOD data from Tiingo.")
    parser.add_argument("--full", action="store_true",
                        help="refetch complete history, ignoring cache")
    parser.add_argument("--tickers", nargs="*", default=UNIVERSE)
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    load_env()
    token = resolve_token()
    data_dir = resolve_data_dir(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    log.info("data store: %s", data_dir)

    session = _session(token)

    for i, ticker in enumerate(args.tickers):
        if i:
            time.sleep(PACING_SECONDS)
        try:
            update_ticker(session, ticker, data_dir, full=args.full)
        except Exception as exc:  # noqa: BLE001
            log.error("%-5s FAILED: %s", ticker, exc)

    panel = build_panel(data_dir)
    closes = panel.xs("adjClose", axis=1, level="field")

    print("\ninception dates:")
    print(closes.apply(lambda s: s.first_valid_index()).sort_values().to_string())
    print(f"\ncommon sample starts: {closes.dropna().index[0].date()}")
    print(f"rows: {len(closes)}   written to: {data_dir}")


if __name__ == "__main__":
    main()