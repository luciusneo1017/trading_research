"""
Panel diagnostics and cleaning for the Tiingo EOD store.

Intended home: libs/data/panel_quality.py

Design: panel.parquet stays raw. Cleaning happens at load time and returns
both a price matrix and a validity mask, so nothing downstream has to guess
whether a value is a real observation or an artefact of the fill policy.
"""

from __future__ import annotations

import pandas as pd


# --- loading --------------------------------------------------------------


def load_field(panel_path, field: str = "adjClose") -> pd.DataFrame:
    """Extract one field from the wide (ticker, field) panel."""
    panel = pd.read_parquet(panel_path)
    return panel.xs(field, axis=1, level=1).sort_index()


# --- diagnostics ----------------------------------------------------------


def diagnose(prices: pd.DataFrame, volumes: pd.DataFrame | None = None,
             jump: float = 0.20) -> pd.DataFrame:
    """Per-ticker data quality report. Run this before trusting anything."""
    rets = prices.pct_change()
    rows = {}

    for t in prices.columns:
        s = prices[t]
        first, last = s.first_valid_index(), s.last_valid_index()

        # NaNs strictly inside the live window are gaps; leading NaNs are
        # just pre-inception and entirely expected.
        live = s.loc[first:last] if first is not None else s.iloc[:0]
        r = rets[t].loc[first:last] if first is not None else rets[t].iloc[:0]

        rows[t] = {
            "start": first.date() if first is not None else None,
            "end": last.date() if last is not None else None,
            "obs": int(live.notna().sum()),
            "interior_nan": int(live.isna().sum()),
            "zero_ret": int((r == 0).sum()),
            "zero_ret_pct": round(100 * (r == 0).mean(), 1) if len(r) else None,
            "max_stale_run": _longest_run(r == 0),
            "big_moves": int((r.abs() > jump).sum()),
            "nonpositive": int((live <= 0).sum()),
        }

        if volumes is not None and t in volumes.columns:
            v = volumes[t].loc[first:last] if first is not None else volumes[t].iloc[:0]
            rows[t]["zero_vol"] = int((v == 0).sum())

    return pd.DataFrame(rows).T


def _longest_run(mask: pd.Series) -> int:
    """Longest consecutive True run — catches stale-price stretches."""
    if mask.empty or not mask.any():
        return 0
    groups = (~mask).cumsum()[mask]
    return int(groups.value_counts().max())


# --- cleaning -------------------------------------------------------------


def clean(prices: pd.DataFrame, max_ffill: int = 3,
          min_obs: int = 250) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (filled prices, validity mask).

    The mask is True where the instrument is live and the price is a real
    observation or a short carry-forward. Position sizing should zero out
    anything the mask excludes rather than relying on NaN propagation.
    """
    prices = prices.sort_index()

    # drop instruments with too little history to estimate anything on
    keep = prices.notna().sum() >= min_obs
    prices = prices.loc[:, keep]

    observed = prices.notna()

    # never fill before an instrument exists
    live = observed.cummax()

    filled = prices.ffill(limit=max_ffill).where(live)
    mask = filled.notna() & live

    return filled, mask


def returns(filled: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Arithmetic returns, blanked wherever either endpoint is invalid."""
    r = filled.pct_change()
    valid = mask & mask.shift(1, fill_value=False)
    return r.where(valid)