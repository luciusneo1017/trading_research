"""Rebuild panel.parquet from every per-ticker file in the store."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

PANEL_NAME = "commods_panel.parquet"


def build_panel(data_dir: Path) -> pd.DataFrame:
    """Union of all per-ticker parquets on disk, keyed (ticker, field).

    Reads the store rather than the current run, so a partial fetch can
    never truncate the panel.
    """
    paths = sorted(p for p in data_dir.glob("*.parquet") if p.name != PANEL_NAME)
    if not paths:
        raise SystemExit(f"no ticker files in {data_dir}")

    frames = {}
    for p in paths:
        df = pd.read_parquet(p)
        if not df.empty:
            frames[p.stem] = df

    panel = pd.concat(frames.values(), axis=1, keys=frames.keys()).sort_index()
    panel.to_parquet(data_dir / PANEL_NAME)
    return panel