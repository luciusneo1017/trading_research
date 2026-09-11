"""Rebuild panel.parquet from every per-ticker file in the store."""
from __future__ import annotations
from pathlib import Path
import pandas as pd




def build_panel(data_dir: Path, panel_name: str) -> pd.DataFrame:
    """Union of all per-ticker parquets on disk, keyed (ticker, field)."""

    paths = sorted(
        p for p in data_dir.glob("*.parquet")
        if p.name != panel_name
    )

    if not paths:
        raise SystemExit(f"no ticker files in {data_dir}")

    frames = {}

    for p in paths:
        df = pd.read_parquet(p)

        # Skip previously-built panel files
        if df.columns.nlevels != 1:
            continue

        if not df.empty:
            frames[p.stem] = df

    panel = pd.concat(
        frames.values(),
        axis=1,
        keys=frames.keys(),
        names=["ticker", "field"],
    ).sort_index()

    panel.to_parquet(data_dir / panel_name)

    return panel