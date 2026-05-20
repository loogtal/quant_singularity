"""OHLCV to DataFrame with standard columns."""

import pandas as pd


def ohlcv_to_df(candles: list) -> pd.DataFrame:
    df = pd.DataFrame(
        candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    return df
