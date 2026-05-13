import types

import pandas as pd


class Panel:
    def __init__(self, data: dict[str, pd.DataFrame]) -> None:
        self.data = types.MappingProxyType(data)

    @property
    def universe(self) -> list[str]:
        return list(self.data["prices"].columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.data["prices"].index

    def universe_at(self, asof) -> list[str]:
        asof = pd.Timestamp(asof)
        prices = self.data["prices"]
        window = prices.loc[prices.index <= asof].iloc[-5:]
        return [col for col in prices.columns if window[col].notna().any()]

    def slice(
        self,
        asof,
        kind: str,
        lookback: int | None = None,
        freq: str = "daily",
        fill: str = "ffill",
    ) -> pd.DataFrame:
        asof = pd.Timestamp(asof)
        df = self.data[kind]
        result = df.loc[df.index <= asof]

        if freq != "daily":
            result = result.resample(freq).last()
            if fill == "ffill":
                result = result.ffill()

        if lookback is not None:
            result = result.iloc[-lookback:]

        return result.copy()
