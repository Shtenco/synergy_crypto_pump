from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HorizonSpec:
    name: str
    horizon_bars: int
    return_threshold: float

    @property
    def target_col(self) -> str:
        return f"target__{self.name}"

    @property
    def mfe_col(self) -> str:
        return f"mfe__{self.name}"

    @property
    def mae_col(self) -> str:
        return f"mae__{self.name}"

    @property
    def end_col(self) -> str:
        return f"end_return__{self.name}"

    @property
    def hit_col(self) -> str:
        return f"hit_bar__{self.name}"


# Defaults assume 5-minute bars.
DEFAULT_HORIZONS: tuple[HorizonSpec, ...] = (
    HorizonSpec("up10_15m", 3, 0.10),
    HorizonSpec("up20_1h", 12, 0.20),
    HorizonSpec("up50_6h", 72, 0.50),
    HorizonSpec("up100_24h", 288, 1.00),
)


@dataclass(slots=True)
class V2Config:
    benchmark_symbol: str = "BTCUSDT"
    feature_windows: tuple[int, ...] = (3, 6, 12, 24, 48, 96)
    z_window: int = 96
    bocd_hazard_bars: int = 40
    bocd_max_run_length: int = 128
    sequence_length: int = 48
    min_train_fraction: float = 0.45
    validation_fraction: float = 0.10
    embargo_bars: int = 12
    n_splits: int = 3
    top_k: int = 10

    @property
    def purge_bars(self) -> int:
        return max(s.horizon_bars for s in DEFAULT_HORIZONS)
