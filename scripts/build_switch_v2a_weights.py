"""Build SWITCH(v2a) weights cache from regime-conditional weight selection.

v2a rule: {R0 → MSR(LW), R5 → MSR(sample), default → MDP(LW)}
Saves to data/cache/portfolio_weights/SWITCH_v2a.parquet
"""
from pathlib import Path

import pandas as pd

RULE = {0: "MSR_ledoit_wolf", 5: "MSR_sample"}
DEFAULT = "MDP_ledoit_wolf"
SUFFIX = "29assets_2003_2026"
WEIGHTS_DIR = Path("data/cache/portfolio_weights")
OUT_PATH = WEIGHTS_DIR / "SWITCH_v2a.parquet"


def main():
    w_msr_lw  = pd.read_parquet(WEIGHTS_DIR / f"MSR_ledoit_wolf_{SUFFIX}.parquet")
    w_msr_smp = pd.read_parquet(WEIGHTS_DIR / f"MSR_sample_{SUFFIX}.parquet")
    w_mdp_lw  = pd.read_parquet(WEIGHTS_DIR / f"MDP_ledoit_wolf_{SUFFIX}.parquet")

    regime_sig = pd.read_parquet("data/cache/regime_signals_2003_2026.parquet")
    dominant_regime = regime_sig["dominant_regime"].dropna()

    idx = w_msr_lw.index
    reb_dates = idx - pd.offsets.BDay(1)
    combined_idx = reb_dates.union(dominant_regime.index).sort_values()
    regime_at_reb = dominant_regime.reindex(combined_idx).ffill().reindex(reb_dates)
    regime_daily = pd.Series(regime_at_reb.values, index=idx)

    default_src = w_mdp_lw.reindex(idx)
    switch_w = default_src.copy()

    for regime_val, stem in RULE.items():
        mask = regime_daily == regime_val
        src = w_msr_lw if stem == "MSR_ledoit_wolf" else w_msr_smp
        switch_w[mask] = src.reindex(idx).loc[mask].values

    switch_w.to_parquet(OUT_PATH)
    print(f"Saved → {OUT_PATH}  shape={switch_w.shape}")
    print(f"Regime breakdown: { {regime_daily.value_counts().to_dict()} }")


if __name__ == "__main__":
    main()
