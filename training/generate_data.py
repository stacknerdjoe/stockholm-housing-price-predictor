"""
Generates a synthetic dataset of Stockholm apartment listings.

Price model:
  price = price_per_sqm(area) * size + rooms_premium - fee_suppression + noise

  - size is the dominant driver, drawn from a distribution conditioned on rooms
    so the two are realistically correlated.
  - rooms adds a small premium on top of what size already captures.
  - monthlyFee has a mild suppression effect reflecting partial capitalisation
    of ongoing costs into the purchase price (Swedish bostadsrätt market norm).
  - Gaussian noise of ~10 % std keeps it from being a clean linear surface.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
N = 2_000

# ── Area price-per-sqm (SEK) ─────────────────────────────────────────────────
# Ordering based on well-known Stockholm market structure:
#   Östermalm/Vasastan → most expensive central areas
#   Södermalm/Norrmalm → upper-mid
#   Kungsholmen/Gärdet → mid
#   Hägersten          → inner-suburban, noticeably cheaper
#   Skärholmen/Hässelby → outer areas, cheapest in the set
AREA_PRICE_PER_SQM: dict[str, float] = {
    "Östermalm":  120_000,
    "Vasastan":   108_000,
    "Södermalm":   94_000,
    "Norrmalm":    91_000,
    "Kungsholmen": 86_000,
    "Gärdet":      83_000,
    "Hägersten":   67_000,
    "Skärholmen":  50_000,
    "Hässelby":    47_000,
}

# ── Room distribution ─────────────────────────────────────────────────────────
ROOM_WEIGHTS = {1: 0.10, 2: 0.35, 3: 0.35, 4: 0.15, 5: 0.05}

# Size (sqm) conditioned on rooms — creates realistic correlation.
# (mean, std) tuples per room count.
ROOM_SIZE_PARAMS: dict[int, tuple[float, float]] = {
    1: (32.0,  6.0),
    2: (52.0,  8.0),
    3: (72.0, 10.0),
    4: (92.0, 12.0),
    5: (116.0, 16.0),
}

ROOMS_PREMIUM_PER_ROOM = 45_000   # SEK — small bonus beyond what size explains
FEE_SUPPRESSION_COEFF  = 30       # SEK off price per SEK of monthly fee
NOISE_FRACTION         = 0.10     # Gaussian noise std as fraction of base price


def generate() -> pd.DataFrame:
    areas = list(AREA_PRICE_PER_SQM.keys())
    area_col = RNG.choice(areas, size=N, replace=True)

    rooms_vals = list(ROOM_WEIGHTS.keys())
    rooms_probs = list(ROOM_WEIGHTS.values())
    rooms_col = RNG.choice(rooms_vals, size=N, p=rooms_probs)

    # Size drawn from room-specific distribution; clip to plausible floor.
    size_col = np.array([
        max(16.0, RNG.normal(*ROOM_SIZE_PARAMS[r]))
        for r in rooms_col
    ])
    size_col = np.round(size_col, 1)

    # Monthly fee loosely correlated with size (larger flats → bigger buildings
    # → varied fees, but larger BRF share → higher absolute fee).
    fee_col = 2_000 + size_col * 18 + RNG.normal(0, 450, N)
    fee_col = np.clip(fee_col, 1_500, 7_000)
    fee_col = np.round(fee_col / 100) * 100  # round to nearest 100 SEK

    # Price model
    price_per_sqm = np.array([AREA_PRICE_PER_SQM[a] for a in area_col])
    base_price = (
        price_per_sqm * size_col
        + rooms_col * ROOMS_PREMIUM_PER_ROOM
        - fee_col * FEE_SUPPRESSION_COEFF
    )
    noise = RNG.normal(0, base_price * NOISE_FRACTION)
    price_col = np.round(base_price + noise, -3).astype(int)  # nearest 1 000
    price_col = np.maximum(price_col, 400_000)

    return pd.DataFrame({
        "area":       area_col,
        "rooms":      rooms_col.astype(int),
        "size":       size_col,
        "monthlyFee": fee_col.astype(int),
        "price":      price_col,
    })


def print_summary(df: pd.DataFrame) -> None:
    print(f"\nGenerated {len(df):,} rows\n")
    print(f"{'Area':<15}  {'Mean price':>13}  {'Median price':>13}  {'n':>5}")
    print("-" * 55)
    summary = (
        df.groupby("area")["price"]
        .agg(mean="mean", median="median", count="count")
        .sort_values("mean", ascending=False)
    )
    for area, row in summary.iterrows():
        print(
            f"{area:<15}  {row['mean']:>12,.0f}  {row['median']:>12,.0f}"
            f"  {row['count']:>5.0f}"
        )
    print()
    print(f"Overall price range: {df['price'].min():,.0f} – {df['price'].max():,.0f} SEK")
    print(f"Overall mean price:  {df['price'].mean():,.0f} SEK")


if __name__ == "__main__":
    df = generate()

    out = Path(__file__).parent.parent / "data" / "stockholm-housing.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"Saved -> {out}")

    print_summary(df)
