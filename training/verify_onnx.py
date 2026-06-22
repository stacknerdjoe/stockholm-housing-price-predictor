"""
Verifies the ONNX export by comparing sklearn Pipeline predictions against
onnxruntime predictions on 5 representative samples.

For a RandomForestRegressor the predictions should be bit-for-bit identical or
differ by at most a few SEK (float32/float64 rounding inside the ONNX runtime).
Anything above WARN_THRESHOLD_SEK signals a genuine conversion problem.

Usage:
    python training/verify_onnx.py
Exit code 0 = PASS, 1 = FAIL (for use in CI).
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

BASE          = Path(__file__).parent.parent
PIPELINE_PATH = BASE / "models" / "housing-price-pipeline.joblib"
ONNX_PATH     = BASE / "models" / "housing-price-model.onnx"

# Five samples chosen to span price tiers and exercise different OHE categories.
SAMPLES: list[dict] = [
    {"area": "Östermalm",  "rooms": 2, "size": 55.0, "monthlyFee": 3_200},
    {"area": "Södermalm",  "rooms": 3, "size": 75.0, "monthlyFee": 4_100},
    {"area": "Vasastan",   "rooms": 1, "size": 32.0, "monthlyFee": 2_200},
    {"area": "Skärholmen", "rooms": 4, "size": 95.0, "monthlyFee": 5_500},
    {"area": "Hägersten",  "rooms": 2, "size": 60.0, "monthlyFee": 3_800},
]

# Differences above this threshold indicate a real problem (wrong feature order,
# OHE category mismatch, dtype coercion, etc.), not just float rounding.
WARN_THRESHOLD_SEK = 100_000

# Maps onnxruntime type strings to numpy dtypes for feed-dict construction.
_ORT_TYPE_TO_NP = {
    "tensor(float)":   np.float32,
    "tensor(double)":  np.float64,
    "tensor(int64)":   np.int64,
    "tensor(int32)":   np.int32,
    "tensor(string)":  object,
}

# Source-of-truth column values pulled from the sample dicts.
_COL_KEYS = ["area", "rooms", "size", "monthlyFee"]


def sklearn_predict(pipeline, samples: list[dict]) -> np.ndarray:
    df = pd.DataFrame(samples)[_COL_KEYS]
    return pipeline.predict(df).astype(np.float64)


def build_ort_feed(sess: ort.InferenceSession, samples: list[dict]) -> dict:
    """
    Build the onnxruntime feed dict by reading the model's declared input names
    and types, then matching them to the sample dicts.

    Each input is a 2-D array of shape [N, 1] to match the [None, 1] shapes
    declared in initial_types during export.
    """
    col_values = {k: [s[k] for s in samples] for k in _COL_KEYS}
    feed = {}
    for inp in sess.get_inputs():
        name = inp.name
        if name not in col_values:
            raise ValueError(
                f"ONNX model declares input {name!r} but no matching column "
                f"found in SAMPLES. Known columns: {_COL_KEYS}"
            )
        dtype = _ORT_TYPE_TO_NP.get(inp.type, object)
        feed[name] = np.array([[v] for v in col_values[name]], dtype=dtype)
    return feed


def ort_predict(sess: ort.InferenceSession, samples: list[dict]) -> np.ndarray:
    feed    = build_ort_feed(sess, samples)
    outputs = sess.run(None, feed)
    # output[0] may be [N] or [N, 1] depending on skl2onnx version
    return np.asarray(outputs[0], dtype=np.float64).ravel()


if __name__ == "__main__":
    print(f"sklearn pipeline : {PIPELINE_PATH}")
    pipeline = joblib.load(PIPELINE_PATH)

    print(f"ONNX model       : {ONNX_PATH}")
    sess = ort.InferenceSession(str(ONNX_PATH))

    print(f"\nONNX declared inputs  : {[i.name for i in sess.get_inputs()]}")
    print(f"ONNX declared outputs : {[o.name for o in sess.get_outputs()]}")
    print()

    sk_preds  = sklearn_predict(pipeline, SAMPLES)
    ort_preds = ort_predict(sess, SAMPLES)
    diffs     = np.abs(sk_preds - ort_preds)

    # ── Results table ─────────────────────────────────────────────────────────
    HDR = (
        f"{'#':<3}  {'Area':<14}  {'Rooms':>5}  {'Size':>6}  {'Fee':>6}"
        f"  {'sklearn (SEK)':>16}  {'ONNX (SEK)':>16}  {'|Diff|':>12}"
    )
    print(HDR)
    print("-" * len(HDR))

    has_failure = False
    for i, (s, sk, op, diff) in enumerate(zip(SAMPLES, sk_preds, ort_preds, diffs), 1):
        flag = "  <-- MISMATCH" if diff > WARN_THRESHOLD_SEK else ""
        if diff > WARN_THRESHOLD_SEK:
            has_failure = True
        print(
            f"{i:<3}  {s['area']:<14}  {s['rooms']:>5}  {s['size']:>6.1f}"
            f"  {s['monthlyFee']:>6}"
            f"  {sk:>16,.0f}  {op:>16,.0f}  {diff:>12,.2f}{flag}"
        )

    # ── Export-fidelity summary ───────────────────────────────────────────────
    print()
    print(f"Max |diff|  : {diffs.max():>14,.2f} SEK")
    print(f"Mean |diff| : {diffs.mean():>14,.2f} SEK")

    # ── Unknown-area / diacritic smoke test ───────────────────────────────────
    #
    # The original API spec example uses "Sodermalm" (no diacritic).
    # Training data uses "Södermalm".  handle_unknown="ignore" silently zeros
    # the entire OHE row for any value not seen at fit time, so the model
    # receives a neighbourhood signal of all-zeros and produces a price driven
    # only by the numeric features.  This section confirms that hypothesis.
    #
    DIVIDER = "=" * len(HDR)
    print(f"\n{DIVIDER}")
    print("UNKNOWN AREA SPELLING TEST")
    print(DIVIDER)

    BASE_FEATURES = {"rooms": 3, "size": 75.0, "monthlyFee": 4_100}
    s_correct = {"area": "Södermalm", **BASE_FEATURES}
    s_unknown  = {"area": "Sodermalm", **BASE_FEATURES}   # API-spec spelling, no ö

    sk_correct  = sklearn_predict(pipeline, [s_correct])[0]
    sk_unknown  = sklearn_predict(pipeline,  [s_unknown])[0]
    ort_correct = ort_predict(sess, [s_correct])[0]
    ort_unknown  = ort_predict(sess,  [s_unknown])[0]

    print(f"\nNumeric features held constant: rooms={BASE_FEATURES['rooms']}, "
          f"size={BASE_FEATURES['size']} m2, fee={BASE_FEATURES['monthlyFee']} SEK/mo\n")
    print(f"{'Area value':<26}  {'sklearn (SEK)':>16}  {'ONNX (SEK)':>16}")
    print("-" * 64)
    print(f"{'Sodermalm  (no diacritic)':<26}  {sk_unknown:>16,.0f}  {ort_unknown:>16,.0f}")
    print(f"{'Södermalm  (training spelling)':<26}  {sk_correct:>16,.0f}  {ort_correct:>16,.0f}")

    drop = sk_correct - sk_unknown
    print(f"\n  Price shift  : {drop:>+14,.0f} SEK  ({drop / sk_correct * 100:+.1f}% vs correct spelling)")

    # Show the raw OHE encoding to make the zeroing explicit.
    ohe        = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    categories = list(ohe.categories_[0])

    def ohe_row(value: str) -> list[int]:
        # Pass a DataFrame so the OHE sees the column name it was fitted with.
        df_single = pd.DataFrame({"area": [value]})
        result = ohe.transform(df_single)
        if hasattr(result, "toarray"):
            result = result.toarray()
        return list(map(int, result[0]))

    row_correct = ohe_row("Södermalm")
    row_unknown  = ohe_row("Sodermalm")

    col_w = max(len(c) for c in categories)
    print(f"\n  OHE encoding (one column per trained area):")
    print(f"  {'Area':<{col_w}}  {'Södermalm':>10}  {'Sodermalm':>10}")
    print(f"  {'-' * col_w}  {'----------':>10}  {'----------':>10}")
    for cat, v_correct, v_unknown in zip(categories, row_correct, row_unknown):
        marker = "  <-- this bit" if v_correct == 1 else ""
        print(f"  {cat:<{col_w}}  {v_correct:>10}  {v_unknown:>10}{marker}")

    if all(v == 0 for v in row_unknown):
        print(
            "\n  CONFIRMED: 'Sodermalm' produces an all-zero OHE row.\n"
            "  The model receives zero neighbourhood signal and falls back to\n"
            "  a price determined only by rooms/size/monthlyFee.\n"
            "\n"
            "  Fix required before production: normalise incoming area strings\n"
            "  to the exact trained spellings (e.g. via a lookup map in the\n"
            "  API controller) before passing them to the inference session."
        )

    # ── Final PASS / FAIL ─────────────────────────────────────────────────────
    print()
    if has_failure:
        print("FAIL -- one or more export-fidelity predictions diverge beyond threshold.")
        print("Likely causes:")
        print("  - Feature columns fed to ONNX in wrong order")
        print("  - OHE category list mismatch (area value not seen during training)")
        print("  - dtype coercion changing a numeric column's values")
        sys.exit(1)
    else:
        print("PASS -- sklearn and ONNX predictions agree on all 5 export-fidelity samples.")
