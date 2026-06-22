"""
Exports models/housing-price-pipeline.joblib to ONNX format.

Inspects the actual training CSV to determine column dtypes before building
initial_types — nothing is guessed.  Each column becomes a separate ONNX input
tensor (required because the pipeline has mixed string + numeric types).

Outputs:
    models/housing-price-model.onnx
    models/model-metadata.json
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import onnx
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    DoubleTensorType,
    FloatTensorType,
    Int64TensorType,
    StringTensorType,
)

BASE          = Path(__file__).parent.parent
DATA_PATH     = BASE / "data"   / "stockholm-housing.csv"
PIPELINE_PATH = BASE / "models" / "housing-price-pipeline.joblib"
ONNX_PATH     = BASE / "models" / "housing-price-model.onnx"
META_PATH     = BASE / "models" / "model-metadata.json"

FEATURES = ["area", "rooms", "size", "monthlyFee"]
OPSET    = 17

# Maps pandas dtype string -> skl2onnx type class.
# pandas 3.0 changed object string columns to report as 'str'; handle both.
_DTYPE_TO_ONNX_CLS = {
    "object":          StringTensorType,
    "str":             StringTensorType,  # pandas 3.0 native string dtype
    "string":          StringTensorType,
    "string[python]":  StringTensorType,
    "int32":           Int64TensorType,   # widen to int64 for ONNX compatibility
    "int64":           Int64TensorType,
    "Int64":           Int64TensorType,   # pandas nullable integer
    "float32":         FloatTensorType,
    "float64":         DoubleTensorType,
    "Float64":         DoubleTensorType,  # pandas nullable float
}

# ONNX TensorProto.DataType enum -> readable name (for output display)
_ELEM_TYPE_NAME = {
    1: "float32", 2: "uint8",   3: "int8",   4: "uint16",
    5: "int16",   6: "int32",   7: "int64",  8: "string",
    9: "bool",   10: "float16", 11: "float64", 12: "uint32",
}


def inspect_dtypes() -> dict[str, str]:
    """Return {column: pandas-dtype-string} by reading the actual training CSV."""
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from CSV: {missing}")
    return {col: str(df[col].dtype) for col in FEATURES}


def build_initial_types(col_dtypes: dict[str, str]) -> list:
    """
    Build the skl2onnx initial_types list from actual column dtypes.
    Each feature column becomes one input tensor of shape [N, 1].
    """
    print("\nColumn dtypes -> ONNX initial_types:")
    initial_types = []
    for col in FEATURES:
        dtype_str = col_dtypes[col]
        cls = _DTYPE_TO_ONNX_CLS.get(dtype_str)
        if cls is None:
            raise ValueError(
                f"No ONNX type mapping for dtype {dtype_str!r} in column {col!r}.\n"
                f"Known dtypes: {list(_DTYPE_TO_ONNX_CLS)}"
            )
        onnx_type = cls([None, 1])
        print(f"  {col:<14} pandas:{dtype_str:<10} -> {type(onnx_type).__name__}")
        initial_types.append((col, onnx_type))
    return initial_types


def extract_area_categories(pipeline) -> list[str]:
    """Pull the exact category list from the fitted OneHotEncoder."""
    ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    return sorted(str(v) for v in ohe.categories_[0])


def format_tensor_info(tt) -> str:
    """Format an ONNX tensor_type proto as 'shape=[...] dtype=...'."""
    shape_parts = []
    for d in tt.shape.dim:
        if d.HasField("dim_value"):
            shape_parts.append(str(d.dim_value))
        elif d.HasField("dim_param"):
            shape_parts.append(d.dim_param)
        else:
            shape_parts.append("?")
    dtype_name = _ELEM_TYPE_NAME.get(tt.elem_type, str(tt.elem_type))
    return f"shape=[{', '.join(shape_parts)}]  dtype={dtype_name}"


def print_onnx_io(model: onnx.ModelProto) -> None:
    """Print every input and output node of the ONNX graph."""
    print("\nONNX inputs:")
    for inp in model.graph.input:
        info = format_tensor_info(inp.type.tensor_type)
        print(f"  name={inp.name!r:<22}  {info}")

    print("\nONNX outputs:")
    for out in model.graph.output:
        info = format_tensor_info(out.type.tensor_type)
        print(f"  name={out.name!r:<22}  {info}")


if __name__ == "__main__":
    print(f"Pipeline : {PIPELINE_PATH}")
    pipeline = joblib.load(PIPELINE_PATH)

    print(f"Data     : {DATA_PATH}")
    col_dtypes    = inspect_dtypes()
    initial_types = build_initial_types(col_dtypes)

    print(f"\nConverting to ONNX (opset={OPSET})...")
    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_types,
        target_opset=OPSET,
    )

    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ONNX_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())

    size_kb = ONNX_PATH.stat().st_size / 1024
    print(f"\nSaved    : {ONNX_PATH}")
    print(f"Size     : {size_kb:.0f} KB")

    loaded = onnx.load(str(ONNX_PATH))
    print_onnx_io(loaded)

    # Write metadata JSON -------------------------------------------------
    area_values = extract_area_categories(pipeline)
    metadata = {
        "version": "1.0.0",
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "features": [
            {"name": col, "dtype": col_dtypes[col]} for col in FEATURES
        ],
        "area_values": area_values,
        "onnx_opset": OPSET,
        "onnx_filename": ONNX_PATH.name,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nMetadata : {META_PATH}")
    print(f"  version     : {metadata['version']}")
    print(f"  trained_at  : {metadata['trained_at']}")
    print(f"  area_values : {area_values}")
