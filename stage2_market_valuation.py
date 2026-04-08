"""
Stage 2: Market Valuation Process
===================================
Scores target and benchmark lands across 10 weighted factors,
then derives the target's estimated price per sqm.

All weights, brackets, and thresholds are loaded from valuation_config.json.
"""

import pandas as pd
import numpy as np
import math
from stage1_benchmark_selection import (
    load_config, score_by_brackets, _pct_diff_score,
    _distance_km, _coord_score
)


# ══════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════

def _safe_float(record, key):
    """Safely extract a float from dict or Series."""
    val = record.get(key)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, str) and val.strip() == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _record_to_dict(row):
    """Convert a DataFrame row to a dict, handling NaN -> None."""
    d = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}


# ══════════════════════════════════════════════
# MULTIPLIER FUNCTIONS (config-driven)
# ══════════════════════════════════════════════

def _mult_address(record, target, config):
    """Address matching using config brackets."""
    brackets = config["scoring_brackets"]["address"]
    if record.get("city") != target.get("city"):
        return brackets["different_city"]
    if record.get("district") == target.get("district"):
        return brackets["same_district"]
    return brackets["same_city_diff_district"]


def _mult_area(record, target, config):
    """Percentage difference between comparable and target area."""
    t_area = _safe_float(target, "area_sqm")
    c_area = _safe_float(record, "area_sqm")
    score = _pct_diff_score(t_area, c_area, config)
    return score if score is not None else 0.00


def _mult_general_location(record, target, config):
    """Haversine distance between subject and comparable."""
    dist_km = _distance_km(
        _safe_float(target, "latitude"), _safe_float(target, "longitude"),
        _safe_float(record, "latitude"), _safe_float(record, "longitude"),
    )
    score = _coord_score(dist_km, config)
    if score is not None:
        return score
    # Fallback: district-based
    fallback = config["scoring_brackets"]["general_location_fallback"]
    if record.get("district") == target.get("district"):
        return fallback["same_district"]
    return fallback["different_district"]


def _mult_commercial_location(record, config):
    """Distance to commercial corridor or grade-based fallback."""
    dist = _safe_float(record, "dist_to_commercial_m")
    if dist is not None:
        return score_by_brackets(dist, "commercial_distance_m", config)
    # Fallback: grade-based
    grade = record.get("commercial_grade")
    grade_map = config["scoring_brackets"]["commercial_grade"]
    return grade_map.get(str(grade), 0.00)


def _mult_admin_location(record, config):
    """Distance to admin center or grade-based fallback."""
    dist = _safe_float(record, "dist_to_admin_m")
    if dist is not None:
        return score_by_brackets(dist, "admin_distance_m", config)
    grade = record.get("admin_grade")
    grade_map = config["scoring_brackets"]["admin_grade"]
    return grade_map.get(str(grade), 0.00)


def _mult_proximity_services(record, config):
    """Distance to services scored using config brackets."""
    dist = _safe_float(record, "dist_to_services_m")
    if dist is None:
        return 0.00
    return score_by_brackets(dist, "services_distance_m", config)


def _mult_proximity_main_road(record, config):
    """Distance to main road scored using config brackets."""
    dist = _safe_float(record, "dist_to_main_road_m")
    if dist is None:
        return 0.00
    return score_by_brackets(dist, "main_road_distance_m", config)


def _mult_plot_proportions(record, config):
    """Frontage-to-depth ratio scored using config brackets."""
    front = _safe_float(record, "frontage_m")
    depth = _safe_float(record, "depth_m")
    if front is None or depth is None or front == 0:
        return 0.00
    ratio = depth / front
    return score_by_brackets(ratio, "plot_ratio", config)


def _mult_far_csr(record, target, config):
    """
    Combined: average of zoning compatibility + FAR/CSR similarity.
    """
    zoning_compat = config["_zoning_compat"]
    t_zoning = target.get("zoning")
    c_zoning = record.get("zoning")
    if t_zoning and c_zoning:
        zoning_score = zoning_compat.get((t_zoning, c_zoning), 0.00)
    else:
        zoning_score = 0.00

    far_score = _pct_diff_score(_safe_float(target, "far"), _safe_float(record, "far"), config)
    csr_score = _pct_diff_score(_safe_float(target, "csr"), _safe_float(record, "csr"), config)

    if far_score is not None and csr_score is not None:
        far_csr_score = (far_score + csr_score) / 2
    elif far_score is not None:
        far_csr_score = far_score
    elif csr_score is not None:
        far_csr_score = csr_score
    else:
        return zoning_score

    return (zoning_score + far_csr_score) / 2


def _mult_utilities(record, config):
    """Count-based: k / 5."""
    count = _safe_float(record, "utilities_count")
    if count is None:
        return 0.00
    return min(count, 5) / 5


# ══════════════════════════════════════════════
# SCORING ENGINE
# ══════════════════════════════════════════════

# Maps factor_key -> function(record, target, config)
# Uniform signature: all accept (record, target, config)
def _get_factor_functions():
    return {
        "address":              lambda r, t, c: _mult_address(r, t, c),
        "area":                 lambda r, t, c: _mult_area(r, t, c),
        "general_location":     lambda r, t, c: _mult_general_location(r, t, c),
        "commercial_location":  lambda r, t, c: _mult_commercial_location(r, c),
        "admin_location":       lambda r, t, c: _mult_admin_location(r, c),
        "proximity_services":   lambda r, t, c: _mult_proximity_services(r, c),
        "proximity_main_road":  lambda r, t, c: _mult_proximity_main_road(r, c),
        "plot_proportions":     lambda r, t, c: _mult_plot_proportions(r, c),
        "far_csr":              lambda r, t, c: _mult_far_csr(r, t, c),
        "utilities":            lambda r, t, c: _mult_utilities(r, c),
    }


def score_property(record: dict, target: dict, config: dict) -> dict:
    """
    Score a single property across active factors only.
    Inactive factors are skipped and their weight redistributed.
    """
    weights = config["stage2_factors"]["weights"]
    is_active = config["stage2_factors"].get("is_active", {})
    factor_funcs = _get_factor_functions()

    # Filter to active factors and redistribute weights
    active_weights = {k: v for k, v in weights.items() if is_active.get(k, True)}
    weight_sum = sum(active_weights.values())

    result = {}
    total_points = 0.0

    for factor_key, raw_weight in active_weights.items():
        # Redistribute: scale so active weights sum to 1.0
        weight = raw_weight / weight_sum if weight_sum > 0 else 0
        func = factor_funcs[factor_key]
        multiplier = func(record, target, config)
        score = weight * multiplier
        total_points += score
        result[f"{factor_key}_multiplier"] = round(multiplier, 4)
        result[f"{factor_key}_score"] = round(score, 4)

    result["total_points"] = round(total_points, 4)
    return result


def run_valuation(target: dict, benchmarks: pd.DataFrame, config: dict) -> dict:
    """
    Full Stage 2 valuation:
      1. Score target across all factors
      2. Score each benchmark
      3. Point value per benchmark
      4. Weighted average (using Stage 1 similarity)
      5. Final target price
    """
    # Score target
    target_scores = score_property(target, target, config)

    # Score benchmarks
    benchmark_results = []
    for _, row in benchmarks.iterrows():
        rec = _record_to_dict(row)
        scores = score_property(rec, target, config)
        scores["land_id"] = rec["land_id"]
        scores["price_per_sqm"] = rec["price_per_sqm"]
        scores["similarity_score"] = rec.get("similarity_score", 1.0)

        if scores["total_points"] > 0:
            scores["point_value"] = round(rec["price_per_sqm"] / scores["total_points"], 4)
        else:
            scores["point_value"] = 0.0
        benchmark_results.append(scores)

    # Weighted average point value
    total_sim = sum(b["similarity_score"] for b in benchmark_results)
    if total_sim > 0:
        weighted_avg_pv = sum(
            b["similarity_score"] * b["point_value"] for b in benchmark_results
        ) / total_sim
    else:
        weighted_avg_pv = np.mean([b["point_value"] for b in benchmark_results])

    # Simple average for comparison
    simple_avg_pv = np.mean([b["point_value"] for b in benchmark_results])

    # Final prices
    target_price_weighted = weighted_avg_pv * target_scores["total_points"]
    target_price_simple = simple_avg_pv * target_scores["total_points"]

    target_area = _safe_float(target, "area_sqm")
    total_val_w = target_price_weighted * target_area if target_area else None
    total_val_s = target_price_simple * target_area if target_area else None

    return {
        "target_scores": target_scores,
        "benchmark_results": benchmark_results,
        "weighted_avg_point_value": round(weighted_avg_pv, 4),
        "simple_avg_point_value": round(simple_avg_pv, 4),
        "target_price_per_sqm_weighted": round(target_price_weighted, 2),
        "target_price_per_sqm_simple": round(target_price_simple, 2),
        "target_total_value_weighted": round(total_val_w, 2) if total_val_w else None,
        "target_total_value_simple": round(total_val_s, 2) if total_val_s else None,
        "weights_used": {
            k: v / sum(
                w for fk, w in config["stage2_factors"]["weights"].items()
                if config["stage2_factors"].get("is_active", {}).get(fk, True)
            )
            for k, v in config["stage2_factors"]["weights"].items()
            if config["stage2_factors"].get("is_active", {}).get(k, True)
        },
    }


# ══════════════════════════════════════════════
# OUTPUT & EXPORT
# ══════════════════════════════════════════════

def print_valuation_report(target: dict, result: dict):
    """Print a formatted valuation report."""
    weights = result["weights_used"]
    factors = list(weights.keys())

    print("\n" + "=" * 80)
    print("STAGE 2: MARKET VALUATION REPORT")
    print("=" * 80)

    ts = result["target_scores"]
    print(f"\n\U0001f3af TARGET LAND SCORING")
    print(f"{'Factor':<25} {'Weight':>8} {'Multiplier':>12} {'Score':>8}")
    print("\u2500" * 55)
    for f in factors:
        print(f"{f:<25} {weights[f]:>8.2f} {ts[f'{f}_multiplier']:>12.4f} {ts[f'{f}_score']:>8.4f}")
    print("\u2500" * 55)
    print(f"{'TOTAL POINTS':<25} {'':>8} {'':>12} {ts['total_points']:>8.4f}")

    print(f"\n\U0001f4ca BENCHMARK SCORING")
    for b in result["benchmark_results"]:
        print(f"\n  {b['land_id']} (price/sqm: {b['price_per_sqm']:,.2f} EGP, "
              f"similarity: {b['similarity_score']:.4f})")
        print(f"  {'Factor':<25} {'Mult':>6} {'Score':>8}")
        sep = "\u2500" * 41
        print(f"  {sep}")
        for f in factors:
            print(f"  {f:<25} {b[f'{f}_multiplier']:>6.2f} {b[f'{f}_score']:>8.4f}")
        print(f"  {sep}")
        print(f"  {'Total Points':<25} {'':>6} {b['total_points']:>8.4f}")
        print(f"  Point Value = {b['price_per_sqm']:,.2f} / {b['total_points']:.4f} "
              f"= {b['point_value']:,.4f} EGP/point")

    print(f"\n{'=' * 80}")
    print(f"\U0001f4b0 FINAL VALUATION")
    print(f"{'=' * 80}")
    print(f"\n  Target Total Points:           {ts['total_points']:.4f}")
    print(f"  Weighted Avg Point Value:       {result['weighted_avg_point_value']:,.4f} EGP/point")
    print(f"  Simple Avg Point Value:         {result['simple_avg_point_value']:,.4f} EGP/point")
    print(f"\n  \u2705 Estimated Price/sqm (weighted): {result['target_price_per_sqm_weighted']:>12,.2f} EGP")
    print(f"     Estimated Price/sqm (simple):   {result['target_price_per_sqm_simple']:>12,.2f} EGP")

    if result["target_total_value_weighted"]:
        area = _safe_float(target, "area_sqm")
        print(f"\n  \U0001f4d0 Area: {area:,.1f} sqm")
        print(f"  \u2705 Total Land Value (weighted):    {result['target_total_value_weighted']:>12,.2f} EGP")
        print(f"     Total Land Value (simple):      {result['target_total_value_simple']:>12,.2f} EGP")


def export_valuation(target: dict, result: dict, output_path: str):
    """Export full audit trail to CSV."""
    rows = []
    factors = list(result["weights_used"].keys())

    # Target row
    ts = result["target_scores"]
    row = {"record_type": "target", "land_id": "TARGET", "price_per_sqm": "N/A",
           "similarity_score": "N/A", "point_value": "N/A"}
    for f in factors:
        row[f"{f}_multiplier"] = ts[f"{f}_multiplier"]
        row[f"{f}_score"] = ts[f"{f}_score"]
    row["total_points"] = ts["total_points"]
    rows.append(row)

    # Benchmark rows
    for b in result["benchmark_results"]:
        row = {"record_type": "benchmark", "land_id": b["land_id"],
               "price_per_sqm": b["price_per_sqm"],
               "similarity_score": b["similarity_score"],
               "point_value": b["point_value"]}
        for f in factors:
            row[f"{f}_multiplier"] = b[f"{f}_multiplier"]
            row[f"{f}_score"] = b[f"{f}_score"]
        row["total_points"] = b["total_points"]
        rows.append(row)

    # Summary row
    rows.append({
        "record_type": "result",
        "land_id": "VALUATION",
        "price_per_sqm": result["target_price_per_sqm_weighted"],
        "similarity_score": "",
        "point_value": result["weighted_avg_point_value"],
        "total_points": result["target_scores"]["total_points"],
    })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

if __name__ == "__main__":
    from stage1_benchmark_selection import load_and_prepare, compute_similarity, format_and_export

    DATA_PATH = "data/land_transactions_egypt.csv"
    CONFIG_PATH = "valuation_config.json"
    STAGE1_OUTPUT = "output/stage1_benchmarks_output.csv"
    STAGE2_OUTPUT = "output/stage2_valuation_output.csv"

    # Load config
    config = load_config(CONFIG_PATH)

    target_land = {
        "city":                "Cairo",
        "district":            "Nasr City",
        "latitude":            30.05,
        "longitude":           31.35,
        "area_sqm":            400,
        "zoning":              "Residential",
        "road_width_m":        12,
        "far":                 1.5,
        "csr":                 0.6,
        "frontage_m":          20,
        "depth_m":             20,
        "commercial_grade":    "B",
        "admin_grade":         "C",
        "utilities_count":     4,
        "dist_to_commercial_m": 350,
        "dist_to_admin_m":     800,
        "dist_to_services_m":  250,
        "dist_to_main_road_m": 100,
    }

    # ── Stage 1 ──
    print("Running Stage 1...")
    df, _ = load_and_prepare(DATA_PATH, target_land["city"], config)
    df = compute_similarity(target_land, df, config)
    top_k = config["stage1_similarity"]["top_k"]
    benchmarks = df.head(top_k).copy()
    print(f"Selected {len(benchmarks)} benchmarks")

    format_and_export(target_land, benchmarks, STAGE1_OUTPUT)
    print(f"\U0001f4e4 Stage 1 output saved to {STAGE1_OUTPUT}\n")

    # ── Stage 2 ──
    result = run_valuation(target_land, benchmarks, config)
    print_valuation_report(target_land, result)

    export_valuation(target_land, result, STAGE2_OUTPUT)
    print(f"\n\U0001f4e4 Stage 2 output saved to {STAGE2_OUTPUT}")
    print("=" * 80)