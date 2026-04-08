"""
Stage 1: Benchmark Land Parcel Selection
=========================================
Revised pipeline:
  Step 1a: Basic filters (asset type, nulls, valid price/area/zoning/road)
  Step 1b: Filter by target city
  Step 1c: Remove price outliers within city
  Step 2:  Derived metrics (price_per_sqm, area_category)
  Step 3:  Similarity scoring (non-price) → top 5
  Step 4:  Validation (no duplicates / too-identical pairs)
  Step 5:  Output

All weights, brackets, and thresholds are loaded from valuation_config.json.
"""

import pandas as pd
import numpy as np
import math
import json
import os


# ══════════════════════════════════════════════
# CONFIG LOADER
# ══════════════════════════════════════════════

def load_config(path: str = None) -> dict:
    """Load configuration from JSON. Falls back to default path."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "valuation_config.json")
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Parse zoning compatibility keys from "A|B" → ("A", "B")
    cfg["_zoning_compat"] = {
        tuple(k.split("|")): v
        for k, v in cfg["zoning_compatibility"].items()
    }
    cfg["_grade_map"] = cfg["scoring_brackets"]["grade_ordinal"]

    return cfg


# ══════════════════════════════════════════════
# BRACKET SCORER (generic, config-driven)
# ══════════════════════════════════════════════

def score_by_brackets(value, bracket_key, config):
    """
    Generic bracket scorer: given a value and a bracket key in config,
    returns the appropriate score.
    Brackets are defined as:
      thresholds: [t1, t2, t3, ...]
      scores:     [s0, s1, s2, ..., s_last]
    If value <= t1 → s0, elif value <= t2 → s1, ... else → s_last
    """
    bracket = config["scoring_brackets"][bracket_key]
    thresholds = bracket["thresholds"]
    scores = bracket["scores"]
    for i, t in enumerate(thresholds):
        if value <= t:
            return scores[i]
    return scores[-1]


# ══════════════════════════════════════════════
# STEPS 1-2: Data Preparation
# ══════════════════════════════════════════════

def load_and_prepare(path: str, target_city: str, config: dict) -> tuple[pd.DataFrame, dict]:
    """
    Step 1a: Basic filters
    Step 1b: Filter by target city
    Step 1c: Outlier removal within city
    Step 2:  Derived metrics
    """
    report = {}

    # ── Load (handle BOM and whitespace in column names) ──
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    report["raw_count"] = len(df)

    # ── Step 1a: Basic filters ──
    rejections = {
        "not_land":             (df["asset_type"] != "Land").sum(),
        "missing_or_zero_price": df["total_price_egp"].isna().sum()
            + (df["total_price_egp"].fillna(0) <= 0).sum(),
        "missing_or_zero_area": df["area_sqm"].isna().sum()
            + (df["area_sqm"].fillna(0) <= 0).sum(),
        "missing_zoning":       df["zoning"].isna().sum(),
        "missing_road_width":   df["road_width_m"].isna().sum(),
    }

    df = (
        df.query("asset_type == 'Land'")
          .dropna(subset=["total_price_egp", "area_sqm", "zoning", "road_width_m"])
          .query("total_price_egp > 0 and area_sqm > 0")
          .copy()
    )
    report["after_basic_filter"] = len(df)

    # ── Step 1b: Filter by target city ──
    before_city = len(df)
    df = df[df["city"] == target_city].copy()
    rejections["different_city"] = before_city - len(df)
    report["after_city_filter"] = len(df)

    # ── Step 1c: Outlier removal within city (thresholds from config) ──
    df["price_per_sqm"] = df["total_price_egp"] / df["area_sqm"]
    lower_pct = config["outlier_removal"]["lower_percentile"]
    upper_pct = config["outlier_removal"]["upper_percentile"]
    p_low = df["price_per_sqm"].quantile(lower_pct)
    p_high = df["price_per_sqm"].quantile(upper_pct)
    before_outlier = len(df)
    df = df[df["price_per_sqm"].between(p_low, p_high)].copy()
    rejections["outlier_price"] = before_outlier - len(df)

    report["step1"] = {
        "rejections": rejections,
        "outlier_thresholds": {"p_low": round(p_low, 0), "p_high": round(p_high, 0)},
        "clean_count": len(df),
    }

    # ── Step 2: Derived metrics (thresholds from config) ──
    area_lower = config["area_categories"]["lower_percentile"]
    area_upper = config["area_categories"]["upper_percentile"]
    p33 = df["area_sqm"].quantile(area_lower)
    p66 = df["area_sqm"].quantile(area_upper)
    df["area_category"] = pd.cut(
        df["area_sqm"],
        bins=[-np.inf, p33, p66, np.inf],
        labels=["Small", "Medium", "Large"],
    )

    df["fd_ratio"] = np.where(
        (df["frontage_m"].notna()) & (df["depth_m"].notna()) & (df["frontage_m"] > 0),
        df["depth_m"] / df["frontage_m"],
        np.nan,
    )

    report["step2"] = {
        "area_thresholds": {"p33": round(p33, 0), "p66": round(p66, 0)},
        "distribution": df["area_category"].value_counts().to_dict(),
    }

    return df, report


# ══════════════════════════════════════════════
# STEP 3: Similarity Scoring
# ══════════════════════════════════════════════

def _pct_diff_score(target_val, candidate_val, config):
    """Percentage difference scored using config brackets."""
    if pd.isna(target_val) or pd.isna(candidate_val):
        return None
    if target_val == 0 and candidate_val == 0:
        return 1.0
    if target_val == 0 or candidate_val == 0:
        return 0.0
    diff = abs(target_val - candidate_val) / max(target_val, candidate_val) * 100
    return score_by_brackets(diff, "pct_diff", config)


def _distance_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km."""
    if any(pd.isna(v) for v in [lat1, lon1, lat2, lon2]):
        return None
    r = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _coord_score(dist_km, config):
    """Distance in km scored using config brackets."""
    if dist_km is None:
        return None
    return score_by_brackets(dist_km, "distance_km", config)


def _grade_score(target_grade, candidate_grade, config):
    """Score based on distance between ordinal grades."""
    grade_map = config["_grade_map"]
    if not target_grade or not candidate_grade:
        return None
    t = grade_map.get(str(target_grade))
    c = grade_map.get(str(candidate_grade))
    if t is None or c is None:
        return None
    diff = abs(t - c)
    return max(0, 1 - diff * 0.25)


def _recency_score(tx_date, ref_date):
    """Score 0-1: transactions within 3 years, linear decay."""
    if pd.isna(tx_date):
        return None
    days = (ref_date - pd.Timestamp(tx_date)).days
    if days < 0:
        return 1.0
    return max(0, 1 - days / 1095)


def compute_similarity(target: dict, candidates: pd.DataFrame,
                       config: dict, ref_date=None) -> pd.DataFrame:
    """
    Compute a weighted similarity score between the target land and
    every candidate. Features with missing data on either side are
    skipped and their weight redistributed.
    """
    weights = config["stage1_similarity"]["weights"]
    is_active = config["stage1_similarity"].get("is_active", {})
    zoning_compat = config["_zoning_compat"]

    # Filter to only active features
    active_weights = {k: v for k, v in weights.items() if is_active.get(k, True)}

    if ref_date is None:
        ref_date = pd.Timestamp("2026-04-07")

    results = []

    for idx, c in candidates.iterrows():
        scores = {}

        # 1. District match
        if "district" in active_weights and target.get("district") and pd.notna(c["district"]):
            scores["district"] = 1.0 if c["district"] == target["district"] else 0.0

        # 2. Coordinate proximity
        if "coordinates" in active_weights:
            dist_km = _distance_km(
                float(target.get("latitude", float("nan"))),
                float(target.get("longitude", float("nan"))),
                c["latitude"], c["longitude"],
            )
            scores["coordinates"] = _coord_score(dist_km, config)

        # 3. Zoning match
        if "zoning" in active_weights:
            t_z = target.get("zoning")
            c_z = c["zoning"]
            if t_z and pd.notna(c_z):
                scores["zoning"] = zoning_compat.get((t_z, c_z), 0.0)

        # 4. Area closeness
        if "area" in active_weights:
            t_area = target.get("area_sqm")
            if t_area:
                scores["area"] = _pct_diff_score(float(t_area), c["area_sqm"], config)

        # 5. Road width closeness
        if "road_width" in active_weights:
            t_road = target.get("road_width_m")
            if t_road:
                scores["road_width"] = _pct_diff_score(float(t_road), c["road_width_m"], config)

        # 6. FAR/CSR similarity
        if "far_csr" in active_weights:
            far_score = _pct_diff_score(
                float(target["far"]) if target.get("far") else None,
                c["far"] if pd.notna(c.get("far")) else None, config,
            )
            csr_score = _pct_diff_score(
                float(target["csr"]) if target.get("csr") else None,
                c["csr"] if pd.notna(c.get("csr")) else None, config,
            )
            if far_score is not None and csr_score is not None:
                scores["far_csr"] = (far_score + csr_score) / 2
            elif far_score is not None:
                scores["far_csr"] = far_score
            elif csr_score is not None:
                scores["far_csr"] = csr_score

        # 7. Grades (commercial + admin averaged)
        if "grades" in active_weights:
            comm_score = _grade_score(target.get("commercial_grade"), c.get("commercial_grade"), config)
            admin_score = _grade_score(target.get("admin_grade"), c.get("admin_grade"), config)
            if comm_score is not None and admin_score is not None:
                scores["grades"] = (comm_score + admin_score) / 2
            elif comm_score is not None:
                scores["grades"] = comm_score
            elif admin_score is not None:
                scores["grades"] = admin_score

        # 8. Utilities
        if "utilities" in active_weights:
            t_util = target.get("utilities_count")
            if t_util is not None and pd.notna(c.get("utilities_count")):
                diff = abs(int(t_util) - int(c["utilities_count"]))
                scores["utilities"] = max(0, 1 - diff / 5)

        # 9. Recency
        if "recency" in active_weights:
            scores["recency"] = _recency_score(c["transaction_date"], ref_date)

        # ── Weighted sum with redistribution (only active features) ──
        available = {k: v for k, v in scores.items() if v is not None}
        if not available:
            total_score = 0.0
        else:
            raw_weight_sum = sum(active_weights[k] for k in available)
            total_score = sum(
                (active_weights[k] / raw_weight_sum) * v
                for k, v in available.items()
            )

        results.append({
            "candidate_idx": idx,
            "similarity_score": round(total_score, 4),
            **{f"sim_{k}": v for k, v in scores.items()},
        })

    score_df = pd.DataFrame(results).set_index("candidate_idx")
    candidates = candidates.join(score_df).sort_values("similarity_score", ascending=False)

    return candidates


# ══════════════════════════════════════════════
# STEP 4: Validation
# ══════════════════════════════════════════════

def validate_benchmarks(benchmarks: pd.DataFrame) -> list[str]:
    """Check for duplicates and excessive similarity between benchmarks."""
    warnings = []
    n = len(benchmarks)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = benchmarks.iloc[i], benchmarks.iloc[j]

            area_diff = abs(a["area_sqm"] - b["area_sqm"]) / max(a["area_sqm"], b["area_sqm"])
            price_diff = abs(a["price_per_sqm"] - b["price_per_sqm"]) / max(a["price_per_sqm"], b["price_per_sqm"])

            if (a["district"] == b["district"] and area_diff < 0.01 and price_diff < 0.01):
                warnings.append(
                    f"Possible duplicate: {a['land_id']} and {b['land_id']} "
                    f"(same district, near-identical area & price)"
                )
            if area_diff < 0.15:
                warnings.append(
                    f"Area diff < 15%: {a['land_id']} ({a['area_sqm']:,.0f}) "
                    f"vs {b['land_id']} ({b['area_sqm']:,.0f}) \u2192 {area_diff:.1%}"
                )
            if price_diff < 0.10:
                warnings.append(
                    f"Price diff < 10%: {a['land_id']} ({a['price_per_sqm']:,.0f}) "
                    f"vs {b['land_id']} ({b['price_per_sqm']:,.0f}) \u2192 {price_diff:.1%}"
                )

    return warnings


# ══════════════════════════════════════════════
# STEP 5: Output
# ══════════════════════════════════════════════

def format_and_export(target: dict, benchmarks: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """Format display table and export CSV."""
    display_cols = {
        "land_id": "Land ID",
        "district": "District",
        "area_sqm": "Area (sqm)",
        "price_per_sqm": "Price/sqm (EGP)",
        "zoning": "Zoning",
        "road_width_m": "Road Width (m)",
        "area_category": "Area Category",
        "similarity_score": "Similarity",
    }
    display = benchmarks[list(display_cols.keys())].rename(columns=display_cols).copy()
    display["Area (sqm)"] = display["Area (sqm)"].map("{:,.1f}".format)
    display["Price/sqm (EGP)"] = display["Price/sqm (EGP)"].map("{:,.2f}".format)
    display["Similarity"] = display["Similarity"].map("{:.4f}".format)

    # Export (include per-feature similarity scores)
    export_cols = [
        "land_id", "district", "area_sqm", "price_per_sqm", "zoning",
        "road_width_m", "area_category", "similarity_score",
        "sim_district", "sim_coordinates", "sim_zoning", "sim_area",
        "sim_road_width", "sim_far_csr", "sim_grades", "sim_utilities", "sim_recency",
        "transaction_date", "latitude", "longitude", "far", "csr",
        "commercial_grade", "admin_grade", "utilities_available",
    ]
    out = benchmarks[[c for c in export_cols if c in benchmarks.columns]].copy()
    out.insert(0, "target_land_city", target["city"])
    out.to_csv(output_path, index=False)

    return display


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

if __name__ == "__main__":
    DATA_PATH = "/home/claude/land_transactions_egypt.csv"
    CONFIG_PATH = "/home/claude/valuation_config.json"
    OUTPUT_PATH = "/home/claude/stage1_benchmarks_output.csv"

    # Load config
    config = load_config(CONFIG_PATH)

    # Example target land
    target_land = {
        "city":             "Cairo",
        "district":         "Nasr City",
        "latitude":         30.05,
        "longitude":        31.35,
        "area_sqm":         400,
        "zoning":           "Residential",
        "road_width_m":     12,
        "far":              1.5,
        "csr":              0.6,
        "frontage_m":       20,
        "depth_m":          20,
        "commercial_grade": "B",
        "admin_grade":      "C",
        "utilities_count":  4,
    }

    print("=" * 70)
    print("STAGE 1: BENCHMARK SELECTION FOR TARGET LAND")
    print("=" * 70)
    print(f"\n\U0001f3af Target Land:")
    for k, v in target_land.items():
        print(f"    {k}: {v}")

    # Steps 1-2
    df, report = load_and_prepare(DATA_PATH, target_land["city"], config)

    print(f"\n\U0001f4e5 Loaded {report['raw_count']} raw records")
    print(f"\n\u2500\u2500 STEP 1: Filtering \u2500\u2500")
    for reason, count in report["step1"]["rejections"].items():
        print(f"    {reason}: {count}")
    t = report["step1"]["outlier_thresholds"]
    print(f"    Outlier thresholds (within {target_land['city']}): "
          f"P_low={t['p_low']:,.0f}, P_high={t['p_high']:,.0f} EGP/sqm")
    print(f"    \u2705 Clean candidate pool: {report['step1']['clean_count']} records")

    print(f"\n\u2500\u2500 STEP 2: Derived Metrics \u2500\u2500")
    t = report["step2"]["area_thresholds"]
    print(f"    Area thresholds: P33={t['p33']:,.0f} sqm, P66={t['p66']:,.0f} sqm")
    print(f"    Distribution: {report['step2']['distribution']}")

    # Step 3: Similarity
    top_k = config["stage1_similarity"]["top_k"]
    df = compute_similarity(target_land, df, config)
    top5 = df.head(top_k).copy()

    print(f"\n\u2500\u2500 STEP 3: Similarity Scoring \u2500\u2500")
    print(f"    Scored {len(df)} candidates")
    print(f"    Score range: {df['similarity_score'].min():.4f} \u2013 {df['similarity_score'].max():.4f}")
    print(f"    Top {top_k} selected")

    sim_cols = [c for c in top5.columns if c.startswith("sim_")]
    print(f"\n    Per-feature scores (top {top_k}):")
    for _, row in top5.iterrows():
        parts = [f"{col.replace('sim_', '')}={row[col]:.2f}" if pd.notna(row[col]) else f"{col.replace('sim_', '')}=N/A" for col in sim_cols]
        print(f"    {row['land_id']}: {', '.join(parts)}")

    # Step 4: Validation
    warnings = validate_benchmarks(top5)
    print(f"\n\u2500\u2500 STEP 4: Validation \u2500\u2500")
    if warnings:
        for w in warnings:
            print(f"    \u26a0\ufe0f  {w}")
    else:
        print("    \u2705 All checks passed")

    # Step 5: Output
    display = format_and_export(target_land, top5, OUTPUT_PATH)
    print(f"\n\u2500\u2500 STEP 5: Final Benchmarks \u2500\u2500\n")
    print(display.to_string(index=False))
    print(f"\n\U0001f4e4 Exported to {OUTPUT_PATH}")
    print("=" * 70)