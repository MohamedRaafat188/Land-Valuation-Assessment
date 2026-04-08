# Land Valuation Using the Market Method

A Python-based land valuation system that estimates the market value of a target land parcel by comparing it against similar parcels with known transaction prices. The system implements the **Sales Comparison Approach (Market Method)** in two stages: benchmark selection and market valuation scoring.

## Overview

The Market Method derives a land's value by:
1. **Finding comparable lands** — selecting the most similar transacted parcels from a dataset
2. **Scoring and pricing** — evaluating both target and comparables across weighted factors, then deriving the target's estimated price per square meter

All weights, scoring brackets, and thresholds are externalized in a JSON configuration file, making the system fully tunable without code changes.

## Project Structure

```
├── valuation_config.json            # All weights, brackets, thresholds (admin-configurable)
├── stage1_benchmark_selection.py    # Stage 1: Data filtering + similarity-based benchmark selection
├── stage2_market_valuation.py       # Stage 2: Factor scoring + pricing engine
├── generate_data.py                 # Synthetic dataset generator (500 records, Egypt market)
├── data/
│   └── land_transactions_egypt.csv  # Synthetic transaction dataset
├── output/
│   ├── stage1_benchmarks_output.csv # Benchmark selection results
│   ├── stage2_valuation_output.csv  # Full scoring audit trail
│   └── Land_Valuation_Technical_Report.docx  # Technical report
└── README.md
```

## How It Works

### Stage 1: Benchmark Selection

Filters the transaction dataset and selects the **top 5 most similar** comparable lands to a given target property.

| Step | Action | Output |
|------|--------|--------|
| 1a | Filter by target city | City-scoped dataset |
| 1b | Basic filters: asset type, nulls, valid price/area, zoning, road width | Filtered candidates |
| 1c | Remove price outliers (P5/P95) within the city | Clean candidate pool |
| 2 | Compute derived metrics: price/sqm, area categories | Enriched candidates |
| 3 | Weighted similarity scoring (9 non-price features) vs target | Ranked candidates |
| 4 | Validate top 5 for duplicates and excessive similarity | Validated benchmarks |
| 5 | Export with per-feature similarity scores | Stage 1 CSV |

**Similarity features:** district match, coordinate proximity (haversine), zoning compatibility, area closeness, road width, FAR/CSR, commercial/admin grades, utilities, and transaction recency.

### Stage 2: Market Valuation

Scores both the target and each benchmark across **10 weighted factors**, then derives the estimated price.

| Factor | Scoring Method |
|--------|---------------|
| Address | District match (same = 1.0, different = 0.6) |
| Area | % difference between target and comparable |
| General Location | Haversine distance (km) with district fallback |
| Commercial Location | Distance to commercial corridor or grade-based |
| Administrative Location | Distance to admin center or grade-based |
| Proximity to Services | Distance brackets (meters) |
| Proximity to Main Road | Distance brackets (meters) |
| Plot Proportions | Depth-to-frontage ratio |
| FAR/CSR | Average of zoning compatibility + FAR/CSR similarity |
| Utilities | Count-based (k/5) |

**Pricing formula:**
```
Point Value (per benchmark) = price_per_sqm / total_points
Weighted Avg Point Value    = Σ(similarity × point_value) / Σ(similarity)
Target Price per sqm        = Weighted Avg Point Value × Target Total Points
```

## Configuration

All tunable parameters live in `valuation_config.json`:

- **Stage 1 & Stage 2 weights** — per-feature/factor importance
- **`is_active` flags** — enable/disable individual features or factors without removing them
- **Scoring brackets** — thresholds and scores for percentage difference, distance (km/m), grade mappings
- **Outlier removal** — configurable percentile thresholds
- **Area categories** — configurable P33/P66 boundaries
- **Zoning compatibility matrix** — pairwise compatibility scores
- **`top_k`** — number of benchmarks to select

When a factor is disabled via `is_active`, its weight is automatically redistributed across the remaining active factors.

## Installation

```bash
pip install pandas numpy
```

## Usage

### Run the full pipeline (Stage 1 + Stage 2)

```bash
python stage2_market_valuation.py
```

This runs both stages and produces two output files in the `output/` folder:
- `stage1_benchmarks_output.csv` — selected benchmarks with similarity scores
- `stage2_valuation_output.csv` — full scoring audit trail and final valuation

### Run Stage 1 only

```bash
python stage1_benchmark_selection.py
```

### Customize the target land

Edit the `target_land` dictionary in the `__main__` block of either script:

```python
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
```

### Adjust weights or disable a factor

Edit `valuation_config.json`:

```json
"stage2_factors": {
    "weights": {
        "address": 0.10,
        "area": 0.15,
        ...
    },
    "is_active": {
        "address": true,
        "area": true,
        "plot_proportions": false,
        ...
    }
}
```

## Output

### Stage 1 Output
Each benchmark includes: land ID, district, area, price/sqm, zoning, road width, similarity score, and per-feature scores (sim_district, sim_coordinates, sim_zoning, sim_area, sim_road_width, sim_far_csr, sim_grades, sim_utilities, sim_recency).

### Stage 2 Output
Full audit trail with: record type (target/benchmark/result), all 10 factor multipliers and weighted scores, total points, point value, and the final estimated price per sqm.

### Example Result

```
Target: Cairo, Nasr City | 400 sqm | Residential
──────────────────────────────────────────────────
Target Total Points:            0.9400
Weighted Avg Point Value:       39,173.87 EGP/point
Estimated Price/sqm:            36,823.44 EGP
Total Estimated Land Value:     14,729,375 EGP
Benchmarks Used:                5 (similarity: 0.53 – 0.61)
```

## Design Decisions

- **City-first filtering** — candidates outside the target's city are removed first, before any other filter. This ensures all subsequent steps (outlier detection, percentile thresholds, rejection counts) reflect the local market only.
- **Weight redistribution** — when a feature has missing data or is disabled via `is_active`, its weight is redistributed across remaining features rather than penalizing with a zero score.
- **Weighted average pricing** — Stage 1 similarity scores are used as weights when averaging point values, giving more influence to closer comparables.
- **Config-driven** — all parameters are externalized in JSON, supporting admin tunability without code changes.

## Synthetic Data

The included dataset (`data/land_transactions_egypt.csv`) contains 500 synthetic land transaction records across 6 Egyptian cities (Cairo, Giza, Alexandria, New Administrative Capital, Mansoura, Tanta) with realistic distributions for prices, areas, zoning types, and intentional data quality issues (missing values, non-land assets, price outliers) for testing the filtering pipeline.