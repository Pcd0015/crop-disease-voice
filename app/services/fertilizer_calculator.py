"""
Basic fertilizer (NPK) calculator — modeled on Plantix's fertilizer
calculator feature. Fully free/offline: static per-crop reference rates,
no API calls.

Rates below are general per-acre nitrogen/phosphorus/potassium (N-P-K)
guidelines commonly used as a starting reference for these crops. They
are NOT a substitute for a soil test or local agricultural extension
guidance — actual needs vary by soil condition, variety, and region.
This is presented to the user as a starting estimate, not a prescription.
"""
from dataclasses import dataclass

# kg per acre, at "full" growth-stage multiplier of 1.0
BASE_RATES_KG_PER_ACRE = {
    "Tomato":       {"N": 55, "P": 40, "K": 55},
    "Potato":       {"N": 60, "P": 40, "K": 60},
    "Corn (maize)": {"N": 70, "P": 30, "K": 30},
    "Apple":        {"N": 40, "P": 20, "K": 40},
    "Grape":        {"N": 30, "P": 15, "K": 40},
    "Bell pepper": {"N": 50, "P": 35, "K": 50},
    "Cherry":       {"N": 35, "P": 18, "K": 35},
    "Peach":        {"N": 35, "P": 18, "K": 35},
    "Strawberry":   {"N": 45, "P": 35, "K": 55},
    "Soybean":      {"N": 15, "P": 25, "K": 30},  # legume — fixes some own nitrogen
    "Squash":       {"N": 45, "P": 30, "K": 45},
    "Blueberry":    {"N": 30, "P": 15, "K": 25},
    "Raspberry":    {"N": 35, "P": 15, "K": 30},
    "Orange (Citrus)": {"N": 55, "P": 20, "K": 45},
    # Added alongside the reference-only disease library entries — general
    # published per-acre starting guidelines for these crops.
    "Rice":                   {"N": 48, "P": 24, "K": 16},
    "Wheat":                  {"N": 48, "P": 24, "K": 16},
    "Cotton":                 {"N": 32, "P": 16, "K": 16},
    "Sugarcane":              {"N": 100, "P": 24, "K": 40},
    "Banana":                 {"N": 80, "P": 24, "K": 100},
    "Mango":                  {"N": 40, "P": 20, "K": 40},
    "Onion":                  {"N": 40, "P": 20, "K": 20},
    "Cabbage":                {"N": 50, "P": 25, "K": 25},
    "Cauliflower":            {"N": 50, "P": 25, "K": 25},
    "Groundnut (Peanut)":     {"N": 8, "P": 20, "K": 16},   # legume — fixes some own nitrogen
    "Chickpea (Gram)":        {"N": 8, "P": 16, "K": 8},    # legume — fixes some own nitrogen
    "Mustard":                {"N": 32, "P": 16, "K": 8},
    "Sunflower":              {"N": 24, "P": 16, "K": 16},
    "Chili (Pepper)":         {"N": 40, "P": 20, "K": 20},
    "Brinjal (Eggplant)":     {"N": 40, "P": 20, "K": 20},
    "Cucumber":               {"N": 30, "P": 16, "K": 30},
    "Watermelon":             {"N": 30, "P": 16, "K": 30},
    "Papaya":                 {"N": 45, "P": 20, "K": 45},
    "Guava":                  {"N": 35, "P": 15, "K": 30},
    "Pomegranate":            {"N": 25, "P": 10, "K": 25},
    "Turmeric":               {"N": 24, "P": 20, "K": 20},
    "Ginger":                 {"N": 30, "P": 24, "K": 20},
    "Garlic":                 {"N": 40, "P": 20, "K": 20},
    "Okra (Lady's finger)":   {"N": 40, "P": 20, "K": 20},
    "Carrot":                 {"N": 24, "P": 20, "K": 24},
    "Spinach":                {"N": 24, "P": 12, "K": 12},
}

# Growth-stage multipliers — early stages need less, peak vegetative/fruiting need more
STAGE_MULTIPLIERS = {
    "Seedling / early growth": 0.4,
    "Vegetative growth": 1.0,
    "Flowering / fruiting": 1.2,
    "Maturation": 0.6,
}


@dataclass
class FertilizerPlan:
    crop: str
    stage: str
    area_acres: float
    n_kg: float
    p_kg: float
    k_kg: float
    note: str


def available_crops() -> list[str]:
    return sorted(BASE_RATES_KG_PER_ACRE.keys())


def available_stages() -> list[str]:
    return list(STAGE_MULTIPLIERS.keys())


def calculate(crop: str, stage: str, area_acres: float) -> FertilizerPlan:
    base = BASE_RATES_KG_PER_ACRE[crop]
    multiplier = STAGE_MULTIPLIERS[stage]
    factor = multiplier * area_acres

    note = (
        "This is a general starting estimate, not a substitute for a soil "
        "test. Actual needs vary by soil condition and local recommendations."
    )
    if crop in ("Soybean", "Groundnut (Peanut)", "Chickpea (Gram)"):
        note += f" {crop} fixes some of its own nitrogen, so N needs are lower than most crops."

    return FertilizerPlan(
        crop=crop,
        stage=stage,
        area_acres=area_acres,
        n_kg=round(base["N"] * factor, 1),
        p_kg=round(base["P"] * factor, 1),
        k_kg=round(base["K"] * factor, 1),
        note=note,
    )
