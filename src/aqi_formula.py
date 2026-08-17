import numpy as np

BREAKPOINTS = {
    "pm2_5": [
        (0, 30, 0, 50),
        (31, 60, 51, 100),
        (61, 90, 101, 200),
        (91, 120, 201, 300),
        (121, 250, 301, 400),
        (250, 500, 401, 500),
    ],
    "pm10": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 250, 101, 200),
        (251, 350, 201, 300),
        (351, 430, 301, 400),
        (430, 600, 401, 500),
    ],
    "no2": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 180, 101, 200),
        (181, 280, 201, 300),
        (281, 400, 301, 400),
        (400, 800, 401, 500),
    ],
    "co": [
        (0, 1000, 0, 50),
        (1001, 2000, 51, 100),
        (2001, 10000, 101, 200),
        (10001, 17000, 201, 300),
        (17001, 34000, 301, 400),
        (34000, 50000, 401, 500),
    ],
}

AQI_CATEGORIES = [
    (0, 50, "Good"),
    (51, 100, "Satisfactory"),
    (101, 200, "Moderate"),
    (201, 300, "Poor"),
    (301, 400, "Very Poor"),
    (401, 500, "Severe"),
]


def calc_sub_index(pollutant, concentration):
    if pollutant not in BREAKPOINTS:
        return 0

    for bp_lo, bp_hi, idx_lo, idx_hi in BREAKPOINTS[pollutant]:
        if bp_lo <= concentration <= bp_hi:
            sub_index = ((idx_hi - idx_lo) / (bp_hi - bp_lo)) * (concentration - bp_lo) + idx_lo
            return round(sub_index)

    if concentration > BREAKPOINTS[pollutant][-1][1]:
        return 500
    return 0


def calculate_aqi(pm2_5, pm10, co, no2):
    sub_indices = {
        "pm2_5": calc_sub_index("pm2_5", pm2_5),
        "pm10": calc_sub_index("pm10", pm10),
        "co": calc_sub_index("co", co),
        "no2": calc_sub_index("no2", no2),
    }
    aqi = max(sub_indices.values())
    dominant = max(sub_indices, key=sub_indices.get)
    return aqi, dominant, sub_indices


def get_category(aqi):
    for lo, hi, label in AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return label
    return "Severe" if aqi > 500 else "Good"


def get_health_advisory(category):
    advisories = {
        "Good": "Minimal impact. Enjoy outdoor activities.",
        "Satisfactory": "Minor breathing discomfort for sensitive people.",
        "Moderate": "Breathing discomfort for people with lung/heart disease.",
        "Poor": "Breathing discomfort on prolonged exposure. Avoid outdoor exertion.",
        "Very Poor": "Respiratory illness on prolonged exposure. Limit outdoor activity.",
        "Severe": "Serious health impacts. Avoid all outdoor activity.",
    }
    return advisories.get(category, "")
