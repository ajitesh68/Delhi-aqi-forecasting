"""CPCB AQI conformance tests.

The previous implementation applied breakpoints to instantaneous hourly
readings, so its output could never match the official bulletin. These
tests pin the parts that were wrong: rolling windows, the three-sub-index
rule, and the requirement that one of them be particulate matter.
"""

import numpy as np
import pandas as pd
import pytest

from src.aqi import (AVERAGING_HOURS, add_rolling_aqi, calculate_aqi, get_category,
                     rolling_averages, sub_index, sub_index_series)


class TestSubIndex:
    @pytest.mark.parametrize("pollutant,concentration,expected", [
        ("pm2_5", 0, 0),
        ("pm2_5", 30, 50),      # band edge
        ("pm2_5", 60, 100),
        ("pm2_5", 90, 200),
        ("pm2_5", 120, 300),
        ("pm2_5", 250, 400),
        ("pm10", 100, 100),
        ("pm10", 250, 200),
        ("no2", 80, 100),
        ("so2", 80, 100),
        ("o3", 100, 100),
        ("co", 2, 100),         # CO breakpoints are mg/m3
        ("nh3", 400, 100),
    ])
    def test_breakpoint_edges(self, pollutant, concentration, expected):
        assert sub_index(pollutant, concentration) == pytest.approx(expected, abs=1)

    def test_midband_is_linear(self):
        # PM2.5 45 sits halfway between 30 and 60, so halfway between 51 and 100
        assert sub_index("pm2_5", 45) == pytest.approx(75.5, abs=1)

    def test_above_scale_caps_at_500(self):
        assert sub_index("pm2_5", 900) == 500

    @pytest.mark.parametrize("bad", [None, float("nan"), -5, "abc"])
    def test_unusable_values_return_none(self, bad):
        assert sub_index("pm2_5", bad) is None

    def test_vectorised_matches_scalar(self):
        values = [0, 10, 30, 45, 60, 79, 95, 150, 250, 300, 450]
        vector = sub_index_series("pm2_5", values)
        for value, got in zip(values, vector):
            assert abs(round(got) - sub_index("pm2_5", value)) <= 1


class TestCalculateAqi:
    def test_max_sub_index_wins_and_names_dominant(self):
        result = calculate_aqi({"pm2_5": 90, "pm10": 100, "no2": 40, "o3": 50})
        assert result["valid"]
        assert result["aqi"] == 200            # PM2.5 at 90
        assert result["dominant"] == "pm2_5"

    def test_requires_three_sub_indices(self):
        result = calculate_aqi({"pm2_5": 90, "pm10": 100})
        assert not result["valid"]
        assert "3" in result["reason"]

    def test_requires_a_pm_reading(self):
        result = calculate_aqi({"no2": 90, "so2": 100, "o3": 120, "co": 3})
        assert not result["valid"]
        assert "PM" in result["reason"]

    def test_thin_coverage_drops_the_pollutant(self):
        # PM2.5 averaged from 4 hours cannot stand in for a 24-hour mean
        result = calculate_aqi({"pm2_5": (300, 4), "pm10": (100, 20),
                                "no2": (40, 20), "o3": (50, 20)})
        assert "pm2_5" not in result["sub_indices"]
        assert result["aqi"] == 100            # falls back to PM10

    def test_full_coverage_keeps_the_pollutant(self):
        result = calculate_aqi({"pm2_5": (300, 20), "pm10": (100, 20),
                                "no2": (40, 20), "o3": (50, 20)})
        assert result["dominant"] == "pm2_5"
        assert result["aqi"] > 400

    def test_empty_input_is_invalid_not_zero(self):
        result = calculate_aqi({})
        assert not result["valid"]
        assert result["aqi"] is None


class TestRollingWindows:
    def test_averaging_periods_follow_cpcb(self):
        assert AVERAGING_HOURS["pm2_5"] == 24
        assert AVERAGING_HOURS["pm10"] == 24
        assert AVERAGING_HOURS["no2"] == 24
        assert AVERAGING_HOURS["co"] == 8      # CO and O3 are 8-hour
        assert AVERAGING_HOURS["o3"] == 8

    def test_rolling_average_uses_its_own_window(self):
        index = pd.date_range("2026-01-01", periods=48, freq="h")
        df = pd.DataFrame({"datetime": index,
                           "pm2_5": np.r_[np.full(24, 100.0), np.full(24, 200.0)],
                           "co": np.r_[np.full(40, 1.0), np.full(8, 5.0)]})
        out = rolling_averages(df)
        # PM2.5 over the trailing 24h is entirely the 200 block
        assert out["pm2_5"][0] == pytest.approx(200, abs=1)
        # CO over the trailing 8h is entirely the 5.0 block
        assert out["co"][0] == pytest.approx(5.0, abs=0.1)

    def test_spike_takes_a_full_day_to_register(self):
        """A one-hour spike must not move a 24-hour average to the spike value."""
        index = pd.date_range("2026-01-01", periods=48, freq="h")
        pm = np.full(48, 60.0)
        pm[-1] = 500.0
        df = pd.DataFrame({"datetime": index, "pm2_5": pm, "pm10": 150.0,
                           "no2": 30.0, "o3": 20.0, "station": "X"})
        out = rolling_averages(df)
        assert out["pm2_5"][0] < 100          # nowhere near 500

    def test_add_rolling_aqi_fills_gaps_in_the_index(self):
        index = pd.date_range("2026-01-01", periods=72, freq="h")
        df = pd.DataFrame({"datetime": index, "station": "X", "pm2_5": 80.0,
                           "pm10": 150.0, "no2": 30.0, "o3": 20.0})
        df = df.drop(index=range(10, 20))     # punch a 10-hour hole
        out = add_rolling_aqi(df)
        assert len(out) == 72                 # reindexed back to continuous
        assert out["aqi"].notna().any()

    def test_all_na_rows_do_not_raise(self):
        index = pd.date_range("2026-01-01", periods=30, freq="h")
        df = pd.DataFrame({"datetime": index, "station": "X",
                           "pm2_5": np.nan, "pm10": np.nan, "no2": np.nan})
        out = add_rolling_aqi(df)
        assert out["aqi"].isna().all()


class TestCategories:
    @pytest.mark.parametrize("aqi,label", [
        (0, "Good"), (50, "Good"), (51, "Satisfactory"), (100, "Satisfactory"),
        (101, "Moderate"), (200, "Moderate"), (201, "Poor"), (300, "Poor"),
        (301, "Very Poor"), (400, "Very Poor"), (401, "Severe"), (700, "Severe"),
    ])
    def test_boundaries(self, aqi, label):
        assert get_category(aqi) == label

    def test_missing_aqi_is_unknown_not_good(self):
        assert get_category(None) == "Unknown"
        assert get_category(float("nan")) == "Unknown"


class TestRealisticDelhiDay:
    def test_severe_winter_day(self):
        """A Delhi winter morning lands in Severe, driven by particulates."""
        result = calculate_aqi({"pm2_5": (310, 24), "pm10": (420, 24),
                                "no2": (95, 24), "so2": (22, 24),
                                "o3": (30, 8), "co": (3.1, 8)})
        assert result["dominant"] == "pm2_5"
        assert result["category"] == "Severe"

    def test_coarse_dust_episode_is_led_by_pm10(self):
        """A dust storm inverts the usual ordering: PM10 outruns PM2.5."""
        result = calculate_aqi({"pm2_5": (310, 24), "pm10": (480, 24),
                                "no2": (95, 24), "o3": (30, 8)})
        assert result["dominant"] == "pm10"
        assert result["category"] == "Severe"

    def test_clean_monsoon_day(self):
        result = calculate_aqi({"pm2_5": (25, 24), "pm10": (60, 24),
                                "no2": (20, 24), "o3": (40, 8)})
        assert result["category"] in ("Good", "Satisfactory")
        assert result["aqi"] <= 100
