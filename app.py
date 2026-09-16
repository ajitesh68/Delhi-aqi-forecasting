"""Delhi AQI — live CPCB ground-station dashboard."""

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Delhi Air | CPCB Live", page_icon="🌫️",
                   layout="wide", initial_sidebar_state="expanded")

from src import analysis, data_loader, maps, metrics, styles, visualizations as viz
from src.aqi import get_category, health_advisory
from src.config import AQI_COLORS, NCR_CITIES, POLLUTANT_LABELS
from src.metrics import SENSITIVITY

styles.inject(st)


def card(body):
    st.markdown(f'<div class="glass">{body}</div>', unsafe_allow_html=True)


def pill(text, kind="muted"):
    return f'<span class="pill pill-{kind}">{text}</span>'


def short(name):
    return str(name).split(",")[0].strip()


# ----------------------------------------------------------------- load

include_ncr = st.sidebar.toggle(
    "Include wider NCR", value=False,
    help="Adds Noida, Ghaziabad, Gurugram and Faridabad. The open data.gov.in "
         "key returns 10 records per request, so each extra city adds a few "
         "seconds on a cold cache.")
cities = tuple(NCR_CITIES) if include_ncr else ("Delhi",)

with st.spinner(f"Reading live CPCB stations ({', '.join(cities[:2])}"
                f"{'...' if len(cities) > 2 else ''})..."):
    stations, meta = data_loader.live_stations(cities)

status = data_loader.store_status()
summary = data_loader.city_summary(stations)

# ----------------------------------------------------------------- sidebar

with st.sidebar:
    st.markdown("### Delhi Air")
    st.caption("Ground-station readings from the CPCB / DPCC monitoring "
               "network, not a satellite model.")

    station_names = sorted(stations["station"].dropna().unique()) if len(stations) else []
    default = next((i for i, s in enumerate(station_names) if "Anand Vihar" in s), 0)
    selected = st.selectbox("Station", station_names, index=default,
                            format_func=short) if station_names else None

    group = st.selectbox("Who is this for?", list(SENSITIVITY.keys()))

    st.markdown("---")
    st.markdown("**Data sources**")
    if meta.get("stale"):
        st.warning("Live feed unreachable — showing the last cached sweep.", icon="⚠️")
    elif meta.get("from_cache") and meta.get("cache_age_min") is not None:
        st.caption(f"Served from a sweep {meta['cache_age_min']:.0f} minutes old. "
                   f"The CPCB feed publishes hourly, so this is current.")
    st.caption(
        f"Live: data.gov.in CPCB feed — {meta.get('total_stations', 0)} stations, "
        f"{meta.get('with_aqi', 0)} with a full AQI.\n\n"
        f"History: OpenAQ CPCB archive — "
        + (f"{status['rows']:,} hours across {status['stations']} stations."
           if not status["empty"] else "still downloading.")
    )
    if meta.get("co_unit") == "unknown":
        st.caption("CO is excluded from AQI here: the feed reports it in units "
                   "that cannot be resolved, and guessing risks a 1000x error.")

# ----------------------------------------------------------------- header

left, right = st.columns([3, 2])
with left:
    st.markdown("# Delhi is breathing")
    if summary:
        reporting = f"{summary['n_reporting']} of {summary['n_total']} stations reporting"
        updated = pd.Timestamp(summary["updated"]).strftime("%d %b, %H:%M")
        st.markdown(
            f"{pill('CPCB ground stations', 'live')} "
            f"{pill(reporting)} {pill('updated ' + updated)}",
            unsafe_allow_html=True)
with right:
    if summary and summary["spread"] > 0:
        st.markdown(
            f'<div class="glass" style="text-align:center">'
            f'<div class="eyebrow">Spread across the city right now</div>'
            f'<div class="metric-value" style="color:{styles.ACCENT}">'
            f'{summary["spread"]} AQI</div>'
            f'<div class="metric-sub">{short(summary["best_station"])} '
            f'{summary["best_aqi"]} &nbsp;&rarr;&nbsp; '
            f'{short(summary["worst_station"])} {summary["worst_aqi"]}. '
            f'A single city-wide number hides this.</div></div>',
            unsafe_allow_html=True)

st.markdown("")

if len(stations) == 0:
    st.error("No live station data available right now, and no cached sweep to "
             "fall back on. The CPCB feed may be rate-limited — try again shortly.")
    st.stop()

# ----------------------------------------------------------------- hero row

row = stations[stations["station"] == selected].iloc[0] if selected else None

c1, c2, c3 = st.columns([1.1, 1, 1])

with c1:
    if row is not None and pd.notna(row["aqi"]):
        colour = AQI_COLORS.get(row["category"], styles.MUTED)
        card(f'<div class="eyebrow">{short(selected)}</div>'
             f'<div class="hero-aqi" style="color:{colour}">{int(row["aqi"])}</div>'
             f'<div class="hero-cat" style="color:{colour}">{row["category"]}</div>'
             f'<div class="metric-sub">Driven by '
             f'<b>{row["dominant_label"]}</b>. {health_advisory(row["category"])}</div>')
    elif row is not None:
        card(f'<div class="eyebrow">{short(selected)}</div>'
             f'<div class="hero-aqi" style="color:{styles.MUTED}">--</div>'
             f'<div class="metric-sub">{row["aqi_reason"] or "No reading"}. '
             f'CPCB needs three pollutants including a PM measurement.</div>')

with c2:
    pm25 = row["pm2_5"] if row is not None and pd.notna(row.get("pm2_5")) else None
    cigs = metrics.cigarette_equivalent(pm25) if pm25 else None
    who = metrics.who_multiple(pm25) if pm25 else None
    if cigs is not None:
        card(f'<div class="eyebrow">A day in this air</div>'
             f'<div class="metric-value">{cigs:g} <span style="font-size:1.1rem;'
             f'font-weight:600;opacity:.6">cigarettes</span></div>'
             f'<div class="metric-sub">PM2.5 is {pm25:.0f} µg/m³, about '
             f'<b>{who}x</b> the WHO daily guideline. Equivalence follows '
             f'Berkeley Earth: 22 µg/m³ over 24 h is roughly one cigarette.</div>')
    else:
        card('<div class="eyebrow">A day in this air</div>'
             '<div class="metric-value" style="opacity:.4">--</div>'
             '<div class="metric-sub">This station is not reporting PM2.5 right now.</div>')

with c3:
    score = metrics.activity_safety_score(row["aqi"], group) if row is not None and pd.notna(row["aqi"]) else None
    headline, detail = metrics.activity_verdict(score)
    if score is not None:
        tone = "#22C55E" if score >= 6.5 else "#FBBF24" if score >= 4.5 else "#EF4444"
        card(f'<div class="eyebrow">Outdoors for a {group.lower()}</div>'
             f'<div class="metric-value" style="color:{tone}">{score}'
             f'<span style="font-size:1.1rem;font-weight:600;opacity:.5">/10</span></div>'
             f'<div class="metric-sub"><b>{headline}.</b> {detail}</div>')
    else:
        card(f'<div class="eyebrow">Outdoors for a {group.lower()}</div>'
             f'<div class="metric-value" style="opacity:.4">--</div>'
             f'<div class="metric-sub">{detail}</div>')

st.markdown("")

# ----------------------------------------------------------------- tabs

(tab_live, tab_when, tab_forecast, tab_history,
 tab_health, tab_about) = st.tabs(
    ["Live map", "When to go out", "Tomorrow", "History",
     "Your exposure", "How this works"])

# --- live map -------------------------------------------------------

with tab_live:
    view = st.radio("View", ["Stations", "Heatmap", "3D columns"],
                    horizontal=True, label_visibility="collapsed")
    m1, m2 = st.columns([2, 1])
    with m1:
        deck = {"Stations": maps.station_map, "Heatmap": maps.heatmap,
                "3D columns": maps.column_map}[view](stations)
        if deck is not None:
            st.pydeck_chart(deck, use_container_width=True)
        else:
            st.info("No stations with coordinates to map yet.")
        if view == "Heatmap":
            st.markdown('<div class="note">The coloured surface is an '
                        'interpolation between monitors, not a measurement. '
                        'The white dots are the actual stations.</div>',
                        unsafe_allow_html=True)

    with m2:
        st.markdown("**Cleanest air right now**")
        ranked = stations.dropna(subset=["aqi"]).sort_values("aqi")
        for i, (_, s) in enumerate(ranked.head(5).iterrows(), 1):
            c = AQI_COLORS.get(s["category"], styles.MUTED)
            st.markdown(
                f'<div class="rank-row"><span class="rank-num">{i}</span>'
                f'<span class="rank-name">{short(s["station"])}</span>'
                f'<span class="rank-aqi" style="color:{c}">{int(s["aqi"])}</span></div>',
                unsafe_allow_html=True)

        st.markdown("**Worst right now**")
        for i, (_, s) in enumerate(ranked.tail(5).iloc[::-1].iterrows(), 1):
            c = AQI_COLORS.get(s["category"], styles.MUTED)
            st.markdown(
                f'<div class="rank-row"><span class="rank-num">{i}</span>'
                f'<span class="rank-name">{short(s["station"])}</span>'
                f'<span class="rank-aqi" style="color:{c}">{int(s["aqi"])}</span></div>',
                unsafe_allow_html=True)

    st.plotly_chart(
        viz.station_ranking_chart(
            stations.dropna(subset=["aqi"])[["station", "aqi"]].rename(
                columns={"aqi": "mean"}),
            highlight=selected,
            title="Every reporting station, ranked"),
        use_container_width=True)

# --- when to go out -------------------------------------------------

with tab_when:
    st.markdown("### When should I go outside?")
    if status["empty"]:
        st.info("Building the answer from real CPCB history — the archive "
                "download is still running. This will fill in automatically.")
    else:
        hist = data_loader.history(stations=[selected] if selected else None, days=120)
        if len(hist) == 0 and selected:
            st.info(f"No stored history for {short(selected)} yet. The archive "
                    "download covers a subset of stations first.")
            hist = data_loader.history(days=120)

        profile = analysis.diurnal_profile(hist, by=None)
        if len(profile) == 0:
            st.info("Not enough stored hours yet to build an hourly profile.")
        else:
            best_hours = profile.nsmallest(3, "mean")["hour"].tolist()
            lo, hi = min(best_hours), max(best_hours)
            worst = profile.loc[profile["mean"].idxmax()]
            best = profile.loc[profile["mean"].idxmin()]

            w1, w2 = st.columns([1, 2])
            with w1:
                card(f'<div class="eyebrow">Typically cleanest</div>'
                     f'<div class="metric-value" style="color:#4ADE80">'
                     f'{int(best["hour"]):02d}:00</div>'
                     f'<div class="metric-sub">Average AQI <b>{best["mean"]:.0f}</b> '
                     f'at this hour, against <b>{worst["mean"]:.0f}</b> at '
                     f'{int(worst["hour"]):02d}:00 — a difference of '
                     f'<b>{worst["mean"] - best["mean"]:.0f} AQI</b> for the same day.'
                     f'</div>')
                st.markdown("")
                card('<div class="eyebrow">Read this correctly</div>'
                     '<div class="metric-sub">This is the average shape of a day '
                     'from real station readings, not a forecast for tomorrow. '
                     'It is built from <b>raw hourly</b> readings, not the '
                     'official 24-hour CPCB index &mdash; a 24-hour rolling mean '
                     'flattens the daily cycle to almost nothing, so it cannot '
                     'answer this question.</div>')
            with w2:
                st.plotly_chart(
                    viz.diurnal_chart(profile, best_window=(lo, hi),
                                      title=f"Hour-by-hour, {short(selected) if selected else 'Delhi'}"),
                    use_container_width=True)

            st.markdown("**Safety score through the day**")
            profile = profile.copy()
            profile["score"] = [metrics.activity_safety_score(v, group)
                                for v in profile["mean"]]
            cols = st.columns(12)
            for i, (_, p) in enumerate(profile.iterrows()):
                if i % 2:
                    continue
                s = p["score"]
                tone = "#22C55E" if s >= 6.5 else "#FBBF24" if s >= 4.5 else "#EF4444"
                with cols[(i // 2) % 12]:
                    st.markdown(
                        f'<div style="text-align:center;padding:6px 0">'
                        f'<div style="font-size:.66rem;color:{styles.MUTED}">'
                        f'{int(p["hour"]):02d}h</div>'
                        f'<div style="font-size:1.05rem;font-weight:800;color:{tone}">'
                        f'{s:g}</div></div>', unsafe_allow_html=True)

# --- history --------------------------------------------------------

with tab_history:
    st.markdown("### What the record shows")
    st.markdown('<div class="note">Everything on this tab is computed from '
                'observed CPCB station readings. The forecasting model plays '
                'no part in any of these charts.</div>', unsafe_allow_html=True)
    st.markdown("")

    if status["empty"]:
        st.info("The CPCB archive download is still running. Historical charts "
                "appear as soon as the first station lands.")
    else:
        period = st.radio("Period", ["24 hours", "7 days", "30 days", "Patterns"],
                          horizontal=True, label_visibility="collapsed")

        # Trailing windows, not the whole store. The archive stops at its
        # last backfill and never moves again, so a view of everything is
        # dominated by a fixed past and looks frozen however long the
        # collector runs. Anchored to now, each view follows the clock.
        if period == "Patterns":
            hist = data_loader.history(days=None)
        else:
            days = {"24 hours": 1, "7 days": 7, "30 days": 30}[period]
            # One station, the one chosen in the sidebar. Forty-four lines
            # over a single day is not a comparison, it is a thicket; the
            # live map is where the spread between stations is read.
            hist = data_loader.history(stations=[selected] if selected else None,
                                       days=days + 1)

        if period == "24 hours":
            cutoff = pd.Timestamp.now().floor("h") - pd.Timedelta(hours=24)
            recent = hist[hist["datetime"] >= cutoff]
            # Sub-indices of the hour itself, not the CPCB headline number.
            # That number is a 24-hour rolling mean, which would flatten the
            # intraday movement this view exists to show -- and cannot be
            # formed at all until a full day of hours has accumulated.
            st.plotly_chart(
                viz.timeseries_chart(recent, value="aqi_instant", by="station",
                                     title="Hourly AQI, last 24 hours",
                                     height=400),
                use_container_width=True)
            n = int(recent["aqi_instant"].notna().sum()) if len(recent) else 0
            st.caption(f"{n:,} observed station-hours since "
                       f"{cutoff:%d %b %H:%M}. Plotted per hour rather than as "
                       "the CPCB index, which averages the 24 hours behind it "
                       "and so would hide the shape of the day.")

        elif period in ("7 days", "30 days"):
            span = 7 if period == "7 days" else 30
            daily = analysis.daily_summary(hist, by="station")
            daily = daily[~daily["sparse"]]
            # A short outage is bridged and drawn dotted; anything wider than
            # a few days stays absent rather than being invented.
            daily = analysis.fill_daily_gaps(daily, by="station", max_run=7)
            floor = pd.Timestamp.now().normalize() - pd.Timedelta(days=span)
            daily = daily[daily["date"] >= floor]

            st.plotly_chart(
                viz.timeseries_chart(
                    daily.rename(columns={"date": "datetime", "mean": "aqi"}),
                    by="station", title=f"Daily mean AQI, last {span} days",
                    height=400),
                use_container_width=True)

            n_est = int(daily["estimated"].sum()) if "estimated" in daily else 0
            note = ("Days with fewer than 12 observed hours are excluded rather "
                    "than averaged from thin data.")
            if n_est:
                note += (f" {n_est} station-days are dotted: the collector had "
                         "no reading for them and the live feed serves only the "
                         "current hour, so they are interpolated, not measured.")
            st.caption(note)

        else:
            st.caption(f"{status['rows']:,} observed hours · "
                       f"{status['stations']} stations · "
                       f"{status['first']:%d %b %Y} to {status['last']:%d %b %Y}")
            monthly = analysis.monthly_summary(hist, by=None)
            st.plotly_chart(viz.monthly_chart(monthly), use_container_width=True)
            yoy = analysis.year_over_year(hist)
            st.plotly_chart(viz.year_over_year_chart(yoy), use_container_width=True)
            if len(yoy) and yoy.attrs.get("doy_range"):
                lo, hi = yoy.attrs["doy_range"]
                st.caption(f"Restricted to days {lo}-{hi} of the year, the window "
                           "both years actually cover, so a partial year is not "
                           "compared against a full one.")

            p1, p2 = st.columns(2)
            with p1:
                st.plotly_chart(viz.day_of_week_chart(analysis.day_of_week_effect(hist)),
                                use_container_width=True)
            with p2:
                mix = analysis.pollutant_mix(hist, window_hours=24 * 30)
                st.plotly_chart(
                    viz.pollutant_radar(mix, title="What drives AQI here (30 days)"),
                    use_container_width=True)

            seasonal = analysis.seasonal_windows(hist)
            if len(seasonal):
                st.markdown("**Diwali windows, measured**")
                st.dataframe(seasonal, use_container_width=True, hide_index=True)

# --- exposure -------------------------------------------------------

with tab_health:
    st.markdown("### Your day, not the city's")
    st.markdown('<div class="note">A city-wide AQI is not your exposure. '
                'Where you spend your hours matters more than the headline '
                'number.</div>', unsafe_allow_html=True)
    st.markdown("")

    e1, e2 = st.columns([1, 1.3])
    with e1:
        names = sorted(stations["station"].dropna().unique())
        home = st.selectbox("Home area", names, index=default, format_func=short)
        work = st.selectbox("Work area", names,
                            index=min(1, len(names) - 1), format_func=short)
        hours_out = st.slider("Hours outdoors (commute, walking)", 0.0, 8.0, 2.0, 0.5)
        hours_work = st.slider("Hours at work", 0.0, 14.0, 8.0, 0.5)
        indoor = st.slider("Indoor air as a fraction of outdoor", 0.2, 1.0, 0.55, 0.05,
                           help="An unfiltered Indian home or office typically "
                                "sits near 0.5-0.6. A running purifier pushes it lower.")

    def pm_of(name):
        sel = stations[stations["station"] == name]
        if len(sel) == 0 or pd.isna(sel.iloc[0].get("pm2_5")):
            return None
        return float(sel.iloc[0]["pm2_5"])

    home_pm, work_pm = pm_of(home), pm_of(work)
    commute_pm = max(v for v in [home_pm, work_pm, 0] if v is not None)
    hours_home = max(0.0, 24 - hours_work - hours_out)

    with e2:
        if home_pm is None and work_pm is None:
            st.info("Neither selected station is reporting PM2.5 right now.")
        else:
            result = metrics.commute_exposure(
                home_pm or 0, work_pm or 0, commute_pm,
                hours_home=hours_home, hours_work=hours_work,
                hours_commute=hours_out, indoor_factor=indoor)
            k1, k2, k3 = st.columns(3)
            k1.metric("Effective PM2.5", f"{result['mean_pm25']:.0f}")
            k2.metric("Cigarettes today", f"{result['cigarettes']:g}")
            k3.metric("WHO guideline", f"{result['who_multiple']}x")
            st.plotly_chart(viz.exposure_chart(result["breakdown"]),
                            use_container_width=True)
            share = result["breakdown"]["Commute (outdoor)"] / max(result["ug_hours"], 1)
            st.markdown(
                f'<div class="note">Those {hours_out:g} outdoor hours are '
                f'{hours_out / 24 * 100:.0f}% of your day but '
                f'<b>{share * 100:.0f}%</b> of your PM2.5 intake. Shifting them '
                f'to a cleaner hour is the cheapest thing you can change.</div>',
                unsafe_allow_html=True)

# --- about ----------------------------------------------------------

# --- tomorrow ------------------------------------------------------

with tab_forecast:
    st.markdown("## The next 24 hours")

    forecast = data_loader.forecast_24h(selected) if selected else None

    if forecast is not None:
        # The forecast starts at the anchor, not at the newest row in the
        # store. Live snapshots can land hours after the archive ends, and
        # reporting those would claim a currency the forecast does not have.
        start = pd.Timestamp(forecast["anchor"])
        lag_hours = (pd.Timestamp.now() - start).total_seconds() / 3600
        if lag_hours > 6:
            st.warning(
                f"This forecasts the 24 hours after **{start:%d %b, %H:%M}**, "
                f"which is {lag_hours / 24:.0f} days ago — not the 24 hours after "
                f"now. The model needs an unbroken week of readings behind its "
                f"starting point, and the most recent stretch that qualifies "
                f"ends there: the OpenAQ archive runs several days behind, and "
                f"the live snapshots collected since are still too sparse to "
                f"bridge the gap. Running `scripts/collect_snapshot.py` hourly "
                f"closes it within a week, after which this becomes a genuine "
                f"forecast.",
                icon="🕐")

    if forecast is None:
        st.info(
            "No forecast for this station yet. The model is trained on the "
            "stations whose full history has been downloaded, and it needs "
            "seven unbroken days of recent readings to forecast from. "
            "Stations outside that set show measurements only.")
    else:
        frame = forecast["frame"]
        card_col, chart_col = st.columns([1, 2.2])

        with chart_col:
            st.plotly_chart(
                viz.forecast_chart(
                    frame["datetime"], frame["aqi"],
                    observed=data_loader.history([selected], days=5),
                    title=(f"{short(selected)}: "
                           f"{frame['datetime'].iloc[0]:%d %b %H:%M} to "
                           f"{frame['datetime'].iloc[-1]:%d %b %H:%M}")),
                use_container_width=True)

        with card_col:
            peak = frame.loc[frame["aqi"].idxmax()] if frame["aqi"].notna().any() else None
            if peak is not None:
                colour = AQI_COLORS.get(peak["category"], styles.MUTED)
                card(f'<div class="eyebrow">Worst hour ahead</div>'
                     f'<div class="metric-value" style="color:{colour}">'
                     f'{int(peak["aqi"])}</div>'
                     f'<div class="metric-sub"><b>{peak["category"]}</b> around '
                     f'{pd.Timestamp(peak["datetime"]):%H:%M}. '
                     f'Forecast PM2.5 peaks near '
                     f'{frame["pm2_5"].max():.0f} µg/m³.</div>')

            observed_pct = forecast["frac_observed"]
            beyond = forecast["frac_beyond_training"]
            tone = "live" if observed_pct > 0.9 else "stale"
            st.markdown(
                f'<div class="glass">'
                f'<div class="eyebrow">How much to trust this</div>'
                f'<div class="metric-sub">'
                f'{pill(f"{observed_pct:.0%} observed input", tone)} '
                f'{pill(f"{beyond:.0%} beyond training range")}<br/><br/>'
                f'The input window was {observed_pct:.0%} real readings. '
                + ("A meaningful share of recent hours sit outside anything "
                   "the model was trained on, so treat this as indicative."
                   if beyond > 0.15 else
                   "Recent conditions are within the range the model has seen.")
                + '</div></div>', unsafe_allow_html=True)

        st.markdown("")
        scorecard = forecast.get("scorecard")
        if scorecard:
            model = scorecard["scores"]["model"]
            base = scorecard["scores"]["persistence"]
            beats = scorecard["beats_persistence"]
            s1, s2, s3 = st.columns(3)
            with s1:
                card(f'<div class="eyebrow">Error on unseen winter</div>'
                     f'<div class="metric-value">{model["mae"]:.0f}'
                     f'<span style="font-size:1rem;opacity:.5"> µg/m³</span></div>'
                     f'<div class="metric-sub">Mean absolute error forecasting '
                     f'PM2.5 24 hours ahead through January 2026, a month held '
                     f'out of training entirely.</div>')
            with s2:
                verdict = "better than" if beats else "worse than"
                tone = "#22C55E" if beats else "#EF4444"
                card(f'<div class="eyebrow">Against doing nothing</div>'
                     f'<div class="metric-value" style="color:{tone}">'
                     f'{abs(scorecard["improvement_pct"]):.1f}%</div>'
                     f'<div class="metric-sub">{verdict} assuming tomorrow '
                     f'repeats today ({base["mae"]:.0f} µg/m³). That baseline '
                     f'is strong at this horizon, which is why it is the '
                     f'bar rather than a formality.</div>')
            with s3:
                card(f'<div class="eyebrow">On the worst hours</div>'
                     f'<div class="metric-value">{model["top_decile_mae"]:.0f}'
                     f'<span style="font-size:1rem;opacity:.5"> µg/m³</span></div>'
                     f'<div class="metric-sub">Error across the dirtiest 10% of '
                     f'hours, against {base["top_decile_mae"]:.0f} for the '
                     f'baseline. Severe episodes are the hardest to call and '
                     f'the ones that matter.</div>')

            st.caption(
                f"Trained on {scorecard['train_windows']:,} windows from "
                f"{len(scorecard['stations'])} stations and scored on "
                f"{scorecard['holdout_windows']:,} windows in January 2026. "
                "The model predicts a departure from persistence rather than a "
                "level, and reads the next 24 hours of forecast weather — wind, "
                "temperature and boundary-layer height — which is the only "
                "thing it knows that persistence does not.")

        st.caption(
            "One honest limitation: the model is trained on a single Delhi "
            "winter, so it under-predicts the sharpest peaks. Compare the "
            "forecast range against the observed line before relying on it "
            "during a severe episode.")

with tab_about:
    st.markdown("### How this works, and what it cannot do")
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("""
**Where the numbers come from**

Live readings are the CPCB / DPCC ground-station feed published through
data.gov.in — the same monitors that produce the official bulletin.
Historical hours come from the OpenAQ archive of the same network.

Open-Meteo is still used, but only for weather. Its air-quality product
is a coarse chemistry-transport model; measured against these stations at
Anand Vihar it correlates **0.40** with hourly PM2.5 and is off by an
average of 32 µg/m³. That is why pollutants are not taken from it.

**AQI is computed the way CPCB computes it**

Sub-indices run on rolling averages — 24 hours for PM2.5, PM10, NO₂, SO₂
and NH₃, 8 hours for CO and O₃ — not on instantaneous readings. A valid
AQI needs at least three pollutants, one of which must be PM. When a
station cannot meet that bar the AQI is shown as unavailable instead of
being computed from whatever happens to be reporting.
        """)
    with a2:
        st.markdown("""
**What is deliberately missing**

CO is dropped from the AQI when its units cannot be resolved. CPCB
publishes CO in mg/m³, OpenAQ relabels the same numbers as ppb, and the
two feeds disagree by a factor that is neither 1 nor 1000. Guessing would
risk a 1000x error, so an unresolvable reading is discarded.

Stubble-fire counts are not a model input. The previous version generated
them from a hardcoded seasonal curve whenever the satellite feed was
unavailable, which is a duplicate of the month-of-year signal wearing a
physics costume.

**Honest limits**

Coverage is uneven: stations drop pollutants for hours at a time, and
periods built from thin data are marked rather than quietly averaged.
The heatmap interpolates between monitors and is not a measurement.
        """)

    if not status["empty"]:
        st.markdown("---")
        st.markdown("**Store contents**")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Observed hours", f"{status['rows']:,}")
        s2.metric("Stations", status["stations"])
        s3.metric("Earliest", f"{status['first']:%b %Y}")
        s4.metric("Archive lag", f"{status['lag_hours']:.0f} h")
        st.caption("The archive runs a few days behind live because that is how "
                   "quickly the upstream network republishes. Live values on the "
                   "map are current; historical charts end where the archive does.")
