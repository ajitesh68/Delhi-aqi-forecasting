import pandas as pd
import streamlit as st

st.set_page_config(page_title="Delhi Air | CPCB Live", page_icon="🌫️",
                   layout="wide", initial_sidebar_state="expanded")

from src import (analysis, auth, data_loader, maps, metrics, styles,
                 visualizations as viz)
from src.aqi import health_advisory
from src.config import AQI_COLORS, RECENT_WINDOW_DAYS
from src.metrics import SENSITIVITY

styles.inject(st)

user = auth.require_login()

def card(body):
    st.markdown(f'<div class="glass">{body}</div>', unsafe_allow_html=True)

def pill(text, kind="muted"):
    return f'<span class="pill pill-{kind}">{text}</span>'

def short(name):
    return str(name).split(",")[0].strip()

with st.spinner("Reading live CPCB stations..."):
    stations, meta = data_loader.live_stations()

status = data_loader.store_status()
summary = data_loader.city_summary(stations)

with st.sidebar:
    st.markdown("### Delhi Air")
    st.caption("Ground-station readings from the CPCB / DPCC monitoring "
               "network, not a satellite model.")

    station_names = sorted(stations["station"].dropna().unique()) if len(stations) else []
    reporting = (set(stations.dropna(subset=["aqi"])["station"])
                if "aqi" in stations.columns else set())
    default = next((i for i, s in enumerate(station_names)
                    if "Anand Vihar" in s and s in reporting),
                   next((i for i, s in enumerate(station_names) if s in reporting), 0))
    selected = st.selectbox("Station", station_names, index=default,
                            format_func=short) if station_names else None

    group = st.selectbox("Who is this for?", list(SENSITIVITY.keys()))

    auth.sidebar_account(user)

    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("**Data sources**")
    if meta.get("stale"):
        st.warning("Live feed unreachable — showing cached readings.", icon="⚠️")
    elif meta.get("from_cache") and meta.get("cache_age_min") is not None:
        st.caption(f"Cached {meta['cache_age_min']:.0f} mins ago from WAQI live feed.")
    st.caption(
        f"Live: WAQI station network — {meta.get('total_stations', 0)} stations reporting.\n\n"
        f"Stored history: "
        + (f"{status['collected_rows']:,} hourly records."
           if status.get("collected_rows") else "none yet.")
    )

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
    if summary:
        st.markdown(
            f'<div class="glass" style="text-align:center">'
            f'<div class="eyebrow">City AQI Range (Cleanest to Most Polluted)</div>'
            f'<div class="metric-value" style="color:{styles.ACCENT}">'
            f'{summary["best_aqi"]} &ndash; {summary["worst_aqi"]}</div>'
            f'<div class="metric-sub">Cleanest: <b>{short(summary["best_station"])} ({summary["best_aqi"]})</b> &bull; '
            f'Highest: <b>{short(summary["worst_station"])} ({summary["worst_aqi"]})</b></div></div>',
            unsafe_allow_html=True)

st.markdown("")

if len(stations) == 0:
    st.error("No live station data available right now, and no cached sweep to "
             "fall back on. The CPCB feed may be rate-limited — try again shortly.")
    st.stop()

row = stations[stations["station"] == selected].iloc[0] if selected else None

c1, c2, c3 = st.columns(3)

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
    pm25_val = row["pm2_5"] if row is not None and pd.notna(row.get("pm2_5")) else (row["aqi"] if row is not None and pd.notna(row.get("aqi")) else None)
    cigs = metrics.cigarette_equivalent(pm25_val) if pm25_val is not None else None
    if cigs is not None:
        c_tone = "#22C55E" if cigs < 2.0 else "#FBBF24" if cigs < 5.0 else "#F97316" if cigs < 10.0 else "#EF4444"
        card(f'<div class="eyebrow">Cigarettes Consumed</div>'
             f'<div class="metric-value" style="color:{c_tone}">🚬 {cigs}'
             f'<span style="font-size:1.1rem;font-weight:600;opacity:.6"> cig / day</span></div>'
             f'<div class="metric-sub">Breathing this air for 24 hours is equivalent to smoking '
             f'<b>~{cigs:g} cigarettes</b> today.</div>')
    else:
        card('<div class="eyebrow">Cigarettes Consumed</div>'
             '<div class="metric-value" style="opacity:.4">--</div>'
             '<div class="metric-sub">Not enough data to calculate cigarette equivalence.</div>')

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

(tab_live, tab_when, tab_forecast, tab_history) = st.tabs(
    ["Live map", "When to go out", "Tomorrow", "History"])

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

with tab_when:
    st.markdown("### When should I go outside?")
    if status["empty"]:
        st.info("Building the answer from real CPCB history — the archive "
                "download is still running. This will fill in automatically.")
    else:
        hist = data_loader.history(stations=[selected] if selected else None,
                                   days=RECENT_WINDOW_DAYS, collected_only=True)
        if len(hist) == 0 and selected:
            st.info(f"No stored history for {short(selected)} yet in recent window. "
                    "Showing citywide recent profile.")
            hist = data_loader.history(days=RECENT_WINDOW_DAYS, collected_only=True)

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

with tab_history:
    st.markdown("### What the record shows")
    st.markdown('<div class="note">Every chart here is computed from observed '
                'CPCB station readings collected hour by hour. The forecasting '
                'model plays no part in any of them.</div>',
                unsafe_allow_html=True)
    st.markdown("")

    if status["empty"]:
        st.info("No hours collected yet. These charts fill in as the hourly "
                "collector runs.")
    else:
        period = st.radio("Period", ["24 hours", "7 days", "30 days"],
                          horizontal=True, label_visibility="collapsed")
        days = {"24 hours": 1, "7 days": 7, "30 days": 30}[period]

        hist = data_loader.history(stations=[selected] if selected else None,
                                   days=days + 1, collected_only=True)

        span_h = 0
        if len(hist):
            span = hist["datetime"].max() - hist["datetime"].min()
            span_h = span.total_seconds() / 3600
        if span_h < days * 24:
            st.info(f"Collection covers {span_h:.0f} hours so far, less than the "
                    f"{days * 24} this view spans. The collector adds an hour "
                    "every hour; the live feed publishes no history, so earlier "
                    "hours cannot be fetched after the fact.", icon="⏳")

        if period == "24 hours":
            cutoff = pd.Timestamp.now().floor("h") - pd.Timedelta(hours=24)
            recent = hist[hist["datetime"] >= cutoff]
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

with tab_forecast:
    st.markdown("## The next 24 hours")

    forecast = data_loader.forecast_24h(selected) if selected else None

    if forecast is not None:
        start = pd.Timestamp(forecast["anchor"])
        lag_hours = (pd.Timestamp.now() - start).total_seconds() / 3600
        if lag_hours > 6:
            st.warning(
                f"This forecasts the 24 hours after **{start:%d %b, %H:%M}**, "
                f"{lag_hours / 24:.0f} days ago — not the 24 hours after now. "
                f"No later hour has enough readings behind it to start from.",
                icon="🕐")
        elif forecast["frac_observed"] < 0.6:
            st.warning(
                f"Built from **{forecast['frac_observed']:.0%} real readings**. "
                f"The rest of the input week is carried forward from the last "
                f"value seen, because collection started recently and the live "
                f"feed cannot backfill. The model was trained on near-complete "
                f"weeks, so treat this as indicative until the window fills — "
                f"it improves every hour the collector runs.",
                icon="⚠️")

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
                     f'Forecast PM2.5 sub-index peaks near '
                     f'{frame["pm2_5"].max():.0f} AQI.</div>')

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


