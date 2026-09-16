from calendar import monthrange
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from data_loader import load_usage_sources
from pricing_client import enrich_meter_records


st.set_page_config(page_title="Azure Cost Intelligence", page_icon="☁️", layout="wide")

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = APP_DIR / "AzureUsage 1.csv"

DARK_HUES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9", "#00a6a6"]
OTHER_COLOR = "#6b6a64"
SURFACE = "#1a1a19"
GRID = "#2c2c2a"
INK_SECONDARY = "#c3c2b7"
INK_MUTED = "#898781"

PRESETS = [
    "All available data",
    "This week",
    "Last 7 days",
    "Last 30 days",
    "Last 12 weeks",
    "This month",
    "Last 6 months",
    "Year to date",
    "Custom range",
]


@st.cache_data(show_spinner="Reading Azure usage data…")
def load_data(sources) -> tuple[pd.DataFrame, dict]:
    return load_usage_sources(sources)


def resolve_date_range(preset: str, custom_range, min_date, max_date):
    end = max_date
    if preset == "All available data":
        start = min_date
    elif preset == "This week":
        start = max(min_date, end - pd.Timedelta(days=end.weekday()))
    elif preset == "Last 7 days":
        start = max(min_date, end - pd.Timedelta(days=6))
    elif preset == "Last 30 days":
        start = max(min_date, end - pd.Timedelta(days=29))
    elif preset == "Last 12 weeks":
        start = max(min_date, end - pd.Timedelta(weeks=12) + pd.Timedelta(days=1))
    elif preset == "This month":
        start = max(min_date, end.replace(day=1))
    elif preset == "Last 6 months":
        start = max(min_date, (pd.Timestamp(end).to_period("M") - 5).start_time.date())
    elif preset == "Year to date":
        start = max(min_date, end.replace(month=1, day=1))
    else:
        if isinstance(custom_range, (tuple, list)) and len(custom_range) == 2:
            start, end = custom_range
        else:
            start, end = min_date, max_date
    return max(start, min_date), min(end, max_date)


def period_start(dates: pd.Series, granularity: str) -> pd.Series:
    if granularity == "Daily":
        return dates.dt.floor("D")
    if granularity == "Weekly":
        # W-SUN periods begin on Monday and end on Sunday.
        return dates.dt.to_period("W-SUN").dt.start_time
    return dates.dt.to_period("M").dt.start_time


def comparison_windows(end_date, cadence: str):
    end = pd.Timestamp(end_date).normalize()
    if cadence == "Daily":
        current_start = end
        previous_start = end - pd.Timedelta(days=1)
        previous_end = previous_start
    elif cadence == "Weekly":
        current_start = end - pd.Timedelta(days=end.weekday())
        elapsed = end - current_start
        previous_start = current_start - pd.Timedelta(days=7)
        previous_end = previous_start + elapsed
    else:
        current_start = end.replace(day=1)
        previous_month_end = current_start - pd.Timedelta(days=1)
        previous_start = previous_month_end.replace(day=1)
        comparable_day = min(end.day, previous_month_end.day)
        previous_end = previous_start.replace(day=comparable_day)
    return current_start, end, previous_start, previous_end


def style_fig(fig, showlegend: bool = True):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font_color=INK_SECONDARY,
        legend_title_text=None,
        showlegend=showlegend,
        margin=dict(l=10, r=10, t=45, b=10),
        hoverlabel=dict(bgcolor="#242422"),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, color=INK_MUTED)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, color=INK_MUTED)
    return fig


def money(value):
    return f"${value:,.2f}"


def percent_change(current, previous):
    if previous == 0:
        return None
    return (current - previous) / previous


def comparison_label(start, end):
    return start.strftime("%b %d") if start == end else f"{start:%b %d}–{end:%b %d}"


def build_meter_catalog(
    dimension_scope: pd.DataFrame,
    filtered: pd.DataFrame,
    current_period: pd.DataFrame,
    previous_period: pd.DataFrame,
) -> pd.DataFrame:
    """Build one auditable row per billing meter for the active filters."""
    catalog = (
        dimension_scope.groupby("ResourceGuid", as_index=False)
        .agg(
            ServiceName=("ServiceName", "first"),
            ServiceType=("ServiceType", "first"),
            ServiceRegion=("ServiceRegion", "first"),
            ServiceResource=("ServiceResource", "first"),
            SubscriptionCount=("SubscriptionName", "nunique"),
            Subscriptions=(
                "SubscriptionName",
                lambda values: ", ".join(sorted(set(values.astype(str)))),
            ),
            FirstUsage=("Date", "min"),
            LastUsage=("Date", "max"),
        )
    )

    def period_totals(frame, prefix):
        return (
            frame.groupby("ResourceGuid", as_index=False)
            .agg(**{
                f"{prefix}Cost": ("Cost", "sum"),
                f"{prefix}Quantity": ("Quantity", "sum"),
            })
        )

    filtered_totals = (
        filtered.groupby("ResourceGuid", as_index=False)
        .agg(
            FilteredCost=("Cost", "sum"),
            FilteredQuantity=("Quantity", "sum"),
            UsageRows=("Date", "size"),
        )
    )
    subscription_costs = (
        filtered.groupby(["ResourceGuid", "SubscriptionName"], as_index=False)["Cost"]
        .sum()
        .sort_values(["ResourceGuid", "Cost"], ascending=[True, False])
    )
    subscription_splits = (
        subscription_costs.assign(
            SubscriptionCost=subscription_costs.apply(
                lambda row: f"{row['SubscriptionName']}: {money(row['Cost'])}", axis=1
            )
        )
        .groupby("ResourceGuid", as_index=False)["SubscriptionCost"]
        .agg("; ".join)
        .rename(columns={"SubscriptionCost": "FilteredSubscriptionCosts"})
    )
    catalog = catalog.merge(filtered_totals, on="ResourceGuid", how="left")
    catalog = catalog.merge(subscription_splits, on="ResourceGuid", how="left")
    catalog["FilteredSubscriptionCosts"] = catalog["FilteredSubscriptionCosts"].fillna(
        "No usage in filtered range"
    )
    catalog = catalog.merge(period_totals(current_period, "Current"), on="ResourceGuid", how="left")
    catalog = catalog.merge(period_totals(previous_period, "Previous"), on="ResourceGuid", how="left")
    numeric_columns = [
        "FilteredCost",
        "FilteredQuantity",
        "UsageRows",
        "CurrentCost",
        "CurrentQuantity",
        "PreviousCost",
        "PreviousQuantity",
    ]
    catalog[numeric_columns] = catalog[numeric_columns].fillna(0)
    catalog["CostChange"] = catalog["CurrentCost"] - catalog["PreviousCost"]
    catalog["QuantityChange"] = catalog["CurrentQuantity"] - catalog["PreviousQuantity"]
    catalog["CostChangePct"] = (catalog["CostChange"] / catalog["PreviousCost"]).where(
        catalog["PreviousCost"].ne(0)
    )
    catalog["QuantityChangePct"] = (
        catalog["QuantityChange"] / catalog["PreviousQuantity"]
    ).where(catalog["PreviousQuantity"].ne(0))
    catalog["EffectiveCostPerUnit"] = (
        catalog["FilteredCost"] / catalog["FilteredQuantity"]
    ).where(catalog["FilteredQuantity"].ne(0))
    return catalog.sort_values("FilteredCost", ascending=False).reset_index(drop=True)


def merge_pricing(catalog: pd.DataFrame, pricing_results: dict) -> pd.DataFrame:
    if not pricing_results:
        return catalog.copy()
    pricing = pd.DataFrame(pricing_results.values())
    return catalog.merge(pricing, on="ResourceGuid", how="left")


def meter_display_label(row) -> str:
    return (
        f"{row['ServiceName']} / {row['ServiceType']} / {row['ServiceResource']} "
        f"({str(row['ResourceGuid'])[:8]})"
    )


st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem;}
        [data-testid="stMetric"] {min-height: 118px;}
        [data-testid="stMetricLabel"] {color: #c3c2b7;}
        div[data-testid="stPlotlyChart"] {border: 1px solid #2c2c2a; border-radius: 10px; overflow: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

title_col, status_col = st.columns([4, 1])
with title_col:
    st.title("Azure Cost Intelligence")
    st.caption("Understand spend, isolate cost drivers, and drill from subscriptions to the finest identity in the export.")

with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader(
        "Upload Azure usage files",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        help="Select one or more CSV or Excel exports. Matching rows are combined for analysis.",
    )
    sources = tuple(uploaded) if uploaded else (DEFAULT_CSV,)

try:
    df, data_meta = load_data(sources)
except (FileNotFoundError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()
subscriptions = sorted(df["SubscriptionName"].unique())
services = sorted(df["ServiceName"].unique())
meters = tuple(sorted(df["ResourceGuid"].unique()))
source_signature = tuple(
    (
        item["Source file"],
        item["Worksheet"],
        item["Rows"],
        round(float(item["Cost"]), 6),
    )
    for item in data_meta["source_summaries"]
)
data_signature = (
    len(df),
    str(min_date),
    str(max_date),
    tuple(subscriptions),
    tuple(services),
    meters,
    source_signature,
)

if st.session_state.get("data_signature") != data_signature:
    st.session_state.data_signature = data_signature
    default_start, default_end = resolve_date_range("Last 6 months", None, min_date, max_date)
    st.session_state.applied_filters = {
        "preset": "Last 6 months",
        "start": default_start,
        "end": default_end,
        "subscriptions": subscriptions,
        "services": services,
        "granularity": "Weekly",
        "top_n": min(7, max(3, len(services))),
    }
    st.session_state.drill_path = []
    st.session_state.pricing_results = {}

applied = st.session_state.applied_filters

with status_col:
    st.metric("Rows loaded", f"{len(df):,}")

with st.sidebar:
    file_word = "file" if data_meta["source_file_count"] == 1 else "files"
    st.caption(
        f"Loaded {data_meta['source_file_count']} {file_word} · {len(df):,} rows · "
        f"{len(subscriptions)} subscriptions"
    )
    with st.expander("Loaded sources"):
        for item in data_meta["source_summaries"]:
            worksheet = "" if item["Worksheet"] == "CSV" else f" / {item['Worksheet']}"
            st.markdown(
                f"**{item['Source file']}{worksheet}**  \n"
                f"{item['Rows']:,} rows · {item['Subscriptions']} subscriptions · {money(item['Cost'])}"
            )
    if data_meta["date_corrections"]:
        st.warning(
            f"Corrected {data_meta['date_corrections']:,} dates that Excel partially converted "
            "with the month and day reversed."
        )
    st.caption(f"Data available: {min_date:%d %b %Y} – {max_date:%d %b %Y}")
    st.header("Filters")
    st.caption("Changes take effect only when you click Apply filters.")

    with st.form("filter_form"):
        preset = st.selectbox("Date preset", PRESETS, index=PRESETS.index(applied["preset"]))
        custom_dates = st.date_input(
            "Custom dates",
            value=(applied["start"], applied["end"]),
            min_value=min_date,
            max_value=max_date,
            disabled=preset != "Custom range",
        )
        selected_subscriptions = st.multiselect(
            "Subscriptions",
            subscriptions,
            default=[value for value in applied["subscriptions"] if value in subscriptions],
        )
        selected_services = st.multiselect(
            "Services",
            services,
            default=[value for value in applied["services"] if value in services],
        )
        granularity = st.segmented_control(
            "Analyse and compare by",
            ["Daily", "Weekly", "Monthly"],
            default=applied["granularity"],
        )
        chart_limit_max = max(3, min(12, len(services)))
        top_n = st.slider(
            "Services shown in charts",
            3,
            chart_limit_max,
            min(applied["top_n"], chart_limit_max),
        )
        submitted = st.form_submit_button("Apply filters", type="primary", width="stretch")

    if submitted:
        start_date, end_date = resolve_date_range(preset, custom_dates, min_date, max_date)
        st.session_state.applied_filters = {
            "preset": preset,
            "start": start_date,
            "end": end_date,
            "subscriptions": selected_subscriptions or subscriptions,
            "services": selected_services or services,
            "granularity": granularity or "Weekly",
            "top_n": top_n,
        }
        st.session_state.drill_path = []
        st.rerun()

    st.divider()
    st.caption(
        f"Applied: **{applied['start']:%d %b %Y} – {applied['end']:%d %b %Y}**  \n"
        f"{len(applied['subscriptions'])} subscriptions · {len(applied['services'])} services"
    )

dimension_mask = (
    df["SubscriptionName"].isin(applied["subscriptions"])
    & df["ServiceName"].isin(applied["services"])
)
dimension_df = df.loc[dimension_mask].copy()
date_mask = (
    (dimension_df["Date"].dt.date >= applied["start"])
    & (dimension_df["Date"].dt.date <= applied["end"])
)
fdf = dimension_df.loc[date_mask].copy()

if fdf.empty:
    st.warning("No usage rows match the applied filters. Change the filters and click Apply filters.")
    st.stop()

subscription_summary = (
    fdf.groupby("SubscriptionName", as_index=False)
    .agg(Cost=("Cost", "sum"), UsageRows=("Cost", "size"), Meters=("ResourceGuid", "nunique"))
    .sort_values("Cost", ascending=False)
)
contributing_subscriptions = subscription_summary["SubscriptionName"].tolist()
noncontributing_subscriptions = sorted(
    set(applied["subscriptions"]) - set(contributing_subscriptions)
)

with st.sidebar:
    st.subheader("Data included")
    for row in subscription_summary.itertuples(index=False):
        st.markdown(f"**{row.SubscriptionName}**  \n{money(row.Cost)} · {row.UsageRows:,} rows")
    if noncontributing_subscriptions:
        st.caption(
            "No rows in the applied date range: " + ", ".join(noncontributing_subscriptions)
        )

granularity = applied["granularity"]
top_n = applied["top_n"]
current_start, current_end, previous_start, previous_end = comparison_windows(applied["end"], granularity)
current_df = dimension_df[
    (dimension_df["Date"] >= current_start) & (dimension_df["Date"] <= current_end)
]
previous_df = dimension_df[
    (dimension_df["Date"] >= previous_start) & (dimension_df["Date"] <= previous_end)
]
meter_catalog = build_meter_catalog(dimension_df, fdf, current_df, previous_df)

total_cost = fdf["Cost"].sum()
current_cost = current_df["Cost"].sum()
previous_cost = previous_df["Cost"].sum()
change = percent_change(current_cost, previous_cost)
observed_days = max((current_end - current_start).days + 1, 1)
if granularity == "Monthly":
    target_days = monthrange(current_end.year, current_end.month)[1]
elif granularity == "Weekly":
    target_days = 7
else:
    target_days = 1
projected_cost = current_cost / observed_days * target_days
daily_cost = fdf.groupby(fdf["Date"].dt.date)["Cost"].sum()
average_daily = daily_cost.mean()
service_cost = fdf.groupby("ServiceName")["Cost"].sum().sort_values(ascending=False)
top_service = service_cost.index[0]
top_service_share = service_cost.iloc[0] / total_cost if total_cost else 0

st.caption(
    f"Showing {len(fdf):,} rows · {applied['start']:%d %b %Y} to {applied['end']:%d %b %Y} · "
    f"{granularity.lower()} comparison"
)
if len(applied["subscriptions"]) > 1:
    if noncontributing_subscriptions:
        st.warning(
            f"{len(applied['subscriptions'])} subscriptions are selected, but this date range contains usage from "
            f"{len(contributing_subscriptions)}: {', '.join(contributing_subscriptions)}."
        )
    else:
        st.info(
            "This view combines data from: " + ", ".join(contributing_subscriptions) + ". "
            "Subscription attribution is shown below and in the meter catalog."
        )

k1, k2, k3, k4, k5 = st.columns(5)
with k1.container(border=True):
    st.metric("Filtered spend", money(total_cost))
with k2.container(border=True):
    st.metric(
        f"Spend · {comparison_label(current_start, current_end)}",
        money(current_cost),
        f"{change:+.1%} vs prior" if change is not None else "No prior comparison",
        delta_color="inverse",
    )
with k3.container(border=True):
    projection_label = "Projected month" if granularity == "Monthly" else "Projected week" if granularity == "Weekly" else "Latest day"
    st.metric(projection_label, money(projected_cost))
with k4.container(border=True):
    st.metric("Average daily spend", money(average_daily))
with k5.container(border=True):
    st.metric("Largest cost area", top_service, f"{top_service_share:.1%} of filtered spend", delta_color="off")

current_by_service = current_df.groupby("ServiceName")["Cost"].sum().rename("Current")
previous_by_service = previous_df.groupby("ServiceName")["Cost"].sum().rename("Previous")
service_comparison = pd.concat([current_by_service, previous_by_service], axis=1).fillna(0)
service_comparison["Change"] = service_comparison["Current"] - service_comparison["Previous"]
service_comparison = service_comparison.sort_values("Change", ascending=False)

region_cost = fdf.groupby("ServiceRegion")["Cost"].sum().sort_values(ascending=False)
top_region = region_cost.index[0]
top_region_share = region_cost.iloc[0] / total_cost if total_cost else 0
largest_increase = service_comparison.index[0] if not service_comparison.empty else None
largest_increase_value = service_comparison.iloc[0]["Change"] if not service_comparison.empty else 0
peak_day = daily_cost.idxmax()
peak_day_cost = daily_cost.max()

st.subheader("Decision signals")
insight_cols = st.columns(3)
with insight_cols[0]:
    if change is None:
        st.info("No comparable prior-period spend is available for the selected dimensions.")
    elif change > 0:
        st.warning(f"Spend is up {change:.1%} versus {comparison_label(previous_start, previous_end)} ({money(current_cost - previous_cost)} more).")
    else:
        st.success(f"Spend is down {abs(change):.1%} versus {comparison_label(previous_start, previous_end)} ({money(abs(current_cost - previous_cost))} less).")
with insight_cols[1]:
    if largest_increase_value > 0:
        st.info(f"{largest_increase} is the largest period-over-period cost driver, adding {money(largest_increase_value)}.")
    else:
        st.info("No service increased in the comparable prior period.")
with insight_cols[2]:
    st.info(f"{top_region} accounts for {top_region_share:.1%} of spend. The highest-cost day was {peak_day:%d %b %Y} at {money(peak_day_cost)}.")

overview_tab, drivers_tab, resources_tab, meters_tab, details_tab = st.tabs(
    ["Overview", "Cost drivers", "Resource drill-through", "Meter catalog", "Details & data quality"]
)

with overview_tab:
    st.subheader("Cost by subscription")
    subscription_chart = subscription_summary.sort_values("Cost", ascending=True)
    subscription_colors = {
        name: DARK_HUES[index % len(DARK_HUES)]
        for index, name in enumerate(sorted(subscription_chart["SubscriptionName"]))
    }
    fig_subscription = px.bar(
        subscription_chart,
        x="Cost",
        y="SubscriptionName",
        orientation="h",
        color="SubscriptionName",
        color_discrete_map=subscription_colors,
        custom_data=["UsageRows", "Meters"],
        height=max(280, 90 * len(subscription_chart) + 120),
    )
    fig_subscription.update_traces(
        texttemplate="$%{x:,.2f}",
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "%{y}<br>Cost: $%{x:,.2f}<br>Rows: %{customdata[0]:,}"
            "<br>Billing meters: %{customdata[1]:,}<extra></extra>"
        ),
    )
    fig_subscription.update_layout(xaxis_title="Cost ($)", yaxis_title=None)
    style_fig(fig_subscription, showlegend=False)
    st.plotly_chart(fig_subscription, width="stretch")

    st.subheader(f"Cost trend · {granularity.lower()}")
    trend_subscription_options = sorted(fdf["SubscriptionName"].unique())
    trend_filter_key = "trend_subscription_filter"
    trend_options_key = "trend_subscription_options"
    prior_trend_options = st.session_state.get(trend_options_key)
    if prior_trend_options != trend_subscription_options:
        prior_selection = st.session_state.get(trend_filter_key, [])
        valid_selection = [
            value for value in prior_selection if value in trend_subscription_options
        ]
        st.session_state[trend_filter_key] = valid_selection or trend_subscription_options
        st.session_state[trend_options_key] = trend_subscription_options

    trend_subscriptions = st.multiselect(
        "Subscriptions in cost trend",
        trend_subscription_options,
        key=trend_filter_key,
        help="This selection changes only the cost trend chart.",
    )

    if not trend_subscriptions:
        st.info("Select at least one subscription to display the cost trend.")
    else:
        trend_df = fdf[fdf["SubscriptionName"].isin(trend_subscriptions)].copy()
        trend_df["Period"] = period_start(trend_df["Date"], granularity)
        trend_service_cost = trend_df.groupby("ServiceName")["Cost"].sum().sort_values(ascending=False)
        ranked_services = list(trend_service_cost.index[:top_n])
        trend_df["ServiceGroup"] = trend_df["ServiceName"].where(
            trend_df["ServiceName"].isin(ranked_services), "Other"
        )
        trend = trend_df.groupby(["Period", "ServiceGroup"], as_index=False)["Cost"].sum()
        service_colors = {
            name: DARK_HUES[i % len(DARK_HUES)] for i, name in enumerate(ranked_services)
        }
        service_colors["Other"] = OTHER_COLOR
        order = [name for name in ranked_services if name in trend["ServiceGroup"].unique()]
        if "Other" in trend["ServiceGroup"].unique():
            order.append("Other")

        fig_trend = px.bar(
            trend,
            x="Period",
            y="Cost",
            color="ServiceGroup",
            color_discrete_map=service_colors,
            category_orders={"ServiceGroup": order},
            labels={"ServiceGroup": "Service"},
        )
        fig_trend.update_layout(xaxis_title=None, yaxis_title="Cost ($)", barmode="stack")
        fig_trend.update_traces(
            hovertemplate="%{x|%d %b %Y}<br>%{fullData.name}: $%{y:,.2f}<extra></extra>"
        )
        style_fig(fig_trend)
        st.plotly_chart(fig_trend, width="stretch")

    st.subheader("Where the money goes")
    tree_data = fdf.groupby(["SubscriptionName", "ServiceName", "ServiceRegion"], as_index=False)["Cost"].sum()
    fig_tree = px.treemap(
        tree_data,
        path=[px.Constant("All spend"), "SubscriptionName", "ServiceName", "ServiceRegion"],
        values="Cost",
        color="Cost",
        color_continuous_scale=["#242422", "#3987e5", "#d95926"],
    )
    fig_tree.update_traces(hovertemplate="%{label}<br>$%{value:,.2f}<br>%{percentRoot:.1%} of total<extra></extra>")
    fig_tree.update_layout(coloraxis_showscale=False)
    style_fig(fig_tree, showlegend=False)
    st.plotly_chart(fig_tree, width="stretch")

with drivers_tab:
    left, right = st.columns(2)
    with left:
        service_breakdown = service_cost.head(top_n).sort_values()
        fig_service = px.bar(
            x=service_breakdown.values,
            y=service_breakdown.index,
            orientation="h",
            labels={"x": "Cost ($)", "y": "Service"},
            title="Highest-cost services",
        )
        fig_service.update_traces(marker_color="#3987e5", hovertemplate="%{y}<br>$%{x:,.2f}<extra></extra>")
        fig_service.update_layout(yaxis_title=None)
        style_fig(fig_service, showlegend=False)
        st.plotly_chart(fig_service, width="stretch")

    with right:
        if previous_df.empty:
            st.info("There is no prior-period data for a service change chart.")
        else:
            driver_data = service_comparison.reindex(
                service_comparison["Change"].abs().nlargest(top_n).index
            ).sort_values("Change")
            driver_data["Direction"] = driver_data["Change"].apply(
                lambda value: "Increase" if value > 0 else "Decrease"
            )
            fig_driver = px.bar(
                driver_data.reset_index(),
                x="Change",
                y="ServiceName",
                orientation="h",
                color="Direction",
                color_discrete_map={"Increase": "#d95926", "Decrease": "#199e70"},
                title=f"Change vs {comparison_label(previous_start, previous_end)}",
            )
            fig_driver.update_traces(hovertemplate="%{y}<br>Change: $%{x:,.2f}<extra></extra>")
            fig_driver.update_layout(xaxis_title="Cost change ($)", yaxis_title=None)
            style_fig(fig_driver)
            st.plotly_chart(fig_driver, width="stretch")

    region_breakdown = region_cost.reset_index()
    fig_region = px.bar(
        region_breakdown,
        x="ServiceRegion",
        y="Cost",
        color="ServiceRegion",
        title="Regional cost concentration",
        color_discrete_sequence=DARK_HUES,
    )
    fig_region.update_layout(xaxis_title=None, yaxis_title="Cost ($)")
    fig_region.update_traces(hovertemplate="%{x}<br>$%{y:,.2f}<extra></extra>")
    style_fig(fig_region, showlegend=False)
    st.plotly_chart(fig_region, width="stretch")

with resources_tab:
    if data_meta["has_resource_identity"] or data_meta["has_resource_group"]:
        available_fields = []
        if data_meta["has_resource_group"]:
            available_fields.append(f"resource groups from `{data_meta['resource_group_source']}`")
        if data_meta["has_resource_identity"]:
            available_fields.append(f"resource names from `{data_meta['identity_source']}`")
        st.success(
            "This export includes " + " and ".join(available_fields) + "."
        )
    else:
        st.warning(
            "This CSV does not include ResourceGroup, ResourceName, or ResourceId. The deepest available level "
            "is the Azure billing meter. A meter identifies how usage is charged, not an individual VM."
        )

    drill_levels = [("SubscriptionName", "Subscription")]
    if data_meta["has_resource_group"]:
        drill_levels.append(("AnalysisResourceGroup", "Resource group"))
    drill_levels.extend([
        ("ServiceName", "Service"),
        ("ServiceType", "Service type"),
        ("ServiceRegion", "Region"),
        ("ServiceResource", "Resource SKU"),
    ])
    if data_meta["has_resource_identity"]:
        drill_levels.append(("AnalysisResourceName", "Resource name"))
    drill_levels.append(("ResourceGuid", "Billing meter"))

    valid_path = []
    scoped = fdf
    for (level, _), value in zip(drill_levels, st.session_state.drill_path):
        if value in scoped[level].unique():
            valid_path.append(value)
            scoped = scoped[scoped[level] == value]
        else:
            break
    st.session_state.drill_path = valid_path

    crumb_cols = st.columns([1] * (len(valid_path) + 1) + [3])
    if crumb_cols[0].button("🏠 All", width="stretch"):
        st.session_state.drill_path = []
        st.rerun()
    for index, value in enumerate(valid_path):
        short_value = value if len(str(value)) <= 28 else f"{str(value)[:25]}…"
        if crumb_cols[index + 1].button(f"{short_value} ✕", width="stretch"):
            st.session_state.drill_path = valid_path[:index]
            st.rerun()

    scoped = fdf
    for (level, _), value in zip(drill_levels, valid_path):
        scoped = scoped[scoped[level] == value]

    level_index = len(valid_path)
    if level_index == len(drill_levels):
        row = scoped.iloc[0]
        meter_id = str(row["ResourceGuid"])
        scoped_quantity = scoped["Quantity"].sum()
        scoped_rate = scoped["Cost"].sum() / scoped_quantity if scoped_quantity else None

        comparison_current = current_df
        comparison_previous = previous_df
        for (level, _), value in zip(drill_levels, valid_path):
            comparison_current = comparison_current[comparison_current[level] == value]
            comparison_previous = comparison_previous[comparison_previous[level] == value]
        scoped_current_cost = comparison_current["Cost"].sum()
        scoped_previous_cost = comparison_previous["Cost"].sum()
        scoped_change = percent_change(scoped_current_cost, scoped_previous_cost)

        st.subheader(f"Billing meter {meter_id[:8]}")
        st.caption(
            f"Subscription: {row['SubscriptionName']}  \n"
            f"{row['ServiceName']} / {row['ServiceType']} / {row['ServiceResource']} / {row['ServiceRegion']}"
        )
        atomic_cols = st.columns(4)
        atomic_cols[0].metric("Cost", money(scoped["Cost"].sum()))
        atomic_cols[1].metric("Quantity", f"{scoped_quantity:,.4f}")
        atomic_cols[2].metric(
            "Effective cost per quantity",
            money(scoped_rate) if scoped_rate is not None else "n.a.",
        )
        atomic_cols[3].metric(
            f"{granularity} cost change",
            f"{scoped_change:+.1%}" if scoped_change is not None else "n.a.",
        )
        if data_meta["has_resource_group"]:
            st.caption(f"Resource group: **{row['AnalysisResourceGroup']}**")
        if data_meta["resource_id_column"]:
            with st.expander("Full Azure Resource ID"):
                st.code(row["AnalysisResourceId"])
        with st.expander("Full billing meter GUID"):
            st.code(meter_id)

        pricing_result = st.session_state.pricing_results.get(meter_id)
        if pricing_result and pricing_result.get("PricingStatus") == "Matched":
            st.subheader("Public retail benchmark")
            pricing_cols = st.columns(4)
            pricing_cols[0].metric("Meter name", pricing_result.get("MeterName") or "n.a.")
            pricing_cols[1].metric("Azure SKU", pricing_result.get("AzureSku") or "n.a.")
            pricing_cols[2].metric("Unit", pricing_result.get("UnitOfMeasure") or "n.a.")
            retail_price = pricing_result.get("RetailPriceUSD")
            pricing_cols[3].metric(
                "Current retail price (USD)",
                money(retail_price) if retail_price is not None else "n.a.",
            )
            st.caption(
                f"{pricing_result.get('ProductName') or 'Product unavailable'} · "
                f"{pricing_result.get('PriceType') or 'Price type unavailable'} · "
                "Public retail pricing may differ from historical or negotiated rates."
            )

        atomic_daily = scoped.groupby(scoped["Date"].dt.date, as_index=False).agg(
            Cost=("Cost", "sum"), Quantity=("Quantity", "sum")
        )
        atomic_daily["EffectiveCostPerUnit"] = (
            atomic_daily["Cost"] / atomic_daily["Quantity"]
        ).where(atomic_daily["Quantity"].ne(0))
        fig_atomic = px.line(atomic_daily, x="Date", y="Cost", markers=True, title="Meter cost over time")
        fig_atomic.update_traces(line_color="#3987e5", marker_color="#3987e5")
        fig_atomic.update_layout(xaxis_title=None, yaxis_title="Cost ($)")
        style_fig(fig_atomic, showlegend=False)
        st.plotly_chart(fig_atomic, width="stretch", key="atomic_trend")
    else:
        current_level, current_label = drill_levels[level_index]
        st.subheader(f"Cost by {current_label}")
        st.caption("Select a bar to move to the next level. Use the breadcrumb above to move back.")
        breakdown = (
            scoped.groupby(current_level, as_index=False)["Cost"]
            .sum()
            .sort_values("Cost", ascending=True)
        )
        chart_height = max(380, min(850, 34 * len(breakdown) + 110))
        fig_drill = px.bar(
            breakdown,
            x="Cost",
            y=current_level,
            orientation="h",
            labels={"Cost": "Cost ($)", current_level: current_label},
            height=chart_height,
        )
        fig_drill.update_traces(
            marker_color="#3987e5",
            selected_marker_color="#d95926",
            hovertemplate="%{y}<br>$%{x:,.2f}<extra>Click to drill in</extra>",
        )
        fig_drill.update_layout(clickmode="event+select")
        fig_drill.update_layout(yaxis_title=None)
        style_fig(fig_drill, showlegend=False)
        event = st.plotly_chart(
            fig_drill,
            on_select="rerun",
            selection_mode="points",
            width="stretch",
            key=f"drill_{level_index}_{len(valid_path)}",
        )
        points = (event or {}).get("selection", {}).get("points", [])
        if points:
            clicked = points[0].get("y")
            if clicked is not None:
                st.session_state.drill_path.append(clicked)
                st.rerun()

with meters_tab:
    st.subheader("Billing meter catalog")
    st.caption(
        "Each row is one Azure billing meter. Service type and SKU explain what was charged; "
        "the GUID itself does not identify an individual resource."
    )

    meter_kpis = st.columns(4)
    meter_kpis[0].metric("Billing meters", f"{len(meter_catalog):,}")
    meter_kpis[1].metric("Meter cost", money(meter_catalog["FilteredCost"].sum()))
    meter_kpis[2].metric("Services", f"{meter_catalog['ServiceName'].nunique():,}")
    meter_kpis[3].metric("SKUs", f"{meter_catalog['ServiceResource'].nunique():,}")

    enrich_col, note_col = st.columns([1, 3])
    with enrich_col:
        enrich_clicked = st.button("Enrich meter details", type="primary", width="stretch")
    with note_col:
        st.caption(
            "Looks up current public USD retail information from Azure. Results are cached for 24 hours. "
            "They are benchmarks, not historical or negotiated prices."
        )

    if enrich_clicked:
        context_columns = [
            "ResourceGuid",
            "ServiceName",
            "ServiceType",
            "ServiceRegion",
            "ServiceResource",
        ]
        with st.spinner(f"Looking up {len(meter_catalog):,} billing meters…"):
            enrichment = enrich_meter_records(meter_catalog[context_columns].to_dict("records"))
        st.session_state.pricing_results.update(
            {result["ResourceGuid"]: result for result in enrichment}
        )

    enriched_catalog = merge_pricing(meter_catalog, st.session_state.pricing_results)
    if st.session_state.pricing_results:
        statuses = pd.Series(
            [result.get("PricingStatus", "Unavailable") for result in st.session_state.pricing_results.values()]
        ).value_counts()
        matched = int(statuses.get("Matched", 0))
        not_found = int(statuses.get("Not found", 0))
        unavailable = int(statuses.get("Unavailable", 0))
        if unavailable:
            st.warning(
                f"Pricing enrichment: {matched} matched, {not_found} not found, and {unavailable} unavailable. "
                "CSV-based analysis remains complete."
            )
        else:
            st.info(f"Pricing enrichment: {matched} matched and {not_found} not found in current public retail data.")

    meter_search = st.text_input(
        "Search meters",
        placeholder="Search service, type, region, SKU, meter name, product, or GUID",
    ).strip()

    catalog_columns = [
        "ResourceGuid",
        "ServiceName",
        "ServiceType",
        "ServiceRegion",
        "ServiceResource",
        "Subscriptions",
        "FilteredSubscriptionCosts",
        "FilteredCost",
        "FilteredQuantity",
        "EffectiveCostPerUnit",
        "CurrentCost",
        "PreviousCost",
        "CostChange",
        "CostChangePct",
        "QuantityChangePct",
        "FirstUsage",
        "LastUsage",
        "SubscriptionCount",
    ]
    pricing_columns = [
        "PricingStatus",
        "MeterName",
        "ProductName",
        "AzureSku",
        "ServiceFamily",
        "UnitOfMeasure",
        "PriceType",
        "RetailPriceUSD",
        "RetailRegion",
        "EffectiveStartDate",
    ]
    catalog_columns.extend(column for column in pricing_columns if column in enriched_catalog.columns)
    catalog_view = enriched_catalog[catalog_columns].copy()
    if meter_search:
        search_mask = catalog_view.astype(str).apply(
            lambda column: column.str.contains(meter_search, case=False, na=False)
        ).any(axis=1)
        catalog_view = catalog_view.loc[search_mask]

    st.dataframe(
        catalog_view,
        width="stretch",
        hide_index=True,
        column_config={
            "ResourceGuid": st.column_config.TextColumn("Billing meter GUID", width="large"),
            "ServiceName": st.column_config.TextColumn("Service"),
            "ServiceType": st.column_config.TextColumn("Service type"),
            "ServiceRegion": st.column_config.TextColumn("Region"),
            "ServiceResource": st.column_config.TextColumn("SKU"),
            "Subscriptions": st.column_config.TextColumn("Subscriptions", width="large"),
            "FilteredSubscriptionCosts": st.column_config.TextColumn(
                "Cost by subscription", width="large"
            ),
            "FilteredCost": st.column_config.NumberColumn("Filtered cost", format="$%.2f"),
            "FilteredQuantity": st.column_config.NumberColumn("Quantity", format="%.4f"),
            "EffectiveCostPerUnit": st.column_config.NumberColumn("Effective cost / quantity", format="$%.6f"),
            "CurrentCost": st.column_config.NumberColumn("Current-period cost", format="$%.2f"),
            "PreviousCost": st.column_config.NumberColumn("Prior comparable cost", format="$%.2f"),
            "CostChange": st.column_config.NumberColumn("Cost change", format="$%.2f"),
            "CostChangePct": st.column_config.NumberColumn("Cost change %", format="percent"),
            "QuantityChangePct": st.column_config.NumberColumn("Quantity change %", format="percent"),
            "FirstUsage": st.column_config.DateColumn("First usage", format="DD MMM YYYY"),
            "LastUsage": st.column_config.DateColumn("Last usage", format="DD MMM YYYY"),
            "RetailPriceUSD": st.column_config.NumberColumn("Current retail price (USD)", format="$%.6f"),
        },
    )
    st.download_button(
        "Download meter catalog",
        enriched_catalog.to_csv(index=False).encode("utf-8"),
        file_name=f"azure_meter_catalog_{applied['start']}_{applied['end']}.csv",
        mime="text/csv",
    )

    st.subheader("Meter analysis")
    meter_options = meter_catalog.loc[meter_catalog["UsageRows"].gt(0), "ResourceGuid"].tolist()
    meter_labels = {
        row["ResourceGuid"]: meter_display_label(row)
        for _, row in meter_catalog.iterrows()
    }
    selected_meter = st.selectbox(
        "Billing meter",
        meter_options,
        format_func=lambda meter_id: meter_labels[meter_id],
    )
    selected_meter_summary = meter_catalog.loc[
        meter_catalog["ResourceGuid"] == selected_meter
    ].iloc[0]
    st.caption(
        f"Subscriptions: {selected_meter_summary['Subscriptions']}  \n"
        f"Filtered cost attribution: {selected_meter_summary['FilteredSubscriptionCosts']}"
    )
    selected_meter_rows = fdf[fdf["ResourceGuid"] == selected_meter].copy()
    selected_meter_rows["Period"] = period_start(selected_meter_rows["Date"], granularity)
    meter_trend = selected_meter_rows.groupby("Period", as_index=False).agg(
        Cost=("Cost", "sum"), Quantity=("Quantity", "sum")
    )
    meter_trend["EffectiveCostPerUnit"] = (
        meter_trend["Cost"] / meter_trend["Quantity"]
    ).where(meter_trend["Quantity"].ne(0))

    cost_col, quantity_col = st.columns(2)
    with cost_col:
        fig_meter_cost = px.bar(meter_trend, x="Period", y="Cost", title="Meter cost")
        fig_meter_cost.update_traces(marker_color="#3987e5", hovertemplate="%{x|%d %b %Y}<br>$%{y:,.2f}<extra></extra>")
        fig_meter_cost.update_layout(xaxis_title=None, yaxis_title="Cost ($)")
        style_fig(fig_meter_cost, showlegend=False)
        st.plotly_chart(fig_meter_cost, width="stretch")
    with quantity_col:
        fig_meter_quantity = px.line(
            meter_trend,
            x="Period",
            y="Quantity",
            markers=True,
            title="Meter quantity",
        )
        fig_meter_quantity.update_traces(line_color="#c98500", marker_color="#c98500")
        fig_meter_quantity.update_layout(xaxis_title=None, yaxis_title="Quantity")
        style_fig(fig_meter_quantity, showlegend=False)
        st.plotly_chart(fig_meter_quantity, width="stretch")

    rate_col, change_col = st.columns(2)
    with rate_col:
        rate_trend = meter_trend.dropna(subset=["EffectiveCostPerUnit"])
        if rate_trend.empty:
            st.info("Effective cost per quantity is unavailable because this meter has zero quantity.")
        else:
            fig_meter_rate = px.line(
                rate_trend,
                x="Period",
                y="EffectiveCostPerUnit",
                markers=True,
                title="Effective cost per quantity",
            )
            fig_meter_rate.update_traces(line_color="#199e70", marker_color="#199e70")
            fig_meter_rate.update_layout(xaxis_title=None, yaxis_title="Cost / quantity ($)")
            style_fig(fig_meter_rate, showlegend=False)
            st.plotly_chart(fig_meter_rate, width="stretch")
    with change_col:
        change_data = meter_catalog.reindex(
            meter_catalog["CostChange"].abs().nlargest(min(12, len(meter_catalog))).index
        ).copy()
        change_data["Meter"] = change_data.apply(
            lambda row: f"{row['ServiceResource']} ({row['ResourceGuid'][:8]})",
            axis=1,
        )
        change_data["Direction"] = change_data["CostChange"].apply(
            lambda value: "Increase" if value > 0 else "Decrease"
        )
        change_data = change_data.sort_values("CostChange")
        fig_meter_change = px.bar(
            change_data,
            x="CostChange",
            y="Meter",
            orientation="h",
            color="Direction",
            color_discrete_map={"Increase": "#d95926", "Decrease": "#199e70"},
            title=f"Largest {granularity.lower()} meter changes",
        )
        fig_meter_change.update_layout(xaxis_title="Cost change ($)", yaxis_title=None)
        fig_meter_change.update_traces(hovertemplate="%{y}<br>$%{x:,.2f}<extra></extra>")
        style_fig(fig_meter_change)
        st.plotly_chart(fig_meter_change, width="stretch")

with details_tab:
    st.subheader("Filtered usage details")
    st.caption(f"{len(fdf):,} rows match the applied sidebar filters.")
    detail_columns = ["Date", "SubscriptionName"]
    if data_meta["source_file_count"] > 1 or data_meta["source_table_count"] > 1:
        detail_columns.extend(["AnalysisSourceFile", "AnalysisSourceSheet"])
    if data_meta["has_resource_group"]:
        detail_columns.append("AnalysisResourceGroup")
    detail_columns.extend(["ServiceName", "ServiceType", "ServiceRegion", "ServiceResource"])
    if data_meta["has_resource_identity"]:
        detail_columns.append("AnalysisResourceName")
    detail_columns.extend(["ResourceGuid", "Quantity", "Cost"])
    details = fdf[detail_columns].sort_values("Date", ascending=False).rename(
        columns={
            "AnalysisResourceGroup": "Resource group",
            "AnalysisResourceName": "Resource name",
            "AnalysisSourceFile": "Source file",
            "AnalysisSourceSheet": "Worksheet",
            "ResourceGuid": "Billing meter GUID",
        }
    )
    st.dataframe(
        details,
        width="stretch",
        hide_index=True,
        column_config={
            "Cost": st.column_config.NumberColumn(format="$%.2f"),
            "Quantity": st.column_config.NumberColumn(format="%.4f"),
            "Date": st.column_config.DateColumn(format="DD MMM YYYY"),
        },
    )
    st.download_button(
        "Download filtered rows",
        fdf[data_meta["download_columns"]].to_csv(index=False).encode("utf-8"),
        file_name=f"azure_usage_{applied['start']}_{applied['end']}.csv",
        mime="text/csv",
    )

    st.subheader("Data quality and interpretation")
    st.caption("Loaded-source reconciliation")
    source_summary = pd.DataFrame(data_meta["source_summaries"])
    st.dataframe(
        source_summary,
        width="stretch",
        hide_index=True,
        column_config={
            "Cost": st.column_config.NumberColumn(format="$%.2f"),
            "First usage": st.column_config.DateColumn(format="DD MMM YYYY"),
            "Last usage": st.column_config.DateColumn(format="DD MMM YYYY"),
        },
    )
    quality_cols = st.columns(5)
    quality_cols[0].metric("Missing required values", "0")
    quality_cols[1].metric("Duplicate rows", f"{df.duplicated(subset=data_meta['original_columns']).sum():,}")
    quality_cols[2].metric("Negative cost rows", f"{(df['Cost'] < 0).sum():,}")
    quality_cols[3].metric("Billing meters", f"{df['ResourceGuid'].nunique():,}")
    quality_cols[4].metric(
        "Resource groups",
        f"{df['AnalysisResourceGroup'].nunique():,}" if data_meta["has_resource_group"] else "Not provided",
    )
    st.caption(
        "Quantity should not be summed across unrelated Azure meters because units can differ. "
        "Cost is the comparable measure used throughout this dashboard."
    )
