import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from datetime import datetime
import io
import re

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FLOWBIX | Telecom Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 0%, #102b3b 0%, transparent 35%),
        linear-gradient(135deg, #080e17 0%, #0a1420 55%, #07121c 100%);
    color: #e8f4ff;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0e1d2b, #08111c);
    border-right: 1px solid #174054;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1600px;
}

.hero {
    background: linear-gradient(115deg, #102b40, #103d43, #102534);
    border: 1px solid #17606a;
    border-radius: 22px;
    padding: 28px 32px;
    margin-bottom: 22px;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.22);
}

.hero-kicker {
    color: #44f4dc;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 3px;
}

.hero-title {
    font-size: clamp(28px, 4vw, 43px);
    font-weight: 800;
    color: #f3fbff;
    margin: 8px 0;
}

.hero-subtitle {
    color: #a9c6d8;
    font-size: 15px;
}

.section-title {
    color: #eaf8ff;
    font-size: 22px;
    font-weight: 750;
    padding: 10px 0;
}

.metric-card {
    background: linear-gradient(145deg, #122737, #0d1b29);
    border: 1px solid #234456;
    border-radius: 16px;
    padding: 18px 20px;
    min-height: 125px;
    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.16);
}

.metric-label {
    color: #9eb8c9;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.metric-value {
    color: #f2fbff;
    font-size: 29px;
    font-weight: 800;
    margin-top: 12px;
    overflow-wrap: anywhere;
}

.metric-note {
    color: #42e7c7;
    font-size: 12px;
    margin-top: 5px;
}

.panel {
    background: rgba(15, 32, 46, 0.90);
    border: 1px solid #203f51;
    border-radius: 17px;
    padding: 18px;
    margin-bottom: 18px;
}

div[data-testid="stTabs"] button {
    color: #b8d3e2;
    font-weight: 700;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #35f1d4;
}

.stButton button, .stDownloadButton button {
    border-radius: 10px;
    border: 1px solid #168b89;
    background: linear-gradient(100deg, #087e81, #0ba68e);
    color: white;
    font-weight: 700;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================

def normalise_columns(df):
    """Normalise column names so the dashboard accepts common variants."""
    df = df.copy()
    df.columns = [
        re.sub(r"_+", "_",
               re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower())
        ).strip("_")
        for c in df.columns
    ]
    return df


def find_column(df, possible_names):
    """Find a column using common name variants."""
    for name in possible_names:
        if name in df.columns:
            return name

    for col in df.columns:
        for name in possible_names:
            if name in col or col in name:
                return col

    return None


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def format_number(value):
    if pd.isna(value):
        return "N/A"

    value = float(value)

    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.2f}K"

    return f"{value:,.2f}"


def make_chart(fig, height=380):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d8eaf5", family="Inter"),
        margin=dict(l=15, r=15, t=55, b=15),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        xaxis=dict(
            gridcolor="rgba(120,160,180,0.12)",
            zerolinecolor="rgba(120,160,180,0.15)"
        ),
        yaxis=dict(
            gridcolor="rgba(120,160,180,0.12)",
            zerolinecolor="rgba(120,160,180,0.15)"
        )
    )
    return fig


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def prepare_dataset(raw_df):
    """
    Prepare an uploaded CSV by mapping common telecom column names.
    Keep original fields and add standard fields where possible.
    """
    df = normalise_columns(raw_df)

    aliases = {
        "timestamp": [
            "timestamp", "datetime", "date_time", "date",
            "time", "datetime_utc"
        ],
        "tower_id": [
            "tower_id", "tower", "cell_id", "site_id",
            "base_station", "base_station_id", "id_tower"
        ],
        "sms_count": [
            "sms_count", "sms", "sms_total", "sms_activity",
            "sms_in", "sms_out", "sms_in_count", "sms_out_count"
        ],
        "call_count": [
            "call_count", "calls", "call_minutes", "call_duration",
            "voice_traffic", "call_total", "call_in", "call_out"
        ],
        "internet_traffic_mb": [
            "internet_traffic_mb", "internet_traffic",
            "data_traffic_mb", "traffic_mb", "total_traffic_mb",
            "internet_mb", "data_volume_mb"
        ],
        "total_traffic_mb": [
            "total_traffic_mb", "internet_traffic_mb",
            "traffic_mb", "data_traffic_mb", "total_traffic"
        ],
        "next_hour_traffic_mb": [
            "next_hour_traffic_mb", "next_hour_traffic",
            "next_hour_traffic_gb", "target", "target_traffic"
        ],
        "active_users": [
            "active_users", "users", "user_count",
            "active_user_count"
        ],
        "signal_dbm": [
            "signal_dbm", "signal_strength", "rsrp", "signal"
        ],
        "tower_capacity_gb": [
            "tower_capacity_gb", "capacity_gb", "tower_capacity",
            "network_capacity_gb"
        ],
        "area_type": [
            "area_type", "region_type", "area", "region"
        ]
    }

    for standard_name, names in aliases.items():
        if standard_name not in df.columns:
            found = find_column(df, names)
            if found is not None:
                df[standard_name] = df[found]

    # Parse time fields
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], errors="coerce"
        )

    # Support separate date and hour fields
    if (
        "timestamp" not in df.columns
        or df["timestamp"].isna().all()
    ):
        if "date" in df.columns:
            date_values = pd.to_datetime(
                df["date"], errors="coerce"
            )

            if "hour" in df.columns:
                hour_values = pd.to_numeric(
                    df["hour"], errors="coerce"
                ).fillna(0)

                df["timestamp"] = (
                    date_values.dt.normalize()
                    + pd.to_timedelta(hour_values, unit="h")
                )
            else:
                df["timestamp"] = date_values

    # Convert standard measurement columns
    numeric_fields = [
        "sms_count",
        "call_count",
        "internet_traffic_mb",
        "total_traffic_mb",
        "next_hour_traffic_mb",
        "active_users",
        "signal_dbm",
        "tower_capacity_gb"
    ]

    for col in numeric_fields:
        if col in df.columns:
            df[col] = safe_numeric(df[col])

    # Derive traffic from available fields if necessary
    if "total_traffic_mb" not in df.columns:
        if "internet_traffic_mb" in df.columns:
            df["total_traffic_mb"] = df["internet_traffic_mb"]
        elif "data_traffic_gb" in df.columns:
            df["total_traffic_mb"] = (
                safe_numeric(df["data_traffic_gb"]) * 1000
            )

    if "internet_traffic_mb" not in df.columns:
        if "total_traffic_mb" in df.columns:
            df["internet_traffic_mb"] = df["total_traffic_mb"]

    # Derive calendar fields from timestamp
    if "timestamp" in df.columns:
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.day_name()
        df["date_only"] = df["timestamp"].dt.date

    # Create a tower label if none exists.
    # This is a placeholder category, not a real measured tower ID.
    if "tower_id" not in df.columns:
        df["tower_id"] = "All Records"

    df["tower_id"] = df["tower_id"].fillna("Unknown").astype(str)

    return df


def get_first_numeric(df, candidates):
    for col in candidates:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            return col
    return None


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <div class="hero-kicker">NETWORK OPERATIONS • DATA INTELLIGENCE</div>
    <div class="hero-title">📡 FLOWBIX TELECOM</div>
    <div class="hero-subtitle">
        Interactive Regional Traffic Monitoring &nbsp; | &nbsp;
        SMS and Calls &nbsp; | &nbsp;
        Machine Learning Predictions &nbsp; | &nbsp;
        Network Analytics
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR: UPLOAD + CONTROLS
# ============================================================

st.sidebar.markdown("## 🎛️ Dashboard Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload your telecom CSV dataset",
    type=["csv", "txt"],
    help="Upload telecom_traffic.csv or a CSV file with equivalent columns."
)

st.sidebar.caption(
    "Your dataset is processed locally by this Streamlit application."
)

# Optional demo data, clearly identified as synthetic
use_demo = st.sidebar.checkbox(
    "Use sample demo data if no file is uploaded",
    value=True
)


def create_demo_data():
    """Create clearly labelled synthetic data for testing the dashboard."""
    rng = np.random.default_rng(42)

    periods = 24 * 30
    times = pd.date_range(
        end=pd.Timestamp.now().floor("h"),
        periods=periods,
        freq="h"
    )

    records = []

    for tower_num in range(1, 7):
        tower = f"DEMO_TOWER_{tower_num:02d}"

        for ts in times:
            hour = ts.hour
            rush = 1.5 if hour in [8, 9, 10, 18, 19, 20, 21] else 1.0
            base = (500 + tower_num * 110) * rush

            traffic = max(30, base + rng.normal(0, 100))
            sms = max(0, int(80 * rush + rng.normal(0, 15)))
            calls = max(0, int(60 * rush + rng.normal(0, 12)))

            records.append({
                "timestamp": ts,
                "tower_id": tower,
                "sms_count": sms,
                "call_count": calls,
                "internet_traffic_mb": traffic,
                "total_traffic_mb": traffic,
                "active_users": max(10, int(traffic / 5)),
                "signal_dbm": rng.normal(-68, 7),
                "tower_capacity_gb": 25,
                "area_type": (
                    "Urban" if tower_num <= 2
                    else "Suburban" if tower_num <= 4
                    else "Rural"
                )
            })

    demo = pd.DataFrame(records)
    demo["hour"] = demo["timestamp"].dt.hour
    demo["day_of_week"] = demo["timestamp"].dt.day_name()
    demo["date_only"] = demo["timestamp"].dt.date

    demo["next_hour_traffic_mb"] = (
        demo.groupby("tower_id")["total_traffic_mb"].shift(-1)
    )

    return demo.dropna(subset=["total_traffic_mb"])


df = None
dataset_label = ""

if uploaded_file is not None:
    try:
        # Read the uploaded file; support regular CSV uploads.
        raw = pd.read_csv(uploaded_file, low_memory=False)
        df = prepare_dataset(raw)
        dataset_label = uploaded_file.name
        st.sidebar.success(
            f"Loaded {len(df):,} rows and {len(df.columns)} columns."
        )
    except Exception as exc:
        st.error(f"Could not read this file as a CSV: {exc}")
        st.info(
            "Please upload a comma-separated CSV file. "
            "A very large raw telecom text file may need preprocessing first."
        )
        st.stop()

elif use_demo:
    df = create_demo_data()
    dataset_label = "Synthetic demo dataset"
    st.info(
        "You are viewing SYNTHETIC DEMO DATA. Upload your actual CSV "
        "from the sidebar to analyse your project dataset."
    )
else:
    st.info(
        "👈 Upload your telecom CSV from the sidebar to start the dashboard."
    )
    st.stop()


if df is None or df.empty:
    st.error("The dataset is empty after loading.")
    st.stop()


# ============================================================
# CHECK REQUIRED DATA
# ============================================================

traffic_col = get_first_numeric(
    df,
    ["total_traffic_mb", "internet_traffic_mb"]
)

if traffic_col is None:
    st.error(
        "I could not identify a numeric traffic column. "
        "Your dataset should contain a column such as "
        "'total_traffic_mb', 'internet_traffic_mb', or 'traffic_mb'."
    )

    st.write("Columns found in the uploaded dataset:")
    st.write(list(df.columns))
    st.stop()

if "total_traffic_mb" not in df.columns:
    df["total_traffic_mb"] = df[traffic_col]

if "internet_traffic_mb" not in df.columns:
    df["internet_traffic_mb"] = df[traffic_col]


# ============================================================
# FILTERS
# ============================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔎 Filters")

tower_options = sorted(df["tower_id"].dropna().astype(str).unique().tolist())

selected_towers = st.sidebar.multiselect(
    "Select towers / regions",
    options=tower_options,
    default=tower_options[:min(6, len(tower_options))]
)

if not selected_towers:
    selected_towers = tower_options

filtered = df[df["tower_id"].astype(str).isin(selected_towers)].copy()

# Date range filter if valid timestamps exist
has_timestamps = (
    "timestamp" in filtered.columns
    and filtered["timestamp"].notna().any()
)

if has_timestamps:
    valid_times = filtered["timestamp"].dropna()
    min_date = valid_times.min().date()
    max_date = valid_times.max().date()

    date_range = st.sidebar.date_input(
        "Select date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range

        filtered = filtered[
            filtered["timestamp"].dt.date.between(start_date, end_date)
        ]

st.sidebar.markdown("---")
st.sidebar.caption(f"Dataset: {dataset_label}")
st.sidebar.caption(f"Filtered rows: {len(filtered):,}")

if filtered.empty:
    st.warning("No rows match your current filters. Select more towers or dates.")
    st.stop()


# ============================================================
# KPI CARDS
# ============================================================

st.markdown('<div class="section-title">⚡ Network Overview</div>',
            unsafe_allow_html=True)

total_traffic = filtered["total_traffic_mb"].sum()
total_sms = (
    filtered["sms_count"].sum()
    if "sms_count" in filtered.columns else None
)
total_calls = (
    filtered["call_count"].sum()
    if "call_count" in filtered.columns else None
)

k1, k2, k3, k4 = st.columns(4)

with k1:
    metric_card("DATA RECORDS", format_number(len(filtered)), "Filtered dataset")

with k2:
    metric_card("TOTAL DATA TRAFFIC", f"{total_traffic / 1000:,.2f} GB",
                "MB converted to decimal GB")

with k3:
    metric_card(
        "TOTAL SMS",
        format_number(total_sms) if total_sms is not None else "Unavailable",
        "Only shown if SMS data exists"
    )

with k4:
    metric_card(
        "CALLS / MINUTES",
        format_number(total_calls) if total_calls is not None else "Unavailable",
        "Uses the uploaded call field"
    )

st.markdown("")


# ============================================================
# TABS
# ============================================================

tab_overview, tab_sms, tab_ml, tab_towers, tab_dataset = st.tabs([
    "📈 Traffic Overview",
    "📞 SMS & Calls",
    "🤖 ML Prediction",
    "🗼 Tower Analytics",
    "📋 Dataset Explorer"
])


# ============================================================
# TAB 1: TRAFFIC OVERVIEW
# ============================================================

with tab_overview:
    st.markdown("### Regional Traffic Intelligence")

    left, right = st.columns([1.6, 1])

    with left:
        if "timestamp" in filtered.columns and filtered["timestamp"].notna().any():
            time_data = filtered.dropna(subset=["timestamp"]).copy()

            time_group = (
                time_data.groupby(
                    pd.Grouper(key="timestamp", freq="D")
                )["total_traffic_mb"]
                .sum()
                .reset_index()
            )

            fig = px.area(
                time_group,
                x="timestamp",
                y="total_traffic_mb",
                title="📊 Total Data Traffic Over Time",
                markers=True
            )

            fig.update_traces(
                line=dict(color="#28e0c2", width=3),
                fillcolor="rgba(40,224,194,0.16)"
            )

            st.plotly_chart(
                make_chart(fig, 420),
                use_container_width=True
            )
        else:
            st.info("No usable timestamp column was found.")

    with right:
        tower_traffic = (
            filtered.groupby("tower_id")["total_traffic_mb"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        fig = px.bar(
            tower_traffic,
            x="total_traffic_mb",
            y="tower_id",
            orientation="h",
            title="🗼 Traffic by Tower",
            color="total_traffic_mb",
            color_continuous_scale=["#176b87", "#2be4c4"]
        )

        fig.update_layout(yaxis={"categoryorder": "total ascending"})

        st.plotly_chart(
            make_chart(fig, 420),
            use_container_width=True
        )

    left, right = st.columns(2)

    with left:
        if "hour" in filtered.columns:
            hourly = filtered.copy()
            hourly["hour"] = safe_numeric(hourly["hour"])
            hourly = hourly.dropna(subset=["hour"])

            if not hourly.empty:
                hourly_group = (
                    hourly.groupby("hour")["total_traffic_mb"]
                    .mean()
                    .reset_index()
                    .sort_values("hour")
                )

                fig = px.line(
                    hourly_group,
                    x="hour",
                    y="total_traffic_mb",
                    markers=True,
                    title="🕒 Average Traffic by Hour"
                )

                fig.update_traces(
                    line=dict(color="#4da3ff", width=3),
                    marker=dict(size=7)
                )

                fig.update_xaxes(dtick=2, title="Hour of Day")
                fig.update_yaxes(title="Average Traffic (MB)")

                st.plotly_chart(
                    make_chart(fig, 350),
                    use_container_width=True
                )

    with right:
        if "area_type" in filtered.columns:
            area_group = (
                filtered.groupby("area_type")["total_traffic_mb"]
                .sum()
                .reset_index()
            )

            fig = px.pie(
                area_group,
                names="area_type",
                values="total_traffic_mb",
                title="🌐 Traffic Share by Area Type",
                hole=0.62,
                color_discrete_sequence=[
                    "#22d3b6", "#3988ff", "#a16eff",
                    "#ffb454", "#ff6680"
                ]
            )

            fig.update_traces(
                textposition="inside",
                textinfo="percent+label"
            )

            st.plotly_chart(
                make_chart(fig, 350),
                use_container_width=True
            )
        else:
            st.info("No area/region category column is available.")

    st.markdown("### 📌 Traffic Summary by Tower")

    summary = (
        filtered.groupby("tower_id")
        .agg(
            Records=("total_traffic_mb", "size"),
            Total_Traffic_MB=("total_traffic_mb", "sum"),
            Average_Traffic_MB=("total_traffic_mb", "mean"),
            Maximum_Traffic_MB=("total_traffic_mb", "max")
        )
        .reset_index()
        .sort_values("Total_Traffic_MB", ascending=False)
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TAB 2: SMS & CALLS
# ============================================================

with tab_sms:
    st.markdown("### 📞 Communication Activity")

    available_sms = "sms_count" in filtered.columns
    available_calls = "call_count" in filtered.columns

    if not available_sms and not available_calls:
        st.warning(
            "The uploaded dataset does not contain identifiable SMS or call "
            "columns. These charts will appear when you upload data with "
            "SMS and call measurements."
        )

    else:
        if available_sms and available_calls:
            a, b = st.columns(2)

            with a:
                sms_by_tower = (
                    filtered.groupby("tower_id")["sms_count"]
                    .sum()
                    .reset_index()
                )

                fig = px.bar(
                    sms_by_tower,
                    x="tower_id",
                    y="sms_count",
                    title="💬 SMS Activity by Tower",
                    color="sms_count",
                    color_continuous_scale=["#176b87", "#2be4c4"]
                )

                st.plotly_chart(
                    make_chart(fig, 370),
                    use_container_width=True
                )

            with b:
                calls_by_tower = (
                    filtered.groupby("tower_id")["call_count"]
                    .sum()
                    .reset_index()
                )

                fig = px.bar(
                    calls_by_tower,
                    x="tower_id",
                    y="call_count",
                    title="☎️ Call Activity by Tower",
                    color="call_count",
                    color_continuous_scale=["#4338ca", "#61a5ff"]
                )

                st.plotly_chart(
                    make_chart(fig, 370),
                    use_container_width=True
                )

        if available_sms and available_calls:
            st.markdown("#### SMS vs Calls")

            comparison = filtered[
                ["tower_id", "sms_count", "call_count"]
            ].copy()

            comparison = comparison.groupby("tower_id").sum().reset_index()

            comparison_long = comparison.melt(
                id_vars="tower_id",
                value_vars=["sms_count", "call_count"],
                var_name="Activity",
                value_name="Count"
            )

            fig = px.bar(
                comparison_long,
                x="tower_id",
                y="Count",
                color="Activity",
                barmode="group",
                title="SMS and Call Comparison by Tower",
                color_discrete_sequence=["#20d6b8", "#638cff"]
            )

            st.plotly_chart(
                make_chart(fig, 400),
                use_container_width=True
            )

        if available_sms and "timestamp" in filtered.columns:
            sms_time = filtered.dropna(subset=["timestamp"]).copy()

            if not sms_time.empty:
                sms_time = (
                    sms_time.groupby(
                        pd.Grouper(key="timestamp", freq="D")
                    )["sms_count"]
                    .sum()
                    .reset_index()
                )

                fig = px.line(
                    sms_time,
                    x="timestamp",
                    y="sms_count",
                    markers=True,
                    title="Daily SMS Activity"
                )

                fig.update_traces(line=dict(color="#26dfbc", width=3))

                st.plotly_chart(
                    make_chart(fig, 330),
                    use_container_width=True
                )

        # Incoming/outgoing analysis only when separate columns actually exist
        incoming_sms_col = find_column(
            filtered, ["sms_in", "sms_in_count", "incoming_sms"]
        )
        outgoing_sms_col = find_column(
            filtered, ["sms_out", "sms_out_count", "outgoing_sms"]
        )
        incoming_call_col = find_column(
            filtered, ["call_in", "calls_in", "incoming_calls"]
        )
        outgoing_call_col = find_column(
            filtered, ["call_out", "calls_out", "outgoing_calls"]
        )

        st.markdown("#### Incoming / Outgoing Data Availability")

        io_cols = st.columns(4)

        io_fields = [
            ("Incoming SMS", incoming_sms_col),
            ("Outgoing SMS", outgoing_sms_col),
            ("Incoming Calls", incoming_call_col),
            ("Outgoing Calls", outgoing_call_col)
        ]

        for col, (label, field) in zip(io_cols, io_fields):
            with col:
                if field:
                    metric_card(
                        label,
                        format_number(filtered[field].sum()),
                        f"Source column: {field}"
                    )
                else:
                    metric_card(
                        label,
                        "Not provided",
                        "Not present in uploaded dataset"
                    )


# ============================================================
# TAB 3: MACHINE LEARNING
# ============================================================

with tab_ml:
    st.markdown("### 🤖 Next-Hour Traffic Prediction")

    st.write(
        "This section trains a Random Forest model on the uploaded dataset "
        "when a suitable target or time sequence is available."
    )

    ml_df = filtered.copy()

    # Select a traffic target
    target_col = None

    if (
        "next_hour_traffic_mb" in ml_df.columns
        and ml_df["next_hour_traffic_mb"].notna().sum() >= 20
    ):
        target_col = "next_hour_traffic_mb"

    elif (
        "next_hour_traffic_gb" in ml_df.columns
        and ml_df["next_hour_traffic_gb"].notna().sum() >= 20
    ):
        target_col = "next_hour_traffic_gb"

    elif "timestamp" in ml_df.columns:
        # Derive the next record's traffic for each tower, using time order.
        ml_df = ml_df.sort_values(["tower_id", "timestamp"]).copy()

        ml_df["_derived_next_traffic"] = (
            ml_df.groupby("tower_id")["total_traffic_mb"].shift(-1)
        )

        target_col = "_derived_next_traffic"

    if target_col is None:
        st.warning(
            "There are not enough records to build a next-hour prediction "
            "target. Upload more traffic records or include a "
            "'next_hour_traffic_mb' column."
        )

    else:
        # Keep a separate display label; avoid confusing GB with MB
        target_display = target_col

        if target_col == "next_hour_traffic_gb":
            ml_df["_ml_target_mb"] = (
                safe_numeric(ml_df[target_col]) * 1000
            )
            target_display = "_ml_target_mb"

        elif target_col == "_derived_next_traffic":
            target_display = "_derived_next_traffic"

        else:
            target_display = target_col

        features = []

        numeric_candidates = [
            "total_traffic_mb",
            "internet_traffic_mb",
            "sms_count",
            "call_count",
            "active_users",
            "signal_dbm",
            "tower_capacity_gb",
            "hour"
        ]

        for col in numeric_candidates:
            if (
                col in ml_df.columns
                and col != target_display
                and ml_df[col].notna().sum() > 0
            ):
                features.append(col)

        if len(features) == 0:
            st.warning(
                "The dataset does not contain enough numeric input features "
                "for machine learning."
            )

        else:
            model_data = ml_df[features + [target_display]].copy()

            for col in features + [target_display]:
                model_data[col] = safe_numeric(model_data[col])

            model_data = model_data.replace(
                [np.inf, -np.inf], np.nan
            ).dropna()

            if len(model_data) < 20:
                st.warning(
                    f"Only {len(model_data)} usable rows remain. "
                    "At least 20 are needed for this dashboard's model."
                )

            else:
                X = model_data[features]
                y = model_data[target_display]

                if y.nunique() < 2:
                    st.warning(
                        "The target has insufficient variation to train "
                        "a useful regression model."
                    )

                else:
                    test_fraction = 0.20

                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=test_fraction,
                        random_state=42
                    )

                    model_choice = st.selectbox(
                        "Choose prediction model",
                        ["Random Forest Regressor", "Linear Regression"]
                    )

                    if model_choice == "Random Forest Regressor":
                        model = RandomForestRegressor(
                            n_estimators=150,
                            max_depth=12,
                            min_samples_leaf=2,
                            random_state=42,
                            n_jobs=-1
                        )
                    else:
                        model = LinearRegression()

                    if st.button("🚀 Train Model & Show Results"):
                        with st.spinner("Training the model..."):
                            model.fit(X_train, y_train)
                            predictions = model.predict(X_test)

                            mae = mean_absolute_error(y_test, predictions)
                            rmse = np.sqrt(
                                mean_squared_error(y_test, predictions)
                            )

                            r2 = r2_score(y_test, predictions)

                            c1, c2, c3 = st.columns(3)

                            with c1:
                                metric_card(
                                    "MAE",
                                    f"{mae:,.2f}",
                                    "Lower is better"
                                )

                            with c2:
                                metric_card(
                                    "RMSE",
                                    f"{rmse:,.2f}",
                                    "Lower is better"
                                )

                            with c3:
                                metric_card(
                                    "R² SCORE",
                                    f"{r2:.3f}",
                                    "Closer to 1 is generally better"
                                )

                            result_df = pd.DataFrame({
                                "Actual": y_test.to_numpy(),
                                "Predicted": predictions
                            })

                            fig = px.scatter(
                                result_df,
                                x="Actual",
                                y="Predicted",
                                title="Actual vs Predicted Traffic",
                                opacity=0.75
                            )

                            lower = min(
                                result_df["Actual"].min(),
                                result_df["Predicted"].min()
                            )
                            upper = max(
                                result_df["Actual"].max(),
                                result_df["Predicted"].max()
                            )

                            fig.add_trace(
                                go.Scatter(
                                    x=[lower, upper],
                                    y=[lower, upper],
                                    mode="lines",
                                    name="Perfect prediction",
                                    line=dict(
                                        color="#ff6b86",
                                        dash="dash"
                                    )
                                )
                            )

                            st.plotly_chart(
                                make_chart(fig, 420),
                                use_container_width=True
                            )

                            st.markdown("#### Feature Importance")

                            if hasattr(model, "feature_importances_"):
                                importance = pd.DataFrame({
                                    "Feature": features,
                                    "Importance": model.feature_importances_
                                }).sort_values(
                                    "Importance", ascending=False
                                )

                                fig = px.bar(
                                    importance,
                                    x="Importance",
                                    y="Feature",
                                    orientation="h",
                                    title="Random Forest Feature Importance",
                                    color="Importance",
                                    color_continuous_scale=[
                                        "#176b87", "#2be4c4"
                                    ]
                                )

                                fig.update_layout(
                                    yaxis={"categoryorder": "total ascending"}
                                )

                                st.plotly_chart(
                                    make_chart(fig, 350),
                                    use_container_width=True
                                )

                            else:
                                st.info(
                                    "Linear Regression does not provide "
                                    "tree-based feature importance in this app."
                                )

                            st.markdown("#### Test Set Predictions")

                            st.dataframe(
                                result_df.head(100),
                                use_container_width=True,
                                hide_index=True
                            )

                            st.download_button(
                                "⬇️ Download Predictions",
                                data=result_df.to_csv(index=False).encode("utf-8"),
                                file_name="telecom_traffic_predictions.csv",
                                mime="text/csv"
                            )

                            st.caption(
                                "Evaluation is based on a random train/test "
                                "split. For rigorous forecasting, use a "
                                "chronological split and avoid future-data "
                                "leakage."
                            )

                    else:
                        st.info(
                            f"Ready to train on {len(model_data):,} usable "
                            f"rows with {len(features)} numeric features."
                        )


# ============================================================
# TAB 4: TOWER ANALYTICS
# ============================================================

with tab_towers:
    st.markdown("### 🗼 Tower Performance & Network Indicators")

    tower_summary = (
        filtered.groupby("tower_id")
        .agg(
            Record_Count=("total_traffic_mb", "size"),
            Total_Traffic_MB=("total_traffic_mb", "sum"),
            Average_Traffic_MB=("total_traffic_mb", "mean"),
            Peak_Traffic_MB=("total_traffic_mb", "max")
        )
        .reset_index()
    )

    if "active_users" in filtered.columns:
        users = (
            filtered.groupby("tower_id")["active_users"]
            .mean()
            .reset_index(name="Average_Active_Users")
        )

        tower_summary = tower_summary.merge(users, on="tower_id", how="left")

    if "signal_dbm" in filtered.columns:
        signals = (
            filtered.groupby("tower_id")["signal_dbm"]
            .mean()
            .reset_index(name="Average_Signal_dBm")
        )

        tower_summary = tower_summary.merge(signals, on="tower_id", how="left")

    left, right = st.columns(2)

    with left:
        fig = px.bar(
            tower_summary.sort_values("Total_Traffic_MB"),
            x="Total_Traffic_MB",
            y="tower_id",
            orientation="h",
            title="Traffic Ranking by Tower",
            color="Total_Traffic_MB",
            color_continuous_scale=["#175b80", "#28e0c2"]
        )

        st.plotly_chart(
            make_chart(fig, 420),
            use_container_width=True
        )

    with right:
        if "active_users" in filtered.columns:
            user_data = (
                filtered.groupby("tower_id")["active_users"]
                .mean()
                .reset_index()
            )

            fig = px.bar(
                user_data,
                x="tower_id",
                y="active_users",
                title="Average Active Users by Tower",
                color="active_users",
                color_continuous_scale=["#4338ca", "#60a5fa"]
            )

            st.plotly_chart(
                make_chart(fig, 420),
                use_container_width=True
            )
        else:
            st.info("Active-user data is not present in the uploaded dataset.")

    st.markdown("#### Tower Data Table")

    st.dataframe(
        tower_summary.sort_values("Total_Traffic_MB", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    # Capacity utilisation only if both traffic and capacity are available
    if "tower_capacity_gb" in filtered.columns:
        st.markdown("#### Capacity Utilisation")

        capacity_data = (
            filtered.groupby("tower_id")
            .agg(
                Average_Traffic_MB=("total_traffic_mb", "mean"),
                Capacity_GB=("tower_capacity_gb", "mean")
            )
            .reset_index()
        )

        capacity_data["Capacity_MB"] = capacity_data["Capacity_GB"] * 1000

        capacity_data["Utilisation_Percent"] = np.where(
            capacity_data["Capacity_MB"] > 0,
            capacity_data["Average_Traffic_MB"]
            / capacity_data["Capacity_MB"] * 100,
            np.nan
        )

        fig = px.bar(
            capacity_data,
            x="tower_id",
            y="Utilisation_Percent",
            title="Estimated Average Capacity Utilisation",
            color="Utilisation_Percent",
            color_continuous_scale=["#2be4c4", "#ffbf69", "#ff5d73"]
        )

        fig.add_hline(
            y=85,
            line_dash="dash",
            line_color="#ff667e",
            annotation_text="85% reference threshold"
        )

        st.plotly_chart(
            make_chart(fig, 370),
            use_container_width=True
        )

        st.caption(
            "This calculation compares average traffic per record with "
            "the capacity field. Confirm that the time units and capacity "
            "definitions are compatible before interpreting it operationally."
        )
    else:
        st.info(
            "Capacity utilisation is unavailable because the uploaded dataset "
            "does not contain a recognised tower capacity column."
        )


# ============================================================
# TAB 5: DATASET EXPLORER
# ============================================================

with tab_dataset:
    st.markdown("### 📋 Dataset Explorer")

    a, b, c = st.columns(3)

    with a:
        metric_card("ROWS", format_number(len(filtered)), "Filtered records")

    with b:
        metric_card("COLUMNS", format_number(len(filtered.columns)), "Available fields")

    with c:
        missing_total = int(filtered.isna().sum().sum())
        metric_card("MISSING CELLS", format_number(missing_total),
                    "Across filtered dataset")

    st.markdown("#### Preview")

    st.dataframe(
        filtered.head(500),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("#### Column Information")

    info_df = pd.DataFrame({
        "Column": filtered.columns,
        "Data Type": [str(filtered[c].dtype) for c in filtered.columns],
        "Missing Values": [
            int(filtered[c].isna().sum()) for c in filtered.columns
        ],
        "Unique Values": [
            int(filtered[c].nunique(dropna=True)) for c in filtered.columns
        ]
    })

    st.dataframe(
        info_df,
        use_container_width=True,
        hide_index=True
    )

    left, right = st.columns(2)

    with left:
        numeric_cols = filtered.select_dtypes(include=np.number).columns.tolist()

        if numeric_cols:
            selected_hist = st.selectbox(
                "Choose a numeric column for histogram",
                numeric_cols
            )

            fig = px.histogram(
                filtered,
                x=selected_hist,
                nbins=35,
                title=f"Distribution of {selected_hist}",
                color_discrete_sequence=["#27d9bb"]
            )

            st.plotly_chart(
                make_chart(fig, 350),
                use_container_width=True
            )

    with right:
        if len(numeric_cols) >= 2:
            corr = filtered[numeric_cols].corr(numeric_only=True)

            fig = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Numeric Feature Correlation Heatmap"
            )

            st.plotly_chart(
                make_chart(fig, 350),
                use_container_width=True
            )

    st.markdown("#### Export Filtered Dataset")

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Filtered Dataset (CSV)",
        data=csv_bytes,
        file_name="filtered_telecom_traffic.csv",
        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown("""
<div style="text-align:center; color:#7998aa; padding:15px;">
    <b style="color:#37e6cc;">FLOWBIX TELECOM</b><br>
    Regional Telecom Traffic Analysis • Interactive Analytics • ML Prediction<br>
    <small>Academic Machine Learning Project</small>
</div>
""", unsafe_allow_html=True)
