import streamlit as st
import pandas as pd
import zipfile
import io
import requests
import altair as alt

st.set_page_config(page_title="Citi Bike Dashboard", layout="wide")

# --- Title and Info ---
st.markdown("# 🚲 NYC Citi Bike Trip Viewer")
st.markdown("A dynamic dashboard to explore NYC Citi Bike trip data.")
st.markdown("---")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("📂 Select a Dataset")
    year = st.selectbox("Year", [2023])
    month = st.selectbox("Month", list(range(1, 13)))

# --- Load Data ---
url = f"https://s3.amazonaws.com/tripdata/JC-{year}{month:02}-citibike-tripdata.csv.zip"
st.markdown(f"🔗 **Data Source**: [{url}]({url})")

@st.cache_data
def load_data(zip_url):
    r = requests.get(zip_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    csv_filename = z.namelist()[0]
    with z.open(csv_filename) as f:
        df = pd.read_csv(f)
    return df

try:
    df = load_data(url)
    st.success("✅ Data loaded successfully!")

    # --- Datetime Handling ---
    datetime_col = "started_at" if "started_at" in df.columns else "starttime"
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df.dropna(subset=[datetime_col], inplace=True)
    df["hour"] = df[datetime_col].dt.hour
    df["day"] = df[datetime_col].dt.day_name()

    # --- Sidebar Filters in Expander ---
    with st.sidebar.expander("🔍 Advanced Filters", expanded=False):
        ride_types = st.multiselect(
            "Rideable Type", 
            options=df["rideable_type"].dropna().unique(), 
            default=list(df["rideable_type"].dropna().unique())
        )

        stations = st.multiselect(
            "Start Station", 
            options=sorted(df["start_station_name"].dropna().unique()), 
            default=list(df["start_station_name"].dropna().unique())
        )

    # --- Apply Filters ---
    filtered_df = df[
        df["rideable_type"].isin(ride_types) & 
        df["start_station_name"].isin(stations)
    ]

    # --- Visualization 1: Trips by Hour ---
    st.subheader("⏱ Trips by Hour of Day")
    hour_chart = filtered_df.groupby("hour").size().reset_index(name="count")
    st.line_chart(hour_chart.set_index("hour"))

    # --- Visualization 2: Top Start Stations ---
    st.subheader("🏆 Top 10 Start Stations")
    top_stations = filtered_df["start_station_name"].value_counts().head(10).reset_index()
    top_stations.columns = ["Station", "Trip Count"]
    st.bar_chart(top_stations.set_index("Station"))

    # --- Visualization 3: Heatmap (Day vs Hour) ---
    st.subheader("🔥 Trip Frequency Heatmap (Day vs Hour)")
    heatmap_data = filtered_df.groupby(["day", "hour"]).size().reset_index(name="count")
    heatmap = alt.Chart(heatmap_data).mark_rect().encode(
        x=alt.X("hour:O", title="Hour of Day"),
        y=alt.Y("day:O", sort=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]),
        color=alt.Color("count:Q", scale=alt.Scale(scheme='reds')),
        tooltip=["day", "hour", "count"]
    ).properties(height=300)
    st.altair_chart(heatmap, use_container_width=True)

    # --- Optional Raw Data ---
    with st.expander("🔽 Show Raw Data"):
        st.dataframe(filtered_df.head(100))

except Exception as e:
    st.error(f"❌ Error loading data: {e}")
