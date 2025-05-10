import streamlit as st
import pandas as pd
import zipfile
import io
import requests
import altair as alt

st.set_page_config(page_title="Citi Bike Dashboard", layout="wide")

# --- Title and Info ---
st.markdown("# 🚲 NYC Citi Bike Data Analysis and Prediction")
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

    # --- Sidebar Filters ---
    with st.sidebar.expander("🔍 Advanced Filters", expanded=False):
        enable_ride_filter = st.checkbox("Filter by Rideable Type", value=False)
        if enable_ride_filter:
            ride_types = st.multiselect(
                "Select Ride Types", 
                options=df["rideable_type"].dropna().unique(),
                default=list(df["rideable_type"].dropna().unique())
            )
        else:
            ride_types = list(df["rideable_type"].dropna().unique())

        enable_station_filter = st.checkbox("Filter by Start Station", value=False)
        if enable_station_filter:
            stations = st.multiselect(
                "Select Start Stations", 
                options=sorted(df["start_station_name"].dropna().unique()),
                default=sorted(df["start_station_name"].dropna().unique())
            )
        else:
            stations = list(df["start_station_name"].dropna().unique())

    # --- Apply Filters ---
    filtered_df = df[
        df["rideable_type"].isin(ride_types) & 
        df["start_station_name"].isin(stations)
    ]

    # --- Insight Summary ---
    total_trips = len(filtered_df)
    top_station = filtered_df["start_station_name"].value_counts().idxmax() if not filtered_df.empty else "N/A"
    st.markdown(
        f"📊 Based on your selection, a total of **{total_trips:,} trips** were recorded. "
        f"The most frequent start station was **{top_station}**."
    )

    # --- Metrics Section ---
    st.subheader("📊 Summary Metrics")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Trips", f"{total_trips:,}")

    with col2:
        if not filtered_df.empty:
            peak_hour = filtered_df["hour"].value_counts().idxmax()
            st.metric("Most Frequent Hour", f"{peak_hour}:00")
        else:
            st.metric("Most Frequent Hour", "N/A")

    with col3:
        st.metric("Top Start Station", top_station if top_station != "N/A" else "N/A")


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

    # --- Enhancement: Map View of Start Stations ---
    st.subheader("🗺️ Map of Start Locations")
    if "start_lat" in filtered_df.columns and "start_lng" in filtered_df.columns:
        st.map(filtered_df.rename(columns={"start_lat": "lat", "start_lng": "lon"}))
    else:
        st.info("ℹ️ Location data (latitude/longitude) not available in this dataset.")

    # --- Enhancement: Download CSV ---
    st.subheader("📥 Export Filtered Data")
    st.download_button("Download CSV", filtered_df.to_csv(index=False), file_name="filtered_citibike_data.csv", mime="text/csv")

    # --- Optional Raw Data ---
    with st.expander("🔽 Show Raw Data"):
        st.dataframe(filtered_df.head(100))

except Exception as e:
    st.error(f"❌ Error loading data: {e}")
