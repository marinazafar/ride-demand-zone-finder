from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Ride Demand Zone Finder",
    page_icon="🚕",
    layout="wide"
)

st.title("Ride Demand Zone Finder")
st.write(
    "Identify potential taxi pickup hotspots in New York City "
    "using HDBSCAN clustering."
)


data_file = Path(__file__).parent / "yellow_tripdata_sample.csv"


if not data_file.exists():
    st.error(
        "Dataset not found. Place yellow_tripdata_2015-01.csv "
        "in the same folder as app.py."
    )
    st.stop()

@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path, nrows=30000)

    data = data.dropna(
        subset=["pickup_latitude", "pickup_longitude"]
    )

    data = data[
        data["pickup_latitude"].between(40.4, 41.0)
        & data["pickup_longitude"].between(-74.3, -73.6)
    ].copy()

    data = data.drop_duplicates()

    return data

df_app = load_data(str(data_file))

st.sidebar.header("Clustering Settings")

min_cluster_size = st.sidebar.slider(
    "Minimum cluster size",
    min_value=20,
    max_value=300,
    value=100,
    step=10
)

min_samples = st.sidebar.slider(
    "Minimum samples",
    min_value=5,
    max_value=30,
    value=10,
    step=5
)

st.write("### Dataset Overview")
st.metric("Pickup records after cleaning", f"{len(df_app):,}")

if st.button("Find Demand Zones"):
    with st.spinner("Running HDBSCAN clustering..."):
        coordinates_app = df_app[
            ["pickup_latitude", "pickup_longitude"]
        ]

        model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples
        )

        labels = model.fit_predict(coordinates_app)
        df_app["cluster"] = labels

        cluster_count = len(set(labels) - {-1})
        noise_count = int((labels == -1).sum())

        col1, col2, col3 = st.columns(3)

        col1.metric("Demand zones found", cluster_count)
        col2.metric("Noise points", f"{noise_count:,}")
        col3.metric("Records analyzed", f"{len(df_app):,}")

        st.write("### Demand Zone Map")

        fig, ax = plt.subplots(figsize=(12, 7))

        noise = df_app[df_app["cluster"] == -1]
        clustered = df_app[df_app["cluster"] != -1]

        ax.scatter(
            noise["pickup_longitude"],
            noise["pickup_latitude"],
            s=3,
            alpha=0.2,
            label="Noise"
        )

        if not clustered.empty:
            ax.scatter(
                clustered["pickup_longitude"],
                clustered["pickup_latitude"],
                c=clustered["cluster"],
                cmap="tab20",
                s=5,
                alpha=0.65
            )

        ax.set_xlabel("Pickup Longitude")
        ax.set_ylabel("Pickup Latitude")
        ax.set_title("Taxi Pickup Clusters")
        ax.legend()
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        hotspots = (
            df_app[df_app["cluster"] != -1]
            .groupby("cluster")
            .agg(
                pickup_count=("cluster", "size"),
                center_latitude=("pickup_latitude", "mean"),
                center_longitude=("pickup_longitude", "mean")
            )
            .sort_values("pickup_count", ascending=False)
            .reset_index()
        )

        st.write("### Largest Demand Zones")
        st.dataframe(hotspots.head(10), use_container_width=True)

        st.download_button(
            "Download clustering results",
            data=df_app.to_csv(index=False).encode("utf-8"),
            file_name="ride_demand_clusters.csv",
            mime="text/csv"
        )
