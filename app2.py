import streamlit as st
from io import BytesIO
import geopandas as gpd
import pandas as pd
import folium
from folium import plugins
from streamlit_folium import folium_static
import os
import numpy as np

# pair programming with GPT4
# https://chat.openai.com/c/33d6abdb-0490-4be9-b605-772e357a1489

# Generate random sample data
num_points = 50
lat_min, lat_max = 34.047, 34.056  # latitude extent
long_min, long_max = -117.82, -117.80  # longitude extent

# Initialize a random seed in session state if it doesn't already exist
if "random_seed" not in st.session_state:
    st.session_state["random_seed"] = np.random.randint(0, 100)

# Use the stored random seed for reproducible randomness
np.random.seed(st.session_state["random_seed"])

sample_data = {
    "INDEX": [i for i in range(1, num_points + 1)],
    "LAT": np.random.uniform(lat_min, lat_max, num_points),
    "LONG": np.random.uniform(long_min, long_max, num_points),
}


def handle_file_upload():
    with st.sidebar:
        # st.caption("Please upload a .xlsx file to get started. After uploading, select the appropriate columns and set the desired distance threshold.")
        msg = "Choose a .xlsx file to get started. Or play around with the sample data (move the distance threshold slider!)."
        uploaded_file = st.file_uploader(msg, type="xlsx")

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.DataFrame(sample_data)  # Use sample data if no file is uploaded
    with st.expander("About the input data"):
        f"{len(df)} rows found."
        df
    return df, uploaded_file


def find_lat_lon_columns(df):
    lat_col, lon_col = None, None

    # Iterate over all columns in the DataFrame
    for col in df.columns:
        col_lower = (col.lower())
        # Check for both full words and abbreviations
        if ("latitude" in col_lower or "lat" in col_lower) and not lat_col:
            lat_col = col  # Assign the first matching latitude column
        elif ("longitude" in col_lower or "lon" in col_lower) and not lon_col:
            lon_col = col  # Assign the first matching longitude column

        # If both columns are found, no need to continue searching
        if lat_col and lon_col:
            break

    return lat_col, lon_col


# Helper function to convert DataFrame to Excel in memory
def convert_df_to_excel(_df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        _df.to_excel(writer, index=False, sheet_name="Sheet1")
    processed_data = output.getvalue()
    return processed_data


# Convert feet to meters
def feet_to_meters(feet):
    return feet * 0.3048


def generate_buffers(gdf, distance):
    # Generate a buffer polygon around each point with the specified distance
    gdf["buffer"] = gdf.geometry.buffer(distance)
    return gdf


def get_bounds(gdf_with_buffers):
    # If your data is in a projected CRS, convert it to WGS 84 (latitude and longitude)
    if gdf_with_buffers.crs != "epsg:4326":
        gdf_with_buffers = gdf_with_buffers.to_crs("epsg:4326")
    bounds = gdf_with_buffers.total_bounds

    # Correct the order of the bounds for Folium:
    # [[southwest_lat, southwest_lon], [northeast_lat, northeast_lon]]
    return [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]


def create_folium_map(gdf, distance_threshold_meters, lat_col, lon_col):
    # Remove rows where lat or lon is NaN
    gdf = gdf.dropna(subset=[lat_col, lon_col])

    # Generate buffers and calculate bounds
    gdf["buffer"] = gdf.apply(
        lambda row: row.geometry.buffer(distance_threshold_meters), axis=1
    )
    bounds = get_bounds(gdf)
    # st.write(bounds)

    # Start with a base map (zoom start will be adjusted with fit_bounds)
    m = folium.Map()

    # Add points to the map
    for _, row in gdf.iterrows():
        # Determine the color based on the presence of a value in distance_feet
        point_color = "red" if pd.notnull(row["distance_feet"]) else "black"
        tooltip_text = (
            str(row[id_col]) if id_col and pd.notnull(row[id_col]) else "No ID"
        )

        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=1,
            color=point_color,
            fill=True,
            fill_color=point_color,
            fill_opacity=1,  # Set fill opacity to 1 for a solid color
            weight=2,
            tooltip=tooltip_text,  # Add tooltip to the marker
            popup=folium.Popup(
                tooltip_text, parse_html=True
            ),  # Add popup to the marker
        ).add_to(m)

        # Add buffers to the map
        folium.Circle(
            location=[row[lat_col], row[lon_col]],
            radius=distance_threshold_meters,
            color="grey",
            weight=1,
        ).add_to(m)

    # Fit the map to the bounds
    m.fit_bounds(bounds)

    # Add fullscreen control to the map
    plugins.Fullscreen(
        position="topright",
        title="Expand me",
        title_cancel="Exit me",
        force_separate_button=True,
    ).add_to(m)

    return m


# Main processing function
def process_data(df, lat_col, lon_col, distance_threshold, id_column=None):
    # Convert DataFrame to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]))
    gdf = gdf.set_crs(epsg=4326)
    projected_gdf = gdf.to_crs(epsg=32611)

    # Use spatial indexing for efficiency
    tree = projected_gdf.sindex

    # Calculate nearby points
    nearby_points = []
    for index, row in projected_gdf.iterrows():
        buffer = row.geometry.buffer(distance_threshold)
        possible_matches_index = list(tree.intersection(buffer.bounds))
        possible_matches = projected_gdf.iloc[possible_matches_index]
        precise_matches = possible_matches[
            possible_matches.distance(row.geometry) <= distance_threshold
        ]
        precise_matches = precise_matches[precise_matches.index != index]

        for _, pm_row in precise_matches.iterrows():
            nearby_point_info = {
                "index": index,
                "distance_feet": round(
                    pm_row.geometry.distance(row.geometry) * 3.28084, 2
                ),
            }
            if id_column:
                nearby_point_info[f"nearby_{id_col}"] = pm_row[id_column]
            nearby_points.append(nearby_point_info)

    nearby_df = pd.DataFrame(nearby_points)

    # Check if nearby_df is empty, if so, create an empty DataFrame with the 'index' column
    if nearby_df.empty:
        nearby_df = pd.DataFrame(columns=["index", "nearby_id", "distance_feet"])
        st.warning(f"No points within {int(distance_threshold_feet)}ft of another!")
    # Perform a left merge to include all original points
    merged_gdf = gdf.merge(nearby_df, how="left", left_index=True, right_on="index")

    # If an ID column is provided, add it to the merged_gdf
    if id_column:
        merged_gdf = merged_gdf.merge(
            df[[id_column]], left_on="index", right_index=True, how="left"
        )
        merged_gdf.rename(columns={id_column: "original_id"}, inplace=True)

    # Drop the extra 'index', 'id_col_y', and 'buffer' columns as they are no longer needed
    columns_to_drop = ["index", f"{id_col}_y", "buffer"]
    columns_to_drop = [
        col for col in columns_to_drop if col in merged_gdf.columns
    ]  # Ensure the column exists before dropping
    merged_gdf.drop(columns=columns_to_drop, inplace=True)
    merged_gdf.rename(columns={f"{id_column}_x": id_column}, inplace=True)

    return merged_gdf


def select_columns(df, uploaded_file):
    lat_col, lon_col = find_lat_lon_columns(df)
    if uploaded_file:  # skip if using sample data
        if not lat_col or not lon_col:
            st.warning("Spatial columns not detected. Select them below.", icon="⚠️")
            lat_col = st.selectbox("Latitude Column", df.columns)
            lon_col = st.selectbox("Longitude Column", df.columns)
        else:
            lat_col = st.selectbox(
                "Latitude Column", df.columns, index=list(df.columns).index(lat_col)
            )
            lon_col = st.selectbox(
                "Longitude Column", df.columns, index=list(df.columns).index(lon_col)
            )

        # Provide an option to select an ID column
        id_col_options = ["None"] + list(df.columns)
        with st.sidebar:
            msg = "Select an ID Column. Associates each point with a unique identifier; if no `nearby_id` field is required, choose 'None'."
            # Default to the first column in the DataFrame
            id_col = st.selectbox(msg, options=id_col_options, index=1)

        # Check if 'None' is selected and set id_col to None
        if id_col == "None":
            id_col = None
    else:  # use hardcoded sample data values
        lat_col, lon_col, id_col = "LAT", "LONG", "INDEX"
    return lat_col, lon_col, id_col


def process_and_display(
    df, lat_col, lon_col, id_col, distance_threshold_meters, uploaded_file=None
):
    if lat_col and lon_col:
        processed_gdf = process_data(
            df, lat_col, lon_col, distance_threshold_meters, id_col
        )
        display_gdf = processed_gdf.drop(columns=["geometry"])

        # Create and display the map
        folium_static(
            create_folium_map(
                processed_gdf, distance_threshold_meters, lat_col, lon_col
            )
        )
        if id_col:
            st.caption(
                f"""
                        Hover/click on points to view {id_col}\n
                        Points nearby (within {distance_threshold_feet}ft) others are red.
                       """
            )
            # Convert to Excel and offer download
        df_xlsx = convert_df_to_excel(display_gdf)

        if uploaded_file:
            short_file_name = os.path.splitext(uploaded_file.name)[0]
        else:
            short_file_name = "sample_data"

        file_name = (
            f"{short_file_name}_{int(distance_threshold_meters * 3.28084)}ft.xlsx"
        )

        st.download_button(
            label=f"📥 Download {file_name}",
            data=df_xlsx,
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        hide_null_distance = st.checkbox("Hide rows with no nearby point", value=True)
        if hide_null_distance:
            # Filter out rows where 'distance_feet' is null
            display_gdf = display_gdf.dropna(subset=["distance_feet"])
        filtered_df = processed_gdf.dropna(subset=["distance_feet"])
        # Display the processed DataFrame
        st.write("Processed Data:", display_gdf)
        if id_col:
            count = filtered_df[id_col].nunique()
            st.info(
                f"{count}/{len(df)} points are nearby (within {int(distance_threshold_feet)}ft of) another."
            )


# Streamlit UI
with st.sidebar:
    "## Spatial Proximity Excel Enrichment"

    df, uploaded_file = handle_file_upload()

    # Define a slider for distance selection
    distance_threshold_feet = st.slider(
        "Distance threshold in feet",
        min_value=25,
        max_value=800,
        value=100,  # default value
        step=25,
        format="%d feet",
    )

    # Define a number input for custom distance thresholds
    custom_distance_threshold_feet = st.number_input(
        "Or enter a custom distance threshold in feet",
        min_value=0.0,
        value=float(
            distance_threshold_feet
        ),  # set the default value to the slider's value
        step=10.0,
        format="%f",
    )

    # Choose which value to use based on whether the custom value differs from the slider
    if custom_distance_threshold_feet != distance_threshold_feet:
        distance_threshold_feet = custom_distance_threshold_feet

    # Now convert the chosen distance threshold in feet to meters for processing
    distance_threshold_meters = feet_to_meters(distance_threshold_feet)

    lat_col, lon_col, id_col = select_columns(df, uploaded_file)

    "---"
    "### How it works"
    """
    This tool augments Excel spreadsheets with proximity analysis capabilities. It requires a spreadsheet containing latitude and longitude coordinates and adds two fields:
    1. **Distance (Feet)**(`distance_feet`): Calculates the distance to each nearby point in feet.
    2. **Nearby Points**(`nearby_*`): Identifies points within a specified distance threshold (default 100 feet).
    The tool allows for adjustment of the distance threshold and outputs an enhanced spreadsheet with spatial proximity details for further analysis.
    """
    "---"
    "### How to use this"
    """
    - Upload your data in the .xlsx format.
    - Select the appropriate columns for latitude, longitude, and an ID (if applicable).
    - Set the distance threshold to find nearby points.
    - The data is processed automatically. You should see the results on the map and can download the output .xlsx file.
    """
    """
    If you have any problems [email](mailto:daniel.mcvey@sce.com) Danny.
    """
    with st.expander("How it was made"):
        """
        This tool is powered by Streamlit, which allows for rapid development of data applications with Python.
        It uses geospatial libraries like Geopandas for the geographic data processing and Folium for creating interactive maps. You can see the source code [here](https://github.com/danmaps/spatial-proximity-excel/).🤓
        """
        "---"

process_and_display(
    df, lat_col, lon_col, id_col, distance_threshold_meters, uploaded_file
)
