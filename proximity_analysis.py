import streamlit as st
from io import BytesIO
import geopandas as gpd
import pandas as pd
import folium
from folium import plugins
from streamlit_folium import folium_static
import os
import numpy as np

st.set_page_config(
    page_title="Spatial Proximity Excel Enrichment",
    page_icon=":world_map:️",
    layout="wide",
)

# Generate random sample data
num_points = 100
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
    "QUANTITY": np.random.randint(100, 1000, num_points),
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
    m = folium.Map(tiles="cartodb-dark-matter")

    # Add points to the map
    for _, row in gdf.iterrows():
        # Determine the color based on the presence of a value in distance_feet
        # st.write(row)
        point_color = "red" if pd.notnull(row["distance_feet"]) else "white"
        tooltip_text = (
            str(row[id_col]) if id_col and pd.notnull(row[id_col]) else "No ID"
        )

        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=2,
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
            weight=2,
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
def process_data(df, lat_col, lon_col, distance_threshold, id_column=None, display_id=None):
    """
    Processes a DataFrame to find nearby points based on a given distance threshold.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        lat_col (str): The name of the column containing the latitude values.
        lon_col (str): The name of the column containing the longitude values.
        distance_threshold (float): The maximum distance (in meters) between two points to be considered nearby.
        id_column (str, optional): The name of the column containing the unique identifier for each point. Defaults to None.
        display_id (str, optional): The name of the column containing the display identifier for each point. Defaults to None.

    Returns:
        pandas.DataFrame: The merged DataFrame containing the original points and their nearby points, along with the distance between them.

    Raises:
        None.

    Notes:
        - The input DataFrame is expected to have columns named 'lat_col' and 'lon_col' containing the latitude and longitude values.
        - If 'id_column' is provided, the resulting DataFrame will also contain the corresponding 'id_column' values.
        - The resulting DataFrame will have columns named 'index', 'nearby_id', and 'distance_feet'.
        - If no points are found within the specified distance threshold, a warning message is displayed and an empty DataFrame is returned.
    """
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
            if display_id:
                nearby_point_info[f"nearby_{display_id}"] = pm_row[display_id]

            if id_column:
                nearby_point_info[f"nearby_{id_column}"] = pm_row[id_column]
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

    # Drop  extra columns
    columns_to_drop = ["index", f"{id_col}_y", f"{display_id}_y", "buffer", f"nearby_{display_id}_y", f"nearby_{id_column}_y", "distance_feet_y"]
    columns_to_drop = [
        col for col in columns_to_drop if col in merged_gdf.columns
    ]  # Ensure the column exists before dropping
    merged_gdf.drop(columns=columns_to_drop, inplace=True)
    
    merged_gdf.rename(columns={f"{display_id}_x": display_id}, inplace=True)
    merged_gdf.rename(columns={f"{id_column}_x": id_column}, inplace=True)
    merged_gdf.rename(columns={f"nearby_{id_column}_x": f"nearby_{id_column}"}, inplace=True)
    merged_gdf.rename(columns={f"nearby_{display_id}_x": f"nearby_{display_id}"}, inplace=True)
    merged_gdf.rename(columns={f"distance_feet_x": "distance_feet"}, inplace=True)
    
    return merged_gdf


def identify_clusters(df, id_col, display_id, sum_col):
    """
    Assigns a group_id to each row based on nearby points.

    Args:
        df (pd.DataFrame): The input DataFrame.
        id_column (str): The name of the column containing the unique identifier for each point.
        sum_col (str): The name of the column containing the values to be summed for each group.

    Returns:
        pd.DataFrame: The input DataFrame with an additional 'group_id' column.
    """

    group_dict = {}
    current_group_id = 0

    def get_or_create_group_id(idx):
        """
        Returns the group_id for the given index.

        If the index is not in the group_dict, creates a new group_id and adds it to the group_dict.

        Args:
            idx (int): The index to get or create the group_id for.

        Returns:
            int: The group_id for the given index.
        """
        nonlocal current_group_id
        if idx in group_dict:
            return group_dict[idx]
        else:
            group_dict[idx] = current_group_id
            current_group_id += 1
            return group_dict[idx]

    # drop duplicates based on id_col
    if id_col:
        df = df.drop_duplicates(subset=[id_col])

    # Add a 'group_id' column to the DataFrame
    df = df.copy()
    df['group_id'] = np.nan

    if display_id or id_col:
        # Use display_id if available, otherwise fall back to id_col
        col_to_use = display_id if display_id else id_col

        # Assign a group_id to each row based on nearby points
        for i, row in df.iterrows():
            nearby_index_col = f'nearby_{col_to_use}'
            if pd.notna(row[nearby_index_col]):
                nearby_index = row[nearby_index_col]
                if row[id_col] in group_dict:
                    group_id = group_dict[row[id_col]]
                else:
                    group_id = get_or_create_group_id(nearby_index)
                df.loc[i, 'group_id'] = group_id
                group_dict[row[id_col]] = group_id

    else:
        # drop group_id column and return the DataFrame
        return df.drop(columns=['group_id'])

    # Add a 'group_sum' column to the DataFrame
    df['group_sum'] = np.nan # this is renamed at the end to "group_{sum_col}"

    # Assign a group_sum to each group
    unique_group_ids = df['group_id'].dropna().unique()
    for group_id in unique_group_ids:
        group_sum = df.loc[df['group_id'] == group_id, sum_col].sum()
        df.loc[df['group_id'] == group_id, 'group_sum'] = group_sum

    return df



def check_and_set_none(cols):
    """
    Check if 'None' is selected and set the value to None, instead of a string containing 'None'.

    Args:
        cols (dict): Dictionary mapping column names to their selected values.

    Returns:
        dict: Updated dictionary with 'None' values set to None.
    """
    return {key: None if value == "None" else value for key, value in cols.items()}


def select_columns(df, uploaded_file):
    """
    Selects the columns from a DataFrame based on the uploaded file.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        uploaded_file (bool): A flag indicating if a file was uploaded.

    Returns:
        Tuple[str, str, Optional[str], str, Optional[str]]: A tuple containing the selected latitude column,
        longitude column, unique ID column, sum column, and optional display ID column.
    """

    lat_col, lon_col = find_lat_lon_columns(df)
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

    # Provide an option to select ID columns
    id_col_options = ["None"] + list(df.columns)
    with st.sidebar:
        id_col = st.selectbox("Select a unique ID", options=id_col_options, index=0)
    
    display_id_options = ["None"] + list(df.columns)
    with st.sidebar:
        display_id = st.selectbox("Select an optional display ID", options=display_id_options, index=0)
    
    # Provide an option to select a sum column
    sum_col = st.sidebar.selectbox("Select a Sum Column", df.columns)

    # Check if 'None' is selected and set the value to None, instead of a string containing 'None'
    cols = {"lat_col": lat_col, "lon_col": lon_col, "id_col": id_col, "sum_col": sum_col, "display_id": display_id}
    cols = check_and_set_none(cols)
    
    # Default values for sample data
    if not uploaded_file:
        lat_col, lon_col, id_col, sum_col, display_id = "LAT", "LONG", "INDEX", "QUANTITY", "INDEX"

    return cols.values()


def process_and_display(
    df, lat_col, lon_col, id_col, display_id, distance_threshold_meters, uploaded_file=None
):
    """
    Process and display spatial proximity analysis results.

    Parameters:
        df (DataFrame): The input DataFrame containing latitude, longitude, and other data.
        lat_col (str): The column name for latitude values.
        lon_col (str): The column name for longitude values.
        id_col (str): The column name to uniquely identify each point.
        distance_threshold_meters (float): The distance threshold in meters to identify nearby points.
        uploaded_file (file, optional): The uploaded Excel file for processing.

    Returns:
        None
    """
    if lat_col and lon_col:
        processed_gdf = process_data(
            df, lat_col, lon_col, distance_threshold_meters, id_col, display_id
        )
        
        processed_gdf = identify_clusters(processed_gdf, id_col, display_id, sum_col)
        display_gdf = processed_gdf.drop(columns=["geometry"])

        # Create and display the map
        folium_static(
            create_folium_map(
                processed_gdf, distance_threshold_meters, lat_col, lon_col
            )
        )

        # Rename group_sum column to f"group_{sum_col}"
        display_gdf = display_gdf.rename(columns={"group_sum": f"group_{sum_col}"})
        
        if id_col:
            st.caption(f"Hover/click on points to view {id_col}")
        st.caption(f"Points nearby (within {distance_threshold_feet}ft) others are red.")

        # Convert to Excel and offer download
        df_xlsx = convert_df_to_excel(display_gdf)

        short_file_name = os.path.splitext(uploaded_file.name)[0] if uploaded_file else "sample_data"

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
            display_gdf = display_gdf.dropna(subset=["distance_feet"])
        filtered_df = processed_gdf.dropna(subset=["distance_feet"])

        # Rename group_sum column to f"group_{sum_col}"
        filtered_df = filtered_df.rename(columns={"group_sum": f"group_{sum_col}"})


        # Display the processed DataFrame
        st.info(f"{len(filtered_df)}/{len(df)} points are nearby (within {int(distance_threshold_feet)}ft of) another.")
        display_gdf

        


# Streamlit UI
            
"## Spatial Proximity Excel Enrichment"

with st.sidebar:
    
    df, uploaded_file = handle_file_upload()

    # Define a slider for distance selection
    distance_threshold_feet = st.slider(
        "Distance threshold in feet",
        min_value=25,
        max_value=800,
        value=100,  # default value
        step=25,
        format="%d feet"
    )

    # Define a number input for custom distance thresholds
    custom_distance_threshold_feet = st.number_input(
        "Or enter a custom distance threshold in feet",
        min_value=0.0,
        value=float(
            distance_threshold_feet
        ),  # set the default value to the slider's value
        step=10.0,
        format="%f"
    )

    # Choose which value to use based on whether the custom value differs from the slider
    if custom_distance_threshold_feet != distance_threshold_feet:
        distance_threshold_feet = custom_distance_threshold_feet

    # Now convert the chosen distance threshold in feet to meters for processing
    distance_threshold_meters = feet_to_meters(distance_threshold_feet)

    lat_col, lon_col, id_col, sum_col, display_id = select_columns(df, uploaded_file)

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
    Questions? Problems? Praise? You can [email Danny](mailto:daniel.mcvey@sce.com).
    """
    with st.expander("How it was made"):
        """
        This tool is powered by Streamlit, which allows for rapid development of data applications with Python.
        It uses geospatial libraries like Geopandas for the geographic data processing and Folium for creating interactive maps. You can see the source code [here](https://github.com/danmaps/spatial-proximity-excel/).🤓
        """
        "---"

process_and_display(
    df, lat_col, lon_col, id_col, display_id, distance_threshold_meters, uploaded_file
)
