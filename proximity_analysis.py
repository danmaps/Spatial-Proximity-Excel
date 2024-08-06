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
    "QUANTITY": np.random.randint(250, 1500, num_points),
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
        col_lower = col.lower()
        # Check for both full words and abbreviations
        if ("latitude" in col_lower or "lat" in col_lower) and not lat_col:
            lat_col = col  # Assign the first matching latitude column
        elif ("longitude" in col_lower or "lon" in col_lower) and not lon_col:
            lon_col = col  # Assign the first matching longitude column

        # If both columns are found, no need to continue searching
        if lat_col and lon_col:
            break

    return lat_col, lon_col


# convert DataFrame to Excel in memory
def convert_df_to_excel(_df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        _df.to_excel(writer, index=False, sheet_name="Sheet1")
    processed_data = output.getvalue()
    return processed_data


def get_bounds(gdf):
    bounds = gdf.total_bounds
    return [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]


# Function to create tooltips based on the row and group information
def create_tooltip(row, gdf, id_col, sum_col):
    if "group_id" in gdf.columns and pd.notnull(row["group_id"]):
        tooltip_text = f"<b>group_id</b> {int(row['group_id'])}<br><b>Total {sum_col}</b> {row['group_sum']}"
        # Add nearby points to the tooltip
        nearby_points = str(gdf[gdf["group_id"] == row["group_id"]][id_col].tolist())[1:-1].replace("'", "")
        tooltip_text += f"<br><b>nearby {id_col}s</b> " + nearby_points
    else:
        tooltip_text = "Not in group"
    
    return tooltip_text


def create_folium_map(gdf, distance_threshold_meters, lat_col, lon_col, id_col):
    # Remove rows where lat or lon is NaN
    gdf = gdf.dropna(subset=[lat_col, lon_col])

    # calculate bounds
    bounds = get_bounds(gdf)

    # Start with a base map (zoom start will be adjusted with fit_bounds)
    m = folium.Map(tiles="cartodb-dark-matter", width='100%', height='100%')

    # Exclude columns with non-serializable data types
    non_serializable_cols = gdf.select_dtypes(include=['datetime64[ns]', 'timedelta64[ns]', 'period[Q]']).columns
    gdf = gdf.drop(columns=non_serializable_cols)

    # Convert specific columns to strings to avoid commas in large numbers
    if 'EQUIP_NUM' in gdf.columns:
        gdf['EQUIP_NUM'] = gdf['EQUIP_NUM'].astype(str)
    if 'nearby_EQUIP_NUM' in gdf.columns:
        gdf['nearby_EQUIP_NUM'] = gdf['nearby_EQUIP_NUM'].astype(str)

    
    # take advantage of the GeoDataFrame structure to set the style of the data
    # create a column style containing each feature’s style in a dictionary
    def style_function(row):
        if pd.isna(row['group_id']): 
            return "gray"
        else:
            # st.write(f"found a match at {row[id_col]}")
            return "red"

    # Add a style column to the gdf
    if "group_id" in gdf.columns: 
        gdf['style'] = gdf.apply(style_function, axis=1)
    else:
        gdf['style'] = "gray"

    # Create a GeoJson layer

    if display_id:
        geojson_layer = folium.GeoJson(
            gdf,
            style_function=lambda x: {
                'markerColor': x['properties']['style'],},
            marker=folium.Marker(icon=folium.Icon(icon='circle', prefix='fa',)),
            popup=folium.GeoJsonPopup(fields=[display_id, id_col, sum_col, "group_id"], localize=True),
            tooltip=folium.GeoJsonTooltip(
                fields=[display_id, id_col, sum_col, "group_id"], localize=True
            )
        ).add_to(m)

    else: # if no display_id, skip trying to add it to the popup and tooltip
        geojson_layer = folium.GeoJson(
            gdf,
            style_function=lambda x: {
                'markerColor': x['properties']['style'],},
            marker=folium.Marker(icon=folium.Icon(icon='circle', prefix='fa',)),
            popup=folium.GeoJsonPopup(fields=[id_col, sum_col, "group_id"], localize=True),
            tooltip=folium.GeoJsonTooltip(
                fields=[id_col, sum_col, "group_id"], localize=True
            )
        ).add_to(m)


    # Add search functionality
    search = folium.plugins.Search(
        layer=geojson_layer,
        geom_type="Point",
        placeholder=f"Search for {id_col}",
        collapsed=False,
        search_label=id_col,
    ).add_to(m)

    # # Add buffers to the map
    # for _, row in gdf.iterrows():
    #     color = "red" if pd.notnull(row["distance_feet"]) else "white"
    #     tooltip_text = create_tooltip(row, gdf, id_col, sum_col)
        
    #     folium.Circle(
    #         location=[row[lat_col], row[lon_col]],
    #         radius=distance_threshold_meters,
    #         color=color,
    #         weight=2,
    #         fill=True,
    #         # tooltip=tooltip_text,
    #         popup=folium.Popup(tooltip_text, parse_html=False),
    #     ).add_to(m)

    # Reproject the GeoDataFrame to a suitable projected CRS (e.g., UTM)
    gdf = gdf.to_crs(epsg=32611)  # Example EPSG code for UTM zone 11N

    # Add a minimum bounding circle to the points by group_id
    if "group_id" in gdf.columns:
        for group_id in gdf["group_id"].unique():
            if use_sum_threshold:
                # Filter groups with sum above threshold
                group_points = gdf[(gdf["group_id"] == group_id) & (gdf["group_sum"] >= sum_threshold)]
            else:
                group_points = gdf[gdf["group_id"] == group_id]
            
            if group_points.empty:
                continue  # Skip empty groups

            # Calculate the centroid of the group
            centroid = group_points.geometry.unary_union.centroid
            
            if centroid.is_empty:
                continue  # Skip if the centroid is empty
            
            # Calculate the maximum distance from the centroid to any point in the group
            max_distance = group_points.geometry.distance(centroid).max()
            
            # Convert the centroid back to WGS84 (lat/lon)
            centroid_wgs84 = gpd.GeoSeries([centroid], crs=gdf.crs).to_crs(epsg=4326).iloc[0]
            
            # tooltip_text = create_tooltip(row, gdf, id_col, sum_col)
            tooltip_text = f"group_id {group_id}, Total: {group_points['group_sum'].sum()}"

            # Ensure the centroid is valid before proceeding
            if not centroid_wgs84.is_empty:
                folium.Circle(
                    location=[centroid_wgs84.y, centroid_wgs84.x],
                    radius=max_distance,
                    color="white",
                    weight=2,
                    fill=True,
                    tooltip=tooltip_text,
                    popup=folium.Popup(tooltip_text, parse_html=False),
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
def process_data(
    df, lat_col, lon_col, distance_threshold, id_column=None, display_id=None
):
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
    columns_to_drop = [
        "index",
        f"{id_col}_y",
        f"{display_id}_y",
        "buffer",
        f"nearby_{display_id}_y",
        f"nearby_{id_column}_y",
        "distance_feet_y",
    ]
    columns_to_drop = [
        col for col in columns_to_drop if col in merged_gdf.columns
    ]  # Ensure the column exists before dropping
    merged_gdf.drop(columns=columns_to_drop, inplace=True)

    merged_gdf.rename(columns={f"{display_id}_x": display_id}, inplace=True)
    merged_gdf.rename(columns={f"{id_column}_x": id_column}, inplace=True)
    merged_gdf.rename(
        columns={f"nearby_{id_column}_x": f"nearby_{id_column}"}, inplace=True
    )
    merged_gdf.rename(
        columns={f"nearby_{display_id}_x": f"nearby_{display_id}"}, inplace=True
    )
    merged_gdf.rename(columns={f"distance_feet_x": "distance_feet"}, inplace=True)

    return merged_gdf


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            self.parent[root_v] = root_u

    def add(self, u):
        if u not in self.parent:
            self.parent[u] = u

def identify_clusters(df, id_col, display_id=None, sum_col=None):
    uf = UnionFind()

    # Drop duplicates based on id_col
    if id_col:
        df = df.drop_duplicates(subset=[id_col])

    df = df.copy()
    df["group_id"] = np.nan

    if display_id or id_col:
        col_to_use = display_id if display_id else id_col

        for i, row in df.iterrows():
            nearby_index_col = f"nearby_{col_to_use}"
            if nearby_index_col in df.columns and pd.notna(row[nearby_index_col]):
                nearby_index = df[df[col_to_use] == row[nearby_index_col]].index[0]
                uf.add(i)
                uf.add(nearby_index)
                uf.union(i, nearby_index)

        # Assign group IDs based on the union-find structure
        for i in df.index:
            if i in uf.parent:
                df.loc[i, "group_id"] = uf.find(i)
                
        # Normalize group IDs to start from 0
        group_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(df["group_id"].dropna().unique()))}
        df["group_id"] = df["group_id"].map(group_id_map)

    else:
        return df.drop(columns=["group_id"])

    # Sum column logic remains unchanged
    df["group_sum"] = np.nan
    unique_group_ids = df["group_id"].dropna().unique()
    for group_id in unique_group_ids:
        group_sum = df.loc[df["group_id"] == group_id, sum_col].sum()
        df.loc[df["group_id"] == group_id, "group_sum"] = group_sum

    return df



def check_and_set_none(cols):
    return {key: None if value == "None" else value for key, value in cols.items()}


def select_columns_ui(df, col_name, default_value):
    if default_value and default_value in df.columns:
        default_index = list(df.columns).index(default_value)
    else:
        default_index = 0
    return st.selectbox(col_name, df.columns, index=default_index)


def select_sidebar_columns(df, msg, options, default_value):
    if default_value and default_value in options:
        default_index = options.index(default_value)
    else:
        default_index = 0
    with st.sidebar:
        return st.selectbox(msg, options=options, index=default_index)


def handle_column_selection(
    df, lat_default, lon_default, id_default, sum_default, display_id_default
):
    lat_col = select_columns_ui(df, "Latitude Column", lat_default)
    lon_col = select_columns_ui(df, "Longitude Column", lon_default)

    id_col_options = ["None"] + list(df.columns)
    display_id_options = ["None"] + list(df.columns)

    id_col = select_sidebar_columns(
        df, "Select a unique ID", id_col_options, id_default
    )
    display_id = select_sidebar_columns(
        df, "Select an optional display ID", display_id_options, display_id_default
    )
    sum_col = st.sidebar.selectbox(
        "Select a Sum Column",
        df.columns,
        index=df.columns.get_loc(sum_default) if sum_default in df.columns else 0,
    )

    cols = {
        "lat_col": lat_col,
        "lon_col": lon_col,
        "id_col": id_col,
        "sum_col": sum_col,
        "display_id": display_id,
    }

    return check_and_set_none(cols)


def select_columns(df, uploaded_file):
    if uploaded_file:
        lat_col_default, lon_col_default = find_lat_lon_columns(df)
    else:
        lat_col_default, lon_col_default = "LAT", "LONG"

    id_col_default = "INDEX"
    sum_col_default = "QUANTITY"
    display_id_default = "INDEX"

    cols = handle_column_selection(
        df,
        lat_col_default,
        lon_col_default,
        id_col_default,
        sum_col_default,
        display_id_default,
    )
    return cols.values()


def create_groups_df(gdf,id_col,display_id,sum_col):
    
    # Group by 'group_id' and aggregate
    grouped_df = gdf.groupby('group_id').agg(
        id_combined=(id_col, lambda x: ','.join(map(str, x))),
        display_combined=(display_id, lambda x: ','.join(map(str, x))),
        group_QUANTITY=(f'group_{sum_col}', 'first')
    ).reset_index()

    # Rename columns
    if not uploaded_file:
        grouped_df = grouped_df.rename(columns={
            'id_combined': id_col+"_id", # no duplicates allowed
            'display_combined': display_id+"_display", # no duplicates allowed
            'group_QUANTITY': f'{sum_col}'
        })

    else:
        grouped_df = grouped_df.rename(columns={
            'id_combined': id_col,
            'display_combined': display_id, # should be distinct from id_col
            'group_QUANTITY': f'{sum_col}'
        })
    
    return grouped_df



def process_and_display(
    df,
    lat_col,
    lon_col,
    id_col,
    display_id,
    distance_threshold_meters,
    sum_threshold=None,
    uploaded_file=None,
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
        if id_col:
            folium_static(
                create_folium_map(
                    processed_gdf, distance_threshold_meters, lat_col, lon_col, id_col
                ),width=1000,height=500
            )
        else:
            st.warning("choose a unique ID column")

        # Rename group_sum column to f"group_{sum_col}"
        display_gdf = display_gdf.rename(columns={"group_sum": f"group_{sum_col}"})

        if id_col:
            st.caption(f"Hover/click on points to view {id_col}")
        st.caption(
            f"Points nearby (within {distance_threshold_feet}ft) others are red."
        )

        offer_download(display_gdf,uploaded_file,distance_threshold_feet)

        hide_null_distance = st.checkbox("Hide rows with no nearby point", value=True)
        if hide_null_distance:
            display_gdf = display_gdf.dropna(subset=["distance_feet"])
        filtered_df = processed_gdf.dropna(subset=["distance_feet"])

        # Rename group_sum column to f"group_{sum_col}"
        filtered_df = filtered_df.rename(columns={"group_sum": f"group_{sum_col}"})

        # Display the processed DataFrame
        st.info(
            f"{len(filtered_df)}/{len(df)} points are nearby (within {int(distance_threshold_feet)}ft of) another."
        )

        # Convert specific columns to strings to avoid commas in large numbers
        if 'EQUIP_NUM' in display_gdf.columns:
            display_gdf['EQUIP_NUM'] = display_gdf['EQUIP_NUM'].astype(str)
        if 'nearby_EQUIP_NUM' in display_gdf.columns:
            display_gdf['nearby_EQUIP_NUM'] = display_gdf['nearby_EQUIP_NUM'].astype(str)
        if 'EQUI1_NUM' in display_gdf.columns:
            display_gdf['EQUI1_NUM'] = display_gdf['EQUI1_NUM'].astype(str)
        display_gdf

        groups_df = create_groups_df(filtered_df, id_col, display_id, sum_col)

        groups_msg = f"There are {len(groups_df)} groups."

        if use_sum_threshold:
            if sum_threshold:
                groups_over_threshold = groups_df[groups_df[f"{sum_col}"] >= sum_threshold]

            # Show info about how many groups are above the sum threshold
            if sum_threshold:
                unique_groups = len(groups_over_threshold["group_id"].unique())
                groups_msg += f" {unique_groups} of them are over {sum_threshold} {sum_col}."
        
        st.info(groups_msg)

        groups_over_threshold

        offer_download(groups_df,uploaded_file,distance_threshold_feet)
        

def offer_download(df,uploaded_file,distance_threshold_feet):
    '''
    Convert the DataFrame to an Excel file and offer download
    '''
    df_xlsx = convert_df_to_excel(df)

    short_file_name = (
        os.path.splitext(uploaded_file.name)[0] if uploaded_file else "sample_data"
    )

    file_name = (
        f"{short_file_name}_{int(distance_threshold_feet)}ft_groups.xlsx"
    )

    st.download_button(
        label=f"📥 Download {file_name}",
        data=df_xlsx,
        file_name=file_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=df.shape,
    )
    # st.write(df.shape, "rows downloaded.")

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
    distance_threshold_feet = custom_distance_threshold_feet if custom_distance_threshold_feet != distance_threshold_feet else distance_threshold_feet

    # Convert the chosen distance threshold in feet to meters for processing
    distance_threshold_meters = distance_threshold_feet* 0.3048

    use_sum_threshold = st.checkbox("Use group sum threshold", value=False)

    if use_sum_threshold:
        sum_threshold = st.slider(
            "Group sum threshold",
            min_value=100,
            max_value=2000,
            value=1000,
            step=100,
            format="%d",
        )
    else:
        sum_threshold = None


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
    df, lat_col, lon_col, id_col, display_id, distance_threshold_meters, sum_threshold, uploaded_file
)
