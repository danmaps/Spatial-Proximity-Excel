import streamlit as st
from io import BytesIO
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import folium_static
import os

# https://chat.openai.com/c/33d6abdb-0490-4be9-b605-772e357a1489

sample_data = {
    "FLOC_ID": ["UG-5709118","UG-5709119","UG-5709120","UG-5709121"],
    "LAT": [34.297416,34.297436,34.297472,34.297491],
    "LONG": [-118.917072,-118.917028,-118.916945,-118.916901]}

# Possible column names for latitude and longitude
default_lat_names = ['LAT', 'lat', 'latitude', 'Latitude']
default_lon_names = ['LONG', 'long', 'longitude', 'Longitude']

def find_lat_lon_columns(df, default_lat_names, default_lon_names):
    lat_col = None
    lon_col = None

    for lat_name in default_lat_names:
        if lat_name in df.columns:
            lat_col = lat_name
            break

    for lon_name in default_lon_names:
        if lon_name in df.columns:
            lon_col = lon_name
            break

    return lat_col, lon_col

# Helper function to convert DataFrame to Excel in memory
def convert_df_to_excel(_df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        _df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

# Convert feet to meters
def feet_to_meters(feet):
    return feet * 0.3048

def generate_buffers(gdf, distance):
    # Generate a buffer polygon around each point with the specified distance
    gdf['buffer'] = gdf.geometry.buffer(distance)
    return gdf

def get_bounds(gdf_with_buffers):

    # If your data is in a projected CRS, convert it to WGS 84 (latitude and longitude)
    if gdf_with_buffers.crs != 'epsg:4326':
        gdf_with_buffers = gdf_with_buffers.to_crs('epsg:4326')

    # Then calculate the bounds
    bounds = gdf_with_buffers.total_bounds

    # Correct the order of the bounds for Folium: [[southwest_lat, southwest_lon], [northeast_lat, northeast_lon]]
    return [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]


def create_folium_map(gdf, distance_threshold_meters, lat_col, lon_col):
    # Generate buffers and calculate bounds
    gdf['buffer'] = gdf.apply(lambda row: row.geometry.buffer(distance_threshold_meters), axis=1)
    bounds = get_bounds(gdf)
    # st.write(bounds)

    # Start with a base map (zoom start will be adjusted with fit_bounds)
    m = folium.Map()

    # Add points to the map
    for _, row in gdf.iterrows():
        # Determine the color based on the presence of a nearby_id
        point_color = 'red' if pd.notnull(row['distance_feet']) else 'black'
        
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=1,  # Keep the radius small for a dot appearance
            color=point_color,
            fill=True,
            fill_color=point_color,
            fill_opacity=1  # Set fill opacity to 1 for a solid color
        ).add_to(m)

        # Add buffers to the map
        folium.Circle(
            location=[row[lat_col], row[lon_col]],
            radius=distance_threshold_meters,
            color='blue'
        ).add_to(m)


    # Fit the map to the bounds
    m.fit_bounds(bounds)

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
        precise_matches = possible_matches[possible_matches.distance(row.geometry) <= distance_threshold]
        precise_matches = precise_matches[precise_matches.index != index]

        for pm_index, pm_row in precise_matches.iterrows():
            nearby_point_info = {
                'index': index,
                'distance_feet': round(pm_row.geometry.distance(row.geometry) * 3.28084, 2)
            }
            if id_column:
                nearby_point_info['nearby_id'] = pm_row[id_column]
            nearby_points.append(nearby_point_info)

    nearby_df = pd.DataFrame(nearby_points)

    # Perform a left merge to include all original points
    merged_gdf = gdf.merge(nearby_df, how='left', left_index=True, right_on='index')

    # If an ID column is provided, add it to the merged_gdf
    if id_column:
        merged_gdf = merged_gdf.merge(df[[id_column]], left_on='index', right_index=True, how='left')
        merged_gdf.rename(columns={id_column: 'original_id'}, inplace=True)
    
    # Drop the 'index', 'OID_y', and 'buffer' columns as they are no longer needed
    columns_to_drop = ['index', f'{id_column}_y', 'buffer']
    columns_to_drop = [col for col in columns_to_drop if col in merged_gdf.columns]  # Ensure the column exists before dropping
    merged_gdf.drop(columns=columns_to_drop, inplace=True)
    merged_gdf.rename(columns={f'{id_column}_x': id_column}, inplace=True)
    
    return merged_gdf


def handle_file_upload():
    with st.sidebar:
        st.caption("Please upload a .xlsx file to get started. After uploading, select the appropriate columns and set the desired distance threshold.")
        uploaded_file = st.file_uploader("Choose a .xlsx file", type="xlsx")

    # uploaded_file = st.file_uploader("Choose a .xlsx file", type="xlsx")
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.DataFrame(sample_data)  # Use sample data if no file is uploaded
    with st.expander("Data Preview"):
        st.write(df.head())
    return df, uploaded_file

def select_columns(df, default_lat_names, default_lon_names):
    lat_col, lon_col = find_lat_lon_columns(df, default_lat_names, default_lon_names)
    if not lat_col or not lon_col:
        st.error("Could not find default latitude and longitude columns. Please select the columns.")
        lat_col = st.selectbox("Select Latitude Column", df.columns)
        lon_col = st.selectbox("Select Longitude Column", df.columns)

    # Provide an option to select an ID column or continue without it

    id_col_options = ['None'] + list(df.columns)
    with st.sidebar:
        # Default to the first column in the DataFrame
        id_col = st.selectbox("Select an ID Column (optional):", options=id_col_options, index=1)

    # Check if 'None' is selected and set id_col to None
    if id_col == 'None':
        id_col = None


    return lat_col, lon_col, id_col

def process_and_display(df, lat_col, lon_col, id_col, distance_threshold_meters, uploaded_file=None):
    if lat_col and lon_col:
        processed_gdf = process_data(df, lat_col, lon_col, distance_threshold_meters, id_col)
        display_gdf = processed_gdf.drop(columns=['geometry'])
        st.write('Processed Data:', display_gdf.head())
        with st.expander("View data on map"):
            # Create and display the map with Folium
            folium_map = create_folium_map(processed_gdf, distance_threshold_meters, lat_col, lon_col)
            folium_static(folium_map)

        # Convert to Excel and offer download
        df_xlsx = convert_df_to_excel(processed_gdf)
        file_name = 'processed_data.xlsx'
        if uploaded_file:
            file_name = f"{os.path.splitext(uploaded_file.name)[0]}_{int(distance_threshold_meters * 3.28084)}ft.xlsx"
        st.download_button(label='📥 Download Result',
                           data=df_xlsx,
                           file_name=file_name,
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


# Streamlit UI
st.title('Spatial Proximity Excel')
with st.sidebar:

    df, uploaded_file = handle_file_upload()

        
    # Define a slider for distance selection
    distance_threshold_feet = st.slider(
        "Select the distance threshold in feet (for custom values, use the text field below)",
        min_value=25,
        max_value=800,
        value=100,  # default value
        step=25,
        format="%d feet"
    )

    # Define a number input for custom distance thresholds
    custom_distance_threshold_feet = st.number_input(
        "Or type a custom distance threshold in feet",
        min_value=0.0,
        value=float(distance_threshold_feet),  # set the default value to the slider's value
        step=10.0,
        format="%f"
    )
    # Choose which value to use based on whether the custom value differs from the slider
    if custom_distance_threshold_feet != distance_threshold_feet:
        distance_threshold_feet = custom_distance_threshold_feet

    # Now convert the chosen distance threshold in feet to meters for processing
    distance_threshold_meters = feet_to_meters(distance_threshold_feet)
    
    st.info("📌 Instructions: Select an ID Column from the dropdown to associate each point with a unique identifier; if no ID is required for your analysis, you may choose 'None'.")

    lat_col, lon_col, id_col = select_columns(df, default_lat_names, default_lon_names)
process_and_display(df, lat_col, lon_col, id_col, distance_threshold_meters, uploaded_file)
