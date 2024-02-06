import streamlit as st
from io import BytesIO
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import folium_static
from folium import plugins

# https://chat.openai.com/c/33d6abdb-0490-4be9-b605-772e357a1489

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
    # Calculate bounds of the buffers
    bounds = gdf_with_buffers['buffer'].total_bounds
    # Bounds are in the form [minx, miny, maxx, maxy], we need to return as [[miny, minx], [maxy, maxx]]
    return [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]

def create_folium_map(gdf, distance_threshold_meters, lat_col, lon_col):
    # Generate buffers and calculate bounds
    gdf['buffer'] = gdf.apply(lambda row: row.geometry.buffer(distance_threshold_meters), axis=1)
    bounds = get_bounds(gdf)

    # Start with a base map (zoom start will be adjusted with fit_bounds)
    m = folium.Map()

    # Add points to the map
    for _, row in gdf.iterrows():
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=3,
            color='red',
            fill=True,
            fill_color='red'
        ).add_to(m)

        # Add buffers to the map
        folium.Circle(
            location=[row[lat_col], row[lon_col]],
            radius=distance_threshold_meters,
            color='blue',
            fill=False
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

     # Update to use the generic 'nearby_id' and include the selected ID column if provided
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
                'distance_feet': pm_row.geometry.distance(row.geometry) * 3.28084
            }
            if id_column:
                nearby_point_info['nearby_id'] = pm_row[id_column]
            nearby_points.append(nearby_point_info)

    nearby_df = pd.DataFrame(nearby_points)

    if id_column:
        nearby_df = nearby_df.merge(df[[id_column]], left_on='index', right_index=True, how='left')

    # Select the relevant columns for merging
    columns_to_merge = ['index', 'nearby_id', 'distance_feet']
    merged_gdf = gdf.merge(nearby_df[columns_to_merge], left_index=True, right_on='index')

    # Drop the 'index' column as it's no longer needed after the merge
    merged_gdf.drop(columns=['index'], inplace=True)
    return merged_gdf

# Streamlit UI
st.title('Spatial Proximity Excel')
uploaded_file = st.file_uploader("Choose a .xlsx file", type="xlsx")


# Define a slider for distance selection
distance_threshold_feet = st.slider(
    "Select the distance threshold in feet (for custom values, use the text field below)",
    min_value=50,
    max_value=1000,
    value=100,  # default value
    step=50,
    format="%d feet"
)

# Define a number input for custom distance thresholds
custom_distance_threshold_feet = st.number_input(
    "Or type a custom distance threshold in feet",
    min_value=0.0,
    value=float(distance_threshold_feet),  # set the default value to the slider's value
    step=0.1,
    format="%.2f"
)
# Choose which value to use based on whether the custom value differs from the slider
if custom_distance_threshold_feet != distance_threshold_feet:
    distance_threshold_feet = custom_distance_threshold_feet

# Now convert the chosen distance threshold in feet to meters for processing
distance_threshold_meters = feet_to_meters(distance_threshold_feet)

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write('Data Preview:', df.head())

    # Find latitude and longitude columns
    lat_col, lon_col = find_lat_lon_columns(df, default_lat_names, default_lon_names)

    if not lat_col or not lon_col:
        st.error("Could not find default latitude and longitude columns. Please select the columns.")
        lat_col = st.selectbox("Select Latitude Column", df.columns)
        lon_col = st.selectbox("Select Longitude Column", df.columns)

    # After uploading and displaying the file, let the user select an ID column
    id_col = None
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.write('Data Preview:', df.head())

        # Provide an option to select an ID column or continue without it
        id_col_options = ['None'] + list(df.columns)
        id_col = st.selectbox("Select an ID Column (optional):", options=id_col_options, index=0)
        if id_col == 'None':
            id_col = None

if uploaded_file and lat_col and lon_col:
    processed_gdf = process_data(df, lat_col, lon_col, distance_threshold_meters, id_col)
    st.write('Processed Data:', processed_gdf.head())
    # Create the map with Folium
    folium_map = create_folium_map(processed_gdf, distance_threshold_meters,lat_col, lon_col)
    
    # Display the map in the Streamlit app
    folium_static(folium_map)
    df_xlsx = convert_df_to_excel(processed_gdf)

    
    st.download_button(label='📥 Download Result',
                       data=df_xlsx,
                       file_name='processed_data.xlsx',
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')




