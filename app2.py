import streamlit as st
from io import BytesIO
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import numpy as np
from pathlib import Path

# Helper function to convert DataFrame to Excel in memory
def convert_df_to_excel(_df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        _df.to_excel(writer, index=False, sheet_name='Sheet1')
        writer.book.close()  # Explicitly close the xlsxwriter Workbook
    processed_data = output.getvalue()
    output.close()  # Close the BytesIO object
    return processed_data



# Main processing function
def process_data(df):
    # Convert DataFrame to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['LONG'], df['LAT']))
    gdf = gdf.set_crs(epsg=4326)

    # Define the distance threshold in meters (100 feet)
    distance_threshold = 100 * 0.3048
    projected_gdf = gdf.to_crs(epsg=32611)

    # Use spatial indexing for efficiency
    tree = projected_gdf.sindex

    nearby_points = []

    for index, row in projected_gdf.iterrows():
        buffer = row.geometry.buffer(distance_threshold)
        possible_matches_index = list(tree.intersection(buffer.bounds))
        possible_matches = projected_gdf.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.distance(row.geometry) <= distance_threshold]
        precise_matches = precise_matches[precise_matches.index != index]

        for pm_index, pm_row in precise_matches.iterrows():
            nearby_points.append({
                'index': index,
                'nearby_index': pm_index,
                'nearby_floc_id': pm_row['FLOC_ID'],
                'distance_feet': pm_row.geometry.distance(row.geometry) * 3.28084
            })


    nearby_df = pd.DataFrame(nearby_points)

    # Select the relevant columns for merging
    columns_to_merge = ['index', 'nearby_floc_id', 'distance_feet']
    merged_gdf = gdf.merge(nearby_df[columns_to_merge], left_index=True, right_on='index')

    # Drop the 'index' column as it's no longer needed after the merge
    merged_gdf.drop(columns=['index','geometry'], inplace=True)
    return merged_gdf

# Streamlit UI
st.title('Spatial Proximity Excel')
uploaded_file = st.file_uploader("Choose a .xlsx file", type="xlsx")

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write('Data Preview:', df.head())

    processed_gdf = process_data(df)
    st.write('Processed Data:', processed_gdf.head())

    df_xlsx = convert_df_to_excel(processed_gdf)
    st.download_button(label='📥 Download Result',
                       data=df_xlsx,
                       file_name='processed_data.xlsx',
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
