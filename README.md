# Proximity Analysis for Excel Spreadsheets

https://spatial-proximity-excel.streamlit.app/

## Features

- **Automatic Detection of Latitude and Longitude Columns**: Supports both full names and abbreviations (e.g., "Latitude" or "Lat", "Longitude" or "Lon").
- **Distance Calculation**: Computes the distance in feet between points, identifying all points within a user-defined distance threshold.
- **Enhanced Spreadsheet Output**: Adds two new columns to the original spreadsheet:
  - **Distance (Feet)**(`distance_feet`): Shows the distance to each nearby point in feet.
  - **Nearby Points**(`nearby_id`): Lists IDs of all points within the specified distance threshold.

## Requirements
- Python 3.x
- Pandas
- Geopandas (for data processing)
- Folium (for mapping)

## Usage

1. Prepare your Excel spreadsheet with latitude and longitude columns.
2. Run the tool and upload your spreadsheet.
3. Set the desired distance threshold.
4. Download the enhanced spreadsheet with proximity analysis results.

## Contributing

Contributions to improve the tool or extend its capabilities are welcome. Please feel free to fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.