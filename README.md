# Proximity Analysis for Excel Spreadsheets

Access the tool [here](https://spatial-proximity-excel.streamlit.app/).

## Screenshot
![image](https://github.com/user-attachments/assets/4cceca3f-826b-49d7-bda8-4f0cacfb91b7)

## Features

- **Automatic Detection of Latitude and Longitude Columns**: Supports both full names and abbreviations (e.g., "Latitude" or "Lat", "Longitude" or "Lon").
- **Distance Calculation**: Computes the distance in feet between points, identifying all points within a user-defined distance threshold.
- **Enhanced Spreadsheet Output**: Adds columns to the original spreadsheet:
  - **Distance (Feet)** (`distance_feet`): Shows the distance to each nearby point in feet.
  - **Nearby Points** (`nearby_id`): Lists IDs of all points within the specified distance threshold.
  - **Group ID** (`group_id`): A unique index of each cluster of points.
  - **Group Sum** (`group_{user specified sum column}`): The sum of the specified column for each point in each cluster.

## How it was made
This tool is powered by Streamlit, which allows for rapid development of data applications with Python. 

## Usage

1. Prepare your Excel spreadsheet with latitude and longitude columns.
2. Upload your spreadsheet.
3. Verify lat/long fields.
4. Choose a distance threshold.
5. Select a column to use for cluster summation.
6. The analysis runs automatically with each change.
7. When it is finished, review the map and download the enhanced spreadsheet with proximity analysis results.

## Contributing

Contributions to improve the tool or extend its capabilities are welcome. Please feel free to fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
