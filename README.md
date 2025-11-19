# Spatial Proximity Enricher

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://spatial-proximity-excel.streamlit.app/)

## Overview

A geospatial analytics tool for performing proximity analysis on Excel spreadsheets. Calculates distances between points, identifies spatial clusters, and generates interactive maps and downloadable reports. Works with latitude/longitude data to analyze spatial relationships and patterns.

![image](https://github.com/user-attachments/assets/4cceca3f-826b-49d7-bda8-4f0cacfb91b7)

## Applications

### Utility & Energy
- Identify equipment (transformers, poles, meters) within specified distances for consolidated maintenance
- Group customer locations and affected assets for crew dispatch
- Cluster work zones for vegetation management
- Analyze asset density across service territories

### Field Services
- Group service calls by geographic proximity
- Allocate equipment and personnel based on spatial clustering
- Analyze service area coverage and territory boundaries

### Facilities & Property Management
- Group properties by location for inspection scheduling
- Identify zones for emergency planning
- Coordinate contractor work areas

## Features

- **Proximity Detection**: Calculate distances between all points in a dataset
- **Spatial Clustering**: Group nearby points using Union-Find algorithm
- **Interactive Maps**: View results with zoom, search, and filtering capabilities
- **Configurable Thresholds**: Adjust distance (feet/meters) and sum thresholds
- **Multiple Export Formats**: Excel, CSV, and GIS shapefiles
- **Large Dataset Support**: Tested with up to 10,000 points
- **Group Aggregation**: Sum values by cluster (e.g., capacity, cost, customer count)
- **Minimum Bounding Circles**: Generate polygons around clusters for GIS integration
- **Point Search**: Locate specific assets or IDs on the map
- **Summary Statistics**: Automatic calculation of group counts and totals

## Quick Start

### Online Access (Recommended)

Access the hosted application instantly at [spatial-proximity-excel.streamlit.app](https://spatial-proximity-excel.streamlit.app/)

**No installation required** - Upload your Excel file and start analyzing immediately.

---

## Local Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

### System Requirements

- **OS**: Windows 10/11, macOS 10.14+, or Linux
- **RAM**: 4GB minimum (8GB recommended for large datasets)
- **Disk Space**: 500MB for Python environment and dependencies

### Installation Instructions

### Windows

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/danmaps/spatial-Proximity-Excel.git
   cd spatial-Proximity-Excel
   ```

2. **Set Up a Virtual Environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application:**

   ```bash
   streamlit run proximity_analysis.py
   ```

5. **Access the Application:**
   - The app will automatically open in your default browser
   - Default URL: `http://localhost:8501`

### Linux and macOS

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/danmaps/spatial-Proximity-Excel.git
   cd spatial-Proximity-Excel
   ```

2. **Set Up a Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application:**

   ```bash
   streamlit run proximity_analysis.py
   ```

5. **Access the Application:**
   - The app will automatically open in your default browser
   - Default URL: `http://localhost:8501`

### Deactivation

To deactivate the virtual environment, use:

- **Windows:** `venv\Scripts\deactivate`
- **Linux/macOS:** `deactivate`

---

## Usage Guide

### Input Data Requirements

Your Excel file (.xlsx) should contain:

- **Latitude column**: Decimal degrees (e.g., 34.0522)
- **Longitude column**: Decimal degrees (e.g., -118.2437)
- **Optional ID column**: Unique identifier for each point
- **Optional quantity column**: Numeric values for sum analysis (e.g., cost, capacity, priority)

**Example Data Structure:**

```
| ID    | LATITUDE | LONGITUDE  | EQUIPMENT_TYPE | QUANTITY |
|-------|----------|------------|----------------|----------|
| E001  | 34.0522  | -118.2437  | Transformer    | 500      |
| E002  | 34.0528  | -118.2441  | Meter          | 150      |
| E003  | 34.0525  | -118.2439  | Pole           | 300      |
```

### Step-by-Step Workflow

1. **Upload Your Data**
   - Click "Choose a .xlsx file" in the sidebar
   - Or use the sample data to explore features

2. **Configure Analysis**
   - Select your latitude and longitude columns
   - Choose a unique ID column (recommended)
   - Select a quantity column for sum analysis (optional)
   - Set distance threshold (default: 100 feet)

3. **Adjust Thresholds**
   - **Distance Threshold**: How far apart points can be to cluster (25-800 feet)
   - **Group Sum Threshold**: Minimum total value to highlight a cluster (optional)

4. **Review Results**
   - **Interactive Map**: Red points are clustered, gray are isolated
   - **Grouped Data**: See all proximity relationships
   - **Groups Summary**: Consolidated view of clusters

5. **Export Results**
   - Download Excel/CSV files with enriched data
   - Export shapefiles for GIS integration
   - Share reports with stakeholders

### Example Workflows

#### Equipment Inspection Routing

1. Upload equipment location spreadsheet (transformers, poles, meters)
2. Set distance threshold (e.g., 200 feet)
3. Review clusters on interactive map
4. Download grouped data showing which assets are near each other
5. Use results to plan consolidated inspection routes

#### Outage Response Analysis

1. Upload customer location data with customer counts
2. Set distance threshold based on service area characteristics
3. Enable group sum threshold to identify high-impact clusters
4. Filter and prioritize based on total affected customers
5. Export results for crew dispatch planning

#### Work Zone Planning

1. Upload work order locations with relevant quantities
2. Set distance threshold appropriate for the work type
3. Review spatial clustering patterns
4. Export shapefiles for contractor or crew assignment
5. Use cluster boundaries for zone-based planning

---

## Troubleshooting

### Common Issues

**Issue: "No points within X feet of another"**

- **Cause**: Distance threshold too small for your data density
- **Solution**: Increase the distance threshold slider

**Issue: Coordinates not displaying correctly**

- **Cause**: Latitude/longitude columns swapped or incorrect format
- **Solution**: Verify lat/lon are in decimal degrees (not DMS format)
- **Example**: Use 34.0522, not 34°03'08"N

**Issue: File upload fails**

- **Cause**: File format incompatibility or corrupted file
- **Solution**: Ensure file is .xlsx (not .xls or .csv), try re-saving in Excel

**Issue: Application runs slowly with large datasets**

- **Cause**: Processing thousands of points with complex calculations
- **Solution**:
  - Filter data before uploading (focus on specific regions/time periods)
  - Increase distance threshold to reduce cluster complexity
  - Use local installation for better performance

**Issue: Shapefile export fails**

- **Cause**: Column names contain special characters or exceed 10 characters
- **Solution**: Application automatically truncates names; check exported files

### Getting Help

- **Email Support**: [daniel.mcvey@sce.com](mailto:daniel.mcvey@sce.com)
- **GitHub Issues**: [Report bugs or request features](https://github.com/danmaps/spatial-proximity-excel/issues)
- **Documentation**: See inline help tooltips in the application

### Performance Tips

- Large datasets (>5000 points): Use local installation with 8GB+ RAM
- Repeated analysis: Column selections persist within session
- Batch processing: Process files separately for better performance
- Dense visualizations: Use filtering to reduce map complexity

---

## Technical Details

### Technology Stack

- **Frontend**: Streamlit
- **Geospatial Processing**: GeoPandas, Shapely, Fiona
- **Visualization**: Folium (Leaflet.js)
- **Data Processing**: Pandas, NumPy, OpenPyXL

### Coordinate Systems

- **Input**: WGS84 (EPSG:4326) - Standard GPS coordinates
- **Processing**: UTM Zone 11N (EPSG:32611) for accurate distance calculations
- **Output**: WGS84 (EPSG:4326) for GIS compatibility

### Clustering Algorithm

Uses Union-Find data structure for spatial clustering:

1. Spatial indexing (R-tree) identifies candidate point pairs
2. Distance calculations performed in projected coordinate system (UTM)
3. Union-Find groups connected points with O(α(n)) complexity
4. Aggregate statistics calculated per cluster
5. Minimum bounding circles generated for visualization

### Performance

- **Processing Speed**: ~1000 points/second
- **Memory**: ~50MB + (dataset size × 2)
- **Tested Range**: Up to 10,000 points
- **Accuracy**: ±1 foot for distances under 1000 feet

---

## Roadmap

### Planned Features

- [ ] Database connectivity (SQL Server, PostgreSQL)
- [ ] REST API for system integration
- [ ] Batch processing interface
- [ ] Network analysis (shortest path, service areas)
- [ ] Time-series analysis for temporal patterns
- [ ] Additional coordinate system support

### Under Consideration

- [ ] Machine learning clustering suggestions
- [ ] 3D visualization with elevation data
- [ ] Additional export formats
- [ ] Enhanced statistics and reporting

---

## Contributing

We welcome contributions from the community!

### How to Contribute

1. **Fork the repository**

2. **Create a feature branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow PEP 8 style guidelines
   - Add comments for complex logic
   - Update documentation as needed

4. **Test thoroughly**
   - Test with various dataset sizes
   - Verify all export formats work
   - Check map rendering on different browsers

5. **Commit your changes**

   ```bash
   git commit -m 'Add feature: description of changes'
   ```

6. **Push to your fork**

   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request**
   - Describe your changes clearly
   - Reference any related issues
   - Include screenshots for UI changes

### Development Guidelines

- Write clear, self-documenting code
- Add docstrings to all functions
- Include type hints where appropriate
- Update README for new features
- Test with real-world data scenarios

---

## Citation

If you use this tool in your research or publications, please cite:

```bibtex
@software{spatial_proximity_enricher,
  author = {McVey, Daniel},
  title = {Spatial Proximity Enricher},
  year = {2025},
  url = {https://github.com/danmaps/spatial-proximity-excel}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Geospatial processing powered by [GeoPandas](https://geopandas.org/)
- Interactive maps by [Folium](https://python-visualization.github.io/folium/)

---

**Made with ❤️ for utility operations and spatial analytics**

*Questions? Contact: [daniel.mcvey@sce.com](mailto:daniel.mcvey@sce.com)*
