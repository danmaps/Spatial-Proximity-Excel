import pandas as pd
import geopandas as gpd
import pytest
from app2 import process_data

@pytest.fixture
def sample_df():
    data = {
        "LAT": [40.712776, 34.052235, 41.878113],
        "LONG": [-74.005974, -118.243683, -87.629799],
        "ID": [1, 2, 3]
    }
    return pd.DataFrame(data)

def test_process_data(sample_df):
    lat_col = "LAT"
    lon_col = "LONG"
    distance_threshold = 1000
    id_column = "ID"
    
    result = process_data(sample_df, lat_col, lon_col, distance_threshold, id_column)
    
    assert isinstance(result, gpd.GeoDataFrame)
    assert "nearby_id" in result.columns
    assert "distance_feet" in result.columns