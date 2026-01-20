"""
@author: Wiktor Wiernasiewicz (klzde9)
last update: 22.08.2024

Parameters:
geometry (shapely.geometry.LineString):
This is a LineString geometry object that represents a road or path. This geometry consists of a sequence of points connected by lines, creating a multi-point line.

The function assumes that the input is a LineString, which is a line created by a series of points. If the geometry passed is not a LineString, the function returns None.

Return value:
slf (float):
The function returns the Straightness Factor (SLF), which is a floating point number. If the path length is 0 (theoretically this should not happen), the function returns 0.

If the geometry passed is not a LineString, the function returns None.
"""


import geopandas as gpd
from shapely.geometry import LineString



def calculate_slf(geometry):
    # Check if the geometry is a LineString
    if isinstance(geometry, LineString):
        # Calculate the Euclidean distance (straight line distance)
        start_point = geometry.coords[0]
        end_point = geometry.coords[-1]
        euclidean_distance = LineString([start_point, end_point]).length
        
        # Calculate the actual length of the way
        actual_length = geometry.length
        
        # Calculate the Straight Line Factor (SLF)
        slf = euclidean_distance / actual_length if actual_length > 0 else 0
        return slf
    else:
        return None


#EXample use

file_path = "C:/Users/klzde9/Desktop/Python_scripts/Calculate_straight_road_factor/files/us-midwest-latest_tunnel.gpkg"
gdf = gpd.read_file(file_path)
gdf.head()
# Applying the function to the GeoDataFrame
gdf['SLF'] = gdf['geometry'].apply(calculate_slf)

# Display the first few rows with the SLF column
gdf[['geometry', 'SLF']].head()