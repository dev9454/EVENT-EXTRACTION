from pathlib import Path
from data_acquisition_utils import Config,read_file,create_map,create_pptx,save_file
import sys
import geopandas as gpd

class EmergencyVehicleRoads:
    """
    Class that extracts roads within 100m of hospitals from OSM geopackage file
    """

    def __init__(self, output_path: Path) -> None:
        self.output_directory = output_path / "45_46_Emergency_Vehicle_Roads"

    def filter(self, buildings_gdf: gpd.GeoDataFrame, roads_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Extracts roads with hospitals nearby from a geodataframe

        Args:
        buildings_gdf (gpd.DataFrame), roads_gdf (gpd.DataFrame)

        Returns
        output_gdf (gpd.DataFrame) Dataframe containing roads with hospitals within 100m
        """
        offset_distance = 0.001  # 0.001 = 100m
        hospitals = buildings_gdf[(buildings_gdf['amenity'] == 'hospital') | (buildings_gdf['building'] == 'hospital')].reset_index(drop=True)
        road_types = ['motorway', 'primary', 'secondary', 'motorway_link', 'primary_link', 'secondary_link', 'residential', 'trunk']
        roads = roads_gdf[roads_gdf['highway'].isin(road_types)].reset_index(drop=True)
        print(roads.crs)
        output_gdf = roads.sjoin_nearest(hospitals, max_distance = offset_distance, distance_col="distance").reset_index(drop=True)
        if len(output_gdf) == 0:
            print("No events found...")
            sys.exit()
        return output_gdf

    def run(self, config: Config) -> None:
            """
            Runs filter function and then saves filtered dataframe to .html map, pptx presentation
            and .gpkg geopackage file
            """
            self.buildings_data = read_file(config.buildings['processed_file_path'])
            self.roads_data = read_file(config.road_type['processed_file_path'])
 
            filtered_data = self.filter(self.buildings_data, self.roads_data)
            comments = ['highway', 'surface', 'lanes', 'maxspeed']
            self.output_directory.mkdir(parents=True, exist_ok=True)
            create_map(filtered_data, self.output_directory / '45_46_Emergency_Vehicle_Roads.html')
            create_pptx(filtered_data, self.output_directory / '45_46_Emergency_Vehicle_Roads.pptx', comments)
            save_file(filtered_data, self.output_directory / '45_46_Emergency_Vehicle_Roads.gpkg')
