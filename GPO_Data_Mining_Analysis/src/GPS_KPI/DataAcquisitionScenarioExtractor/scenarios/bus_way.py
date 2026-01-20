from pathlib import Path
from data_acquisition_utils import Config,read_file,create_map,create_pptx,save_file
import geopandas as gpd

class BusWay:
    """
    Class that extracts bus lanes from OSM geopackage file
    """

    def __init__(self, output_path: Path):
        self.output_directory = output_path / "47_Bus_Way"

    def filter(self, road_type: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Extracts bus lanes from geodataframes

        Args:
        gdf (gpd.DataFrame) Dataframe for road_type

        Returns
        output_gdf (gpd.DataFrame) Dataframe containing geometries of roads with busways/lanes
        """
        road_type = road_type[road_type['access:lanes'] == '']  # filter for access tag
        excluded_values = ['', 'no']
        output_gdf = road_type[~road_type['bus:lanes'].isin(excluded_values) | (~gdf['busway'].isin(excluded_values))]
        output_gdf.reset_index(drop=True, inplace=True)
        if len(output_gdf) == 0:
            print("No events found...")
        return output_gdf

    def run(self, config: Config) -> None:
        """
        Runs filter function and then saves filtered dataframe to .html map, pptx presentation
        and .gpkg geopackage file
        """
        road_type = read_file(config.road_type['processed_file_path'])
        filtered_data = self.filter(road_type)
        comments = ['highway', 'surface', 'lanes', 'maxspeed']
        self.output_directory.mkdir(parents=True, exist_ok=True)
        create_map(filtered_data, self.output_directory / '47_Bus_Way.html')
        create_pptx(filtered_data, self.output_directory / '47_Bus_Way.pptx', comments)
        save_file(filtered_data, self.output_directory / '47_Bus_Way.gpkg')
