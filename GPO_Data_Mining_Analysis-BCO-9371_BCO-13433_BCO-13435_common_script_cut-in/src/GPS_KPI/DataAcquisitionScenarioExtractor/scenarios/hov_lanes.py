from pathlib import Path
from data_acquisition_utils import Config, read_file, create_map, create_pptx, save_file
import sys
import geopandas as gpd


class HovLanes:
    """
    Class that extracts HOV_lanes scenarios from OSM geopackage file
    """

    def __init__(self, output_path: Path) -> None:
        self.output_directory = output_path / "79_HOV_Lane"

    def filter(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Extracts HOV_lane rows from geodataframe

        Args:
        gdf (gpd.DataFrame)

        Returns
        output_gdf (gpd.DataFrame) Dataframe with HOV_lane filtered rows
        """
        excluded_values = ['', 'no']
        output_gdf = gdf[~gdf['hov:lanes'].isin(excluded_values) | (~gdf['hov'].isin(excluded_values)) | (~gdf['hov:conditional'].isin(excluded_values)) | (~gdf['hov:minimum'].isin(excluded_values))].copy()
        output_gdf.reset_index(drop=True, inplace=True)

        return output_gdf

    def run(self, config: Config) -> None:
        """
        Runs filter function and then saves filtered dataframe to .html map, pptx presentation
        and .gpkg geopackage file
        """
        self.config = config
        self.input_file_path = config.road_type['processed_file_path']
        self.data = read_file(self.input_file_path)

        filtered_data = self.filter(self.data)

        if len(filtered_data) == 0:
            print("No events found...")
        else:
            comments = ['length', 'highway', 'surface', 'lanes', 'maxspeed']
            self.output_directory.mkdir(parents=True, exist_ok=True)
            create_map(filtered_data, self.output_directory / '79_HOV_Lane.html')
            create_pptx(filtered_data, self.output_directory / '79_HOV_Lane.pptx', comments)
            save_file(filtered_data, self.output_directory / '79_HOV_Lane.gpkg')
