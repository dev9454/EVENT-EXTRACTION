import geopandas as gpd
from pathlib import Path
import sys
import data_acquisition_utils


class Intersections:
    """
    Class that extracts intersection scenarios from road network data.
    """

    def __init__(self, output_path):
        self.scenario = '19_49_Intersections'
        self.output_directory = Path(output_path) / self.scenario

    @staticmethod
    def filter(intersections: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Filters the GeoDataFrame to include only intersections where:
        - 'highway_1' or 'highway_2' tag is either 'trunk', 'primary', or 'secondary'
        - contains valid geometries representing points of intersection.

        Args:
        intersections (gpd.GeoDataFrame): Dataframe containing road intersections.

        Returns:
        gpd.GeoDataFrame: Filtered intersections.
        """

        valid_highways = intersections.loc[
            (intersections['highway_1'].isin(['trunk', 'primary', 'secondary'])) |
            (intersections['highway_2'].isin(['trunk', 'primary', 'secondary']))
            ].copy()

        return valid_highways

    def run(self, config) -> None:
        """
        Executes the filtering process, creates map, saves files in HTML, GPKG, and PPTX formats.

        Args:
        config: Configuration object containing paths and settings.
        """

        gdf_intersections = data_acquisition_utils.read_file(config.intersections['processed_file_path'])

        gdf_filtered = self.filter(gdf_intersections)

        if len(gdf_filtered) == 0:
            print("No events found...")
        else:
            self.output_directory.mkdir(parents=True, exist_ok=True)

            comments = ['length_1', 'highway_1', 'surface_1', 'lanes_1', 'maxspeed_1']

            data_acquisition_utils.create_map(gdf_filtered, self.output_directory / f'{self.scenario}.html')
            data_acquisition_utils.create_pptx(gdf_filtered, self.output_directory / f'{self.scenario}.pptx', comments)
            data_acquisition_utils.save_file(gdf_filtered, self.output_directory / f'{self.scenario}.gpkg')