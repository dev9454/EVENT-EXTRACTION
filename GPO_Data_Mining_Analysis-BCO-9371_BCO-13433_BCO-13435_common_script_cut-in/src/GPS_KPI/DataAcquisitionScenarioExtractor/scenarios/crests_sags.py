import data_acquisition_utils
import geopandas as gpd


class CrestsSags:
    """
    Class that extracts crests and sags scenarios from OSM geopackage file.
    """

    def __init__(self, output_path):
        self.scenario = '88_Crests_Sags'
        self.output_directory = output_path / self.scenario

    @staticmethod
    def filter(road_type: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Filter function for crests and sags.

        :param road_type: Road_type GeoDataFrame
        :return: Filtered GeoDataFrame with crests or sags.
        """
        road_type = road_type[road_type['access'] == '']
        crests_sags = road_type[road_type['incline'] != ''].reset_index().drop(['index'], axis=1)
        return crests_sags

    def run(self, config: data_acquisition_utils.Config) -> None:
        """
        Run function for crests and sags scenarios, returns none, saves map (.html), presentation (.pptx)
        and geopackage (.gpkg) files on specified location
        :param config: Config object
        :return: None
        """
        road_type = data_acquisition_utils.read_file(config.road_type['processed_file_path'])

        gdf_filtered = self.filter(road_type)
        if len(gdf_filtered) == 0:
            print("No events found...")
        else:
            comments = []
            self.output_directory.mkdir(parents=True, exist_ok=True)
            data_acquisition_utils.create_map(gdf_filtered, self.output_directory / '88_Crests_Sags.html')
            data_acquisition_utils.create_pptx(gdf_filtered, self.output_directory / '88_Crests_Sags.pptx', comments)
            data_acquisition_utils.save_file(gdf_filtered, self.output_directory / '88_Crests_Sags.gpkg')
