import data_acquisition_utils
import geopandas as gpd


class SpeedScenes:
    """
    Class that extracts Speed Scenes scenarios from OSM geopackage files
    """

    def __init__(self, output_path):
        self.scenario = '94_SpeedScenes'
        self.output_directory = output_path

    @staticmethod
    def filter_turn_lanes(road_type: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Extracts turn lanes rows from geodataframe

        Args:
        road_type (gpd.DataFrame)

        Returns
        turn_lanes (gpd.DataFrame) GeoDataFrame with turn lanes filtered rows

        """
        turn_lanes = road_type[road_type['turn:lanes'].str.contains('\Aleft|\Aright|[^_]left|[^_]right')] \
            .reset_index().drop(columns='index')
        turn_lanes = turn_lanes[turn_lanes['access'] == '']
        return turn_lanes

    @staticmethod
    def filter_roundabout(roundabout: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Extracts roundabout rows from geodataframe

        Args:
        roundabout (gpd.DataFrame)

        Returns
        roundabout (gpd.DataFrame) GeoDataFrame with roundabout filtered rows
        """
        roundabout = roundabout.loc[roundabout['highway'].isin(['secondary', 'tertiary', 'primary', 'residential',
                                                                'service', 'trunk', 'tertiary_link',
                                                                'living_street'])].reset_index().drop(columns='index')
        roundabout.rename(columns={'object_name': 'name'}, inplace=True)
        roundabout = roundabout[roundabout['access'] == ''].reset_index().drop(columns='index')
        return roundabout

    @staticmethod
    def filter_parking(parking: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Extracts parkings rows from geodataframe

        Args:
        parking (gpd.DataFrame)

        Returns
        parking (gpd.DataFrame) GeoDataFrame with parking filtered rows
        """
        return parking

    @staticmethod
    def filter_speed_bumps(traffic_calming: gpd.GeoDataFrame, road_type: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Extracts speed_bumps rows from geodataframe

        Args:
        traffic_calming (gpd.DataFrame)
        road_type (gpd.DataFrame)

        Returns
        speed_bumps (gpd.DataFrame) GeoDataFrame with speed bumps with merged roads
        """
        road_type = road_type[road_type['access'] == '']
        return data_acquisition_utils.get_intersection(traffic_calming, road_type)

    @staticmethod
    def filter_interchanges(road_type: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Extracts interchanges rows from geodataframe

        Args:
        road_type (gpd.DataFrame)

        Returns
        interchanges (gpd.DataFrame) GeoDataFrame with interchanges filtered rows
        """
        road_type = road_type[road_type['access'] == '']
        interchanges = road_type[road_type['highway'] == 'motorway_link'].reset_index().drop(columns='index')
        return interchanges

    def run(self, config: data_acquisition_utils.Config) -> None:
        """
        Runs filter function and then saves filtered dataframe to .html map, pptx presentation
        and .gpkg geopackage file

        """
        turn_lanes_output_directory = self.output_directory / '94_Turn_Lanes'
        roundabout_output_directory = self.output_directory / '94_Roundabout'
        parking_output_directory = self.output_directory / '94_Parking'
        speed_bumps_output_directory = self.output_directory / '94_Speed_Bumps'
        interchanges_output_directory = self.output_directory / '96_Interchanges'

        road_type = data_acquisition_utils.read_file(config.road_type['processed_file_path'])
        roundabout = data_acquisition_utils.read_file(config.roundabout['processed_file_path'])
        parking = data_acquisition_utils.read_file(config.parking['processed_file_path'])
        traffic_calming = data_acquisition_utils.read_file(config.traffic_calming['processed_file_path'])

        # Turn_Lanes
        turn_lanes_filtered = self.filter_turn_lanes(road_type)
        if len(turn_lanes_filtered) == 0:
            print("No turn_lanes events found...")
        else:
            turn_lanes_output_directory.mkdir(parents=True, exist_ok=True)
            comments_turn_lanes = []
            data_acquisition_utils.create_map(turn_lanes_filtered, turn_lanes_output_directory / '94_Turn_Lanes.html')
            data_acquisition_utils.create_pptx(turn_lanes_filtered, turn_lanes_output_directory / '94_Turn_Lanes.pptx',
                                               comments_turn_lanes)
            data_acquisition_utils.save_file(turn_lanes_filtered, turn_lanes_output_directory / '94_Turn_Lanes.gpkg')

        # Roundabout
        roundabout_filtered = self.filter_roundabout(roundabout)
        if len(roundabout_filtered) == 0:
            print("No roundabout events found...")
        else:
            roundabout_output_directory.mkdir(parents=True, exist_ok=True)
            comments_roundabout = []
            data_acquisition_utils.create_map(roundabout_filtered, roundabout_output_directory / '94_Roundabout.html')
            data_acquisition_utils.create_pptx(roundabout_filtered, roundabout_output_directory / '94_Roundabout.pptx',
                                               comments_roundabout)
            data_acquisition_utils.save_file(roundabout_filtered, roundabout_output_directory / '94_Roundabout.gpkg')

        # Parking
        parking_filtered = self.filter_parking(parking)
        if len(parking_filtered) == 0:
            print("No parking events found...")
        else:
            parking_output_directory.mkdir(parents=True, exist_ok=True)
            comments_parking = []
            data_acquisition_utils.create_map(parking_filtered, parking_output_directory / '94_Parking.html')
            data_acquisition_utils.create_pptx(parking_filtered, parking_output_directory / '94_Parking.pptx',
                                               comments_parking)
            data_acquisition_utils.save_file(parking_filtered, parking_output_directory / '94_Parking.gpkg')

        # Speed_Bumps
        speed_bumps_filtered = self.filter_speed_bumps(traffic_calming, road_type)
        if len(speed_bumps_filtered) == 0:
            print("No speed bump events found...")
        else:
            speed_bumps_output_directory.mkdir(parents=True, exist_ok=True)
            comments_speed_bumps = []
            data_acquisition_utils.create_map(speed_bumps_filtered, speed_bumps_output_directory / '94_Speed_Bump.html')
            data_acquisition_utils.create_pptx(speed_bumps_filtered, speed_bumps_output_directory / '94_Speed_Bumps.pptx',
                                               comments_speed_bumps)
            data_acquisition_utils.save_file(speed_bumps_filtered.drop(['geometry', 'b_geometry'], axis=1)
                                             , speed_bumps_output_directory / '94_Speed_Bumps.gpkg')

        # Interchanges
        interchanges_filtered = self.filter_interchanges(road_type)
        if len(interchanges_filtered) == 0:
            print("No interchange events found...")
        else:
            interchanges_output_directory.mkdir(parents=True, exist_ok=True)
            comments_interchanges = []
            data_acquisition_utils.create_map(interchanges_filtered, interchanges_output_directory / '96_Interchanges.html')
            data_acquisition_utils.create_pptx(interchanges_filtered, interchanges_output_directory / '96_Interchanges.pptx'
                                               , comments_interchanges)
            data_acquisition_utils.save_file(interchanges_filtered, interchanges_output_directory / '96_Interchanges.gpkg')
