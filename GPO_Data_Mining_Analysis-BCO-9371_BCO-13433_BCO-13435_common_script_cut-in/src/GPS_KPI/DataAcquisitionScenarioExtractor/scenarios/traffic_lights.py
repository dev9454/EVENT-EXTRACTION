import data_acquisition_utils
import geopandas as gpd
from data_acquisition_utils import get_intersection, create_map, create_pptx, read_file, save_file

''' author  : Padmesh (uqgqif)
    version : 1.0 '''


class TrafficLights():

    def __init__(self, output_path):
        self.output_directory_name = "57_58_59_60_Traffic_Lights"
        self.output_directory = output_path / self.output_directory_name

    @staticmethod
    def filter(traffic_signals, road_type):
        filtered = get_intersection(traffic_signals, road_type)
        return filtered

    def run(self, config):
        '''
         Run function to ...................

        :param config: config_object
        :param output_path: output_path
        :return:
        '''

        traffic_signals = read_file(config.traffic_signals['processed_file_path'])
        # traffic_signals = read_file(r"C:\Work\BCO-12581\Gpkg\GCC\gcc-states-20241007_traffic_signals.pkl")
        road_type = read_file(config.road_type['processed_file_path'])
        # road_type = read_file(r"C:\Work\BCO-12581\Gpkg\GCC\gcc-states-20241007_road_type.pkl")
        filtered = self.filter(road_type, traffic_signals)
        if len(filtered) == 0:
            print("No events found...")
        else:
            # Keep only one geometry dtype
            for col in filtered.columns:
                if col != filtered.geometry.name and filtered[col].dtype == 'geometry':
                    filtered[col] = filtered[col].astype(str)

            self.output_directory.mkdir(parents=True, exist_ok=True)
            create_map(filtered, self.output_directory / str(self.output_directory_name + '.html'))
            comments = ['highway_left', 'lanes', 'maxspeed', 'surface']
            create_pptx(filtered, self.output_directory / str(self.output_directory_name + '.pptx'), comments)
            save_file(filtered, self.output_directory / str(self.output_directory_name + '.gpkg'))

