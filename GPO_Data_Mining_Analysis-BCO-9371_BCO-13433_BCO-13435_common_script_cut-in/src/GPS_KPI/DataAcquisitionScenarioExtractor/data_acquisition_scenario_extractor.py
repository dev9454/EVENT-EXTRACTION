import json
from data_acquisition_utils import Config, RoadType, Area, extract_directory
import os
import importlib
from pathlib import Path


def import_modules():
    """
    Imports classes from all scripts in ./scenarios location
    Class names will be based on script name, for example hov_lanes.py -> HovLanes class

    Returns:
    class_names (list)
    """
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    scenarios_path = os.path.join(current_script_path, 'scenarios')
    scenarios_files = [f for f in os.listdir(scenarios_path) if f.endswith('.py')]
    class_names = []
    for file in scenarios_files:
        file_basename = file[:-3]
        class_name = "".join([i.capitalize() for i in file_basename.split("_")])
        class_names.append(class_name)
        module = importlib.import_module(f"scenarios.{file_basename}")
        globals()[class_name] = getattr(module, class_name)
    return class_names


class DataAcquisitionMain:
    """
    Class designed to run different scripts to extract scenarios.
    Config object is loaded based on a json config file.
    """

    def __init__(self, json_config_path: str, scenarios_list: list) -> None:
        self.json_config_path = json_config_path
        self.output_path = extract_directory(self.json_config_path) / 'DataAcquisitionScenarioExtractor'
        self.scenarios_list = scenarios_list
        self.config = None

    def set_config(self) -> None:
        """
        Creates a Config attribute based on json config file
        """
        with open(self.json_config_path, 'r') as file:
            data = json.load(file)

        arguments_dict = {}
        for file in data["Road_types"]:
            arguments = {}
            road_type = file['road_type']
            arguments['name'] = road_type
            arguments['area'] = Area(*file['area'])
            arguments['raw_file_path'] = file['raw']
            arguments['processed_file_path'] = file['processed']
            arguments['processed_merged_file_path'] = file['merged_processed']
            arguments_dict[road_type] = arguments

        self.config = Config(
            config_path=self.json_config_path,
            region=data["Region"],
            **arguments_dict,
        )

    def run_scenarios(self) -> None:
        """
        Dynamically runs run method on a list of classes extracted from .scenarios location
        """
        for scenario in self.scenarios_list:
            print(f"--- Running {scenario.__name__} scenario ---")
            obj = scenario(self.output_path)
            obj.run(self.config)


def main() -> None:
    class_names = import_modules()
    json_config_file_path = r"/mnt/usmidet/projects/STLA-THUNDER/8-Users/AlgoGroup/GeoData/RoadScenarios_PKL_files/CES2025/CES2025_config.json"
    data_acquisition = DataAcquisitionMain(json_config_file_path, [eval(s) for s in class_names])
    data_acquisition.set_config()
    data_acquisition.run_scenarios()


if __name__ == "__main__":
    main()