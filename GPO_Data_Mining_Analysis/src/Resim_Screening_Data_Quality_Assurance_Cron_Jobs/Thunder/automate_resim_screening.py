import os
import shutil
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

source_directory = r"/mnt/usmidet/projects/STLA-THUNDER/7-Tools/ResimScreening/filelists_by_data_set"
target_directory = r"/mnt/usmidet/projects/STLA-THUNDER/11-Development/Filelist/ResimScreening"

def create_folder(target_directory):
    today = date.today().strftime("%B_%d_%Y")
    folder_path = os.path.join(target_directory, today)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

def copy_files(source_directory, target_directory):
    today_folder = create_folder(target_directory)
    shutil.copytree(source_directory, today_folder, dirs_exist_ok=True)

def delete_files(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(("_delta", "_missing", "README")):
                os.remove(os.path.join(root, file))

def compare_files(today_folder, yesterday_folder):
    delta_folder = os.path.join(today_folder, "Delta")
    if not os.path.exists(delta_folder):
        os.makedirs(delta_folder)

    mf4_files = []
    for root, dirs, files in os.walk(today_folder):
        for file in files:
            if "MF4" in file:
                mf4_files.append(os.path.join(root, file))

    for mf4_file in mf4_files:
        delta_file = os.path.join(delta_folder, os.path.basename(mf4_file) + "_delta")
        missing_file = os.path.join(delta_folder, os.path.basename(mf4_file) + "_missing")

        with open(mf4_file, "r") as today_file:
            today_lines = set(today_file.readlines())

        if os.path.exists(yesterday_folder):
            yesterday_files = set()

            for root, dirs, files in os.walk(yesterday_folder):
                yesterday_file = ""
                for file in files:
                    if os.path.basename(mf4_file) == file:
                        yesterday_file = os.path.join(root, file)
                        break
                if os.path.exists(yesterday_file) and os.path.isfile(yesterday_file):
                    with open(yesterday_file, "r") as yes_file:
                        yesterday_lines = set(yes_file.readlines())

                    new_lines = today_lines - yesterday_lines
                    if new_lines:
                        with open(delta_file, "w") as d_file:
                            d_file.writelines(sorted(new_lines))

                    missing_lines = yesterday_lines - today_lines
                    if missing_lines:
                        with open(missing_file, "w") as miss_file:
                            miss_file.writelines(sorted(missing_lines))

def compare_folders(today_folder, yesterday_folder, readme_file):
    if os.path.exists(yesterday_folder):
        today_files = set()
        yesterday_files = set()

        for root, dirs, files in os.walk(today_folder):
            for file in files:
                today_files.add(os.path.join(root, file))

        for root, dirs, files in os.walk(yesterday_folder):
            for file in files:
                yesterday_files.add(os.path.join(root, file))

        missing_files = yesterday_files - today_files
        if missing_files:
            with open(readme_file, "a") as readme:
                readme.write("Files missing in today's folder:\n")
                for file in missing_files:
                    readme.write(file + "\n")

        new_files = today_files - yesterday_files
        if new_files:
            with open(readme_file, "a") as readme:
                readme.write("New files in today's folder:\n")
                for file in new_files:
                    readme.write(file + "\n")


prev_day_dir = os.path.join(target_directory,(date.today() - timedelta(days=1)).strftime("%B_%d_%Y"))
today_dir = os.path.join(target_directory,date.today().strftime("%B_%d_%Y"))
readme_file = os.path.join(target_directory, today_dir,"README")

copy_files(source_directory, target_directory)
delete_files(today_dir)
compare_files(today_dir, prev_day_dir)
compare_folders(today_dir,  prev_day_dir, readme_file)