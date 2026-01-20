import os
import shutil
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from automate_resim_screening import compare_files,compare_folders
from automate_resim_screening import delete_files as delete_delta_n_missing
      
year = 2024
month = "March"
start_date = 26
end_date = 31
create_backup = False

parallel_execution = True

if end_date < start_date:
    raise ValueError("End date must be greater than start date.")

directory = "/mnt/usmidet/projects/STLA-THUNDER/11-Development/Filelist/ResimScreening/"

if create_backup:
    backup_dir = os.path.join(directory, 'Backup')

    for i in range(start_date, end_date + 1):
        source_directory = os.path.join(directory, f"{month}_{i}_{year}")
        target_directory = os.path.join(backup_dir, f"{month}_{i}_{year}")
        shutil.copytree(source_directory, target_directory, dirs_exist_ok=True)

# delete_delta_n_missing(target_directory)

def process_directory(i):
    source_directory = os.path.join(directory, f"{month}_{i}_{year}")
    target_directory = os.path.join(directory, f"{month}_{i + 1}_{year}")
    delete_delta_n_missing(target_directory)
    compare_files(target_directory,source_directory)
    compare_folders(target_directory, source_directory, os.path.join(target_directory, "README"))

if parallel_execution : 
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(start_date, end_date):
            futures.append(executor.submit(process_directory, i))
        
        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                pbar.update(1)
else : 
    process_directory(start_date)
