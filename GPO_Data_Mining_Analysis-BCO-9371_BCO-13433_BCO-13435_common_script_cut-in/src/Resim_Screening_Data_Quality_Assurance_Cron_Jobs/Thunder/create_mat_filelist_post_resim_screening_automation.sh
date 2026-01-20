#!/bin/bash

#author : ioiny8
# The script is scheduled to run every day at 5:15 AM EST using the crontab.
# 15 5 * * * /mnt/usmidet/projects/STLA-THUNDER/7-Tools/DMA_Venv/Resim_Screening_Cron_job/create_mat_filelist_post_resim_screening_automation.sh
# The above crontab entry runs the bash script every day at 5:15 AM.

cd /mnt/usmidet/projects/STLA-THUNDER/7-Tools/DMA_Venv/Resim_Screening_Cron_job
source /mnt/usmidet/projects/STLA-THUNDER/7-Tools/hpc_tools_py_venv/bin/activate
base_dir="/mnt/usmidet/projects/STLA-THUNDER/11-Development/Filelist/ResimScreening"
today=$(date +"%B_%d_%Y")
# today="April_19_2024"
date_folder="${base_dir}/${today}"

if [ -d "$date_folder" ]; then
    dirs_in_datedir=$(find "${date_folder}" -mindepth 1 -maxdepth 1 -type d ! -name "Delta")

    flat_mf4_delta_file_paths=""
    expected_delta_mf4_files=""

    exist_Delta_dir=false

    # Check if the 'Delta' folder exists within the date folder
    if [ -d "${date_folder}/Delta" ]; then
        # Code to be executed if the 'Delta' folder exists
        exist_Delta_dir=true
    fi

    for dir in $dirs_in_datedir; do
        # Get the base name of the directory
        dir_name=$(basename "$dir")
        # Append the suffix "_MF4_flat_delta" to the base name
        expected_delta_mf4_files+="${dir_name}_MF4_flat_delta"
        expected_delta_mf4_file="${date_folder}/Delta/${dir_name}_MF4_flat_delta"
        # Check if the expected delta file exists within the Delta folder
        if [ -f $expected_delta_mf4_file ]; then
            # Append the full file path to flat_mf4_delta_file_paths
            #create mat file list out of the mf4 delta file
            python create_mat_filelist.py -i "${expected_delta_mf4_file}"
            temp_flat_list="${date_folder}/Delta/${dir_name}_Mat_flat_delta_dma"
            flat_mat_delta_file_paths+="$temp_flat_list "
            temp_cont_list="${date_folder}/Delta/${dir_name}_Mat_cont_delta_dma"
            python /mnt/usmidet/users/mjxk2s/STLA_THUNDER/Scripts/resim_filelist_util.py -l $temp_flat_list -o $temp_cont_list -op convert_cont  --log_number_string_size 4
            flat_mat_delta_file_paths+="$temp_cont_list "
        fi
    done

    echo "Available delta files:" $flat_mat_delta_file_paths
    echo "$flat_mat_delta_file_paths" | tr ' ' '\n' > "${date_folder}/delta_filelist"
else
    echo "Date folder does not exist: $date_folder"
fi

chmod 777 -R -f "$base_dir"

