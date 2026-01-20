#!/bin/bash

year=2024
month="March"
start_date=26
end_date=31
create_backup=false

parallel_execution=true

if [ $end_date -lt $start_date ]; then
    echo "End date must be greater than start date."
    exit 1
fi

directory="/mnt/usmidet/projects/STLA-THUNDER/11-Development/Filelist/ResimScreening/"

process_directory() {
    i=$1
    temp_flatList="$directory${month}_${i}_${year}/Delta/KX_W5U_RWUP_AEB_NA_Mat_flat_delta_dma"
    temp_cont_list="$directory${month}_${i}_${year}/Delta/KX_W5U_RWUP_AEB_NA_Mat_cont_delta_dma"
    command="/mnt/usmidet/users/mjxk2s/STLA_THUNDER/Scripts/resim_filelist_util.py -l $temp_flatList -o $temp_cont_list -op convert_cont --log_number_string_size 4"
    $command
}

if [ $parallel_execution = true ]; then
    for ((i=start_date; i<end_date; i++)); do
        process_directory $i &
    done
    wait
else
    process_directory $start_date
fi
