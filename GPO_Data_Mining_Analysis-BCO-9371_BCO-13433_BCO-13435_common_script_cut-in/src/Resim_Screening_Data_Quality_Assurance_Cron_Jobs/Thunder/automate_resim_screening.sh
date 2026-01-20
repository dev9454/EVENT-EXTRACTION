#!/bin/bash

# author : ioiny8
# The script is scheduled to run every day at 5:00 AM EST using the crontab.
# 0 5 * * * /mnt/usmidet/projects/STLA-THUNDER/7-Tools/DMA_Venv/Resim_Screening_Cron_job/automate_resim_screening.sh
# The above crontab entry runs the bash script every day at 5:00 AM.
start_time=$(date +%s)
current_date=$(date +"%B_%d_%Y")

source /mnt/usmidet/projects/STLA-THUNDER/7-Tools/DMA_Venv/data_analytics_thunder/bin/activate
python  $(dirname $0)/automate_resim_screening.py
deactivate

chmod 777 -R -f /mnt/usmidet/projects/STLA-THUNDER/11-Development/Filelist/ResimScreening/*

end_time=$(date +%s)
execution_time=$((end_time - start_time))

echo "Completed execution for :$current_date  Time taken for execution: $execution_time seconds" >> /mnt/usmidet/projects/STLA-THUNDER/11-Development/Filelist/ResimScreening/cron.log
