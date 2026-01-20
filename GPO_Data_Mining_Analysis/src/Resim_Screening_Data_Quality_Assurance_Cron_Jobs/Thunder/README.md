# Resim Screening File List Creation Cron Job

## Overview

The scripts support the Resim Screening file list creation cron job, which automates the process of generating flat and continuous file lists from MF4 and MAT files created based on Taylor's pipeline.

## Workflow

The cron job invokes the automate_resim_screening.sh shell script, which in turn calls the automate_resim_screening.py Python script. The convert_flat_to_continuous_resimscreening script is used to create a continuous file list from the input MF4 and MAT files.
