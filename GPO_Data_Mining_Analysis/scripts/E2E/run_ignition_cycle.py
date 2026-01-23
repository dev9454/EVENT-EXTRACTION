import os
import sys
import time
import re
import json
import logging
from pathlib import Path
from functools import reduce
from datetime import datetime, timezone
from termcolor import colored

from psac.interface import PSACInterface
from find_da_generic import FindDaGeneric

# -------------------------------------------------
# Config & Security
# -------------------------------------------------
ENABLE_NEXUS = True
SERVER_URL = "https://nexus.aptiv.com/api"
NEXUS_EMAIL = "Dev.Gupta@aptiv.com"
# In production, consider: NEXUS_PASSWORD = os.getenv("NEXUS_PASSWORD")
NEXUS_PASSWORD = "Aptiv@123Aptiv@123"

# -------------------------------------------------
# Path setup
# -------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------------------------------------
# Production Logging Setup
# -------------------------------------------------
log_file = OUTPUT_DIR / f"extraction_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

# -------------------------------------------------
# Global Metrics for Summary Report
# -------------------------------------------------
metrics = {
    "files_processed": 0,
    "events_extracted": 0,
    "events_uploaded": 0,
    "events_out_of_bounds": 0,
    "errors": 0
}

# -------------------------------------------------
# Time helpers
# -------------------------------------------------
def secondsToStr(t):
    return "%d:%02d:%02d.%03d" % \
        reduce(lambda ll, b: divmod(ll[0], b) + ll[1:], [(t * 1000,), 1000, 60, 60])

def iso_to_epoch(ts: str) -> float:
    """Converts Nexus ISO string to epoch float for comparison."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()

def epoch_to_nexus_iso(ts: float) -> str:
    """Converts epoch strictly to UTC ISO format to match Nexus Z-suffix."""
    dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt_utc.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

# -------------------------------------------------
# Simplified Nexus Matching Logic
# -------------------------------------------------
def find_log_best_match(pi, run_name):
    """Searches dataOrigin directly by removing the '_merged' suffix."""
    search_term = run_name.replace("_merged", "")
    logging.info(f"STEP 2: Matching to Nexus Log - Seeking '{search_term}'")

    try:
        logs = pi.getLogList(query={"dataOrigin": {"$regex": f".*{search_term}.*"}})
        if logs:
            log_obj = logs[0]
            logging.info(colored(f"  [FOUND] Match: {log_obj.get('_id')} | {log_obj.get('dataOrigin')}", "green"))
            return log_obj
    except Exception as e:
        logging.error(f"Nexus API search error: {e}")
        metrics["errors"] += 1

    logging.warning(colored(f"  [FAIL] No log found matching: {search_term}", "red"))
    return None

# -------------------------------------------------
# Upload logic with Duplicate Prevention
# -------------------------------------------------
def process_nexus_upload(pi, event_records, base_name):
    log_obj = find_log_best_match(pi, base_name)
    if not log_obj:
        return

    log_id = log_obj["_id"]
    n_start_iso = log_obj["startTime"]
    n_end_iso = log_obj["endTime"]
    n_start_epoch = iso_to_epoch(n_start_iso)
    n_end_epoch = iso_to_epoch(n_end_iso)

    logging.info(f"  [BOUNDS] Nexus Start: {n_start_iso} | End: {n_end_iso}")
    logging.info(colored(f"STEP 3: Verifying {len(event_records)} extracted events", "cyan"))

    for entry in event_records:
        metrics["events_extracted"] += 1
        nexus_event_type = entry.get("nexus_event_type", "unknown")
        
        if not nexus_event_type or nexus_event_type == "unknown":
            continue

        da_start = entry["event_start_cTime"]
        da_end = entry["event_end_cTime"]
        script_start_iso = epoch_to_nexus_iso(da_start)
        script_end_iso = epoch_to_nexus_iso(da_end)

        # Bounds Check
        if (da_start < n_start_epoch) or (da_start > n_end_epoch):
            logging.warning(f"  [OUT OF BOUNDS] {nexus_event_type} at {script_start_iso}")
            metrics["events_out_of_bounds"] += 1
            continue
        
        # Upload
        if ENABLE_NEXUS:
            try:
                # Use 'algorithm' emitter and 'return_existing' for production safety
                pi.createEvent(
                    start=script_start_iso, 
                    end=script_end_iso,
                    event_name=nexus_event_type,
                    log=log_id,
                    emitter="manual",
                    return_existing=True
                )
                logging.info(colored(f"  [SUCCESS] Uploaded {nexus_event_type}", "green"))
                metrics["events_uploaded"] += 1
            except Exception as e:
                logging.error(f"  [FAILURE] Upload error for {nexus_event_type}: {e}")
                metrics["errors"] += 1

# -------------------------------------------------
# Main Execution Loop
# -------------------------------------------------
def main():
    program = "E2E"
    config_name = "config_e2e_v1_da_basic.yaml"
    data_dir = PROJECT_ROOT / "data" / program / "extracted_data"
    local_config = str(PROJECT_ROOT / "src" / "eventExtraction" / "data" / program / config_name)

    # Input file list
    input_files = ["SDV_E2EML_M7_20260103_090957_0020_merged.mat"]

    pi = PSACInterface(url=SERVER_URL, user=NEXUS_EMAIL, pass_word=NEXUS_PASSWORD)
    extractor = FindDaGeneric()
    start_run_time = time.time()

    logging.info(colored(f"{'='*30} START PRODUCTION RUN {'='*30}", "blue"))

    for mat in input_files:
        mat_path = data_dir / mat
        if not mat_path.exists():
            logging.error(f"File not found: {mat_path}")
            continue

        metrics["files_processed"] += 1
        logging.info(colored(f"STEP 1: Processing {mat}", "blue"))
        
        try:
            result = extractor.run(str(mat_path), local=True, program=program, config_path=local_config)
            
            if isinstance(result, dict) and "output_data_da" in result:
                records = result["output_data_da"].to_dict(orient="records")
                
                # Save persistent JSON copy
                out_file = OUTPUT_DIR / f"{mat_path.stem}.json"
                with open(out_file, "w") as f:
                    json.dump(records, f, indent=2)
                
                process_nexus_upload(pi, records, mat_path.stem)
            else:
                logging.error(f"No valid extraction data in {mat}")
        except Exception as e:
            logging.error(f"Critical processing error for {mat}: {e}")
            metrics["errors"] += 1

    # Final Summary Report
    duration = secondsToStr(time.time() - start_run_time)
    logging.info(colored(f"\n{'='*30} FINAL SUMMARY REPORT {'='*30}", "blue"))
    logging.info(f"Total Time:         {duration}")
    logging.info(f"Files Processed:    {metrics['files_processed']}")
    logging.info(f"Events Extracted:   {metrics['events_extracted']}")
    logging.info(colored(f"Events Uploaded:    {metrics['events_uploaded']}", "green"))
    logging.info(colored(f"Events Out-of-Bound: {metrics['events_out_of_bounds']}", "yellow"))
    logging.info(colored(f"Total Errors:       {metrics['errors']}", "red" if metrics['errors'] > 0 else "green"))
    logging.info(f"Log file saved to:  {log_file}")
    logging.info(colored(f"{'='*80}", "blue"))

if __name__ == "__main__":
    main()