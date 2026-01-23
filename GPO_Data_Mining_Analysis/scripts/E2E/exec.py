import argparse
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
SERVER_URL = "https://nexus.aptiv.com/api"
NEXUS_EMAIL = "Dev.Gupta@aptiv.com"
NEXUS_PASSWORD = "Aptiv@123Aptiv@123"
ENABLE_NEXUS = True

# Global Metrics
metrics = {
    "files_processed": 0,
    "events_extracted": 0,
    "events_uploaded": 0,
    "events_out_of_bounds": 0,
    "errors": 0
}

# -------------------------------------------------
# Time Helpers
# -------------------------------------------------
def secondsToStr(t):
    return "%d:%02d:%02d.%03d" % \
        reduce(lambda ll, b: divmod(ll[0], b) + ll[1:], [(t * 1000,), 1000, 60, 60])

def iso_to_epoch(ts: str) -> float:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()

def epoch_to_nexus_iso(ts: float) -> str:
    # Converts epoch strictly to UTC ISO format to match Nexus Z-suffix
    dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt_utc.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

# -------------------------------------------------
# Nexus Logic
# -------------------------------------------------
def find_log_best_match(pi, run_name):
    search_term = run_name.replace("_merged", "")
    logging.info(f"STEP 2: Matching to Nexus Log - Seeking '{search_term}'")
    try:
        logs = pi.getLogList(query={"dataOrigin": {"$regex": f".*{search_term}.*"}})
        if logs:
            log_obj = logs[0]
            # Print Log details same as your requested trace
            logging.info(colored(f"   [FOUND] Match: {log_obj.get('_id')} | {log_obj.get('dataOrigin')}", "green"))
            return log_obj
    except Exception as e:
        logging.error(f"Nexus API search error: {e}")
        metrics["errors"] += 1
    logging.warning(colored(f"   [FAIL] No log found matching: {search_term}", "red"))
    return None

def process_nexus_upload(pi, event_records, base_name):
    log_obj = find_log_best_match(pi, base_name)
    if not log_obj:
        return

    log_id = log_obj["_id"]
    n_start_iso = log_obj["startTime"]
    n_end_iso = log_obj["endTime"]
    n_start_epoch = iso_to_epoch(n_start_iso)
    n_end_epoch = iso_to_epoch(n_end_iso)

    # Logging Nexus Bounds
    logging.info(f"   [BOUNDS] Nexus Window: {n_start_iso} to {n_end_iso}")
    logging.info(colored(f"STEP 3: Verifying {len(event_records)} events", "cyan"))

    for entry in event_records:
        metrics["events_extracted"] += 1
        nexus_event_type = entry.get("nexus_event_type", "unknown")
        
        if not nexus_event_type or nexus_event_type == "unknown":
            continue

        da_start = entry["event_start_cTime"]
        da_end = entry["event_end_cTime"]
        script_start_iso = epoch_to_nexus_iso(da_start)
        script_end_iso = epoch_to_nexus_iso(da_end)

        # Print Event Check details
        logging.info(f"\n   Checking Event: {colored(nexus_event_type, 'yellow')}")
        logging.info(f"      Script Start: {script_start_iso}")
        logging.info(f"      Script End:   {script_end_iso}")

        # Bounds Check
        if (da_start < n_start_epoch) or (da_start > n_end_epoch):
            logging.warning(colored("      [OUT OF BOUNDS] Skipping upload.", "red"))
            metrics["events_out_of_bounds"] += 1
            continue
        
        logging.info(colored("      [IN BOUNDS] Verification successful.", "green"))

        if ENABLE_NEXUS:
            try:
                pi.createEvent(
                    start=script_start_iso, 
                    end=script_end_iso,
                    event_name=nexus_event_type,
                    log=log_id,
                    emitter="manual", # Ensures it shows in DMA
                    return_existing=True
                )
                logging.info(colored("      [SUCCESS] Event created.", "green"))
                metrics["events_uploaded"] += 1
            except Exception as e:
                logging.error(f"      [FAILURE] Upload error: {e}")
                metrics["errors"] += 1

# -------------------------------------------------
# Main Execution
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Nexus Event Extraction Pipeline")
    parser.add_argument("--matfiles", nargs='+', required=True, help="Path to .mat files")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(exist_ok=True, parents=True)

    log_file = output_dir / f"extraction_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]
    # Dynamically find the config in the project structure
    local_config = str(project_root / "src/eventExtraction/data/E2E/config_e2e_v1_da_basic.yaml")

    pi = PSACInterface(url=SERVER_URL, user=NEXUS_EMAIL, pass_word=NEXUS_PASSWORD)
    extractor = FindDaGeneric()
    start_run_time = time.time()

    logging.info(colored(f"{'='*30} START PRODUCTION RUN {'='*30}", "blue"))

    for mat_str in args.matfiles:
        mat_path = Path(mat_str).resolve()
        if not mat_path.exists():
            logging.error(f"File not found: {mat_path}")
            metrics["errors"] += 1
            continue

        metrics["files_processed"] += 1
        logging.info(colored(f"\nSTEP 1: Processing {mat_path.name}", "blue"))
        
        try:
            result = extractor.run(str(mat_path), local=True, program="E2E", config_path=local_config)
            
            if isinstance(result, dict) and "output_data_da" in result:
                records = result["output_data_da"].to_dict(orient="records")
                
                # Save JSON result to provided output directory
                out_file = output_dir / f"{mat_path.stem}.json"
                with open(out_file, "w") as f:
                    json.dump(records, f, indent=2)
                logging.info(colored(f"[OUTPUT SAVED] {out_file}", "green"))
                
                process_nexus_upload(pi, records, mat_path.stem)
            else:
                logging.error(f"No valid extraction data in {mat_path.name}")
                metrics["errors"] += 1
        except Exception as e:
            logging.error(f"Critical processing error for {mat_path.name}: {e}")
            metrics["errors"] += 1

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