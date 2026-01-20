#!/usr/bin/env python3

import os
import sys
import argparse
import json
import logging
from glob import glob
from datetime import datetime

from find_da_generic import FindDaGeneric


# =======================
# LOGGING
# =======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# =======================
# HELPERS
# =======================
def find_mat_files(root):
    """Recursively find *_merged.mat files"""
    return sorted(
        glob(os.path.join(root, "**", "*_merged.mat"), recursive=True)
    )


def write_output(out_dict, output_dir):
    if out_dict is None:
        raise RuntimeError("DA returned None")

    if not isinstance(out_dict, dict):
        raise RuntimeError(f"DA failed: {out_dict}")

    if not out_dict:
        raise RuntimeError("DA returned empty output dictionary")

    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, "da_output.json")

    with open(out_path, "w") as f:
        json.dump(
            {
                k: v.to_dict(orient="records")
                for k, v in out_dict.items()
            },
            f,
            indent=2,
        )

    logging.info(f"DA output written to {out_path}")


def run_da_on_mat(mat_file, config_path, output_dir):
    logging.info(f"Processing MAT : {mat_file}")
    logging.info(f"Output dir     : {output_dir}")

    da = FindDaGeneric()

    out = da.run(
        file_name=[mat_file],
        #is_continuous=False,
        config_path=config_path,
    )

    write_output(out, output_dir)


# =======================
# MAIN
# =======================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--local-mat", help="Path to a single MAT file")
    parser.add_argument(
        "--local-ignition", help="Path to ignition directory"
    )
    parser.add_argument(
        "--config",
        default="gpo_core/eventExtraction/data/E2E/config_e2e_v1_da_basic.yaml",
        help="DA config file",
    )

    args = parser.parse_args()

    if not args.local_mat and not args.local_ignition:
        parser.error("Provide --local-mat or --local-ignition")

    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        raise RuntimeError(f"Config file not found: {config_path}")

    start_time = datetime.now()

    # =======================
    # SINGLE MAT MODE
    # =======================
    if args.local_mat:
        mat_file = os.path.abspath(args.local_mat)

        if not os.path.exists(mat_file):
            raise RuntimeError(f"MAT file not found: {mat_file}")

        run_mode = "LOCAL_MAT"
        ignition_name = os.path.splitext(os.path.basename(mat_file))[0]

        output_dir = os.path.join(
            "EXTRACTED_EVENTS", ignition_name
        )

        logging.info(f"RUN MODE      : {run_mode}")
        run_da_on_mat(mat_file, config_path, output_dir)

    # =======================
    # IGNITION MODE
    # =======================
    if args.local_ignition:
        ignition_root = os.path.abspath(args.local_ignition)

        if not os.path.isdir(ignition_root):
            raise RuntimeError(
                f"Ignition path not found: {ignition_root}"
            )

        mat_files = find_mat_files(ignition_root)

        if not mat_files:
            raise RuntimeError("No *_merged.mat files found")

        ignition_name = os.path.basename(ignition_root)

        logging.info("RUN MODE      : LOCAL_IGNITION")
        logging.info(f"Ignition path : {ignition_root}")
        logging.info(f"Found {len(mat_files)} MAT files")

        for mat in mat_files:
            mat_name = os.path.basename(os.path.dirname(mat))

            output_dir = os.path.join(
                "EXTRACTED_EVENTS", ignition_name, mat_name
            )

            run_da_on_mat(mat, config_path, output_dir)

    elapsed = (datetime.now() - start_time).total_seconds()
    logging.info("DA EXECUTION COMPLETED")
    logging.info(f"Time taken : {elapsed:.2f} sec")


# =======================
if __name__ == "__main__":
    main()
