import argparse
import subprocess
import os
import sys
from pathlib import Path

def run_azure_cmd(cmd):
    res = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    if res.returncode != 0:
        raise RuntimeError(f"Azure CLI failed:\n{res.stderr}")
    return res.stdout.strip().splitlines()

def list_all_blobs(container, prefix):
    cmd = (
        f'az storage blob list '
        f'--container-name {container} '
        f'--account-name nexusintermediateus '
        f'--auth-mode login '
        f'--prefix "{prefix}" '
        f'--query "[].name" -o tsv'
    )
    return run_azure_cmd(cmd)

def download_blob(container, blob_name, out_root):
    local_path = Path(out_root) / blob_name
    local_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = (
        f'az storage blob download '
        f'--container-name {container} '
        f'--account-name nexusintermediateus '
        f'--auth-mode login '
        f'--name "{blob_name}" '
        f'--file "{local_path}" '
        f'--overwrite'
    )
    subprocess.run(cmd, shell=True, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", required=True)
    ap.add_argument("--ignition-path", required=True)
    ap.add_argument("--out-dir", default="DOWNLOADED_MATS")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"[INFO] Container     : {args.container}")
    print(f"[INFO] Ignition path : {args.ignition_path}")
    print(f"[INFO] Output dir    : {args.out_dir}")
    print(f"[INFO] Dry run       : {args.dry_run}")

    blobs = list_all_blobs(args.container, args.ignition_path)

    mat_blobs = [b for b in blobs if b.endswith("_merged.mat")]

    if not mat_blobs:
        print("[WARNING] No MAT files found")
        sys.exit(0)

    print(f"[INFO] Found {len(mat_blobs)} MAT files")

    for blob in mat_blobs:
        print(f"[INFO] Downloading: {blob}")
        if not args.dry_run:
            download_blob(args.container, blob, args.out_dir)

    print("[INFO] Download completed")

if __name__ == "__main__":
    main()
