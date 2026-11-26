"""Download only the PVDAQ daily files for one system and date window.

Run from the project folder:   python -m modules.download
No AWS account is needed (anonymous access to the public OEDI data lake).
Safe to stop and run again: files already downloaded are skipped.
"""
import re
from pathlib import Path

import pandas as pd

import config

CSV_ROOT = "oedi-data-lake/pvdaq/csv"


def file_date(path):
    """Date of a daily file, from its name (2021_06_01 / 2021-06-01 / 20210601)
    or from year=/month=/day= folders. Returns None if no date can be found."""
    for m in re.finditer(r"(?<!\d)(?=(\d{4})[-_]?(\d{2})[-_]?(\d{2})(?!\d))", Path(path).name):
        if not 2000 <= int(m[1]) <= 2100:      # skip things like the system id 1332
            continue
        try:
            return pd.Timestamp(int(m[1]), int(m[2]), int(m[3]))
        except ValueError:
            continue
    y = re.search(r"year=(\d{4})", path)
    mo = re.search(r"month=(\d{1,2})", path)
    d = re.search(r"day=(\d{1,2})", path)
    if y and mo and d:
        try:
            return pd.Timestamp(int(y[1]), int(mo[1]), int(d[1]))
        except ValueError:
            return None
    return None


def find_system_folder(fs, system_id, root=CSV_ROOT, max_depth=3):
    """Search the data lake folders (a few levels deep) for this system's folder."""
    wanted = {f"system_id={system_id}", str(system_id), f"system_{system_id}",
              f"system-{system_id}", f"{system_id}_"}
    level = [root.rstrip("/")]
    for _ in range(max_depth):
        nxt = []
        for folder in level:
            for entry in fs.ls(folder, detail=False):
                name = entry.rstrip("/").split("/")[-1]
                if name in wanted:
                    return entry
                if "." not in name:        # a folder, not a file
                    nxt.append(entry)
        level = nxt
    return None


def files_for_window(fs, system_folder, start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    keep, skipped = [], 0
    for f in fs.find(system_folder):
        if not f.lower().endswith((".csv", ".parquet")):
            continue
        day = file_date(f)
        if day is None:
            skipped += 1
        elif start <= day <= end:
            keep.append((f, day))
    return sorted(keep, key=lambda x: x[1]), skipped


def main():
    import s3fs

    fs = s3fs.S3FileSystem(anon=True)
    prefix = getattr(config, "S3_SYSTEM_PREFIX", None)
    if not prefix:
        print(f"Looking for system {config.SYSTEM_ID} under {CSV_ROOT} ...")
        prefix = find_system_folder(fs, config.SYSTEM_ID)
    if not prefix:
        print("\nCould not find the system's folder automatically. The top-level folders are:")
        for p in fs.ls(CSV_ROOT, detail=False)[:30]:
            print("  ", p)
        raise SystemExit("\nSend this list to your trainer/assistant, or set S3_SYSTEM_PREFIX "
                         "in config.py to the folder for your system.")
    print("System folder:", prefix)

    files, skipped = files_for_window(fs, prefix, config.START, config.END)
    if skipped:
        print(f"({skipped} files had no date in the path and were skipped)")
    if not files:
        raise SystemExit("No files in the date window. Check START/END in config.py.")
    print(f"{len(files)} daily files in {config.START} to {config.END}")

    out = Path(config.RAW_FOLDER)
    out.mkdir(parents=True, exist_ok=True)
    done = 0
    for i, (f, day) in enumerate(files, 1):
        local = out / f"{config.SYSTEM_ID}_{day:%Y-%m-%d}{Path(f).suffix.lower()}"
        if not local.exists():
            fs.get(f, str(local))
            done += 1
        if i % 25 == 0 or i == len(files):
            print(f"  {i}/{len(files)} ({done} newly downloaded)")
    print(f"Done. Files are in {out.resolve()}")


if __name__ == "__main__":
    main()
