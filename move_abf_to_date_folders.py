"""Move ABF files named YYYY_MM_DD_NNNN.abf into per-day YYYY-MM-DD folders.

Usage:
    python move_abf_to_date_folders.py [FOLDER] [--dry-run]

FOLDER defaults to the current directory. Date folders are created inside
FOLDER. Files that do not match the pattern are left alone, and an existing
file at the destination is never overwritten.
"""

import argparse
import re
import shutil
from pathlib import Path

ABF_NAME = re.compile(r"^(\d{4})_(\d{2})_(\d{2})_\d{4}\.abf$", re.IGNORECASE)


def move_abf_files(folder, dry_run=False):
    folder = Path(folder)
    moved = skipped = 0
    for path in sorted(folder.iterdir()):
        match = ABF_NAME.match(path.name)
        if not path.is_file() or not match:
            continue
        dest_dir = folder / "-".join(match.groups())
        dest = dest_dir / path.name
        if dest.exists():
            print(f"skip (already exists): {dest}")
            skipped += 1
            continue
        print(f"{path.name} -> {dest_dir.name}/")
        if not dry_run:
            dest_dir.mkdir(exist_ok=True)
            shutil.move(str(path), str(dest))
        moved += 1
    verb = "would move" if dry_run else "moved"
    print(f"{verb} {moved} file(s), skipped {skipped}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", nargs="?", default=".",
                        help="folder containing the .abf files (default: current directory)")
    parser.add_argument("--dry-run", action="store_true",
                        help="show what would be moved without moving anything")
    args = parser.parse_args()
    move_abf_files(args.folder, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
