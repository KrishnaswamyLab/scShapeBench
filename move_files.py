import shutil
from pathlib import Path

SRC_ROOT = Path("submissions_old")
DST_ROOT = Path("submissions")

for file_path in SRC_ROOT.rglob("*"):
    if not file_path.is_file():
        continue

    # Expect structure:
    # submissions_old/SCD-0001/k25/dreeb/...
    try:
        _, scd_id, k_folder, user, *relative_inside_dreeb = file_path.parts
    except ValueError:
        # skip anything not matching expected depth
        continue

    # Rebuild relative path inside dreeb folder (if any subfolders exist)
    rel_inside = Path(*relative_inside_dreeb)

    dest_path = DST_ROOT / user / k_folder / scd_id / rel_inside

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Moving: {file_path} -> {dest_path}")
    shutil.move(str(file_path), str(dest_path))

print("Done.")