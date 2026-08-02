# Step5 finalize: byte-copy the 90 airplane PNGs from arm1 (step2_shadow/stage5_dataset)
# into the final arm2 dataset so it is a complete 490 (90 airplane + 400 ship).
# Airplanes are identical between arm1 and final arm2 by design -> copy arm1's bytes verbatim.
# Also merge airplane rows from arm1's airplane_labels.csv into arm2's all/dataset_labels.csv
# and copy airplane PNGs into arm2's all/ folder. Ship CSV/rows remain untouched.
import os, glob, shutil, csv, hashlib

ARM1 = r"PATH_TO_ARM1_DATASET"
ARMS = {
    "arm2": r"PATH_TO_FINAL_ARM2_DATASET",
}

def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f: h.update(f.read())
    return h.hexdigest()

def read_csv(p):
    with open(p, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f)), None
    # header captured via DictReader.fieldnames below

a1_air_pngs = sorted(glob.glob(os.path.join(ARM1, "airplane", "*.png")))
assert len(a1_air_pngs) == 90, f"arm1 airplane count != 90: {len(a1_air_pngs)}"

# arm1 airplane csv (rows + header)
with open(os.path.join(ARM1, "airplane", "airplane_labels.csv"), newline='', encoding='utf-8') as f:
    rdr = csv.DictReader(f)
    a1_air_rows = list(rdr)
    air_header = rdr.fieldnames

for arm, base in ARMS.items():
    plane_dir = os.path.join(base, "airplane")
    all_dir = os.path.join(base, "all")
    os.makedirs(plane_dir, exist_ok=True)
    os.makedirs(all_dir, exist_ok=True)
    n_copied = 0
    for src in a1_air_pngs:
        nm = os.path.basename(src)
        dst_plane = os.path.join(plane_dir, nm)
        dst_all = os.path.join(all_dir, nm)
        shutil.copy2(src, dst_plane)
        shutil.copy2(src, dst_all)
        assert md5(src) == md5(dst_plane) == md5(dst_all), f"copy byte mismatch {nm}"
        n_copied += 1
    # write airplane_labels.csv (byte-equivalent rows from arm1)
    with open(os.path.join(plane_dir, "airplane_labels.csv"), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=air_header)
        w.writeheader(); w.writerows(a1_air_rows)
    # merge airplane rows into all/dataset_labels.csv (ship rows already written by generator)
    all_csv = os.path.join(all_dir, "dataset_labels.csv")
    with open(all_csv, newline='', encoding='utf-8') as f:
        rdr = csv.DictReader(f); ship_rows = list(rdr); all_header = rdr.fieldnames
    # ensure header compatibility
    assert all_header == air_header, f"{arm}: all csv header != airplane header\n{all_header}\n{air_header}"
    merged = a1_air_rows + ship_rows  # airplanes first (matches typical ordering); order not load-bearing
    with open(all_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=all_header)
        w.writeheader(); w.writerows(merged)
    n_air_png = len(glob.glob(os.path.join(plane_dir, "*.png")))
    n_ship_png = len(glob.glob(os.path.join(base, "ship", "*.png")))
    n_all_png = len(glob.glob(os.path.join(all_dir, "*.png")))
    print(f"[{arm}] airplane copied={n_copied} | airplane_png={n_air_png} ship_png={n_ship_png} "
          f"all_png={n_all_png} | all_csv rows={len(merged)} (air {len(a1_air_rows)} + ship {len(ship_rows)})")
    assert n_air_png == 90 and n_ship_png == 400 and n_all_png == 490, f"{arm} count check FAILED"
    print(f"[{arm}] COMPLETE 490 dataset OK")

print("\nFINALIZE DONE: arm2 now has 490 images (90 airplane byte-copied from arm1 + 400 ship).")
