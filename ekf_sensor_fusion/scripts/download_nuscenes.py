"""Download the nuScenes v1.0-mini split (~4 GB) from the public AWS bucket.

nuScenes mini is hosted in the Registry of Open Data on AWS and is readable
anonymously — no account or token needed. This pulls it into data/nuscenes/.

    python scripts/download_nuscenes.py

On Colab this runs the same way; the tracks are then extracted with
kf_fusion.nuscenes_extract.
"""
from __future__ import annotations
import sys, tarfile, time
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.client import Config

BUCKET = "motional-nuscenes"
KEY = "public/v1.0/v1.0-mini.tgz"
DEST = Path(__file__).resolve().parents[1] / "data" / "nuscenes"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    if (DEST / "v1.0-mini").exists():
        print(f"already extracted at {DEST}")
        return
    s3 = boto3.client("s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED))
    size = s3.head_object(Bucket=BUCKET, Key=KEY)["ContentLength"]
    tgz = DEST / "v1.0-mini.tgz"
    print(f"downloading {size/1e9:.2f} GB -> {tgz}")
    seen = [0]; t0 = time.time()
    def cb(n):
        seen[0] += n
        if seen[0] % (500 * 1024 * 1024) < n:
            print(f"  {seen[0]/1e9:.2f}/{size/1e9:.2f} GB", flush=True)
    s3.download_file(BUCKET, KEY, str(tgz), Callback=cb,
                     Config=boto3.s3.transfer.TransferConfig(max_concurrency=8))
    print("extracting…")
    with tarfile.open(tgz) as tf:
        tf.extractall(DEST)
    tgz.unlink()
    print(f"done in {time.time()-t0:.0f}s -> {DEST}")


if __name__ == "__main__":
    main()
