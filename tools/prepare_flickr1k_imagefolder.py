import argparse
import ast
import csv
import json
import os
import shutil
import subprocess
import zipfile


def extract_images(zip_path, extract_dir):
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)


def locate_image(extracted_root, filename):
    direct = os.path.join(extracted_root, filename)
    if os.path.exists(direct):
        return direct
    for root, _, files in os.walk(extracted_root):
        if filename in files:
            return os.path.join(root, filename)
    return None


def clone_flickr_repo(clone_url, flickr_root):
    parent = os.path.dirname(flickr_root)
    if parent:
        os.makedirs(parent, exist_ok=True)
    print(f"Cloning Flickr30k repo from {clone_url} ...")
    subprocess.run(["git", "clone", clone_url, flickr_root], check=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare local Flickr30k into imagefolder format.")
    parser.add_argument("--flickr_root", type=str, default="flickr30k", help="Folder containing zip and csv.")
    parser.add_argument(
        "--clone_url",
        type=str,
        default="git@hf.co:datasets/nlphuji/flickr30k",
        help="HF dataset git URL used when local folder/files are missing.",
    )
    parser.add_argument(
        "--no_clone",
        action="store_true",
        help="Disable auto-clone and require existing local files.",
    )
    parser.add_argument("--images_zip", type=str, default="flickr30k-images.zip", help="Zip file name inside flickr_root.")
    parser.add_argument("--annotations_csv", type=str, default="flickr_annotations_30k.csv", help="CSV annotations file name.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Split to export.")
    parser.add_argument("--count", type=int, default=1000, help="Max samples to export.")
    parser.add_argument("--out_dir", type=str, default="data/flickr1k_local", help="Output imagefolder directory.")
    args = parser.parse_args()

    flickr_root = os.path.abspath(args.flickr_root)
    zip_path = os.path.join(flickr_root, args.images_zip)
    csv_path = os.path.join(flickr_root, args.annotations_csv)
    extracted_dir = os.path.join(flickr_root, "images")

    if (not os.path.exists(zip_path) or not os.path.exists(csv_path)) and not args.no_clone:
        if not os.path.exists(flickr_root):
            clone_flickr_repo(args.clone_url, flickr_root)
        else:
            print("Local flickr_root exists but required files are missing.")
            print("Skipping auto-clone because target directory already exists.")

    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"Missing zip file: {zip_path}. Ensure repo has LFS files (git lfs pull) or provide correct --flickr_root."
        )
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing annotations csv: {csv_path}")

    if not os.path.exists(extracted_dir) or not os.listdir(extracted_dir):
        print(f"Extracting images from {zip_path} ...")
        extract_images(zip_path, extracted_dir)

    os.makedirs(args.out_dir, exist_ok=True)
    meta_path = os.path.join(args.out_dir, "metadata.jsonl")

    written = 0
    with open(csv_path, "r", encoding="utf-8") as f, open(meta_path, "w", encoding="utf-8") as w:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") != args.split:
                continue

            filename = row["filename"].strip()
            src = locate_image(extracted_dir, filename)
            if not src:
                continue

            captions = ast.literal_eval(row["raw"])
            caption = captions[0].strip() if captions else ""

            dst_name = f"img_{written:05d}.jpg"
            dst_path = os.path.join(args.out_dir, dst_name)
            shutil.copy2(src, dst_path)
            w.write(json.dumps({"file_name": dst_name, "text": caption}, ensure_ascii=False) + "\n")

            written += 1
            if written >= args.count:
                break

    print(f"Saved {written} samples to: {args.out_dir}")
    print(f"Created: {meta_path}")
    print("Done.")

if __name__ == "__main__":
    main()