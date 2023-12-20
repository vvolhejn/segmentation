import argparse
import functools
from pathlib import Path
import multiprocessing
import sys
import re

import tqdm.auto

from segmentation import gcp
from segmentation.cutting import extract_cutouts


def get_image_index(filename: str) -> int:
    match = re.match(r"sa_([0-9]+).*", filename)
    if match is None:
        raise ValueError(f"Could not extract image index from {filename}")
    
    return int(match.groups()[0])


def save_cutouts_for_image(
    image_path: Path,
    output_dir: Path,
    error_on_missing_file: bool = True,
    gcp_prefix: str | None = None,
):
    try:
        for cutout_index, cutout in extract_cutouts(image_path):
            image_index = get_image_index(image_path.name)
            filename = f"{image_index:08d}_{cutout_index:05d}.webp"
            cutout.save(output_dir / filename)

            if gcp_prefix:
                gcp.upload_blob(output_dir / filename, gcp_prefix + filename)
    except FileNotFoundError:
        if error_on_missing_file:
            raise
        else:
            # The image indices are _almost_ continuous but not quite, some of them
            # are skipped. Just ignore them.
            return


def main(
    input_dir: Path,
    output_dir: Path,
    max_n_images: int,
    gcp_prefix: str | None = None,
    parallel: bool = True,
):
    output_dir.mkdir(exist_ok=True)

    f = functools.partial(
        save_cutouts_for_image, output_dir=output_dir, gcp_prefix=gcp_prefix
    )
    image_paths = tqdm.auto.tqdm(sorted(input_dir.glob("sa_*.jpg"))[:max_n_images])

    if not parallel:
        for path in image_paths:
            f(path)
    else:
        with multiprocessing.Pool(8) as pool:
            pool.map(f, image_paths, chunksize=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", "-i", type=Path, required=True)
    parser.add_argument("--output-dir", "-o", type=Path, required=True)
    parser.add_argument("--max-n-images", type=int, default=100)
    parser.add_argument("--gcp-prefix", type=str, default=None)
    parser.add_argument("--no-parallel", action="store_false", dest="parallel")
    args = parser.parse_args()

    gcp_prefix = args.gcp_prefix
    if gcp_prefix is not None:
        if not gcp_prefix.startswith("cutouts/v"):
            print("gcp-prefix should start with 'cutouts/v1' or a different version")
            sys.exit(1)

        if not gcp_prefix.endswith("/"):
            print("gcp-prefix should end with a slash, adding automatically")
            gcp_prefix += "/"

    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_n_images=args.max_n_images,
        gcp_prefix=gcp_prefix,
        parallel=args.parallel,
    )
