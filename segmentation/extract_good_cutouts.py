import argparse
import functools
from pathlib import Path
import multiprocessing

import tqdm.auto

from segmentation.cutting import extract_cutouts


def save_cutouts_for_index(
    image_index: int, output_dir: Path, error_on_missing_file: bool = True
):
    try:
        for cutout_index, cutout in extract_cutouts(image_index):
            cutout.save(output_dir / f"{image_index:08d}_{cutout_index:05d}.webp")
    except FileNotFoundError:
        if error_on_missing_file:
            raise
        else:
            # The image indices are _almost_ continuous but not quite, some of them
            # are skipped. Just ignore them.
            return


def main(output_dir: Path, max_index: int):
    output_dir.mkdir(exist_ok=True)

    f = functools.partial(save_cutouts_for_index, output_dir=args.output_dir)

    with multiprocessing.Pool(8) as pool:
        pool.map(f, tqdm.auto.trange(1, max_index + 1), chunksize=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", "-o", type=Path, required=True)
    parser.add_argument("--max-index", type=int, default=100)
    args = parser.parse_args()

    main(args.output_dir, args.max_index)
