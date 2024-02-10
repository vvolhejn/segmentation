import functools
import json
import multiprocessing
from typing import Iterator
from PIL import Image
from PIL.Image import Image as ImageType
from pathlib import Path
import numpy as np
import skimage.measure


import pycocotools.mask
import tqdm.auto

Annotation = dict


def load(image_path: Path) -> tuple[ImageType, list[Annotation]]:
    image = Image.open(image_path)

    with open(image_path.with_suffix(".json")) as f:
        data = json.load(f)

    return image, data["annotations"]


def is_good_cutout(image: ImageType) -> bool:
    is_big_enough = image.width >= 256 or image.height >= 256
    if not is_big_enough:
        return False

    if any(get_cut_off_sides(image)):
        return False

    if not is_connected(image):
        return False

    return True


def extract_cutouts(image_path: Path) -> Iterator[tuple[int, ImageType]]:
    image, annotations = load(image_path)

    for i, annotation in enumerate(annotations):
        mask = pycocotools.mask.decode(annotation["segmentation"])

        image.putalpha(Image.fromarray(mask * 255, mode="L"))
        cutout = image.crop(image.getbbox())

        if not is_good_cutout(cutout):
            continue

        cutout = to_standard_cutout(cutout)
        yield i, cutout


def save_cutouts_for_index(image_index, output_dir: Path):
    # We already have the cutouts for this image
    if list(output_dir.glob(f"{image_index}_*")):
        return

    found = False

    for crop_index, crop in enumerate(extract_cutouts(image_index)):
        crop.save(output_dir / f"{image_index}_{crop_index}.png")
        found = True

    if not found:
        (output_dir / f"{image_index}_no_cutouts.txt").touch()


def save_cutouts(output_dir: Path, max_n_images: int = 1000, parallel: bool = True):
    f = functools.partial(save_cutouts_for_index, output_dir=output_dir)

    output_dir.mkdir(exist_ok=True)

    if parallel:
        with multiprocessing.Pool() as pool:
            pool.map(f, tqdm.auto.trange(1, max_n_images), chunksize=1)
    else:
        for i in tqdm.auto.trange(1, max_n_images):
            f(i)


def to_standard_cutout(image: ImageType) -> ImageType:
    size = 256
    # scale image to (256, 256), preserving aspect ratio and centering
    if image.width > image.height:
        image = image.resize((size, int(image.height / image.width * size)))
    else:
        image = image.resize((int(image.width / image.height * size), size))

    image = image.crop(
        (
            (image.width - size) // 2,
            (image.height - size) // 2,
            (image.width + size) // 2,
            (image.height + size) // 2,
        )
    )

    return image


def is_connected(image: ImageType) -> bool:
    assert image.mode == "RGBA", "Image is not RGBA"
    n_components = skimage.measure.label(np.array(image)[:, :, 3] > 0).max()
    return n_components == 1


def is_top_cut_off(image: ImageType) -> bool:
    assert image.mode == "RGBA", "Image is not RGBA"

    image = image.crop(image.getbbox())

    is_filled = np.array(image)[:, :, -1] > 0

    # Measure the width of the edge in the first few rows. If the width
    # of the very first row is similar to the next rows, it means that
    # the top of the image is cut off. Otherwise, we'd expect the width
    # of the image to grow as we go down the image.
    n_rows = 5
    edge_width = np.sum(is_filled[:n_rows, :]) / n_rows
    top_row_width = np.sum(is_filled[0, :])

    # If the edge is thin, it should look ok even if it is cut off.
    if edge_width < 0.2 * image.width:
        return False

    return top_row_width >= 0.85 * edge_width


def get_cut_off_sides(image: ImageType) -> tuple[bool, bool, bool, bool]:
    """Computes which sides of the image are cut off.

    Returns:
        a tuple of booleans in the order top, right, bottom, left, saying
        whether the corresponding side is cut off.
    """
    # crop image to content
    image = image.crop(image.getbbox())

    res = []

    for i in range(4):
        res.append(is_top_cut_off(image))

        # Rotates 90 deg counterclockwise, which means that the order
        # of the sides will be top, right, bottom, left
        image = image.transpose(Image.ROTATE_90)

    return tuple(res)
