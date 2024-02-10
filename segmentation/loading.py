import itertools
from pathlib import Path
from typing import Iterable
from PIL import Image
from PIL.Image import Image as ImageType

DATA_DIR = Path(__file__).parent.parent / "data"


def images_in_dir(dir: Path):
    # Start with images from the dir itself
    yield from sorted(dir.glob("*.webp"))

    # Then go into subdirs. We sort them so that the order is deterministic
    # and we avoid sorting one big list
    subdirs = sorted(dir.glob("*/"))
    for subdir in subdirs:
        yield from images_in_dir(subdir)


def iterate_images(dir: Path, max_n_images: int | None = None) -> Iterable[ImageType]:
    for path in itertools.islice(images_in_dir(dir), max_n_images):
        yield Image.open(path)
