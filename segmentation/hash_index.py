from pathlib import Path
from PIL import Image
import numpy as np
import tqdm.auto
import faiss

from segmentation.loading import DATA_DIR, iterate_images


def get_example_image():
    rng = np.random.default_rng(0)
    return Image.fromarray((rng.random((256, 256, 4)) * 256).astype(np.uint8))


def check_hash_function(hash_function) -> int:
    """Returns the number of elements the hash function returns."""
    example_image = get_example_image()
    example_hash = hash_function(example_image)
    assert isinstance(example_hash, np.ndarray)
    assert (
        example_hash.dtype == np.float32
    ), f"Hash must be float32, got {example_hash.dtype}"
    assert example_hash.ndim == 1

    if len(example_hash) > 256:
        raise ValueError(
            "For 1M images, a hash of more than 256 elements would "
            f"be over 2GB in total. Got {len(example_hash)} elements."
        )

    return len(example_hash)


class HashIndex:
    def __init__(
        self,
        hash_function: callable,
        cache_file: Path,
        max_n_images: int | None = None,
    ):
        self.hash_function = hash_function
        self.cache_file = cache_file
        self.max_n_images = max_n_images

        hash_size = check_hash_function(hash_function)
        self.hash_size = hash_size

        if cache_file.exists():
            hashes = np.load(cache_file)
            self.make_index(hashes)
        else:
            self.make_index()

    def make_index(self, hashes: np.ndarray | None = None) -> None:
        if hashes is None:
            hashes = self.compute_hashes()

        print(hashes.shape)

        self.index = faiss.IndexFlatL2(self.hash_size)
        self.index.add(hashes)

    def compute_hashes(self):
        hashes = []

        for image in tqdm.auto.tqdm(
            iterate_images(DATA_DIR / "cutouts2", max_n_images=self.max_n_images),
            total=self.max_n_images,
        ):
            h = self.hash_function(image)
            assert len(h) == self.hash_size
            hashes.append(h)

        return np.array(hashes)

    def get_closest(self, image: Image.Image, n: int = 10):
        h = self.hash_function(image)
        distances, indices = self.index.search(h[np.newaxis, :], n)
        return indices[0]
