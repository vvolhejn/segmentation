import diskcache

from PIL.Image import Image as ImageType
from PIL import Image
import numpy as np
import tqdm.auto

from segmentation.loading import DATA_DIR, images_in_dir
from segmentation.optimization import select_best
from segmentation.placement import to_mask

from segmentation.placement import place_image
from pydantic import BaseModel

ESCHER_CACHE_DIR = DATA_DIR / "escher" / "cache"
ESCHER_IMAGES_DIR = DATA_DIR / "escher" / "images"


def is_reasonable(image: ImageType) -> bool:
    return 0.1 < to_mask(image).mean() < 0.95


def set_alpha_to_1(image: ImageType):
    # set alpha from 255 to 1 and keep the 0 alpha at 0
    image = image.convert("RGBA")
    data = np.array(image)
    data[..., 3] = np.where(data[..., 3] > 0, 1, 0)
    image = Image.fromarray(data)
    return image


class TilingConfig(BaseModel):
    delta1: tuple[int, int]
    delta2: tuple[int, int]


def place_tiled(canvas: ImageType, reference_image: ImageType, config: TilingConfig):
    dx1, dy1 = config.delta1
    dx2, dy2 = config.delta2

    for i in range(-20, 20):
        for j in range(-20, 20):
            # skip if the image would be completely out of bounds
            x, y = (i * dx1 + j * dx2, i * dy1 + j * dy2)
            if x + reference_image.width < 0 or y + reference_image.height < 0:
                continue
            if x > canvas.width or y > canvas.height:
                continue

            canvas = place_image(
                canvas,
                reference_image,
                (x, y),
                allow_out_of_bounds=True,
            )

    return canvas


def score_tiling(
    canvas: ImageType,
    reference_image: ImageType,
    config: TilingConfig,
) -> list[float]:
    reference_image_1 = set_alpha_to_1(reference_image)
    canvas = place_tiled(canvas, reference_image_1, config)
    alpha = np.array(canvas)[..., 3]

    # The fraction of empty space is an affine combination of frac_exact
    # and frac_overlap. Linear combinations of these two will therefore also
    # implicitly penalize/encourage empty space.
    frac_exact = np.mean(alpha == 1)
    frac_overlap = np.mean(alpha > 1)

    return frac_exact - frac_overlap


def iter_deltas():
    for dx in range(0, 256, 32):
        for dy in range(0, 256, 32):
            yield (dx, dy)


def iter_configs():
    for delta1 in iter_deltas():
        for delta2 in iter_deltas():
            # check that delta2 is "on the right" of delta1 to reduce duplicates
            if delta1[0] * delta2[1] - delta1[1] * delta2[0] < 0:
                yield TilingConfig(delta1=delta1, delta2=delta2)


def main():
    cache = diskcache.Cache(directory=ESCHER_CACHE_DIR)
    canvas = Image.new("RGBA", (256, 256), 0)

    ESCHER_IMAGES_DIR.mkdir(exist_ok=True)

    n_skipped = 0

    progress_bar = tqdm.auto.tqdm(
        # islice(images_in_dir(DATA_DIR / "cutouts2"), 0, 200, 10)
        images_in_dir(DATA_DIR / "cutouts2")
    )

    for path in progress_bar:
        reference_image = Image.open(path)
        progress_bar.set_description(f"Processing {path.name}")
        progress_bar.set_postfix(n_skipped=n_skipped)

        if not is_reasonable(reference_image):
            n_skipped += 1
            continue

        best_config, best_score = select_best(
            tqdm.auto.tqdm(iter_configs(), leave=False),
            lambda x: score_tiling(canvas, reference_image, x),
        )

        cache[f"v1:{path}"] = {
            "score": best_score,
            "config": best_config.model_dump(mode="json"),
            "path": path,
            "image": place_tiled(canvas, reference_image, best_config),
        }

        place_tiled(canvas, reference_image, best_config).save(
            ESCHER_IMAGES_DIR / f"{path.stem}.png"
        )
        # print(path, best_score, best_config)
        # display(place_tiled(canvas, reference_image, best_config))
        # display(
        #     show_comparison(
        #         reference_image, place_tiled(canvas, reference_image, best_config)
        #     )
        # )

        # break


if __name__ == "__main__":
    main()
