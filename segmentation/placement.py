from typing import Iterable
import numpy as np
from PIL import Image
from PIL.Image import Image as ImageType
import skimage
from IPython.display import display

CANVAS_SIZE = (512, 512)
PlacementProposal = tuple[ImageType, tuple[int, int]]


def show(image: ImageType | np.ndarray) -> None:
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            if image.dtype == bool:
                image = image.astype(np.uint8) * 255
            else:
                image = image.astype(np.uint8)
        image = Image.fromarray(image)

    display(image)


def place_image(
    background: ImageType, foreground: ImageType, position: tuple[int, int] = (0, 0)
) -> ImageType | None:
    x, y = position
    if x < 0 or y < 0:
        raise ValueError("Position must be positive")
    if (
        x + foreground.width > background.width
        or y + foreground.height > background.height
    ):
        raise ValueError("Foreground does not fit in the background")
    background = background.copy()
    # paste respecting alpha
    background.alpha_composite(foreground, (x, y))
    return background


def expand_cutout_canvas(
    cutout: ImageType,
    position: tuple[int, int],
    canvas_size: tuple[int, int] = CANVAS_SIZE,
) -> ImageType:
    placed = Image.new("RGBA", canvas_size, color=(0, 0, 0, 0))
    placed.paste(cutout, tuple(position))
    return placed


def expand_randomly(
    cutout: ImageType,
    canvas_size: tuple[int, int] = CANVAS_SIZE,
    rng: np.random.Generator | None = None,
) -> ImageType:
    if rng is None:
        rng = np.random.default_rng()
    position = (
        rng.integers(0, canvas_size[0] - cutout.width),
        rng.integers(0, canvas_size[1] - cutout.height),
    )
    return expand_cutout_canvas(cutout, position, canvas_size)


def has_overlap(
    image: ImageType, mask: ImageType, position: tuple[int, int] = (0, 0)
) -> bool:
    # Convert to NumPy arrays
    image_np = to_mask(image)
    mask_np = to_mask(mask)

    # Get the dimensions of the mask
    mask_height, mask_width = mask_np.shape

    # Position to apply the mask on the image
    x_pos, y_pos = position

    # Extract the relevant area from the image
    image_area = image_np[y_pos : y_pos + mask_height, x_pos : x_pos + mask_width]

    # Check if the mask can be applied
    if image_area.shape[:2] != mask_np.shape:
        raise ValueError("Mask does not fit in the specified position of the image")

    # Apply mask and check if any pixels overlap
    return np.any(image_area[mask_np])


def propose_random_positions(
    image: ImageType,
    canvas_size: tuple[int, int],
    attempts: int = 100,
    rng: np.random.Generator | None = None,
) -> Iterable[PlacementProposal]:
    if rng is None:
        rng = np.random.default_rng()

    for i in range(attempts):
        position = (
            rng.integers(0, canvas_size[0] - image.width),
            rng.integers(0, canvas_size[1] - image.height),
        )
        yield image, position


def propose_grid(
    image: ImageType,
    canvas_size: tuple[int, int],
    grid_size: int | tuple[int, int],
):
    effective_size = (
        canvas_size[0] - image.width,
        canvas_size[1] - image.height,
    )
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            position = (
                int(x / (grid_size[0] - 1) * effective_size[0]),
                int(y / (grid_size[1] - 1) * effective_size[1]),
            )
            yield image, position


def to_mask(image: np.ndarray | ImageType) -> np.ndarray:
    """Returns a mask from the alpha channel as a boolean NumPy array (false = transparent)."""
    if isinstance(image, Image.Image):
        if image.mode == "L":
            return np.array(image) > 0
        else:
            assert image.mode == "RGBA", f"Image is not RGBA, it's {image.mode}"
            return np.array(image)[:, :, 3] > 0
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            return image.astype(bool)
        elif image.ndim == 3:
            assert image.shape[2] == 4, "Image is not RGBA"
            return image[:, :, 3] > 0
        else:
            raise ValueError(f"Image has invalid number of dimensions: {image.ndim}")
    else:
        raise ValueError(f"Image has invalid type: {type(image)}")


def get_tightness(
    bg_mask: np.ndarray | ImageType, fg_mask: np.ndarray | ImageType
) -> int:
    bg_mask = to_mask(bg_mask)
    fg_mask = to_mask(fg_mask)
    assert bg_mask.shape == fg_mask.shape, "Masks have different sizes"

    diameter = 10

    def pad(mask, value=0):
        return np.pad(
            mask,
            ((diameter, diameter), (diameter, diameter)),
            constant_values=value,
        )

    bg_mask = pad(bg_mask, value=True)
    # display(Image.fromarray(bg_mask.astype(np.uint8) * 255))
    fg_mask = pad(fg_mask, value=False)

    fg_expanded = skimage.morphology.dilation(
        fg_mask, skimage.morphology.disk(diameter)
    )
    fg_edge = fg_expanded & ~fg_mask
    edge_visible = bg_mask & fg_edge

    return np.sum(edge_visible)


def place_greedily(
    canvas: ImageType, proposals: Iterable[PlacementProposal]
) -> ImageType | None:
    for i, (cur_cutout, position) in enumerate(proposals):
        # position = rng.integers(0, 512, size=2)
        cur_cutout = expand_cutout_canvas(cur_cutout, position, canvas_size=canvas.size)

        if not has_overlap(canvas, cur_cutout):
            return place_image(canvas, cur_cutout)
    return None


def place_best_tightness(
    canvas: ImageType, proposals: Iterable[PlacementProposal]
) -> ImageType | None:
    best = None
    best_tightness = -np.inf

    for i, (cur_cutout, position) in enumerate(proposals):
        # position = rng.integers(0, 512, size=2)
        cur_cutout = expand_cutout_canvas(cur_cutout, position, canvas_size=canvas.size)

        if not has_overlap(canvas, cur_cutout):
            tightness = get_tightness(bg_mask=canvas, fg_mask=cur_cutout)
            if tightness > best_tightness:
                best = cur_cutout
                best_tightness = tightness

    return place_image(canvas, best) if best is not None else None
