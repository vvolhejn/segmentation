from PIL import Image
from PIL.Image import Image as ImageType
import numpy as np
import skimage.feature

# def get_shape_hash(image: ImageType, size: int) -> np.ndarray:
#     image = image.resize((size, size), Image.BILINEAR)
#     image = image.convert("L")
#     image = np.array(image).reshape(-1)
#     return image


def bool_array_to_bytes(bool_array: np.ndarray) -> np.ndarray:
    flattened = bool_array.flatten()
    # assert len(flattened) == 64, "Array must have exactly 64 elements"
    packed = np.packbits(flattened)
    # int64 = packed.view(np.uint64)
    # return int64[0]
    return packed


def get_shape_hash(image: ImageType, size: int = 8):
    assert image.mode == "RGBA"
    image = image.resize((size, size), Image.NEAREST)
    binary_mask = np.array(image)[:, :, 3] > 0
    return bool_array_to_bytes(binary_mask)


def get_shape_hash_float(image: ImageType, size: int = 8):
    assert image.mode == "RGBA"
    image = image.resize((size, size), Image.BILINEAR)
    return np.array(image)[:, :, 3].flatten() / 255


def get_edges_hash(image: ImageType, size: int = 8):
    image = image.resize((size * 4, size * 4))

    if image.mode == "RGBA":
        arr = np.array(image)[:, :, 3]
    else:
        image = image.convert("L")
        arr = np.array(image)

    edges = skimage.feature.canny(arr)
    edges = (
        Image.fromarray(edges.astype(np.uint8) * 255).resize(
            (size, size), Image.BILINEAR
        )
        # .resize((256, 256), Image.NEAREST)
    )
    edges_arr = (np.array(edges).flatten() / 255).astype(np.float32)
    return edges_arr
