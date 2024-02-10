from PIL import Image
from PIL.Image import Image as ImageType


def make_more_transparent(image: ImageType, coef: float):
    assert image.mode == "RGBA"
    assert 0 <= coef <= 1
    # Split the image into RGB and Alpha channels
    r, g, b, alpha = image.split()

    # Multiply the alpha channel by 0.4
    mod_alpha = alpha.point(lambda p: p * coef)

    # Merge the channels back
    new_image = Image.merge("RGBA", (r, g, b, mod_alpha))

    return new_image


def show_comparison(original: ImageType, similar: ImageType, alpha: float = 0.1):
    assert original.mode == "RGBA"
    assert similar.mode == "RGBA"
    assert original.size == similar.size

    original_transparent = make_more_transparent(original, coef=alpha)

    similar = similar.copy()
    similar.alpha_composite(original_transparent)

    # Create a new image with twice the width
    new_image = Image.new("RGBA", (original.width * 2, original.height))
    new_image.paste(original, (0, 0))
    new_image.paste(similar, (original.width, 0))

    return new_image
