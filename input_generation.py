from PIL import Image, ImageDraw, ImageFont
import io
import cv2
import numpy as np
import enum


WIDTH = 224
HEIGHT = 224


class Color(enum.Enum):
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    BLUE = (0, 0, 255)
    RED = (255, 0, 0)

    def highlight(self):
        if self == Color.WHITE:
            return Color.BLACK.value
        else:
            return Color.WHITE.value


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    return np.asarray(img) / 255


def numpy_to_pil(img: np.ndarray) -> Image:
    return Image.fromarray(np.uint8(img * 255))


def overlay_text_on_image(image: Image, adversarial_text: str) -> Image:
    """
    Overlays a given text once on an image specified by an image path. Returns
    an image in memory in PIL format.
    """

    image_data = io.open(image_path, "rb").read()
    image = Image.open(io.BytesIO(image_data))
    img = cv2.resize(
        pil_to_numpy(image),
        dsize=(
            224,
            224,
        ),
        interpolation=cv2.INTER_LINEAR,
    )
    image = numpy_to_pil(img)

    # Draw the image
    font = ImageFont.truetype(
        "/usr/share/fonts/liberation/DejaVuSans.ttf", 20
    )
    draw = ImageDraw.Draw(image)

    for x in range(2):
        for y in range(2):
            draw.text(
                (20 + 100 * x, 20 + 100 * y),
                adversarial_text,
                Color.WHITE.value,
                font=font,
                stroke_width=2,
                stroke_fill=Color.WHITE.highlight(),
            )

    return image


def rasterize_text(text: str) -> Image:
    """
    Places black text in the middle of a 224x224 white image.
    """

    # Create empty 224x224 image with a white background
    image = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
    font = ImageFont.truetype(
        "/usr/share/fonts/liberation/DejaVuSans.ttf", 20
    )
    draw = ImageDraw.Draw(image)

    text_width, text_height = draw.textsize(text, font)

    if text_width > WIDTH or text_height > HEIGHT:
        raise Exception("Text is too large: " + text)

    draw.text(
        ((WIDTH - text_width) / 2, (HEIGHT - text_height) / 2),
        text,
        Color.BLACK.value,
        font=font,
        stroke_width=2,
        stroke_fill=Color.BLACK.highlight(),
    )

    return image
