import os


class InvalidImage(Exception):
    def __init__(self) -> None:
        self.message = "Неверный формат изображения!"


def is_valid_img(image) -> bool:
    return image is None or not (len(image.shape) != 3 or image.shape[-1] != 3)


def path_exists(path: str = None) -> bool:
    if path and os.path.exists(path):
        return True
    return False


class FaceMissing(Exception):
    """Raised when face is not found in an image
    Attributes:
        message: (str) Exception message
    """

    def __init__(self) -> None:
        self.message = "Face not found!!"
