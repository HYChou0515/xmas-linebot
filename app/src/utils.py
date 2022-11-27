import base64
from io import BytesIO
import numpy as np
from PIL import Image


def bio_to_image(bio: BytesIO, to="cv2"):
    if to == "cv2":
        image = Image.open(bio).convert("RGB")
        image = np.array(image)
        return image.copy()


def get_written_bio(func):
    bio = BytesIO()
    func(bio)
    bio.seek(0)
    return bio


def bio_to_base64(bio):
    b64 = base64.b64encode(bio.read()).decode("utf-8")
    bio.seek(0)
    return b64


def base64_to_bio(b64):
    bio = get_written_bio(lambda bio: bio.write(base64.b64decode(b64.encode("utf-8"))))
    bio.seek(0)
    return bio


class ImageConverter:
    def __init__(self, *, bio=None):
        self._base64 = None
        self._cv2_image = None
        if bio is not None:
            bio.seek(0)
            self._bio = bio

    @property
    def base64(self):
        if self._base64 is None:
            self._base64 = bio_to_base64(self._bio)
        return self._base64

    @property
    def cv2_image(self):
        if self._cv2_image is None:
            image = Image.open(self._bio).convert("RGB")
            image = np.array(image)
            self._cv2_image = image.copy()
        return self._cv2_image
