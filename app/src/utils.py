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
