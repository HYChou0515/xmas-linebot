from typing import Optional, List
from fastapi import FastAPI
from pydantic import BaseModel
import tempfile
import base64
import io
from fastapi.logger import logger

import os
import sys
import skimage.io
import matplotlib.pyplot as plt


class Payload(BaseModel):
    imageb64: Optional[str]


class RetBody(BaseModel):
    visimageb64: Optional[str]
    rois: Optional[List[List[int]]]
    class_ids: Optional[List[int]]
    class_names: Optional[List[str]]
    scores: Optional[List[float]]
    masks: Optional[List[List[List[bool]]]]


# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils, visualize
import mrcnn.model as modellib

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    logger.info("Start download mask_rcnn_coco.h5...")
    utils.download_trained_weights(COCO_MODEL_PATH)


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = [
    "BG",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

model = None


def go_serve(payload):
    global model
    if model is None:
        # Create model object in inference mode.
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

        # Load weights trained on MS-COCO
        model.load_weights(COCO_MODEL_PATH, by_name=True)

    # Load a random image from the images folder
    with tempfile.NamedTemporaryFile("wb") as f:
        f.write(base64.b64decode(payload.imageb64.encode("utf-8")))
        image = skimage.io.imread(f.name)

    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]

    fig, ax = plt.subplots(1, figsize=(image.shape[1] / 96, image.shape[0] / 96))
    visualize.display_instances(
        image,
        r["rois"],
        r["masks"],
        r["class_ids"],
        class_names,
        r["scores"],
        ax=ax,
    )
    bio = io.BytesIO()
    fig.tight_layout()
    fig.savefig(bio, format="JPEG", dpi=96)
    bio.seek(0)
    body = RetBody(
        visimageb64=base64.b64encode(bio.read()).decode("utf-8"),
        rois=r["rois"].tolist(),
        class_ids=r["class_ids"].tolist(),
        class_names=[class_names[x] for x in r["class_ids"]],
        scores=r["scores"].tolist(),
        masks=r["masks"].tolist(),
    )
    return body


app = FastAPI()


@app.post("/", response_model=RetBody)
async def serve(payload: Payload):
    return go_serve(payload)
