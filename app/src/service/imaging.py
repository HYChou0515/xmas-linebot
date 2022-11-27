import asyncio
import tempfile
from io import BytesIO
from typing import Optional
import json

import cv2
import httpx
import numpy as np
import requests
import skimage
from PIL import Image

from WBsRGB import WBsRGB
from database.dao.imaging import ImagingDao
from models import WhiteBalanceSetting, UseUpgradedModel, GamutMapping
from utils import get_written_bio, ImageConverter

image_db = {}


class ImagingService:
    def __init__(self):
        self.image_dao = ImagingDao()

    async def get_baseurl(self):
        # get ngrok api tunnels public url
        a = requests.get("http://localhost:4040/api/tunnels")
        return json.loads(a.text)["tunnels"][0]["public_url"]

    def get_white_balance_setting(self):
        return WhiteBalanceSetting.parse_obj(
            {
                "use_upgraded_model": UseUpgradedModel.OLD_MODEL,
                "gamut_mapping": GamutMapping.CLIPPING,
            }
        )

    def white_balance(self, image, wb_setting: Optional[WhiteBalanceSetting] = None):
        if wb_setting is None:
            wb_setting = self.get_white_balance_setting()
        wb_model = WBsRGB(
            gamut_mapping=wb_setting.gamut_mapping,
            upgraded=wb_setting.use_upgraded_model,
        )
        return wb_model.correctImage(image)

    async def upload_image(self, user_id, image_io: BytesIO):
        return await self.image_dao.create_image(user_id, image_io)

    async def get_image(self, token):
        return await self.image_dao.get_image(token)

    def get_red_mask(self, image):
        image = self.white_balance(image)
        bio = get_written_bio(
            lambda bio: Image.fromarray(np.uint8(image * 255)).save(bio, format="JPEG")
        )

        with tempfile.NamedTemporaryFile("wb") as f:
            f.write(bio.read())
            bio.seek(0)
            image = skimage.io.imread(f.name)

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Threshold of blue in HSV space
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        redmask = cv2.bitwise_or(mask1, mask2)

        return redmask

    async def do_mrcnn(self, client, image_b64):
        req = client.post(
            "http://192.168.1.233:8000/",
            data=json.dumps({"imageb64": image_b64}),
            timeout=None,
        )
        return await req

    def get_resized(self, image):
        max_pixel = 3500000
        thumbnail_pixel = max_pixel // 100
        total_pixel = image.shape[0] * image.shape[1]
        thumbnail_ratio = np.sqrt(thumbnail_pixel / total_pixel)
        ratio = np.sqrt(max_pixel / total_pixel)

        thumbnail = skimage.transform.resize(
            image,
            (
                int(np.ceil(image.shape[0] * thumbnail_ratio)),
                int(np.ceil(image.shape[1] * thumbnail_ratio)),
            ),
        )

        result = skimage.transform.resize(
            image,
            (
                int(np.ceil(image.shape[0] * ratio)),
                int(np.ceil(image.shape[1] * ratio)),
            ),
        )
        return result, thumbnail

    async def handle_image(self, user_id, image_io):
        image = ImageConverter(bio=image_io)
        async with httpx.AsyncClient() as client:
            task = asyncio.create_task(self.do_mrcnn(client, image.base64))
            redmask = self.get_red_mask(image.cv2_image)
            ret = await task

        if ret.status_code != 200:
            raise ValueError("mrcnn error")

        retobj = json.loads(ret.text)
        ret_masks = retobj["masks"]
        mask_list = set(ret_masks[2])
        texts = []
        images = []

        for mi in mask_list:
            classname = retobj["class_names"][mi]
            score = retobj["scores"][mi]
            masks = np.array(ret_masks)
            objmask_idx = masks[:2, masks[2] == mi]
            objmask = np.zeros(image.cv2_image.shape[:2], dtype="uint8")
            objmask[objmask_idx[0], objmask_idx[1]] = 255
            _mask = cv2.bitwise_and(redmask, objmask)
            result = cv2.bitwise_and(image.cv2_image, image.cv2_image, mask=objmask)
            texts.append(
                f"{mi + 1}:{classname}({score:.02%}):{np.count_nonzero(_mask) / np.count_nonzero(objmask):.02%}"
            )
            images.append(result)
            sep = np.zeros((10, result.shape[1], 3), dtype="uint8") + 255
            images.append(sep)

        result = np.concatenate(images)
        result, thumbnail = self.get_resized(result)
        result_token = self.upload_image(
            user_id,
            get_written_bio(
                lambda bio: Image.fromarray(np.uint8(result * 255)).save(
                    bio, format="JPEG"
                )
            ),
        )
        thumbnail_token = self.upload_image(
            user_id,
            get_written_bio(
                lambda bio: Image.fromarray(np.uint8(thumbnail * 255)).save(
                    bio, format="JPEG"
                )
            ),
        )
        text = "\r\n".join(texts)
        return text, await result_token, await thumbnail_token
