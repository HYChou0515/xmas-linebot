from collections import Counter
from loguru import logger
from io import BytesIO
from typing import Optional
import json
from joblib import Parallel, delayed

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

    @staticmethod
    def get_red_mask(image):
        # image = self.white_balance(image)
        # image = np.array(image*255, dtype='uint8')
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Threshold of red in HSV space
        h = hsv[:, :, 0]
        s = hsv[:, :, 1] / 255
        v = hsv[:, :, 2] / 255
        # bound = np.nonzero(((s + 0.16) * (v + 0.05) >= 0.08) & ((h + 30) % 180 <= 40))
        bound = np.nonzero(((s + 0.16) * (v + 0.05) >= 0.23) & ((h + 30) % 180 <= 40))
        redmask = np.zeros(hsv.shape[:2], dtype="uint8")
        redmask[bound] = 255
        # Image.fromarray(cv2.bitwise_and(image, image, mask=(rm))).show()
        # Image.fromarray(image).show()
        #
        # kernel = np.zeros((19, 19), dtype='uint8')
        # for i in range(10):
        #     kernel[i,i:19-i] = 1
        #
        # rm = redmask.copy()
        # for i in range(1):
        #     a = np.zeros_like(rm)
        #     for i in range(4):
        #         aa = sig.convolve2d(rm, kernel, mode="same")
        #         aa[aa<kernel.sum()] = 0
        #         aa[aa>=kernel.sum()] = 1
        #         a = cv2.bitwise_or(aa, a)
        #         kernel = np.rot90(kernel)
        #     rm = cv2.bitwise_and(rm, a)
        _, labels = cv2.connectedComponents(redmask)
        cnt = Counter(labels.reshape(-1))

        # cnt = set(k for k in )
        def f(_x):
            x, y = _x
            if x == 0:
                if cnt.get(y) < 200:
                    return x
                return 0
            if cnt.get(y) < 400:
                return 0
            return x

        ff = np.vectorize(f, signature="(n)->()")
        rm = ff(np.stack([redmask, labels], -1))
        rm = rm.astype("uint8")
        logger.info("redmask done")
        return rm

    @staticmethod
    def do_mrcnn(image_b64):
        req = httpx.post(
            # "http://127.0.0.1:8000/",
            "http://192.168.1.233:8000/",
            data=json.dumps({"imageb64": image_b64}),
            timeout=None,
        )
        logger.info("mrcnn done")
        return req

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

    def handle_image(self, image_io):
        image = ImageConverter(bio=image_io)
        ret, redmask = Parallel(n_jobs=2, prefer="threads")(
            (
                delayed(self.do_mrcnn)(image.base64),
                delayed(self.get_red_mask)(image.cv2_image),
            )
        )

        if ret.status_code != 200:
            raise ValueError("mrcnn error")

        retobj = json.loads(ret.text)
        ret_masks = retobj["masks"]
        masks = np.array(ret_masks)
        mask_list = set(ret_masks[2])
        # find largest mask
        ms, mi = None, None
        for i in mask_list:
            s = sum(masks[2] == i)
            if ms is None or ms < s:
                ms = s
                mi = i

        classname = retobj["class_names"][mi]
        score = retobj["scores"][mi]
        objmask_idx = masks[:2, masks[2] == mi]
        objmask = np.zeros(image.cv2_image.shape[:2], dtype="uint8")
        objmask[objmask_idx[0], objmask_idx[1]] = 255
        _mask = cv2.bitwise_and(redmask, objmask)
        result = cv2.bitwise_and(image.cv2_image, image.cv2_image, mask=objmask)
        text = (
            f"{classname}"
            f"\ntotal:{np.count_nonzero(objmask)} pixels"
            f"\nred:{np.count_nonzero(_mask)} pixels "
            f"({np.count_nonzero(_mask) / np.count_nonzero(objmask):.02%})"
        )
        _, thresh = cv2.threshold(_mask, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 0), 3)

        strips = np.zeros_like(image.cv2_image)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if 13 < (i + j) % 40 <= 23:
                    strips[i, j] = [0, 255, 0]
        strips = cv2.bitwise_and(strips, strips, mask=_mask)
        # colored_redmask = np.broadcast_to(np.array([0, 255, 0], dtype='uint8'), result.shape)
        # colored_redmask = cv2.bitwise_and(colored_redmask, colored_redmask, mask=_mask)
        result = cv2.addWeighted(result, 1.0, strips, 0.5, 0)

        result, thumbnail = self.get_resized(result)
        return text, result, thumbnail
