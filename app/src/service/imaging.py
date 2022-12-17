from collections import Counter
from functools import lru_cache

import pandas as pd
from loguru import logger
from io import BytesIO
from typing import Optional
import json
from joblib import Parallel, delayed
import importlib
import resources
import cv2
import httpx
import numpy as np
import requests
import skimage

from WBsRGB import WBsRGB
from database.dao.imaging import ImagingDao
from models import (
    WhiteBalanceSetting,
    UseUpgradedModel,
    GamutMapping,
    UserConfig,
    ObjectDetectionOptions,
)
from utils import ImageConverter

image_db = {}


class ImagingService:
    DEFAULT_USER_CONFIG = UserConfig.parse_obj(
        {
            "object_detection": "mrcnn",
        }
    )

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

    async def delete_user_config(self, user_id):
        await self.image_dao.delete_user_config(user_id)

    async def update_user_config(
        self, user_id, *, object_detection: ObjectDetectionOptions
    ):
        user_config = await self.get_or_create_user_config(user_id)
        user_config.object_detection = object_detection
        await self.image_dao.update_user_config(user_id, user_config)
        user_config = await self.get_or_create_user_config(user_id)
        return user_config

    async def get_or_create_user_config(self, user_id):
        try:
            return await self.image_dao.get_user_config(user_id)
        except ValueError:
            await self.image_dao.create_user_config(user_id, self.DEFAULT_USER_CONFIG)
            return await self.image_dao.get_user_config(user_id)

    @staticmethod
    def get_color_mask(image, color):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Threshold of red in HSV space
        h = hsv[:, :, 0]
        s = hsv[:, :, 1] / 255
        v = hsv[:, :, 2] / 255
        colormask = np.zeros(hsv.shape[:2], dtype="uint8")
        if "red" in color:
            # bound = np.nonzero(((s + 0.16) * (v + 0.05) >= 0.08) & ((h + 30) % 180 <= 40))
            bound = np.nonzero(
                ((s + 0.16) * (v + 0.05) >= 0.23) & ((h + 30) % 180 <= 40)
            )
            colormask[bound] = 255
        if "green" in color:
            with importlib.resources.files(resources).joinpath("green.csv").open(
                "r"
            ) as f:
                df = pd.read_csv(f)
                df = df.fillna(method="ffill")
                df.iloc[:, 1:4] /= 100
            s_range1 = list(
                zip(*(df.groupby("h")["s"].min().reset_index().itertuples(index=False)))
            )
            s_range2 = list(
                zip(*(df.groupby("h")["s"].max().reset_index().itertuples(index=False)))
            )
            all_h = df["h"].drop_duplicates().sort_values().to_numpy()
            h_dict = {_h: df[df["h"] == _h] for _h in all_h}

            @lru_cache(maxsize=None)
            def is_in_green(_h, _s, _v):
                if not (all_h[0] <= _h <= all_h[-1]):
                    return False
                s1 = np.interp(_h, s_range1[0], s_range1[1])
                s2 = np.interp(_h, s_range2[0], s_range2[1])
                if not (s1 <= _s <= s2):
                    return False
                if _h in h_dict:
                    hhh = h_dict[_h]
                else:
                    i = np.searchsorted(all_h, _h)
                    hl, hr = all_h[i - 1], all_h[i]
                    ratio = (_h - hl) / (hr - hl)
                    hhh = h_dict[hl].reset_index(drop=True) * ratio + h_dict[
                        hr
                    ].reset_index(drop=True) * (1 - ratio)
                v1 = hhh["v1"]
                v2 = hhh["v2"]
                ss = hhh["s"]
                np.interp(_s, ss, v2)
                if not (np.interp(_s, ss, v1) <= _v <= np.interp(_s, ss, v2)):
                    return False
                return True

            bound = np.zeros_like(s)
            for i in range(bound.shape[0]):
                for j in range(bound.shape[1]):
                    bound[i, j] = is_in_green(h[i, j] * 2.0, s[i, j], v[i, j])
            # bound = np.nonzero(((s + 0.16) * (v + 0.05) >= 0.08) & ((h + 30) % 180 <= 40))
            bound = np.nonzero(bound)
            colormask[bound] = 255

        _, labels = cv2.connectedComponents(colormask)
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
        rm = ff(np.stack([colormask, labels], -1))
        rm = rm.astype("uint8")
        logger.info(f"{color} mask done")
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

    def original_figure_mode(self, image_io):
        image = ImageConverter(bio=image_io)
        result = image.cv2_image.copy()
        # _, result = self.get_resized(result)
        # result = (result*255).astype('uint8')
        # self.get_color_mask(result, 'red')

        (redmask,) = Parallel(n_jobs=1, prefer="threads")(
            (delayed(self.get_color_mask)(result, ("red", "green")),)
        )
        _mask = redmask.copy()
        ttl_size = result.shape[0] * result.shape[1]
        result = result.copy()
        text = (
            f"\ntotal:{ttl_size} pixels"
            f"\nred/green:{np.count_nonzero(_mask)} pixels "
            f"({np.count_nonzero(_mask) / ttl_size:.02%})"
        )
        _, thresh = cv2.threshold(_mask, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 0), 3)
        strips = np.zeros_like(result)
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

    def mrcnn_mode(self, image_io):
        image = ImageConverter(bio=image_io)
        ret, redmask = Parallel(n_jobs=2, prefer="threads")(
            (
                delayed(self.do_mrcnn)(image.base64),
                delayed(self.get_color_mask)(image.cv2_image, ("red", "green")),
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

    def handle_image(self, image_io, *, object_detection=ObjectDetectionOptions.MRCNN):
        if object_detection == ObjectDetectionOptions.MRCNN:
            return self.mrcnn_mode(image_io)
        if object_detection == ObjectDetectionOptions.NONE:
            return self.original_figure_mode(image_io)
