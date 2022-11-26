import os
import pickle
from io import BytesIO
from typing import Optional
from uuid import uuid4
import json
import requests

from WBsRGB import WBsRGB
from models import WhiteBalanceSetting, UseUpgradedModel, GamutMapping
from utils import get_written_bio

image_db = {}


class ImagingService:
    def __init__(self):
        ...

    def get_baseurl(self):
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

    def get_image(self, token: str):
        global image_db
        if os.path.exists("imagedb.pkl"):
            image_db = pickle.load(open("imagedb.pkl", "rb"))
        return get_written_bio(lambda bio: bio.write(image_db[token]))

    def upload_image(self, image_io: BytesIO):
        token = str(uuid4())
        global image_db
        if os.path.exists("imagedb.pkl"):
            image_db = pickle.load(open("imagedb.pkl", "rb"))
        image_db[token] = image_io.read()
        pickle.dump(image_db, open("imagedb.pkl", "wb"))
        return token
