import cv2
import skimage
import asyncio
import json
import tempfile
from io import BytesIO

import numpy as np
import httpx
from PIL import Image
from linebot import LineBotApi
from linebot.models import TextMessage, TextSendMessage, ImageMessage, ImageSendMessage

from config import config
from service.imaging import ImagingService
from utils import bio_to_image, get_written_bio, bio_to_base64

line_bot_api = LineBotApi(config.LINE_CHANNEL_ACCESS_TOKEN)


def handle_follow(event) -> None:
    """Event - User follow LINE Bot
    Args:
        event (LINE Event Object): Refer to https://developers.line.biz/en/reference/messaging-api/#follow-event
    """
    user_id = event.source.user_id
    print(f"User follow! user_id: {user_id}")


def handle_unfollow(event) -> None:
    """Event - User ban LINE Bot
    Args:
        event (LINE Event Object): Refer to https://developers.line.biz/en/reference/messaging-api/#unfollow-event
    """
    user_id = event.source.user_id
    print(f"User leave! user_id: {user_id}")


def handle_text_message(event) -> None:
    """Event - User sent message
    Args:
        event (LINE Event Object): Refer to https://developers.line.biz/en/reference/messaging-api/#message-event
    """
    reply_token = event.reply_token

    # Text message
    if isinstance(event.message, TextMessage):
        # Get user sent message
        user_message = event.message.text

        # Reply with same message
        messages = TextSendMessage(text=user_message)

        line_bot_api.reply_message(reply_token=reply_token, messages=messages)


async def handle_image_message(event) -> None:
    imaging_service = ImagingService()
    reply_token = event.reply_token
    if isinstance(event.message, ImageMessage):
        message_content = line_bot_api.get_message_content(event.message.id)
        bio = BytesIO()
        for chunk in message_content.iter_content():
            bio.write(chunk)
        image = bio_to_image(bio)
        image = imaging_service.white_balance(image)
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
        result = cv2.bitwise_and(image, image, mask=redmask)

        async with httpx.AsyncClient() as client:
            req = client.post(
                "http://192.168.1.233:8000/",
                data=json.dumps({"imageb64": bio_to_base64(bio)}),
                timeout=None,
            )
            task = asyncio.create_task(req)
            ret = await task

        message = []
        api_root = imaging_service.get_baseurl()
        if ret.status_code == 200:
            # b64 = json.loads(ret.text)['visimageb64']
            # bio = base64_to_bio(b64)
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
                objmask = np.zeros(image.shape[:2], dtype="uint8")
                objmask[objmask_idx[0], objmask_idx[1]] = 255
                _mask = cv2.bitwise_and(redmask, objmask)
                result = cv2.bitwise_and(image, image, mask=objmask)
                texts.append(
                    f"{mi+1}:{classname}({score:.02%}):{np.count_nonzero(_mask)/np.count_nonzero(objmask):.02%}"
                )
                images.append(result)
                sep = np.zeros((10, result.shape[1], 3), dtype="uint8") + 255
                images.append(sep)
            result = np.concatenate(images)
            text = "\r\n".join(texts)
            max_pixel = 3500000
            thumbnail_pixel = max_pixel // 100
            thumbnail_ratio = np.sqrt(
                thumbnail_pixel / (result.shape[0] * result.shape[1])
            )
            thumbnail = skimage.transform.resize(
                result,
                (
                    int(np.ceil(result.shape[0] * thumbnail_ratio)),
                    int(np.ceil(result.shape[1] * thumbnail_ratio)),
                ),
            )

            ratio = np.sqrt(max_pixel / (result.shape[0] * result.shape[1]))
            result = skimage.transform.resize(
                result,
                (
                    int(np.ceil(result.shape[0] * ratio)),
                    int(np.ceil(result.shape[1] * ratio)),
                ),
            )

            result_token = imaging_service.upload_image(
                get_written_bio(
                    lambda bio: Image.fromarray(np.uint8(result * 255)).save(
                        bio, format="JPEG"
                    )
                )
            )
            thumbnail_token = imaging_service.upload_image(
                get_written_bio(
                    lambda bio: Image.fromarray(np.uint8(thumbnail * 255)).save(
                        bio, format="JPEG"
                    )
                )
            )

            message.append(TextSendMessage(text=text))
            message.append(
                ImageSendMessage(
                    original_content_url=f"{api_root}/tea/{result_token}",
                    preview_image_url=f"{api_root}/tea/{thumbnail_token}",
                )
            )

            line_bot_api.reply_message(reply_token=reply_token, messages=message)
        else:
            line_bot_api.reply_message(
                reply_token=reply_token, messages=TextSendMessage(text="Error")
            )
