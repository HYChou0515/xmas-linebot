import asyncio
from io import BytesIO

import numpy as np
from PIL import Image
from linebot import LineBotApi
from linebot.models import TextMessage, TextSendMessage, ImageMessage, ImageSendMessage

from config import config
from service.imaging import ImagingService
from utils import get_written_bio

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
    user_id = event.source.user_id
    if isinstance(event.message, ImageMessage):
        message_content = line_bot_api.get_message_content(event.message.id)
        bio = BytesIO()
        for chunk in message_content.iter_content():
            bio.write(chunk)
        try:
            text, result, thumbnail = imaging_service.handle_image(bio)
            bio.seek(0)
            await imaging_service.upload_image(
                user_id,
                get_written_bio(
                    lambda b: Image.open(bio).save(
                        b, format="JPEG"
                    )
                ),
            )
            result_token = await imaging_service.upload_image(
                user_id,
                get_written_bio(
                    lambda bio: Image.fromarray(np.uint8(result * 255)).save(
                        bio, format="JPEG"
                    )
                ),
            )
            thumbnail_token = await imaging_service.upload_image(
                user_id,
                get_written_bio(
                    lambda bio: Image.fromarray(np.uint8(thumbnail * 255)).save(
                        bio, format="JPEG"
                    )
                ),
            )
            api_root = imaging_service.get_baseurl()
            api_root = await api_root
            message = [
                TextSendMessage(text=text),
                ImageSendMessage(
                    original_content_url=f"{api_root}/imagedb/{result_token}",
                    preview_image_url=f"{api_root}/imagedb/{thumbnail_token}",
                ),
            ]
            line_bot_api.reply_message(reply_token=reply_token, messages=message)
        except Exception:
            line_bot_api.reply_message(
                reply_token=reply_token, messages=TextSendMessage(text="哭哭 我壞掉了 可以再給我一次機會ㄇ")
            )
