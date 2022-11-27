import asyncio
from io import BytesIO

from linebot import LineBotApi
from linebot.models import TextMessage, TextSendMessage, ImageMessage, ImageSendMessage

from config import config
from service.imaging import ImagingService

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
            result = imaging_service.handle_image(user_id, bio)
            api_root = imaging_service.get_baseurl()
            text, result_token, thumbnail_token = await result
            api_root = await api_root
            message = [
                TextSendMessage(text=text),
                ImageSendMessage(
                    original_content_url=f"{api_root}/imagedb/{result_token}",
                    preview_image_url=f"{api_root}/imagedb/{thumbnail_token}",
                ),
            ]
            line_bot_api.reply_message(reply_token=reply_token, messages=message)
        except ValueError:
            line_bot_api.reply_message(
                reply_token=reply_token, messages=TextSendMessage(text="Error")
            )
