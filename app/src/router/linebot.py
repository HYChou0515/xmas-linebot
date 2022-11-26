import asyncio
import os
import tempfile

from fastapi import APIRouter, HTTPException, Request, Depends, BackgroundTasks
from linebot import WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    TextMessage,
    FollowEvent,
    UnfollowEvent,
    MessageEvent,
    ImageMessage,
)

from service import ImagingService
from fastapi.responses import FileResponse

import service
from config import config

router = APIRouter()
handler = WebhookHandler(config.LINE_CHANNEL_SECRET)

loop = asyncio.get_event_loop()


@router.post("/")
@router.post("/callback")
async def callback(request: Request) -> str:
    """LINE Bot webhook callback
    Args:
        request (Request): Request Object.
    Raises:
        HTTPException: Invalid Signature Error
    Returns:
        str: OK
    """
    signature = request.headers["X-Line-Signature"]
    body = await request.body()

    # handle webhook body
    try:
        handler.handle(body.decode(), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Missing Parameter")
    return "OK"


def get_named_temporary_file():
    return tempfile.NamedTemporaryFile(delete=False)


def delete_temporary_file(name):
    os.remove(name)


@router.get("/tea/{token}")
async def tea(
    token: str,
    background_tasks: BackgroundTasks,
    imaging_service: ImagingService = Depends(),
    tmp: tempfile.NamedTemporaryFile = Depends(get_named_temporary_file),
) -> FileResponse:
    bio = imaging_service.get_image(token)
    background_tasks.add_task(lambda: delete_temporary_file(tmp.name))
    tmp.write(bio.read())
    return FileResponse(tmp.name)


@handler.add(FollowEvent)
def handle_follow(event) -> None:
    """Event - User follow LINE Bot
    Args:
        event (LINE Event Object): Refer to https://developers.line.biz/en/reference/messaging-api/#follow-event
    """
    service.handle_follow(event=event)


@handler.add(UnfollowEvent)
def handle_unfollow(event) -> None:
    """Event - User ban LINE Bot
    Args:
        event (LINE Event Object): Refer to https://developers.line.biz/en/reference/messaging-api/#unfollow-event
    """
    service.handle_unfollow(event=event)


@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event) -> None:
    """Event - User sent message
    Args:
        event (LINE Event Object): Refer to https://developers.line.biz/en/reference/messaging-api/#message-event
    """
    service.handle_text_message(event=event)


@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event) -> None:
    """Event - User sent message
    Args:
        event (LINE Event Object): Refer to https://developers.line.biz/en/reference/messaging-api/#message-event
    """
    asyncio.create_task(service.handle_image_message(event=event))
