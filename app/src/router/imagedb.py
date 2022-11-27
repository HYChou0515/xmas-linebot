import os
import tempfile

from fastapi import APIRouter, Depends, BackgroundTasks
from service import ImagingService
from fastapi.responses import FileResponse

router = APIRouter()


def get_named_temporary_file():
    return tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")


def delete_temporary_file(name):
    os.remove(name)


@router.get("/{token}")
async def tea(
    token: str,
    background_tasks: BackgroundTasks,
    imaging_service: ImagingService = Depends(),
    tmp: tempfile.NamedTemporaryFile = Depends(get_named_temporary_file),
) -> FileResponse:
    image_bytes = await imaging_service.get_image(token)
    background_tasks.add_task(lambda: delete_temporary_file(tmp.name))
    tmp.write(image_bytes)
    return FileResponse(tmp.name)
