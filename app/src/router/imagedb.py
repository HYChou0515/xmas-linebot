import tempfile

from fastapi import APIRouter, Depends

from service import ImagingService
from fastapi.responses import FileResponse

router = APIRouter()


@router.get("/{token}")
async def get_image(
    token: str,
    imaging_service: ImagingService = Depends(),
    tmp: tempfile.NamedTemporaryFile = Depends(),
):
    bio = imaging_service.get_image(token)
    tmp.write(bio.read())
    return FileResponse(tmp.name)
