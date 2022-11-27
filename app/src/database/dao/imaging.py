import asyncio
from uuid import uuid4

from sqlalchemy import insert

from database.dao import DBSessionMixin, ImageStorageMixin
import database as db


class ImagingDao(DBSessionMixin, ImageStorageMixin):
    async def create_image(self, user_id, image_io):
        token = str(uuid4())
        await asyncio.gather(
            self.save_image(image_io, token),
            self.execute(
                insert(db.UserImage).values(
                    {
                        db.UserImage.user_id.name: user_id,
                        db.UserImage.image_id.name: token,
                    }
                )
            ),
        )
        return token

    async def get_image(self, token):
        return await self.load_image(token)
