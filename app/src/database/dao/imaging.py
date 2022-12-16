import asyncio
from uuid import uuid4

from sqlalchemy import insert, select, update, delete

from database.dao import DBSessionMixin, ImageStorageMixin
import database as db
from models import UserConfig


class ImagingDao(DBSessionMixin, ImageStorageMixin):
    async def delete_user_config(self, user_id):
        await self.execute(delete(db.UserState).where(db.UserState.user_id == user_id))

    async def update_user_config(self, user_id, user_config: UserConfig):
        await self.execute(
            update(db.UserState)
            .where(db.UserState.user_id == user_id)
            .values({db.UserState.state.name: user_config.json()})
        )

    async def get_user_config(self, user_id):
        user_config = await self.select_df(
            select(db.UserState.state).where(db.UserState.user_id == user_id)
        )
        if user_config is None or user_config.empty:
            raise ValueError("Cannot create user config")
        user_config = user_config.iloc[0, 0]
        user_config = UserConfig.parse_raw(user_config)
        return user_config

    async def create_user_config(self, user_id, user_config: UserConfig):
        await self.execute(
            insert(db.UserState).values(
                {
                    db.UserState.user_id.name: user_id,
                    db.UserState.state.name: user_config.json(),
                }
            )
        )

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
