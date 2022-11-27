import os

import sqlalchemy

from database import SQLITE_DB_PATH, IMAGE_PATH
from database.tables import Base
import pandas as pd

db_client = None


def get_db():
    global db_client
    if db_client is None:
        db_client = sqlalchemy.create_engine(
            f"sqlite:///{SQLITE_DB_PATH}", isolation_level=None
        )
        db_client.execute("pragma journal_mode=wal;")
        Base.metadata.create_all(db_client)
    return db_client


class DBSessionMixin:
    def __init__(self):
        super().__init__()
        self.db = get_db()

    async def select_df(self, sql):
        return pd.read_sql(sql, self.db)

    async def execute(self, sql):
        return self.db.execute(sql)


class ImageStorageMixin:
    async def save_image(self, image, name):
        fpath = f"{IMAGE_PATH}/{name}"
        if os.path.exists(fpath):
            raise FileExistsError(fpath)
        with open(fpath, "wb") as f:
            f.write(image.read())

    async def load_image(self, name):
        fpath = f"{IMAGE_PATH}/{name}"
        if not os.path.exists(fpath):
            raise FileNotFoundError(fpath)
        with open(fpath, "rb") as f:
            return f.read()
