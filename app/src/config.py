import os
import attrs as at


@at.define
class Config:
    LINE_CHANNEL_ACCESS_TOKEN: str = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
    LINE_CHANNEL_SECRET: str = os.environ.get("LINE_CHANNEL_SECRET")


config = Config()
