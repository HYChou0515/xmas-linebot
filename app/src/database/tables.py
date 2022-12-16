from sqlalchemy import Column, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class UserState(Base):
    __tablename__ = "user_state"

    user_id = Column(String, primary_key=True)
    state = Column(String)


class UserImage(Base):
    __tablename__ = "user_image"

    image_id = Column(String, primary_key=True)
    user_id = Column(String)
