from router import linebot, imagedb
from fastapi import FastAPI

app = FastAPI()

app.include_router(linebot)
app.include_router(imagedb, prefix="/imagedb")
