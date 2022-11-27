import os
from .tables import *

DB_DIR = f"{os.path.dirname(__file__)}/.db"
IMAGE_PATH = f"{DB_DIR}/images"
SQLITE_DB_PATH = f"{DB_DIR}/sqlite.db"

if not os.path.exists(DB_DIR):
    os.makedirs(DB_DIR, exist_ok=True)
if not os.path.exists(IMAGE_PATH):
    os.makedirs(IMAGE_PATH, exist_ok=True)
