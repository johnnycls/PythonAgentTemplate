from __future__ import annotations
import os

from dotenv import load_dotenv


def load_env() -> dict[str, str]:
    load_dotenv()
    return {k: v for k, v in os.environ.items() if v}


env = load_env()