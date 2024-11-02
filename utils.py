import os

from dotenv import load_dotenv

load_dotenv()


def load_env(name: str) -> str:
    data = os.getenv(name)
    return data if data else ''


def load_env_as_number(name: str) -> int:
    data = load_env(name)
    return int(data) if data else 0
