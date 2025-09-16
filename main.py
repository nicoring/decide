import logfire

from decide.app import build_page
from decide.config import settings

if settings.logfire_token is not None:
    logfire.configure(token=settings.logfire_token)
    logfire.instrument_pydantic_ai()


if __name__ == "__main__":
    build_page()
