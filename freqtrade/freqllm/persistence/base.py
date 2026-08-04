"""FreqLLM ORM base class — separate from freqtrade's ModelBase to use an independent DB."""

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase


class FreqLLMModelBase(DeclarativeBase):
    """Base class for all FreqLLM ORM models.

    Uses its own DeclarativeBase so that FreqLLM tables live in a separate
    SQLite database from freqtrade's core trade tables.
    """

    metadata = MetaData()
