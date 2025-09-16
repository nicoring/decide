from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Iterator, Literal, Self

from pydantic import BaseModel, Field
from pydantic_ai import ModelRetry

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

type ColumnDescription = Annotated[
    NumberColumnDescription | CategoricalColumnDescription | DatetimeColumnDescription,
    Field(discriminator="kind"),
]


class NumberColumnDescription(BaseModel):
    kind: Literal["number"] = "number"
    min: float
    max: float
    mean: float
    median: float
    std: float

    @classmethod
    def from_series(cls, series: pd.Series) -> Self:
        return cls(
            min=series.min(),
            max=series.max(),
            mean=series.mean(),
            median=series.median(),
            std=series.std(),
        )


class CategoricalColumnDescription(BaseModel):
    kind: Literal["categorical"] = "categorical"
    most_common_value: str
    most_common_value_counts: dict[str, int]

    @classmethod
    def from_series(cls, series: pd.Series) -> Self:
        return cls(
            most_common_value=series.mode()[0],
            most_common_value_counts=series.value_counts().nlargest(20).to_dict(),
        )


class DatetimeColumnDescription(BaseModel):
    kind: Literal["datetime"] = "datetime"
    min: datetime
    max: datetime

    @classmethod
    def from_series(cls, series: pd.Series) -> Self:
        return cls(
            min=series.min(),
            max=series.max(),
        )


def get_column_context(series: pd.Series) -> ColumnDescription | None:
    if is_numeric_dtype(series.dtype):
        return NumberColumnDescription.from_series(series)
    elif is_datetime64_any_dtype(series.dtype):
        return DatetimeColumnDescription.from_series(series)
    elif isinstance(series.dtype, pd.CategoricalDtype):
        return CategoricalColumnDescription.from_series(series)
    else:
        return None


class ColumnContext(BaseModel):
    name: str
    unique: int
    missing: int
    column_details: ColumnDescription | None

    @classmethod
    def from_series(cls, series: pd.Series) -> Self:
        column_details = get_column_context(series)
        return cls(
            name=str(series.name),
            unique=series.nunique(),
            missing=series.isnull().sum(),
            column_details=column_details,
        )


class DataframeContext(BaseModel):
    name: str
    length: int
    columns: list[ColumnContext]
    description: str | None = None
    sql: str | None = None

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        name: str,
        sql: str | None = None,
        description: str | None = None,
    ) -> Self:
        return cls(
            name=name,
            length=len(df),
            columns=[ColumnContext.from_series(df[col]) for col in df],
            description=description,
            sql=sql,
        )


@dataclass
class DataStore:
    dataframes: dict[str, pd.DataFrame] = field(default_factory=dict)
    contexts: dict[str, DataframeContext] = field(default_factory=dict)

    def __contains__(self, name: str) -> bool:
        return name in self.dataframes

    def __len__(self) -> int:
        return len(self.dataframes)

    def __iter__(self) -> Iterator[str]:
        return iter(self.dataframes.keys())

    def items(self) -> Iterator[tuple[str, pd.DataFrame]]:
        return iter(self.dataframes.items())

    def delete(self, name: str) -> None:
        del self.dataframes[name]
        del self.contexts[name]

    def store(
        self,
        value: pd.DataFrame,
        name: str,
        sql: str | None = None,
        description: str | None = None,
    ) -> str:
        """Store the output in deps under the given name"""
        if name in self.dataframes:
            raise ModelRetry(
                f"Error: {name} is already a valid dataframe. Check the previous messages and try again."
            )
        self.dataframes[name] = value
        self.contexts[name] = DataframeContext.from_df(value, name, sql, description)
        return name

    def get(self, name: str) -> pd.DataFrame:
        if name not in self.dataframes:
            raise ModelRetry(
                f"Error: {name} is not a valid variable reference. Check the previous messages and try again."
            )
        return self.dataframes[name]

    def get_context(self, name: str) -> DataframeContext:
        if name not in self.contexts:
            raise ModelRetry(
                f"Error: {name} is not a valid variable reference. Check the previous messages and try again."
            )
        return self.contexts[name]

    def keys(self) -> list[str]:
        return list(self.dataframes.keys())

    def values(self) -> list[pd.DataFrame]:
        return list(self.dataframes.values())
