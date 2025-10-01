from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Iterator, Literal, Self

from pydantic import BaseModel, Field
from pydantic_ai import ModelRetry

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from sklearn.base import BaseEstimator

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


def get_column_description(series: pd.Series) -> ColumnDescription | None:
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
        column_details = get_column_description(series)
        return cls(
            name=str(series.name),
            unique=series.nunique(),
            missing=series.isnull().sum(),
            column_details=column_details,
        )


class DataframeContext(BaseModel):
    length: int
    columns: list[ColumnContext]

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
    ) -> Self:
        return cls(
            length=len(df),
            columns=[ColumnContext.from_series(df[col]) for col in df],
        )


@dataclass
class _BaseData:
    df: pd.DataFrame
    name: str
    description: str
    context: DataframeContext

    def _base_context(self) -> str:
        return f"""
Name: {self.name}
Description: {self.description}
Context: {self.context.model_dump_json()}
        """


@dataclass
class StaticData(_BaseData):
    @classmethod
    def from_df(cls, df: pd.DataFrame, name: str, description: str) -> Self:
        return cls(
            df=df,
            name=name,
            description=description,
            context=DataframeContext.from_df(df),
        )

    def get_context(self) -> str:
        context_str = f"""
Dataframe uploaded by the user with the following metadata:
{self._base_context()}
        """
        return context_str


@dataclass
class SQLData(_BaseData):
    input_data: list[str]
    sql: str

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        name: str,
        description: str,
        input_data: list[str],
        sql: str,
    ) -> Self:
        return cls(
            df=df,
            context=DataframeContext.from_df(df),
            name=name,
            description=description,
            input_data=input_data,
            sql=sql,
        )

    def get_context(self) -> str:
        context_str = f"""
Dataframe derived from the following dataframes and using the following SQL query with this metadata:
Input dataframes: {", ".join(self.input_data)}
SQL query: {self.sql}
{self._base_context()}
        """
        return context_str


@dataclass
class ModelData(_BaseData):
    input_data: str
    model: BaseEstimator

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        name: str,
        description: str,
        input_data: str,
        model: BaseEstimator,
    ) -> Self:
        return cls(
            df=df,
            context=DataframeContext.from_df(df),
            name=name,
            description=description,
            input_data=input_data,
            model=model,
        )

    def get_context(self) -> str:
        context_str = f"""
Model fitted to the following dataframe with this metadata:
Input dataframe: {self.input_data}
Model: {str(self.model)}
The stored dataframe are the predictions of the model, with this metadata:
{self._base_context()}
        """
        return context_str


type StoredDataframe = StaticData | SQLData | ModelData


@dataclass
class DataStore:
    data: dict[str, StoredDataframe] = field(default_factory=dict)

    def __contains__(self, name: str) -> bool:
        return name in self.data

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[str]:
        return iter(self.data.keys())

    def items(self) -> Iterator[tuple[str, StoredDataframe]]:
        return iter(self.data.items())

    def delete(self, name: str) -> None:
        del self.data[name]

    def store(
        self,
        value: StoredDataframe,
        name: str,
    ) -> None:
        """Store the output in deps under the given name"""
        if name in self.data:
            raise ModelRetry(
                f"Error: {name} is already a valid dataframe. Check the previous messages and try again."
            )
        self.data[name] = value

    def store_static(self, df: pd.DataFrame, name: str, description: str) -> StaticData:
        data = StaticData.from_df(df, name, description)
        self.data[name] = data
        return data

    def store_sql(
        self,
        df: pd.DataFrame,
        name: str,
        description: str,
        input_data: list[str],
        sql: str,
    ) -> SQLData:
        data = SQLData.from_df(df, name, description, input_data, sql)
        self.data[name] = data
        return data

    def store_model(
        self,
        df: pd.DataFrame,
        name: str,
        description: str,
        input_data: str,
        model: BaseEstimator,
    ) -> ModelData:
        data = ModelData.from_df(df, name, description, input_data, model)
        self.data[name] = data
        return data

    def get(self, name: str) -> StoredDataframe:
        if name not in self.data:
            raise ModelRetry(
                f"Error: {name} is not a valid variable reference. Check the previous messages and try again."
            )
        return self.data[name]

    def get_data(self, name: str) -> pd.DataFrame:
        if name not in self.data:
            raise ModelRetry(
                f"Error: {name} is not a valid variable reference. Check the previous messages and try again."
            )
        return self.data[name].df

    def get_context(self, name: str) -> str:
        if name not in self.data:
            raise ModelRetry(
                f"Error: {name} is not a valid variable reference. Check the previous messages and try again."
            )
        return self.data[name].get_context()

    def keys(self) -> list[str]:
        return list(self.data.keys())

    def values(self) -> list[StoredDataframe]:
        return list(self.data.values())
