from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

type EstimatorSchema = Annotated[
    PipelineSchema
    | RegressorSchema
    | ClassifierSchema
    | TransformerSchema
    | DecomposerSchema
    | FeatureUnionSchema
    | ColumnTransformerSchema
    | ImputerSchema,
    Field(discriminator="kind"),
]

EstimatorSchemaAdapter: TypeAdapter[EstimatorSchema] = TypeAdapter(EstimatorSchema)


def estimator_factory(data: dict) -> EstimatorSchema:
    return EstimatorSchemaAdapter.validate_python(data)


def build_estimator(data: dict | EstimatorSchema) -> BaseEstimator:
    estimator_schema = estimator_factory(data) if isinstance(data, dict) else data
    estimator = estimator_schema.build()
    estimator = estimator.set_output(transform="pandas")
    return estimator


class PipelineSchema(BaseModel):
    kind: Literal["pipeline"] = "pipeline"
    steps: list[EstimatorSchema]

    def build(self) -> Pipeline:
        return make_pipeline(*[step.build() for step in self.steps])


class FeatureUnionSchema(BaseModel):
    kind: Literal["feature_union"] = "feature_union"
    transformers: list[EstimatorSchema]

    def build(self) -> FeatureUnion:
        return FeatureUnion(*[transformer.build() for transformer in self.transformers])


class ColumnTransformerSchema(BaseModel):
    kind: Literal["column_transformer"] = "column_transformer"
    transformers: list[tuple[str, EstimatorSchema, list[str]]] = Field(
        description="""
        List of tuples of the form (name, transformer, feature_names) consisting of these parts:
        - name: the name of the transformer
        - transformer: the transformer to apply, can be a single transformer or a pipeline of transformers or any of the EstimatorSchema types
        - feature_names: the names of the features to apply the transformer to (needs to be a list even with a single feature)
        Examples:
        ```
        [["scale", {"kind": "transformer", "name": "StandardScaler"}, ["feature1", "feature2"]], ["encode", {"kind": "transformer", "name": "OneHotEncoder"}, ["feature3"]]]
        ```
        """
    )
    remainder: Literal["drop", "passthrough"] = "drop"

    def build(self) -> ColumnTransformer:
        transformers = [
            (
                name,
                transformer.build(),
                feature_names,
            )
            for name, transformer, feature_names in self.transformers
        ]
        return ColumnTransformer(transformers, remainder=self.remainder)


class RegressorSchema(BaseModel):
    kind: Literal["regressor"] = "regressor"
    name: Literal["LinearRegression", "Ridge", "Lasso", "ElasticNet"]

    def build(self) -> BaseEstimator:
        match self.name:
            case "LinearRegression":
                return LinearRegression()
            case "Ridge":
                return Ridge()
            case "Lasso":
                return Lasso()
            case "ElasticNet":
                return ElasticNet()


class ClassifierSchema(BaseModel):
    kind: Literal["classifier"] = "classifier"
    name: Literal[
        "LogisticRegression", "RandomForestClassifier", "KNeighborsClassifier", "SVC"
    ]

    def build(self) -> BaseEstimator:
        match self.name:
            case "LogisticRegression":
                return LogisticRegression()
            case "RandomForestClassifier":
                return RandomForestClassifier()
            case "KNeighborsClassifier":
                return KNeighborsClassifier()
            case "SVC":
                return SVC()

class TransformerSchema(BaseModel):
    kind: Literal["transformer"] = "transformer"
    name: Literal[
        "StandardScaler",
        "MinMaxScaler",
        "MaxAbsScaler",
        "RobustScaler",
        "PowerTransformer",
        "OneHotEncoder",
    ]

    def build(self) -> BaseEstimator:
        match self.name:
            case "StandardScaler":
                return StandardScaler()
            case "MinMaxScaler":
                return MinMaxScaler()
            case "MaxAbsScaler":
                return MaxAbsScaler()
            case "RobustScaler":
                return RobustScaler()
            case "PowerTransformer":
                return PowerTransformer()
            case "OneHotEncoder":
                return OneHotEncoder(sparse_output=False)


class DecomposerSchema(BaseModel):
    kind: Literal["decomposer"] = "decomposer"
    name: Literal["PCA", "FastICA"]
    n_components: int

    def build(self) -> BaseEstimator:
        match self.name:
            case "PCA":
                return PCA(n_components=self.n_components)
            case "FastICA":
                return FastICA(n_components=self.n_components)


class ImputerSchema(BaseModel):
    kind: Literal["imputer"] = "imputer"
    strategy: Literal["mean", "median", "most_frequent", "constant"]
    fill_value: float | str | None = None

    def build(self) -> SimpleImputer:
        imputer = SimpleImputer(strategy=self.strategy, fill_value=self.fill_value)
        return imputer
