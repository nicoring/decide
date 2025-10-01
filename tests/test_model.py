import pytest

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from decide.model import (
    ClassifierSchema,
    PipelineSchema,
    RegressorSchema,
    TransformerSchema,
    build_estimator,
    estimator_factory,
)


@pytest.mark.parametrize(
    "data,estimator_type",
    [
        (
            {
                "kind": "regressor",
                "name": "LinearRegression",
            },
            RegressorSchema,
        ),
        (
            {
                "kind": "classifier",
                "name": "LogisticRegression",
            },
            ClassifierSchema,
        ),
        (
            {
                "kind": "transformer",
                "name": "StandardScaler",
            },
            TransformerSchema,
        ),
    ],
)
def test_base_estimator(data, estimator_type):
    estimator = estimator_factory(data)
    assert estimator.kind == data["kind"]
    assert isinstance(estimator, estimator_type)
    assert estimator.model_dump() == data


@pytest.mark.parametrize(
    "data,estimator_types",
    [
        (
            {
                "kind": "pipeline",
                "steps": [
                    {
                        "kind": "transformer",
                        "name": "StandardScaler",
                    },
                    {
                        "kind": "classifier",
                        "name": "LogisticRegression",
                    },
                ],
            },
            [TransformerSchema, ClassifierSchema],
        ),
        (
            {
                "kind": "pipeline",
                "steps": [],
            },
            [],
        ),
        (
            {
                "kind": "pipeline",
                "steps": [
                    {
                        "kind": "transformer",
                        "name": "StandardScaler",
                    },
                    {
                        "kind": "transformer",
                        "name": "RobustScaler",
                    },
                    {
                        "kind": "transformer",
                        "name": "MinMaxScaler",
                    },
                    {
                        "kind": "classifier",
                        "name": "LogisticRegression",
                    },
                ],
            },
            [TransformerSchema, TransformerSchema, TransformerSchema, ClassifierSchema],
        ),
    ],
)
def test_pipeline(data, estimator_types):
    estimator = estimator_factory(data)
    assert estimator.kind == "pipeline"
    assert isinstance(estimator, PipelineSchema)
    assert len(estimator.steps) == len(estimator_types)
    for step, estimator_type, step_data in zip(
        estimator.steps, estimator_types, data["steps"]
    ):
        assert isinstance(step, estimator_type)
        assert step.model_dump() == step_data


def test_build_estimator():
    estimator = build_estimator(
        {
            "kind": "pipeline",
            "steps": [
                {
                    "kind": "transformer",
                    "name": "StandardScaler",
                },
                {
                    "kind": "classifier",
                    "name": "LogisticRegression",
                },
            ],
        }
    )
    assert isinstance(estimator, Pipeline)
    assert len(estimator.steps) == 2
    assert isinstance(estimator.steps[0][1], StandardScaler)
    assert isinstance(estimator.steps[1][1], LogisticRegression)


def test_complex_estimator():
    df = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [1, 2, 3, 4, 5],
            "feature3": ["a", "a", "b", "b", "c"],
            "target": [1, 2, 3, 4, 5],
        }
    )

    schema = {
        "kind": "pipeline",
        "steps": [
            {
                "kind": "column_transformer",
                "transformers": [
                    [
                        "scale",
                        {"kind": "transformer", "name": "StandardScaler"},
                        ["feature1", "feature2"],
                    ],
                    [
                        "encode",
                        {"kind": "transformer", "name": "OneHotEncoder"},
                        ["feature3"],
                    ],
                ],
            },
            {
                "kind": "regressor",
                "name": "LinearRegression",
            },
        ],
    }

    estimator = build_estimator(schema)
    estimator.fit(df[["feature1", "feature2", "feature3"]], df["target"])
    predictions = estimator.predict(df[["feature1", "feature2", "feature3"]])
    np.testing.assert_allclose(df["target"], predictions)
