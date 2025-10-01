import sys

sys.path.append("../decide")

import logfire
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators.common import EqualsExpected, IsInstance

from decide.agent import ModelFailure, model_agent
from decide.model import ClassifierSchema, EstimatorSchema, RegressorSchema

happy_dataset = Dataset(
    cases=[
        Case(
            name="lin_reg",
            inputs="Build a linear regression model.",
            expected_output=RegressorSchema(name="LinearRegression"),
        ),
        Case(
            name="log_reg",
            inputs="Build a logistic regression model.",
            expected_output=ClassifierSchema(name="LogisticRegression"),
        ),
        Case(
            name="rf_reg",
            inputs="Build a random forest classifier model.",
            expected_output=ClassifierSchema(name="RandomForestClassifier"),
        ),
        Case(
            name="ridge_reg",
            inputs="Build a ridge regression model.",
            expected_output=RegressorSchema(name="Ridge"),
        ),
        Case(
            name="lasso_reg",
            inputs="Build a lasso regression model.",
            expected_output=RegressorSchema(name="Lasso"),
        ),
        Case(
            name="svc_class",
            inputs="Build a support vector classifier.",
            expected_output=ClassifierSchema(name="SVC"),
        ),
        Case(
            name="rf_class",
            inputs="Build a random forest classifier.",
            expected_output=ClassifierSchema(name="RandomForestClassifier"),
        ),
        Case(
            name="log_reg_alt",
            inputs="Build a logistic regression classifier.",
            expected_output=ClassifierSchema(name="LogisticRegression"),
        ),
    ],
    evaluators=[EqualsExpected()]
)

sad_dataset = Dataset(
    cases=[
        Case(
            name="dtree_class",
            inputs="Build a decision tree classifier.",
            expected_output=ModelFailure(explanation="Decision tree classifiers are not supported."),
        ),
        Case(
            name="neural_nets",
            inputs="Build a neural network classifier.",
            expected_output=ModelFailure(explanation="Neural networks are not supported."),
        ),
    ],
    evaluators=[IsInstance(ModelFailure)]
)



async def run_model_agent(input: str) -> EstimatorSchema:
    return (await model_agent.run(input)).output

if __name__ == "__main__":
    from decide.config import settings

    if settings.logfire_token is not None:
        logfire.configure(
            token=settings.logfire_token,
            environment='development',
            service_name='evals',
        )
    report = happy_dataset.evaluate_sync(run_model_agent)
    report.print(include_input=True, include_output=True, include_durations=False)
    report = sad_dataset.evaluate_sync(run_model_agent)
    report.print(include_input=True, include_output=True, include_durations=False)
