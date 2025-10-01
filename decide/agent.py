from typing import Any, Callable, Literal

import duckdb
import sqlfluff
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.models.openai import Model, OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

import pandas as pd

import statsmodels.formula.api as smf

from decide.config import settings
from decide.model import EstimatorSchema, EstimatorSchemaAdapter, build_estimator
from decide.storage import DataStore


def build_model(temperature: float) -> Model:
    return OpenAIChatModel(
        model_name=settings.model_name,
        provider=OllamaProvider(base_url=settings.ollama_url),
        settings={
            "temperature": temperature,
        },
    )


_model = build_model(temperature=settings.temperature)
_model_agent_model = build_model(temperature=0.0)  # Use zero temperature for reliable structured output

agent = Agent(
    model=_model,
    deps_type=DataStore,
    tools=[duckduckgo_search_tool()],
)


class ModelFailure(BaseModel):
    """An unrecoverable failure. Only use this when the prompt requested an invalid model."""

    explanation: str

model_agent = Agent[None, EstimatorSchema | ModelFailure](
    name="model_agent",
    system_prompt=f"""
You are a model agent that builds scikit-learn models from natural language prompts.

Your task is to output ONLY valid JSON matching the EstimatorSchema.

CRITICAL RULES:
1. Output ONLY valid JSON - no markdown, no code blocks, no explanations
2. Always include the "kind" field as the discriminator
3. For simple models, use the appropriate kind: "regressor", "classifier", "transformer", "decomposer", or "imputer"
4. For complex models with multiple steps, use "pipeline" with a list of steps

VALID OUTPUT EXAMPLES:

For "linear regression":
{{"kind": "regressor", "name": "LinearRegression"}}

For "logistic regression":
{{"kind": "classifier", "name": "LogisticRegression"}}

For "ridge regression":
{{"kind": "regressor", "name": "Ridge"}}

For "lasso regression":
{{"kind": "regressor", "name": "Lasso"}}

For "random forest classifier":
{{"kind": "classifier", "name": "RandomForestClassifier"}}

For "support vector classifier":
{{"kind": "classifier", "name": "SVC"}}

For a pipeline with scaling and regression:
{{"kind": "pipeline", "steps": [{{"kind": "transformer", "name": "StandardScaler"}}, {{"kind": "regressor", "name": "LinearRegression"}}]}}

For unsupported models (e.g., decision trees, neural networks):
{{"explanation": "Decision tree classifiers are not supported."}}

FULL SCHEMA FOR REFERENCE:
{EstimatorSchemaAdapter.json_schema()}

Remember: Output ONLY the JSON object, nothing else.
    """,
    model=_model_agent_model,
    output_type=EstimatorSchema | ModelFailure,
    output_retries=3,
)


@agent.system_prompt
def system_prompt() -> str:
    prompt = """
You are a Data Science Assistant.
You collaborate with a domain expert (the user).

ROLE:
- You are the data science and statistical modeling expert.
- The user is the domain expert who provides context and interpretation of results.
- You have access to:
  1. A set of dataframes, with the name of the dataframe as the key and the first dataframe as the "initial" dataframe.
  2. Tools to run SQL queries via DuckDB (results are stored as new dataframes).
  3. Statistical and econometric models from statsmodels.

OBJECTIVES:
1. Collaboration: Work with the user to clarify goals and extract relevant domain knowledge before choosing methods.
2. Tool Selection: Based on user input and goals, choose the appropriate tool (DuckDB query, statistical test, regression, etc.).
3. Interpretation: Provide clear, domain-relevant interpretations of results, not just technical output.
4. Iteration: Suggest next steps, refinements, or additional analyses based on results and user feedback.
5. Transparency: Always explain why a tool/model was chosen, state assumptions, and mention limitations.

COLLABORATION STYLE:
- Ask clarifying questions before running analyses.
- Encourage the user to frame hypotheses or specify the relationships of interest.
- Treat the user as the authority on domain knowledge.
- Provide concise, interpretable outputs (avoid unnecessary jargon unless requested).
- Follow an iterative workflow: ask → analyze → interpret → refine.

EXAMPLE BEHAVIORS:
- If asked about relationships between variables: propose regression or correlation analysis, but first confirm the causal direction or hypothesis.
- If asked for a summary: suggest aggregations or transformations using DuckDB queries.
- If the user is uncertain: propose alternative approaches, explain trade-offs, and guide them to a decision.

Your ultimate goal is to combine your data science expertise with the user's domain expertise to produce meaningful, trustworthy insights.

Always respond in English.
"""
    return prompt


@agent.instructions
def instructions(ctx: RunContext[DataStore]) -> str:
    return f"The available dataframes are: {ctx.deps.keys()}"


def check_columns(df: pd.DataFrame, name: str, *cols: str) -> None:
    for col in cols:
        if col not in df.columns:
            raise ModelRetry(
                f"Error: {col} is not a valid column in the dataframe {name}. Check the previous messages and try again."
            )


@agent.tool
def get_dataframe_columns(ctx: RunContext[DataStore], name: str) -> list[str]:
    """
    Get the columns of a dataframe

    Args:
        name
            the name of the dataframe

    Returns:
        list[str]
            contains the columns of the dataframe
    """
    return ctx.deps.get_data(name).columns.tolist()


@agent.tool
def get_dataframe_context(ctx: RunContext[DataStore], name: str) -> str:
    """
    Get the full context of a dataframe with metadata

    Args:
        name
            the name of the dataframe

    Returns:
        str
            the context with metadata of the dataframe including the name, length, columns, description, and sql
            as well as the context of each column where applicable (number, categorical, datetime).
    """
    context = ctx.deps.get_context(name)
    return context


@agent.tool
def delete_dataframe(ctx: RunContext[DataStore], name: str) -> None:
    """
    Delete a dataframe
    """
    if name not in ctx.deps:
        raise ModelRetry(
            f"Error: {name} is not a valid dataframe. Check the previous messages and try again."
        )
    ctx.deps.delete(name)


@agent.tool(retries=3)
def run_duckdb_query(
    ctx: RunContext[DataStore],
    names: list[str],
    new_name: str,
    sql: str,
    description: str,
    replace: bool = False,
) -> str:
    """
    Run a DuckDB query on set of dataframes, the result is stored as a new dataframe.

    Args:
        names
            the names of the dataframes. These dataframes are accessible as tables with the same name in the query.
        new_name
            the name of the new dataframe to store the result of the DuckDB query.
        sql
            the DuckDB query to run. This query is executed using the duckdb.query_df function. MUST comply with the duckdb syntax.
        description
            the description of the new dataframe that is stored in the context and be used later.
        replace
            whether to replace an existing dataframe with the same name
    Returns:
        str
            the context with metadata of the new dataframe that is stored under the given name.
    """
    if new_name in ctx.deps and not replace:
        raise ModelRetry(
            f"Error: {new_name} is already a valid dataframe. Check the previous messages and try again."
        )
    sql_query = sqlfluff.fix(sql, dialect="duckdb")

    conn = duckdb.connect()
    for name in names:
        data = ctx.deps.get_data(name)
        conn.register(name, data)
    try:
        result = conn.sql(sql).df()
    except Exception as e:
        raise ModelRetry(f"Error: {e}. Check the previous messages and try again.")
    finally:
        conn.close()
    sql_data = ctx.deps.store_sql(
        df=result,
        name=new_name,
        input_data=names,
        sql=sql_query,
        description=description,
    )
    context = sql_data.get_context()
    return context


@agent.tool
def head(ctx: RunContext[DataStore], name: str) -> str:
    """
    Get the first 5 rows of a dataframe

    Args:
        name
            the name of the dataframe
    """
    return ctx.deps.get_data(name).head(5).to_string()


def try_statsmodels(
    callable: Callable[..., Any],
    formula: str,
    data: pd.DataFrame,
    kwargs: dict[str, Any] | None = None,
) -> str:
    if kwargs is None:
        kwargs = {}
    try:
        return callable(formula, data=data, **kwargs).fit().summary().as_text()
    except Exception as e:
        raise ModelRetry(f"Error: {e}. Check the previous messages and try again.")


@agent.tool
def statsmodels_regression(
    ctx: RunContext[DataStore],
    name: str,
    regression_type: Literal[
        "ols",
        "wls",
        "glm",
        "rlm",
        "logit",
        "probit",
        "poisson",
        "negativebinomial",
        "quantreg",
        "phreg",
        "ordinal_gee",
        "nominal_gee",
        "gee",
        "glmgam",
        "conditional_logit",
        "conditional_mnlogit",
        "conditional_poisson",
    ],
    formula: str,
) -> str:
    """
    Run a statsmodels regression

    Args:
        name
            the name of the dataframe
        regression_type
            the type of regression to run
        formula
            the formula to use for the regression

    Returns:
        str
            the summary of the regression
    """
    df = ctx.deps.get_data(name)
    model = getattr(smf, regression_type)
    return try_statsmodels(model, formula, df)


@agent.tool
def statsmodels_mixedlm_regression(
    ctx: RunContext[DataStore], name: str, formula: str, group_col: str
) -> str:
    """
    Run a statsmodels MixedLM regression
    """
    df = ctx.deps.get_data(name)
    return try_statsmodels(smf.mixedlm, formula, df, kwargs={"groups": df[group_col]})


@agent.tool(retries=3)
async def fit_model(
    ctx: RunContext[DataStore],
    name: str,
    features: list[str],
    target: str,
    model_prompt: str,
    description: str,
) -> str:
    """
    Fit a scikit learn model to a dataframe, using a natural language prompt to describe the desired model or pipeline.

    This tool allows you to specify, in plain English, the type of model or modeling pipeline you want to fit (e.g., regression, classification, feature engineering, or a combination thereof). The prompt will be interpreted by a sub-agent, which will generate a structured model specification and fit it to the provided dataframe.

    Usage:
        - Use this tool when you want to fit a model or pipeline to a dataframe, but do not want to specify the exact model code.
        - The features and target variables must be specified in the context of the dataframe.
        - The model_prompt should clearly describe the modeling goal and any special requirements (e.g., regularization, feature selection, etc.). It should not mention any specifics of how to fit the model this will be done byt this tool call.
        - The dataframe to use must already exist in the context under the given name.

    Args:
        name (str): The name of the dataframe to use for modeling. This must match one of the available dataframes in the context.
        features (list[str]): The features to use for modeling. This must be a list of columns in the dataframe.
        target (str): The target variable to predict. This must be a column in the dataframe.
        model_prompt (str): A natural language description of the model or modeling pipeline to fit. Be as specific as possible about the modeling objective, variables, and any constraints or preferences. Do not give instructions, just describe the model.
        description (str): A description of the model to store in the context.
    Returns:
        str: Metadata and context for the predictions, which are automatically stored as a new dataframe named "{name}_predictions".
            - The returned context includes information about the predictions and the model used.
            - The predictions dataframe can be accessed using the new name.

    Notes:
        - If the model_prompt is invalid or cannot be translated into a valid model, an error will be raised.
        - The tool supports complex pipelines, including transformers and multiple modeling steps.
    """
    data = ctx.deps.get(name)
    df = data.df
    check_columns(df, name, *features, target)

    feature_metadata = [
        col.model_dump_json() for col in data.context.columns if col.name in features
    ]
    target_metadata = [
        col.model_dump_json() for col in data.context.columns if col.name == target
    ][0]

    prompt = f"""
    Generate a model schema for a model that predicts the target variable {target} using the following features: {features}.

    Metadata for the features: {feature_metadata}
    Metadata for the target: {target_metadata}

    The model is described as follows:
    {model_prompt}

    DO NOT include any other text than the model schema.
    """

    result = await model_agent.run(prompt)
    output = result.output
    if isinstance(output, ModelFailure):
        raise ModelRetry(
            f"Error: {output.explanation}. Check the previous messages and try again."
        )
    estimator = build_estimator(output)
    try:
        estimator.fit(df[features], df[target])
        predictions = estimator.predict(df[features])
    except Exception as e:
        raise ModelRetry(
            f"Error fitting the model: {e}. Check the previous messages and try again."
        )
    data = ctx.deps.store_model(
        df=pd.DataFrame(predictions, columns=["predicted_" + target]),
        name=f"{name}_predictions",
        description=description,
        input_data=name,
        model=estimator,
    )
    context = data.get_context()
    return context
