from typing import Any, Callable, Literal

import duckdb
import sqlfluff
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

import pandas as pd

import statsmodels.formula.api as smf

from decide.config import settings
from decide.storage import DataframeContext, DataStore


def build_agent() -> Agent[DataStore]:
    model = OpenAIChatModel(
        model_name=settings.model_name,
        provider=OllamaProvider(base_url=settings.ollama_url),
        settings={
            "temperature": settings.temperature,
        },
    )
    agent = Agent(
        model=model,
        deps_type=DataStore,
        tools=[duckduckgo_search_tool()],
    )
    return agent


agent = build_agent()


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
    return ctx.deps.get(name).columns.tolist()


@agent.tool
def get_dataframe_context(ctx: RunContext[DataStore], name: str) -> DataframeContext:
    """
    Get the full context of a dataframe with metadata

    Args:
        name
            the name of the dataframe

    Returns:
        DataframeContext
            the context with metadata of the dataframe including the name, length, columns, description, and sql
            as well as the context of each column where applicable (number, categorical, datetime).
    """
    return ctx.deps.get_context(name)


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
    name: str,
    new_name: str,
    sql: str,
    replace: bool = False,
    description: str | None = None,
) -> DataframeContext:
    """
    Run a DuckDB query on a dataframe, the result is stored as a new dataframe.

    Args:
        name
            the name of the dataframe. This dataframe is accessible as a table with the same name in the query.
        new_name
            the name of the new dataframe to store the result of the DuckDB query.
        sql
            the DuckDB query to run. This query is executed using the duckdb.query_df function.
        replace
            whether to replace an existing dataframe with the same name
        description
            the description of the new dataframe that is stored in the context and be used later.
    Returns:
        DataframeContext
            the context with metadata of the new dataframe that is stored under the given name.
    """

    check_columns(ctx.deps.get(name), name)
    if new_name in ctx.deps and not replace:
        raise ModelRetry(
            f"Error: {new_name} is already a valid dataframe. Check the previous messages and try again."
        )
    data = ctx.deps.get(name)
    sql_query = sqlfluff.fix(sql, dialect="duckdb")
    try:
        result = duckdb.query_df(df=data, virtual_table_name=name, sql_query=sql).df()
    except Exception as e:
        raise ModelRetry(f"Error: {e}. Check the previous messages and try again.")
    name = ctx.deps.store(result, new_name, sql=sql_query, description=description)
    context = ctx.deps.get_context(name)
    return context


@agent.tool
def head(ctx: RunContext[DataStore], name: str) -> str:
    """
    Get the first 5 rows of a dataframe

    Args:
        name
            the name of the dataframe
    """
    return ctx.deps.get(name).head(5).to_string()


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
    df = ctx.deps.get(name)
    model = getattr(smf, regression_type)
    return try_statsmodels(model, formula, df)


@agent.tool
def statsmodels_mixedlm_regression(
    ctx: RunContext[DataStore], name: str, formula: str, group_col: str
) -> str:
    """
    Run a statsmodels MixedLM regression
    """
    df = ctx.deps.get(name)
    return try_statsmodels(smf.mixedlm, formula, df, kwargs={"groups": df[group_col]})
