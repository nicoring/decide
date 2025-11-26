from typing import Any, Callable, Literal

import arviz as az
import duckdb
import pymc as pm
import sqlfluff
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.models.openai import Model, OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
import logfire

import numpy as np
import pandas as pd

import statsmodels.formula.api as smf

from decide.config import settings
from decide.model import EstimatorSchema, EstimatorSchemaAdapter, build_estimator
from decide.storage import DataStore
from decide.validation import execute_validated_code


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


class BayesianModelCode(BaseModel):
    """Generated PyMC model code."""

    code: str


class BayesianModelFailure(BaseModel):
    """An unrecoverable failure. Only use this when the prompt requested an invalid Bayesian model."""

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


bayesian_agent = Agent[None, BayesianModelCode | BayesianModelFailure](
    name="bayesian_agent",
    system_prompt="""
You are a Bayesian modeling expert that generates PyMC model code from natural language descriptions.

Your task is to output ONLY valid Python code that defines a function called `build_model`.

CRITICAL RULES:
1. Output ONLY Python code - no markdown, no code blocks, no explanations
2. The code must define a function: `def build_model(df, features, target):`
3. The function must return a PyMC model object
4. DO NOT include any import statements - pm, np, and pd are already available as globals
5. Use modern PyMC syntax with pm (not pymc)
6. The model should use the provided df, features, and target parameters

FUNCTION SIGNATURE:
```python
def build_model(df, features, target):
    # pm, np, and pd are already available
    # IMPORTANT: Convert all features to numeric - PyMC cannot handle object dtypes
    # Use .astype('category').cat.codes for categorical/string columns
    # Your model code here
    return model
```

DATA TYPE HANDLING:
- PyMC requires numeric arrays - ALWAYS convert object/string columns to numeric codes
- For categorical variables: use df[col].astype('category').cat.codes.values
- For numeric columns: use df[col].values directly
- Check the metadata to determine column types

CRITICAL - DATA STANDARDIZATION:
- ALWAYS standardize continuous predictors: X_scaled = (X - X.mean()) / X.std()
- This dramatically improves sampler convergence and mixing
- Do NOT standardize binary variables (0/1) or categorical codes
- Standardization is essential for hierarchical models

EXAMPLE OUTPUTS:

For "linear regression with normal priors" (all numeric features):
```python
def build_model(df, features, target):
    # Convert to numpy arrays, ensuring numeric dtype
    X = df[features].select_dtypes(include=[np.number]).values.astype(float)

    # Standardize features for better convergence
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_scaled = (X - X_mean) / (X_std + 1e-8)

    y = df[target].values.astype(float)

    with pm.Model() as model:
        # Priors (tighter for standardized data)
        alpha = pm.Normal('intercept', mu=0, sigma=2)
        beta = pm.Normal('coefficients', mu=0, sigma=2, shape=X.shape[1])
        sigma = pm.HalfNormal('sigma', sigma=1)

        # Linear model
        mu = alpha + pm.math.dot(X_scaled, beta)

        # Likelihood
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    return model
```

For "logistic regression" (numeric features):
```python
def build_model(df, features, target):
    X = df[features].select_dtypes(include=[np.number]).values.astype(float)

    # Standardize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_scaled = (X - X_mean) / (X_std + 1e-8)

    y = df[target].values.astype(int)

    with pm.Model() as model:
        # Priors (tighter for standardized data)
        alpha = pm.Normal('intercept', mu=0, sigma=2)
        beta = pm.Normal('coefficients', mu=0, sigma=2, shape=X.shape[1])

        # Logistic model
        p = pm.math.invlogit(alpha + pm.math.dot(X_scaled, beta))

        # Likelihood
        y_obs = pm.Bernoulli('y_obs', p=p, observed=y)

    return model
```

For "hierarchical logistic regression with group effects":
```python
def build_model(df, features, target):
    # Note: Pclass should NOT be in features list, access it directly from df
    groups = df['Pclass'].astype('category').cat.codes.values
    n_groups = len(df['Pclass'].unique())

    # Process features
    X_list = []
    is_continuous = []

    for col in features:
        if pd.api.types.is_numeric_dtype(df[col]):
            x_col = df[col].values.astype(float)
            is_continuous.append(len(np.unique(x_col)) > 10)
            X_list.append(x_col)
        else:
            X_list.append(df[col].astype('category').cat.codes.values.astype(float))
            is_continuous.append(False)

    X = np.column_stack(X_list)

    # Standardize continuous features only
    X_scaled = X.copy()
    for i, is_cont in enumerate(is_continuous):
        if is_cont:
            X_scaled[:, i] = (X[:, i] - X[:, i].mean()) / (X[:, i].std() + 1e-8)

    y = df[target].values.astype(int)

    with pm.Model() as model:
        # Hyperpriors (tighter for standardized data)
        mu_alpha = pm.Normal('mu_alpha', mu=0, sigma=2)
        sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=1)

        # Non-centered parameterization for group intercepts
        alpha_offset = pm.Normal('alpha_offset', mu=0, sigma=1, shape=n_groups)
        alpha = pm.Deterministic('alpha', mu_alpha + alpha_offset * sigma_alpha)

        # Global slopes
        beta = pm.Normal('beta', mu=0, sigma=2, shape=X.shape[1])

        # Logistic model
        p = pm.math.invlogit(alpha[groups] + pm.math.dot(X_scaled, beta))

        # Likelihood
        y_obs = pm.Bernoulli('y_obs', p=p, observed=y)

    return model
```

For "linear regression with categorical predictors":
```python
def build_model(df, features, target):
    # Handle mixed numeric and categorical features
    X_list = []
    is_continuous = []

    for col in features:
        if pd.api.types.is_numeric_dtype(df[col]):
            x_col = df[col].values.astype(float)
            # Check if truly continuous (not binary/categorical codes)
            is_continuous.append(len(np.unique(x_col)) > 10)
            X_list.append(x_col)
        else:
            # Convert categorical/object to numeric codes
            X_list.append(df[col].astype('category').cat.codes.values.astype(float))
            is_continuous.append(False)

    X = np.column_stack(X_list)

    # Standardize only continuous features
    X_scaled = X.copy()
    for i, is_cont in enumerate(is_continuous):
        if is_cont:
            X_scaled[:, i] = (X[:, i] - X[:, i].mean()) / (X[:, i].std() + 1e-8)

    y = df[target].values.astype(float)

    with pm.Model() as model:
        alpha = pm.Normal('intercept', mu=0, sigma=2)
        beta = pm.Normal('coefficients', mu=0, sigma=2, shape=X.shape[1])
        sigma = pm.HalfNormal('sigma', sigma=1)

        mu = alpha + pm.math.dot(X_scaled, beta)
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    return model
```

For "hierarchical model with group effects":
```python
def build_model(df, features, target):
    # First feature is the grouping variable (categorical)
    groups = df[features[0]].astype('category').cat.codes.values
    n_groups = len(df[features[0]].unique())

    # Remaining features are predictors
    X_list = []
    is_continuous = []

    for col in features[1:]:
        if pd.api.types.is_numeric_dtype(df[col]):
            x_col = df[col].values.astype(float)
            is_continuous.append(len(np.unique(x_col)) > 10)
            X_list.append(x_col)
        else:
            X_list.append(df[col].astype('category').cat.codes.values.astype(float))
            is_continuous.append(False)

    X = np.column_stack(X_list) if X_list else np.zeros((len(df), 1))

    # Standardize continuous features
    X_scaled = X.copy()
    for i, is_cont in enumerate(is_continuous):
        if is_cont:
            X_scaled[:, i] = (X[:, i] - X[:, i].mean()) / (X[:, i].std() + 1e-8)

    y = df[target].values.astype(float)

    with pm.Model() as model:
        # Hyperpriors (tighter for better convergence)
        mu_alpha = pm.Normal('mu_alpha', mu=0, sigma=2)
        sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=2)

        # Group-level intercepts (non-centered parameterization)
        alpha_offset = pm.Normal('alpha_offset', mu=0, sigma=1, shape=n_groups)
        alpha = pm.Deterministic('alpha', mu_alpha + alpha_offset * sigma_alpha)

        # Global slopes (tighter priors)
        beta = pm.Normal('beta', mu=0, sigma=2, shape=X.shape[1])

        # Error
        sigma = pm.HalfNormal('sigma', sigma=1)

        # Linear model
        mu = alpha[groups] + pm.math.dot(X_scaled, beta)

        # Likelihood
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    return model
```

IMPORTANT NOTES:
- DO NOT include import statements - pm, np, pd are pre-imported
- CRITICAL: PyMC cannot handle object dtypes - ALWAYS convert to numeric:
  * Use pd.api.types.is_numeric_dtype() to check if numeric
  * Use .astype('category').cat.codes.values for strings/categoricals
  * Use .select_dtypes(include=[np.number]) to filter numeric columns
  * Ensure target is .astype(float) or .astype(int) as appropriate
- CRITICAL: Standardize continuous features: X_scaled = (X - X.mean()) / X.std()
  * This is ESSENTIAL for convergence in complex models
  * Only standardize truly continuous variables (>10 unique values)
  * Do NOT standardize binary (0/1) or small categorical codes
- Use tighter priors for standardized data (sigma=2 instead of sigma=10)
- For hierarchical models, use non-centered parameterization for better sampling
- For count data, use Poisson or NegativeBinomial likelihoods
- For binary outcomes, use Bernoulli or Categorical (ensure y is int)
- Always use X.shape[1] not len(features) for beta shape
- Return the model object at the end

Remember: Output ONLY the Python code for the function, nothing else. NO imports.
    """,
    model=_model_agent_model,
    output_type=BayesianModelCode | BayesianModelFailure,
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


@agent.tool
async def fit_bayesian_model(
    ctx: RunContext[DataStore],
    name: str,
    features: list[str],
    target: str,
    model_description: str,
    description: str,
    n_samples: int = 2000,
    n_tune: int = 2000,
    n_chains: int = 4,
    target_accept: float = 0.95,
) -> str:
    """
    Fit a Bayesian model using PyMC to a dataframe, using natural language to describe the model structure.

    This tool generates PyMC code from your description, validates it for security, executes it safely,
    and performs MCMC sampling to obtain posterior distributions.

    Usage:
        - Use this tool when you want probabilistic inference with uncertainty quantification
        - Ideal for: hierarchical models, regularization via priors, small sample sizes, causal inference
        - The model_description should clearly specify the model structure, priors, and likelihood

    Args:
        name: The name of the dataframe to use for modeling
        features: List of feature column names to use as predictors
        target: The target variable column name to predict
        model_description: Natural language description of the Bayesian model structure
            Examples:
            - "linear regression with weakly informative normal priors"
            - "logistic regression with regularizing priors on coefficients"
            - "hierarchical linear model with varying intercepts by group"
        description: A description of this analysis to store in the context
        n_samples: Number of posterior samples to draw (default: 2000)
        n_tune: Number of tuning/warmup samples (default: 2000, increased for complex models)
        n_chains: Number of MCMC chains to run (default: 4 for better convergence diagnostics)
        target_accept: Target acceptance rate for NUTS sampler (default: 0.95, higher = more accurate but slower)

    Returns:
        str: Metadata and diagnostics for the fitted model, including:
            - Posterior summary statistics (mean, sd, hdi)
            - Convergence diagnostics (Rhat, effective sample size)
            - Information about the stored posterior samples

    Notes:
        - Results are stored as "{name}_bayesian" containing posterior samples
        - The tool validates generated code for security before execution
        - Check Rhat < 1.01 and ESS > 400 for reliable inference
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

    base_prompt = f"""
    Generate PyMC model code that predicts the target variable '{target}' using features: {features}.

    Feature metadata: {feature_metadata}
    Target metadata: {target_metadata}

    Model description:
    {model_description}

    Output ONLY the Python code for the build_model function. Do not include markdown formatting or explanations.
    """

    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        # Build prompt with error feedback on retry
        if attempt == 0:
            prompt = base_prompt
        else:
            prompt = f"""
{base_prompt}

PREVIOUS ATTEMPT FAILED with the following error:
{last_error}

Please fix the code to address this error. Common fixes:
- Ensure all features are converted to numeric (use pd.api.types.is_numeric_dtype() to check)
- Standardize continuous features (len(np.unique(x)) > 10)
- Use .astype(float) or .astype(int) appropriately
- Check array shapes before matrix operations
- Ensure the function returns the model object

Output ONLY the corrected Python code.
"""

        try:
            # Generate code
            result = await bayesian_agent.run(prompt)
            output = result.output

            if isinstance(output, BayesianModelFailure):
                last_error = output.explanation
                continue

            model_code = output.code

            # Validate and execute the code
            allowed_globals = {"pm": pm, "np": np, "pd": pd}
            namespace = execute_validated_code(model_code, allowed_globals)
            build_model_func = namespace.get("build_model")

            if build_model_func is None:
                last_error = "Generated code did not define 'build_model' function."
                continue

            # Build the model
            model = build_model_func(df, features, target)

            # Sample from the posterior with better settings for complex models
            with model:
                idata = pm.sample(
                    draws=n_samples,
                    tune=n_tune,
                    chains=n_chains,
                    target_accept=target_accept,
                    return_inferencedata=True,
                )

            # Check convergence
            summary = az.summary(idata)
            max_rhat = summary['r_hat'].max()
            min_ess = summary['ess_bulk'].min()

            convergence_warnings = []
            if max_rhat > 1.0:
                convergence_warnings.append(
                    f"WARNING: Poor convergence detected. Max R-hat = {max_rhat:.3f} (should be < 1.01). "
                    "The chains have not converged. Consider: (1) increasing n_tune, (2) standardizing features, "
                    "(3) using tighter priors, or (4) simplifying the model."
                )
            if min_ess < 400:
                convergence_warnings.append(
                    f"WARNING: Low effective sample size. Min ESS = {min_ess:.0f} (should be > 400). "
                    "Posterior estimates may be unreliable. Consider increasing n_samples or n_chains."
                )

            if convergence_warnings:
                warning_msg = "\n\n".join(convergence_warnings)
                raise ModelRetry(
                    f"Bayesian model sampling completed but convergence diagnostics failed:\n\n{warning_msg}\n\n"
                    "Please retry with better settings or a simpler model."
                )

            # Extract posterior samples as dataframe
            posterior_df = az.extract(idata, num_samples=500).to_dataframe().reset_index()

            # Success! Store results and return
            bayesian_data = ctx.deps.store_bayesian(
                df=posterior_df,
                name=f"{name}_bayesian",
                description=description,
                input_data=name,
                model_code=model_code,
                idata=idata,
            )

            return bayesian_data.get_context()

        except ModelRetry:
            # Convergence issues should bubble up immediately - caller can decide to retry with better params
            raise
        except Exception as e:
            last_error = str(e)
            logfire.error("Error fitting Bayesian model: {e}", e=e)
            if attempt < max_retries - 1:
                # Try again with error feedback
                continue
            else:
                # Final attempt failed
                raise ModelRetry(
                    f"Error fitting Bayesian model after {max_retries} attempts. "
                    f"Last error: {last_error}. "
                    f"Please check your model description and try again."
                )


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
