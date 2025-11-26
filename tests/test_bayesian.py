import pytest

import numpy as np
import pandas as pd
import pymc as pm

from decide.validation import (
    CodeValidationError,
    execute_validated_code,
    validate_pymc_code,
)


class TestValidation:
    """Test AST validation for security."""

    def test_valid_simple_model(self):
        """Test that valid PyMC code passes validation."""
        code = """
def build_model(df, features, target):
    X = df[features].values
    y = df[target].values

    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=len(features))
        sigma = pm.HalfNormal('sigma', sigma=1)

        mu = alpha + pm.math.dot(X, beta)
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    return model
"""
        is_valid, errors, func_name = validate_pymc_code(code)
        assert is_valid
        assert len(errors) == 0
        assert func_name == "build_model"

    def test_any_import_blocked(self):
        """Test that all import statements are blocked."""
        code = """
import os

def build_model(df, features, target):
    return None
"""
        is_valid, errors, _ = validate_pymc_code(code)
        assert not is_valid
        assert any("Import statements are not allowed" in e for e in errors)

    def test_pymc_import_blocked(self):
        """Test that even 'safe' imports like pymc are blocked (since injected as globals)."""
        code = """
import pymc as pm

def build_model(df, features, target):
    return None
"""
        is_valid, errors, _ = validate_pymc_code(code)
        assert not is_valid
        assert any("Import statements are not allowed" in e for e in errors)
        assert any("pm, np, pd" in e for e in errors)

    def test_dangerous_function_blocked(self):
        """Test that dangerous function calls are blocked."""
        code = """
def build_model(df, features, target):
    eval('print("dangerous")')
    return None
"""
        is_valid, errors, _ = validate_pymc_code(code)
        assert not is_valid
        assert any("Dangerous function call: eval" in e for e in errors)

    def test_file_operations_blocked(self):
        """Test that file operations are blocked."""
        code = """
def build_model(df, features, target):
    with open('/etc/passwd', 'r') as f:
        data = f.read()
    return None
"""
        is_valid, errors, _ = validate_pymc_code(code)
        assert not is_valid
        assert any("Dangerous function call: open" in e for e in errors)

    def test_no_function_definition(self):
        """Test that code without function definition fails."""
        code = """
x = 5
"""
        is_valid, errors, _ = validate_pymc_code(code)
        assert not is_valid
        assert any("must define a function" in e for e in errors)

    def test_syntax_error(self):
        """Test that syntax errors are caught."""
        code = """
def build_model(df, features, target)
    return None
"""
        is_valid, errors, _ = validate_pymc_code(code)
        assert not is_valid
        assert any("Syntax error" in e for e in errors)


class TestExecution:
    """Test safe code execution."""

    def test_execute_valid_code(self):
        """Test executing valid code."""
        code = """
def build_model(df, features, target):
    return "model"
"""
        namespace = execute_validated_code(code, {"np": np, "pm": pm, "pd": pd})
        assert "build_model" in namespace
        assert callable(namespace["build_model"])

        # Test function execution
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = namespace["build_model"](df, ["a"], "b")
        assert result == "model"

    def test_execute_invalid_code_raises(self):
        """Test that invalid code raises CodeValidationError."""
        code = """
import os

def build_model(df, features, target):
    return None
"""
        with pytest.raises(CodeValidationError) as exc_info:
            execute_validated_code(code, {"np": np, "pm": pm, "pd": pd})
        assert "Code validation failed" in str(exc_info.value)
        assert "Import statements are not allowed" in str(exc_info.value)


class TestBayesianModelIntegration:
    """Integration tests for Bayesian model building."""

    def test_validation_retry(self):
        """Test that validation errors trigger retry with error feedback."""
        from decide.validation import validate_pymc_code

        # First attempt - code with import (will fail validation)
        bad_code = """
import os

def build_model(df, features, target):
    return None
"""
        is_valid, errors, _ = validate_pymc_code(bad_code)
        assert not is_valid
        assert "Import statements are not allowed" in errors[0]

        # After retry - corrected code without imports
        good_code = """
def build_model(df, features, target):
    X = df[features].select_dtypes(include=[np.number]).values.astype(float)
    y = df[target].values.astype(float)

    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sigma=2)
        beta = pm.Normal('beta', mu=0, sigma=2, shape=X.shape[1])
        sigma = pm.HalfNormal('sigma', sigma=1)

        mu = alpha + pm.math.dot(X, beta)
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    return model
"""
        is_valid, errors, _ = validate_pymc_code(good_code)
        assert is_valid
        assert len(errors) == 0

    def test_linear_regression_model(self):
        """Test building and sampling from a simple linear regression model."""
        code = """
def build_model(df, features, target):
    X = df[features].values
    y = df[target].values

    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=len(features))
        sigma = pm.HalfNormal('sigma', sigma=1)

        mu = alpha + pm.math.dot(X, beta)
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    return model
"""
        # Create synthetic data
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 2)
        y = 2 + 3 * X[:, 0] - 1.5 * X[:, 1] + np.random.randn(n) * 0.5

        df = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "y": y})

        # Execute code and build model
        namespace = execute_validated_code(code, {"pm": pm, "np": np, "pd": pd})
        build_model_func = namespace["build_model"]
        model = build_model_func(df, ["x1", "x2"], "y")

        # Sample from posterior
        with model:
            idata = pm.sample(
                draws=100, tune=100, chains=2, return_inferencedata=True, random_seed=42
            )

        # Check that sampling succeeded
        assert idata is not None
        assert "posterior" in idata.groups()
        assert "alpha" in idata.posterior.data_vars
        assert "beta" in idata.posterior.data_vars

    def test_logistic_regression_model(self):
        """Test building a logistic regression model."""
        code = """
def build_model(df, features, target):
    X = df[features].values
    y = df[target].values.astype(int)

    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sigma=5)
        beta = pm.Normal('beta', mu=0, sigma=5, shape=len(features))

        p = pm.math.invlogit(alpha + pm.math.dot(X, beta))
        y_obs = pm.Bernoulli('y_obs', p=p, observed=y)

    return model
"""
        # Create synthetic binary classification data
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 2)
        z = 1 + 2 * X[:, 0] - X[:, 1]
        p = 1 / (1 + np.exp(-z))
        y = (np.random.rand(n) < p).astype(int)

        df = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "y": y})

        # Execute code and build model
        namespace = execute_validated_code(code, {"pm": pm, "np": np, "pd": pd})
        build_model_func = namespace["build_model"]
        model = build_model_func(df, ["x1", "x2"], "y")

        # Sample from posterior
        with model:
            idata = pm.sample(
                draws=100, tune=100, chains=2, return_inferencedata=True, random_seed=42
            )

        # Check that sampling succeeded
        assert idata is not None
        assert "posterior" in idata.groups()
