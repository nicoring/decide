"""AST validation for safe execution of generated PyMC code."""

import ast
from typing import Any


class CodeValidationError(Exception):
    """Raised when generated code fails security validation."""

    pass


ALLOWED_BUILTINS = {
    "range",
    "len",
    "enumerate",
    "zip",
    "list",
    "dict",
    "set",
    "tuple",
    "str",
    "int",
    "float",
    "bool",
    "print",
    "abs",
    "min",
    "max",
    "sum",
}


class SafetyValidator(ast.NodeVisitor):
    """AST visitor that validates code safety."""

    def __init__(self):
        self.errors: list[str] = []
        self.has_function_def = False
        self.function_name: str | None = None

    def visit_Import(self, node: ast.Import) -> Any:
        """Block all import statements - modules are injected as globals."""
        for alias in node.names:
            self.errors.append(
                f"Import statements are not allowed: 'import {alias.name}'. "
                f"Required modules (pm, np, pd) are available as globals."
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        """Block all from-import statements - modules are injected as globals."""
        module_name = node.module or "."
        self.errors.append(
            f"Import statements are not allowed: 'from {module_name} import ...'. "
            f"Required modules (pm, np, pd) are available as globals."
        )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        """Track function definitions."""
        if not self.has_function_def:
            self.has_function_def = True
            self.function_name = node.name
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        """Validate function calls for dangerous operations."""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            dangerous_funcs = {"eval", "exec", "compile", "__import__", "open"}
            if func_name in dangerous_funcs:
                self.errors.append(f"Dangerous function call: {func_name}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        """Check for dangerous attribute access."""
        dangerous_attrs = {"__import__", "__loader__", "__builtins__"}
        if node.attr in dangerous_attrs:
            self.errors.append(f"Dangerous attribute access: {node.attr}")
        self.generic_visit(node)


def validate_pymc_code(code: str) -> tuple[bool, list[str], str | None]:
    """
    Validate PyMC code for security.

    Args:
        code: Python code string to validate

    Returns:
        Tuple of (is_valid, errors, function_name)
        - is_valid: Whether the code passed validation
        - errors: List of validation error messages
        - function_name: Name of the defined function, if any
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"Syntax error: {e}"], None

    validator = SafetyValidator()
    validator.visit(tree)

    if not validator.has_function_def:
        validator.errors.append("Code must define a function")

    return len(validator.errors) == 0, validator.errors, validator.function_name


def execute_validated_code(
    code: str, allowed_globals: dict[str, Any]
) -> dict[str, Any]:
    """
    Execute validated code in a restricted namespace.

    Args:
        code: Validated Python code
        allowed_globals: Dictionary of allowed global variables

    Returns:
        The namespace after execution (containing defined functions)

    Raises:
        CodeValidationError: If code fails validation
        Exception: If code execution fails
    """
    is_valid, errors, function_name = validate_pymc_code(code)

    if not is_valid:
        raise CodeValidationError(
            f"Code validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    # Create restricted namespace
    namespace: dict[str, Any] = {
        "__builtins__": {k: __builtins__[k] for k in ALLOWED_BUILTINS},  # type: ignore
        **allowed_globals,
    }

    # Execute code
    exec(code, namespace)

    return namespace
