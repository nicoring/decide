#!/bin/bash

set -e

uv run mypy decide
uv run ruff check --fix --unsafe-fixes
uv run ruff format
