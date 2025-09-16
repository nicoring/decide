# Decide - Data Analysis Assistant

Decide is a Streamlit-based data analysis assistant that lets you chat with your data using open-source LLMs via [Ollama](https://ollama.com/).

## Quickstart

### 1. Requirements

- Python 3.9+
- [Ollama](https://ollama.com/) running locally (for LLM inference)
- [uv](https://github.com/astral-sh/uv) (for fast Python dependency management)

### 2. Run LLM locally

Install Ollama from [https://ollama.com/download](https://ollama.com/download).

Start Ollama and pull a model, e.g. `gpt-oss:20b`:

```bash
ollama pull gpt-oss:20b
```

**Important:** Make sure Ollama is running as a service:

```bash
ollama serve
```

Keep this terminal window open as Ollama needs to be running for the application to work.

### 3. Install Dependencies

Install the project dependencies using uv:

```bash
uv sync
```

### 4. Run the Application

You can run the application in two ways:

**Option 1: Using the provided script**
```bash
./scripts/run.sh
```

**Option 2: Using uv directly**
```bash
uv run streamlit run main.py
```

The application will start and be available at [http://localhost:8501](http://localhost:8501) in your browser.


## Linting, formatting and type checking

This project uses `ruff` for litning and formatting and `mypy` for type checking. You can run all with:

```bash
./scripts/check.sh
```

## Logging

This project uses [Logfire](https://logfire.pydantic.dev/) for observability and monitoring.

### Setup Logfire (Optional)

1. **Get a Logfire token:**
   - Sign up at [https://logfire.pydantic.dev/](https://logfire.pydantic.dev/)
   - Create a new project and get your token

2. **Configure the token:**
   Create a `.env` file in the project root:
   ```bash
   echo "LOGFIRE_TOKEN=your_token_here" > .env
   ```


### Viewing Logs

Once configured, you can view your logs and metrics at your Logfire dashboard URL, which will be displayed in the terminal when the application starts.
