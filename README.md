# EDA Workflow

An AI-powered exploratory data analysis workflow that performs consistent, first-pass analysis of datasets using LangChain and LangGraph. The workflow runs a fixed set of analysis tools, uses an LLM to extract observations after each step, and synthesizes findings into a summary with actionable recommendations.

## How It Works

The workflow follows a sequential process:

1. **Analyze**: Runs a fixed set of predefined analysis tools on the dataset
2. **Observe**: After each tool, the LLM extracts concise observations from the results
3. **Synthesize**: Once all tools have run, the LLM summarizes findings and provides actionable recommendations

This approach combines deterministic pandas-based analysis with LLM-powered interpretation.

## Analysis Steps

The workflow performs the following analysis steps in sequence:

1. **Profile Dataset**
   - Dataset shape (rows, columns)
   - Column data types
   - Numeric column summary statistics (mean, std, min, max, quartiles)
   - Categorical column value distributions (top 10 values)

2. **Analyze Missingness**
   - Missing value counts and percentages per column
   - Columns with high missingness (>20%)
   - Complete rows count and percentage

3. **Detect Duplicates**
   - Duplicate row detection
   - Duplicate ID column detection
   - Data quality red flags

4. **Analyze Distributions**
   - Skewness and kurtosis for numeric columns
   - Jarque-Bera normality test approximation
   - Distribution shape assessment

5. **Detect Outliers**
   - IQR method (1.5 × IQR rule)
   - Z-score method (threshold = 3)
   - Outlier counts and sample values

6. **Compute Aggregates**
   - Group-by aggregations on categorical columns
   - Mean and count statistics for numeric columns
   - Top 10 categories per grouping variable

7. **Analyze Relationships**
   - Correlation matrix for numeric variables
   - Contingency tables for categorical variable pairs (top 5 combinations)
   - Cross-tabulation summaries

## Setup

### Prerequisites

- **Python 3.10 or 3.11**
- **Poetry** (dependency manager)
- **OpenAI API Key**

### Installation Steps

1. **Install Poetry** (if not already installed):
   
   **Windows (PowerShell)**:
   ```powershell
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
   ```
   
   **macOS/Linux**:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
   
   After installation, restart your terminal. If `poetry` command is not found:
   - **Windows**: Add `%APPDATA%\Python\Scripts` to your system PATH
   - **macOS/Linux**: Add `export PATH="$HOME/.local/bin:$PATH"` to your `~/.bashrc` or `~/.zshrc`

2. **Install dependencies**:
   ```bash
   poetry install
   ```
   
   This will install all dependencies with the exact versions specified in `poetry.lock`, ensuring consistency across all environments.

3. **Set up your OpenAI API key**:
   
   Create a `.env` file in the project root:
   
   **Windows**:
   ```powershell
   echo OPENAI_API_KEY=sk-your-key-here > .env
   ```
   
   **macOS/Linux**:
   ```bash
   echo "OPENAI_API_KEY=sk-your-key-here" > .env
   ```
   
   Or manually create a `.env` file with:
   ```
   OPENAI_API_KEY=sk-your-key-here
   ```

### Multiple Python Versions?

If you have multiple Python versions installed and want to use a specific one:

```bash
# Tell Poetry which Python to use
poetry env use python3.11  # or python3.10

# Then install dependencies
poetry install
```

Poetry will create a virtual environment with your chosen Python version.

## Usage

### Python API

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from eda_workflow.eda_workflow import EDAWorkflow

load_dotenv()

# Initialize the workflow with an LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
workflow = EDAWorkflow(model=llm)

# Run analysis on a dataset
workflow.invoke_workflow("data/cafe_sales.csv")

# Retrieve results
summary = workflow.get_summary()              # str: 2-3 sentence summary
recommendations = workflow.get_recommendations()  # list[str]: 3-5 actionable items
observations = workflow.get_observations()    # dict[str, list[str]]: per-step observations
results = workflow.get_results()              # dict: full pandas analysis results
```

**Optional Parameters:**

```python
# Enable logging to save results
workflow = EDAWorkflow(
    model=llm,
    log=True,                    # Save analysis to logs/
    log_path="custom/log/dir"    # Custom log directory (optional)
)

# Use with a checkpointer for state persistence
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
workflow = EDAWorkflow(model=llm, checkpointer=checkpointer)
workflow.invoke_workflow("data/cafe_sales.csv", config={"configurable": {"thread_id": "1"}})
```

### Running the Example

```bash
poetry run python example_usage.py
```

This runs a full analysis on the sample dataset and prints the results for each step.

### Visualizing the Workflow Graph

You can generate a visual diagram of the LangGraph workflow:

```python
workflow._compiled_graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
```

This creates a PNG diagram showing all analysis nodes and their connections.

## Project Structure

```
eda-workflow/
├── data/
│   └── cafe_sales.csv             # Sample dataset
├── eda_workflow/
│   ├── __init__.py
│   ├── eda_workflow.py            # Main workflow class and graph
│   └── prompts/                   # LLM prompt templates
│       ├── extract_observations_system.txt
│       ├── extract_observations_human.txt
│       ├── synthesize_findings_system.txt
│       └── synthesize_findings_human.txt
├── .env                           # Environment variables (create this)
├── example_usage.py               # Example script
├── pyproject.toml                 # Dependencies configuration
├── poetry.lock                    # Locked dependency versions
└── README.md
```

**Important**: The `poetry.lock` file is committed to ensure all users get identical, tested dependency versions.

## Key Features

- **Fixed Analysis Pipeline**: Deterministic, reproducible analysis steps using pandas
- **LLM-Powered Insights**: Extracts human-readable observations after each step
- **Actionable Recommendations**: Synthesizes findings into practical next steps
- **Structured Output**: Returns summary, recommendations, observations, and raw results
- **State Persistence**: Optional checkpointer support for saving workflow state
- **Graph Visualization**: Generate visual diagrams of the workflow
- **Logging Support**: Save analysis results to files for later review

## Requirements

- Python 3.10 or 3.11
- OpenAI API Key
- Poetry (dependency manager)

## Output Format

The workflow returns structured results in four formats:

### 1. Summary

A concise 2-3 sentence summary of key findings across all analysis steps.

```python
summary = workflow.get_summary()
# Example: "The dataset contains 500 rows and 8 columns with minimal missing data (< 5%). 
# Numeric distributions show moderate skewness in price columns, and correlation analysis 
# reveals strong relationships between quantity and total_sales."
```

### 2. Recommendations

A list of 3-5 actionable recommendations based on the analysis.

```python
recommendations = workflow.get_recommendations()
# Example:
# [
#   "Investigate outliers in the price column using domain knowledge",
#   "Consider imputation strategies for the 3.2% missing values in customer_id",
#   "Explore the strong correlation between quantity and total_sales for predictive modeling",
#   "Validate duplicate records in transaction_id column (2.1% duplication rate)"
# ]
```

### 3. Observations

Step-by-step observations extracted by the LLM after each analysis tool runs.

```python
observations = workflow.get_observations()
# Example:
# {
#   "profile_dataset": [
#     "Dataset has 500 rows and 8 columns",
#     "Contains 5 numeric and 3 categorical columns"
#   ],
#   "analyze_missingness": [
#     "Customer_id has 3.2% missing values",
#     "94.5% of rows are complete"
#   ],
#   ...
# }
```

### 4. Results

Raw pandas analysis results for each step (dictionaries with detailed statistics).

```python
results = workflow.get_results()
# Example:
# {
#   "profile_dataset": {
#     "shape": {"rows": 500, "columns": 8},
#     "dtypes": {...},
#     "numeric_summary": {...}
#   },
#   "analyze_missingness": {...},
#   ...
# }
```

## Workflow Architecture

The workflow uses **LangGraph** to orchestrate sequential analysis steps:

1. Each analysis step (e.g., profile_dataset, analyze_missingness) is a node
2. After each analysis node, an LLM extraction node generates observations
3. Analysis steps run deterministically using pandas
4. LLM steps interpret results and provide insights
5. Final synthesis node combines all observations into summary and recommendations

This architecture ensures:

- **Reproducibility**: Same data always produces same analysis results
- **Transparency**: Raw results available alongside LLM interpretations
- **Modularity**: Easy to add/remove analysis steps
- **Efficiency**: LLM only interprets, doesn't perform analysis

## Customization

### Using Different LLMs

The workflow supports any LangChain-compatible LLM:

```python
# Anthropic Claude
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

# Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

workflow = EDAWorkflow(model=llm)
```

### Modifying Prompts

Prompt templates are stored in `eda_workflow/prompts/`:

- `extract_observations_system.txt` - System prompt for observation extraction
- `extract_observations_human.txt` - Human prompt for observation extraction
- `synthesize_findings_system.txt` - System prompt for final synthesis
- `synthesize_findings_human.txt` - Human prompt for final synthesis

Edit these files to customize how the LLM interprets results.

## Limitations

- **Fixed Analysis Steps**: Cannot dynamically add/remove analysis tools at runtime
- **CSV Input Only**: Currently supports CSV files only (not Excel, JSON, etc.)
- **Token Limits**: Large datasets may hit LLM context windows; results are truncated automatically
- **English Only**: Prompts and outputs are in English
- **Pandas-Based**: Limited by pandas capabilities (not suitable for huge datasets that don't fit in memory)

## License

This project is provided as-is for educational and commercial use.
