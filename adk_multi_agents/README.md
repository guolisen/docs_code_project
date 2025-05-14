# Multi-Agent System with Google ADK

This repository contains a demonstration of a multi-agent system built using Google's Agent Development Kit (ADK). The system consists of specialized agents for billing and technical support, coordinated by a main routing agent.

## Files

- `main.py`: Python script version of the multi-agent system
- `multi_agent_demo.ipynb`: Jupyter notebook with detailed explanations, visualizations, and agent path analysis

## Prerequisites

Before running the code, ensure you have the following installed:

1. Python 3.13 or higher
2. Jupyter Notebook or JupyterLab (for notebook version)
3. [uv](https://github.com/astral-sh/uv) - A faster, more reliable Python package installer and resolver

### Install uv
Linux:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Windows:
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Environment Setup with uv

```bash
# Create a virtual environment
uv venv

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies from pyproject.toml
uv sync

# Or install dependencies manually
uv pip install google-adk google-generativeai openai matplotlib networkx
```

## API Configuration

This demo uses OpenAI's models through LiteLlm. You'll need to:

1. Set up an OpenAI API key
2. Configure the environment variables in the code:
   ```python
   os.environ['OPENAI_API_KEY'] = ''
   ```

## Running the Code

### Option 1: Running the Python Script

1. Navigate to the project directory in your terminal:
   ```
   cd /path/to/adk_multi_agents
   ```

2. Run the script:
   ```
   python main.py
   ```

### Option 2: Using Jupyter Notebook

1. Navigate to the project directory in your terminal:
   ```
   cd /path/to/adk_multi_agents
   ```

2. Launch Jupyter Notebook:
   ```
   jupyter notebook
   ```

3. In the browser window that opens, click on `multi_agent_demo.ipynb` to open the notebook.

4. Run the cells by clicking the "Run" button or pressing Shift+Enter for each cell.
   - You can run all cells at once by selecting "Cell" > "Run All" from the menu.

### Option 3: Using VS Code

1. Open VS Code and navigate to the project directory.

2. Install the "Jupyter" extension if you haven't already.

3. Open the `multi_agent_demo.ipynb` file.

4. Run cells using the "Run Cell" button that appears above each cell or by pressing Shift+Enter.

## What the Demo Does

The multi-agent system demonstrates:

1. **Agent Definition**: Setting up specialized agents for billing and technical support using LLMs (OpenAI's GPT-3.5-turbo model)

2. **Coordinator Architecture**: Creating a coordinator agent that intelligently routes requests to the appropriate specialized agent

3. **Query Processing**: How different user queries are processed through the multi-agent system

4. **Agent Invoke Path Analysis**: Extracting and visualizing the path a query takes through the multi-agent system:
   - Support queries: Coordinator → Support Agent
   - Billing queries: Coordinator → Billing Agent

5. **Visualization**: The notebook includes functions to visualize agent invoke paths using networkx and matplotlib

## Example Queries

The demo includes examples of different query types:

1. **Support Query**: "I can't access my email account, it says my password is incorrect"
   - This query is routed from the Coordinator to the Support Agent

2. **Billing Query**: "I was charged twice for my subscription last month"
   - This query is routed from the Coordinator to the Billing Agent

## Troubleshooting

- **ModuleNotFoundError**: If you encounter errors about missing modules, ensure you've installed all required packages using `uv pip install` or `uv sync`.

- **API Key Issues**: If you encounter authentication errors, check that your OpenAI API key is properly configured.

- **Memory Issues**: If the notebook runs out of memory, try restarting the kernel and running only the cells you need.

## Additional Resources

- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [LiteLlm Documentation](https://litellm.ai/docs)
