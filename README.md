# LangGraph_Research_Assistant_Agent

## Setup on a new machine

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh   # Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Clone and install (uv fetches Python 3.12 automatically, deps pinned by uv.lock)
git clone https://github.com/ARNAB-bughunter/LangGraph_Research_Assistant_Agent.git
cd LangGraph_Research_Assistant_Agent
uv sync

# 3. Add your keys
cp .env.example .env   # then fill in the values

# 4. Run the LangGraph dev server
uv run langgraph dev
```
