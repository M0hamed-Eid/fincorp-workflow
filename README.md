# FinCorp Multi-Source LCEL Workflow

A proof of concept for routing financial and market-news queries,
retrieving mock data concurrently, and validating LLM outputs
with Pydantic.

## Technologies

- Python 3.12
- LangChain Core / LCEL
- OpenAI GPT-4o-mini
- Pydantic
- uv

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/).

Clone the repository:

```bash
git clone https://github.com/M0hamed-Eid/fincorp-workflow.git
cd fincorp-workflow
```

Install the locked dependencies:

```bash
uv sync --locked
```

Create the local environment file:

```bash
cp .env.example .env
```

Edit `.env` and set your OpenAI API key:

```dotenv
OPENAI_API_KEY=your-openai-api-key
```

## Run

```bash
uv run python workflow_architecture.py
```

Running with the real OpenAI model requires an API key and may
incur API usage charges.

## Project Files

| File | Purpose |
|---|---|
| `workflow_architecture.py` | LCEL workflow implementation |
| `fincorp_data_schema.md` | Provided data schema reference |
| `design_proposal.md` | Architecture proposal |
| `pyproject.toml` | Project configuration and dependencies |
| `uv.lock` | Locked dependency versions |
| `.env.example` | Environment variable template |

## Architecture Proposal

Read the [design proposal](design_proposal.md).