# Data Protection Index - Phase 1: Document Discovery

Automated system for discovering and collecting data protection legal documents across countries.

## Overview

Phase 1 automatically:
1. Identifies relevant legal documents for a country (using LLM)
2. Generates targeted search queries
3. Executes web searches via SerpAPI
4. Filters results by relevance (using LLM)
5. Outputs top 3-5 URLs per document type

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env and add your API keys
```

## Usage

```bash
# Run discovery for a country
python -m src.main Chile

# Verbose output
python -m src.main Chile --verbose

# Custom config
python -m src.main Germany --config config/config.yaml
```

## Output

Results are saved to `data/outputs/{country_name}/discovery_results.json`

Example output structure:
```json
{
  "country": {"name": "Chile", ...},
  "documents": [
    {
      "document": {"document_type": "data_protection_law", ...},
      "top_results": [
        {"url": "...", "relevance_score": 9.5, ...}
      ]
    }
  ]
}
```

## Testing

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# With coverage
pytest --cov=src
```

## Project Structure

```
src/
├── core/              # Pipeline components
├── models/            # Data models
├── clients/           # API clients (LLM, search)
├── utils/             # Utilities (config, logging)
└── prompts/           # LLM prompt templates
```

## Documentation

See `PHASE1_IMPLEMENTATION_PLAN.md` for detailed architecture.
See `BRANCHING_STRATEGY.md` for development workflow.
