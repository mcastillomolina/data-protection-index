# Data Protection Index

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Document Discovery | ✅ Complete |
| **Phase 2** | Document Retrieval & Processing | 🔜 Next |
| **Phase 3** | Information Extraction & Storage | 📋 Planned |

---

## Phase 1: Document Discovery

✅ **Complete** — Automated AI-powered system for discovering data protection legal documents across countries using LLMs and web search.

## Overview

Phase 1 is a complete 4-step AI pipeline:

1. **Document Identification** - LLM identifies relevant legal documents for a country
2. **Query Generation** - LLM generates optimized search queries (5 per document)
3. **Web Search** - Executes searches via SerpAPI with deduplication
4. **Relevance Filtering** - LLM scores results (0-10) and returns top URLs

**Input:** Country name
**Output:** JSON file with top 5 URLs per document type, scored and ranked

## Quick Start

```bash
# 1. Activate virtual environment
pyenv activate dpi

# 2. Set up API keys in .env
cp .env.example .env
# Edit .env: Add GROQ_API_KEY (free), OPENAI_API_KEY or ANTHROPIC_API_KEY, and SERPAPI_KEY

# 3. Run discovery for Chile
python -m src.main Chile --verbose

# 4. Check results
cat data/outputs/Chile/discovery_results_latest.json
```

## Features

✅ **LLM Integration**
- Groq (Llama 3.3, Mixtral, Gemma2 — free tier available)
- OpenAI (GPT-4, GPT-4 Turbo, GPT-4o, GPT-3.5)
- Anthropic (Claude 3 Opus, Sonnet, Haiku)
- Switchable via `config/config.yaml` — one provider per run
- Automatic retry with exponential backoff
- Cost tracking per request

✅ **Smart Search**
- SerpAPI integration (Google Search)
- Country and language localization
- Site restrictions (official government domains)
- Rate limiting and deduplication

✅ **AI-Powered Scoring**
- 0-10 relevance scoring with reasoning
- Authority source detection
- Confidence levels (high/medium/low)
- Batch processing for large result sets

✅ **Complete Pipeline**
- End-to-end orchestration
- Progress tracking
- Comprehensive error handling
- JSON output with metadata

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
```

Edit `.env`:
```bash
OPENAI_API_KEY=sk-...        # Or ANTHROPIC_API_KEY
SERPAPI_KEY=...              # Get free 100 searches/month
```

## Usage

### Basic

```bash
python -m src.main Chile
```

### With Options

```bash
# Verbose output with progress bars
python -m src.main Chile --verbose

# Limit scope (faster/cheaper)
python -m src.main Chile --max-documents 3 --queries-per-doc 3 -v

# Custom output directory
python -m src.main Germany --output-dir ./results

# Full help
python -m src.main --help
```

### Multiple Countries

```bash
for country in Chile Germany "United Kingdom"; do
    python -m src.main "$country" -v
done
```

## Output

Results saved to: `data/outputs/{country_name}/discovery_results_{timestamp}.json`

Structure:
```json
{
  "country": {
    "name": "Chile",
    "iso_code": "CL",
    "official_languages": ["es"],
    "region": "Latin America"
  },
  "documents": [
    {
      "document": {
        "official_name": "Ley 19.628 sobre Protección de la Vida Privada",
        "document_type": "data_protection_law",
        "priority_score": 10,
        "expected_language": "es"
      },
      "top_results": [
        {
          "search_result": {
            "url": "https://www.bcn.cl/leychile/navegar?idNorma=141599",
            "title": "Ley 19628 - Protección de la Vida Privada"
          },
          "relevance_score": 9.5,
          "is_likely_official": true,
          "confidence": "high",
          "reasoning": "Official legislative database..."
        }
      ]
    }
  ],
  "total_documents_identified": 8,
  "total_urls_found": 34,
  "metadata": {
    "processing_time_seconds": 247.3,
    "llm_model": "gpt-4o-mini",
    "search_provider": "serpapi"
  }
}
```

## Cost Estimation

Per country (approximate):
- **LLM:** ~$0.26 (with GPT-4o-mini)
- **Search:** Free tier (100/month) or ~$0.40
- **Total:** ~$0.26-0.66 per country

See [README_USAGE.md](README_USAGE.md) for cost reduction tips.

## Testing

```bash
# Test individual components
python test_llm_clients.py        # Test LLM clients
python test_search_client.py      # Test search client
python test_prompts.py            # Test prompts
python test_core_pipeline.py      # Test full pipeline

# Unit tests
pytest tests/unit/ -v

# With coverage
pytest --cov=src --cov-report=html
```

## Project Structure

```
src/
├── main.py                    # CLI entry point
├── core/                      # Pipeline components
│   ├── document_identifier.py # LLM document identification
│   ├── query_generator.py     # LLM query generation
│   ├── search_executor.py     # Search execution
│   └── relevance_filter.py    # LLM relevance scoring
├── models/                    # Pydantic data models
│   ├── country.py
│   └── document.py
├── clients/                   # API clients
│   ├── llm_client.py          # Abstract base
│   ├── groq_client.py         # Groq implementation (free tier)
│   ├── openai_client.py       # OpenAI implementation
│   ├── anthropic_client.py    # Anthropic implementation
│   └── search_client.py       # SerpAPI client
├── prompts/                   # LLM prompt templates
│   ├── document_identification.py
│   ├── query_generation.py
│   └── relevance_scoring.py
└── utils/                     # Utilities
    ├── config.py              # Configuration management
    └── logger.py              # Logging setup

config/
├── config.yaml                # Main configuration
├── document_types.yaml        # Document type definitions
└── countries.yaml             # Country metadata

data/outputs/                  # Discovery results (Phase 1 output / Phase 2 input)
tests/                         # Unit and integration tests
```

## Configuration

Edit `config/config.yaml` to customize:
- LLM provider and model
- Search settings
- Pipeline parameters
- Output formats

Add countries in `config/countries.yaml`

See [README_USAGE.md](README_USAGE.md) for detailed configuration guide.

## Documentation

- **[README_USAGE.md](README_USAGE.md)** - Complete usage guide
- **[IMPLEMENTATION_PROGRESS.md](garbage/IMPLEMENTATION_PROGRESS.md)** - Implementation tracker
- **[PHASE1_IMPLEMENTATION_PLAN.md](garbage/PHASE1_IMPLEMENTATION_PLAN.md)** - Architecture details
- **Component docs:**
  - [LLM_CLIENTS_IMPLEMENTATION.md](LLM_CLIENTS_IMPLEMENTATION.md)
  - [SEARCH_CLIENT_IMPLEMENTATION.md](SEARCH_CLIENT_IMPLEMENTATION.md)
  - [PROMPTS_IMPLEMENTATION.md](PROMPTS_IMPLEMENTATION.md)

## Implementation Stats

**Total Lines of Code:** ~10,000+
- Core pipeline: 1,360 lines
- Clients: 900 lines
- Prompts: 1,060 lines
- Models: 400 lines
- Tests: 2,000+ lines
- Configuration: 200 lines
- Documentation: 4,000+ lines

**Components:**
- ✅ 4 Core pipeline classes
- ✅ 3 LLM client implementations
- ✅ 1 Search client
- ✅ 3 Prompt template modules
- ✅ 2 Data model modules
- ✅ Configuration system
- ✅ Logging system
- ✅ CLI interface

## Supported Countries

Pre-configured in `config/countries.yaml`:
- Chile
- Germany
- United Kingdom

Add more by editing the config file.

## Requirements

- Python 3.10+
- API keys:
  - OpenAI OR Anthropic (for LLM)
  - SerpAPI (for search)

---

## Phase 2: Document Retrieval & Processing (Next)

Phase 1 produces a JSON file with the top 5 URLs per document type per country. Phase 2 takes those URLs and processes the actual documents.

**Goals:**
1. **Document Retrieval** — Download PDFs and HTML pages from the discovered URLs
2. **Text Extraction** — Parse PDFs (pdfplumber/pypdf) and HTML (BeautifulSoup) into clean text
3. **Document Storage** — Store raw + extracted text, keyed by country + document type

**Input:** `data/outputs/{country}/discovery_results_latest.json` (Phase 1 output)
**Output:** Structured text corpus per country, ready for Phase 3 extraction

---

## Phase 3: Information Extraction & Storage (Planned)

Phase 3 uses LLMs to extract structured data protection information from the processed documents.

**Goals:**
1. **Structured Extraction** — Key provisions, enforcement mechanisms, penalties, rights
2. **Database Storage** — Persistent store (PostgreSQL or similar) indexed by country + topic
3. **Comparison Layer** — Cross-country analysis and index scoring

## License

See [LICENSE](LICENSE) file.

## Support

- Documentation: See docs folder
- Issues: Check logs in `logs/discovery.log`
- Config: Review `config/config.yaml`
