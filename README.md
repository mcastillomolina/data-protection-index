# Data Protection Index

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Document Discovery | ✅ Complete |
| **Phase 2** | Document Retrieval & Text Extraction | ✅ Complete |
| **Phase 3** | Information Extraction & Storage | 📋 Planned |

---

## Phase 1: Document Discovery

✅ **Complete** — Automated AI-powered system for discovering data protection legal documents across countries using LLMs and web search.

### Overview

Phase 1 is a complete 4-step AI pipeline:

1. **Document Identification** - LLM identifies relevant legal documents for a country
2. **Query Generation** - LLM generates optimized search queries (5 per document)
3. **Web Search** - Executes searches via SerpAPI with deduplication
4. **Relevance Filtering** - LLM scores results (0-10) and returns top URLs

**Input:** Country name
**Output:** `data/outputs/{country}/discovery_results_latest.json` — top 5 URLs per document type, scored and ranked

---

## Phase 2: Document Retrieval & Text Extraction

✅ **Complete** — Downloads the discovered URLs and extracts clean text from PDFs and HTML pages, producing a structured text corpus for Phase 3.

### Overview

Phase 2 is a deterministic (no LLM) 2-step pipeline that runs automatically after Phase 1:

1. **Document Retrieval** - Downloads content from discovered URLs via HTTP with retry/backoff
2. **Text Extraction** - Extracts clean text from PDFs (pdfplumber) and HTML pages (BeautifulSoup)

For each document, Phase 2 tries all top URLs in relevance order and uses the first that yields usable text.

**Input:** `data/outputs/{country}/discovery_results_latest.json` (Phase 1 output)
**Output:** `data/outputs/{country}/retrieval_results_latest.json` — extracted text per document

## Quick Start

```bash
# 1. Activate virtual environment
pyenv activate dpi

# 2. Set up API keys in .env
cp .env.example .env
# Edit .env: Add GROQ_API_KEY (free), OPENAI_API_KEY or ANTHROPIC_API_KEY, and SERPAPI_KEY

# 3. Run the full pipeline for Chile (Phase 1 + Phase 2)
python -m src.main Chile --verbose

# 4. Check Phase 1 results (discovered URLs)
cat data/outputs/Chile/discovery_results_latest.json

# 5. Check Phase 2 results (extracted text)
cat data/outputs/Chile/retrieval_results_latest.json
```

## Features

✅ **LLM Integration** (Phase 1)
- Groq (Llama 3.3, Mixtral, Gemma2 — free tier available)
- OpenAI (GPT-4, GPT-4 Turbo, GPT-4o, GPT-3.5)
- Anthropic (Claude 3 Opus, Sonnet, Haiku)
- Switchable via `config/config.yaml` — one provider per run
- Automatic retry with exponential backoff
- Cost tracking per request

✅ **Smart Search** (Phase 1)
- SerpAPI integration (Google Search)
- Country and language localization
- Site restrictions (official government domains)
- Rate limiting and deduplication

✅ **AI-Powered Scoring** (Phase 1)
- 0-10 relevance scoring with reasoning
- Authority source detection
- Confidence levels (high/medium/low)
- Batch processing for large result sets

✅ **Document Retrieval** (Phase 2)
- HTTP download with retry and exponential backoff
- Automatic content-type detection (PDF vs HTML)
- Tries all top URLs in relevance order; uses first with extractable text
- No LLM cost — fully deterministic

✅ **Text Extraction** (Phase 2)
- PDF extraction via pdfplumber (no temp files)
- HTML extraction via BeautifulSoup/lxml (strips scripts, nav, footer)
- Text cleaning: whitespace normalization, null-byte removal
- Minimum length threshold to filter empty/garbled extractions

✅ **Complete Pipeline**
- Phases 1 & 2 run automatically in sequence with a single command
- `--discovery-only` flag to run Phase 1 alone
- Progress tracking and comprehensive error handling
- JSON output with metadata at every stage

## Installation

```bash
# Activate the virtualenv
pyenv activate dpi

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
```

Edit `.env`:
```bash
OPENAI_API_KEY=sk-...        # Or ANTHROPIC_API_KEY or GROQ_API_KEY (free)
SERPAPI_KEY=...              # Get free 100 searches/month
```

## Usage

### Basic

```bash
# Full pipeline: Phase 1 (discovery) + Phase 2 (retrieval)
python -m src.main Chile
```

### With Options

```bash
# Verbose output with progress bars
python -m src.main Chile --verbose

# Phase 1 only (skip document retrieval)
python -m src.main Chile --discovery-only

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

Results are saved in two files per country under `data/outputs/{country_name}/`:

### Phase 1: `discovery_results_latest.json`

```json
{
  "country": { "name": "Chile", "iso_code": "CL", ... },
  "documents": [
    {
      "document": {
        "official_name": "Ley 19.628 sobre Protección de la Vida Privada",
        "document_type": "data_protection_law",
        "priority_score": 10
      },
      "top_results": [
        {
          "search_result": { "url": "https://www.bcn.cl/...", "title": "Ley 19628" },
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
  "metadata": { "processing_time_seconds": 247.3, "llm_model": "llama-3.3-70b-versatile" }
}
```

### Phase 2: `retrieval_results_latest.json`

```json
{
  "country": { "name": "Chile", "iso_code": "CL", ... },
  "documents": [
    {
      "document": { "official_name": "Ley 19.628 ...", "document_type": "data_protection_law" },
      "status": "success",
      "successful_url": "https://www.dipres.gob.cl/.../ley19628.pdf",
      "attempted_urls": ["https://...", "https://..."],
      "content": {
        "url": "https://www.dipres.gob.cl/.../ley19628.pdf",
        "content_type": "pdf",
        "extracted_text": "LEY Nº 19.628 SOBRE PROTECCIÓN DE LA VIDA PRIVADA...",
        "char_count": 14832,
        "extraction_success": true
      }
    }
  ],
  "total_documents": 8,
  "successful_retrievals": 7,
  "failed_retrievals": 1,
  "metadata": { "phase": "2", "processing_time_seconds": 38.2 }
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
# Unit tests only (fast, no API calls)
pytest tests/unit/ -v

# With coverage
pytest tests/unit/ --cov=src --cov-report=html

# Integration tests (require API keys + network)
pytest tests/integration/ -v

# Run all tests
pytest --cov=src --cov-report=html
```

## Project Structure

```
src/
├── main.py                    # CLI entry point (orchestrates Phase 1 + Phase 2)
├── core/                      # Pipeline components
│   ├── document_identifier.py # [Phase 1] LLM document identification
│   ├── query_generator.py     # [Phase 1] LLM query generation
│   ├── search_executor.py     # [Phase 1] SerpAPI search execution
│   ├── relevance_filter.py    # [Phase 1] LLM relevance scoring
│   ├── country_resolver.py    # Country lookup with LLM enrichment fallback
│   ├── document_retriever.py  # [Phase 2] HTTP download with retry
│   └── text_extractor.py      # [Phase 2] PDF and HTML text extraction
├── models/                    # Pydantic data models
│   ├── country.py
│   ├── document.py            # Phase 1 models (DocumentMetadata, DiscoveryOutput, ...)
│   └── retrieval.py           # Phase 2 models (DocumentContent, RetrievalOutput, ...)
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
├── config.yaml                # Main configuration (llm, search, pipeline, retrieval)
├── document_types.yaml        # Document type definitions
└── countries.yaml             # Country metadata (auto-enriched for unknown countries)

data/outputs/{country}/
├── discovery_results_latest.json   # Phase 1 output: top scored URLs per document
└── retrieval_results_latest.json   # Phase 2 output: extracted text per document

tests/
├── unit/                      # Fast unit tests (no API calls)
│   ├── test_document_retriever.py
│   ├── test_text_extractor.py
│   └── test_search_client.py
└── integration/               # End-to-end tests (require API keys)
```

## Configuration

Edit `config/config.yaml` to customize:
- `llm` — provider (openai/anthropic/groq), model, temperature
- `search` — max results per query, rate limiting
- `pipeline` — min relevance score, deduplication
- `retrieval` — HTTP timeout, retries, min text length for extraction
- `output` — output directory, format

Add countries in `config/countries.yaml`. Unknown countries are automatically enriched via LLM and cached back to the file.

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
