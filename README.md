# Data Protection Index

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Document Discovery | ✅ Complete |
| **Phase 2** | Document Retrieval & Text Extraction | ✅ Complete |
| **Phase 3** | Information Extraction & PostgreSQL Storage | ✅ Complete |

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
# Edit .env: Add GROQ_API_KEY (free), OPENAI_API_KEY or ANTHROPIC_API_KEY, SERPAPI_KEY,
# and DATABASE_URL=postgresql://dpi:dpi@localhost:5433/dpi

# 3. Start PostgreSQL (Phase 3)
docker compose up -d

# 4. Run the full pipeline for Chile (Phase 1 + Phase 2 + Phase 3)
python -m src.main Chile --verbose

# 5. Check results
cat data/outputs/Chile/discovery_results_latest.json   # Phase 1: discovered URLs
cat data/outputs/Chile/retrieval_results_latest.json   # Phase 2: extracted text
cat data/outputs/Chile/extraction_results_latest.json  # Phase 3: structured data
```

## Features

✅ **LLM Integration** (Phases 1 & 3)
- Groq (Llama 3.3, Mixtral, Gemma2 — free tier available)
- OpenAI (GPT-4, GPT-4 Turbo, GPT-4o, GPT-3.5)
- Anthropic (Claude 3 Opus, Sonnet, Haiku)
- Mistral
- Switchable via `config/config.yaml` — separate providers for discovery (Phase 1) and extraction (Phase 3)
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

✅ **Structured Information Extraction** (Phase 3)
- Language detection via langdetect (deterministic, no LLM)
- Three-tier section splitting: universal article patterns → language-specific patterns → paragraph fallback
- LLM extraction per section — six fields: key provisions, subject rights, enforcement body, penalties, lawful basis, notes
- Always extracts in English regardless of source document language
- Cross-section aggregation with deduplication

✅ **PostgreSQL Storage** (Phase 3)
- Four normalized tables: countries, documents, section_extractions, document_extractions
- GIN indexes on JSONB `extracted_fields` for fast querying
- Idempotent upserts — safe to re-run without duplicating data
- Docker Compose for local dev (PostgreSQL 16-alpine, port 5433)
- `--skip-db` flag for JSON-only output without database

✅ **Complete Pipeline**
- All three phases run automatically in sequence with a single command
- `--discovery-only`, `--skip-extraction`, `--extraction-only`, `--skip-db` flags for partial runs
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
DATABASE_URL=postgresql://dpi:dpi@localhost:5433/dpi  # Phase 3
```

Start PostgreSQL (Phase 3):
```bash
docker compose up -d
```

## Usage

### Basic

```bash
# Full pipeline: Phase 1 + Phase 2 + Phase 3
python -m src.main Chile
```

### With Options

```bash
# Verbose output with progress bars
python -m src.main Chile --verbose

# Phase 1 only (skip retrieval and extraction)
python -m src.main Chile --discovery-only

# Phases 1+2 only (skip Phase 3 extraction)
python -m src.main Chile --skip-extraction

# Phase 3 only (reads existing retrieval_results_latest.json)
python -m src.main Chile --extraction-only

# Phase 3 without DB writes (JSON output only)
python -m src.main Chile --extraction-only --skip-db

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

Results are saved in three files per country under `data/outputs/{country_name}/`:

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

### Phase 3: `extraction_results_latest.json`

```json
{
  "country": { "name": "Chile", "iso_code": "CL", ... },
  "documents": [
    {
      "document": { "official_name": "Ley 19.628 ...", "document_type": "data_protection_law" },
      "detected_language": "es",
      "split_tier_used": "tier1",
      "total_sections": 45,
      "sections_with_signal": 38,
      "enforcement_authority": "Consejo para la Transparencia",
      "aggregated_fields": {
        "key_provisions": ["Personal data must be collected for specific, explicit purposes", ...],
        "data_subject_rights": ["Right of access", "Right to rectification", "Right to deletion", ...],
        "enforcement_body": "Consejo para la Transparencia",
        "penalties": ["Fines up to 50 UTM for minor violations", "Up to 100 UTM for serious violations"],
        "lawful_basis": ["Consent of the data subject", "Legal obligation", "Public interest"],
        "notes": "[§12] Cross-border transfer provisions may conflict with GDPR adequacy requirements"
      },
      "status": "success",
      "processing_time_seconds": 62.4,
      "llm_provider": "groq",
      "llm_model": "llama-3.3-70b-versatile"
    }
  ],
  "total_documents": 8,
  "successful_extractions": 7,
  "failed_extractions": 1,
  "metadata": { "phase": "3", "processing_time_seconds": 180.2 }
}
```

## Cost Estimation

Per country (approximate):
- **Phase 1 LLM:** ~$0.26 (with GPT-4o-mini)
- **Search:** Free tier (100/month) or ~$0.40
- **Phase 3 LLM:** ~$0.05-0.20 (with Groq free tier or GPT-4o-mini; ~300-400 calls per country)
- **Total:** ~$0.31-0.86 per country

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
├── main.py                      # CLI entry point (orchestrates Phases 1 + 2 + 3)
├── core/                        # Pipeline components
│   ├── document_identifier.py   # [Phase 1] LLM document identification
│   ├── query_generator.py       # [Phase 1] LLM query generation
│   ├── search_executor.py       # [Phase 1] SerpAPI search execution
│   ├── relevance_filter.py      # [Phase 1] LLM relevance scoring
│   ├── country_resolver.py      # Country lookup with LLM enrichment fallback
│   ├── document_retriever.py    # [Phase 2] HTTP download with retry
│   ├── text_extractor.py        # [Phase 2] PDF and HTML text extraction
│   ├── language_detector.py     # [Phase 3] langdetect wrapper (no LLM)
│   ├── section_splitter.py      # [Phase 3] Three-tier regex section splitter
│   └── information_extractor.py # [Phase 3] LLM per-section extraction + aggregation
├── db/                          # Database layer (Phase 3)
│   ├── schema.py                # PostgreSQL DDL (CREATE TABLE IF NOT EXISTS)
│   └── writer.py                # DatabaseWriter — idempotent upserts via psycopg2
├── models/                      # Pydantic data models
│   ├── country.py
│   ├── document.py              # Phase 1 models (DocumentMetadata, DiscoveryOutput, ...)
│   ├── retrieval.py             # Phase 2 models (DocumentContent, RetrievalOutput, ...)
│   └── extraction.py            # Phase 3 models (SectionExtractionResult, ExtractionOutput, ...)
├── clients/                     # API clients
│   ├── llm_client.py            # Abstract base
│   ├── groq_client.py           # Groq implementation (free tier)
│   ├── openai_client.py         # OpenAI implementation
│   ├── anthropic_client.py      # Anthropic implementation
│   ├── mistral_client.py        # Mistral implementation
│   └── search_client.py         # SerpAPI client
├── prompts/                     # LLM prompt templates
│   ├── document_identification.py
│   ├── query_generation.py
│   ├── relevance_scoring.py
│   └── information_extraction.py  # [Phase 3] Per-section extraction prompt + schema
└── utils/                       # Utilities
    ├── config.py                # Configuration management (includes ExtractionConfig)
    └── logger.py                # Logging setup

config/
├── config.yaml                  # Main configuration (llm, search, pipeline, retrieval, extraction)
├── document_types.yaml          # Document type definitions
├── countries.yaml               # Country metadata (auto-enriched for unknown countries)
└── extraction_schema.yaml       # Per-document-type field expectations for Phase 3

docker-compose.yml               # PostgreSQL 16-alpine on port 5433

data/outputs/{country}/
├── discovery_results_latest.json    # Phase 1 output: top scored URLs per document
├── retrieval_results_latest.json    # Phase 2 output: extracted text per document
└── extraction_results_latest.json   # Phase 3 output: structured extraction + aggregated fields

tests/
├── unit/                        # Fast unit tests (no API calls, all I/O mocked)
│   ├── test_document_retriever.py
│   ├── test_text_extractor.py
│   ├── test_search_client.py
│   ├── test_language_detector.py
│   ├── test_section_splitter.py
│   ├── test_information_extractor.py
│   └── test_database_writer.py
└── integration/                 # End-to-end tests (require API keys + DB)
    └── test_phase3_pipeline.py
```

## Configuration

Edit `config/config.yaml` to customize:
- `llm` — provider (openai/anthropic/groq/mistral), model, temperature (Phase 1)
- `search` — max results per query, rate limiting
- `pipeline` — min relevance score, deduplication
- `retrieval` — HTTP timeout, retries, min text length for extraction
- `extraction` — llm_provider, llm_model, min_section_chars (Phase 3, separate from Phase 1 LLM)
- `output` — output directory, format

Add countries in `config/countries.yaml`. Unknown countries are automatically enriched via LLM and cached back to the file.

Set `DATABASE_URL` in `.env` and run `docker compose up -d` before using Phase 3.

See [README_USAGE.md](README_USAGE.md) for detailed configuration guide.

## Caching

The pipeline caches every expensive operation so repeat runs are fast and free. All file caches are stored under `data/cache/`; phase output files double as caches for downstream phases.

### Cache layers

| Layer | Location | Keyed on | Phase |
|---|---|---|---|
| Document identification | `data/cache/documents/` | country ISO + known-docs fingerprint | P1 Step 1 |
| Query generation | `data/cache/queries/` | doc name + ISO + n_queries | P1 Step 2 |
| Web search results | `data/cache/search/` | query string + country + language | P1 Step 3 |
| Relevance scoring | `data/cache/relevance/` | doc name + country + sorted URLs | P1 Step 4 |
| Document retrieval | `data/outputs/{Country}/retrieval_results_latest.json` | per URL | P2 |
| Extraction results | `data/outputs/{Country}/extraction_results_latest.json` | per document name | P3 |
| Criterion scores | PostgreSQL `criterion_scores` table | country + criterion + year + model | P4 |

### How it works per phase

- **P1** — All four LLM/search steps check their cache first. A full cache hit means zero LLM calls and zero SerpAPI calls.
- **P2** — URLs already present in the previous `retrieval_results_latest.json` are served from disk; only new URLs trigger HTTP downloads.
- **P3** — Documents with a non-failed entry in `extraction_results_latest.json` are skipped entirely; their cached `aggregated_fields` flow straight into the output.
- **P4** — Criteria already recorded in `criterion_scores` for the same model and year are returned from the DB without calling the LLM.

### Disabling the cache

**Per run** (recommended for first runs or debugging):
```bash
python -m src.main Canada --no-cache
```

**Permanently** — set in `config/config.yaml`:
```yaml
pipeline:
  enable_caching: false
```

**Selectively** — delete the relevant directory or file:
```bash
rm -rf data/cache/documents/   # re-identify docs on next run
rm -rf data/cache/search/      # re-execute web searches
rm data/outputs/Canada/extraction_results_latest.json  # re-extract Canada
```

### Cache TTL

Web search results (`data/cache/search/`) expire after **30 days** (controlled by `pipeline.cache_ttl_seconds` in `config.yaml`). All other file caches have no TTL — delete the file to force a refresh. PostgreSQL criterion scores are overwritten on re-run only when `--no-cache` is passed.

## Documentation

- **[README_USAGE.md](README_USAGE.md)** - Complete usage guide
- **[IMPLEMENTATION_PROGRESS.md](garbage/IMPLEMENTATION_PROGRESS.md)** - Implementation tracker
- **[PHASE1_IMPLEMENTATION_PLAN.md](garbage/PHASE1_IMPLEMENTATION_PLAN.md)** - Architecture details
- **Component docs:**
  - [LLM_CLIENTS_IMPLEMENTATION.md](LLM_CLIENTS_IMPLEMENTATION.md)
  - [SEARCH_CLIENT_IMPLEMENTATION.md](SEARCH_CLIENT_IMPLEMENTATION.md)
  - [PROMPTS_IMPLEMENTATION.md](PROMPTS_IMPLEMENTATION.md)

## Implementation Stats

**Total Lines of Code:** ~12,000+
- Core pipeline: ~1,700 lines (Phase 3 adds language_detector, section_splitter, information_extractor)
- Database layer: ~300 lines (schema.py + writer.py)
- Clients: ~900 lines (4 LLM providers + search)
- Prompts: ~1,100 lines (Phase 3 adds information_extraction.py)
- Models: ~460 lines (Phase 3 adds extraction.py)
- Tests: ~2,600+ lines (Phase 3 adds 4 unit + 1 integration test file)
- Configuration: ~255 lines (config.yaml + extraction_schema.yaml)
- Documentation: 4,000+ lines

**Components:**
- ✅ 7 Core pipeline classes (4 Phase 1, 2 Phase 2, 3 Phase 3)
- ✅ 4 LLM client implementations (OpenAI, Anthropic, Groq, Mistral)
- ✅ 1 Search client
- ✅ 4 Prompt template modules
- ✅ 3 Data model modules
- ✅ PostgreSQL persistence layer (schema + idempotent writer)
- ✅ Docker Compose for local database
- ✅ Configuration system (separate extraction LLM config)
- ✅ Logging system
- ✅ CLI interface (6 run-mode flags)

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

## Phase 3: Information Extraction & Storage

✅ **Complete** — LLM-based structured extraction from processed documents, persisted in PostgreSQL for querying and cross-country comparison.

### Overview

Phase 3 is a 4-step pipeline that runs automatically after Phase 2:

1. **Language Detection** — Identifies document language (ISO 639-1) using langdetect (deterministic, no LLM)
2. **Section Splitting** — Splits text using a three-tier regex fallback: universal article patterns → language-specific labels → paragraph chunks
3. **Per-Section Extraction** — One LLM call per section extracts six fields in English: key provisions, data subject rights, enforcement body, penalties, lawful basis, notes
4. **Aggregation + Storage** — Merges section results (dedup lists, first-wins scalars) and upserts into PostgreSQL + JSON output

**Input:** `data/outputs/{country}/retrieval_results_latest.json` (Phase 2 output)
**Output:**
- `data/outputs/{country}/extraction_results_latest.json` — structured extraction per document
- PostgreSQL tables: `countries`, `documents`, `section_extractions`, `document_extractions`

## License

See [LICENSE](LICENSE) file.

## Support

- Documentation: See docs folder
- Issues: Check logs in `logs/discovery.log`
- Config: Review `config/config.yaml`
