"""DDL statements for Phase 3 PostgreSQL schema."""

CREATE_CRITERIA = """
CREATE TABLE IF NOT EXISTS criteria (
    id             INTEGER PRIMARY KEY,
    name           VARCHAR(100) NOT NULL,
    description    TEXT,
    dimension      VARCHAR(20)  NOT NULL,
    document_types TEXT[]       DEFAULT '{}',
    rubric         TEXT
);
"""

SEED_CRITERIA = """
INSERT INTO criteria (id, name, dimension, document_types) VALUES
  (1,  'Constitutional protection',                 'legal',
        ARRAY['constitution','court_decision']),
  (2,  'Statutory protection',                      'legal',
        ARRAY['data_protection_law','regulation','dpa_regulation']),
  (3,  'Privacy enforcement',                       'enforcement',
        ARRAY['enforcement_report','dpa_annual_report']),
  (4,  'Identity cards and biometrics',             'mixed',
        ARRAY['biometrics_id_law']),
  (5,  'Data-sharing',                              'mixed',
        ARRAY['data_protection_law','international_treaty']),
  (6,  'Visual surveillance',                       'enforcement',
        ARRAY['surveillance_law']),
  (7,  'Communication interception',                'mixed',
        ARRAY['surveillance_law','data_retention_law']),
  (8,  'Workplace monitoring',                      'enforcement',
        ARRAY['workplace_privacy_law']),
  (9,  'Government access to data',                 'mixed',
        ARRAY['data_protection_law','surveillance_law']),
  (10, 'Communications data retention',             'legal',
        ARRAY['data_retention_law']),
  (11, 'Surveillance of medical/financial/movement','enforcement',
        ARRAY['surveillance_law']),
  (12, 'Border and trans-border issues',            'enforcement',
        ARRAY['border_surveillance_law','international_treaty']),
  (13, 'Leadership',                                'legal',
        ARRAY['international_treaty']),
  (14, 'Democratic safeguards',                     'enforcement',
        ARRAY['court_decision'])
ON CONFLICT (id) DO NOTHING;
"""

CREATE_COUNTRIES = """
CREATE TABLE IF NOT EXISTS countries (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(255) NOT NULL,
    iso_code    CHAR(2)      NOT NULL UNIQUE,
    region      VARCHAR(100),
    languages   TEXT[],
    aliases     TEXT[]    DEFAULT '{}',
    created_at  TIMESTAMP DEFAULT NOW()
);
"""

CREATE_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS documents (
    id                SERIAL PRIMARY KEY,
    country_id        INTEGER REFERENCES countries(id) ON DELETE CASCADE,
    document_type     VARCHAR(100) NOT NULL,
    official_name     TEXT        NOT NULL,
    source_url        TEXT,
    content_type      VARCHAR(50),
    char_count        INTEGER,
    detected_language VARCHAR(10),
    criteria_ids      INTEGER[]   DEFAULT '{}',
    retrieved_at      TIMESTAMP,
    created_at        TIMESTAMP DEFAULT NOW(),
    UNIQUE (country_id, official_name)
);
"""

CREATE_SECTION_EXTRACTIONS = """
CREATE TABLE IF NOT EXISTS section_extractions (
    id                      SERIAL PRIMARY KEY,
    document_id             INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    section_index           INTEGER NOT NULL,
    section_header          TEXT,
    section_text_original   TEXT NOT NULL,
    split_tier_used         VARCHAR(10) NOT NULL,
    extracted_fields        JSONB,
    all_null                BOOLEAN DEFAULT FALSE,
    llm_provider            VARCHAR(50),
    llm_model               VARCHAR(100),
    processing_time_seconds FLOAT,
    error_message           TEXT,
    extracted_at            TIMESTAMP DEFAULT NOW(),
    UNIQUE (document_id, section_index)
);
"""

CREATE_DOCUMENT_EXTRACTIONS = """
CREATE TABLE IF NOT EXISTS document_extractions (
    id                      SERIAL PRIMARY KEY,
    document_id             INTEGER REFERENCES documents(id) ON DELETE CASCADE UNIQUE,
    extracted_fields        JSONB,
    enforcement_authority   VARCHAR(255),
    effective_date          DATE,
    max_fine_amount         NUMERIC(20, 2),
    fine_currency           VARCHAR(10),
    total_sections          INTEGER,
    sections_with_signal    INTEGER,
    split_tier_used         VARCHAR(10),
    detected_language       VARCHAR(10),
    status                  VARCHAR(50) DEFAULT 'pending',
    extracted_at            TIMESTAMP,
    error_message           TEXT,
    created_at              TIMESTAMP DEFAULT NOW()
);
"""

CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_countries_aliases     ON countries USING GIN (aliases);
CREATE INDEX IF NOT EXISTS idx_documents_criteria    ON documents USING GIN (criteria_ids);
CREATE INDEX IF NOT EXISTS idx_section_ext_document  ON section_extractions(document_id);
CREATE INDEX IF NOT EXISTS idx_section_ext_fields    ON section_extractions USING GIN (extracted_fields);
CREATE INDEX IF NOT EXISTS idx_doc_ext_document      ON document_extractions(document_id);
CREATE INDEX IF NOT EXISTS idx_doc_ext_fields        ON document_extractions USING GIN (extracted_fields);
CREATE INDEX IF NOT EXISTS idx_doc_ext_status        ON document_extractions(status);
CREATE INDEX IF NOT EXISTS idx_documents_country     ON documents(country_id);
CREATE INDEX IF NOT EXISTS idx_documents_type        ON documents(document_type);
"""

ALTER_DOCUMENTS_OPACITY = """
ALTER TABLE documents
  ADD COLUMN IF NOT EXISTS information_opacity BOOLEAN DEFAULT FALSE;
"""

# ── D.0c ─────────────────────────────────────────────────────────────────────
# Run separately in ensure_schema() so failure can be caught and surfaced with
# actionable instructions (the image must be pgvector/pgvector:pg16).
CREATE_VECTOR_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector;"

ALTER_SECTION_EXTRACTIONS_EMBEDDING = """
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'section_extractions' AND column_name = 'embedding'
  ) THEN
    ALTER TABLE section_extractions ADD COLUMN embedding vector(768);
  ELSE
    -- Change dimension to 768 if needed (safe: all values are NULL before first run)
    ALTER TABLE section_extractions ALTER COLUMN embedding TYPE vector(768);
  END IF;
END $$;

ALTER TABLE section_extractions
  ADD COLUMN IF NOT EXISTS embedding_model VARCHAR(50);

DROP INDEX IF EXISTS idx_section_embeddings;
CREATE INDEX IF NOT EXISTS idx_section_embeddings
  ON section_extractions
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100)
  WHERE embedding IS NOT NULL;
"""

# ── D.1 ──────────────────────────────────────────────────────────────────────
ALTER_DOCUMENT_EXTRACTIONS_PI_FIELDS = """
ALTER TABLE document_extractions
  ADD COLUMN IF NOT EXISTS constitutional_privacy_right BOOLEAN,
  ADD COLUMN IF NOT EXISTS dpa_exists                   BOOLEAN,
  ADD COLUMN IF NOT EXISTS dpa_independence             VARCHAR(30),
  ADD COLUMN IF NOT EXISTS data_retention_period        VARCHAR(100),
  ADD COLUMN IF NOT EXISTS interception_legal_standard  VARCHAR(100),
  ADD COLUMN IF NOT EXISTS information_opacity_flag     BOOLEAN DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS dimension                    VARCHAR(20);
"""

# ── D.2 ──────────────────────────────────────────────────────────────────────
CREATE_ENFORCEMENT_RECORDS = """
CREATE TABLE IF NOT EXISTS enforcement_records (
    id                       SERIAL PRIMARY KEY,
    country_id               INTEGER REFERENCES countries(id),
    document_id              INTEGER REFERENCES documents(id),
    source_type              VARCHAR(30) NOT NULL,
    source_url               TEXT,
    source_domain            VARCHAR(255),
    source_language          CHAR(5)     DEFAULT 'en',
    enforcing_body           VARCHAR(255),
    subject_entity           VARCHAR(255),
    pi_criterion_number      INTEGER,
    sanction_type            VARCHAR(50),
    sanction_amount          NUMERIC,
    sanction_currency        CHAR(3),
    sanction_date            DATE,
    summary                  TEXT,
    raw_text                 TEXT,
    reliability_score        FLOAT       DEFAULT 0.8,
    information_opacity_flag BOOLEAN     DEFAULT FALSE,
    created_at               TIMESTAMPTZ DEFAULT NOW(),
    updated_at               TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_enforcement_records_country
  ON enforcement_records(country_id);
CREATE INDEX IF NOT EXISTS idx_enforcement_records_criterion
  ON enforcement_records(pi_criterion_number);
CREATE INDEX IF NOT EXISTS idx_enforcement_records_date
  ON enforcement_records(sanction_date);
"""

# ── D.3 ──────────────────────────────────────────────────────────────────────
CREATE_TRUSTED_SOURCES = """
CREATE TABLE IF NOT EXISTS trusted_sources (
    id                  SERIAL PRIMARY KEY,
    country_code        CHAR(2),
    pi_criterion_number INTEGER,
    domain              VARCHAR(255) NOT NULL,
    source_type         VARCHAR(20)  NOT NULL,
    language            CHAR(5)      DEFAULT 'en',
    search_engine       VARCHAR(20)  DEFAULT 'google',
    reliability_score   FLOAT        DEFAULT 1.0,
    requires_search     BOOLEAN      DEFAULT TRUE,
    geo_restriction     VARCHAR(100),
    last_validated      DATE,
    notes               TEXT,
    created_at          TIMESTAMPTZ  DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_trusted_sources_domain_country_criterion
  ON trusted_sources(
    domain,
    COALESCE(country_code, ''),
    COALESCE(pi_criterion_number, 0)
  );
"""

SEED_TRUSTED_SOURCES = """
INSERT INTO trusted_sources (domain, source_type, pi_criterion_number, requires_search, reliability_score) VALUES
  ('constituteproject.org', 'institutional', 1,    false, 1.0),
  ('hudoc.echr.coe.int',    'institutional', 1,    false, 1.0),
  ('gdprhub.eu',            'institutional', 3,    true,  0.9),
  ('edpb.europa.eu',        'institutional', 3,    false, 1.0),
  ('freedomhouse.org',      'ngo',           14,   false, 0.9),
  ('v-dem.net',             'institutional', 14,   false, 0.95),
  ('rsf.org',               'ngo',           14,   false, 0.85),
  ('treaty.un.org',         'institutional', 13,   false, 1.0),
  ('privacyinternational.org', 'ngo',        NULL, true,  0.85),
  ('eff.org',               'ngo',           NULL, true,  0.8),
  ('accessnow.org',         'ngo',           NULL, true,  0.8)
ON CONFLICT DO NOTHING;
"""

# ── F.1 ──────────────────────────────────────────────────────────────────────
CREATE_CRITERION_SCORES = """
CREATE TABLE IF NOT EXISTS criterion_scores (
    id                   SERIAL PRIMARY KEY,
    country_id           INTEGER REFERENCES countries(id),
    criterion_number     INTEGER NOT NULL,
    criterion_name       VARCHAR(100),
    dimension            VARCHAR(20),

    legal_subscore       FLOAT,
    enforcement_subscore FLOAT,
    criterion_score      FLOAT NOT NULL,

    confidence           VARCHAR(10),
    evidence_count       INTEGER,
    information_opacity  BOOLEAN DEFAULT FALSE,

    rationale            TEXT,
    evidence_gaps        TEXT,
    key_sources          JSONB,

    model_used           VARCHAR(100),
    reference_year       INTEGER,

    created_at           TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (country_id, criterion_number, reference_year, model_used)
);

CREATE INDEX IF NOT EXISTS idx_criterion_scores_country
  ON criterion_scores(country_id);
CREATE INDEX IF NOT EXISTS idx_criterion_scores_criterion
  ON criterion_scores(criterion_number);
"""

# ── G.1 ──────────────────────────────────────────────────────────────────────
CREATE_COUNTRY_INDEX_SCORES = """
CREATE TABLE IF NOT EXISTS country_index_scores (
    id                   SERIAL PRIMARY KEY,
    country_id           INTEGER REFERENCES countries(id),
    reference_year       INTEGER NOT NULL,

    legal_score          FLOAT,
    enforcement_score    FLOAT,
    final_score          FLOAT NOT NULL,

    pi_category          VARCHAR(100),
    rank                 INTEGER,

    criteria_count       INTEGER,
    missing_criteria     JSONB,
    opacity_affected     INTEGER,

    model_used           VARCHAR(100),
    confidence_weighting BOOLEAN DEFAULT TRUE,
    missing_strategy     VARCHAR(20),

    created_at           TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (country_id, reference_year, model_used)
);
"""

# ── E.1 ──────────────────────────────────────────────────────────────────────
CREATE_EXTERNAL_INDICATORS = """
CREATE TABLE IF NOT EXISTS external_indicators (
    id                   SERIAL PRIMARY KEY,
    country_id           INTEGER REFERENCES countries(id),
    pi_criterion_number  INTEGER NOT NULL,
    dimension            VARCHAR(20) NOT NULL,
    source_name          VARCHAR(100) NOT NULL,
    source_year          INTEGER,
    indicator_name       VARCHAR(100),
    indicator_value      FLOAT,
    indicator_normalised FLOAT,
    raw_data             JSONB,
    notes                TEXT,
    created_at           TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (country_id, pi_criterion_number, source_name, indicator_name, source_year)
);

CREATE INDEX IF NOT EXISTS idx_external_indicators_country_criterion
  ON external_indicators(country_id, pi_criterion_number);
"""

ALL_STATEMENTS = [
    CREATE_CRITERIA,
    CREATE_COUNTRIES,
    CREATE_DOCUMENTS,
    CREATE_SECTION_EXTRACTIONS,
    CREATE_DOCUMENT_EXTRACTIONS,
    CREATE_INDEXES,
    SEED_CRITERIA,
    ALTER_DOCUMENTS_OPACITY,
    # D.0c — vector extension handled separately in ensure_schema(); these run after
    ALTER_SECTION_EXTRACTIONS_EMBEDDING,
    ALTER_DOCUMENT_EXTRACTIONS_PI_FIELDS,
    CREATE_ENFORCEMENT_RECORDS,
    CREATE_TRUSTED_SOURCES,
    SEED_TRUSTED_SOURCES,
    CREATE_CRITERION_SCORES,
    CREATE_COUNTRY_INDEX_SCORES,
    CREATE_EXTERNAL_INDICATORS,
]
