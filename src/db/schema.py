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

ALL_STATEMENTS = [
    CREATE_CRITERIA,
    CREATE_COUNTRIES,
    CREATE_DOCUMENTS,
    CREATE_SECTION_EXTRACTIONS,
    CREATE_DOCUMENT_EXTRACTIONS,
    CREATE_INDEXES,
    SEED_CRITERIA,
    ALTER_DOCUMENTS_OPACITY,
]
