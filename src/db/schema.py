"""DDL statements for Phase 3 PostgreSQL schema."""

CREATE_COUNTRIES = """
CREATE TABLE IF NOT EXISTS countries (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(255) NOT NULL,
    iso_code    CHAR(2)      NOT NULL UNIQUE,
    region      VARCHAR(100),
    languages   TEXT[],
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
CREATE INDEX IF NOT EXISTS idx_section_ext_document  ON section_extractions(document_id);
CREATE INDEX IF NOT EXISTS idx_section_ext_fields    ON section_extractions USING GIN (extracted_fields);
CREATE INDEX IF NOT EXISTS idx_doc_ext_document      ON document_extractions(document_id);
CREATE INDEX IF NOT EXISTS idx_doc_ext_fields        ON document_extractions USING GIN (extracted_fields);
CREATE INDEX IF NOT EXISTS idx_doc_ext_status        ON document_extractions(status);
CREATE INDEX IF NOT EXISTS idx_documents_country     ON documents(country_id);
CREATE INDEX IF NOT EXISTS idx_documents_type        ON documents(document_type);
"""

ALL_STATEMENTS = [
    CREATE_COUNTRIES,
    CREATE_DOCUMENTS,
    CREATE_SECTION_EXTRACTIONS,
    CREATE_DOCUMENT_EXTRACTIONS,
    CREATE_INDEXES,
]
