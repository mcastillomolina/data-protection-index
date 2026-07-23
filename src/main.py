"""
Main entry point for the Data Protection Index pipeline.

Orchestrates Phase 1 (Document Discovery), Phase 2 (Document Retrieval),
and Phase 3 (Information Extraction):

Phase 1:
1. Identify relevant documents for a country
2. Generate search queries for each document
3. Execute web searches
4. Filter results by relevance → top URLs per document

Phase 2:
5. Download content from each discovered URL
6. Extract clean text from PDFs and HTML pages

Phase 3:
7. Detect language of each document
8. Split into sections (three-tier regex)
9. Extract structured fields via LLM (one call per section)
10. Aggregate across sections and write to PostgreSQL + JSON
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from src.core import DocumentIdentifier, QueryGenerator, SearchExecutor, RelevanceFilter
from src.core import DocumentRetriever, TextExtractor
from src.core.country_resolver import resolve_country
from src.core.language_detector import LanguageDetector
from src.core.section_splitter import SectionSplitter
from src.core.information_extractor import InformationExtractor
from src.core.criterion_extractor import CriterionExtractor, _load_document_dimension
from src.db.writer import DatabaseWriter
from src.models.country import Country
from src.models.document import DocumentWithResults, DiscoveryOutput
from src.models.retrieval import DocumentContent, RetrievedDocument, RetrievalOutput
from src.models.extraction import (
    SectionExtractionResult,
    DocumentExtractionResult,
    ExtractionOutput,
)
from src.utils.config import Config
from src.utils.logger import setup_logger

DEMO_CRITERIA_NUMBERS = [1, 2, 3]
DEMO_CRITERIA_NAMES = {
    1: "Constitutional Protection",
    2: "Statutory Protection",
    3: "Privacy Enforcement",
}


def discover_documents_for_country(
    country_name: str,
    config: Config,
    db_writer: Optional[DatabaseWriter] = None,
    output_dir: Optional[Path] = None,
    max_documents: Optional[int] = None,
    queries_per_document: int = 5,
    top_urls_per_document: int = 5,
    verbose: bool = False,
    demo_mode: bool = False,
) -> DiscoveryOutput:
    """
    Main pipeline function for discovering documents for a country.

    This orchestrates all components:
    - DocumentIdentifier: Find relevant documents
    - QueryGenerator: Create search queries
    - SearchExecutor: Execute searches
    - RelevanceFilter: Score and filter results

    Args:
        country_name: Name of the country to analyze
        config: Configuration object
        output_dir: Optional output directory (uses config default if None)
        max_documents: Optional limit on number of documents to process
        queries_per_document: Number of search queries per document
        top_urls_per_document: Number of top URLs to return per document
        verbose: Whether to enable verbose logging

    Returns:
        DiscoveryOutput with all discovered documents and URLs

    Raises:
        ValueError: If country not found in config
        Exception: If pipeline fails
    """
    start_time = datetime.now()

    logger.info("="*60)
    logger.info(f"Starting document discovery for: {country_name}")
    logger.info("="*60)

    # Resolve country from config, or enrich via LLM and cache if not found
    country = resolve_country(country_name, config, db_writer=db_writer)
    country_metadata = country.metadata

    logger.info(f"Country: {country.name} ({country.iso_code})")
    logger.info(f"Languages: {', '.join(country.official_languages)}")
    logger.info(f"Region: {country.region}")

    # Initialize components
    llm_client = config.get_llm_client()
    search_client = config.get_search_client()

    # Step 1: Identify documents
    logger.info("\nStep 1/4: Identifying relevant documents...")
    _cache_dir = Path(config.pipeline.cache_dir) if config.pipeline.enable_caching else None
    identifier = DocumentIdentifier(
        llm_client=llm_client,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        cache_dir=_cache_dir,
        demo_mode=demo_mode,
    )

    known_docs = country_metadata.get("known_documents", {})
    documents = identifier.identify_documents(
        country=country,
        known_documents=known_docs if known_docs else None
    )

    logger.info(f"✓ Identified {len(documents)} documents")

    if not documents:
        logger.warning("No documents identified, stopping pipeline")
        return _create_empty_output(country, start_time)

    # Limit documents if specified
    if max_documents and len(documents) > max_documents:
        logger.info(f"Limiting to top {max_documents} documents by priority")
        documents.sort(key=lambda d: d.priority_score, reverse=True)
        documents = documents[:max_documents]

    # Step 2: Generate search queries
    logger.info(f"\nStep 2/4: Generating search queries ({queries_per_document} per document)...")
    generator = QueryGenerator(
        llm_client=llm_client,
        temperature=config.llm.temperature,
        queries_per_document=queries_per_document,
        cache_dir=Path(config.pipeline.cache_dir) if config.pipeline.enable_caching else None,
    )

    known_sources = country_metadata.get("search_hints", [])
    all_queries = generator.generate_queries_for_multiple(
        documents=documents,
        country=country,
        known_sources=known_sources if known_sources else None
    )

    total_queries = sum(len(queries) for queries in all_queries.values())
    logger.info(f"✓ Generated {total_queries} queries for {len(documents)} documents")

    # Step 3: Execute searches
    logger.info(f"\nStep 3/4: Executing web searches...")
    executor = SearchExecutor(
        search_client=search_client,
        max_results_per_query=2 if demo_mode else config.search.max_results_per_query,
        enable_deduplication=config.pipeline.enable_deduplication,
        show_progress=verbose
    )

    # Get country code for localized search
    country_code = country.iso_code.lower() if country.iso_code else None
    language = country.official_languages[0] if country.official_languages else None

    search_results = executor.execute_searches_by_document(
        queries_by_doc=all_queries,
        country_code=country_code,
        language=language
    )

    total_results = sum(len(results) for results in search_results.values())
    logger.info(f"✓ Collected {total_results} search results")

    # Step 4: Filter by relevance (skipped in demo mode)
    from src.models.document import ScoredResult as _ScoredResult
    document_results = []

    if demo_mode:
        total_raw = sum(len(r) for r in search_results.values())
        logger.info(f"Demo mode: relevance filter skipped ({total_raw} results → retrieval)")
        for document in documents:
            doc_id = document.official_name
            results = search_results.get(doc_id, [])
            scored = [
                _ScoredResult(
                    search_result=r,
                    relevance_score=8.0,
                    reasoning="Demo mode — filter skipped",
                    is_likely_official=False,
                    confidence="medium",
                )
                for r in results[:top_urls_per_document]
            ]
            document_results.append(DocumentWithResults(
                document=document,
                top_results=scored,
                search_queries_used=all_queries.get(doc_id, [])
            ))
    else:
        logger.info(f"\nStep 4/4: Filtering results by relevance (top {top_urls_per_document} per document)...")
        relevance_filter = RelevanceFilter(
            llm_client=llm_client,
            temperature=0.2,
            max_tokens=config.llm.max_tokens,
            min_relevance_score=config.pipeline.min_relevance_score,
            cache_dir=_cache_dir,
        )

        for document in documents:
            doc_id = document.official_name
            results = search_results.get(doc_id, [])

            if not results:
                logger.warning(f"No search results for '{doc_id}'")
                document_results.append(DocumentWithResults(
                    document=document,
                    top_results=[],
                    search_queries_used=all_queries.get(doc_id, [])
                ))
                continue

            logger.info(f"Scoring {len(results)} results for '{doc_id}'")

            scored_results = relevance_filter.filter_results_batch(
                document=document,
                results=results,
                country_name=country.name,
                batch_size=10,
                top_n=top_urls_per_document
            )

            document_results.append(DocumentWithResults(
                document=document,
                top_results=scored_results,
                search_queries_used=all_queries.get(doc_id, [])
            ))

            logger.info(f"✓ Found {len(scored_results)} relevant results for '{doc_id}'")

    # Create output
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()

    output = DiscoveryOutput(
        country=country,
        documents=document_results,
        timestamp=end_time,
        total_documents_identified=len(documents),
        total_urls_found=sum(len(d.top_results) for d in document_results),
        metadata={
            "phase": "1",
            "version": "1.0",
            "llm_provider": config.llm.provider,
            "llm_model": config.llm.model,
            "search_provider": config.search.provider,
            "processing_time_seconds": processing_time,
            "queries_per_document": queries_per_document,
            "top_urls_per_document": top_urls_per_document,
            "min_relevance_score": config.pipeline.min_relevance_score
        }
    )

    logger.info("\n" + "="*60)
    logger.info("Discovery complete!")
    logger.info(f"Documents identified: {output.total_documents_identified}")
    logger.info(f"URLs found: {output.total_urls_found}")
    logger.info(f"Processing time: {processing_time:.1f}s")
    logger.info("="*60)

    return output


def save_discovery_output(output: DiscoveryOutput, output_dir: Path) -> Path:
    """
    Save discovery output to JSON file.

    Args:
        output: DiscoveryOutput object
        output_dir: Directory to save output

    Returns:
        Path to saved file
    """
    # Create country-specific directory
    country_dir = output_dir / output.country.name.replace(" ", "_")
    country_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = output.timestamp.strftime("%Y%m%d_%H%M%S")
    filename = f"discovery_results_{timestamp}.json"
    output_file = country_dir / filename

    # Convert to dict for JSON serialization
    output_dict = output.model_dump(mode='json')

    # Save with pretty printing
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_dict, f, indent=2, ensure_ascii=False)

    logger.info(f"Output saved to: {output_file}")

    # Also save a "latest" version
    latest_file = country_dir / "discovery_results_latest.json"
    with open(latest_file, 'w', encoding='utf-8') as f:
        json.dump(output_dict, f, indent=2, ensure_ascii=False)

    logger.info(f"Latest output: {latest_file}")

    return output_file


def print_summary(output: DiscoveryOutput) -> None:
    """
    Print a human-readable summary of the discovery results.

    Args:
        output: DiscoveryOutput object
    """
    print("\n" + "="*70)
    print(f"DISCOVERY SUMMARY: {output.country.name}")
    print("="*70)

    print(f"\n📊 Statistics:")
    print(f"   Documents identified: {output.total_documents_identified}")
    print(f"   Total URLs found: {output.total_urls_found}")
    print(f"   Processing time: {output.metadata.get('processing_time_seconds', 0):.1f}s")

    print(f"\n📄 Top Documents:")
    for i, doc_result in enumerate(output.documents[:5], 1):
        doc = doc_result.document
        print(f"\n   {i}. {doc.official_name}")
        print(f"      Type: {doc.document_type}")
        print(f"      Priority: {doc.priority_score}/10")
        print(f"      URLs found: {len(doc_result.top_results)}")

        if doc_result.top_results:
            best = doc_result.top_results[0]
            print(f"      Best match (score {best.relevance_score:.1f}/10):")
            print(f"        {best.search_result.url}")
            print(f"        Official: {best.is_likely_official}")

    print("\n" + "="*70 + "\n")


def retrieve_documents_from_output(
    discovery_output: DiscoveryOutput,
    config: Config,
    verbose: bool = False,
    output_dir: Optional[Path] = None,
) -> RetrievalOutput:
    """
    Phase 2: Download and extract text from the URLs discovered in Phase 1.

    For each document, tries all top_results URLs in relevance order and uses
    the first one that yields extractable text.

    Args:
        discovery_output: Phase 1 output with scored URLs per document
        config: Configuration object
        verbose: Whether verbose logging is enabled
        output_dir: If provided, load previously extracted content from
            retrieval_results_latest.json to skip re-downloading cached URLs.

    Returns:
        RetrievalOutput with extracted text per document
    """
    start_time = datetime.now()

    logger.info("="*60)
    logger.info("Phase 2: Document Retrieval & Text Extraction")
    logger.info("="*60)

    # Build URL → DocumentContent cache from previous retrieval run
    url_cache: dict[str, DocumentContent] = {}
    if output_dir is not None:
        country_dir = output_dir / discovery_output.country.name.replace(" ", "_")
        cached_retrieval = country_dir / "retrieval_results_latest.json"
        if cached_retrieval.exists():
            try:
                with open(cached_retrieval, encoding="utf-8") as f:
                    prev = json.load(f)
                for rd in prev.get("documents", []):
                    c = rd.get("content")
                    if c and c.get("extraction_success") and c.get("extracted_text"):
                        url_cache[c["url"]] = DocumentContent(**c)
                logger.info(f"Retrieval cache loaded: {len(url_cache)} previously extracted URL(s)")
            except Exception as e:
                logger.warning(f"Could not load retrieval cache from {cached_retrieval}: {e}")

    retriever = DocumentRetriever(
        timeout=config.retrieval.timeout,
        max_retries=config.retrieval.max_retries,
        retry_delay=config.retrieval.retry_delay,
        user_agent=config.retrieval.user_agent,
    )
    extractor = TextExtractor(min_text_length=config.retrieval.min_text_length)

    retrieved_docs: list[RetrievedDocument] = []

    docs_iter = discovery_output.documents
    if verbose:
        from tqdm import tqdm
        docs_iter = tqdm(docs_iter, desc="Retrieving documents")

    for doc_result in docs_iter:
        doc = doc_result.document
        attempted_urls = [r.search_result.url for r in doc_result.top_results]

        if not attempted_urls:
            logger.warning(f"No URLs for '{doc.official_name}' — skipping")
            retrieved_docs.append(RetrievedDocument(
                document=doc,
                attempted_urls=[],
                status="no_results",
            ))
            continue

        logger.info(f"Retrieving '{doc.official_name}' ({len(attempted_urls)} URL(s))")

        content: Optional[DocumentContent] = None
        successful_url: Optional[str] = None

        for url in attempted_urls:
            # Check cache before attempting HTTP download
            if url in url_cache:
                content = url_cache[url]
                successful_url = url
                logger.info(f"[CACHE HIT] '{doc.official_name}' — {content.char_count:,} chars from {url}")
                break

            result = retriever.retrieve(url)
            if result is None:
                continue

            raw_bytes, content_type = result
            text = extractor.extract(raw_bytes, content_type)

            if text is not None:
                content = DocumentContent(
                    url=url,
                    content_type=content_type,
                    extracted_text=text,
                    char_count=len(text),
                    extraction_success=True,
                )
                successful_url = url
                logger.info(f"✓ '{doc.official_name}' — {len(text):,} chars from {url}")
                break
            else:
                logger.debug(f"Extraction yielded no usable text for {url}")

        if content is None:
            logger.warning(f"✗ All URLs failed for '{doc.official_name}'")

        retrieved_docs.append(RetrievedDocument(
            document=doc,
            content=content,
            successful_url=successful_url,
            attempted_urls=attempted_urls,
            status="success" if content else "failed",
        ))

    retriever.close()

    successful = sum(1 for d in retrieved_docs if d.status == "success")
    failed = sum(1 for d in retrieved_docs if d.status == "failed")
    no_results = sum(1 for d in retrieved_docs if d.status == "no_results")

    processing_time = (datetime.now() - start_time).total_seconds()

    logger.info("\n" + "="*60)
    logger.info("Retrieval complete!")
    logger.info(f"Successful: {successful} | Failed: {failed} | No URLs: {no_results}")
    logger.info(f"Processing time: {processing_time:.1f}s")
    logger.info("="*60)

    return RetrievalOutput(
        country=discovery_output.country,
        documents=retrieved_docs,
        total_documents=len(retrieved_docs),
        successful_retrievals=successful,
        failed_retrievals=failed,
        metadata={
            "phase": "2",
            "version": "1.0",
            "processing_time_seconds": processing_time,
            "min_text_length": config.retrieval.min_text_length,
        },
    )


def save_retrieval_output(output: RetrievalOutput, output_dir: Path) -> Path:
    """
    Save retrieval output to JSON file.

    Args:
        output: RetrievalOutput object
        output_dir: Directory to save output

    Returns:
        Path to saved file
    """
    country_dir = output_dir / output.country.name.replace(" ", "_")
    country_dir.mkdir(parents=True, exist_ok=True)

    timestamp = output.timestamp.strftime("%Y%m%d_%H%M%S")
    filename = f"retrieval_results_{timestamp}.json"
    output_file = country_dir / filename

    output_dict = output.model_dump(mode="json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=2, ensure_ascii=False)

    logger.info(f"Retrieval output saved to: {output_file}")

    latest_file = country_dir / "retrieval_results_latest.json"
    with open(latest_file, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=2, ensure_ascii=False)

    logger.info(f"Latest retrieval output: {latest_file}")

    return output_file


def print_retrieval_summary(output: RetrievalOutput) -> None:
    """Print a human-readable summary of retrieval results."""
    print("\n" + "="*70)
    print(f"RETRIEVAL SUMMARY: {output.country.name}")
    print("="*70)

    print(f"\n📊 Statistics:")
    print(f"   Documents processed: {output.total_documents}")
    print(f"   Successfully retrieved: {output.successful_retrievals}")
    print(f"   Failed: {output.failed_retrievals}")
    print(f"   Processing time: {output.metadata.get('processing_time_seconds', 0):.1f}s")

    print(f"\n📄 Documents:")
    for doc_result in output.documents:
        doc = doc_result.document
        status_icon = "✅" if doc_result.status == "success" else ("❌" if doc_result.status == "failed" else "⚠️")
        print(f"\n   {status_icon} {doc.official_name}")
        if doc_result.content:
            print(f"      Source: {doc_result.content.content_type.upper()} — {doc_result.content.char_count:,} chars")
            print(f"      URL: {doc_result.successful_url}")
        elif doc_result.status == "failed":
            print(f"      Tried {len(doc_result.attempted_urls)} URL(s) — all failed")

    print("\n" + "="*70 + "\n")


def extract_information_from_retrieval(
    retrieval_output: RetrievalOutput,
    config: Config,
    db_writer: Optional[DatabaseWriter],
    verbose: bool = False,
    cached_extractions: Optional[dict] = None,
    demo_mode: bool = False,
    disable_gate2: bool = False,
) -> ExtractionOutput:
    """
    Phase 3: Extract structured information from retrieved documents.

    For each successfully retrieved document:
      1. Detect language (langdetect, no LLM)
      2. Split into sections (three-tier regex, no LLM)
      3. Extract fields via LLM (one call per section)
      4. Aggregate across sections
      5. Upsert to PostgreSQL (if db_writer provided) and return ExtractionOutput

    Args:
        retrieval_output: Phase 2 output with extracted text per document
        config: Configuration object
        db_writer: Optional DatabaseWriter; if None, skips DB writes
        verbose: Whether verbose logging is enabled

    Returns:
        ExtractionOutput with extraction results per document
    """
    start_time = datetime.now()

    logger.info("=" * 60)
    logger.info("Phase 3: Information Extraction")
    logger.info("=" * 60)

    country = retrieval_output.country

    lang_detector = LanguageDetector()
    section_splitter = SectionSplitter()
    llm_client = config.get_extraction_llm_client()
    extractor = CriterionExtractor(
        llm_client=llm_client,
        min_section_chars=config.extraction.min_section_chars,
        country_name=country.name,
        disable_gate2=disable_gate2,
    )
    country_id: Optional[int] = None
    if db_writer:
        country_id = db_writer.upsert_country(country)

    doc_results: list[DocumentExtractionResult] = []
    successful = 0
    failed = 0

    docs_iter = retrieval_output.documents
    if verbose:
        from tqdm import tqdm
        docs_iter = tqdm(docs_iter, desc="Extracting documents")

    for retrieved_doc in docs_iter:
        doc = retrieved_doc.document

        if cached_extractions and doc.official_name in cached_extractions:
            logger.info(f"[CACHE HIT] Extraction '{doc.official_name}' — skipping LLM calls")
            doc_results.append(cached_extractions[doc.official_name])
            successful += 1
            continue

        doc_start = datetime.now()
        dimension = _load_document_dimension(doc.document_type)
        logger.info(f"CriterionExtractor — {doc.document_type} — dimension: {dimension}")

        if retrieved_doc.status != "success" or retrieved_doc.content is None:
            logger.warning(f"Skipping '{doc.official_name}' — no retrieved text")
            doc_results.append(
                DocumentExtractionResult(
                    document=doc,
                    detected_language="unknown",
                    split_tier_used="none",
                    total_sections=0,
                    sections_with_signal=0,
                    sections=[],
                    aggregated_fields={},
                    status="failed",
                    error_message="No retrieved text from Phase 2",
                    processing_time_seconds=0.0,
                    llm_provider=config.extraction.llm_provider,
                    llm_model=config.extraction.llm_model,
                )
            )
            failed += 1
            continue

        text = retrieved_doc.content.extracted_text
        logger.info(f"Processing '{doc.official_name}' ({len(text):,} chars)")

        # Step 1: Detect language
        detected_lang = lang_detector.detect_with_fallback(text, country.official_languages)
        logger.info(f"  Language: {detected_lang}")

        # Step 2: Split into sections
        sections = section_splitter.split(text, detected_lang)
        tier_used = sections[0].tier_used if sections else "tier3"
        logger.info(f"  Sections: {len(sections)} (tier={tier_used})")

        if demo_mode and len(sections) > 20:
            logger.info(f"Demo mode: capped at 20/{len(sections)} sections for '{doc.official_name}'")
            sections = sections[:20]

        # Step 3 + 4: Extract and aggregate
        _tok_before_p = llm_client.total_usage.prompt_tokens
        _tok_before_c = llm_client.total_usage.completion_tokens
        try:
            section_results, aggregated = extractor.extract_document(retrieved_doc, sections)
        except Exception as exc:
            logger.error(f"Extraction failed for '{doc.official_name}': {exc}")
            doc_results.append(
                DocumentExtractionResult(
                    document=doc,
                    detected_language=detected_lang,
                    split_tier_used=tier_used,
                    total_sections=len(sections),
                    sections_with_signal=0,
                    sections=[],
                    aggregated_fields={},
                    status="failed",
                    error_message=str(exc),
                    processing_time_seconds=(datetime.now() - doc_start).total_seconds(),
                    llm_provider=config.extraction.llm_provider,
                    llm_model=config.extraction.llm_model,
                )
            )
            failed += 1
            continue

        sections_with_signal = sum(1 for r in section_results if not r.all_null)
        status = (
            "success" if sections_with_signal > 0
            else ("partial" if section_results else "failed")
        )

        elapsed = (datetime.now() - doc_start).total_seconds()

        # Step 5: Upsert to DB
        if db_writer and country_id is not None:
            doc_id = db_writer.upsert_document(
                country_id,
                retrieved_doc,
                detected_lang,
                information_opacity=retrieved_doc.document.information_opacity,
            )
            for sr in section_results:
                db_writer.upsert_section_extraction(
                    doc_id, sr,
                    llm_provider=config.extraction.llm_provider,
                    llm_model=config.extraction.llm_model,
                    extraction_dimension=dimension,
                )
            db_writer.upsert_document_extraction(
                doc_id,
                aggregated,
                metadata={
                    "total_sections": len(sections),
                    "sections_with_signal": sections_with_signal,
                    "split_tier_used": tier_used,
                    "detected_language": detected_lang,
                    "status": status,
                    "extraction_dimension": dimension,
                },
            )

        doc_results.append(
            DocumentExtractionResult(
                document=doc,
                detected_language=detected_lang,
                split_tier_used=tier_used,
                total_sections=len(sections),
                sections_with_signal=sections_with_signal,
                sections=section_results,
                aggregated_fields=aggregated,
                enforcement_authority=aggregated.get("enforcement_body"),
                status=status,
                processing_time_seconds=elapsed,
                llm_provider=config.extraction.llm_provider,
                llm_model=config.extraction.llm_model,
            )
        )

        _doc_prompt = llm_client.total_usage.prompt_tokens - _tok_before_p
        _doc_completion = llm_client.total_usage.completion_tokens - _tok_before_c
        logger.info(
            f"  Done '{doc.official_name}': {sections_with_signal}/{len(sections)} "
            f"sections with signal ({elapsed:.1f}s) | "
            f"tokens in={_doc_prompt:,} out={_doc_completion:,} total={_doc_prompt + _doc_completion:,}"
        )
        successful += 1 if status != "failed" else 0
        failed += 1 if status == "failed" else 0

    processing_time = (datetime.now() - start_time).total_seconds()

    _total = llm_client.total_usage
    _avg = _total.total_tokens // max(sum(d.total_sections for d in doc_results), 1)
    logger.info(
        f"\n{'═' * 42}\n"
        f"TOKEN USAGE SUMMARY — Phase 3 Extraction\n"
        f"  Input tokens:  {_total.prompt_tokens:>12,}\n"
        f"  Output tokens: {_total.completion_tokens:>12,}\n"
        f"  Total tokens:  {_total.total_tokens:>12,}\n"
        f"  Avg per section: {_avg:>10,}\n"
        f"{'═' * 42}"
    )

    logger.info("\n" + "=" * 60)
    logger.info("Extraction complete!")
    logger.info(f"Successful: {successful} | Failed: {failed}")
    logger.info(f"Processing time: {processing_time:.1f}s")
    logger.info("=" * 60)

    return ExtractionOutput(
        country=country,
        documents=doc_results,
        total_documents=len(doc_results),
        successful_extractions=successful,
        failed_extractions=failed,
        metadata={
            "phase": "3",
            "version": "1.0",
            "llm_provider": config.extraction.llm_provider,
            "llm_model": config.extraction.llm_model,
            "processing_time_seconds": processing_time,
        },
    )


def save_extraction_output(output: ExtractionOutput, output_dir: Path) -> Path:
    """Save extraction output to JSON file."""
    country_dir = output_dir / output.country.name.replace(" ", "_")
    country_dir.mkdir(parents=True, exist_ok=True)

    timestamp = output.timestamp.strftime("%Y%m%d_%H%M%S")
    filename = f"extraction_results_{timestamp}.json"
    output_file = country_dir / filename

    output_dict = output.model_dump(mode="json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=2, ensure_ascii=False)

    logger.info(f"Extraction output saved to: {output_file}")

    latest_file = country_dir / "extraction_results_latest.json"
    with open(latest_file, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=2, ensure_ascii=False)

    logger.info(f"Latest extraction output: {latest_file}")
    return output_file


def print_extraction_summary(output: ExtractionOutput) -> None:
    """Print a human-readable summary of extraction results."""
    print("\n" + "=" * 70)
    print(f"EXTRACTION SUMMARY: {output.country.name}")
    print("=" * 70)

    print(f"\n📊 Statistics:")
    print(f"   Documents processed: {output.total_documents}")
    print(f"   Successful: {output.successful_extractions}")
    print(f"   Failed: {output.failed_extractions}")
    print(f"   Processing time: {output.metadata.get('processing_time_seconds', 0):.1f}s")

    print(f"\n📄 Documents:")
    for doc_result in output.documents:
        doc = doc_result.document
        status_icon = "✅" if doc_result.status == "success" else (
            "⚠️" if doc_result.status == "partial" else "❌"
        )
        print(f"\n   {status_icon} {doc.official_name}")
        print(
            f"      Lang: {doc_result.detected_language} | "
            f"Tier: {doc_result.split_tier_used} | "
            f"Sections: {doc_result.sections_with_signal}/{doc_result.total_sections} with signal"
        )
        if doc_result.enforcement_authority:
            print(f"      Enforcement: {doc_result.enforcement_authority}")
        agg = doc_result.aggregated_fields
        if agg.get("key_provisions"):
            print(f"      Key provisions: {len(agg['key_provisions'])}")
        if agg.get("data_subject_rights"):
            print(f"      Subject rights: {len(agg['data_subject_rights'])}")

    print("\n" + "=" * 70 + "\n")


def _get_country_info_for_scoring(
    db_writer: "DatabaseWriter",
    dsn: str,
    country_name: str,
) -> tuple[int, str, str, str]:
    """
    Fetch country_id, canonical name, iso_code, and information_environment
    from the DB for use by CriterionScorer.

    Exits with code 1 if the country is not found.
    Returns (country_id, name, iso_code, information_environment).
    information_environment defaults to 'open' if the column does not yet exist.
    """
    import psycopg2
    import psycopg2.extras

    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'countries'
                  AND column_name = 'information_environment'
                """
            )
            has_ie_col = cur.fetchone() is not None

        select_cols = (
            "id, name, iso_code, information_environment"
            if has_ie_col
            else "id, name, iso_code"
        )
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT {select_cols}
                FROM countries
                WHERE name ILIKE %s OR %s = ANY(aliases)
                LIMIT 1
                """,
                (country_name, country_name),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        logger.error(
            f"Country '{country_name}' not found in the database. "
            "Run the full pipeline first to populate it."
        )
        sys.exit(1)

    info_env = row.get("information_environment", "open") or "open"
    return int(row["id"]), row["name"], row["iso_code"].strip(), info_env


def _print_scoring_summary(
    country_name: str,
    reference_year: int,
    scores: list,
) -> None:
    """Print one line per scored criterion to stdout."""
    print("\n" + "=" * 70)
    print(f"SCORING SUMMARY: {country_name} ({reference_year})")
    print("=" * 70)
    print(f"{'#':<4} {'Criterion':<45} {'Score':<7} {'Conf':<8} {'Opacity'}")
    print("-" * 70)
    for s in sorted(scores, key=lambda x: x.criterion_number):
        opacity_flag = "[opacity]" if s.information_opacity else ""
        print(
            f"{s.criterion_number:<4} {s.criterion_name:<45} "
            f"{s.criterion_score:<7.2f} {s.confidence:<8} {opacity_flag}"
        )
    print("=" * 70)
    print(f"Total scored: {len(scores)}/14 criteria")
    print()


def _create_empty_output(country: Country, start_time: datetime) -> DiscoveryOutput:
    """Create an empty DiscoveryOutput for failed pipelines."""
    return DiscoveryOutput(
        country=country,
        documents=[],
        timestamp=datetime.now(),
        total_documents_identified=0,
        total_urls_found=0,
        metadata={
            "phase": "1",
            "version": "1.0",
            "processing_time_seconds": (datetime.now() - start_time).total_seconds(),
            "error": "No documents identified"
        }
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Phase 1: Document Discovery for Data Protection Index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Discover documents for Chile
  python -m src.main Chile

  # Verbose output
  python -m src.main Chile --verbose

  # Limit to 5 documents, 3 queries each
  python -m src.main Germany --max-documents 5 --queries-per-doc 3

  # Custom output directory
  python -m src.main "United Kingdom" --output-dir ./results
        """
    )

    parser.add_argument(
        "country",
        nargs="?",
        type=str,
        default=None,
        help=(
            "Country name (e.g., 'Chile', 'Germany', 'United Kingdom'). "
            "Not required for --score-all or --export-index."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to configuration file (default: config/config.yaml)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for results (default: from config)"
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        help="Maximum number of documents to process (default: unlimited)"
    )
    parser.add_argument(
        "--queries-per-doc",
        type=int,
        default=5,
        help="Number of search queries per document (default: 5)"
    )
    parser.add_argument(
        "--top-urls",
        type=int,
        default=5,
        help="Number of top URLs to return per document (default: 5)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging and progress bars"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save output to file (print only)"
    )
    parser.add_argument(
        "--discovery-only",
        action="store_true",
        help="Run Phase 1 only (skip document retrieval)"
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Run Phases 1 + 2 but skip Phase 3 information extraction"
    )
    parser.add_argument(
        "--extraction-only",
        action="store_true",
        help="Run Phase 3 only (reads existing retrieval_results_latest.json)"
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Skip PostgreSQL writes during Phase 3 (JSON output only)"
    )
    parser.add_argument(
        "--populate-embeddings",
        action="store_true",
        help="After Phase 3, embed all non-null sections with the configured embedding model"
    )
    parser.add_argument(
        "--embeddings-only",
        action="store_true",
        help="Skip Phases 1–3; embed pending sections for a country already in the DB"
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help=(
            "Run Phase 4 scoring only (Phases 1–3 + embeddings must already be complete). "
            "Scores all 14 PI criteria for the country and writes to criterion_scores table."
        ),
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Reference year for scoring (default: current year)",
    )
    parser.add_argument(
        "--fetch-external",
        action="store_true",
        help=(
            "Fetch external source data (Freedom House, V-Dem, RSF, GDPRhub) "
            "for the given country into external_indicators table. Requires country arg."
        ),
    )
    parser.add_argument(
        "--score-all",
        action="store_true",
        help=(
            "Score all countries in the DB that have embedded sections. "
            "Writes to criterion_scores. No country arg required."
        ),
    )
    parser.add_argument(
        "--benchmark-models",
        action="store_true",
        help=(
            "Run CriterionScorer with each model in scoring.benchmark_models from config. "
            "Writes one row per model to criterion_scores. Requires country arg."
        ),
    )
    parser.add_argument(
        "--export-index",
        action="store_true",
        help=(
            "Aggregate criterion scores, rank countries, and export index to "
            "data/outputs/index/. No country arg required. Use with --format."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="Export format for --export-index (default: csv).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "Disable all file caches for this run — forces fresh LLM and network calls. "
            "Equivalent to setting pipeline.enable_caching: false in config.yaml."
        ),
    )
    parser.add_argument(
        "--disable-gate2",
        action="store_true",
        help=(
            "Disable the signal-vocabulary pre-filter (Gate 2) during Phase 3 extraction. "
            "Use when running a country whose language is not yet in the vocabulary list — "
            "all sections that pass Gate 1 (structural noise) will be sent to the LLM. "
            "Safe in demo mode due to the 20-section cap."
        ),
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Run the full pipeline end-to-end with reduced scope (3 docs, 3 criteria, "
            "no relevance filter). Results are written to DB for Streamlit dashboard demos."
        ),
    )

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logger(level=log_level)

    logger.info("Data Protection Index - Phase 1: Document Discovery")
    logger.info(f"Country: {args.country}")

    try:
        # Load configuration
        config = Config(args.config)
        config.validate()

        # --no-cache overrides config so every downstream reader sees it immediately
        if args.no_cache:
            config.pipeline.enable_caching = False

        # Convenience variable used throughout main() for all cache-aware constructors
        cache_dir = Path(config.pipeline.cache_dir) if config.pipeline.enable_caching else None

        if args.demo:
            print(
                "\n╔══════════════════════════════════════════════════╗\n"
                "║            DPI PIPELINE — DEMO MODE              ║\n"
                f"║  Country: {args.country:<39}║\n"
                "║  Criteria: 3  |  Max docs: 3  |  Queries: 6     ║\n"
                "║  Estimated time: ~3 minutes                      ║\n"
                "╚══════════════════════════════════════════════════╝\n"
            )
        _demo_start = datetime.now()

        # Determine output directory
        output_dir = args.output_dir if args.output_dir else Path(config.output.directory)

        # Initialise DB writer early so country resolution can use it
        import os
        db_writer: Optional[DatabaseWriter] = None
        dsn = os.getenv("DATABASE_URL")
        if not args.skip_db:
            if dsn:
                db_writer = DatabaseWriter(dsn)
                db_writer.ensure_schema()
            else:
                logger.warning("DATABASE_URL not set — skipping DB writes")

        # ------------------------------------------------------------------
        # Validate: country is required unless a no-country flag is set
        # ------------------------------------------------------------------
        _no_country_flags = args.score_all or args.export_index
        if args.country is None and not _no_country_flags:
            logger.error(
                "A country name is required unless you use --score-all or --export-index."
            )
            sys.exit(1)

        # ------------------------------------------------------------------
        # --fetch-external: ingest institutional data for one country
        # ------------------------------------------------------------------
        if args.fetch_external:
            if args.country is None:
                logger.error("--fetch-external requires a country name argument")
                sys.exit(1)
            if not dsn:
                logger.error("--fetch-external requires DATABASE_URL to be set")
                sys.exit(1)
            if not db_writer:
                db_writer = DatabaseWriter(dsn)
                db_writer.ensure_schema()

            reference_year = args.year if args.year else datetime.now().year
            country_id, country_name_resolved, iso_code, _ = (
                _get_country_info_for_scoring(db_writer, dsn, args.country)
            )

            from src.core.external_source_fetcher import ExternalSourceFetcher
            fetcher = ExternalSourceFetcher(dsn=dsn)
            results = fetcher.fetch_all(
                country_iso=iso_code,
                country_id=country_id,
                year=reference_year,
                country_name=country_name_resolved,
            )
            print(f"\n✅ External sources fetched for '{country_name_resolved}':")
            for source, count in results.items():
                print(f"   {source}: {count} row(s) written")
            db_writer.close()
            sys.exit(0)

        # ------------------------------------------------------------------
        # --score-all: score all countries with embedded sections
        # ------------------------------------------------------------------
        if args.score_all:
            for flag_name, flag_val in [
                ("--discovery-only",  args.discovery_only),
                ("--extraction-only", args.extraction_only),
                ("--embeddings-only", args.embeddings_only),
                ("--score-only",      args.score_only),
                ("--benchmark-models", args.benchmark_models),
            ]:
                if flag_val:
                    logger.error(f"--score-all cannot be combined with {flag_name}")
                    sys.exit(1)
            if not dsn:
                logger.error("--score-all requires DATABASE_URL to be set")
                sys.exit(1)
            if not db_writer:
                db_writer = DatabaseWriter(dsn)
                db_writer.ensure_schema()

            reference_year = args.year if args.year else datetime.now().year

            import psycopg2 as _psycopg2
            import psycopg2.extras as _extras
            _conn = _psycopg2.connect(dsn)
            try:
                with _conn.cursor(cursor_factory=_extras.RealDictCursor) as _cur:
                    _cur.execute(
                        """
                        SELECT DISTINCT d.country_id
                        FROM documents d
                        JOIN section_extractions se ON se.document_id = d.id
                        WHERE se.embedding IS NOT NULL
                        """
                    )
                    _country_ids = [r["country_id"] for r in _cur.fetchall()]
            finally:
                _conn.close()

            if not _country_ids:
                logger.warning("--score-all: no countries with embedded sections found in DB")
                db_writer.close()
                sys.exit(0)

            logger.info(f"--score-all: scoring {len(_country_ids)} country(ies)")
            embedding_client = config.get_embedding_client()
            llm_client = config.get_llm_client()

            from src.core.criterion_scorer import CriterionScorer as _CS
            scorer = _CS(
                llm_client=llm_client,
                embedding_client=embedding_client,
                dsn=dsn,
                model_name=config.llm.model,
                cosine_threshold=config.scoring.cosine_similarity_threshold,
                max_sections=config.scoring.max_sections_per_criterion,
            )

            for _cid in _country_ids:
                _conn2 = _psycopg2.connect(dsn)
                try:
                    with _conn2.cursor(cursor_factory=_extras.RealDictCursor) as _cur2:
                        _cur2.execute(
                            "SELECT id, name, iso_code FROM countries WHERE id = %s",
                            (_cid,),
                        )
                        _crow = _cur2.fetchone()
                finally:
                    _conn2.close()
                if _crow is None:
                    logger.warning(f"--score-all: country_id={_cid} not found — skipping")
                    continue
                _scores = scorer.score_all_criteria(
                    country_id=_crow["id"],
                    country_name=_crow["name"],
                    country_code=_crow["iso_code"].strip(),
                    reference_year=reference_year,
                    skip_if_scored=config.pipeline.enable_caching,
                )
                _print_scoring_summary(_crow["name"], reference_year, _scores)

            db_writer.close()
            sys.exit(0)

        # ------------------------------------------------------------------
        # --benchmark-models: score one country with multiple models
        # ------------------------------------------------------------------
        if args.benchmark_models:
            if args.country is None:
                logger.error("--benchmark-models requires a country name argument")
                sys.exit(1)
            if args.score_all or args.score_only:
                logger.error("--benchmark-models cannot be combined with --score-all or --score-only")
                sys.exit(1)
            if not dsn:
                logger.error("--benchmark-models requires DATABASE_URL to be set")
                sys.exit(1)
            if not config.scoring.benchmark_models:
                logger.error(
                    "scoring.benchmark_models is empty in config.yaml — "
                    "add model strings to use --benchmark-models"
                )
                sys.exit(1)
            if not db_writer:
                db_writer = DatabaseWriter(dsn)
                db_writer.ensure_schema()

            reference_year = args.year if args.year else datetime.now().year
            country_id, country_name_resolved, iso_code, info_env = (
                _get_country_info_for_scoring(db_writer, dsn, args.country)
            )
            embedding_client = config.get_embedding_client()

            from src.core.criterion_scorer import CriterionScorer as _CS2
            for model_str in config.scoring.benchmark_models:
                logger.info(f"--benchmark-models: scoring with model '{model_str}'")
                # model_str format: "provider/model-name"
                if "/" in model_str:
                    _provider, _model = model_str.split("/", 1)
                else:
                    _provider, _model = config.llm.provider, model_str

                # Temporarily override config to build correct client
                _orig_provider = config.llm.provider
                _orig_model    = config.llm.model
                config.llm.provider = _provider
                config.llm.model    = _model
                try:
                    _bm_llm = config.get_llm_client()
                finally:
                    config.llm.provider = _orig_provider
                    config.llm.model    = _orig_model

                _bm_scorer = _CS2(
                    llm_client=_bm_llm,
                    embedding_client=embedding_client,
                    dsn=dsn,
                    model_name=model_str,
                    cosine_threshold=config.scoring.cosine_similarity_threshold,
                    max_sections=config.scoring.max_sections_per_criterion,
                )
                _bm_scores = _bm_scorer.score_all_criteria(
                    country_id=country_id,
                    country_name=country_name_resolved,
                    country_code=iso_code,
                    reference_year=reference_year,
                    information_environment=info_env,
                    skip_if_scored=config.pipeline.enable_caching,
                )
                _print_scoring_summary(f"{args.country} [{model_str}]", reference_year, _bm_scores)

            db_writer.close()
            sys.exit(0)

        # ------------------------------------------------------------------
        # --export-index: aggregate, rank, and export all countries
        # ------------------------------------------------------------------
        if args.export_index:
            if not dsn:
                logger.error("--export-index requires DATABASE_URL to be set")
                sys.exit(1)
            if not db_writer:
                db_writer = DatabaseWriter(dsn)
                db_writer.ensure_schema()

            reference_year = args.year if args.year else datetime.now().year

            from src.core.index_aggregator import IndexAggregator
            aggregator = IndexAggregator(dsn=dsn, scoring_config=config.scoring)
            ranked = aggregator.rank_countries(reference_year=reference_year, model_used=None)
            out_path = aggregator.export_index(
                reference_year=reference_year,
                model_used=None,
                fmt=args.format,
            )
            print(f"\n✅ Index exported ({len(ranked)} countries) → {out_path}")
            db_writer.close()
            sys.exit(0)

        # --score-only mutual-exclusion guard
        if args.score_only:
            for flag_name, flag_val in [
                ("--discovery-only", args.discovery_only),
                ("--extraction-only", args.extraction_only),
                ("--embeddings-only", args.embeddings_only),
            ]:
                if flag_val:
                    logger.error(f"--score-only cannot be combined with {flag_name}")
                    sys.exit(1)
            if not dsn:
                logger.error("--score-only requires DATABASE_URL to be set")
                sys.exit(1)

        # --embeddings-only requires a DB connection
        if args.embeddings_only and not dsn:
            logger.error("--embeddings-only requires DATABASE_URL to be set")
            sys.exit(1)

        # ------------------------------------------------------------------
        # --embeddings-only: skip all phases, embed pending sections from DB
        # ------------------------------------------------------------------
        if args.embeddings_only:
            if not db_writer:
                db_writer = DatabaseWriter(dsn)
                db_writer.ensure_schema()
            country_id = db_writer.get_country_id_by_name(args.country)
            if country_id is None:
                logger.error(
                    f"Country '{args.country}' not found in the database. "
                    "Run the full pipeline first to populate it."
                )
                sys.exit(1)
            embedding_client = config.get_embedding_client()
            from src.core.embedding_populator import EmbeddingPopulator
            populator = EmbeddingPopulator(dsn=dsn, embedding_client=embedding_client)
            n = populator.populate(country_id)
            print(f"\n✅ Embedded {n} sections for '{args.country}' "
                  f"using {embedding_client.model}")
            db_writer.close()
            sys.exit(0)

        # ------------------------------------------------------------------
        # --score-only: skip Phases 1–3; score all 14 criteria from DB
        # ------------------------------------------------------------------
        if args.score_only:
            if not db_writer:
                db_writer = DatabaseWriter(dsn)
                db_writer.ensure_schema()

            reference_year = args.year if args.year else datetime.now().year

            country_id, country_name_resolved, iso_code, info_env = (
                _get_country_info_for_scoring(db_writer, dsn, args.country)
            )

            embedding_client = config.get_embedding_client()
            llm_client = config.get_llm_client()

            from src.core.criterion_scorer import CriterionScorer
            scorer = CriterionScorer(
                llm_client=llm_client,
                embedding_client=embedding_client,
                dsn=dsn,
                model_name=config.llm.model,
                cosine_threshold=config.scoring.cosine_similarity_threshold,
                max_sections=config.scoring.max_sections_per_criterion,
            )

            scores = scorer.score_all_criteria(
                country_id=country_id,
                country_name=country_name_resolved,
                country_code=iso_code,
                reference_year=reference_year,
                information_environment=info_env,
                skip_if_scored=config.pipeline.enable_caching,
            )

            _print_scoring_summary(args.country, reference_year, scores)

            db_writer.close()
            sys.exit(0)

        # ------------------------------------------------------------------
        # --extraction-only: skip Phases 1+2, load existing retrieval output
        # ------------------------------------------------------------------
        if args.extraction_only:
            country_dir = output_dir / args.country.replace(" ", "_")
            retrieval_file = country_dir / "retrieval_results_latest.json"
            if not retrieval_file.exists():
                logger.error(f"No retrieval output found at {retrieval_file}")
                sys.exit(1)

            with open(retrieval_file, encoding="utf-8") as f:
                retrieval_data = json.load(f)

            from src.models.retrieval import RetrievalOutput as _RO
            retrieval_output = _RO.model_validate(retrieval_data)

        else:
            # Phase 1: discovery
            _p1_start = datetime.now()
            output = discover_documents_for_country(
                country_name=args.country,
                config=config,
                db_writer=db_writer,
                output_dir=output_dir,
                max_documents=3 if args.demo else args.max_documents,
                queries_per_document=2 if args.demo else args.queries_per_doc,
                top_urls_per_document=args.top_urls,
                verbose=args.verbose,
                demo_mode=args.demo,
            )

            if not args.no_save:
                output_file = save_discovery_output(output, output_dir)
                print(f"\n✅ Discovery results saved to: {output_file}")

            print_summary(output)

            if args.demo:
                _p1_elapsed = (datetime.now() - _p1_start).total_seconds()
                total_searches = sum(
                    len(d.search_queries_used) for d in output.documents
                )
                print(
                    f"✅ P1 complete ({_p1_elapsed:.0f}s) — "
                    f"{output.total_documents_identified} documents, {total_searches} searches"
                )

            if args.discovery_only:
                sys.exit(0)

            # Phase 2: retrieval
            _p2_start = datetime.now()
            retrieval_output = retrieve_documents_from_output(
                discovery_output=output,
                config=config,
                verbose=args.verbose,
                output_dir=output_dir,
            )

            if not args.no_save:
                retrieval_file = save_retrieval_output(retrieval_output, output_dir)
                print(f"\n✅ Retrieval results saved to: {retrieval_file}")

            print_retrieval_summary(retrieval_output)

            if args.demo:
                _p2_elapsed = (datetime.now() - _p2_start).total_seconds()
                print(
                    f"✅ P2 complete ({_p2_elapsed:.0f}s) — "
                    f"{retrieval_output.successful_retrievals} documents retrieved"
                )

        # Phase 3: extraction
        if not args.skip_extraction and not args.discovery_only:
            cached_extractions: dict = {}
            if cache_dir is not None and args.country:
                _country_dir = output_dir / args.country.replace(" ", "_")
                _prev_extraction = _country_dir / "extraction_results_latest.json"
                if _prev_extraction.exists():
                    try:
                        with open(_prev_extraction, encoding="utf-8") as _f:
                            _prev = json.load(_f)
                        from src.models.extraction import ExtractionOutput as _EO
                        _prev_out = _EO.model_validate(_prev)
                        cached_extractions = {
                            d.document.official_name: d
                            for d in _prev_out.documents
                            if d.status != "failed"
                        }
                        logger.info(
                            f"Extraction cache loaded: {len(cached_extractions)} doc(s)"
                        )
                    except Exception as _e:
                        logger.warning(f"Could not load extraction cache: {_e}")

            _p3_start = datetime.now()
            extraction_output = extract_information_from_retrieval(
                retrieval_output=retrieval_output,
                config=config,
                db_writer=db_writer,
                verbose=args.verbose,
                cached_extractions=cached_extractions,
                demo_mode=args.demo,
                disable_gate2=args.disable_gate2,
            )

            if not args.no_save:
                extraction_file = save_extraction_output(extraction_output, output_dir)
                print(f"\n✅ Extraction results saved to: {extraction_file}")

            print_extraction_summary(extraction_output)

            if args.demo:
                _p3_elapsed = (datetime.now() - _p3_start).total_seconds()
                total_sections = sum(d.total_sections for d in extraction_output.documents)
                print(f"✅ P3 complete ({_p3_elapsed:.0f}s) — {total_sections} sections extracted")

            # Optional: embed non-null sections after Phase 3
            if args.populate_embeddings and db_writer and dsn:
                country_id = db_writer.get_country_id_by_name(args.country)
                if country_id is not None:
                    embedding_client = config.get_embedding_client()
                    from src.core.embedding_populator import EmbeddingPopulator
                    populator = EmbeddingPopulator(
                        dsn=dsn, embedding_client=embedding_client
                    )
                    n = populator.populate(country_id)
                    print(f"\n✅ Embedded {n} sections for '{args.country}' "
                          f"using {embedding_client.model}")
                else:
                    logger.warning(
                        f"Could not find country_id for '{args.country}' — "
                        "skipping embedding population"
                    )

            # --demo: run embeddings → external sources → scoring end-to-end
            if args.demo and db_writer and dsn:
                reference_year = args.year or datetime.now().year

                # Embeddings
                _emb_start = datetime.now()
                _emb_country_id = db_writer.get_country_id_by_name(args.country)
                if _emb_country_id is not None:
                    _emb_client = config.get_embedding_client()
                    from src.core.embedding_populator import EmbeddingPopulator as _EP
                    _populator = _EP(dsn=dsn, embedding_client=_emb_client)
                    _n_embedded = _populator.populate(_emb_country_id)
                    print(
                        f"✅ Embeddings complete ({(datetime.now() - _emb_start).total_seconds():.0f}s)"
                        f" — {_n_embedded} section(s) embedded"
                    )
                else:
                    logger.warning("Demo: could not find country_id for embeddings — skipping")

                # External sources (failures are expected and acceptable)
                _ext_start = datetime.now()
                try:
                    _ext_country_id, _ext_country_name, _ext_iso, _ = (
                        _get_country_info_for_scoring(db_writer, dsn, args.country)
                    )
                    from src.core.external_source_fetcher import ExternalSourceFetcher as _ESF
                    _fetcher = _ESF(dsn=dsn)
                    _ext_results = _fetcher.fetch_all(
                        country_iso=_ext_iso,
                        country_id=_ext_country_id,
                        year=reference_year,
                        country_name=_ext_country_name,
                    )
                    _ext_rows = sum(_ext_results.values())
                except Exception as _ext_err:
                    logger.warning(f"Demo: external sources error (non-fatal): {_ext_err}")
                    _ext_country_id, _ext_country_name, _ext_iso = None, args.country, ""
                    _ext_rows = 0
                print(
                    f"✅ External sources complete ({(datetime.now() - _ext_start).total_seconds():.0f}s)"
                    f" — {_ext_rows} row(s) written"
                )

                # Scoring — DEMO_CRITERIA_NUMBERS only
                _score_start = datetime.now()
                if _ext_country_id is not None:
                    from src.core.criterion_scorer import CriterionScorer as _DemoCS
                    _demo_scorer = _DemoCS(
                        llm_client=config.get_llm_client(),
                        embedding_client=config.get_embedding_client(),
                        dsn=dsn,
                        model_name=config.llm.model,
                        cosine_threshold=config.scoring.cosine_similarity_threshold,
                        max_sections=config.scoring.max_sections_per_criterion,
                    )
                    _demo_scores = _demo_scorer.score_all_criteria(
                        country_id=_ext_country_id,
                        country_name=_ext_country_name,
                        country_code=_ext_iso,
                        reference_year=reference_year,
                        criteria_filter=DEMO_CRITERIA_NUMBERS,
                    )
                    print(
                        f"✅ Scoring complete ({(datetime.now() - _score_start).total_seconds():.0f}s)"
                        f" — 3 criteria scored"
                    )

                    # Final summary
                    _total_elapsed = (datetime.now() - _demo_start).total_seconds()
                    _mins, _secs = divmod(int(_total_elapsed), 60)
                    print("\n" + "=" * 40)
                    print(f"🏁 Demo complete in {_mins}m {_secs}s — {args.country}")
                    print()
                    print(f"{'Criterion':<40} {'Score':>5}")
                    print("─" * 46)
                    _score_map = {s.criterion_number: s for s in _demo_scores}
                    for _num in DEMO_CRITERIA_NUMBERS:
                        _s = _score_map.get(_num)
                        _name = DEMO_CRITERIA_NAMES.get(_num, f"Criterion {_num}")
                        _val = f"{_s.criterion_score:.2f}" if _s else "N/A"
                        print(f"{_name:<40} {_val:>5}")
                    print("=" * 40 + "\n")
                else:
                    logger.warning("Demo: scoring skipped — country not found in DB after P1-P3")

        if db_writer:
            db_writer.close()

        # Exit successfully
        sys.exit(0)

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        sys.exit(130)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
