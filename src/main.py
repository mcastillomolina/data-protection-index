"""
Main entry point for the Data Protection Index pipeline.

Orchestrates Phase 1 (Document Discovery) and Phase 2 (Document Retrieval):

Phase 1:
1. Identify relevant documents for a country
2. Generate search queries for each document
3. Execute web searches
4. Filter results by relevance → top URLs per document

Phase 2:
5. Download content from each discovered URL
6. Extract clean text from PDFs and HTML pages
7. Save structured text corpus for Phase 3
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
from src.models.country import Country
from src.models.document import DocumentWithResults, DiscoveryOutput
from src.models.retrieval import DocumentContent, RetrievedDocument, RetrievalOutput
from src.utils.config import Config
from src.utils.logger import setup_logger


def discover_documents_for_country(
    country_name: str,
    config: Config,
    output_dir: Optional[Path] = None,
    max_documents: Optional[int] = None,
    queries_per_document: int = 5,
    top_urls_per_document: int = 5,
    verbose: bool = False
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
    country = resolve_country(country_name, config)
    country_metadata = country.metadata

    logger.info(f"Country: {country.name} ({country.iso_code})")
    logger.info(f"Languages: {', '.join(country.official_languages)}")
    logger.info(f"Region: {country.region}")

    # Initialize components
    llm_client = config.get_llm_client()
    search_client = config.get_search_client()

    # Step 1: Identify documents
    logger.info("\nStep 1/4: Identifying relevant documents...")
    identifier = DocumentIdentifier(
        llm_client=llm_client,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens
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
        queries_per_document=queries_per_document
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
        max_results_per_query=config.search.max_results_per_query,
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

    # Step 4: Filter by relevance
    logger.info(f"\nStep 4/4: Filtering results by relevance (top {top_urls_per_document} per document)...")
    relevance_filter = RelevanceFilter(
        llm_client=llm_client,
        temperature=0.2,  # Lower for consistent scoring
        min_relevance_score=config.pipeline.min_relevance_score
    )

    document_results = []
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

        scored_results = relevance_filter.filter_results(
            document=document,
            results=results,
            country_name=country.name,
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
) -> RetrievalOutput:
    """
    Phase 2: Download and extract text from the URLs discovered in Phase 1.

    For each document, tries all top_results URLs in relevance order and uses
    the first one that yields extractable text.

    Args:
        discovery_output: Phase 1 output with scored URLs per document
        config: Configuration object
        verbose: Whether verbose logging is enabled

    Returns:
        RetrievalOutput with extracted text per document
    """
    start_time = datetime.now()

    logger.info("="*60)
    logger.info("Phase 2: Document Retrieval & Text Extraction")
    logger.info("="*60)

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
        type=str,
        help="Country name (e.g., 'Chile', 'Germany', 'United Kingdom')"
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

        # Determine output directory
        output_dir = args.output_dir if args.output_dir else Path(config.output.directory)

        # Run pipeline
        output = discover_documents_for_country(
            country_name=args.country,
            config=config,
            output_dir=output_dir,
            max_documents=args.max_documents,
            queries_per_document=args.queries_per_doc,
            top_urls_per_document=args.top_urls,
            verbose=args.verbose
        )

        # Save Phase 1 output (unless disabled)
        if not args.no_save:
            output_file = save_discovery_output(output, output_dir)
            print(f"\n✅ Discovery results saved to: {output_file}")

        # Print Phase 1 summary
        print_summary(output)

        # Phase 2: retrieve and extract text from discovered URLs
        if not args.discovery_only:
            retrieval_output = retrieve_documents_from_output(
                discovery_output=output,
                config=config,
                verbose=args.verbose,
            )

            if not args.no_save:
                retrieval_file = save_retrieval_output(retrieval_output, output_dir)
                print(f"\n✅ Retrieval results saved to: {retrieval_file}")

            print_retrieval_summary(retrieval_output)

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
