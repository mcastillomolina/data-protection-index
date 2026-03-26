"""
Document identifier using LLM.

This module uses an LLM to identify relevant legal documents for a given country.
"""

from typing import List, Optional
from loguru import logger

from src.clients.llm_client import LLMClient
from src.models.country import Country
from src.models.document import DocumentMetadata
from src.prompts.document_identification import (
    SYSTEM_PROMPT,
    DOCUMENT_IDENTIFICATION_SCHEMA,
    create_identification_prompt,
)


class DocumentIdentifier:
    """
    Uses LLM to identify relevant legal documents for a country.

    This class leverages a language model to identify what legal documents
    exist for a given country in the data protection and privacy domain.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        temperature: float = 0.3,
        max_tokens: int = 2000
    ):
        """
        Initialize document identifier.

        Args:
            llm_client: LLM client instance
            temperature: Sampling temperature (0.3 recommended for consistency)
            max_tokens: Maximum tokens for LLM response
        """
        self.llm_client = llm_client
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info("Initialized DocumentIdentifier")

    def identify_documents(
        self,
        country: Country,
        known_documents: Optional[dict] = None,
        document_types: Optional[List[str]] = None
    ) -> List[DocumentMetadata]:
        """
        Identify relevant documents for a country using LLM.

        Args:
            country: Country object with metadata
            known_documents: Optional dict of known document names by type
            document_types: Optional list of specific document types to focus on

        Returns:
            List of DocumentMetadata objects

        Raises:
            ValueError: If LLM response is invalid
            Exception: If LLM call fails
        """
        logger.info(f"Identifying documents for {country.name}")

        # Create prompt
        prompt = create_identification_prompt(
            country_name=country.name,
            iso_code=country.iso_code,
            official_languages=country.official_languages,
            government_domains=country.government_domains,
            region=country.region,
            known_documents=known_documents,
            document_types=document_types
        )

        try:
            # Call LLM
            logger.debug(f"Calling LLM with temperature={self.temperature}")
            response = self.llm_client.complete_json(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                schema=DOCUMENT_IDENTIFICATION_SCHEMA,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Validate response structure
            if "documents" not in response:
                raise ValueError("LLM response missing 'documents' field")

            documents_data = response["documents"]
            logger.info(f"LLM identified {len(documents_data)} documents")

            # Convert to DocumentMetadata objects
            documents = []
            for doc_data in documents_data:
                try:
                    # Create DocumentMetadata with required fields
                    doc = DocumentMetadata(
                        document_type=doc_data["document_type"],
                        official_name=doc_data["official_name"],
                        description=doc_data["description"],
                        expected_language=doc_data["expected_language"],
                        priority_score=doc_data["priority_score"],
                        alternate_names=doc_data.get("alternate_names", []),
                        expected_file_types=doc_data.get("expected_file_types", ["pdf", "html"])
                    )
                    documents.append(doc)
                    logger.debug(
                        f"Created DocumentMetadata: {doc.official_name} "
                        f"(priority: {doc.priority_score})"
                    )
                except Exception as e:
                    logger.warning(f"Failed to create DocumentMetadata: {e}")
                    logger.debug(f"Document data: {doc_data}")
                    continue

            if not documents:
                logger.warning("No valid documents created from LLM response")

            # Log metadata if present
            if "metadata" in response:
                metadata = response["metadata"]
                logger.info(
                    f"LLM metadata - Total: {metadata.get('total_documents', 'N/A')}, "
                    f"Notes: {metadata.get('notes', 'None')}"
                )

            return documents

        except ValueError as e:
            logger.error(f"Invalid LLM response: {e}")
            raise

        except Exception as e:
            logger.error(f"Error identifying documents: {e}")
            raise

    def identify_documents_simple(self, country_name: str) -> List[DocumentMetadata]:
        """
        Simplified method that takes just a country name.

        Args:
            country_name: Name of the country

        Returns:
            List of DocumentMetadata objects

        Raises:
            ValueError: If country information is insufficient
        """
        # Create a minimal Country object
        country = Country(
            name=country_name,
            iso_code="",  # Will be empty but that's ok for simple use
            official_languages=["en"],  # Default assumption
            government_domains=[],
            region="",
            metadata={}
        )

        return self.identify_documents(country)

    def get_high_priority_documents(
        self,
        country: Country,
        min_priority: int = 8
    ) -> List[DocumentMetadata]:
        """
        Get only high-priority documents for a country.

        Args:
            country: Country object
            min_priority: Minimum priority score (default 8)

        Returns:
            List of high-priority DocumentMetadata objects
        """
        all_documents = self.identify_documents(country)

        high_priority = [
            doc for doc in all_documents
            if doc.priority_score >= min_priority
        ]

        logger.info(
            f"Filtered to {len(high_priority)}/{len(all_documents)} "
            f"documents with priority >= {min_priority}"
        )

        return high_priority

    def get_documents_by_type(
        self,
        country: Country,
        document_type: str
    ) -> List[DocumentMetadata]:
        """
        Get documents of a specific type for a country.

        Args:
            country: Country object
            document_type: Type of document (e.g., "data_protection_law")

        Returns:
            List of DocumentMetadata objects of specified type
        """
        all_documents = self.identify_documents(
            country,
            document_types=[document_type]
        )

        # Filter to exact type (in case LLM returns others)
        typed_docs = [
            doc for doc in all_documents
            if doc.document_type == document_type
        ]

        logger.info(
            f"Found {len(typed_docs)} documents of type '{document_type}'"
        )

        return typed_docs
