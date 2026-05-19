"""Phase 2: HTTP document retrieval with retry and content-type detection."""

import time
from typing import Optional
from urllib.parse import urlparse

import httpx
from loguru import logger


class DocumentRetriever:
    """Downloads documents from URLs, detecting whether each is a PDF or HTML page."""

    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        user_agent: str = "DataProtectionIndex/1.0",
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = httpx.Client(
            timeout=httpx.Timeout(timeout),
            headers={"User-Agent": user_agent},
            follow_redirects=True,
        )
        logger.info("Initialized DocumentRetriever")

    def retrieve(self, url: str) -> Optional[tuple[bytes, str]]:
        """
        Download content from url.

        Returns (raw_bytes, content_type) where content_type is 'pdf', 'html', or 'unknown'.
        Returns None if all retries are exhausted.
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"Downloading (attempt {attempt}/{self.max_retries}): {url}")
                response = self.session.get(url)
                response.raise_for_status()

                content_type = self._detect_content_type(url, response.headers, response.content)
                logger.debug(f"Downloaded {len(response.content):,} bytes ({content_type}) from {url}")
                return response.content, content_type

            except httpx.HTTPStatusError as e:
                last_error = e
                logger.warning(f"HTTP {e.response.status_code} for {url} (attempt {attempt})")
                # 4xx errors are not retryable
                if 400 <= e.response.status_code < 500:
                    break
            except httpx.RequestError as e:
                last_error = e
                logger.warning(f"Request error for {url} (attempt {attempt}): {e}")

            if attempt < self.max_retries:
                sleep_time = self.retry_delay * (2 ** (attempt - 1))
                logger.debug(f"Retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)

        logger.error(f"Failed to download {url} after {self.max_retries} attempts: {last_error}")
        return None

    def _detect_content_type(
        self,
        url: str,
        headers: httpx.Headers,
        content: bytes,
    ) -> str:
        """Detect whether content is PDF or HTML.

        Priority: Content-Type header → URL extension → magic bytes → default html.
        """
        content_type_header = headers.get("content-type", "").lower()

        if "application/pdf" in content_type_header:
            return "pdf"
        if "text/html" in content_type_header:
            return "html"

        # Fall back to URL extension
        path = urlparse(url).path.lower()
        if path.endswith(".pdf"):
            return "pdf"
        if path.endswith((".html", ".htm")):
            return "html"

        # PDF magic bytes
        if content[:4] == b"%PDF":
            return "pdf"

        return "html"

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
