from io import BytesIO
from datetime import date
import re
from urllib.parse import urljoin, urlparse

import arxiv
import requests
from pypdf import PdfReader
from smolagents import tool

MAX_PDF_BYTES = 15 * 1024 * 1024
MAX_PDF_PAGES = 12
MAX_PDF_TEXT_CHARS = 12000
ARXIV_HOSTS = {"arxiv.org", "export.arxiv.org"}


@tool
def search_arxiv(
    query: str,
    max_results: int = 3,
    category: str = "",
    author: str = "",
    start_date: str = "",
    end_date: str = "",
) -> list:
    """Searches arXiv with optional category, author, and submission-date filters.

    Args:
        query: The search terms to look for on arXiv.
        max_results: How many papers to return. Defaults to 3.
        category: Optional arXiv category such as cs.AI.
        author: Optional author name.
        start_date: Optional inclusive start date in YYYY-MM-DD format.
        end_date: Optional inclusive end date in YYYY-MM-DD format.
    """
    query = " ".join(query.split())
    if not query or len(query) > 250:
        raise ValueError("Search terms must be between 1 and 250 characters.")
    if not 1 <= int(max_results) <= 10:
        raise ValueError("max_results must be between 1 and 10.")

    query_parts = [f"({query})"]
    category = category.strip()
    if category:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9-]*(?:\.[A-Za-z0-9]{2})?", category):
            raise ValueError("Enter a valid arXiv category, such as cs.AI.")
        query_parts.append(f"cat:{category}")

    author = " ".join(author.split())
    if len(author) > 100:
        raise ValueError("Author filter must be 100 characters or fewer.")
    if author:
        author = author.replace('"', "").replace("\\", "")
        query_parts.append(f'au:"{author}"')

    start_date = start_date.strip()
    end_date = end_date.strip()
    if start_date or end_date:
        start = date.fromisoformat(start_date) if start_date else date(1991, 1, 1)
        end = date.fromisoformat(end_date) if end_date else date.today()
        if start > end:
            raise ValueError("Start date must be on or before end date.")
        query_parts.append(
            f"submittedDate:[{start:%Y%m%d}0000 TO {end:%Y%m%d}2359]"
        )

    search_query = " AND ".join(query_parts)
    client = arxiv.Client()
    search = arxiv.Search(
        query=search_query,
        max_results=int(max_results),
        sort_by=arxiv.SortCriterion.Relevance,
    )
    results = []
    for paper in client.results(search):
        author_names = [author.name for author in paper.authors]
        results.append({
            "title": paper.title,
            "authors": ", ".join(author_names),
            "author_names": author_names,
            "year": paper.published.year,
            "url": paper.entry_id,
            "pdf_url": paper.pdf_url,
            "abstract": paper.summary,
            "category": paper.primary_category,
            "doi": paper.doi,
        })
    return results


def read_arxiv_pdf_pages(pdf_url: str) -> list:
    current_url = pdf_url
    response = None
    for _ in range(4):
        parsed_url = urlparse(current_url)
        if (
            parsed_url.scheme != "https"
            or parsed_url.hostname not in ARXIV_HOSTS
            or parsed_url.port not in (None, 443)
            or not parsed_url.path.startswith("/pdf/")
        ):
            raise ValueError("Only arXiv PDF URLs are allowed.")

        response = requests.get(
            current_url,
            headers={"User-Agent": "ArxivResearchAgent/1.0"},
            timeout=(5, 30),
            stream=True,
            allow_redirects=False,
        )
        if not response.is_redirect:
            break
        redirect_url = response.headers.get("Location")
        response.close()
        if not redirect_url:
            raise ValueError("The arXiv PDF redirect was invalid.")
        current_url = urljoin(current_url, redirect_url)
    else:
        raise ValueError("Too many redirects while retrieving the arXiv PDF.")

    try:
        response.raise_for_status()
        pdf_bytes = bytearray()
        for chunk in response.iter_content(chunk_size=64 * 1024):
            pdf_bytes.extend(chunk)
            if len(pdf_bytes) > MAX_PDF_BYTES:
                raise ValueError("The PDF exceeds the 15 MB size limit.")
    finally:
        response.close()

    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError("The arXiv response was not a valid PDF.")

    reader = PdfReader(BytesIO(pdf_bytes), strict=False)
    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs cannot be read.")

    extracted_pages = []
    remaining_chars = MAX_PDF_TEXT_CHARS
    for page_number, page in enumerate(reader.pages):
        if page_number >= MAX_PDF_PAGES or remaining_chars <= 0:
            break
        page_text = (page.extract_text() or "").strip()
        if page_text:
            page_text = page_text[:remaining_chars]
            extracted_pages.append({"page": page_number + 1, "text": page_text})
            remaining_chars -= len(page_text)

    if not extracted_pages:
        raise ValueError("No readable text was found in the PDF.")
    return extracted_pages


def read_arxiv_pdf(pdf_url: str) -> str:
    pages = read_arxiv_pdf_pages(pdf_url)
    return "\n\n".join(
        f"[Page {page['page']}]\n{page['text']}" for page in pages
    )