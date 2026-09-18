import arxiv
from smolagents import tool


@tool
def search_arxiv(query: str, max_results: int = 3) -> list:
    """Searches arXiv for academic papers matching a query.

    Args:
        query: The search terms to look for on arXiv.
        max_results: How many papers to return. Defaults to 3.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    results = []
    for paper in client.results(search):
        # Extract authors into a comma-separated string
        authors = ", ".join([author.name for author in paper.authors])
        # Append as a structured dictionary rather than a raw string
        results.append({
            "title": paper.title,
            "authors": authors,
            "year": paper.published.year,
            "url": paper.entry_id,
            "abstract": paper.summary
        })
    return results