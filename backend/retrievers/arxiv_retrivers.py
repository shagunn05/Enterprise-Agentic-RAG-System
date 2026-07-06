import arxiv

def get_arxiv_retriever(query: str, max_results: int = 3, sort_by_date: bool = False) -> str:
    sort_criterion = arxiv.SortCriterion.SubmittedDate if sort_by_date else arxiv.SortCriterion.Relevance
    
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_criterion,
        sort_order=arxiv.SortOrder.Descending
    )
    client = arxiv.Client()
    results = list(client.results(search))

    if not results:
        return "No relevant arxiv papers found."

    output = ""
    for i, result in enumerate(results):
        output += f"\nResult {i+1}\n"
        output += f"Title: {result.title}\n"
        output += f"Published: {result.published.strftime('%Y-%m-%d')}\n"
        output += f"Authors: {', '.join([a.name for a in result.authors])}\n"
        output += f"Summary: {result.summary[:500]}\n"

    return output