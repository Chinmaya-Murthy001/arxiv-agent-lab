import gradio as gr
import spaces

from tools import search_arxiv

# Satisfy ZeroGPU startup detection without allocating a GPU for hosted inference.
@spaces.GPU
def decoy_function():
    pass

def agent_chat(user_input):
    query = user_input.strip()
    if not query:
        return "Enter search terms to find papers on arXiv."

    try:
        papers = search_arxiv(query=query, max_results=3)
    except Exception:
        return "The arXiv search failed. Please try again."

    if not papers:
        return "No arXiv papers found for that search."

    markdown_output = "### arXiv Search Results\n\n"
    for paper in papers:
        title = paper["title"].replace("\n", " ").strip()
        authors = paper["authors"].replace("\n", " ").strip()
        abstract = paper["abstract"].replace("\n", " ").strip()
        markdown_output += (
            f"**[{title}]({paper['url']})**\n\n"
            f"By: {authors}  \n"
            f"Year: {paper['year']}\n\n"
            f"{abstract}\n\n---\n\n"
        )

    return markdown_output

demo = gr.Interface(
    fn=agent_chat,
    inputs=gr.Textbox(lines=2, placeholder="Search arXiv, e.g. agentic AI or cs.AI reinforcement learning."),
    outputs=gr.Markdown(label="Agent Output"),
    title="Autonomous arXiv Research Agent",
    description="Live paper metadata from arXiv"
)

demo.launch()