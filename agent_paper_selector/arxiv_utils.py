import os
import tempfile
from typing import Dict, List
import arxiv
import pymupdf4llm
from langchain.text_splitter import RecursiveCharacterTextSplitter


def fetch_papers(topic: str, max_results: int = 10) -> List[Dict]:
    """
    Fetch a number of papers from arxiv given a topic and a maximum number of them
    :param topic:
    :param max_results:
    :return:
    """
    client = arxiv.Client()
    search = arxiv.Search(query=topic,
                           max_results=max_results,
                           sort_by=arxiv.SortCriterion.SubmittedDate)
    papers = []
    for result in client.results(search):
        papers.append({
            "title": result.title,
            "abstract": result.summary,
            "published": result.published.strftime("%Y-%m-%d"),
            "id": os.path.basename(result.entry_id)
        })
    return papers


def split_text(text, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


def add_content_to_papers(papers: List[Dict]) -> List[Dict]:
    """
    Add the content of the paper to each of the dictionaries. It is assume that each paper
    has the key 'id'.
    :param papers: A list o dictionaries with at least the key 'id'
    :return:
    """
    temporary_file = 'temporary_file.pdf'
    for paper in papers:
        try:
            paper_obj = next(arxiv.Client().results(arxiv.Search(id_list=[paper['id']])))
            paper_obj.download_pdf(filename=temporary_file)
        except arxiv.HTTPError as e:
            print(f"Paper id {paper['id']} not found")

        content = pymupdf4llm.to_markdown(temporary_file)
        paper['content'] = content

    if os.path.exists(temporary_file):
        os.remove(temporary_file)
    return papers
