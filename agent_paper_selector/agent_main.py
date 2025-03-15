import csv
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from crewai import Crew, Task, Agent
import pandas as pd
from pydantic import Field

from agents import CategorisationAgent
from arxiv_utils import fetch_papers, split_text, add_content_to_papers


class PaperSelector(dataclass):
    output_filename: str
    user_context: str
    days_to_go_back_in_search: Optional[Any] = Field(
        default=1,
        description="the number of days to search papers. 1 day means to search only today, 2 days yesterday and today and so on",
    )
    topic: str
    summarisation_agent: Agent = None
    categorisation_agent: Agent = None

    def __post_init__(self):
        """Initialize Agents and Tasks"""
        self.get_agents()
        self.get_tasks()

    def get_agents(self):
        """Define the agents"""
        self.summarisation_agent = Agent(
            role="Research Paper Summariser",
            goal="Summarise academic papers efficiently.",
            backstory="Expert in academic literature, summarising research papers with key insights.",
            verbose=True
        )

        self.categorisation_agent = Agent(
            role="Paper Evaluator",
            goal="Evaluate research papers and categorize them based on their relevance to user context.",
            backstory="Experienced in academic review, capable of assessing paper relevance.",
            verbose=True
        )

    def get_tasks(self):
        """Define CrewAI tasks"""
        self.summarisation_task = Task(
            description=("Summarise the following research paper's content efficiently."
                         "{text}"),
            agent=self.summarisation_agent,
            expected_output="A structured summary of each research paper.",
        )

        self.initial_categorisation_task = Task(
            description=(
                "Given the following paper summary:\n{summary}\n"
                "And the user's context for rating papers:\n{user_context}\n"
                "Rate the paper as: 'Not worth reading', 'May be worth reading', or 'Worth reading'."
            ),
            agent=self.categorisation_agent,
            expected_output="A category assigned to each paper based on its abstract and user context.",
        )

        self.final_categorisation_task = Task(
            description=(
                "Re-evaluate papers after full content is summarized.\n"
                "Update their categorization based on deeper insights."
                "Given the following paper summary:\n{summary}\n"
                "And the user's context for rating papers:\n{user_context}\n"
                "Rate the paper as: 'Not worth reading', 'May be worth reading', or 'Worth reading'."
            ),
            agent=self.categorisation_agent,
            expected_output="A refined category assigned to each paper after full content analysis.",
        )

    # Function to save results as CSV
    def save_results(self, papers: List[Dict], filename: str = "papers.csv"):
        df = pd.DataFrame(papers)
        print(df.head())
        df.to_csv(filename)

    def _initial_assessment(self, papers):
        paper_subset = []
        for paper in papers:
            assessment = self.categorisation_agent.kickoff(inputs={"text": paper["summary"], 'user_context': self.user_context})
            if 'not worth' not in assessment.lower():
                paper_subset.append(paper)

        return paper_subset

    def _summarise_papers(self, papers):
        papers_with_updated_summary = []
        for paper in papers:
            updated_paper = paper.copy()
            content = updated_paper.pop('content')
            if len(content) > 4000:  # If too long for a single LLM call
                chunks = split_text(content)
                chunk_summaries = [self.summarization_agent.kickoff(inputs={"text": chunk}) for chunk in chunks]
                final_summary = " ".join(chunk_summaries)
            else:
                final_summary = self.summarization_agent.kickoff(inputs={"text": content})

            updated_paper['summary'] = final_summary
            papers_with_updated_summary.append(updated_paper)

        return papers_with_updated_summary

    def _final_assessment(self, papers):
        for paper in papers:
            assessment = self.categorisation_agent.kickoff(inputs={"text": paper["summary"], 'user_context': self.user_context})
            paper['assessment'] = assessment
            if 'content' in paper:
                del paper['content']

        return papers


    # Main function to run CrewAI pipeline
    def run(self, max_results: int = 10):
        papers = fetch_papers(self.topic, max_results)

        subset_papers = self._initial_assessment(papers)
        subset_papers = add_content_to_papers(subset_papers)
        subset_papers = self._summarise_papers(subset_papers)
        subset_papers = self._final_assessment(subset_papers)

        self.save_results(subset_papers)


if __name__ == "__main__":
    runner = PaperSelector(topic="computer vision",
                  user_context="I am interested in image segmentation.",
                  max_results=5)
    runner.run()
