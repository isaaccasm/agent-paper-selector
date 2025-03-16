import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from crewai import Crew, Task, Agent, Process
import pandas as pd
from pydantic import Field

from agent_paper_selector.arxiv_utils import fetch_papers, split_text, add_content_to_papers
from llm_utils.load_keys import load_open_ai_key


os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo' #'gpt-4o'
load_open_ai_key()


@dataclass
class PaperSelector:
    output_filename: str
    user_context: str
    topic: str
    max_results: int = 10
    days_to_go_back_in_search: Optional[Any] = Field(
        default=1,
        description="the number of days to search papers. 1 day means to search only today, 2 days yesterday and today and so on",
    )

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
            verbose=True,
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
            description=("Summarise the following research paper's content efficiently.\n\n "
                         "{text}\n\n"
                         "Get the main ideas, its pros and cons and how it compares to the state-of-the-art "
                         "if provided. If there is any new type of deep layer or loss. If the paper presents a new"
                         "dataset for a specific purpose then get the purpose, number of samples and number of samples "
                         "per class."),
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

    def save_results(self, papers: List[Dict], filename: str = "papers.csv"):
        """Save the results as a CSV file"""
        df = pd.DataFrame(papers)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

    def run(self):
        """Main function to execute the workflow using CrewAI"""
        # Step 1: Fetch Papers
        papers = fetch_papers(self.topic, self.max_results)

        # Step 2: Perform initial categorization based on abstract
        initial_results = []
        for paper in papers:
            rating = self.initial_categorisation_task.execute(inputs={
                "text": paper["abstract"],
                "user_context": self.user_context
            })
            if "not worth" not in rating.lower():
                paper["initial_rating"] = rating
                initial_results.append(paper)

        # Step 3: Download full paper content
        enriched_papers = add_content_to_papers(initial_results)

        # Step 4: Summarise the full content
        max_size_chunk = 4000
        for paper in enriched_papers:
            content = paper.pop("content")
            if len(content) > max_size_chunk:
                chunks = split_text(content, chunk_size=max_size_chunk)
                chunk_summaries = [self.summarisation_task.execute(inputs={"text": chunk}) for chunk in chunks]
                final_summary = " ".join(chunk_summaries)
            else:
                final_summary = self.summarisation_task.execute(inputs={"text": content})
            paper["summary"] = final_summary

        # Step 5: Final categorization
        for paper in enriched_papers:
            final_rating = self.final_categorisation_task.execute(inputs={
                "text": paper["summary"],
                "user_context": self.user_context
            })
            paper["final_rating"] = final_rating

        # Step 6: Save results
        self.save_results(enriched_papers)


if __name__ == "__main__":
    selector = PaperSelector(
        topic="computer vision",
        user_context=("I am interested in papers that present innovative and useful ideas. "
                      "In addition, they need to be for general purpose and not for a specif domain like medicine. "
                      "My order of preference is object detection, classification, segmentation and anomaly detection"
                      "Only get papers that presents a dataset or use deep learning."),
        output_filename="papers.csv",
        max_results=5
    )
    selector.run()
    Crew()