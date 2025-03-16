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
    max_size_chunk: int = 4000
    days_to_go_back_in_search: Optional[Any] = Field(
        default=1,
        description="the number of days to search papers. 1 day means to search only today, 2 days yesterday and today and so on",
    )

    def __post_init__(self):
        """Initialize Agents and Tasks"""
        self.get_agents()
        self.get_tasks()
        self.get_crews()

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
                         "if provided in the paper. If there is any new type of deep layer or loss provide a proper "
                         "explanation about its advantages. If the paper presents a new"
                         "dataset for a specific purpose then get the purpose, number of samples and number of samples "
                         "per class. Do not provide information about the bibliography unless it is absolutely necessary"
                         "to understand the paper. Try to make the summary succint, but it is more important to provide "
                         "an understandable summary."),
            agent=self.summarisation_agent,
            expected_output="A structured summary of the paper.",
        )

        self.categorisation_task = Task(
            description=(
                "Given the following paper summary:\n{summary}\n"
                "And the user's context for rating papers:\n{user_context}\n"
                "Rate the paper as: 'Not worth reading', 'May be worth reading', or 'Worth reading'."
                "Think before replying"
            ),
            agent=self.categorisation_agent,
            expected_output=("The thinking process between tags <thinking> </thinking> and the output rate and only the "
                             "output rate without any other text 'Not worth reading', 'May be worth reading',"
                            " or 'Worth reading' inside tags <output></output>. Any other text can be outside the tags"),
        )

    def get_crews(self):
        self.categorization_crew = Crew(
            agents=[self.categorisation_agent],  # Only the categorizer is needed
            tasks=[self.categorisation_task],
            process=Process.sequential
        )

        self.summarisation_crew = Crew(
            agents=[self.summarisation_agent],  # Both summarization & final categorization
            tasks=[self.summarisation_task],
            process=Process.sequential
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
        filtered_papers = []
        for paper in papers:
            result = self.categorization_crew.kickoff(inputs={
                "summary": paper["abstract"],
                "user_context": self.user_context
            })

            # Store result and filter out irrelevant papers
            paper["initial_rating"] = result.raw
            if "not worth" not in result.raw.lower():  # Only keep relevant papers
                filtered_papers.append(paper)

        # Step 3: Download full paper content
        enriched_papers = add_content_to_papers(filtered_papers)

        # Step 4: Summarise the full content and perform classification again
        for paper in enriched_papers:
            content = paper["content"]

            # Chunk large content for summarization
            if len(content) > self.max_size_chunk:
                chunks = split_text(content, chunk_size=self.max_size_chunk)
                chunk_summaries = [self.summarisation_crew.kickoff(inputs={"text": chunk}).raw for chunk in chunks]
                final_summary = " ".join(chunk_summaries)
            else:
                final_summary = self.summarisation_crew.kickoff(inputs={"text": content}).raw

            # Store summarized content
            paper["summary"] = final_summary

            # Run final categorization
            final_rating = self.categorization_crew.kickoff(inputs={
                "summary": paper["summary"],
                "user_context": self.user_context
            })

            paper["final_rating"] = final_rating.raw

        # Step 5: Save results
        self.save_results(enriched_papers)


if __name__ == "__main__":
    selector = PaperSelector(
        topic="computer vision",
        user_context=("I am interested in papers that present innovative and useful ideas. "
                      "In addition, they need to be for general purpose and not for a specif domain like medicine. "
                      "My order of preference is object detection, classification, segmentation and anomaly detection"
                      "Only get papers that presents a dataset or use deep learning."),
        output_filename="papers.csv",
        max_results=2
    )
    selector.run()
