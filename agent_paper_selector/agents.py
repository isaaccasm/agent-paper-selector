from typing import List, Dict
from crewai import Task, Agent


class CategorisationAgent(Agent):
    def run(self, papers: List[Dict], user_context: str) -> List[Dict]:
        categorised_papers = []
        for paper in papers:
            rating = self.rate_paper(paper["title"], paper["summary"], user_context)
            categorised_papers.append({**paper, "rating": rating})
        return categorised_papers

    def rate_paper(self, title: str, summary: str, user_context: str) -> str:
        """ Calls the LLM to evaluate and categorise a paper. """
        prompt = f"""
        You are an AI research assistant evaluating the relevance of research papers.

        **User's Interest:** {user_context}
        **Paper Title:** {title}
        **Paper Summary:** {summary}

        **Task:** 
        Based on the summary and user's interest, rate the paper as:
        - "Worth Reading" if it strongly aligns with the user's interest.
        - "May Be Worth Reading" if it has some relevance.
        - "Not Worth Reading" if it is unrelated.

        **Response Format:** 
        - Only return one of: Worth Reading, May Be Worth Reading, Not Worth Reading.
        """

        response = client.completions.create(
            model="gpt-4-turbo",  # Use GPT-4 Turbo or a model of your choice
            prompt=prompt,
            max_tokens=10,
            temperature=0.2  # Keep it deterministic
        )

        return response.choices[0].text.strip()
