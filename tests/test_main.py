import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from agent_paper_selector.crew_main import PaperSelector  # Replace 'your_module' with actual module name


class TestPaperSelector(unittest.TestCase):

    @patch("agent_paper_selector.crew_main.fetch_papers")
    @patch("agent_paper_selector.crew_main.add_content_to_papers")
    @patch("agent_paper_selector.crew_main.split_text")
    @patch("agent_paper_selector.crew_main.Crew.kickoff")
    @patch("agent_paper_selector.crew_main.pd.DataFrame.to_csv")
    def test_run(self, mock_to_csv, mock_kickoff, mock_split_text, mock_add_content, mock_fetch_papers):
        # Mock fetch_papers
        mock_fetch_papers.return_value = [
            {"id": "1234.5678", "title": "Test Paper", "abstract": "Test abstract."}
        ]

        # Mock initial categorization
        mock_kickoff.side_effect = [
            "May be worth reading",  # Initial categorization
            "Summarised content",  # Summary
            "Worth reading"  # Final categorization
        ]

        # Mock add_content_to_papers
        mock_add_content.return_value = [
            {"id": "1234.5678", "title": "Test Paper", "abstract": "Test abstract.", "content": "Full paper content"}
        ]

        # Mock split_text (no splitting needed in this case)
        mock_split_text.return_value = ["Summarised content"]

        selector = PaperSelector(
            topic="computer vision",
            user_context="I want general-purpose deep learning papers.",
            output_filename="test_papers.csv",
            max_results=1
        )

        selector.run()

        # Assertions
        mock_fetch_papers.assert_called_once()
        mock_add_content.assert_called_once()
        mock_kickoff.assert_called()
        mock_to_csv.assert_called_once()


if __name__ == "__main__":
    unittest.main()
