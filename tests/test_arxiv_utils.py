import unittest
from unittest.mock import patch, MagicMock
from agent_paper_selector.arxiv_utils import fetch_papers, split_text, add_content_to_papers


class TestArxivFunctions(unittest.TestCase):

    @patch('agent_paper_selector.arxiv_utils.arxiv.Client')
    def test_fetch_papers(self, mock_client):
        # Mock the result returned by arxiv
        mock_paper = MagicMock()
        mock_paper.title = "Test Paper"
        mock_paper.summary = "This is a test summary."
        mock_paper.published.strftime.return_value = "2025-03-16"
        mock_paper.entry_id = "http://arxiv.org/abs/1234.5678"

        mock_client_instance = mock_client.return_value
        mock_client_instance.results.return_value = [mock_paper]

        papers = fetch_papers("test topic", max_results=1)
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]['title'], "Test Paper")
        self.assertEqual(papers[0]['summary'], "This is a test summary.")
        self.assertEqual(papers[0]['published'], "2025-03-16")
        self.assertEqual(papers[0]['id'], "http://arxiv.org/abs/1234.5678")

    def test_split_text(self):
        text = "This is a long text. " * 100  # Creating a long text
        chunks = split_text(text, chunk_size=50, chunk_overlap=10)
        self.assertTrue(all(len(chunk) <= 50 for chunk in chunks))
        self.assertGreater(len(chunks), 1)

    @patch('agent_paper_selector.arxiv_utils.arxiv.Client')
    @patch('agent_paper_selector.arxiv_utils.pymupdf4llm.to_markdown')
    def test_add_content_to_papers(self, mock_to_markdown, mock_client):
        # Mocking paper fetch
        mock_paper = MagicMock()
        mock_paper.download_pdf = MagicMock()

        mock_client_instance = mock_client.return_value
        mock_client_instance.results.return_value = [mock_paper]

        # Mocking PDF to markdown conversion
        mock_to_markdown.return_value = "Extracted content from PDF"

        papers = [{"id": "1234.5678"}]
        enriched_papers = add_content_to_papers(papers)

        self.assertEqual(len(enriched_papers), 1)
        self.assertIn("content", enriched_papers[0])
        self.assertEqual(enriched_papers[0]["content"], "Extracted content from PDF")


if __name__ == '__main__':
    unittest.main()
