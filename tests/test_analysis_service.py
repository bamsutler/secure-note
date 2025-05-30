import unittest
from unittest.mock import patch, MagicMock

from src.analysis_service import AnalysisService, OllamaProvider
from src.config_service import ConfigurationService

class TestAnalysisService(unittest.TestCase):

    def _get_config_side_effect(self, *args, **kwargs):
        # print(f"Mock config_service.get called with: {args}")
        if args == ('models', 'ollama_api_url'):
            return "http://mock-ollama/api/generate"
        if args == ('models', 'llm_model_name'):
            return "mock-ollama-model"
        if args == ('paths', 'prompt_templates'):
            return "mock_prompt_templates_dir"
        if args == ('prompts', 'core_instructions.txt'):
            return "Core instructions text."
        if args == ('prompts', 'title_prompt.txt'):
            return "Title prompt text."
        if args == ('prompts', 'summary_prompt.txt'):
            return "Summary prompt text."
        if args == ('analysis', 'default_title'):
            return "Default Analysis Title"
        if args == ('models', 'temperature'):
            return 0.1
        if args == ('models', 'top_k'):
            return 50
        if args == ('models', 'top_p'):
            return 0.9
        if args == ('models', 'num_ctx'):
            return 2048
        # Add other config gets as needed by AnalysisService
        return MagicMock()

    @patch('src.analysis_service.OllamaProvider')
    @patch('src.analysis_service.config_service', new_callable=MagicMock)
    def setUp(self, mock_module_config_service, mock_ollama_provider_class):
        self.mock_config_service_global = mock_module_config_service
        self.mock_config_service_global.get.side_effect = self._get_config_side_effect

        self.test_instance_config_service = MagicMock(spec=ConfigurationService)
        self.test_instance_config_service.get.side_effect = self._get_config_side_effect

        self.mock_ollama_provider_instance = MagicMock(spec=OllamaProvider)
        mock_ollama_provider_class.return_value = self.mock_ollama_provider_instance
        
        self.analysis_service = AnalysisService(provider_type="ollama")
        self.assertEqual(self.analysis_service.provider, self.mock_ollama_provider_instance)
        mock_ollama_provider_class.assert_called_once()

    def test_analyze_transcription_success(self):
        test_transcription = "This is a test transcription about a meeting discussing project alpha and its deadlines."
        expected_model_used = "mock-ollama-model"
        
        mock_ollama_analysis_result = {
            "title": "Project Alpha Meeting",
            "summary": "The meeting covered project alpha and its upcoming deadlines.",
            "key_topics": ["project alpha", "deadlines"],
            "action_items": ["Follow up on deadline for task A."],
            "open_questions": ["When is the final deadline?"],
            "model_used": expected_model_used,
            "full_markdown_response": "# Project Alpha Meeting...",
            "error": None
        }
        self.mock_ollama_provider_instance.analyze.return_value = mock_ollama_analysis_result

        results = self.analysis_service.analyze_transcription(test_transcription)

        self.mock_ollama_provider_instance.analyze.assert_called_once_with(test_transcription, model_config=None)
        
        self.assertIsNotNone(results)
        self.assertEqual(results['title'], mock_ollama_analysis_result['title'])
        self.assertEqual(results['summary'], mock_ollama_analysis_result['summary'])
        self.assertEqual(results['action_items'], mock_ollama_analysis_result['action_items'])
        self.assertEqual(results['model_used'], expected_model_used)
        self.assertIsNone(results.get('error'))

    def test_analyze_transcription_empty_input(self):
        results = self.analysis_service.analyze_transcription("")
        self.assertIsNotNone(results)
        self.assertIn("error", results)
        self.assertEqual(results['error'], "Transcription was empty.")
        self.mock_ollama_provider_instance.analyze.assert_not_called()

    @patch('src.analysis_service.json.loads')
    def test_analyze_transcription_llm_returns_invalid_json(self, mock_json_loads):
        test_transcription = "A valid transcription that will lead to invalid JSON output from LLM."
        expected_model_used = "mock-ollama-model"

        self.mock_ollama_provider_instance.analyze.return_value = {
            "title": "Error processing", 
            "summary": "", 
            "key_topics": [], 
            "action_items": [], 
            "open_questions": [],
            "model_used": expected_model_used,
            "error": "Ollama returned invalid JSON",
            "full_markdown_response": "Ollama analysis failed..."
        }

        results = self.analysis_service.analyze_transcription(test_transcription)

        self.assertIsNotNone(results)
        self.assertIn("error", results)
        self.assertTrue("Ollama returned invalid JSON" in results['error'])
        self.assertEqual(results['model_used'], expected_model_used)
        mock_json_loads.assert_not_called()

    def test_analyze_transcription_llm_api_error(self):
        test_transcription = "Test transcription for API error scenario."
        expected_model_used = "mock-ollama-model"
        
        self.mock_ollama_provider_instance.analyze.return_value = {
            "title": "API Error", 
            "summary": "", 
            "key_topics": [], 
            "action_items": [], 
            "open_questions": [],
            "model_used": expected_model_used,
            "error": "Ollama API request failed - Connection error",
            "full_markdown_response": "Ollama analysis failed..."
        }

        results = self.analysis_service.analyze_transcription(test_transcription)

        self.assertIsNotNone(results)
        self.assertIn("error", results)
        self.assertIn("Ollama API request failed - Connection error", results['error'])
        self.assertEqual(results.get('model_used'), expected_model_used)

    def test_analyze_transcription_llm_returns_no_choices_or_empty_content(self):
        test_transcription = "Test when LLM returns no choices or empty content."
        expected_model_used = "mock-ollama-model"

        self.mock_ollama_provider_instance.analyze.return_value = {
            "title": "", 
            "summary": "", 
            "key_topics": [], 
            "action_items": [], 
            "open_questions": [],
            "model_used": expected_model_used,
            "error": "Analysis completed but all sections resulted in errors or empty content.",
            "full_markdown_response": "No analysis content generated."
        }

        results = self.analysis_service.analyze_transcription(test_transcription)

        self.assertIsNotNone(results)
        self.assertIn("error", results)
        self.assertIn("errors or empty content", results['error'])
        self.assertEqual(results['model_used'], expected_model_used)

if __name__ == '__main__':
    unittest.main() 