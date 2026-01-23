"""
Tests unitaires pour les chains LangChain
"""

import pytest
from unittest.mock import Mock, patch
import sys
import os

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import LangChainApp


@pytest.fixture
def mock_env():
    """Mock des variables d'environnement"""
    with patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-key-123',
        'MODEL_NAME': 'gpt-3.5-turbo',
        'TEMPERATURE': '0.7',
        'MAX_TOKENS': '500'
    }):
        yield


@pytest.fixture
def app(mock_env):
    """Fixture pour l'application"""
    with patch('app.ChatOpenAI'):
        return LangChainApp()


class TestSimpleChain:
    """Tests pour la simple chain"""
    
    def test_simple_chain_returns_string(self, app):
        """Test que simple_chain retourne une string"""
        with patch.object(app, 'llm') as mock_llm:
            mock_chain = Mock()
            mock_chain.run.return_value = "Explication test"
            
            with patch('app.LLMChain', return_value=mock_chain):
                result = app.simple_chain_demo("test topic")
                assert isinstance(result, str)
                assert len(result) > 0
    
    def test_simple_chain_calls_llm(self, app):
        """Test que simple_chain appelle le LLM"""
        with patch.object(app, 'llm') as mock_llm:
            mock_chain = Mock()
            mock_chain.run.return_value = "Test response"
            
            with patch('app.LLMChain', return_value=mock_chain) as mock_chain_class:
                app.simple_chain_demo("transformers")
                mock_chain_class.assert_called_once()
                mock_chain.run.assert_called_once()


class TestSequentialChain:
    """Tests pour la sequential chain"""
    
    def test_sequential_chain_returns_dict(self, app):
        """Test que sequential_chain retourne un dict"""
        with patch.object(app, 'llm') as mock_llm:
            mock_chain = Mock()
            mock_chain.return_value = {
                "summary": "Résumé test",
                "analysis": "Analyse test"
            }
            
            with patch('app.SequentialChain', return_value=mock_chain):
                result = app.sequential_chain_demo("Test text")
                assert isinstance(result, dict)
                assert "summary" in result
                assert "analysis" in result
    
    def test_sequential_chain_processes_text(self, app):
        """Test que sequential_chain traite le texte correctement"""
        test_text = "Ceci est un texte de test pour l'analyse."
        
        with patch.object(app, 'llm') as mock_llm:
            mock_chain = Mock()
            mock_chain.return_value = {
                "summary": "Court résumé",
                "analysis": "Sentiment positif"
            }
            
            with patch('app.SequentialChain', return_value=mock_chain):
                result = app.sequential_chain_demo(test_text)
                assert result["summary"]
                assert result["analysis"]


class TestQADemo:
    """Tests pour le Q&A avec contexte"""
    
    def test_qa_returns_string(self, app):
        """Test que qa_demo retourne une string"""
        context = "Le RAG combine retrieval et génération."
        question = "Qu'est-ce que le RAG?"
        
        with patch.object(app, 'llm') as mock_llm:
            mock_chain = Mock()
            mock_chain.run.return_value = "Le RAG est..."
            
            with patch('app.LLMChain', return_value=mock_chain):
                result = app.qa_demo(context, question)
                assert isinstance(result, str)
                assert len(result) > 0
    
    def test_qa_uses_context(self, app):
        """Test que qa_demo utilise le contexte"""
        context = "Test context"
        question = "Test question"
        
        with patch.object(app, 'llm') as mock_llm:
            mock_chain = Mock()
            mock_chain.run.return_value = "Answer"
            
            with patch('app.LLMChain', return_value=mock_chain):
                result = app.qa_demo(context, question)
                mock_chain.run.assert_called_once()
                call_kwargs = mock_chain.run.call_args[1]
                assert call_kwargs['context'] == context
                assert call_kwargs['question'] == question


class TestFewShot:
    """Tests pour le few-shot learning"""
    
    def test_few_shot_returns_classification(self, app):
        """Test que few_shot retourne une classification"""
        with patch.object(app, 'llm') as mock_llm:
            mock_chain = Mock()
            mock_chain.run.return_value = "positif"
            
            with patch('app.LLMChain', return_value=mock_chain):
                result = app.few_shot_demo("J'adore ce produit!")
                assert isinstance(result, str)
                assert result in ["positif", "négatif", "neutre"] or len(result) > 0


class TestAppInitialization:
    """Tests pour l'initialisation de l'application"""
    
    def test_app_requires_api_key(self):
        """Test que l'app nécessite une clé API"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                LangChainApp()
    
    def test_app_initializes_with_valid_config(self, mock_env):
        """Test que l'app s'initialise avec une config valide"""
        with patch('app.ChatOpenAI'):
            app = LangChainApp()
            assert app.llm is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
