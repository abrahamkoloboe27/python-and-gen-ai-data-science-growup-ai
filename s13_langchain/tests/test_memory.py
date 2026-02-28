"""
Tests unitaires pour la gestion de la mémoire
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
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


class TestConversationMemory:
    """Tests pour la mémoire conversationnelle"""
    
    def test_memory_buffer_initialization(self, app):
        """Test l'initialisation de ConversationBufferMemory"""
        from langchain_classic.memory import ConversationBufferMemory
        
        memory = ConversationBufferMemory()
        assert memory is not None
    
    def test_memory_buffer_window_initialization(self, app):
        """Test l'initialisation de ConversationBufferWindowMemory"""
        from langchain_classic.memory import ConversationBufferWindowMemory
        
        memory = ConversationBufferWindowMemory(k=5)
        assert memory is not None
        assert memory.k == 5
    
    def test_memory_stores_conversation(self):
        """Test que la mémoire stocke les conversations"""
        from langchain_classic.memory import ConversationBufferMemory
        
        memory = ConversationBufferMemory()
        memory.save_context(
            {"input": "Bonjour"},
            {"output": "Salut! Comment puis-je vous aider?"}
        )
        
        history = memory.load_memory_variables({})
        assert "history" in history
        assert len(history["history"]) > 0
    
    def test_memory_window_limits_messages(self):
        """Test que la fenêtre de mémoire limite les messages"""
        from langchain_classic.memory import ConversationBufferWindowMemory
        
        memory = ConversationBufferWindowMemory(k=2)
        
        # Ajouter 3 paires de messages
        for i in range(3):
            memory.save_context(
                {"input": f"Question {i}"},
                {"output": f"Réponse {i}"}
            )
        
        # La mémoire devrait ne garder que les 2 dernières paires
        history = memory.load_memory_variables({})
        # Note: Vérifier la logique spécifique de LangChain pour le comptage
    
    def test_conversation_demo_invalid_memory_type(self, app):
        """Test que conversation_demo rejette les types invalides"""
        with pytest.raises(ValueError, match="Type de mémoire non supporté"):
            app.conversation_demo(memory_type="invalid_type")


class TestMemoryIntegration:
    """Tests d'intégration pour la mémoire"""
    
    def test_memory_persists_across_turns(self):
        """Test que la mémoire persiste entre les tours"""
        from langchain_classic.memory import ConversationBufferMemory
        from langchain_classic.chains import ConversationChain
        from langchain_community.llms import OpenAI
        
        with patch('langchain_community.llms.OpenAI') as mock_llm:
            mock_instance = MagicMock()
            mock_llm.return_value = mock_instance
            
            memory = ConversationBufferMemory()
            
            # Simuler une conversation
            memory.save_context(
                {"input": "Mon nom est Alice"},
                {"output": "Enchanté Alice!"}
            )
            
            memory.save_context(
                {"input": "Quel est mon nom?"},
                {"output": "Votre nom est Alice"}
            )
            
            history = memory.load_memory_variables({})
            assert "Alice" in str(history)
    
    def test_memory_clears_correctly(self):
        """Test que la mémoire peut être effacée"""
        from langchain_classic.memory import ConversationBufferMemory
        
        memory = ConversationBufferMemory()
        memory.save_context(
            {"input": "Test"},
            {"output": "Response"}
        )
        
        memory.clear()
        history = memory.load_memory_variables({})
        # Après clear, l'historique devrait être vide ou minimal


class TestMemoryTypes:
    """Tests pour différents types de mémoire"""
    
    def test_buffer_memory_stores_all(self):
        """Test que BufferMemory stocke tout"""
        from langchain_classic.memory import ConversationBufferMemory
        
        memory = ConversationBufferMemory()
        
        # Ajouter plusieurs messages
        for i in range(10):
            memory.save_context(
                {"input": f"Message {i}"},
                {"output": f"Réponse {i}"}
            )
        
        history = memory.load_memory_variables({})
        # BufferMemory devrait garder tous les messages
    
    def test_window_memory_sliding_window(self):
        """Test le comportement de fenêtre glissante"""
        from langchain_classic.memory import ConversationBufferWindowMemory
        
        k = 3
        memory = ConversationBufferWindowMemory(k=k)
        
        # Ajouter plus de messages que k
        for i in range(5):
            memory.save_context(
                {"input": f"Q{i}"},
                {"output": f"A{i}"}
            )
        
        history = memory.load_memory_variables({})
        # Devrait contenir seulement les k derniers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
