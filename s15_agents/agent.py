"""
S15 — ReAct Agent Implementation
Agent autonome utilisant le pattern ReAct (Reasoning + Acting)
"""

import os
import re
import json
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import logging

from tools import get_tools, get_tools_description, Tool

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Charger les variables d'environnement
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))


class ReactAgent:
    """
    Agent ReAct (Reasoning + Acting)
    Utilise un LLM pour raisonner et choisir des actions (outils) à exécuter
    """
    
    def __init__(
        self,
        tools: Optional[Dict[str, Tool]] = None,
        max_iterations: int = MAX_ITERATIONS,
        verbose: bool = True
    ):
        """
        Initialiser l'agent
        
        Args:
            tools: Dictionnaire des outils disponibles
            max_iterations: Nombre maximum d'itérations
            verbose: Afficher les logs détaillés
        """
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY non trouvée dans .env")
        
        self.tools = tools or get_tools()
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # Initialiser le LLM
        self.llm = ChatOpenAI(
            model_name=MODEL_NAME,
            temperature=TEMPERATURE
        )
        
        # Construire le system prompt
        self.system_prompt = self._build_system_prompt()
        
        logger.info(f"✅ ReAct Agent initialisé avec {len(self.tools)} outils")
    
    def _build_system_prompt(self) -> str:
        """Construire le prompt système pour l'agent"""
        
        tools_desc = get_tools_description()
        
        prompt = f"""Tu es un agent autonome qui utilise des outils pour résoudre des tâches.

OUTILS DISPONIBLES:
{tools_desc}

FORMAT DE RÉPONSE:
Tu dois répondre en utilisant EXACTEMENT ce format:

Thought: [ton raisonnement sur ce qu'il faut faire]
Action: [nom_de_l_outil]
Action Input: [paramètres de l'outil au format JSON]

OU, si tu as la réponse finale:

Thought: J'ai maintenant toutes les informations nécessaires
Final Answer: [ta réponse finale]

RÈGLES IMPORTANTES:
1. Utilise un outil à la fois
2. Attends l'observation avant de continuer
3. Si un outil échoue, essaie une autre approche
4. Ne répète pas les mêmes actions
5. Sois concis dans tes pensées

EXEMPLE:
Question: Combien font 15% de 200?
Thought: Je dois calculer 15% de 200, ce qui équivaut à 200 * 0.15
Action: calculator
Action Input: {{"expression": "200 * 0.15"}}
Observation: 30.0
Thought: J'ai le résultat du calcul
Final Answer: 15% de 200 font 30.

Commence maintenant!"""
        
        return prompt
    
    def run(self, task: str) -> str:
        """
        Exécuter l'agent sur une tâche
        
        Args:
            task: Tâche à accomplir
            
        Returns:
            Réponse finale
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 AGENT DÉMARRÉ")
        logger.info(f"{'='*60}")
        logger.info(f"Tâche: {task}\n")
        
        # Historique de conversation
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Question: {task}")
        ]
        
        # Boucle ReAct
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"\n--- Itération {iteration}/{self.max_iterations} ---")
            
            # Obtenir la pensée/action du LLM
            response = self.llm(messages)
            content = response.content
            
            if self.verbose:
                print(f"\n🤔 Agent Response (Iteration {iteration}):")
                print(content)
            
            # Parser la réponse
            if "Final Answer:" in content:
                # L'agent a terminé
                final_answer = self._extract_final_answer(content)
                logger.info(f"\n{'='*60}")
                logger.info(f"✅ AGENT TERMINÉ")
                logger.info(f"{'='*60}")
                logger.info(f"Réponse: {final_answer}")
                return final_answer
            
            # Parser l'action
            try:
                thought, action, action_input = self._parse_action(content)
                
                logger.info(f"💭 Thought: {thought}")
                logger.info(f"🔧 Action: {action}")
                logger.info(f"📥 Input: {action_input}")
                
                # Exécuter l'outil
                observation = self._execute_tool(action, action_input)
                logger.info(f"👁️  Observation: {observation}")
                
                # Ajouter à l'historique
                messages.append(response)
                messages.append(HumanMessage(content=f"Observation: {observation}"))
                
            except Exception as e:
                error_msg = f"Erreur: {str(e)}"
                logger.error(error_msg)
                messages.append(HumanMessage(content=error_msg))
        
        # Max iterations atteintes
        logger.warning(f"⚠️  Nombre maximum d'itérations atteint ({self.max_iterations})")
        return "Je n'ai pas pu terminer la tâche dans le nombre d'itérations autorisé."
    
    def _parse_action(self, content: str) -> Tuple[str, str, Dict]:
        """
        Parser le contenu pour extraire thought, action, et input
        
        Args:
            content: Contenu de la réponse du LLM
            
        Returns:
            (thought, action, action_input)
        """
        # Extraire thought
        thought_match = re.search(r'Thought:\s*(.+?)(?=\nAction:|$)', content, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else "No thought"
        
        # Extraire action
        action_match = re.search(r'Action:\s*(\w+)', content)
        if not action_match:
            raise ValueError("Aucune action trouvée dans la réponse")
        action = action_match.group(1).strip()
        
        # Extraire action input
        input_match = re.search(r'Action Input:\s*({.+?}|\[.+?\]|.+?)(?=\n|$)', content, re.DOTALL)
        if not input_match:
            raise ValueError("Aucun input d'action trouvé")
        
        input_str = input_match.group(1).strip()
        
        # Parser le JSON si c'est un dict
        if input_str.startswith('{'):
            try:
                action_input = json.loads(input_str)
            except json.JSONDecodeError:
                # Fallback: extraire manuellement
                action_input = self._extract_params(input_str)
        else:
            # Simple string input
            action_input = {"input": input_str}
        
        return thought, action, action_input
    
    def _extract_params(self, input_str: str) -> Dict:
        """Extraire les paramètres d'une string"""
        params = {}
        # Pattern: "key": "value" ou key: value
        matches = re.findall(r'["\']?(\w+)["\']?\s*:\s*["\']?([^,"\']+)["\']?', input_str)
        for key, value in matches:
            # Essayer de convertir en nombre si possible
            try:
                value = float(value) if '.' in value else int(value)
            except ValueError:
                pass
            params[key] = value
        return params
    
    def _extract_final_answer(self, content: str) -> str:
        """Extraire la réponse finale"""
        match = re.search(r'Final Answer:\s*(.+)', content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return content
    
    def _execute_tool(self, tool_name: str, params: Dict) -> str:
        """
        Exécuter un outil
        
        Args:
            tool_name: Nom de l'outil
            params: Paramètres de l'outil
            
        Returns:
            Observation (résultat)
        """
        if tool_name not in self.tools:
            return f"Erreur: Outil '{tool_name}' n'existe pas. Outils disponibles: {list(self.tools.keys())}"
        
        tool = self.tools[tool_name]
        
        try:
            result = tool.execute(**params)
            return result
        except Exception as e:
            return f"Erreur lors de l'exécution de {tool_name}: {str(e)}"
    
    def list_tools(self) -> List[str]:
        """Lister les outils disponibles"""
        return list(self.tools.keys())


# ============================================================
# FONCTION PRINCIPALE
# ============================================================

def main():
    """Fonction principale pour tester l'agent"""
    
    print("\n" + "="*60)
    print("🤖 REACT AGENT DEMO")
    print("="*60)
    
    try:
        # Créer l'agent
        agent = ReactAgent(verbose=True)
        
        # Exemples de tâches
        test_tasks = [
            "Combien coûte un billet Paris-Londres?",
            "Calcule 15% de 200",
            "Quelle est l'heure actuelle?",
            "Convertis 100 EUR en USD",
        ]
        
        # Menu interactif
        while True:
            print("\n" + "-"*60)
            print("Options:")
            print("1. Tester avec une tâche prédéfinie")
            print("2. Entrer une tâche personnalisée")
            print("3. Tester toutes les tâches prédéfinies")
            print("4. Lister les outils disponibles")
            print("0. Quitter")
            print("-"*60)
            
            choice = input("\nVotre choix: ").strip()
            
            if choice == "0":
                print("\n👋 Au revoir!")
                break
            
            elif choice == "1":
                print("\nTâches prédéfinies:")
                for i, task in enumerate(test_tasks, 1):
                    print(f"{i}. {task}")
                
                idx = input("\nNuméro de la tâche: ").strip()
                try:
                    task = test_tasks[int(idx) - 1]
                    result = agent.run(task)
                    print(f"\n✨ Résultat Final:\n{result}\n")
                except (ValueError, IndexError):
                    print("❌ Choix invalide")
            
            elif choice == "2":
                task = input("\nVotre tâche: ").strip()
                if task:
                    result = agent.run(task)
                    print(f"\n✨ Résultat Final:\n{result}\n")
            
            elif choice == "3":
                for task in test_tasks:
                    print(f"\n{'='*60}")
                    print(f"Test: {task}")
                    print(f"{'='*60}")
                    result = agent.run(task)
                    print(f"\n✨ Résultat Final:\n{result}\n")
                    input("Appuyez sur Entrée pour continuer...")
            
            elif choice == "4":
                print("\n📦 Outils disponibles:")
                for tool_name in agent.list_tools():
                    tool = agent.tools[tool_name]
                    print(f"\n• {tool_name}")
                    print(f"  {tool.description}")
            
            else:
                print("\n❌ Choix invalide")
    
    except Exception as e:
        logger.error(f"Erreur: {e}")
        print(f"\n❌ Erreur: {e}")


if __name__ == "__main__":
    main()
