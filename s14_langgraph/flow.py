"""
S14 — LangGraph Flow Implementation
Pipeline: Retrieve → Summarize → Decide → Action/Escalate
"""

import os
from typing import TypedDict, List, Literal, Annotated
from dotenv import load_dotenv
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    # Fallback pour anciennes versions de LangChain
    from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from langgraph.graph import StateGraph, END
import random
import logging

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
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))


# ============================================================
# DÉFINITION DE L'ÉTAT
# ============================================================

class GraphState(TypedDict):
    """État partagé entre tous les nœuds du graphe"""
    query: str                    # Requête utilisateur
    documents: List[str]          # Documents récupérés
    summary: str                  # Résumé généré
    confidence: float             # Score de confiance (0-1)
    action: str                   # Action finale
    metadata: dict                # Métadonnées additionnelles


# ============================================================
# SIMULATION D'UNE BASE VECTORIELLE (pour démo)
# ============================================================

KNOWLEDGE_BASE = [
    {
        "id": 1,
        "text": "Pour réinitialiser votre mot de passe, allez dans Paramètres > Sécurité > Réinitialiser le mot de passe.",
        "category": "compte"
    },
    {
        "id": 2,
        "text": "Notre service client est disponible du lundi au vendredi de 9h à 18h. Vous pouvez nous contacter par email ou téléphone.",
        "category": "support"
    },
    {
        "id": 3,
        "text": "La livraison standard prend 3-5 jours ouvrables. La livraison express est disponible pour 10€ supplémentaires.",
        "category": "livraison"
    },
    {
        "id": 4,
        "text": "Pour annuler votre abonnement, rendez-vous dans Mon Compte > Abonnement > Annuler. Le remboursement est traité sous 7 jours.",
        "category": "abonnement"
    },
    {
        "id": 5,
        "text": "Nos produits sont garantis 2 ans. En cas de défaut, contactez notre service après-vente avec votre numéro de commande.",
        "category": "garantie"
    },
]


def simulate_vector_search(query: str, top_k: int = 3) -> List[dict]:
    """
    Simule une recherche vectorielle (en production, utiliser FAISS/Milvus)
    """
    # Simple keyword matching pour la démo
    query_lower = query.lower()
    scored_docs = []
    
    for doc in KNOWLEDGE_BASE:
        score = 0
        text_lower = doc["text"].lower()
        
        # Score basique basé sur les mots communs
        query_words = set(query_lower.split())
        text_words = set(text_lower.split())
        common_words = query_words.intersection(text_words)
        score = len(common_words)
        
        if score > 0:
            scored_docs.append((score, doc))
    
    # Trier par score et prendre top_k
    scored_docs.sort(reverse=True, key=lambda x: x[0])
    results = [doc for score, doc in scored_docs[:top_k]]
    
    # Si aucun résultat, retourner quelques docs aléatoires
    if not results:
        results = random.sample(KNOWLEDGE_BASE, min(top_k, len(KNOWLEDGE_BASE)))
    
    return results


# ============================================================
# NŒUDS DU GRAPHE
# ============================================================

def retrieve_node(state: GraphState) -> GraphState:
    """
    Nœud 1: Récupérer les documents pertinents
    """
    query = state["query"]
    logger.info(f"🔍 RETRIEVE: Recherche pour '{query}'")
    
    # Recherche vectorielle simulée
    docs = simulate_vector_search(query, top_k=3)
    doc_texts = [doc["text"] for doc in docs]
    
    logger.info(f"   → {len(doc_texts)} documents récupérés")
    
    return {
        **state,
        "documents": doc_texts,
        "metadata": {
            **state.get("metadata", {}),
            "num_docs_retrieved": len(doc_texts)
        }
    }


def summarize_node(state: GraphState) -> GraphState:
    """
    Nœud 2: Résumer et générer une réponse
    """
    query = state["query"]
    documents = state["documents"]
    
    logger.info(f"📝 SUMMARIZE: Génération de réponse")
    
    # Créer le prompt
    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(documents)])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tu es un assistant qui répond aux questions basées sur des documents fournis. Sois concis et précis."),
        ("human", f"""Question: {query}

Contexte:
{context}

Réponds à la question en te basant sur le contexte. Si la réponse n'est pas dans le contexte, dis-le clairement.""")
    ])
    
    # Appeler le LLM
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE)
    messages = prompt.format_messages()
    response = llm(messages)
    summary = response.content
    
    # Calculer un score de confiance (simplifié)
    confidence = calculate_confidence(summary, documents)
    
    logger.info(f"   → Résumé généré (confiance: {confidence:.2f})")
    
    return {
        **state,
        "summary": summary,
        "confidence": confidence,
        "metadata": {
            **state.get("metadata", {}),
            "summary_length": len(summary)
        }
    }


def calculate_confidence(summary: str, documents: List[str]) -> float:
    """
    Calculer un score de confiance basé sur la réponse
    """
    # Heuristiques simples pour la démo
    confidence = 0.5
    
    # Augmenter si la réponse contient des informations des documents
    for doc in documents:
        doc_words = set(doc.lower().split())
        summary_words = set(summary.lower().split())
        overlap = len(doc_words.intersection(summary_words))
        if overlap > 5:
            confidence += 0.2
    
    # Diminuer si la réponse contient des phrases d'incertitude
    uncertainty_phrases = ["je ne sais pas", "pas dans le contexte", "je ne peux pas"]
    for phrase in uncertainty_phrases:
        if phrase in summary.lower():
            confidence -= 0.3
    
    # Borner entre 0 et 1
    confidence = max(0.0, min(1.0, confidence))
    
    return confidence


def decide_node(state: GraphState) -> Literal["action", "escalate"]:
    """
    Nœud 3: Décider de la prochaine étape basée sur la confiance
    """
    confidence = state["confidence"]
    
    logger.info(f"🤔 DECIDE: Confiance = {confidence:.2f}, Seuil = {CONFIDENCE_THRESHOLD}")
    
    if confidence >= CONFIDENCE_THRESHOLD:
        logger.info(f"   → Route vers ACTION")
        return "action"
    else:
        logger.info(f"   → Route vers ESCALATE")
        return "escalate"


def action_node(state: GraphState) -> GraphState:
    """
    Nœud 4: Exécuter l'action appropriée
    """
    logger.info(f"✅ ACTION: Fournir la réponse")
    
    action = f"Réponse fournie: {state['summary']}"
    
    return {
        **state,
        "action": action,
        "metadata": {
            **state.get("metadata", {}),
            "final_node": "action"
        }
    }


def escalate_node(state: GraphState) -> GraphState:
    """
    Nœud 5: Escalader vers un humain
    """
    logger.info(f"⚠️  ESCALATE: Confiance faible, escalade nécessaire")
    
    action = (
        f"Escalade vers support humain requise.\n"
        f"Réponse tentative: {state['summary']}\n"
        f"Confiance: {state['confidence']:.2f}"
    )
    
    return {
        **state,
        "action": action,
        "metadata": {
            **state.get("metadata", {}),
            "final_node": "escalate"
        }
    }


# ============================================================
# CONSTRUCTION DU GRAPHE
# ============================================================

class RAGWorkflow:
    """Classe principale pour le workflow RAG avec LangGraph"""
    
    def __init__(self):
        """Initialiser le workflow"""
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY non trouvée dans .env")
        
        # Créer le graphe
        self.graph = self._build_graph()
        logger.info("✅ Workflow RAG initialisé")
    
    def _build_graph(self) -> StateGraph:
        """Construire le graphe d'états"""
        
        # Créer le graphe
        workflow = StateGraph(GraphState)
        
        # Ajouter les nœuds
        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("summarize", summarize_node)
        workflow.add_node("action", action_node)
        workflow.add_node("escalate", escalate_node)
        
        # Définir le point d'entrée
        workflow.set_entry_point("retrieve")
        
        # Ajouter les edges
        workflow.add_edge("retrieve", "summarize")
        
        # Ajouter conditional edge (branchement)
        workflow.add_conditional_edges(
            "summarize",
            decide_node,
            {
                "action": "action",
                "escalate": "escalate"
            }
        )
        
        # Terminer les deux branches
        workflow.add_edge("action", END)
        workflow.add_edge("escalate", END)
        
        # Compiler le graphe
        return workflow.compile()
    
    def run(self, query: str) -> GraphState:
        """
        Exécuter le workflow avec une requête
        
        Args:
            query: Question de l'utilisateur
            
        Returns:
            État final avec la réponse
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 DÉBUT DU WORKFLOW")
        logger.info(f"{'='*60}")
        logger.info(f"Query: {query}\n")
        
        # État initial
        initial_state: GraphState = {
            "query": query,
            "documents": [],
            "summary": "",
            "confidence": 0.0,
            "action": "",
            "metadata": {}
        }
        
        # Exécuter le graphe
        final_state = self.graph.invoke(initial_state)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✨ FIN DU WORKFLOW")
        logger.info(f"{'='*60}\n")
        
        return final_state
    
    def visualize(self, output_path: str = "workflow_graph.png"):
        """
        Visualiser le graphe (nécessite graphviz)
        
        Args:
            output_path: Chemin de sortie pour l'image
        """
        try:
            from langchain.graphs import Graph
            # Note: La visualisation nécessite des dépendances supplémentaires
            logger.info(f"💡 Pour visualiser le graphe, installez: pip install pygraphviz")
        except ImportError:
            logger.warning("Visualisation non disponible (installer pygraphviz)")


# ============================================================
# FONCTION PRINCIPALE
# ============================================================

def main():
    """Fonction principale pour tester le workflow"""
    
    print("\n" + "="*60)
    print("🔀 LANGGRAPH WORKFLOW DEMO")
    print("="*60)
    
    try:
        # Créer le workflow
        workflow = RAGWorkflow()
        
        # Exemples de requêtes à tester
        test_queries = [
            "Comment réinitialiser mon mot de passe?",
            "Quels sont les délais de livraison?",
            "Quelle est la couleur du ciel?",  # Question sans réponse dans la KB
        ]
        
        # Menu interactif
        while True:
            print("\n" + "-"*60)
            print("Options:")
            print("1. Tester avec une requête prédéfinie")
            print("2. Entrer une requête personnalisée")
            print("3. Tester toutes les requêtes prédéfinies")
            print("0. Quitter")
            print("-"*60)
            
            choice = input("\nVotre choix: ").strip()
            
            if choice == "0":
                print("\n👋 Au revoir!")
                break
            
            elif choice == "1":
                print("\nRequêtes prédéfinies:")
                for i, q in enumerate(test_queries, 1):
                    print(f"{i}. {q}")
                
                idx = input("\nNuméro de la requête: ").strip()
                try:
                    query = test_queries[int(idx) - 1]
                    result = workflow.run(query)
                    print_result(result)
                except (ValueError, IndexError):
                    print("❌ Choix invalide")
            
            elif choice == "2":
                query = input("\nVotre requête: ").strip()
                if query:
                    result = workflow.run(query)
                    print_result(result)
            
            elif choice == "3":
                for query in test_queries:
                    print(f"\n{'='*60}")
                    print(f"Test: {query}")
                    print(f"{'='*60}")
                    result = workflow.run(query)
                    print_result(result)
                    input("\nAppuyez sur Entrée pour continuer...")
            
            else:
                print("\n❌ Choix invalide")
    
    except Exception as e:
        logger.error(f"Erreur: {e}")
        print(f"\n❌ Erreur: {e}")


def print_result(result: GraphState):
    """Afficher le résultat formaté"""
    print("\n" + "="*60)
    print("📊 RÉSULTAT")
    print("="*60)
    print(f"\n🔍 Documents récupérés: {len(result['documents'])}")
    print(f"\n📝 Résumé:\n{result['summary']}")
    print(f"\n📊 Confiance: {result['confidence']:.2f}")
    print(f"\n✅ Action:\n{result['action']}")
    print(f"\n📈 Métadonnées: {result['metadata']}")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
