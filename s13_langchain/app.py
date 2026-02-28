"""
S13 — LangChain Application
Application démontrant les patterns LangChain essentiels:
- Chains (LLM, Sequential)
- Memory (Conversation)
- Q&A avec contexte
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_classic.chains import LLMChain, SequentialChain, SimpleSequentialChain, ConversationChain
from langchain_classic.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, SystemMessage
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
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "500"))


class LangChainApp:
    """Application principale démontrant les patterns LangChain"""
    
    def __init__(self):
        """Initialiser l'application"""
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY non trouvée dans .env")
        
        # Initialiser le modèle
        self.llm = ChatOpenAI(
            model_name=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        logger.info(f"✅ LangChain app initialisée avec {MODEL_NAME}")
    
    # ============================================================
    # PATTERN 1: Simple Chain
    # ============================================================
    
    def simple_chain_demo(self, topic: str) -> str:
        """
        Démonstration d'une chain simple avec prompt template
        
        Args:
            topic: Sujet à expliquer
            
        Returns:
            Explication générée
        """
        logger.info(f"Simple chain: topic='{topic}'")
        
        # Créer le prompt template
        template = """Tu es un professeur expert. Explique le concept suivant de manière claire et concise.

Concept: {topic}

Explication:"""
        
        prompt = PromptTemplate(
            input_variables=["topic"],
            template=template
        )
        
        # Créer la chain
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # Exécuter
        response = chain.run(topic=topic)
        logger.info(f"Réponse reçue: {len(response)} caractères")
        
        return response
    
    # ============================================================
    # PATTERN 2: Sequential Chain
    # ============================================================
    
    def sequential_chain_demo(self, text: str) -> dict:
        """
        Démonstration d'une sequential chain (résumé → analyse)
        
        Args:
            text: Texte à analyser
            
        Returns:
            Dict avec résumé et analyse
        """
        logger.info(f"Sequential chain: text length={len(text)}")
        
        # Chain 1: Résumé
        summary_template = """Résume le texte suivant en 2-3 phrases:

Texte: {text}

Résumé:"""
        
        summary_prompt = PromptTemplate(
            input_variables=["text"],
            template=summary_template
        )
        
        summary_chain = LLMChain(
            llm=self.llm,
            prompt=summary_prompt,
            output_key="summary"
        )
        
        # Chain 2: Analyse de sentiment
        analysis_template = """Analyse le sentiment du résumé suivant et donne un score de 1 à 5.

Résumé: {summary}

Analyse (sentiment et score):"""
        
        analysis_prompt = PromptTemplate(
            input_variables=["summary"],
            template=analysis_template
        )
        
        analysis_chain = LLMChain(
            llm=self.llm,
            prompt=analysis_prompt,
            output_key="analysis"
        )
        
        # Combiner les chains
        overall_chain = SequentialChain(
            chains=[summary_chain, analysis_chain],
            input_variables=["text"],
            output_variables=["summary", "analysis"],
            verbose=True
        )
        
        result = overall_chain({"text": text})
        logger.info("Sequential chain terminée")
        
        return result
    
    # ============================================================
    # PATTERN 3: Conversation avec Mémoire
    # ============================================================
    
    def conversation_demo(self, memory_type: str = "buffer_window"):
        """
        Démonstration d'une conversation avec mémoire
        
        Args:
            memory_type: Type de mémoire ('buffer', 'buffer_window')
        """
        logger.info(f"Conversation démarrée (memory_type={memory_type})")
        
        # Choisir le type de mémoire
        if memory_type == "buffer":
            memory = ConversationBufferMemory()
        elif memory_type == "buffer_window":
            memory = ConversationBufferWindowMemory(k=5)  # Garde 5 derniers messages
        else:
            raise ValueError(f"Type de mémoire non supporté: {memory_type}")
        
        # Créer la conversation chain
        conversation = ConversationChain(
            llm=self.llm,
            memory=memory,
            verbose=True
        )
        
        print("\n" + "="*60)
        print("💬 CONVERSATION INTERACTIVE")
        print("="*60)
        print("Tapez vos messages (ou 'quit' pour quitter)")
        print("La mémoire conserve le contexte de la conversation\n")
        
        while True:
            user_input = input("Vous: ")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                logger.info("Conversation terminée")
                break
            
            try:
                response = conversation.predict(input=user_input)
                print(f"\nAssistant: {response}\n")
                logger.info(f"Tour de conversation: input={len(user_input)} chars, output={len(response)} chars")
            except Exception as e:
                logger.error(f"Erreur: {e}")
                print(f"\n❌ Erreur: {e}\n")
    
    # ============================================================
    # PATTERN 4: Q&A avec Contexte
    # ============================================================
    
    def qa_demo(self, context: str, question: str) -> str:
        """
        Démonstration de Q&A avec contexte fourni
        
        Args:
            context: Contexte/document pour répondre
            question: Question à répondre
            
        Returns:
            Réponse basée sur le contexte
        """
        logger.info(f"Q&A: context={len(context)} chars, question='{question}'")
        
        # Créer le prompt template
        template = """Réponds à la question en te basant UNIQUEMENT sur le contexte fourni.
Si la réponse n'est pas dans le contexte, dis "Je ne peux pas répondre d'après le contexte fourni."

Contexte: {context}

Question: {question}

Réponse:"""
        
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
        
        # Créer la chain
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # Exécuter
        response = chain.run(context=context, question=question)
        logger.info(f"Réponse Q&A générée")
        
        return response
    
    # ============================================================
    # PATTERN 5: Few-Shot Learning
    # ============================================================
    
    def few_shot_demo(self, user_input: str) -> str:
        """
        Démonstration de few-shot learning avec exemples
        
        Args:
            user_input: Entrée utilisateur à classifier
            
        Returns:
            Classification
        """
        logger.info(f"Few-shot classification: input='{user_input}'")
        
        template = """Classifie les phrases suivantes par sentiment (positif, négatif, neutre).

Exemples:
Phrase: "J'adore ce produit!"
Sentiment: positif

Phrase: "C'est décevant."
Sentiment: négatif

Phrase: "Le produit est arrivé."
Sentiment: neutre

Maintenant, classifie cette phrase:
Phrase: {input}
Sentiment:"""
        
        prompt = PromptTemplate(
            input_variables=["input"],
            template=template
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(input=user_input)
        
        logger.info(f"Classification: {response.strip()}")
        
        return response.strip()


def main():
    """Fonction principale pour tester l'application"""
    
    print("\n" + "="*60)
    print("🦜⛓️  LANGCHAIN PATTERNS DEMO")
    print("="*60)
    
    try:
        app = LangChainApp()
        
        # Menu interactif
        while True:
            print("\n" + "-"*60)
            print("Choisissez une démo:")
            print("1. Simple Chain (explication de concept)")
            print("2. Sequential Chain (résumé + analyse)")
            print("3. Conversation avec mémoire")
            print("4. Q&A avec contexte")
            print("5. Few-shot classification")
            print("0. Quitter")
            print("-"*60)
            
            choice = input("\nVotre choix: ").strip()
            
            if choice == "0":
                print("\n👋 Au revoir!")
                break
            
            elif choice == "1":
                topic = input("\nSujet à expliquer: ")
                response = app.simple_chain_demo(topic)
                print(f"\n📝 Explication:\n{response}\n")
            
            elif choice == "2":
                text = input("\nTexte à analyser: ")
                result = app.sequential_chain_demo(text)
                print(f"\n📊 Résumé:\n{result['summary']}\n")
                print(f"📊 Analyse:\n{result['analysis']}\n")
            
            elif choice == "3":
                memory_type = input("\nType de mémoire (buffer/buffer_window): ").strip()
                if memory_type not in ["buffer", "buffer_window"]:
                    memory_type = "buffer_window"
                app.conversation_demo(memory_type)
            
            elif choice == "4":
                print("\nExemple de contexte:")
                context = """Le RAG (Retrieval-Augmented Generation) est une technique qui combine 
la recherche d'information (retrieval) avec la génération de texte. 
Il permet aux LLMs d'accéder à des connaissances externes via un système de recherche vectorielle."""
                
                question = input("\nVotre question: ")
                response = app.qa_demo(context, question)
                print(f"\n💡 Réponse:\n{response}\n")
            
            elif choice == "5":
                phrase = input("\nPhrase à classifier: ")
                sentiment = app.few_shot_demo(phrase)
                print(f"\n🎯 Sentiment: {sentiment}\n")
            
            else:
                print("\n❌ Choix invalide\n")
    
    except Exception as e:
        logger.error(f"Erreur: {e}")
        print(f"\n❌ Erreur: {e}")


if __name__ == "__main__":
    main()
