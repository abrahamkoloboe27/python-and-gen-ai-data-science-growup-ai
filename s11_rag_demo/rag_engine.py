"""
RAG Engine - Core logic for Retrieval Augmented Generation
"""
import os
import time
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import pickle


class RAGEngine:
    """
    Moteur RAG combinant recherche vectorielle et génération LLM
    """
    
    def __init__(
        self,
        index_path: str = "index/faiss_index.bin",
        data_path: str = "index/documents.pkl",
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        llm_model: str = "gpt-3.5-turbo",
        openai_api_key: Optional[str] = None
    ):
        """
        Initialiser le moteur RAG
        
        Args:
            index_path: Chemin vers l'index FAISS
            data_path: Chemin vers les documents
            embedding_model: Modèle pour les embeddings
            llm_model: Modèle LLM pour la génération
            openai_api_key: Clé API OpenAI
        """
        # Charger le modèle d'embeddings
        print(f"🔄 Chargement du modèle d'embeddings: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Charger l'index FAISS et les documents
        print(f"🔄 Chargement de l'index FAISS: {index_path}")
        self.index = faiss.read_index(index_path)
        
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        self.documents = data['documents']
        
        # Initialiser le client OpenAI
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=api_key)
            self.llm_model = llm_model
        else:
            self.client = None
            print("⚠️ OpenAI API key not provided - generation disabled")
        
        print(f"✅ RAG Engine initialized with {self.index.ntotal} documents")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Récupérer les documents les plus pertinents
        
        Args:
            query: Question de l'utilisateur
            top_k: Nombre de documents à retourner
            
        Returns:
            Liste de documents avec scores
        """
        start_time = time.time()
        
        # Créer l'embedding de la query
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        
        # Rechercher dans l'index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Construire les résultats
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            doc = self.documents.iloc[idx]
            results.append({
                "rank": i + 1,
                "doc_id": doc.get('id', f"doc_{idx}"),
                "title": doc.get('title', 'Untitled'),
                "content": doc.get('content', doc.get('text', '')),
                "chunk": doc.get('chunk', doc.get('text', '')),
                "metadata": doc.get('metadata', {}),
                "distance": float(dist),
                "score": float(1 / (1 + dist))  # Approximation de similarité
            })
        
        retrieval_time = (time.time() - start_time) * 1000  # en ms
        
        return results, retrieval_time
    
    def generate(
        self,
        question: str,
        context_docs: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """
        Générer une réponse avec le LLM basée sur les documents récupérés
        
        Args:
            question: Question de l'utilisateur
            context_docs: Documents récupérés
            temperature: Température pour la génération
            max_tokens: Nombre max de tokens à générer
            
        Returns:
            Réponse générée avec métadonnées
        """
        if not self.client:
            return {
                "answer": "LLM generation not available (no API key)",
                "generation_time_ms": 0,
                "tokens": 0
            }
        
        start_time = time.time()
        
        # Construire le contexte à partir des documents
        context = self._build_context(context_docs)
        
        # Construire le prompt
        prompt = self._build_prompt(question, context)
        
        # Appeler le LLM
        messages = [
            {"role": "system", "content": "Tu es un assistant utile qui répond aux questions basées sur le contexte fourni. Si le contexte ne contient pas l'information, dis-le clairement."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        answer = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        
        generation_time = (time.time() - start_time) * 1000  # en ms
        
        return {
            "answer": answer,
            "generation_time_ms": generation_time,
            "tokens": tokens_used,
            "model": self.llm_model
        }
    
    def query(
        self,
        question: str,
        top_k: int = 3,
        include_sources: bool = True,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Pipeline RAG complet: retrieve + generate
        
        Args:
            question: Question de l'utilisateur
            top_k: Nombre de documents à récupérer
            include_sources: Inclure les sources dans la réponse
            temperature: Température pour la génération
            
        Returns:
            Réponse complète avec sources et métadonnées
        """
        # Étape 1: Retrieval
        retrieved_docs, retrieval_time = self.retrieve(question, top_k)
        
        # Étape 2: Generation
        generation_result = self.generate(question, retrieved_docs, temperature)
        
        # Construire la réponse finale
        result = {
            "question": question,
            "answer": generation_result["answer"],
            "metadata": {
                "model": generation_result.get("model", "N/A"),
                "retrieval_time_ms": retrieval_time,
                "generation_time_ms": generation_result["generation_time_ms"],
                "total_time_ms": retrieval_time + generation_result["generation_time_ms"],
                "tokens_used": generation_result["tokens"],
                "top_k": top_k
            }
        }
        
        if include_sources:
            result["sources"] = [
                {
                    "rank": doc["rank"],
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "chunk": doc["chunk"][:200] + "..." if len(doc["chunk"]) > 200 else doc["chunk"],
                    "score": round(doc["score"], 3)
                }
                for doc in retrieved_docs
            ]
        
        return result
    
    def _build_context(self, docs: List[Dict[str, Any]], max_length: int = 2000) -> str:
        """
        Construire le contexte à partir des documents récupérés
        
        Args:
            docs: Documents récupérés
            max_length: Longueur max du contexte (en caractères)
            
        Returns:
            Contexte formaté
        """
        context_parts = []
        current_length = 0
        
        for doc in docs:
            chunk = doc.get('chunk', doc.get('content', ''))
            title = doc.get('title', 'Document')
            
            doc_text = f"[{title}]\n{chunk}\n"
            doc_length = len(doc_text)
            
            if current_length + doc_length > max_length:
                # Tronquer si nécessaire
                remaining = max_length - current_length
                if remaining > 100:  # Au moins 100 caractères
                    doc_text = doc_text[:remaining] + "...\n"
                    context_parts.append(doc_text)
                break
            
            context_parts.append(doc_text)
            current_length += doc_length
        
        return "\n".join(context_parts)
    
    def _build_prompt(self, question: str, context: str) -> str:
        """
        Construire le prompt pour le LLM
        
        Args:
            question: Question de l'utilisateur
            context: Contexte des documents
            
        Returns:
            Prompt formaté
        """
        prompt = f"""Contexte:
{context}

Question: {question}

Instructions:
- Réponds à la question en te basant UNIQUEMENT sur le contexte fourni ci-dessus.
- Si le contexte ne contient pas assez d'information pour répondre, dis-le clairement.
- Sois précis et concis.
- Cite les sources si pertinent (ex: "Selon [Titre du document]...")

Réponse:"""
        
        return prompt
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtenir les statistiques de l'index
        
        Returns:
            Statistiques
        """
        return {
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal,
            "embedding_dimension": self.index.d,
            "embedding_model": str(self.embedding_model),
            "llm_model": self.llm_model if self.client else "N/A"
        }
