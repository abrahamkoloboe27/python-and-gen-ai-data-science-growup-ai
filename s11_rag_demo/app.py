"""
FastAPI Application for RAG Demo
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv

from rag_engine import RAGEngine

# Charger les variables d'environnement
load_dotenv()

# Initialiser FastAPI
app = FastAPI(
    title="RAG Demo API",
    description="Retrieval Augmented Generation endpoint",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialiser le moteur RAG
rag_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialiser le RAG engine au démarrage"""
    global rag_engine
    
    index_path = os.getenv("INDEX_PATH", "index/faiss_index.bin")
    data_path = os.getenv("DATA_PATH", "index/documents.pkl")
    embedding_model = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
    llm_model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    
    try:
        rag_engine = RAGEngine(
            index_path=index_path,
            data_path=data_path,
            embedding_model=embedding_model,
            llm_model=llm_model
        )
        print("✅ RAG Engine initialized successfully")
    except Exception as e:
        print(f"⚠️ Failed to initialize RAG Engine: {e}")
        print("API will run in limited mode (no RAG functionality)")


# Modèles Pydantic
class QueryRequest(BaseModel):
    """Requête pour une question RAG"""
    question: str = Field(..., description="Question de l'utilisateur")
    top_k: int = Field(3, ge=1, le=10, description="Nombre de documents à récupérer")
    include_sources: bool = Field(True, description="Inclure les sources dans la réponse")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Température pour la génération")


class RetrieveRequest(BaseModel):
    """Requête pour retrieval uniquement"""
    query: str = Field(..., description="Query de recherche")
    top_k: int = Field(5, ge=1, le=20, description="Nombre de documents à récupérer")


class QueryResponse(BaseModel):
    """Réponse à une question RAG"""
    question: str
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any]


# Endpoints
@app.get("/")
async def root():
    """Page d'accueil"""
    return {
        "message": "RAG Demo API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "query": "/query (POST)",
            "retrieve": "/retrieve (POST)",
            "stats": "/index/stats"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if rag_engine is None:
        return {
            "status": "degraded",
            "message": "RAG engine not initialized"
        }
    
    return {
        "status": "healthy",
        "message": "RAG engine is running"
    }


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Endpoint principal RAG: retrieve + generate
    
    Args:
        request: Requête avec question et paramètres
        
    Returns:
        Réponse générée avec sources
    """
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    try:
        result = rag_engine.query(
            question=request.question,
            top_k=request.top_k,
            include_sources=request.include_sources,
            temperature=request.temperature
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/retrieve")
async def retrieve_documents(request: RetrieveRequest):
    """
    Endpoint de retrieval uniquement (sans génération)
    
    Args:
        request: Requête de recherche
        
    Returns:
        Documents récupérés avec scores
    """
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    try:
        results, retrieval_time = rag_engine.retrieve(
            query=request.query,
            top_k=request.top_k
        )
        
        return {
            "query": request.query,
            "results": results,
            "metadata": {
                "retrieval_time_ms": retrieval_time,
                "top_k": request.top_k
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")


@app.get("/index/stats")
async def get_index_stats():
    """
    Obtenir les statistiques de l'index
    
    Returns:
        Statistiques de l'index
    """
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    try:
        stats = rag_engine.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


# Pour lancer directement avec python app.py
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
