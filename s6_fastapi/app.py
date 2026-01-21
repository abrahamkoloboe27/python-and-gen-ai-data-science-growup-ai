"""
Séance 6 — FastAPI : Servir des modèles ML via API REST

Ce fichier contient une application FastAPI simple pour servir un modèle de classification de texte.
L'API expose des endpoints pour la prédiction, la santé du service, et la documentation.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import List
from datetime import datetime
import uvicorn

# Configuration de l'application
app = FastAPI(
    title="API de Classification de Texte",
    description="API pour classifier des documents textuels (FAQ vs Blog)",
    version="1.0.0"
)

# Modèle Pydantic pour la validation des requêtes
class TextInput(BaseModel):
    """Modèle pour l'input de prédiction"""
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "text": "Comment réinitialiser mon mot de passe ?"
            }]
        }
    )
    
    text: str = Field(..., min_length=1, max_length=10000, description="Texte à classifier")


class BatchTextInput(BaseModel):
    """Modèle pour les prédictions en batch"""
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "texts": [
                    "Comment réinitialiser mon mot de passe ?",
                    "Dans cet article, nous explorons les tendances du machine learning."
                ]
            }]
        }
    )
    
    texts: List[str] = Field(..., min_items=1, max_items=100, description="Liste de textes à classifier")


class PredictionResponse(BaseModel):
    """Modèle pour la réponse de prédiction"""
    text: str
    label: str
    probability: float
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Modèle pour la réponse de prédictions en batch"""
    predictions: List[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    """Modèle pour la réponse de healthcheck"""
    status: str
    timestamp: str
    model_loaded: bool
    version: str


# Simulateur de modèle simple (à remplacer par un vrai modèle)
class SimpleTextClassifier:
    """
    Simulateur de modèle de classification.
    Dans un cas réel, vous chargeriez un modèle entraîné (pickle, joblib, etc.)
    """
    
    def __init__(self):
        self.loaded = True
        self.classes = ["FAQ", "Blog"]
    
    def predict(self, text: str) -> tuple:
        """
        Prédiction simple basée sur la longueur du texte.
        Dans un cas réel, utilisez votre modèle entraîné.
        """
        # Règle simple: textes courts = FAQ, textes longs = Blog
        text_length = len(text)
        word_count = len(text.split())
        
        # Logique de classification simplifiée
        if text_length < 100 or word_count < 15:
            label = "FAQ"
            prob = min(0.95, 0.7 + (100 - text_length) / 200)
        else:
            label = "Blog"
            prob = min(0.95, 0.6 + (text_length - 100) / 1000)
        
        # Assurer que la probabilité est entre 0.5 et 1.0
        prob = max(0.5, min(1.0, prob))
        
        return label, round(prob, 4)


# Initialisation du modèle
model = SimpleTextClassifier()


@app.get("/", tags=["Root"])
async def root():
    """Point d'entrée racine de l'API"""
    return {
        "message": "API de Classification de Texte",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Endpoint de santé pour vérifier que l'API fonctionne correctement.
    Utile pour les orchestrateurs (Kubernetes, Docker Swarm, etc.)
    """
    return HealthResponse(
        status="healthy" if model.loaded else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model.loaded,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(input_data: TextInput):
    """
    Endpoint de prédiction pour un seul texte.
    
    Args:
        input_data: TextInput contenant le texte à classifier
        
    Returns:
        PredictionResponse avec le label et la probabilité
        
    Raises:
        HTTPException: Si le modèle n'est pas chargé ou si une erreur survient
    """
    try:
        if not model.loaded:
            raise HTTPException(status_code=503, detail="Modèle non disponible")
        
        # Prédiction
        label, probability = model.predict(input_data.text)
        
        return PredictionResponse(
            text=input_data.text[:100] + "..." if len(input_data.text) > 100 else input_data.text,
            label=label,
            probability=probability,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(input_data: BatchTextInput):
    """
    Endpoint de prédiction pour plusieurs textes en batch.
    
    Args:
        input_data: BatchTextInput contenant une liste de textes
        
    Returns:
        BatchPredictionResponse avec les prédictions pour tous les textes
        
    Raises:
        HTTPException: Si le modèle n'est pas chargé ou si une erreur survient
    """
    try:
        if not model.loaded:
            raise HTTPException(status_code=503, detail="Modèle non disponible")
        
        predictions = []
        timestamp = datetime.now().isoformat()
        
        for text in input_data.texts:
            label, probability = model.predict(text)
            predictions.append(
                PredictionResponse(
                    text=text[:100] + "..." if len(text) > 100 else text,
                    label=label,
                    probability=probability,
                    timestamp=timestamp
                )
            )
        
        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction batch: {str(e)}")


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Informations sur le modèle chargé"""
    return {
        "model_type": "SimpleTextClassifier",
        "classes": model.classes,
        "loaded": model.loaded,
        "description": "Classificateur simple basé sur la longueur du texte (démo)"
    }


# Point d'entrée pour exécuter l'application
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Rechargement automatique en développement
        log_level="info"
    )
