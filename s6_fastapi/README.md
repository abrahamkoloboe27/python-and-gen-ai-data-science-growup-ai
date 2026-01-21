# Séance 6 — FastAPI : Serveur applicatif pour modèles ML

## 🎯 Objectifs
- Exposer un modèle ML via une API REST
- Comprendre la structure d'une application FastAPI
- Implémenter des endpoints de prédiction et de healthcheck
- Valider les données d'entrée avec Pydantic
- Conteneuriser l'application avec Docker

---

## 📁 Structure du projet

```
s6_fastapi/
├── app.py              # Application FastAPI principale
├── requirements.txt    # Dépendances Python
├── Dockerfile         # Configuration Docker
└── README.md          # Ce fichier
```

---

## 🚀 Installation et démarrage

### Option 1 : Installation locale

1. **Créer un environnement virtuel** (recommandé):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

2. **Installer les dépendances**:
```bash
pip install -r requirements.txt
```

3. **Lancer l'application**:
```bash
python app.py
```

Ou avec uvicorn directement:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

4. **Accéder à l'API**:
   - API: http://localhost:8000
   - Documentation interactive (Swagger): http://localhost:8000/docs
   - Documentation alternative (ReDoc): http://localhost:8000/redoc

### Option 2 : Avec Docker

1. **Construire l'image Docker**:
```bash
docker build -t fastapi-text-classifier .
```

2. **Lancer le conteneur**:
```bash
docker run -p 8000:8000 fastapi-text-classifier
```

3. **Accéder à l'API**:
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs

---

## 📡 Endpoints disponibles

### 1. Root (`GET /`)
Point d'entrée de l'API avec les informations de base.

**Exemple avec curl**:
```bash
curl http://localhost:8000/
```

**Réponse**:
```json
{
  "message": "API de Classification de Texte",
  "version": "1.0.0",
  "endpoints": {
    "docs": "/docs",
    "health": "/health",
    "predict": "/predict",
    "predict_batch": "/predict/batch"
  }
}
```

### 2. Health Check (`GET /health`)
Vérifier l'état de santé de l'API.

**Exemple avec curl**:
```bash
curl http://localhost:8000/health
```

**Réponse**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.123456",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### 3. Predict (`POST /predict`)
Prédire la catégorie d'un seul texte.

**Exemple avec curl**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Comment réinitialiser mon mot de passe ?"}'
```

**Réponse**:
```json
{
  "text": "Comment réinitialiser mon mot de passe ?",
  "label": "FAQ",
  "probability": 0.8523,
  "timestamp": "2024-01-15T10:30:00.123456"
}
```

**Exemple avec un texte long (Blog)**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Dans cet article approfondi, nous explorons les dernières avancées en intelligence artificielle et leurs implications pour l'"'"'avenir de la technologie. Les chercheurs du monde entier travaillent sur des modèles de plus en plus sophistiqués qui transforment notre façon d'"'"'interagir avec les machines."}'
```

**Réponse**:
```json
{
  "text": "Dans cet article approfondi, nous explorons les dernières avancées en intelligence artific...",
  "label": "Blog",
  "probability": 0.9234,
  "timestamp": "2024-01-15T10:31:00.123456"
}
```

### 4. Batch Predict (`POST /predict/batch`)
Prédire la catégorie de plusieurs textes en une seule requête.

**Exemple avec curl**:
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Comment contacter le support ?",
      "Dans cet article, nous explorons les tendances du machine learning pour 2024.",
      "Quels sont vos horaires d'"'"'ouverture ?"
    ]
  }'
```

**Réponse**:
```json
{
  "predictions": [
    {
      "text": "Comment contacter le support ?",
      "label": "FAQ",
      "probability": 0.8312,
      "timestamp": "2024-01-15T10:32:00.123456"
    },
    {
      "text": "Dans cet article, nous explorons les tendances du machine learning pour 2024.",
      "label": "Blog",
      "probability": 0.8876,
      "timestamp": "2024-01-15T10:32:00.123456"
    },
    {
      "text": "Quels sont vos horaires d'ouverture ?",
      "label": "FAQ",
      "probability": 0.8654,
      "timestamp": "2024-01-15T10:32:00.123456"
    }
  ],
  "count": 3
}
```

### 5. Model Info (`GET /model/info`)
Obtenir des informations sur le modèle chargé.

**Exemple avec curl**:
```bash
curl http://localhost:8000/model/info
```

**Réponse**:
```json
{
  "model_type": "SimpleTextClassifier",
  "classes": ["FAQ", "Blog"],
  "loaded": true,
  "description": "Classificateur simple basé sur la longueur du texte (démo)"
}
```

---

## 🧪 Tests avec Python

Vous pouvez aussi tester l'API avec Python:

```python
import requests

# URL de l'API
url = "http://localhost:8000"

# Test health check
response = requests.get(f"{url}/health")
print(response.json())

# Test prédiction simple
data = {"text": "Comment puis-je annuler ma commande ?"}
response = requests.post(f"{url}/predict", json=data)
print(response.json())

# Test prédiction batch
data = {
    "texts": [
        "Quelle est votre politique de retour ?",
        "Dans ce tutoriel complet, nous allons explorer en détail..."
    ]
}
response = requests.post(f"{url}/predict/batch", json=data)
print(response.json())
```

---

## 📊 Documentation interactive

FastAPI génère automatiquement une documentation interactive:

1. **Swagger UI**: http://localhost:8000/docs
   - Interface interactive pour tester tous les endpoints
   - Voir les schémas de requêtes/réponses
   - Exécuter des requêtes directement depuis le navigateur

2. **ReDoc**: http://localhost:8000/redoc
   - Documentation alternative plus lisible
   - Parfait pour partager avec d'autres développeurs

---

## 🔧 Personnalisation

### Remplacer le modèle simulé

Pour utiliser un vrai modèle entraîné (par exemple avec scikit-learn ou PyCaret):

1. **Charger votre modèle dans `app.py`**:
```python
import joblib

class RealTextClassifier:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.loaded = True
        
    def predict(self, text):
        # Preprocessing si nécessaire
        prediction = self.model.predict([text])[0]
        probability = self.model.predict_proba([text])[0].max()
        return prediction, probability

# Initialiser avec votre modèle
model = RealTextClassifier('path/to/your/model.pkl')
```

2. **Ajouter les dépendances nécessaires** dans `requirements.txt`:
```
scikit-learn==1.4.0
joblib==1.3.2
```

### Ajouter l'authentification

Pour sécuriser l'API:

```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends

security = HTTPBearer()

@app.post("/predict")
async def predict(
    input_data: TextInput,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # Vérifier le token
    if credentials.credentials != "votre-token-secret":
        raise HTTPException(status_code=401, detail="Token invalide")
    # ... reste du code
```

---

## 🐳 Déploiement Docker

### Docker Compose (optionnel)

Créez un `docker-compose.yml`:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
```

Lancez avec:
```bash
docker-compose up -d
```

---

## 📝 Exercices

1. **Modifier le modèle**:
   - Remplacez le `SimpleTextClassifier` par un vrai modèle entraîné
   - Testez avec vos propres données

2. **Ajouter des endpoints**:
   - Endpoint `/retrain` pour réentraîner le modèle
   - Endpoint `/metrics` pour les statistiques d'utilisation
   - Endpoint `/feedback` pour collecter du feedback

3. **Améliorer la validation**:
   - Ajouter plus de validations Pydantic
   - Gérer les cas d'erreur spécifiques
   - Ajouter des limites de rate limiting

4. **Monitoring**:
   - Ajouter des logs structurés
   - Intégrer Prometheus pour les métriques
   - Configurer des alertes

5. **Tests**:
   - Écrire des tests unitaires avec pytest
   - Tester tous les endpoints
   - Tester les cas d'erreur

---

## 🔗 Ressources

- [Documentation FastAPI](https://fastapi.tiangolo.com/)
- [Documentation Pydantic](https://docs.pydantic.dev/)
- [Documentation Uvicorn](https://www.uvicorn.org/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

---

## 💡 Conseils

- Utilisez toujours la validation Pydantic pour les données d'entrée
- Gérez les erreurs avec des HTTPException appropriées
- Documentez vos endpoints avec des docstrings
- Utilisez le mode `--reload` uniquement en développement
- Mettez en place un monitoring en production
- Versionnez votre API (v1, v2, etc.)

---

## ✅ Checklist de production

Avant de déployer en production:

- [ ] Remplacer le modèle simulé par un vrai modèle
- [ ] Ajouter l'authentification/autorisation
- [ ] Configurer CORS si nécessaire
- [ ] Ajouter du rate limiting
- [ ] Mettre en place des logs
- [ ] Configurer le monitoring
- [ ] Écrire des tests
- [ ] Optimiser les performances
- [ ] Configurer HTTPS
- [ ] Documenter l'API pour les utilisateurs
