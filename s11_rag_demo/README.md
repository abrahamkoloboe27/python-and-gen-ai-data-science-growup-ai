# S11 — RAG Demo: Pipeline End-to-End

## 🎯 Objectifs
- Concevoir un pipeline RAG complet (ingestion → embed → index → retrieve → generate)
- Implémenter un endpoint FastAPI qui combine retrieval et génération
- Comprendre le chunking, les métadatas, et la gestion du context window
- Déployer un RAG local fonctionnel

## 📋 Architecture

```
┌─────────────┐
│  Documents  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Chunking   │  (découpage intelligent)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Embedding  │  (sentence-transformers)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ FAISS Index │  (indexation vectorielle)
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌──────────┐
│   Retrieval │────▶│   LLM    │
│   (top-k)   │     │ (GPT-3.5)│
└─────────────┘     └────┬─────┘
                         │
                         ▼
                    ┌──────────┐
                    │ Response │
                    └──────────┘
```

## 🚀 Installation

```bash
cd s11_rag_demo
pip install -r requirements.txt
```

## 📁 Structure du projet

```
s11_rag_demo/
├── README.md              # Ce fichier
├── requirements.txt       # Dépendances
├── indexer.py            # Script d'indexation
├── app.py                # API FastAPI
├── rag_engine.py         # Logique RAG
├── data/                 # Données sources
│   └── sample_docs.json
├── index/                # Index FAISS (généré)
│   ├── faiss_index.bin
│   └── documents.pkl
└── .env.example          # Configuration
```

## 🔧 Configuration

1. Créer un fichier `.env`:
```bash
OPENAI_API_KEY=votre_clé_api
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=3
```

## 📚 Étape 1: Indexation

### Préparer les données
Placez vos documents dans `data/sample_docs.json`:
```json
[
  {
    "id": "doc1",
    "title": "Introduction au ML",
    "content": "Le machine learning est...",
    "metadata": {"category": "ML", "author": "Alice"}
  }
]
```

### Lancer l'indexation
```bash
python indexer.py --input data/sample_docs.json --output index/
```

Cette étape va:
1. Charger les documents
2. Les découper en chunks
3. Générer les embeddings
4. Créer l'index FAISS
5. Sauvegarder l'index sur disque

## 🌐 Étape 2: Lancer l'API

```bash
python app.py
# ou
uvicorn app:app --reload
```

L'API sera disponible sur `http://localhost:8000`

## 📖 Utilisation de l'API

### Endpoints disponibles

#### 1. Health Check
```bash
curl http://localhost:8000/health
```

#### 2. RAG Query
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Qu'\''est-ce que le machine learning?",
    "top_k": 3,
    "include_sources": true
  }'
```

Réponse:
```json
{
  "question": "Qu'est-ce que le machine learning?",
  "answer": "Le machine learning est une branche...",
  "sources": [
    {
      "doc_id": "doc1",
      "title": "Introduction au ML",
      "chunk": "Le machine learning est...",
      "score": 0.89
    }
  ],
  "metadata": {
    "model": "gpt-3.5-turbo",
    "retrieval_time_ms": 45,
    "generation_time_ms": 1200
  }
}
```

#### 3. Retrieve Only (sans génération)
```bash
curl -X POST "http://localhost:8000/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Python pour data science",
    "top_k": 5
  }'
```

#### 4. Index Stats
```bash
curl http://localhost:8000/index/stats
```

## 🔍 Fonctionnalités avancées

### Chunking intelligent
Le script supporte plusieurs stratégies:
- **Fixed size**: Chunks de taille fixe
- **Sentence-based**: Découpage par phrases
- **Semantic**: Découpage sémantique (expérimental)

### Métadonnées enrichies
Chaque chunk contient:
- `doc_id`: ID du document source
- `chunk_id`: ID du chunk
- `title`: Titre du document
- `category`: Catégorie
- `start_pos`: Position de début dans le document

### Reranking (optionnel)
Améliore la pertinence des résultats après retrieval initial.

### Context window management
Gestion automatique pour ne pas dépasser la limite du modèle.

## 🧪 Tests

```bash
# Tester l'indexation
pytest tests/test_indexer.py

# Tester l'API
pytest tests/test_api.py

# Tester le RAG engine
pytest tests/test_rag_engine.py
```

## 📊 Monitoring et Logs

L'API logge automatiquement:
- Temps de retrieval
- Temps de génération
- Nombre de tokens utilisés
- Coût estimé

Exemple de log:
```
[2024-01-22 10:30:45] INFO: Query received: "Qu'est-ce que le ML?"
[2024-01-22 10:30:45] INFO: Retrieved 3 documents in 45ms
[2024-01-22 10:30:46] INFO: Generated response in 1.2s (250 tokens)
[2024-01-22 10:30:46] INFO: Estimated cost: $0.0015
```

## 🔒 Sécurité

- Validation des inputs utilisateur
- Rate limiting (10 requêtes/min par IP)
- Sanitization des queries
- Pas de stockage des requêtes sensibles

## 🚢 Déploiement

### Docker
```bash
docker build -t rag-demo .
docker run -p 8000:8000 --env-file .env rag-demo
```

### Docker Compose
```bash
docker-compose up -d
```

## 💡 Bonnes pratiques

1. **Chunking**: Ajuster `CHUNK_SIZE` selon votre use case
   - FAQ: 200-300 tokens
   - Articles: 500-1000 tokens
   - Documentation technique: 300-500 tokens

2. **Top-k**: Commencer avec 3-5, ajuster selon le besoin
   - Plus grand = plus de contexte mais plus de bruit
   - Plus petit = plus précis mais peut manquer d'info

3. **Prompt template**: Customiser selon votre domaine
   - Spécifier le ton et le style
   - Ajouter des instructions de format
   - Gérer les cas où rien n'est trouvé

4. **Caching**: Cache les embeddings fréquents
   - Réduire les appels API
   - Améliorer la latence

## 🐛 Dépannage

### "Index not found"
```bash
# Réindexer
python indexer.py --input data/sample_docs.json --output index/
```

### "API key invalid"
```bash
# Vérifier le .env
cat .env | grep OPENAI_API_KEY
```

### "Memory error with FAISS"
```bash
# Réduire la taille des chunks ou utiliser IVF index
python indexer.py --index-type ivf --nlist 100
```

## 📚 Ressources

- [RAG Paper (Lewis et al.)](https://arxiv.org/abs/2005.11401)
- [LangChain RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [FastAPI Best Practices](https://fastapi.tiangolo.com/tutorial/)

## 🎓 Exercices

### Exercice 1: Ajouter un document
1. Ajouter un nouveau document dans `data/`
2. Réindexer
3. Tester une query sur ce document

### Exercice 2: Customiser le prompt
1. Modifier le prompt template dans `rag_engine.py`
2. Tester différents styles de réponse
3. Comparer les résultats

### Exercice 3: Implémenter le reranking
1. Ajouter une étape de reranking après retrieval
2. Utiliser un modèle cross-encoder
3. Mesurer l'amélioration du recall

### Exercice 4: Ajouter des filtres
1. Permettre le filtrage par catégorie
2. Filtrer par date
3. Combiner plusieurs filtres

## ✅ Checklist

- [ ] Dépendances installées
- [ ] Index créé avec succès
- [ ] API lance correctement
- [ ] Health check passe
- [ ] Query RAG fonctionne
- [ ] Sources retournées correctement
- [ ] Logs fonctionnels
- [ ] Tests passent

---

**Mini-projet S11 complété! 🎉**

## 📞 Support

Pour toute question:
1. Vérifier les logs
2. Consulter ce README
3. Tester avec `curl` les endpoints
4. Vérifier le format des données d'entrée
