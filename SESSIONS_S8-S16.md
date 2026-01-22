# 📚 Sessions S8 à S16 — Guide Complet GenAI & LLM

Ce document fournit une vue d'ensemble des matériels de cours pour les séances 8 à 16, couvrant les fondamentaux des LLMs jusqu'à l'orchestration avancée.

## 📋 Table des matières

- [S8 — Fondamentaux LLM & Transformers](#s8--fondamentaux-llm--transformers)
- [S9 — OpenAI API & Prompt Engineering](#s9--openai-api--prompt-engineering)
- [S10 — Embeddings & Recherche Vectorielle](#s10--embeddings--recherche-vectorielle)
- [S11 — RAG: Architecture & Patterns](#s11--rag-architecture--patterns)
- [S12 — Vector DB & Production](#s12--vector-db--production)
- [S13 — LangChain: Patterns Pratiques](#s13--langchain-patterns-pratiques)
- [S14 — LangGraph & Orchestration](#s14--langgraph--orchestration)
- [S15 — Agents Avancés](#s15--agents-avancés)
- [S16 — Flow Control & State Management](#s16--flow-control--state-management)

---

## S8 — Fondamentaux LLM & Transformers

### 🎯 Objectifs
- Comprendre l'architecture Transformer
- Maîtriser les concepts d'embeddings et tokens
- Comprendre le context window et ses implications
- Apprendre les hyperparamètres clés (temperature, top-k, etc.)

### 📁 Fichiers
- **Document technique**: `s8_llm_fundamentals.md`

### 📚 Contenu
1. **Architecture Transformer**
   - Self-Attention et Multi-Head Attention
   - Encodeur vs Décodeur vs Encodeur-Décodeur
   - Positional Encoding
   - Architecture GPT

2. **Tokenisation**
   - Types de tokenisation (caractères, mots, subword)
   - BPE (Byte Pair Encoding)
   - Implications pour le coût et la performance

3. **Embeddings vs Logits**
   - Représentations vectorielles
   - Conversion en probabilités
   - Flow de génération autoregressive

4. **Context Window**
   - Limites et implications
   - Stratégies de gestion (summarization, chunking, RAG)
   - Tradeoffs coût/latence

5. **Hyperparamètres de Génération**
   - Temperature: contrôle de la créativité
   - Top-k et Top-p sampling
   - Frequency/presence penalties

6. **Coût et Latence**
   - Facteurs de coût
   - Time to First Token (TTFT)
   - Optimisations

### 🎓 Exercice
- Note technique résumant les concepts essentiels
- 3 prompts expérimentaux avec outputs documentés
- Analyse de l'impact des hyperparamètres

### 🔧 Prérequis
- Compréhension basique du ML
- Familiarité avec les réseaux de neurones

---

## S9 — OpenAI API & Prompt Engineering

### 🎯 Objectifs
- Maîtriser les appels API OpenAI (Chat, Completion, Embeddings)
- Appliquer les bonnes pratiques de prompt design
- Comprendre few-shot vs zero-shot learning
- Tester et comparer différents prompts

### 📁 Fichiers
- **Notebook**: `notebooks/s9_openai_prompts.ipynb`

### 📚 Contenu
1. **Configuration API**
   - Installation et setup
   - Gestion des clés API
   - Comptage de tokens

2. **Appels API de Base**
   - Chat Completion
   - Embeddings
   - Modération de contenu

3. **Prompt Engineering: Techniques**
   - Zero-shot prompting
   - Few-shot learning
   - Chain-of-Thought (CoT)
   - Structured outputs

4. **Expérimentation avec 10 Prompts**
   - Résumé basique
   - Résumé avec contraintes
   - Bullet points structurés
   - Audience cible
   - Q&A factuelle et analytique
   - Extraction JSON
   - Styles variés

5. **Analyse Comparative**
   - Métriques de tokens
   - Visualisations
   - Comparaison de qualité

6. **Bonnes Pratiques**
   - Instructions claires
   - Contexte et rôle
   - Format de sortie
   - Température appropriée

7. **Sécurité**
   - Validation des inputs
   - Détection d'injections
   - Modération de contenu

### 🎓 Exercice
- 10 prompts testés pour résumé/Q&A
- Comparaison et documentation des résultats
- Sauvegarde en JSON

### 🔧 Prérequis
```bash
pip install openai python-dotenv tiktoken
```

### ⚠️ Important
Créer un fichier `.env` avec `OPENAI_API_KEY=votre_clé`

---

## S10 — Embeddings & Recherche Vectorielle

### 🎯 Objectifs
- Comprendre les embeddings et leur création
- Maîtriser la recherche de similarité (nearest neighbour)
- Implémenter un index FAISS local
- Évaluer recall et precision du retrieval

### 📁 Fichiers
- **Notebook**: `notebooks/s10_embeddings.ipynb`

### 📚 Contenu
1. **Création d'Embeddings**
   - Modèles: Sentence-BERT, OpenAI
   - Génération batch
   - Dimensions et propriétés

2. **Métriques de Similarité**
   - Cosine similarity
   - Distance euclidienne
   - Matrice de similarité

3. **FAISS: Index Flat**
   - Création d'index brute force
   - Recherche exacte
   - Sauvegarder/charger

4. **FAISS: Index IVF**
   - Clustering pour acceleration
   - Tradeoff vitesse/précision
   - Benchmark de performance

5. **Évaluation**
   - Ground truth definition
   - Recall@k et Precision@k
   - Métriques moyennes
   - Visualisations

6. **Concepts Avancés**
   - Types d'index (IVFPQ, HNSW)
   - Librairies alternatives (Milvus, Weaviate, Pinecone)
   - Métriques de distance

### 🎓 Exercice
- Créer embeddings pour dataset de documents
- Implémenter index FAISS
- Tester queries et mesurer recall/precision
- Sauvegarder index localement

### 🔧 Prérequis
```bash
pip install faiss-cpu sentence-transformers pandas numpy scikit-learn
```

---

## S11 — RAG: Architecture & Patterns

### 🎯 Objectifs
- Concevoir un pipeline RAG end-to-end
- Implémenter un endpoint FastAPI combinant retrieval et génération
- Comprendre chunking, métadonnées, et context window management
- Déployer un RAG local fonctionnel

### 📁 Fichiers
```
s11_rag_demo/
├── README.md              # Documentation
├── requirements.txt       # Dépendances
├── indexer.py            # Script d'indexation
├── app.py                # API FastAPI
├── rag_engine.py         # Logique RAG
├── data/                 # Données sources
│   └── sample_docs.json
├── index/                # Index FAISS (généré)
└── .env.example          # Configuration
```

### 📚 Contenu
1. **Architecture RAG**
   - Pipeline: ingestion → embed → index → retrieve → generate
   - Composants clés
   - Flow de données

2. **Indexation**
   - Chunking intelligent (fixed size, sentence-based)
   - Métadonnées enrichies
   - Génération d'embeddings
   - Création d'index FAISS

3. **API FastAPI**
   - Endpoint `/query`: RAG complet
   - Endpoint `/retrieve`: Retrieval uniquement
   - Endpoint `/index/stats`: Statistiques
   - Health check

4. **RAG Engine**
   - Retrieval avec top-k
   - Context building
   - Prompt template
   - Génération LLM

5. **Fonctionnalités Avancées**
   - Reranking (optionnel)
   - Context window management
   - Métadonnées filtrées
   - Logging et monitoring

### 🎓 Exercice
- Indexer le dataset fourni
- Lancer l'API FastAPI
- Tester les endpoints
- Ajouter de nouveaux documents
- Customiser le prompt template

### 🔧 Installation
```bash
cd s11_rag_demo
pip install -r requirements.txt
python indexer.py --input data/sample_docs.json --output index/
python app.py
```

### 🌐 Utilisation
```bash
# Health check
curl http://localhost:8000/health

# RAG query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Qu'\''est-ce que le machine learning?", "top_k": 3}'
```

---

## S12 — Vector DB & Production

### 🎯 Objectifs
- Adapter le choix d'index en production
- Comprendre persistance, backup, et sharding
- Comparer FAISS vs Milvus
- Mesurer latence et recall@k à grande échelle

### 📁 Fichiers
- **Notebook**: `notebooks/s12_vectordb.ipynb`

### 📚 Contenu
1. **Comparaison d'Index FAISS**
   - Flat: exact, lent
   - IVF: approximatif, rapide
   - HNSW: graph-based, très rapide
   - Benchmarks sur 10K documents

2. **Métriques de Performance**
   - Latence moyenne, P95, P99
   - Recall@k
   - Throughput
   - Memory usage

3. **Milvus: Vector Database**
   - Architecture distribuée
   - Collections et partitions
   - Opérations CRUD
   - Index types supportés

4. **FAISS vs Milvus**
   - Cas d'usage
   - Scalabilité
   - Features
   - Coût opérationnel

5. **Production Considerations**
   - Persistance et backup
   - Monitoring et alerting
   - Embedding freshness
   - Scaling strategies

6. **Décision Matrix**
   - Critères de choix
   - Tableau comparatif
   - Recommandations

### 🎓 Exercice
- Benchmark FAISS (Flat, IVF, HNSW) sur 10K docs
- Setup Milvus (optionnel si disponible)
- Comparer latence et recall@k
- Créer rapport de performance

### 🔧 Prérequis
```bash
pip install faiss-cpu pymilvus pandas numpy matplotlib
```

---

## S13 — LangChain: Patterns Pratiques

### 🎯 Objectifs
- Architecturer des apps GenAI avec LangChain
- Maîtriser Chains, Tools, Agents, Memory
- Implémenter une app Q&A avec mémoire
- Écrire des tests unitaires

### 📁 Fichiers
```
s13_langchain/
├── README.md
├── requirements.txt
├── app.py                # Application principale
├── tests/
│   ├── test_chains.py
│   └── test_memory.py
```

### 📚 Contenu
1. **Introduction LangChain**
   - Composants clés
   - LLM wrappers
   - Architecture

2. **Chains**
   - LLMChain: prompt + LLM
   - SequentialChain: chaîner plusieurs étapes
   - TransformChain: transformation de données

3. **Memory**
   - ConversationBufferMemory: historique complet
   - ConversationBufferWindowMemory: N derniers messages
   - Persistence

4. **Prompts**
   - PromptTemplate
   - ChatPromptTemplate
   - Few-shot examples

5. **Application Pratique**
   - CLI interactive
   - Q&A avec contexte
   - Conversation avec mémoire

### 🎓 Exercice
- Créer une chain Q&A personnalisée
- Implémenter mémoire conversationnelle
- Ajouter tests unitaires
- Tester différents patterns

### 🔧 Installation
```bash
cd s13_langchain
pip install -r requirements.txt
python app.py
```

---

## S14 — LangGraph & Orchestration

### 🎯 Objectifs
- Modéliser flows complexes avec graphes
- Créer pipelines réutilisables
- Implémenter retrieve → summarize → action
- Gérer state et transitions

### 📁 Fichiers
```
s14_langgraph/
├── README.md
├── requirements.txt
└── flow.py               # Implémentation du flow
```

### 📚 Contenu
1. **Concepts LangGraph**
   - StateGraph
   - Nodes et Edges
   - Conditional routing
   - State management

2. **Flow Implementation**
   - Node 1: Retrieval (vector search simulé)
   - Node 2: Summarize (résumé du contexte)
   - Node 3: Decide (routing conditionnel)
   - Node 4a: Action (réponse générée)
   - Node 4b: Escalate (besoin humain)

3. **Orchestration**
   - Composition de services
   - Gestion d'erreurs
   - Retry logic
   - Logging

4. **Testing**
   - Scénarios multiples
   - Validation du routing
   - State inspection

### 🎓 Exercice
- Implémenter un flow personnalisé
- Ajouter un node de validation
- Tester avec différents inputs
- Créer un diagramme ASCII de votre flow

### 🔧 Installation
```bash
cd s14_langgraph
pip install -r requirements.txt
python flow.py
```

---

## S15 — Agents Avancés

### 🎯 Objectifs
- Comprendre les types d'agents (planner/actor, tool-using)
- Implémenter un agent multi-tool
- Gérer l'exécution sécurisée des tools
- Créer des scénarios de test réalistes

### 📁 Fichiers
```
s15_agents/
├── README.md
├── requirements.txt
├── agent.py              # Agent ReAct
├── tools.py              # Définitions des tools
└── scenario_test.py      # Scénarios de test
```

### 📚 Contenu
1. **Architecture d'Agent**
   - ReAct pattern (Reasoning + Acting)
   - Tool interface
   - Execution loop

2. **Tools Disponibles**
   - web_search: recherche simulée
   - calculator: calculs sécurisés (AST-based)
   - currency_converter: conversion de devises
   - time: date/heure
   - string_length: utilitaire

3. **Agent Implementation**
   - Parsing des actions
   - Tool execution
   - Error handling
   - Max iterations

4. **Scénarios de Test**
   - "Plan a travel with budget"
   - Calculs multi-étapes
   - Recherche + analyse
   - Gestion d'erreurs

5. **Sécurité**
   - Sandboxing des tools
   - Input validation
   - Pas d'eval() dangereux

### 🎓 Exercice
- Ajouter un nouveau tool personnalisé
- Créer un scénario complexe
- Tester les limites de l'agent
- Implémenter retry logic

### 🔧 Installation
```bash
cd s15_agents
pip install -r requirements.txt
python scenario_test.py
```

---

## S16 — Flow Control & State Management

### 🎯 Objectifs
- Gérer la complexité des conversations longues
- Implémenter state persisté avec SQLite
- Créer des checkpoints et rollbacks
- Gérer la reprise après crash

### 📁 Fichiers
- **Notebook**: `notebooks/s16_flow_control.ipynb`

### 📚 Contenu
1. **State Management**
   - SQLite pour persistence
   - Schema de base de données
   - CRUD operations

2. **Checkpoints**
   - Sauvegarder l'état à chaque étape
   - Metadata (timestamp, user, status)
   - Historique complet

3. **Rollback**
   - Retour à un checkpoint précédent
   - Gestion des erreurs
   - Recovery automatique

4. **Workflows**
   - Conversation multi-étapes
   - Conditional branching
   - Parallel execution (simulation)

5. **Recovery**
   - Reprise après crash
   - Replay des étapes
   - Validation de cohérence

6. **Visualisation**
   - Timeline des états
   - Graphiques de progression
   - Statistiques

### 🎓 Exercice
- Implémenter un workflow avec 5+ étapes
- Tester le crash recovery
- Créer des branches conditionnelles
- Analyser l'historique des états

### 🔧 Prérequis
```bash
pip install sqlite3 pandas matplotlib
```

---

## 🔄 Workflow Complet GenAI

Voici comment utiliser ces matériels dans un projet end-to-end:

### 1. Fondations (S8-S9)
- **S8**: Comprendre les concepts LLM
- **S9**: Maîtriser les APIs et prompts

### 2. Retrieval (S10-S11)
- **S10**: Créer des embeddings et index FAISS
- **S11**: Déployer un RAG avec FastAPI

### 3. Production (S12)
- **S12**: Optimiser avec vector databases

### 4. Orchestration (S13-S14)
- **S13**: Structurer avec LangChain
- **S14**: Orchestrer des flows complexes

### 5. Agents (S15-S16)
- **S15**: Créer des agents autonomes
- **S16**: Gérer l'état et la persistence

---

## 🛠️ Installation Globale

Pour installer toutes les dépendances:

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances de base
pip install openai sentence-transformers faiss-cpu langchain \
            fastapi uvicorn pandas numpy matplotlib jupyter \
            python-dotenv tiktoken pydantic

# Pour LangGraph (optionnel)
pip install langgraph

# Pour Milvus (optionnel)
pip install pymilvus
```

---

## 📝 Exercices Transversaux

### Projet Intégré: Système Q&A Intelligent

**Phase 1: Fondations (S8-S10)**
1. Analyser les besoins en tokens
2. Créer et tester des prompts
3. Générer embeddings pour la knowledge base

**Phase 2: RAG (S11-S12)**
4. Implémenter le pipeline RAG
5. Optimiser avec vector DB
6. Mesurer performances

**Phase 3: Orchestration (S13-S14)**
7. Structurer avec LangChain
8. Créer un flow multi-étapes
9. Ajouter de la mémoire

**Phase 4: Agent (S15-S16)**
10. Transformer en agent autonome
11. Ajouter des tools
12. Implémenter state management

---

## 🔗 Ressources Supplémentaires

### Documentation Officielle
- [OpenAI Platform](https://platform.openai.com/docs)
- [LangChain Docs](https://python.langchain.com/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [FastAPI](https://fastapi.tiangolo.com/)

### Papers Fondamentaux
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Transformers)
- [BERT](https://arxiv.org/abs/1810.04805)
- [GPT-3](https://arxiv.org/abs/2005.14165)
- [RAG](https://arxiv.org/abs/2005.11401)
- [ReAct](https://arxiv.org/abs/2210.03629)

### Tutoriels Recommandés
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)
- [LangChain Tutorials](https://python.langchain.com/docs/tutorials/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

---

## 💡 Conseils Pédagogiques

### Pour les Instructeurs

**Ordre recommandé:**
- Suivre l'ordre S8 → S16 (progression logique)
- S8-S9: Théorie et API (2-3h chacun)
- S10-S11: RAG pratique (3-4h chacun)
- S12: Production (2-3h)
- S13-S14: Orchestration (2-3h chacun)
- S15-S16: Agents avancés (3-4h chacun)

**Points d'attention:**
- S8: Bien expliquer tokens et context window
- S9: Insister sur la sécurité des prompts
- S10-S11: Focus sur la qualité du retrieval
- S12: Tradeoffs performance vs coût
- S13-S14: Patterns réutilisables
- S15-S16: Robustesse et error handling

**Évaluation:**
- Mini-projet RAG (S11): 25%
- Benchmark Vector DB (S12): 15%
- Application LangChain (S13): 20%
- Agent multi-tool (S15): 20%
- Projet final intégré: 20%

### Pour les Étudiants

**Prérequis:**
- Python intermédiaire
- Bases de ML/DL
- API REST concepts
- Git et ligne de commande

**Méthode d'apprentissage:**
1. Lire la documentation de la séance
2. Exécuter tous les exemples
3. Modifier les paramètres
4. Faire les exercices
5. Créer un mini-projet personnel

**Ressources d'aide:**
- Documentation inline dans le code
- README de chaque session
- Discord/Slack de la formation
- Stack Overflow
- GitHub Issues

---

## ✅ Checklist de Complétion

### S8 - Fondamentaux LLM
- [ ] Comprendre architecture Transformer
- [ ] Maîtriser tokenisation
- [ ] Différencier embeddings et logits
- [ ] Gérer context window
- [ ] Configurer hyperparamètres

### S9 - OpenAI API
- [ ] Configurer API OpenAI
- [ ] Faire des appels Chat/Embeddings
- [ ] Tester 10 prompts variés
- [ ] Comparer les résultats
- [ ] Implémenter sécurité

### S10 - Embeddings & Vector Search
- [ ] Créer embeddings
- [ ] Implémenter FAISS Flat
- [ ] Implémenter FAISS IVF
- [ ] Mesurer recall/precision
- [ ] Sauvegarder index

### S11 - RAG
- [ ] Indexer documents
- [ ] Lancer API FastAPI
- [ ] Tester endpoints
- [ ] Analyser performances
- [ ] Customiser prompts

### S12 - Vector DB Production
- [ ] Benchmark FAISS (3 types)
- [ ] Tester Milvus (optionnel)
- [ ] Mesurer latence P95/P99
- [ ] Créer rapport de décision

### S13 - LangChain
- [ ] Créer une chain
- [ ] Implémenter mémoire
- [ ] Écrire tests unitaires
- [ ] Tester patterns

### S14 - LangGraph
- [ ] Créer un StateGraph
- [ ] Implémenter routing conditionnel
- [ ] Tester flow complet
- [ ] Créer diagramme

### S15 - Agents
- [ ] Implémenter agent ReAct
- [ ] Créer 2+ tools personnalisés
- [ ] Tester scénarios complexes
- [ ] Gérer erreurs

### S16 - State Management
- [ ] Implémenter persistence SQLite
- [ ] Créer checkpoints
- [ ] Tester rollback
- [ ] Démo crash recovery

---

## 🐛 Dépannage Commun

### S9: "Invalid API key"
```bash
# Vérifier .env
cat .env | grep OPENAI_API_KEY

# Tester la clé
python -c "import openai; openai.api_key='YOUR_KEY'; print('OK')"
```

### S10-S11: "Index not found"
```bash
# Réindexer
python indexer.py --input data/sample_docs.json --output index/
```

### S12: "Memory error with FAISS"
```bash
# Réduire dataset ou utiliser IVF
python indexer.py --index-type ivf --nlist 50
```

### S13: "LangChain module not found"
```bash
# Installer version spécifique
pip install langchain==0.1.0 langchain-openai
```

### S15: "Tool execution failed"
```bash
# Vérifier logs
python agent.py --debug

# Tester tool individuellement
python -c "from tools import calculator; print(calculator('2+2'))"
```

---

## 📧 Support

Pour toute question:
1. Consulter le README de la session
2. Vérifier les exemples de code
3. Rechercher dans les issues GitHub
4. Poster sur le forum de la formation
5. Contacter les instructeurs

---

## 📜 Licence

Matériel pédagogique fourni à des fins éducatives.
Les bibliothèques utilisées ont leurs propres licences.

---

**Bon apprentissage et bienvenue dans le monde de la GenAI! 🚀🤖**
