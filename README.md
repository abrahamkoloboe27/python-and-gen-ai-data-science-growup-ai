# 🚀 Python & GenAI Data Science - GrowUp AI

Formation complète en Data Science, Machine Learning et Intelligence Artificielle Générative.

## 📚 Vue d'Ensemble

Ce repository contient les matériels de cours, notebooks, et projets pour la formation **Python & GenAI Data Science**. La formation est structurée en 16 sessions progressives couvrant des statistiques de base jusqu'aux agents LLM avancés.

### 🎯 Objectifs de la Formation

- Maîtriser Python pour la Data Science
- Comprendre et appliquer le Machine Learning
- Créer des APIs et interfaces utilisateur
- Développer des applications GenAI avec LLMs
- Déployer des systèmes RAG en production
- Orchestrer des agents IA autonomes

## 📋 Structure du Cours

### 📊 Module 1: Fondamentaux Data Science (S1-S7)

**Sessions S3-S7** - *Voir [SESSIONS_S3-S7.md](./SESSIONS_S3-S7.md) pour détails*

- **S3**: Statsmodels & analyses statistiques
- **S4**: Scikit-learn & pipelines ML
- **S5**: PyCaret pour prototypage rapide
- **S6**: FastAPI - Serveurs applicatifs
- **S7**: Streamlit - Prototypage d'interface

### 🤖 Module 2: GenAI & LLMs (S8-S16)

**Sessions S8-S16** - *Voir [SESSIONS_S8-S16.md](./SESSIONS_S8-S16.md) pour détails complets*

#### Fondamentaux LLM (S8-S10)
- **S8**: Fondamentaux LLM & Transformers
- **S9**: OpenAI API & Prompt Engineering  
- **S10**: Embeddings & Recherche Vectorielle (RAG Intro)

#### RAG & Production (S11-S12)
- **S11**: RAG - Architecture & Patterns (Mini-projet FastAPI)
- **S12**: Vector Databases & Production (FAISS vs Milvus)

#### Orchestration (S13-S14)
- **S13**: LangChain - Patterns Pratiques (Chains, Memory, Tools)
- **S14**: LangGraph & Orchestration de Flows

#### Agents Avancés (S15-S16)
- **S15**: Agents Avancés - Design Patterns & Capabilities
- **S16**: Flow Control & State Management

## 🗂️ Organisation du Repository

```
.
├── README.md                    # Ce fichier
├── SESSIONS_S3-S7.md           # Guide détaillé S3-S7
├── SESSIONS_S8-S16.md          # Guide détaillé S8-S16
│
├── notebooks/                   # Jupyter notebooks
│   ├── s3_stats.ipynb          # Statsmodels
│   ├── s4_ml_pipeline.ipynb    # Scikit-learn
│   ├── s5_pycaret.ipynb        # PyCaret
│   ├── s9_openai_prompts.ipynb # OpenAI API
│   ├── s10_embeddings.ipynb    # Embeddings & FAISS
│   ├── s12_vectordb.ipynb      # Vector Databases
│   └── s16_flow_control.ipynb  # State Management
│
├── s6_fastapi/                  # Projet FastAPI
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
│
├── s7_streamlit/                # Projet Streamlit
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
│
├── s8_llm_fundamentals.md       # Note technique LLM
│
├── s11_rag_demo/                # Mini-projet RAG
│   ├── app.py                   # API FastAPI
│   ├── indexer.py              # Indexation
│   ├── rag_engine.py           # Moteur RAG
│   ├── data/                   # Documents
│   └── README.md
│
├── s13_langchain/               # Application LangChain
│   ├── app.py
│   ├── tests/
│   └── README.md
│
├── s14_langgraph/               # Orchestration LangGraph
│   ├── flow.py
│   └── README.md
│
└── s15_agents/                  # Agents Multi-tool
    ├── agent.py
    ├── tools.py
    ├── scenario_test.py
    └── README.md
```

## 🚀 Démarrage Rapide

### Prérequis

- Python 3.9+
- pip ou conda
- Git

### Installation

```bash
# Cloner le repository
git clone https://github.com/abrahamkoloboe27/python-and-gen-ai-data-science-growup-ai.git
cd python-and-gen-ai-data-science-growup-ai

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances de base
pip install pandas numpy matplotlib jupyter
```

### Dépendances par Module

#### Module 1: Data Science (S3-S7)
```bash
pip install statsmodels scipy scikit-learn pycaret[full] \
            fastapi uvicorn[standard] streamlit plotly \
            pandas numpy matplotlib jupyter
```

#### Module 2: GenAI (S8-S16)
```bash
pip install openai sentence-transformers faiss-cpu langchain \
            tiktoken python-dotenv pydantic
```

## 📖 Guide d'Utilisation

### Pour les Étudiants

1. **Commencer par les fondamentaux** (S3-S7)
   - Suivre l'ordre des sessions
   - Exécuter tous les notebooks
   - Faire les exercices pratiques

2. **Progresser vers GenAI** (S8-S16)
   - Lire les documents théoriques
   - Tester les APIs
   - Créer vos propres projets

3. **Projet Final**
   - Intégrer plusieurs concepts
   - Déployer en production
   - Documenter votre travail

### Pour les Instructeurs

- Chaque session = 2-4 heures de cours
- Matériel prêt à l'emploi
- Exercices progressifs
- Évaluations suggérées dans les guides

## 🎓 Exercices et Projets

### Mini-Projets Intégrés

1. **Système de Classification ML** (S3-S7)
   - Analyse statistique → Pipeline ML → API → Interface

2. **Assistant Q&A Intelligent** (S8-S16)
   - Embeddings → RAG → LangChain → Agent

### Exercices Par Session

Voir les fichiers `SESSIONS_*.md` pour les exercices détaillés de chaque session.

## 🔧 Configuration

### Variables d'Environnement

Pour les sessions GenAI (S9+), créer un fichier `.env`:

```bash
# OpenAI
OPENAI_API_KEY=votre_clé_api

# Configuration Embeddings
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2

# Configuration LLM
LLM_MODEL=gpt-3.5-turbo
```

## 📊 Datasets

Les notebooks utilisent des datasets d'exemple. Pour vos propres projets:
- Respecter le format CSV/JSON documenté
- Nettoyer les données avant utilisation
- Vérifier les licences des datasets publics

## 🔗 Ressources Complémentaires

### Documentation
- [OpenAI Platform](https://platform.openai.com/docs)
- [LangChain](https://python.langchain.com/)
- [Scikit-learn](https://scikit-learn.org/)
- [FastAPI](https://fastapi.tiangolo.com/)

### Papers Importants
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformers
- [BERT](https://arxiv.org/abs/1810.04805)
- [GPT-3](https://arxiv.org/abs/2005.14165)
- [RAG](https://arxiv.org/abs/2005.11401)

### Tutoriels
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

## 🤝 Contribution

Les contributions sont bienvenues!

1. Fork le repository
2. Créer une branche (`git checkout -b feature/amelioration`)
3. Commit vos changements (`git commit -m 'Ajout de...'`)
4. Push vers la branche (`git push origin feature/amelioration`)
5. Ouvrir une Pull Request

## 📝 Licence

Ce matériel pédagogique est fourni à des fins éducatives.

## 📧 Contact & Support

Pour questions ou assistance:
- Ouvrir une Issue GitHub
- Consulter les guides de session
- Vérifier la FAQ dans les README spécifiques

## ⭐ Remerciements

Merci à tous les contributeurs et à la communauté open-source pour les outils extraordinaires qui rendent cette formation possible.

---

**Bonne formation! 🚀 Happy Learning! 📚**
