# S13 — LangChain Patterns

## 🎯 Objectifs
- Comprendre l'architecture LangChain et ses composants
- Implémenter des chains pour orchestrer des LLMs
- Gérer la mémoire conversationnelle
- Créer un système de Q&A avec contexte
- Maîtriser les templates de prompts

## 📋 Architecture

```
┌─────────────────────────────────────────────┐
│           LangChain Application             │
├─────────────────────────────────────────────┤
│                                             │
│  ┌────────────┐    ┌──────────────┐       │
│  │  Prompts   │───▶│   Chains     │       │
│  │ Templates  │    │ (Sequential) │       │
│  └────────────┘    └──────┬───────┘       │
│                           │                │
│  ┌────────────┐    ┌──────▼───────┐       │
│  │   Memory   │◀───│     LLM      │       │
│  │(Conversation)│    │   (OpenAI)   │       │
│  └────────────┘    └──────┬───────┘       │
│                           │                │
│  ┌────────────┐    ┌──────▼───────┐       │
│  │   Tools    │◀───│   Outputs    │       │
│  │(Search,etc)│    │   (Parsed)   │       │
│  └────────────┘    └──────────────┘       │
│                                             │
└─────────────────────────────────────────────┘
```

## 🚀 Installation

```bash
cd s13_langchain
pip install -r requirements.txt
```

## 📁 Structure du projet

```
s13_langchain/
├── README.md              # Ce fichier
├── requirements.txt       # Dépendances
├── app.py                # Application principale
├── .env.example          # Configuration
└── tests/                # Tests unitaires
    ├── __init__.py
    ├── test_chains.py
    └── test_memory.py
```

## 🔧 Configuration

Créer un fichier `.env`:
```bash
OPENAI_API_KEY=votre_clé_api
MODEL_NAME=gpt-3.5-turbo
TEMPERATURE=0.7
MAX_TOKENS=500
```

## 📚 Concepts Clés

### 1. Chains (Chaînes)
Les chains permettent d'enchaîner plusieurs opérations:
- **LLMChain**: Chain basique (prompt → LLM → output)
- **SequentialChain**: Enchaînement séquentiel de chains
- **SimpleSequentialChain**: Version simplifiée (output → input suivant)

### 2. Memory (Mémoire)
Gestion du contexte conversationnel:
- **ConversationBufferMemory**: Stocke tout l'historique
- **ConversationBufferWindowMemory**: Garde les N derniers messages
- **ConversationSummaryMemory**: Résume l'historique progressivement

### 3. Prompts
Templates réutilisables pour formater les entrées:
- Variables dynamiques
- Exemples few-shot
- Instructions système

## 🌐 Utilisation

### Lancer l'application
```bash
python app.py
```

### Exemples d'utilisation

#### 1. Simple Chain
```python
from app import simple_chain_demo

response = simple_chain_demo("Explique-moi les transformers")
print(response)
```

#### 2. Conversation avec Mémoire
```python
from app import conversation_demo

conversation_demo()
# Tapez vos messages, l'historique est conservé
```

#### 3. Q&A avec Contexte
```python
from app import qa_demo

context = """
Le RAG (Retrieval-Augmented Generation) combine la recherche 
d'information et la génération de texte.
"""

response = qa_demo(context, "Comment fonctionne le RAG?")
print(response)
```

## 🔍 Patterns Implémentés

### Pattern 1: Chain Simple (LLMChain)
Utilisez pour des tâches simples avec prompt template.

**Use case**: Classification, extraction, résumé basique

### Pattern 2: Sequential Chain
Enchaînez plusieurs étapes de traitement.

**Use case**: 
- Génération puis traduction
- Résumé puis analyse de sentiment
- Extraction puis validation

### Pattern 3: Conversation avec Mémoire
Gardez le contexte sur plusieurs tours.

**Use case**:
- Chatbots
- Assistants conversationnels
- Support client

### Pattern 4: Q&A avec RAG
Combinez retrieval et génération.

**Use case**:
- FAQ dynamique
- Documentation search
- Knowledge base

## 🧪 Tests

```bash
# Tous les tests
pytest tests/

# Tests spécifiques
pytest tests/test_chains.py
pytest tests/test_memory.py

# Avec coverage
pytest --cov=. tests/
```

## 📊 Monitoring

L'application logge automatiquement:
- Prompts envoyés
- Tokens utilisés
- Latence des requêtes
- Erreurs et exceptions

Exemple de log:
```
[2024-01-22 10:30:45] INFO: Chain 'simple_chain' invoked
[2024-01-22 10:30:45] INFO: Prompt: "Explique-moi..."
[2024-01-22 10:30:46] INFO: Response received (120 tokens)
[2024-01-22 10:30:46] INFO: Cost: $0.0012
```

## 💡 Bonnes Pratiques

### 1. Gestion de la Mémoire
```python
# ❌ Mauvais: mémoire illimitée
memory = ConversationBufferMemory()

# ✅ Bon: limiter la mémoire
memory = ConversationBufferWindowMemory(k=5)  # Garde 5 derniers messages
```

### 2. Error Handling
```python
# ✅ Toujours gérer les erreurs
try:
    response = chain.run(input)
except Exception as e:
    logger.error(f"Chain failed: {e}")
    response = "Désolé, une erreur s'est produite."
```

### 3. Prompt Templates
```python
# ✅ Utiliser des templates réutilisables
template = PromptTemplate(
    input_variables=["topic"],
    template="Explique {topic} en 3 phrases."
)
```

### 4. Token Management
```python
# ✅ Surveiller les tokens
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    response = chain.run(input)
    print(f"Tokens: {cb.total_tokens}, Cost: ${cb.total_cost}")
```

## 🔒 Sécurité

- Validation des inputs utilisateur
- Sanitization des outputs
- Rate limiting
- Pas de données sensibles dans les logs
- Variables d'environnement pour les secrets

## 🚢 Déploiement

### Docker
```bash
docker build -t langchain-app .
docker run -p 8000:8000 --env-file .env langchain-app
```

### Avec FastAPI
```python
# Wrappez l'app dans une API REST
from fastapi import FastAPI

app = FastAPI()

@app.post("/chat")
async def chat(message: str):
    return {"response": chain.run(message)}
```

## 🐛 Dépannage

### "API key not found"
```bash
# Vérifier le .env
cat .env | grep OPENAI_API_KEY
```

### "Memory full"
```python
# Réduire la fenêtre de mémoire
memory = ConversationBufferWindowMemory(k=3)
```

### "Prompt too long"
```python
# Utiliser ConversationSummaryMemory
memory = ConversationSummaryMemory(llm=llm)
```

## 📚 Ressources

- [LangChain Documentation](https://python.langchain.com/)
- [LangChain Cookbook](https://github.com/langchain-ai/langchain/tree/master/cookbook)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [LangChain Templates](https://github.com/langchain-ai/langchain/tree/master/templates)

## 🎓 Exercices

### Exercice 1: Custom Chain
1. Créez une chain qui traduit puis résume un texte
2. Utilisez SequentialChain
3. Testez avec différents textes

### Exercice 2: Mémoire Optimisée
1. Implémentez ConversationSummaryMemory
2. Comparez avec BufferWindowMemory
3. Mesurez la différence de tokens

### Exercice 3: Q&A avec Sources
1. Créez une chain Q&A qui cite ses sources
2. Ajoutez un retriever (FAISS)
3. Retournez les sources avec la réponse

### Exercice 4: Multi-Chain Pipeline
1. Créez un pipeline: extraction → classification → résumé
2. Gérez les erreurs à chaque étape
3. Loggez les métriques

## ✅ Checklist

- [ ] Dépendances installées
- [ ] .env configuré
- [ ] Tests passent
- [ ] Simple chain fonctionne
- [ ] Mémoire conversationnelle OK
- [ ] Q&A avec contexte OK
- [ ] Logs fonctionnels

---

**Mini-projet S13 — LangChain Patterns 🦜⛓️**

## 📞 Support

Pour toute question:
1. Vérifier les logs
2. Consulter la documentation LangChain
3. Tester avec des exemples simples
4. Vérifier les quotas API
