# S14 — LangGraph & Orchestration

## 🎯 Objectifs
- Comprendre les graphes de workflows avec LangGraph
- Implémenter un pipeline retrieve → summarize → action
- Gérer les états et les transitions entre nœuds
- Orchestrer des agents complexes avec des branches conditionnelles

## 📋 Architecture du Flow

```
                    ┌──────────────┐
                    │    START     │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   RETRIEVE   │  (Recherche vectorielle)
                    │              │
                    │ - Query user │
                    │ - Search DB  │
                    │ - Get top-k  │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  SUMMARIZE   │  (Résumé LLM)
                    │              │
                    │ - Aggregate  │
                    │ - LLM call   │
                    │ - Generate   │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   DECIDE     │  (Branchement)
                    │              │
                    │ Confidence?  │
                    └──┬───────┬───┘
                       │       │
            ┌──────────┘       └──────────┐
            │                              │
            ▼                              ▼
     ┌──────────────┐              ┌──────────────┐
     │    ACTION    │              │   ESCALATE   │
     │              │              │              │
     │ - Execute    │              │ - Human      │
     │ - Respond    │              │ - Review     │
     └──────┬───────┘              └──────┬───────┘
            │                              │
            └──────────┬───────────────────┘
                       │
                       ▼
                ┌──────────────┐
                │     END      │
                └──────────────┘
```

## 🚀 Installation

```bash
cd s14_langgraph
pip install -r requirements.txt
```

## 📁 Structure du projet

```
s14_langgraph/
├── README.md              # Ce fichier
├── requirements.txt       # Dépendances
├── flow.py               # Pipeline LangGraph
└── .env.example          # Configuration
```

## 🔧 Configuration

Créer un fichier `.env`:
```bash
OPENAI_API_KEY=votre_clé_api
MODEL_NAME=gpt-3.5-turbo
TEMPERATURE=0.3
CONFIDENCE_THRESHOLD=0.7
```

## 📚 Concepts Clés

### 1. StateGraph
Le graphe d'états définit:
- **Nodes**: Fonctions de traitement
- **Edges**: Transitions entre nœuds
- **Conditional Edges**: Branches basées sur des conditions

### 2. State Management
L'état est partagé entre tous les nœuds:
```python
class GraphState(TypedDict):
    query: str              # Requête utilisateur
    documents: List[str]    # Documents récupérés
    summary: str           # Résumé généré
    confidence: float      # Score de confiance
    action: str            # Action à effectuer
```

### 3. Nodes (Nœuds)
Chaque nœud est une fonction qui:
- Reçoit l'état actuel
- Effectue un traitement
- Retourne l'état mis à jour

### 4. Conditional Routing
Décisions basées sur l'état:
```python
def decide_next_step(state):
    if state["confidence"] > 0.7:
        return "action"
    else:
        return "escalate"
```

## 🌐 Utilisation

### Lancer le flow
```bash
python flow.py
```

### Exemple d'utilisation programmatique

```python
from flow import RAGWorkflow

# Créer le workflow
workflow = RAGWorkflow()

# Exécuter avec une requête
result = workflow.run("Comment réinitialiser mon mot de passe?")

print(f"Summary: {result['summary']}")
print(f"Action: {result['action']}")
print(f"Confidence: {result['confidence']}")
```

## 🔍 Détail des Nœuds

### Node 1: Retrieve
**Fonction**: Rechercher des documents pertinents
```python
def retrieve_node(state):
    query = state["query"]
    documents = vector_search(query, top_k=5)
    return {"documents": documents}
```

### Node 2: Summarize
**Fonction**: Résumer les documents récupérés
```python
def summarize_node(state):
    docs = state["documents"]
    summary = llm.summarize(docs)
    confidence = calculate_confidence(summary, docs)
    return {"summary": summary, "confidence": confidence}
```

### Node 3: Decide
**Fonction**: Décider du prochain nœud
```python
def decide_node(state):
    if state["confidence"] > THRESHOLD:
        return "action"
    else:
        return "escalate"
```

### Node 4: Action
**Fonction**: Exécuter l'action appropriée
```python
def action_node(state):
    action = determine_action(state["summary"])
    return {"action": action}
```

### Node 5: Escalate
**Fonction**: Escalader vers un humain
```python
def escalate_node(state):
    return {"action": "human_review_required"}
```

## 🧪 Tests

```bash
# Tester le flow complet
python flow.py --test

# Tester avec différentes requêtes
python flow.py --query "Votre question ici"
```

## 📊 Visualisation du Graphe

Le script génère automatiquement une visualisation du graphe:
```python
workflow.visualize("workflow_graph.png")
```

## 💡 Patterns Avancés

### Pattern 1: Parallel Execution
Exécuter plusieurs nœuds en parallèle
```python
# Rechercher dans plusieurs sources en parallèle
graph.add_node("retrieve_db", retrieve_from_db)
graph.add_node("retrieve_api", retrieve_from_api)
graph.add_node("merge", merge_results)
```

### Pattern 2: Retry Logic
Réessayer en cas d'échec
```python
def node_with_retry(state, max_retries=3):
    for attempt in range(max_retries):
        try:
            return process(state)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            continue
```

### Pattern 3: Human-in-the-Loop
Demander validation humaine
```python
def human_review_node(state):
    # Pause et attendre input
    approved = request_human_approval(state)
    if approved:
        return "continue"
    else:
        return "abort"
```

## 🔒 Gestion des Erreurs

Le workflow gère automatiquement:
- Timeouts
- Erreurs API
- États invalides
- Boucles infinies (max iterations)

```python
workflow = RAGWorkflow(
    max_iterations=10,
    timeout=30
)
```

## 🚢 Déploiement

### Avec FastAPI
```python
from fastapi import FastAPI
from flow import RAGWorkflow

app = FastAPI()
workflow = RAGWorkflow()

@app.post("/query")
async def process_query(query: str):
    result = workflow.run(query)
    return result
```

## 📚 Ressources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Examples](https://github.com/langchain-ai/langgraph/tree/main/examples)
- [State Machines Guide](https://en.wikipedia.org/wiki/Finite-state_machine)
- [Workflow Orchestration Patterns](https://docs.temporal.io/workflows)

## 🎓 Exercices

### Exercice 1: Ajouter un nœud
1. Ajoutez un nœud de validation après retrieve
2. Filtrez les documents de faible qualité
3. Mesurez l'impact sur le recall

### Exercice 2: Branchement multiple
1. Modifiez decide_node pour 3+ branches
2. Ajoutez une branche pour "clarification needed"
3. Testez avec des requêtes ambiguës

### Exercice 3: Parallel Retrieval
1. Implémentez la recherche parallèle (DB + API)
2. Mergez les résultats
3. Comparez la latence vs séquentiel

### Exercice 4: Stateful Conversation
1. Ajoutez un état de conversation
2. Gardez l'historique entre appels
3. Implémentez des follow-up questions

## ✅ Checklist

- [ ] Dépendances installées
- [ ] .env configuré
- [ ] Flow s'exécute sans erreur
- [ ] Graphe visualisé
- [ ] Tous les nœuds testés
- [ ] Branching conditionnel fonctionne
- [ ] Gestion d'erreurs testée

---

**Mini-projet S14 — LangGraph & Orchestration 🔀**

## 📞 Support

Pour toute question:
1. Vérifier les logs de chaque nœud
2. Visualiser le graphe
3. Tester chaque nœud individuellement
4. Vérifier les transitions d'état
