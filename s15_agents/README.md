# S15 — Advanced Agents

## 🎯 Objectifs
- Comprendre l'architecture des agents autonomes
- Implémenter un agent multi-tools (web_search + calculator)
- Maîtriser le ReAct pattern (Reasoning + Acting)
- Créer des scénarios de test réalistes
- Gérer les tool calls et error handling

## 📋 Architecture Agent

```
┌────────────────────────────────────────────┐
│           AGENT (ReAct Loop)               │
├────────────────────────────────────────────┤
│                                            │
│  1. OBSERVE: Analyser la requête          │
│                                            │
│  2. THINK: Raisonner sur l'action          │
│      ├─ Quel outil utiliser?              │
│      └─ Quels paramètres?                 │
│                                            │
│  3. ACT: Exécuter l'outil                 │
│      ├─ web_search()                      │
│      ├─ calculator()                      │
│      └─ custom_tool()                     │
│                                            │
│  4. OBSERVE: Analyser le résultat          │
│                                            │
│  5. REPEAT ou RESPOND                      │
│      └─ Continuer ou répondre?            │
│                                            │
└────────────────────────────────────────────┘

            ┌─────────────┐
            │   TOOLS     │
            ├─────────────┤
            │ web_search  │
            │ calculator  │
            │ custom      │
            └─────────────┘
```

## 🚀 Installation

```bash
cd s15_agents
pip install -r requirements.txt
```

## 📁 Structure du projet

```
s15_agents/
├── README.md              # Ce fichier
├── requirements.txt       # Dépendances
├── agent.py              # Agent multi-tools
├── tools.py              # Définitions des outils
├── scenario_test.py      # Tests de scénarios
└── .env.example          # Configuration
```

## 🔧 Configuration

Créer un fichier `.env`:
```bash
OPENAI_API_KEY=votre_clé_api
MODEL_NAME=gpt-4
TEMPERATURE=0.2
MAX_ITERATIONS=10
```

## 📚 Concepts Clés

### 1. ReAct Pattern
**Re**asoning + **Act**ing en boucle:
- **Thought**: Raisonnement sur la prochaine action
- **Action**: Exécution d'un outil
- **Observation**: Résultat de l'action
- **Repeat**: Jusqu'à avoir la réponse finale

### 2. Tool Definition
Chaque outil doit avoir:
```python
{
    "name": "calculator",
    "description": "Calcule des expressions mathématiques",
    "parameters": {
        "expression": "string (ex: '2 + 2')"
    }
}
```

### 3. Agent Loop
```python
while not done:
    thought = agent.think()
    action, params = agent.parse_action(thought)
    observation = tools.execute(action, params)
    done = agent.should_finish(observation)
```

### 4. Error Recovery
L'agent doit gérer:
- Outil inexistant
- Paramètres invalides
- Erreurs d'exécution
- Boucles infinies

## 🌐 Utilisation

### Lancer l'agent
```bash
python agent.py
```

### Exemple programmatique

```python
from agent import ReactAgent
from tools import get_tools

# Créer l'agent
tools = get_tools()
agent = ReactAgent(tools=tools)

# Exécuter une tâche
result = agent.run("Combien coûte un billet Paris-Londres?")
print(result)
```

### Tester les scénarios
```bash
python scenario_test.py
```

## 🔍 Outils Disponibles

### Tool 1: Web Search (stub)
**Usage**: Rechercher des informations en ligne
```python
web_search(query: str) -> str
```

**Exemple**: "Prix moyen d'un billet Paris-Londres"

### Tool 2: Calculator
**Usage**: Effectuer des calculs
```python
calculator(expression: str) -> float
```

**Exemple**: "150 * 1.2 + 50"

### Tool 3: Date/Time (bonus)
**Usage**: Obtenir la date/heure actuelle
```python
get_current_time() -> str
```

## 🎯 Scénarios de Test

### Scénario 1: Travel Planning
**Tâche**: "Planifie un voyage à Paris avec un budget de 1000€"

**Étapes attendues**:
1. Recherche prix vols
2. Recherche prix hôtels
3. Calcul budget restant
4. Recommandations

### Scénario 2: Complex Calculation
**Tâche**: "Si un produit coûte 49€ et il y a -20%, combien ça fait?"

**Étapes attendues**:
1. Parse la question
2. Calcul: 49 * 0.8
3. Réponse formatée

### Scénario 3: Multi-step Research
**Tâche**: "Quelle est la capitale de la France et combien d'habitants?"

**Étapes attendues**:
1. Recherche capitale
2. Recherche population
3. Synthèse

## 💡 Patterns Avancés

### Pattern 1: Tool Chaining
Enchaîner plusieurs outils automatiquement
```python
# Recherche → Calcul → Validation
result = agent.run_chain([
    ("web_search", {"query": "prix X"}),
    ("calculator", {"expression": "prix * quantity"}),
    ("validate", {"result": result})
])
```

### Pattern 2: Parallel Tool Execution
Exécuter plusieurs outils en parallèle
```python
results = agent.run_parallel([
    ("web_search", {"query": "hôtels Paris"}),
    ("web_search", {"query": "restaurants Paris"})
])
```

### Pattern 3: Human-in-the-Loop
Demander confirmation avant action
```python
if tool.requires_confirmation:
    approved = ask_user(f"Execute {tool.name}?")
    if not approved:
        return "Action cancelled"
```

### Pattern 4: Tool Result Caching
Cache les résultats pour éviter les appels répétés
```python
@lru_cache(maxsize=100)
def web_search(query: str) -> str:
    # Cached search
    pass
```

## 🧪 Tests

### Tests unitaires
```bash
pytest -v
```

### Tests de scénarios
```bash
python scenario_test.py --verbose
```

### Tests de performance
```bash
python scenario_test.py --benchmark
```

## 📊 Monitoring

L'agent logge automatiquement:
- Chaque pensée (thought)
- Chaque action executée
- Chaque observation
- Nombre d'itérations
- Temps total

Exemple de log:
```
[Iteration 1] Thought: Je dois chercher le prix...
[Iteration 1] Action: web_search("prix billet Paris-Londres")
[Iteration 1] Observation: Environ 80-150€
[Iteration 2] Thought: Je peux répondre maintenant
[Iteration 2] Action: Final Answer
```

## 🔒 Sécurité

### Limitations
- Max iterations (éviter boucles infinies)
- Tool whitelist (uniquement outils autorisés)
- Input validation
- Output sanitization

### Sandbox
```python
agent = ReactAgent(
    tools=tools,
    max_iterations=10,
    allowed_tools=["web_search", "calculator"],
    sandbox_mode=True
)
```

## 🚢 Déploiement

### Avec FastAPI
```python
from fastapi import FastAPI
from agent import ReactAgent

app = FastAPI()
agent = ReactAgent(tools=get_tools())

@app.post("/agent")
async def run_agent(task: str):
    result = agent.run(task)
    return {"result": result}
```

## 🐛 Dépannage

### "Agent stuck in loop"
```python
# Augmenter max_iterations ou améliorer les prompts
agent = ReactAgent(tools, max_iterations=15)
```

### "Tool not found"
```python
# Vérifier que l'outil est bien enregistré
print(agent.list_tools())
```

### "Invalid tool parameters"
```python
# Améliorer la description de l'outil
tool.description = "Détails précis avec exemples..."
```

## 📚 Ressources

- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)

## 🎓 Exercices

### Exercice 1: Nouvel Outil
1. Créez un outil `weather(city: str)`
2. Intégrez-le à l'agent
3. Testez: "Quel temps fait-il à Paris?"

### Exercice 2: Error Handling
1. Forcez une erreur dans un outil
2. Vérifiez que l'agent la gère
3. Ajoutez un retry mechanism

### Exercice 3: Multi-Agent
1. Créez 2 agents (researcher + planner)
2. Faites-les collaborer sur une tâche
3. Mesurez l'amélioration

### Exercice 4: Tool Composition
1. Créez un outil qui utilise d'autres outils
2. Implémentez "smart_search" = search + summarize
3. Testez sur des requêtes complexes

## ✅ Checklist

- [ ] Dépendances installées
- [ ] .env configuré
- [ ] Agent s'exécute
- [ ] Tous les outils testés
- [ ] Scénarios passent
- [ ] Error handling OK
- [ ] Logs compréhensibles
- [ ] Performance acceptable

---

**Mini-projet S15 — Advanced Agents 🤖**

## 📞 Support

Pour toute question:
1. Vérifier les logs détaillés
2. Tester chaque outil individuellement
3. Simplifier la tâche si l'agent est bloqué
4. Ajuster max_iterations si nécessaire
