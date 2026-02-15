# S9 - Groq et Prompt Engineering Pratique

Ce dossier contient deux notebooks pour la séance 9 (S9) du cours sur l'intelligence artificielle générative.

## 📚 Notebooks Disponibles

### 1. `s9_groq_prompts.ipynb` - API Native Groq
Notebook utilisant le SDK natif de Groq.

**Contenu:**
- Configuration de l'API Groq
- Appels API de base avec le client Groq
- Techniques de prompt engineering (zero-shot, few-shot, chain-of-thought)
- 10 prompts expérimentaux pour résumé/Q&A
- Chaînes de prompts
- Instructions de sécurité
- Analyse comparative des résultats

**Modèle principal:** `llama-3.3-70b-versatile`

### 2. `s9_groq_openai_compatible.ipynb` - API OpenAI Compatible
Notebook utilisant le SDK OpenAI avec l'endpoint compatible de Groq.

**Contenu:**
- Configuration du client OpenAI avec l'endpoint Groq
- Appels API avec interface OpenAI
- Même techniques de prompt engineering
- 10 prompts expérimentaux identiques
- Streaming des réponses
- Guide de migration OpenAI → Groq
- Comparaison des deux approches

**Modèles supportés:** 
- `llama-3.3-70b-versatile` (défaut)
- `llama-3.1-70b-versatile`
- `llama-3.1-8b-instant`
- `mixtral-8x7b-32768`
- `gemma2-9b-it`

## 🎯 Objectifs Pédagogiques

1. **Maîtrise des APIs** - Appels API (Chat Completions)
2. **Prompt Engineering** - Bonnes pratiques de design de prompts
3. **Few-shot vs Zero-shot** - Comprendre les différences
4. **Sécurité** - Instructions de sécurité et validation
5. **Comparaison** - Native vs OpenAI-compatible

## 📋 Livrables

Chaque notebook produit:
- ✅ 10 prompts testés pour une tâche de résumé/Q&A
- ✅ Comparaison des outputs
- ✅ Documentation des prompt templates
- ✅ Fichiers JSON avec les résultats

## 🚀 Installation

```bash
# Pour le notebook natif Groq
pip install groq python-dotenv

# Pour le notebook OpenAI-compatible
pip install openai python-dotenv
```

## 🔑 Configuration

Créez un fichier `.env` à la racine du projet:

```env
GROQ_API_KEY=votre_clé_api_groq
```

Obtenez votre clé API sur: https://console.groq.com/

## 💡 Quelle Approche Choisir?

### Utilisez l'approche **Native Groq** si:
- ✅ Vous commencez un nouveau projet spécifiquement pour Groq
- ✅ Vous voulez utiliser des features spécifiques à Groq
- ✅ Vous voulez minimiser les dépendances
- ✅ Performance maximale est critique

### Utilisez l'approche **OpenAI-Compatible** si:
- ✅ Vous avez déjà du code OpenAI existant
- ✅ Vous voulez faciliter la migration entre providers
- ✅ Vous utilisez des frameworks comme LangChain, LlamaIndex
- ✅ Vous voulez une interface standardisée

## 📊 Exemples de Prompts Testés

Les deux notebooks testent les mêmes 10 types de prompts:

1. **Résumé basique** - Résumé standard du texte
2. **Résumé en 3 points** - Format structuré
3. **Bullet points** - Format liste
4. **Pour enfant de 10 ans** - Simplification
5. **Pour expert technique** - Jargon technique
6. **Q&A: Avantages et défis** - Extraction d'information
7. **Q&A: Concepts clés** - Identification de concepts
8. **Extraction JSON** - Format structuré
9. **Style académique** - Style formel
10. **Style tweet** - Format court (280 caractères)

## 🔒 Sécurité

Les notebooks incluent:
- Validation des inputs
- Filtrage de contenu
- Instructions de sécurité dans les prompts
- Protection contre l'injection de prompts

## 📈 Résultats

Chaque exécution génère un fichier JSON avec:
- Texte source
- Prompts utilisés
- Réponses obtenues
- Statistiques de tokens
- Métriques de performance

Fichiers générés:
- `groq_prompt_results.json` (natif)
- `groq_openai_compatible_results.json` (OpenAI-compatible)

## 🎓 Exercices Pratiques

Les notebooks incluent des exercices pour:
1. Créer vos propres prompts personnalisés
2. Optimiser l'usage de tokens
3. Créer des chaînes de prompts complexes
4. Tester la sécurité avec des injections

## 📚 Ressources Supplémentaires

- [Groq Documentation](https://console.groq.com/docs)
- [Groq Quickstart](https://console.groq.com/docs/quickstart)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [Best Practices](https://platform.openai.com/docs/guides/prompt-engineering)

## 📝 Notes

- Les notebooks sont conçus pour être exécutés de manière interactive
- Chaque cellule peut être exécutée indépendamment
- Les résultats peuvent varier selon la température et le modèle
- Un délai de 0.5s est ajouté entre les appels pour éviter le rate limiting

## 🤝 Contribution

Ces notebooks font partie du cours **Python and Gen AI Data Science** de GrowUp AI.

Pour toute question ou suggestion, consultez le repository principal.

---

**Bonne exploration de Groq et du prompt engineering! 🚀🤖**
