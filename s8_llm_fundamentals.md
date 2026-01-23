# S8 — Fondamentaux LLM & Transformers

## 📚 Note Technique : Concepts Essentiels

### 🎯 Objectifs de la séance
- Comprendre l'architecture Transformer
- Maîtriser les concepts d'embeddings et tokens
- Comprendre le context window et ses implications
- Apprendre les hyperparamètres clés (temperature, top-k, etc.)

---

## 1. Architecture Transformer

### 1.1 Principe de base
Les **Transformers** sont une architecture de réseaux de neurones introduite en 2017 (Vaswani et al., "Attention is All You Need"). Ils ont révolutionné le traitement du langage naturel.

**Composants clés:**
- **Self-Attention**: Mécanisme permettant au modèle de pondérer l'importance de chaque mot par rapport aux autres
- **Multi-Head Attention**: Plusieurs mécanismes d'attention en parallèle pour capturer différentes relations
- **Feed-Forward Networks**: Couches fully-connected après l'attention
- **Positional Encoding**: Injection d'information sur la position des tokens
- **Layer Normalization**: Normalisation pour stabiliser l'entraînement

### 1.2 Encodeur vs Décodeur
- **Encodeur**: Traite l'input et produit des représentations contextuelles (ex: BERT)
- **Décodeur**: Génère l'output de manière autorégressive (ex: GPT)
- **Encodeur-Décodeur**: Combinaison pour tâches de traduction (ex: T5, BART)

### 1.3 Architecture GPT (Generative Pre-trained Transformer)
```
Input → Tokenization → Embeddings → 
→ Positional Encoding → 
→ N × [Multi-Head Attention → Feed-Forward] → 
→ Output Layer (Logits) → 
→ Sampling → Generated Text
```

---

## 2. Tokenisation

### 2.1 Qu'est-ce qu'un token?
Un **token** est l'unité de base traitée par un LLM. Ce n'est pas toujours un mot!

**Types de tokenisation:**
- **Caractères**: Chaque caractère = 1 token (ex: "chat" = 4 tokens)
- **Mots**: Chaque mot = 1 token (vocabulaire énorme)
- **Subword**: Compromis intelligent (BPE, WordPiece, SentencePiece)

### 2.2 Exemple avec GPT (BPE - Byte Pair Encoding)
```
Texte: "tokenization"
Tokens: ["token", "ization"]  # 2 tokens

Texte: "L'intelligence artificielle"
Tokens: ["L", "'", "intelligence", " art", "ific", "ielle"]  # 6 tokens
```

### 2.3 Implications pratiques
- **Coût**: Facturé au nombre de tokens (input + output)
- **Context window**: Limité en tokens, pas en caractères
- **Performance**: Mots rares = plusieurs tokens = moins efficace

**Règle approximative**: 
- Anglais: ~1 token = 4 caractères = 0.75 mots
- Français: ~1 token = 3-3.5 caractères (plus de tokens pour même texte)

---

## 3. Embeddings vs Logits

### 3.1 Embeddings
Les **embeddings** sont des représentations vectorielles denses de tokens.

```python
# Exemple conceptuel
token_id = 1234
embedding = embedding_layer[token_id]  # Vector de dimension 768, 1024, 4096...
# embedding = [0.12, -0.34, 0.56, ..., 0.23]
```

**Caractéristiques:**
- Dimension fixe (ex: 768 pour BERT-base, 12288 pour GPT-4)
- Tokens similaires ont embeddings proches (distance cosinus)
- Appris pendant l'entraînement
- Contextuels (pour Transformers): même mot a différents embeddings selon contexte

### 3.2 Logits
Les **logits** sont les scores bruts de sortie avant softmax.

```python
# Dernier layer du modèle
logits = model_output  # Vector de taille = taille vocabulaire (ex: 50,000)
# logits = [2.3, -1.2, 5.6, ..., 0.8]

# Conversion en probabilités
probs = softmax(logits)
# probs = [0.002, 0.0001, 0.05, ..., 0.001]  # Somme = 1
```

**Usage:**
- **Génération**: Sélection du prochain token via sampling des logits
- **Classification**: Argmax des logits pour prédire la classe

### 3.3 Flow complet
```
Input text → Tokens → Embeddings → 
→ Transformer Layers → 
→ Logits → Sampling → Next Token → 
→ Repeat (autoregressive)
```

---

## 4. Context Window

### 4.1 Définition
Le **context window** est la quantité maximale de tokens que le modèle peut traiter simultanément.

**Exemples:**
- GPT-3.5-turbo: 4,096 tokens (~3,000 mots)
- GPT-4: 8,192 tokens (version standard)
- GPT-4-32k: 32,768 tokens (~24,000 mots)
- Claude 2: 100,000 tokens (~75,000 mots)

### 4.2 Implications
**Limites:**
- Input + Output ≤ Context Window
- Au-delà: tokens anciens sont oubliés (truncation)
- Mémoire limitée pour conversations longues

**Stratégies:**
- **Summarization**: Résumer l'historique
- **Chunking**: Découper les documents longs
- **Sliding window**: Garder les N derniers tokens
- **Retrieval**: Chercher l'info pertinente (RAG)

### 4.3 Coût vs Context Window
Plus le context window est grand:
- Plus le coût est élevé (computation O(n²) avec self-attention)
- Plus la latence augmente
- Meilleure cohérence sur longs textes

---

## 5. Hyperparamètres de Génération

### 5.1 Temperature
Contrôle la "créativité" du modèle en ajustant les probabilités.

```python
# Avant sampling
logits = [2.0, 1.0, 0.5]

# Avec temperature = 1.0 (par défaut)
probs = softmax(logits / 1.0) = [0.59, 0.24, 0.17]

# Avec temperature = 0.1 (plus déterministe)
probs = softmax(logits / 0.1) = [0.84, 0.11, 0.05]

# Avec temperature = 2.0 (plus aléatoire)
probs = softmax(logits / 2.0) = [0.46, 0.30, 0.24]
```

**Usage:**
- **Temperature basse (0.1-0.5)**: Réponses précises, factuelles, répétables
- **Temperature moyenne (0.7-1.0)**: Équilibre créativité/cohérence
- **Temperature haute (1.5-2.0)**: Créatif, varié, potentiellement incohérent

### 5.2 Top-k Sampling
Limite le sampling aux k tokens les plus probables.

```python
# Probabilités: [0.4, 0.3, 0.15, 0.10, 0.05]
# Top-k = 3
# Sample uniquement parmi: [0.4, 0.3, 0.15]
# Renormalisation: [0.47, 0.35, 0.18]
```

**Effet:** Évite de choisir des tokens très improbables.

### 5.3 Top-p (Nucleus Sampling)
Limite le sampling aux tokens dont la probabilité cumulative ≤ p.

```python
# Probabilités: [0.4, 0.3, 0.15, 0.10, 0.05]
# Top-p = 0.8
# Cumul: [0.4, 0.7, 0.85, 0.95, 1.0]
# Sample parmi les 3 premiers (0.4 + 0.3 + 0.15 = 0.85 > 0.8)
```

**Effet:** Adaptif selon la distribution (plus intelligent que top-k).

### 5.4 Autres paramètres
- **max_tokens**: Nombre maximum de tokens générés
- **frequency_penalty**: Pénalise les répétitions (0 à 2)
- **presence_penalty**: Encourage nouveaux topics (0 à 2)
- **stop_sequences**: Liste de tokens pour arrêter la génération

---

## 6. Coût et Latence

### 6.1 Facteurs de coût
**Tarification OpenAI (exemple GPT-4):**
- Input: $0.03 / 1K tokens
- Output: $0.06 / 1K tokens

**Calcul:**
```
Prompt: 1,000 tokens → $0.03
Réponse: 500 tokens → $0.03
Total: $0.06 par requête
```

**Optimisations:**
- Modèles moins chers (GPT-3.5: 10x moins cher)
- Prompts plus courts
- Caching de prompts systèmes
- Batch requests

### 6.2 Latence
**Time to First Token (TTFT):**
- Temps avant le premier token généré
- Dépend de: taille du prompt, charge du serveur

**Throughput:**
- Tokens par seconde
- ~20-50 tokens/sec pour GPT-4
- ~100+ tokens/sec pour GPT-3.5

**Tradeoffs:**
- **Context window large**: Plus lent, plus cher, meilleure qualité
- **Temperature basse**: Plus rapide (moins de sampling), moins varié
- **Streaming**: Meilleure UX (tokens progressifs) mais même coût

---

## 7. Embeddings pour Retrieval

### 7.1 Embeddings de texte
Contrairement aux embeddings de tokens, les **text embeddings** représentent des phrases/documents entiers.

**Modèles populaires:**
- OpenAI: text-embedding-3-small (1536 dimensions)
- Sentence-BERT: Multi-lingual embeddings
- Cohere: embed-multilingual-v3.0

### 7.2 Utilisation
```python
# Pseudo-code
text = "Comment réinitialiser mon mot de passe?"
embedding = get_embedding(text)  # [0.12, -0.34, ..., 0.56] (1536 dims)

# Similarity search
query_emb = get_embedding("reset password")
similarity = cosine_similarity(embedding, query_emb)  # 0.87 (très similaire)
```

### 7.3 Propriétés
- **Sémantique**: Textes similaires en sens = embeddings proches
- **Multilingue**: Certains modèles alignent plusieurs langues
- **Fixed dimension**: Tous les textes → même dimension

---

## 8. Conclusion

### Points clés à retenir
1. **Transformers** = architecture à base d'attention, base des LLMs modernes
2. **Tokens** = unités de traitement (≠ mots), impactent coût et capacités
3. **Embeddings** = représentations vectorielles denses
4. **Logits** = scores bruts avant sampling
5. **Context window** = mémoire limitée du modèle
6. **Temperature/Top-k/Top-p** = contrôle de la génération
7. **Coût** = fonction du nombre de tokens (input + output)
8. **Latence** = tradeoff avec qualité et context window

### Ressources
- [Attention Is All You Need (paper)](https://arxiv.org/abs/1706.03762)
- [OpenAI Tokenizer](https://platform.openai.com/tokenizer)
- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

---

## 📝 Exercice Pratique

### Partie 1: Analyse de tokenisation
1. Utilisez l'outil [OpenAI Tokenizer](https://platform.openai.com/tokenizer)
2. Testez différents textes (anglais vs français)
3. Calculez le ratio caractères/tokens

### Partie 2: Expérimentation avec prompts
Créez 3 prompts expérimentaux documentant l'impact des hyperparamètres:

#### Prompt 1: Impact de la temperature
**Tâche**: Génération créative
**Prompt**: "Écris une histoire courte sur un robot qui découvre l'art."

Tester:
- Temperature = 0.2
- Temperature = 1.0
- Temperature = 1.8

**Observations**: [À documenter]

#### Prompt 2: Top-k vs Top-p
**Tâche**: Réponse technique
**Prompt**: "Explique en 3 phrases comment fonctionne le gradient descent."

Tester:
- Top-k = 10
- Top-p = 0.9
- Sans restriction

**Observations**: [À documenter]

#### Prompt 3: Context window
**Tâche**: Résumé
**Prompt**: [Insérer un texte long de 2000 mots]
"Résume ce texte en 100 mots."

Tester avec:
- Texte complet (dans la limite)
- Texte tronqué (au-delà de la limite)

**Observations**: [À documenter]

---

**Document créé pour la séance S8 — Fondamentaux LLM & Transformers**
