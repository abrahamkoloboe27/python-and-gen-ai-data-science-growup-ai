# 📄 RAG PDF Chat — LangChain + OpenAI + Streamlit

Application interactive permettant de **chatter avec n'importe quel document PDF** grâce à la technique RAG (**Retrieval-Augmented Generation**), LangChain et OpenAI.

---

## 🎯 Fonctionnalités

- 📂 **Upload PDF** — Téléchargez n'importe quel PDF directement depuis l'interface
- 💬 **Chat conversationnel** — Posez des questions en langage naturel sur le document
- 🧠 **Mémoire** — La conversation garde le contexte des échanges précédents
- 📚 **Sources citées** — Chaque réponse affiche les extraits utilisés et les numéros de page
- ⚡ **Streaming** — Les réponses s'affichent en temps réel
- ⚙️ **Paramètres ajustables** — Configurez tout depuis la barre latérale

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   PHASE D'INDEXATION                    │
│                                                         │
│  PDF ──▶ PyPDFLoader ──▶ TextSplitter ──▶ Embeddings   │
│                                              │          │
│                                         FAISS Index     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    PHASE DE REQUÊTE                     │
│                                                         │
│  Question ──▶ Embeddings ──▶ Retriever ──▶ Top-K docs  │
│                                              │          │
│  Réponse ◀── ChatOpenAI ◀── Prompt ◀──────  │          │
│                          (contexte + hist.)             │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation & Lancement

### 1. Cloner / naviguer dans le dossier

```bash
cd s13_langchain_rag_pdf
```

### 2. Créer un environnement virtuel (recommandé)

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer la clé API OpenAI

```bash
# Copier le fichier d'exemple
cp .env.example .env

# Éditer .env et renseigner votre clé API
nano .env   # ou ouvrez le fichier dans votre éditeur
```

Contenu du `.env` :
```env
OPENAI_API_KEY=sk-votre-clé-api-openai
```

> **Astuce** : Vous pouvez aussi saisir la clé directement dans la barre latérale de l'application.

### 5. Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvre automatiquement dans votre navigateur sur `http://localhost:8501`.

---

## 🖥️ Utilisation

### Étape 1 — Configurer (optionnel)
Ouvrez la barre latérale (⚙️) et ajustez les paramètres selon vos besoins.

### Étape 2 — Uploader votre PDF
Glissez-déposez ou cliquez sur **"Téléchargez votre document PDF"**.

### Étape 3 — Chatter
Tapez votre question dans la barre en bas et appuyez sur Entrée.

---

## ⚙️ Paramètres de la barre latérale

| Paramètre | Description | Valeur par défaut |
|-----------|-------------|-------------------|
| **Modèle OpenAI** | LLM pour générer les réponses | `gpt-4o-mini` |
| **Modèle Embeddings** | Pour vectoriser le texte | `text-embedding-3-small` |
| **Température** | Créativité des réponses (0 = précis) | `0.0` |
| **Type de RAG** | Stratégie de recherche | `Naïf (Basic RAG)` |
| **Top-k** | Nombre d'extraits récupérés | `4` |
| **Seuil de score** | (Similarity + Score) Score minimum pour inclure un extrait | `0.5` |
| **Taille des chunks** | Taille de chaque extrait en caractères | `800` |
| **Chevauchement** | Overlap entre chunks consécutifs | `100` |
| **Tours de mémoire** | Nombre de tours de conversation conservés | `5` |

---

## 🔍 Types de RAG disponibles

| Type | Description | Avantage |
|------|-------------|---------|
| **Naïf (Basic RAG)** | Recherche par similarité cosinus | Simple et rapide |
| **MMR (Diversifié)** | Max Marginal Relevance | Évite les résultats redondants |
| **Similarity + Score** | Filtre par score minimum | Résultats plus précis |

---

## 📁 Structure du projet

```
s13_langchain_rag_pdf/
├── app.py              # Application Streamlit principale
├── requirements.txt    # Dépendances Python
├── .env.example        # Template de configuration
└── README.md           # Ce fichier
```

---

## 💡 Conseils d'utilisation

### Pour de meilleurs résultats

- **Chunk size** : 600–1000 chars pour documents techniques, 300–600 pour FAQ
- **Top-k** : Augmentez à 5–8 si les réponses sont incomplètes
- **Température** : Gardez à 0.0 pour des réponses factuelles
- **Type de RAG** : Utilisez MMR si les réponses sont répétitives

### Exemples de questions

- *"Quel est le sujet principal de ce document ?"*
- *"Résume les points clés du chapitre 3"*
- *"Quelles sont les recommandations mentionnées ?"*
- *"Explique le concept de [terme] mentionné page X"*

---

## 🔒 Sécurité & Confidentialité

- ✅ Les PDFs sont traités localement (non envoyés à OpenAI bruts)
- ✅ Seuls les **extraits pertinents** sont transmis au LLM
- ✅ La clé API reste dans votre environnement local
- ⚠️ Ne partagez pas votre `.env` dans un dépôt public

---

## 🐛 Dépannage

### "API key not found"
→ Vérifiez que votre `.env` contient `OPENAI_API_KEY=sk-...` ou saisissez-la dans la sidebar.

### "Error processing PDF"
→ Vérifiez que le PDF n'est pas protégé par mot de passe ou corrompu.

### Réponses hors sujet
→ Réduisez la température à 0.0 et assurez-vous que le chunk_size est adapté.

### Réponses incomplètes
→ Augmentez `top_k` à 6–8 pour récupérer plus de contexte.

### Application lente
→ Utilisez `gpt-4o-mini` + `text-embedding-3-small` pour un meilleur rapport vitesse/qualité.

---

## 📚 Technologies utilisées

| Technologie | Version | Rôle |
|-------------|---------|------|
| **Streamlit** | ≥1.32 | Interface utilisateur |
| **LangChain** | ≥0.2 | Orchestration RAG |
| **LangChain-OpenAI** | ≥0.1 | Intégration OpenAI |
| **FAISS** | ≥1.7 | Index vectoriel |
| **PyPDF** | ≥4.0 | Extraction PDF |
| **OpenAI** | ≥1.0 | LLM + Embeddings |

---

*Projet RAG PDF Chat — Session S13 LangChain | GrowUP AI 🚀*
