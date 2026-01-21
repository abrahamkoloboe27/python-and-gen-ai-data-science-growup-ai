# Séance 7 — Streamlit : Interface de prototypage

## 🎯 Objectifs
- Construire un dashboard interactif pour démontrer des modèles ML
- Utiliser les composants Streamlit (sidebar, container, expander)
- Intégrer des visualisations Plotly
- Créer une interface de recherche de documents

---

## 📁 Structure du projet

```
s7_streamlit/
├── app.py              # Application Streamlit principale
├── requirements.txt    # Dépendances Python
└── README.md          # Ce fichier
```

---

## 🚀 Installation et démarrage

### Installation locale

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
streamlit run app.py
```

4. **Accéder à l'application**:
   - L'application s'ouvre automatiquement dans votre navigateur
   - URL par défaut: http://localhost:8501

---

## 📋 Fonctionnalités

### 1. Upload de fichiers CSV
- Glissez-déposez ou sélectionnez un fichier CSV
- Format supporté: CSV avec au moins une colonne de texte
- Aperçu automatique des données chargées

### 2. Indexation des documents
- Cliquez sur "Indexer les documents" pour préparer la recherche
- Création d'un index inversé pour une recherche rapide
- Indicateur visuel de l'état d'indexation

### 3. Recherche de documents
Plusieurs modes de recherche disponibles:

**Mode "Contient"**: Recherche de sous-chaînes
```
Exemple: "machine learning"
Trouve: documents contenant exactement "machine learning"
```

**Mode "Mots-clés"**: Recherche par mots exacts
```
Exemple: "python data"
Trouve: documents contenant les deux mots "python" ET "data"
```

**Mode "Regex"**: Recherche avec expressions régulières
```
Exemple: "python|java"
Trouve: documents contenant "python" OU "java"
```

### 4. Affichage des résultats

**Mode Cartes**: Affichage enrichi
- Texte du document (preview de 500 caractères)
- Métadonnées disponibles
- Mise en page claire et lisible

**Mode Tableau**: Affichage complet
- Toutes les colonnes visibles
- Tri interactif
- Filtrage facile

### 5. Visualisations
- Histogrammes automatiques pour les colonnes numériques
- Graphiques interactifs avec Plotly
- Statistiques descriptives

### 6. Export des résultats
- Téléchargement des résultats au format CSV
- Nom de fichier horodaté
- Conservation de toutes les colonnes

---

## 📊 Format du fichier CSV

Votre fichier doit contenir au minimum:
- Une colonne avec du texte (nommée 'text', 'content', ou autre)
- Optionnellement: d'autres colonnes avec des métadonnées

### Exemple de structure

```csv
text,category,date,author
"Comment réinitialiser mon mot de passe ?","FAQ","2024-01-01","Support"
"Dans cet article, nous explorons l'IA","Blog","2024-01-02","Jean Dupont"
"Quels sont vos horaires ?","FAQ","2024-01-03","Support"
```

### Créer un fichier de test

Vous pouvez utiliser ce code Python pour créer un fichier de test:

```python
import pandas as pd

data = {
    'text': [
        "Comment réinitialiser mon mot de passe ?",
        "Dans cet article, nous explorons l'intelligence artificielle",
        "Quels sont vos horaires d'ouverture ?",
        "Le machine learning révolutionne l'industrie",
    ],
    'category': ['FAQ', 'Blog', 'FAQ', 'Blog'],
    'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04']
}

df = pd.DataFrame(data)
df.to_csv('mes_documents.csv', index=False)
```

---

## 🎨 Composants Streamlit utilisés

### Layout
- `st.sidebar`: Barre latérale pour les contrôles
- `st.container()`: Conteneurs pour organiser le contenu
- `st.expander()`: Sections pliables/dépliables
- `st.columns()`: Disposition en colonnes

### Widgets d'entrée
- `st.file_uploader()`: Upload de fichiers
- `st.text_input()`: Champ de texte
- `st.selectbox()`: Liste déroulante
- `st.checkbox()`: Case à cocher
- `st.button()`: Boutons
- `st.radio()`: Boutons radio

### Affichage
- `st.dataframe()`: Tableaux interactifs
- `st.metric()`: Métriques/indicateurs
- `st.plotly_chart()`: Graphiques Plotly
- `st.markdown()`: Texte formaté en Markdown
- `st.success()`, `st.info()`, `st.warning()`, `st.error()`: Messages colorés

### Interaction
- `st.spinner()`: Indicateur de chargement
- `st.download_button()`: Téléchargement de fichiers
- `st.session_state`: Gestion de l'état

---

## 🔧 Personnalisation

### Modifier le thème

Créez un fichier `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#F63366"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Ajouter des fonctionnalités

**1. Intégration d'un modèle ML**:
```python
import joblib

# Charger le modèle
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

# Prédiction
if st.button("Prédire"):
    prediction = model.predict([text])[0]
    st.success(f"Prédiction: {prediction}")
```

**2. Connexion à une base de données**:
```python
import sqlite3

@st.cache_resource
def get_connection():
    return sqlite3.connect('database.db')

conn = get_connection()
df = pd.read_sql_query("SELECT * FROM documents", conn)
```

**3. Ajout de graphiques avancés**:
```python
import plotly.graph_objects as go

fig = go.Figure(data=[
    go.Bar(x=df['category'], y=df['count'])
])
fig.update_layout(title="Documents par catégorie")
st.plotly_chart(fig)
```

---

## 📝 Exercices

### Exercice 1: Améliorer la recherche
- Ajouter la recherche par similarité (TF-IDF + cosine similarity)
- Implémenter le highlight des mots-clés dans les résultats
- Ajouter des suggestions de recherche

### Exercice 2: Statistiques avancées
- Créer un dashboard avec plusieurs KPIs
- Ajouter des graphiques de distribution (box plots, violin plots)
- Implémenter des filtres interactifs par date, catégorie, etc.

### Exercice 3: Intégration ML
- Charger un modèle de classification entraîné
- Ajouter un endpoint de prédiction en temps réel
- Afficher les probabilités et l'importance des features

### Exercice 4: Cache et performance
- Utiliser `@st.cache_data` pour le chargement des données
- Optimiser l'indexation pour de gros volumes
- Implémenter la pagination des résultats

### Exercice 5: Multi-pages
- Créer plusieurs pages (recherche, statistiques, admin)
- Utiliser `st.navigation()` pour la navigation
- Organiser le code en modules

---

## 🚀 Déploiement

### Streamlit Cloud

1. Pushez votre code sur GitHub
2. Connectez-vous à [share.streamlit.io](https://share.streamlit.io)
3. Déployez votre application en quelques clics
4. Gratuit pour les projets publics!

### Docker

Créez un `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Construisez et lancez:
```bash
docker build -t streamlit-app .
docker run -p 8501:8501 streamlit-app
```

### Heroku

Créez un `Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

Déployez:
```bash
heroku create
git push heroku main
```

---

## 💡 Bonnes pratiques

1. **Performance**:
   - Utilisez `@st.cache_data` pour les données
   - Utilisez `@st.cache_resource` pour les modèles
   - Évitez les calculs lourds à chaque interaction

2. **UX/UI**:
   - Utilisez des spinners pour les opérations longues
   - Donnez du feedback à l'utilisateur (success, warning, error)
   - Organisez le contenu avec des expandeurs

3. **Code**:
   - Séparez la logique en fonctions
   - Utilisez le session_state pour l'état global
   - Documentez vos fonctions

4. **Sécurité**:
   - Validez les fichiers uploadés
   - Ne stockez pas de données sensibles dans le code
   - Utilisez secrets.toml pour les credentials

---

## 🔗 Ressources

- [Documentation Streamlit](https://docs.streamlit.io/)
- [Galerie d'exemples](https://streamlit.io/gallery)
- [Composants communautaires](https://streamlit.io/components)
- [Cheat sheet](https://docs.streamlit.io/library/cheatsheet)
- [Forum communautaire](https://discuss.streamlit.io/)

---

## 🐛 Dépannage

### L'application ne démarre pas
```bash
# Vérifier l'installation
pip list | grep streamlit

# Réinstaller si nécessaire
pip install --upgrade streamlit
```

### Erreur de port
```bash
# Spécifier un port différent
streamlit run app.py --server.port 8502
```

### Problèmes de cache
```bash
# Vider le cache
streamlit cache clear
```

---

## ✅ Checklist

Avant de partager votre application:

- [ ] Testé avec différents fichiers CSV
- [ ] Gestion des erreurs implémentée
- [ ] Documentation complète
- [ ] Code commenté et organisé
- [ ] Thème personnalisé (optionnel)
- [ ] Performance optimisée
- [ ] Tests avec de gros fichiers
- [ ] Interface responsive
- [ ] Messages d'aide clairs
- [ ] Export fonctionnel

---

## 🎓 Pour aller plus loin

- Intégrer des modèles de deep learning (Hugging Face)
- Ajouter un système de RAG (Retrieval-Augmented Generation)
- Créer des visualisations 3D avec Plotly
- Implémenter l'authentification
- Ajouter du streaming de données en temps réel
- Créer des composants Streamlit personnalisés
