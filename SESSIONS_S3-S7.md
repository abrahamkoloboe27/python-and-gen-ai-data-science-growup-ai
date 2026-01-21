# 📚 Sessions S3 à S7 — Guide Complet

Ce document fournit une vue d'ensemble des matériels de cours pour les séances 3 à 7.

## 📋 Table des matières

- [S3 — Statsmodels & analyses statistiques](#s3--statsmodels--analyses-statistiques)
- [S4 — Scikit-learn : Pipeline ML rapide](#s4--scikit-learn--pipeline-ml-rapide)
- [S5 — PyCaret pour prototypage](#s5--pycaret-pour-prototypage)
- [S6 — FastAPI : Serveurs applicatifs](#s6--fastapi--serveurs-applicatifs)
- [S7 — Streamlit : Prototypage d'interface](#s7--streamlit--prototypage-dinterface)

---

## S3 — Statsmodels & analyses statistiques

### 🎯 Objectifs
- Réaliser des tests statistiques simples
- Comprendre et implémenter des régressions linéaires
- Maîtriser les statistiques descriptives avancées
- Effectuer des tests d'hypothèses

### 📁 Fichiers
- **Notebook**: `notebooks/s3_stats.ipynb`

### 📚 Contenu
1. **Statistiques descriptives avancées**
   - Moyennes, médianes, écarts-types
   - Asymétrie (skewness) et aplatissement (kurtosis)
   - Matrices de corrélation

2. **Tests d'hypothèses**
   - Test de normalité (Shapiro-Wilk)
   - Test t de Student
   - Comparaison de moyennes

3. **Régression linéaire (OLS)**
   - Modèle univarié (simple)
   - Modèle multivarié (multiple)
   - Diagnostics de régression
   - Interprétation des résultats

4. **Exercice pratique**
   - Créer un modèle OLS pour prédire la difficulté d'un document
   - Analyser les coefficients et leur significativité
   - Vérifier les hypothèses de la régression

### 🔧 Bibliothèques nécessaires
```bash
pip install statsmodels scipy pandas numpy matplotlib
```

### 🚀 Démarrage rapide
```bash
jupyter notebook notebooks/s3_stats.ipynb
```

---

## S4 — Scikit-learn : Pipeline ML rapide

### 🎯 Objectifs
- Comprendre et utiliser les pipelines scikit-learn
- Maîtriser le preprocessing de données textuelles
- Implémenter une classification binaire baseline
- Évaluer les modèles avec plusieurs métriques

### 📁 Fichiers
- **Notebook**: `notebooks/s4_ml_pipeline.ipynb`

### 📚 Contenu
1. **Préparation des données**
   - Création d'un dataset de classification (FAQ vs Blog)
   - Train/test split stratifié
   - Analyse exploratoire

2. **TF-IDF Vectorization**
   - Conversion texte → vecteurs numériques
   - Paramètres: max_features, ngram_range, min_df

3. **Pipelines scikit-learn**
   - Structure: preprocessing + modèle
   - Trois modèles testés: Logistic Regression, Naive Bayes, Random Forest

4. **Évaluation**
   - Métriques: Accuracy, Precision, Recall, F1-Score
   - Matrices de confusion
   - Validation croisée
   - Classification report

5. **Exercice pratique**
   - Construire un pipeline personnalisé
   - Expérimenter avec différents paramètres
   - Comparer les performances

### 🔧 Bibliothèques nécessaires
```bash
pip install scikit-learn pandas numpy matplotlib
```

### 🚀 Démarrage rapide
```bash
jupyter notebook notebooks/s4_ml_pipeline.ipynb
```

---

## S5 — PyCaret pour prototypage

### 🎯 Objectifs
- Prototyper des modèles ML rapidement
- Comparer automatiquement plusieurs algorithmes
- Optimiser les hyperparamètres facilement
- Exporter et sauvegarder les modèles

### 📁 Fichiers
- **Notebook**: `notebooks/s5_pycaret.ipynb`

### 📚 Contenu
1. **Préparation des données**
   - Dataset avec features textuelles extraites
   - Classification: Technical vs General documents

2. **Configuration PyCaret**
   - Setup automatique avec preprocessing
   - Normalisation, gestion des valeurs manquantes

3. **Comparaison de modèles**
   - Comparaison automatique de 15+ algorithmes
   - Tri par métrique (F1, Accuracy, etc.)

4. **Optimisation**
   - Tuning automatique des hyperparamètres
   - Création d'ensembles (blending, stacking)

5. **Visualisations**
   - Matrice de confusion
   - Courbe ROC
   - Importance des features
   - Courbe Precision-Recall

6. **Export et déploiement**
   - Finalisation du modèle
   - Sauvegarde en pickle
   - Chargement et prédiction

7. **Exercice pratique**
   - AutoML pour une tâche de classification
   - Comparaison et optimisation
   - Export du meilleur modèle

### 🔧 Bibliothèques nécessaires
```bash
pip install pycaret[full]
```

### 🚀 Démarrage rapide
```bash
jupyter notebook notebooks/s5_pycaret.ipynb
```

---

## S6 — FastAPI : Serveurs applicatifs

### 🎯 Objectifs
- Exposer un modèle ML via une API REST
- Implémenter des endpoints de prédiction
- Valider les données avec Pydantic
- Conteneuriser l'application avec Docker

### 📁 Fichiers
```
s6_fastapi/
├── app.py              # Application FastAPI
├── requirements.txt    # Dépendances
├── Dockerfile         # Configuration Docker
└── README.md          # Documentation
```

### 📚 Contenu
1. **Application FastAPI**
   - Endpoints: `/`, `/health`, `/predict`, `/predict/batch`, `/model/info`
   - Validation Pydantic des requêtes
   - Gestion des erreurs

2. **Modèle de prédiction**
   - Simulateur simple (basé sur longueur de texte)
   - Structure pour remplacer par un vrai modèle

3. **Documentation automatique**
   - Swagger UI: `/docs`
   - ReDoc: `/redoc`

4. **Conteneurisation**
   - Dockerfile optimisé
   - Configuration production-ready

### 🔧 Installation
```bash
cd s6_fastapi
pip install -r requirements.txt
```

### 🚀 Démarrage rapide

**Option 1: Local**
```bash
cd s6_fastapi
python app.py
# ou
uvicorn app:app --reload
```

**Option 2: Docker**
```bash
cd s6_fastapi
docker build -t fastapi-text-classifier .
docker run -p 8000:8000 fastapi-text-classifier
```

### 🧪 Tests
```bash
# Health check
curl http://localhost:8000/health

# Prédiction simple
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Comment réinitialiser mon mot de passe ?"}'

# Prédiction batch
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Question 1?", "Article long..."]}'
```

### 📖 Documentation
Accédez à la documentation interactive:
- http://localhost:8000/docs (Swagger)
- http://localhost:8000/redoc (ReDoc)

---

## S7 — Streamlit : Prototypage d'interface

### 🎯 Objectifs
- Construire un dashboard interactif
- Uploader et indexer des documents CSV
- Implémenter une recherche par mots-clés
- Créer des visualisations Plotly

### 📁 Fichiers
```
s7_streamlit/
├── app.py              # Application Streamlit
├── requirements.txt    # Dépendances
└── README.md          # Documentation
```

### 📚 Contenu
1. **Interface utilisateur**
   - Sidebar avec upload de fichiers
   - Configuration de la recherche
   - Statistiques en temps réel

2. **Fonctionnalités**
   - Upload CSV
   - Indexation des documents
   - Recherche (3 modes: Contient, Mots-clés, Regex)
   - Affichage des résultats (Cartes ou Tableau)
   - Export CSV

3. **Visualisations**
   - Histogrammes avec Plotly
   - Métriques interactives
   - Graphiques dynamiques

4. **Composants utilisés**
   - `st.sidebar`, `st.container`, `st.expander`
   - `st.file_uploader`, `st.text_input`, `st.selectbox`
   - `st.dataframe`, `st.plotly_chart`
   - `st.session_state` pour la gestion d'état

### 🔧 Installation
```bash
cd s7_streamlit
pip install -r requirements.txt
```

### 🚀 Démarrage rapide
```bash
cd s7_streamlit
streamlit run app.py
```

L'application s'ouvre automatiquement sur http://localhost:8501

### 📊 Format du fichier CSV
```csv
text,category,date
"Texte du document 1","FAQ","2024-01-01"
"Texte du document 2","Blog","2024-01-02"
```

---

## 🔄 Workflow complet

Voici comment utiliser ces matériels dans un workflow de bout en bout:

1. **S3**: Analyser statistiquement vos données
   - Comprendre les distributions
   - Tester des hypothèses
   - Modèle OLS pour features importance

2. **S4**: Construire un pipeline ML baseline
   - Preprocessing automatisé
   - Classification avec TF-IDF
   - Évaluation rigoureuse

3. **S5**: Prototypage rapide avec PyCaret
   - Comparer 15+ algorithmes
   - Optimiser automatiquement
   - Exporter le meilleur modèle

4. **S6**: Servir le modèle via API
   - Charger le modèle PyCaret/sklearn
   - Exposer via FastAPI
   - Conteneuriser avec Docker

5. **S7**: Créer une interface démo
   - Dashboard Streamlit
   - Appel de l'API FastAPI
   - Visualisation des résultats

---

## 🛠️ Installation globale

Pour installer toutes les dépendances d'un coup:

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer toutes les dépendances
pip install statsmodels scipy scikit-learn pycaret[full] \
            fastapi uvicorn[standard] streamlit plotly \
            pandas numpy matplotlib jupyter
```

---

## 📝 Exercices transversaux

### Projet intégré: Système de classification de documents

1. **Données** (S3):
   - Analyser statistiquement un corpus de documents
   - Identifier les features importantes

2. **Modèle** (S4-S5):
   - Créer un pipeline scikit-learn
   - Comparer avec PyCaret AutoML
   - Exporter le meilleur modèle

3. **API** (S6):
   - Charger le modèle dans FastAPI
   - Implémenter prédiction + feedback
   - Tester avec curl

4. **Interface** (S7):
   - Créer un dashboard Streamlit
   - Intégrer l'API FastAPI
   - Permettre upload, classification, visualisation

---

## 🔗 Ressources supplémentaires

### Documentation officielle
- [Statsmodels](https://www.statsmodels.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [PyCaret](https://pycaret.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://docs.streamlit.io/)

### Tutoriels recommandés
- [Real Python - Statistical Tests](https://realpython.com/python-statistics/)
- [Kaggle - ML Pipeline Tutorial](https://www.kaggle.com/learn/intro-to-machine-learning)
- [PyCaret Official Tutorials](https://pycaret.org/tutorial/)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [Streamlit Gallery](https://streamlit.io/gallery)

---

## 💡 Conseils pédagogiques

### Pour les instructeurs

1. **Ordre recommandé**:
   - Suivre l'ordre S3 → S4 → S5 → S6 → S7
   - Chaque séance construit sur la précédente

2. **Durée suggérée**:
   - S3: 2-3 heures (stats + régression)
   - S4: 2-3 heures (pipelines + évaluation)
   - S5: 1.5-2 heures (AutoML rapide)
   - S6: 2-3 heures (API + Docker)
   - S7: 2-3 heures (Interface + intégration)

3. **Points d'attention**:
   - S3: Bien expliquer les p-values et R²
   - S4: Insister sur la validation train/test
   - S5: Montrer les limites de l'AutoML
   - S6: Sécurité et bonnes pratiques
   - S7: UX et performance

### Pour les étudiants

1. **Prérequis**:
   - Python de base
   - Pandas et NumPy
   - Notions de ML (optionnel pour S3)

2. **Méthode d'apprentissage**:
   - Exécuter tous les exemples
   - Modifier les paramètres
   - Faire les exercices
   - Créer son propre projet

3. **Ressources d'aide**:
   - Documentation officielle
   - Stack Overflow
   - GitHub Issues des bibliothèques
   - Forums communautaires

---

## ✅ Checklist de complétion

Pour chaque séance, vérifiez que vous pouvez:

### S3 - Statsmodels
- [ ] Calculer des statistiques descriptives
- [ ] Effectuer un test d'hypothèse
- [ ] Construire un modèle OLS
- [ ] Interpréter les résultats

### S4 - Scikit-learn
- [ ] Créer un pipeline complet
- [ ] Vectoriser du texte avec TF-IDF
- [ ] Évaluer un modèle de classification
- [ ] Faire une validation croisée

### S5 - PyCaret
- [ ] Configurer PyCaret
- [ ] Comparer plusieurs modèles
- [ ] Optimiser un modèle
- [ ] Exporter un modèle

### S6 - FastAPI
- [ ] Créer une API REST
- [ ] Implémenter des endpoints
- [ ] Valider les données d'entrée
- [ ] Conteneuriser avec Docker

### S7 - Streamlit
- [ ] Créer une interface interactive
- [ ] Uploader et traiter des fichiers
- [ ] Créer des visualisations
- [ ] Gérer l'état de l'application

---

## 🐛 Dépannage

### Problèmes courants

**ImportError: No module named 'X'**
```bash
pip install X
```

**Jupyter kernel not found**
```bash
python -m ipykernel install --user
```

**Port déjà utilisé (FastAPI/Streamlit)**
```bash
# FastAPI
uvicorn app:app --port 8001

# Streamlit
streamlit run app.py --server.port 8502
```

**Problèmes de mémoire (PyCaret)**
```python
# Réduire le dataset
df_sample = df.sample(n=1000)
```

---

## 📧 Support

Pour toute question ou problème:
1. Consultez d'abord les README spécifiques
2. Vérifiez la documentation officielle
3. Recherchez sur Stack Overflow
4. Ouvrez une issue sur GitHub (si applicable)

---

## 📜 Licence

Ce matériel pédagogique est fourni à des fins éducatives.
Les bibliothèques utilisées ont leurs propres licences (voir documentation respective).

---

**Bon apprentissage! 🚀**
