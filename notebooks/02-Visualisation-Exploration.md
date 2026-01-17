# Séance 2 — Visualisation rapide & exploration (Matplotlib / Plotly)

## 🎯 Objectifs
- Visualiser des distributions et des séries temporelles pour diagnostics
- Comprendre quand utiliser chaque type de graphique
- Apprendre à interpréter les visualisations
- Créer des graphiques interactifs avec Plotly Express

---

## 📚 Introduction

La visualisation de données est essentielle pour comprendre rapidement vos données, détecter des anomalies, et communiquer vos résultats. Dans cette séance, nous allons explorer les graphiques les plus courants et apprendre à les utiliser efficacement.

**Bibliothèques nécessaires :**
```python
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
```

**Installation :**
```bash
pip install pandas numpy plotly matplotlib
```

---

## 🔄 Dataset d'exemple

Pour cette séance, nous utiliserons un jeu de données simple sur les ventes. Créons-le directement :

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Créer un dataset de ventes
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')

data = {
    'date': dates,
    'ventes': np.random.normal(1000, 200, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 300,
    'produit': np.random.choice(['A', 'B', 'C', 'D'], len(dates)),
    'region': np.random.choice(['Nord', 'Sud', 'Est', 'Ouest'], len(dates)),
    'prix': np.random.uniform(10, 100, len(dates))
}

df = pd.DataFrame(data)
df['ventes'] = df['ventes'].clip(lower=0)  # Pas de ventes négatives
print(df.head())
print(f"\nDimensions: {df.shape}")
```

---

## 📊 1. Histogrammes

### 📌 À quoi ça sert ?
Les histogrammes montrent la **distribution d'une variable numérique**. Ils permettent de voir :
- La forme de la distribution (normale, asymétrique, etc.)
- Les valeurs les plus fréquentes
- La présence de valeurs extrêmes

### 🔍 Quand l'utiliser ?
- Pour comprendre la distribution d'une variable
- Pour détecter des anomalies ou valeurs aberrantes
- Pour vérifier si vos données suivent une distribution normale

### 💻 Exemple avec Plotly

```python
import plotly.express as px

# Histogramme simple
fig = px.histogram(df, 
                   x='ventes', 
                   nbins=30,
                   title='Distribution des ventes',
                   labels={'ventes': 'Montant des ventes (€)'},
                   color_discrete_sequence=['#636EFA'])

fig.update_layout(
    xaxis_title='Ventes (€)',
    yaxis_title='Fréquence',
    showlegend=False
)

fig.show()
```

### 🎨 Histogramme par catégorie

```python
# Histogramme avec plusieurs catégories
fig = px.histogram(df, 
                   x='ventes', 
                   color='produit',
                   nbins=25,
                   title='Distribution des ventes par produit',
                   barmode='overlay',  # 'overlay', 'group', ou 'stack'
                   opacity=0.7)

fig.show()
```

### 📖 Comment interpréter ?
- **Pic unique** : Distribution normale, données cohérentes
- **Plusieurs pics** : Présence de sous-groupes différents
- **Asymétrie** : Données biaisées (ex: revenus, souvent asymétriques à droite)
- **Valeurs isolées** : Outliers potentiels à investiguer

---

## 📦 2. Boxplots (Boîtes à moustaches)

### 📌 À quoi ça sert ?
Les boxplots montrent la **distribution et la dispersion** des données à travers 5 statistiques clés :
- Minimum
- Premier quartile (Q1, 25%)
- Médiane (Q2, 50%)
- Troisième quartile (Q3, 75%)
- Maximum

### 🔍 Quand l'utiliser ?
- Pour comparer des distributions entre groupes
- Pour identifier rapidement les outliers
- Pour voir la dispersion et la symétrie des données

### 💻 Exemple avec Plotly

```python
# Boxplot simple
fig = px.box(df, 
             y='ventes',
             title='Distribution des ventes',
             points='outliers')  # 'outliers', 'all', False

fig.update_layout(yaxis_title='Ventes (€)')
fig.show()
```

### 🎨 Boxplot par catégorie

```python
# Comparer plusieurs groupes
fig = px.box(df, 
             x='produit', 
             y='ventes',
             color='produit',
             title='Distribution des ventes par produit',
             points='outliers')

fig.update_layout(
    xaxis_title='Produit',
    yaxis_title='Ventes (€)'
)

fig.show()
```

### 🎨 Boxplot horizontal par région

```python
# Boxplot horizontal
fig = px.box(df, 
             x='ventes', 
             y='region',
             color='region',
             title='Distribution des ventes par région',
             orientation='h')

fig.show()
```

### 📖 Comment interpréter ?

```
        Maximum ──────┐
                      │
        Q3 ────────┐  │
                   │  │
        Médiane ───┼──┤  ← La boîte représente 50% des données
                   │  │
        Q1 ────────┘  │
                      │
        Minimum ──────┘

        • Points isolés = Outliers
```

- **Boîte large** : Données très dispersées
- **Boîte étroite** : Données concentrées
- **Médiane au centre** : Distribution symétrique
- **Médiane décalée** : Distribution asymétrique
- **Points au-delà des moustaches** : Outliers (valeurs > Q3 + 1.5×IQR ou < Q1 - 1.5×IQR)

---

## 📈 3. Séries temporelles

### 📌 À quoi ça sert ?
Les graphiques de séries temporelles montrent l'**évolution d'une variable dans le temps**. Ils permettent de :
- Identifier des tendances
- Détecter la saisonnalité
- Repérer des anomalies temporelles

### 🔍 Quand l'utiliser ?
- Pour analyser l'évolution temporelle
- Pour prévoir des tendances futures
- Pour comparer plusieurs séries temporelles

### 💻 Exemple simple avec Plotly

```python
# Série temporelle simple
fig = px.line(df, 
              x='date', 
              y='ventes',
              title='Évolution des ventes en 2023',
              labels={'date': 'Date', 'ventes': 'Ventes (€)'})

fig.update_layout(
    hovermode='x unified',
    xaxis_title='Date',
    yaxis_title='Ventes (€)'
)

fig.show()
```

### 🎨 Série temporelle avec moyenne mobile

```python
# Ajouter une moyenne mobile pour voir la tendance
df_sorted = df.sort_values('date')
df_sorted['moyenne_mobile_7j'] = df_sorted['ventes'].rolling(window=7).mean()

fig = go.Figure()

# Données brutes
fig.add_trace(go.Scatter(
    x=df_sorted['date'],
    y=df_sorted['ventes'],
    mode='lines',
    name='Ventes quotidiennes',
    line=dict(color='lightblue', width=1),
    opacity=0.5
))

# Moyenne mobile
fig.add_trace(go.Scatter(
    x=df_sorted['date'],
    y=df_sorted['moyenne_mobile_7j'],
    mode='lines',
    name='Moyenne mobile (7 jours)',
    line=dict(color='red', width=2)
))

fig.update_layout(
    title='Ventes avec tendance (moyenne mobile 7 jours)',
    xaxis_title='Date',
    yaxis_title='Ventes (€)',
    hovermode='x unified'
)

fig.show()
```

### 🎨 Comparer plusieurs séries

```python
# Ventes par produit dans le temps
df_produit = df.groupby(['date', 'produit'])['ventes'].sum().reset_index()

fig = px.line(df_produit, 
              x='date', 
              y='ventes',
              color='produit',
              title='Évolution des ventes par produit',
              labels={'date': 'Date', 'ventes': 'Ventes (€)'})

fig.update_layout(hovermode='x unified')
fig.show()
```

### 📖 Comment interpréter ?
- **Tendance croissante/décroissante** : Évolution générale
- **Pics réguliers** : Saisonnalité (ex: ventes élevées en fin de mois)
- **Pics isolés** : Événements ponctuels (promotions, anomalies)
- **Variabilité** : Stabilité ou volatilité des données

---

## 🎯 4. Introduction à Plotly Express interactif

### 📌 Pourquoi Plotly ?
Plotly offre des **graphiques interactifs** qui permettent :
- Zoom et pan
- Hover pour voir les détails
- Sélection de données
- Export d'images
- Facile à partager (HTML)

### 💻 Scatter plot interactif

```python
# Nuage de points avec informations au survol
fig = px.scatter(df, 
                 x='prix', 
                 y='ventes',
                 color='produit',
                 size='ventes',
                 hover_data=['region', 'date'],
                 title='Relation entre prix et ventes',
                 labels={'prix': 'Prix (€)', 'ventes': 'Ventes (€)'})

fig.update_traces(marker=dict(opacity=0.6))
fig.show()
```

### 🎨 Graphique à barres interactif

```python
# Ventes moyennes par région
ventes_region = df.groupby('region')['ventes'].agg(['mean', 'std']).reset_index()

fig = px.bar(ventes_region, 
             x='region', 
             y='mean',
             error_y='std',
             title='Ventes moyennes par région (avec écart-type)',
             labels={'mean': 'Ventes moyennes (€)', 'region': 'Région'},
             color='region')

fig.update_layout(showlegend=False)
fig.show()
```

### 🎨 Heatmap (carte de chaleur)

```python
# Ventes par jour de la semaine et produit
df['jour_semaine'] = df['date'].dt.day_name()
heatmap_data = df.groupby(['produit', 'jour_semaine'])['ventes'].mean().reset_index()
heatmap_pivot = heatmap_data.pivot(index='produit', columns='jour_semaine', values='ventes')

# Réordonner les jours (ordre chronologique)
import calendar
jours_ordre = list(calendar.day_name)
heatmap_pivot = heatmap_pivot.reindex(columns=[j for j in jours_ordre if j in heatmap_pivot.columns])

fig = px.imshow(heatmap_pivot,
                labels=dict(x="Jour de la semaine", y="Produit", color="Ventes moyennes"),
                title='Ventes moyennes par produit et jour de la semaine',
                color_continuous_scale='Viridis')

fig.show()
```

### 🎨 Graphique combiné (subplots)

```python
from plotly.subplots import make_subplots

# Créer une grille de graphiques
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Distribution des ventes', 'Ventes par produit',
                    'Évolution temporelle', 'Prix vs Ventes'),
    specs=[[{"type": "histogram"}, {"type": "box"}],
           [{"type": "scatter"}, {"type": "scatter"}]]
)

# Histogramme
fig.add_trace(
    go.Histogram(x=df['ventes'], name='Ventes', nbinsx=30),
    row=1, col=1
)

# Boxplot
for produit in df['produit'].unique():
    fig.add_trace(
        go.Box(y=df[df['produit']==produit]['ventes'], name=produit),
        row=1, col=2
    )

# Série temporelle
df_sorted = df.sort_values('date')
fig.add_trace(
    go.Scatter(x=df_sorted['date'], y=df_sorted['ventes'], 
               mode='lines', name='Ventes'),
    row=2, col=1
)

# Scatter
fig.add_trace(
    go.Scatter(x=df['prix'], y=df['ventes'], 
               mode='markers', name='Prix vs Ventes', opacity=0.5),
    row=2, col=2
)

fig.update_layout(height=800, showlegend=False, title_text="Dashboard de visualisation")
fig.show()
```

---

## 📋 Résumé : Quel graphique choisir ?

| Type de graphique | Utilisation | Exemple |
|-------------------|-------------|---------|
| **Histogramme** | Distribution d'une variable numérique | Âges, revenus, scores |
| **Boxplot** | Comparer des distributions, détecter outliers | Salaires par département |
| **Série temporelle** | Évolution dans le temps | Ventes mensuelles, cours boursiers |
| **Scatter plot** | Relation entre 2 variables | Prix vs demande |
| **Bar chart** | Comparer des catégories | Ventes par région |
| **Heatmap** | Visualiser une matrice de valeurs | Corrélations, patterns temporels |

---

## 💡 Bonnes pratiques

1. **Toujours titrer vos graphiques** : Un titre clair explique ce que montre le graphique
2. **Labelliser les axes** : Avec les unités (€, %, kg, etc.)
3. **Choisir des couleurs appropriées** : Cohérentes et accessibles
4. **Ne pas surcharger** : Un graphique = un message principal
5. **Interactivité avec Plotly** : Permet d'explorer les données en profondeur
6. **Contextualiser** : Ajouter des lignes de référence, moyennes, etc.

---

## 🔧 Comparaison Matplotlib vs Plotly

### Matplotlib
```python
import matplotlib.pyplot as plt

# Exemple simple
plt.figure(figsize=(10, 6))
plt.hist(df['ventes'], bins=30, edgecolor='black')
plt.title('Distribution des ventes')
plt.xlabel('Ventes (€)')
plt.ylabel('Fréquence')
plt.grid(True, alpha=0.3)
plt.show()
```

**Avantages** : Rapide, grande communauté, personnalisation fine  
**Inconvénients** : Graphiques statiques, moins intuitif pour l'interactivité

### Plotly
```python
import plotly.express as px

fig = px.histogram(df, x='ventes', nbins=30, title='Distribution des ventes')
fig.show()
```

**Avantages** : Interactif, moderne, facile à partager, bonne API  
**Inconvénients** : Fichiers HTML plus lourds, moins de contrôle fin

**Recommandation** : Utilisez **Plotly** pour l'exploration et les présentations, **Matplotlib** pour les publications scientifiques.

---

## 🎓 Exercices pratiques

### Exercice 1 : Histogrammes
Créez un histogramme des prix et identifiez :
- La fourchette de prix la plus fréquente
- S'il y a des outliers

### Exercice 2 : Boxplots
Comparez les distributions de ventes entre régions avec un boxplot. Quelle région a :
- Les ventes les plus élevées en médiane ?
- La plus grande dispersion ?
- Le plus d'outliers ?

### Exercice 3 : Séries temporelles
Créez un graphique montrant l'évolution des ventes avec une moyenne mobile de 30 jours. Identifiez :
- La tendance générale (croissance/décroissance)
- Les périodes de forte/faible activité

### Exercice 4 : Dashboard
Créez un dashboard avec 4 graphiques montrant :
1. Distribution des ventes
2. Ventes par produit (boxplot)
3. Évolution temporelle
4. Corrélation prix-ventes

---

## 📚 Ressources complémentaires

- [Documentation Plotly](https://plotly.com/python/)
- [Galerie d'exemples Plotly](https://plotly.com/python/plotly-express/)
- [Documentation Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/) : Alternative élégante basée sur Matplotlib

---

## 🎯 Points clés à retenir

✅ **Histogrammes** : Pour les distributions  
✅ **Boxplots** : Pour comparer et détecter les outliers  
✅ **Séries temporelles** : Pour l'évolution dans le temps  
✅ **Plotly** : Pour l'interactivité et l'exploration  
✅ **Un graphique = Un message** : Restez simple et clair  
✅ **Contexte** : Toujours titrer et labelliser

---

**Prochaine séance** : Manipulation avancée de données avec Pandas et préparation pour le machine learning ! 🚀
