"""
Séance 7 — Streamlit : Interface de prototypage pour démontrer des modèles

Cette application Streamlit permet de :
- Uploader un fichier CSV de documents
- Indexer les documents pour la recherche
- Effectuer une recherche par mots-clés
- Prévisualiser les résultats
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import re

# Configuration de la page
st.set_page_config(
    page_title="Recherche de Documents",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("🔍 Application de Recherche de Documents")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Section Upload
    st.subheader("📤 Upload de données")
    uploaded_file = st.file_uploader(
        "Choisissez un fichier CSV",
        type=['csv'],
        help="Le fichier doit contenir au moins une colonne 'text' ou 'content'"
    )
    
    st.markdown("---")
    
    # Section Paramètres de recherche
    st.subheader("🔧 Paramètres de recherche")
    search_mode = st.selectbox(
        "Mode de recherche",
        ["Contient (sous-chaîne)", "Mots-clés (exact)", "Regex"],
        help="Choisissez le mode de recherche"
    )
    
    case_sensitive = st.checkbox(
        "Sensible à la casse",
        value=False,
        help="Respecter les majuscules/minuscules"
    )
    
    st.markdown("---")
    
    # Section Statistiques
    if 'df' in st.session_state and st.session_state.df is not None:
        st.subheader("📊 Statistiques")
        st.metric("Nombre de documents", len(st.session_state.df))
        if 'indexed' in st.session_state and st.session_state.indexed:
            st.success("✅ Documents indexés")
        else:
            st.warning("⚠️ Documents non indexés")

# Initialisation du session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'indexed' not in st.session_state:
    st.session_state.indexed = False
if 'index_data' not in st.session_state:
    st.session_state.index_data = {}

# Fonction pour charger les données
@st.cache_data
def load_data(file):
    """Charger un fichier CSV"""
    try:
        df = pd.read_csv(file)
        return df, None
    except Exception as e:
        return None, str(e)

# Fonction pour indexer les documents
@st.cache_data
def index_documents(_df, text_column):
    """Créer un index simple pour la recherche"""
    index = {}
    for idx, row in _df.iterrows():
        text = str(row[text_column]).lower()
        words = set(re.findall(r'\w+', text))
        for word in words:
            if word not in index:
                index[word] = []
            index[word].append(idx)
    return index

# Fonction de recherche
def search_documents(df, query, text_column, mode="contains", case_sensitive=False, index_data=None):
    """Rechercher dans les documents"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    if not case_sensitive:
        query = query.lower()
    
    results_mask = pd.Series([False] * len(df))
    
    # Utiliser l'index pour la recherche par mots-clés si disponible
    if mode == "Mots-clés (exact)" and index_data is not None:
        query_words = set(re.findall(r'\w+', query))
        if not case_sensitive:
            query_words = {word.lower() for word in query_words}
        
        # Trouver les documents contenant tous les mots-clés
        matching_indices = None
        for word in query_words:
            word_lower = word.lower() if not case_sensitive else word
            if word_lower in index_data:
                word_indices = set(index_data[word_lower])
                if matching_indices is None:
                    matching_indices = word_indices
                else:
                    matching_indices = matching_indices.intersection(word_indices)
            else:
                # Si un mot n'est pas trouvé, aucun résultat
                matching_indices = set()
                break
        
        if matching_indices:
            results_mask[list(matching_indices)] = True
        return df[results_mask]
    
    # Recherche classique pour les autres modes
    for idx, row in df.iterrows():
        text = str(row[text_column])
        if not case_sensitive:
            text = text.lower()
        
        if mode == "Contient (sous-chaîne)":
            if query in text:
                results_mask[idx] = True
        elif mode == "Mots-clés (exact)":
            words = set(re.findall(r'\w+', text))
            query_words = set(re.findall(r'\w+', query))
            if query_words.issubset(words):
                results_mask[idx] = True
        elif mode == "Regex":
            try:
                if re.search(query, text, re.IGNORECASE if not case_sensitive else 0):
                    results_mask[idx] = True
            except re.error:
                st.error("Erreur: Expression régulière invalide")
                return pd.DataFrame()
    
    return df[results_mask]

# Section principale
if uploaded_file is not None:
    # Charger les données
    if st.session_state.df is None:
        with st.spinner("Chargement des données..."):
            df, error = load_data(uploaded_file)
            if error:
                st.error(f"❌ Erreur lors du chargement: {error}")
            else:
                st.session_state.df = df
                st.session_state.indexed = False
                st.success(f"✅ Fichier chargé avec succès! ({len(df)} lignes)")
    
    df = st.session_state.df
    
    if df is not None:
        # Afficher un aperçu
        with st.expander("👀 Aperçu des données", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Lignes", len(df))
            with col2:
                st.metric("Colonnes", len(df.columns))
            with col3:
                st.metric("Mémoire", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        # Sélection de la colonne de texte
        text_columns = df.select_dtypes(include=['object']).columns.tolist()
        if not text_columns:
            st.error("❌ Aucune colonne de type texte trouvée dans le fichier")
        else:
            text_column = st.selectbox(
                "Sélectionnez la colonne contenant le texte",
                text_columns,
                index=0 if 'text' in text_columns else (text_columns.index('content') if 'content' in text_columns else 0)
            )
            
            # Bouton d'indexation
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("🗂️ Indexer les documents", type="primary", use_container_width=True):
                    with st.spinner("Indexation en cours..."):
                        st.session_state.index_data = index_documents(df, text_column)
                        st.session_state.indexed = True
                        st.success(f"✅ {len(df)} documents indexés!")
            
            st.markdown("---")
            
            # Section de recherche
            st.header("🔍 Recherche")
            
            search_query = st.text_input(
                "Entrez votre requête de recherche",
                placeholder="Ex: machine learning, data science, etc.",
                help="Entrez un ou plusieurs mots-clés pour rechercher dans les documents"
            )
            
            if search_query:
                with st.spinner("Recherche en cours..."):
                    # Passer l'index si disponible
                    index_to_use = st.session_state.index_data if st.session_state.indexed else None
                    results = search_documents(
                        df, 
                        search_query, 
                        text_column, 
                        mode=search_mode,
                        case_sensitive=case_sensitive,
                        index_data=index_to_use
                    )
                
                st.markdown("---")
                
                # Afficher les résultats
                if len(results) > 0:
                    st.success(f"✅ {len(results)} résultat(s) trouvé(s)")
                    
                    # Statistiques des résultats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Résultats", len(results))
                    with col2:
                        st.metric("% du total", f"{len(results)/len(df)*100:.1f}%")
                    with col3:
                        st.metric("Documents restants", len(df) - len(results))
                    
                    st.markdown("---")
                    
                    # Options d'affichage
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        sort_column = st.selectbox(
                            "Trier par",
                            results.columns.tolist(),
                            index=0
                        )
                    with col2:
                        sort_order = st.selectbox(
                            "Ordre",
                            ["Croissant", "Décroissant"],
                            index=0
                        )
                    
                    # Trier les résultats
                    results_sorted = results.sort_values(
                        by=sort_column,
                        ascending=(sort_order == "Croissant")
                    )
                    
                    # Affichage des résultats
                    st.subheader("📄 Résultats de la recherche")
                    
                    # Option: affichage en cartes ou en tableau
                    display_mode = st.radio(
                        "Mode d'affichage",
                        ["Cartes", "Tableau"],
                        horizontal=True
                    )
                    
                    if display_mode == "Cartes":
                        # Affichage en cartes
                        for idx, row in results_sorted.iterrows():
                            with st.container():
                                st.markdown(f"**Document #{idx}**")
                                
                                # Afficher le texte avec highlight (simple)
                                text = str(row[text_column])
                                if not case_sensitive:
                                    # Highlight simple (sans regex)
                                    display_text = text[:500] + "..." if len(text) > 500 else text
                                else:
                                    display_text = text[:500] + "..." if len(text) > 500 else text
                                
                                st.markdown(f"> {display_text}")
                                
                                # Afficher les autres colonnes
                                other_cols = [col for col in results_sorted.columns if col != text_column]
                                if other_cols:
                                    cols = st.columns(len(other_cols))
                                    for i, col in enumerate(other_cols):
                                        with cols[i]:
                                            st.caption(f"**{col}:** {row[col]}")
                                
                                st.markdown("---")
                    else:
                        # Affichage en tableau
                        st.dataframe(results_sorted, use_container_width=True)
                    
                    # Option d'export
                    st.markdown("---")
                    st.subheader("💾 Export des résultats")
                    
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        csv = results_sorted.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Télécharger CSV",
                            data=csv,
                            file_name=f"resultats_recherche_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    # Visualisation (si colonnes numériques disponibles)
                    numeric_cols = results_sorted.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    if numeric_cols:
                        st.markdown("---")
                        st.subheader("📊 Visualisations")
                        
                        viz_col = st.selectbox("Sélectionnez une colonne à visualiser", numeric_cols)
                        
                        fig = px.histogram(
                            results_sorted,
                            x=viz_col,
                            title=f"Distribution de {viz_col}",
                            labels={viz_col: viz_col, 'count': 'Nombre de documents'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.warning("⚠️ Aucun résultat trouvé pour cette requête")
                    st.info("💡 Essayez de modifier votre requête ou le mode de recherche")

else:
    # Message d'accueil
    st.info("👋 Bienvenue! Uploadez un fichier CSV pour commencer.")
    
    st.markdown("""
    ### 📚 Guide d'utilisation
    
    1. **Upload**: Utilisez la sidebar pour uploader un fichier CSV contenant vos documents
    2. **Sélection**: Choisissez la colonne contenant le texte à indexer
    3. **Indexation**: Cliquez sur "Indexer les documents" pour préparer la recherche
    4. **Recherche**: Entrez vos mots-clés dans la barre de recherche
    5. **Exploration**: Consultez les résultats et exportez-les si nécessaire
    
    ### 📋 Format du fichier CSV
    
    Votre fichier doit contenir au minimum:
    - Une colonne avec du texte (nommée 'text', 'content', ou autre)
    - Optionnellement: d'autres colonnes avec des métadonnées
    
    Exemple de structure:
    ```
    text,category,date
    "Texte du document 1","FAQ","2024-01-01"
    "Texte du document 2","Blog","2024-01-02"
    ```
    """)
    
    # Exemple de données
    st.markdown("---")
    st.subheader("📄 Exemple de fichier CSV")
    
    example_data = pd.DataFrame({
        'text': [
            "Comment réinitialiser mon mot de passe ?",
            "Dans cet article, nous explorons l'intelligence artificielle",
            "Quels sont vos horaires d'ouverture ?",
            "Le machine learning révolutionne l'industrie",
            "Comment contacter le support technique ?"
        ],
        'category': ['FAQ', 'Blog', 'FAQ', 'Blog', 'FAQ'],
        'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']
    })
    
    st.dataframe(example_data, use_container_width=True)
    
    # Télécharger l'exemple
    csv_example = example_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger cet exemple",
        data=csv_example,
        file_name="exemple_documents.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("**Séance 7 — Streamlit : Prototypage d'interface**")
    st.caption("Développé avec ❤️ et Streamlit")
