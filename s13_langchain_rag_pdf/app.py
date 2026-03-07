"""
S13 — RAG PDF Chat Application
Application Streamlit pour chatter avec un document PDF
en utilisant LangChain et OpenAI.
"""

import os
import tempfile
import time
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
load_dotenv()

st.set_page_config(
    page_title="📄 RAG PDF Chat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Styles CSS personnalisés
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #f0f2f6;
        margin-bottom: 1.5rem;
    }
    .chat-message-user {
        background: #e8f4fd;
        border-radius: 12px 12px 4px 12px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
    }
    .chat-message-bot {
        background: #f0f7ee;
        border-radius: 12px 12px 12px 4px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
    }
    .source-badge {
        display: inline-block;
        background: #e2e8f0;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.8rem;
        margin: 2px;
    }
    .metric-card {
        background: #f8fafc;
        border-radius: 8px;
        padding: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Fonctions utilitaires
# ─────────────────────────────────────────────────────────────

def get_rag_type_config(rag_type: str) -> dict:
    """Retourne la configuration selon le type de RAG sélectionné."""
    configs = {
        "Naïf (Basic RAG)": {
            "search_type": "similarity",
            "description": "Recherche simple par similarité cosinus.",
        },
        "MMR (Diversifié)": {
            "search_type": "mmr",
            "description": "Max Marginal Relevance — équilibre pertinence et diversité.",
        },
        "Similarity + Score": {
            "search_type": "similarity_score_threshold",
            "description": "Filtre les résultats sous un seuil de score minimum.",
        },
    }
    return configs.get(rag_type, configs["Naïf (Basic RAG)"])


def format_documents(docs) -> str:
    """Formate les documents récupérés en texte structuré."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Document")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Extrait {i} — Page {page}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


@st.cache_resource(show_spinner=False)
def build_vectorstore(
    pdf_bytes: bytes,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
) -> tuple:
    """
    Construit le vectorstore FAISS à partir d'un PDF.
    Mis en cache pour éviter de reconstruire à chaque interaction.

    Returns:
        Tuple (vectorstore, nb_chunks, nb_pages)
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        # Charger le PDF
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        nb_pages = len(pages)

        # Découper en chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        chunks = splitter.split_documents(pages)
        nb_chunks = len(chunks)

        # Créer les embeddings et le vectorstore
        embeddings = OpenAIEmbeddings(model=embedding_model)
        vectorstore = FAISS.from_documents(chunks, embeddings)

        return vectorstore, nb_chunks, nb_pages
    finally:
        os.unlink(tmp_path)


def build_rag_chain(
    vectorstore,
    llm_model: str,
    temperature: float,
    top_k: int,
    rag_type: str,
    score_threshold: float,
    max_memory_turns: int,
):
    """
    Construit le pipeline RAG avec mémoire conversationnelle.

    Returns:
        chain (RunnableWithMessageHistory)
    """
    rag_config = get_rag_type_config(rag_type)

    # Configurer le retriever selon le type de RAG
    search_kwargs = {"k": top_k}
    if rag_type == "MMR (Diversifié)":
        search_kwargs["fetch_k"] = top_k * 3
        search_kwargs["lambda_mult"] = 0.5
    elif rag_type == "Similarity + Score":
        search_kwargs["score_threshold"] = score_threshold

    retriever = vectorstore.as_retriever(
        search_type=rag_config["search_type"],
        search_kwargs=search_kwargs,
    )

    # Initialiser le LLM
    llm = ChatOpenAI(
        model=llm_model,
        temperature=temperature,
        streaming=True,
    )

    # Prompt RAG avec historique
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Tu es un assistant expert en analyse de documents.
Réponds à la question de l'utilisateur en te basant UNIQUEMENT sur les extraits du document fournis.

Règles importantes :
- Si la réponse n'est pas dans les extraits, dis clairement "Je ne trouve pas cette information dans le document."
- Cite les numéros de page lorsque c'est possible.
- Sois précis, clair et concis.
- Réponds toujours en français sauf si l'utilisateur parle une autre langue.

Extraits du document :
{context}
""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    # Chain principale
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_documents(
                retriever.invoke(x["question"])
            )
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    # Gestion de la mémoire par session
    def get_session_history(session_id: str):
        if session_id not in st.session_state.get("chat_histories", {}):
            if "chat_histories" not in st.session_state:
                st.session_state.chat_histories = {}
            st.session_state.chat_histories[session_id] = ChatMessageHistory()
        return st.session_state.chat_histories[session_id]

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    ), retriever


# ─────────────────────────────────────────────────────────────
# Interface principale
# ─────────────────────────────────────────────────────────────

def main():
    # ── En-tête ──────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>📄 RAG PDF Chat</h1>
        <p>Téléchargez un PDF et posez vos questions en langage naturel</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Barre latérale ────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Clé API
        st.subheader("🔑 OpenAI API")
        api_key = st.text_input(
            "Clé API OpenAI",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Votre clé API OpenAI (sk-...)",
        )
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        st.divider()

        # Modèle LLM
        st.subheader("🤖 Modèle")
        llm_model = st.selectbox(
            "Modèle OpenAI",
            options=[
                "gpt-4o-mini",
                "gpt-4o",
                "gpt-4-turbo",
                "gpt-3.5-turbo",
            ],
            index=0,
            help="Modèle utilisé pour générer les réponses",
        )

        embedding_model = st.selectbox(
            "Modèle d'Embeddings",
            options=[
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-ada-002",
            ],
            index=0,
            help="Modèle pour la représentation vectorielle des textes",
        )

        temperature = st.slider(
            "🌡️ Température",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="0 = déterministe et factuel (recommandé pour le RAG), 1.0 = réponses plus créatives",
        )

        st.divider()

        # Type de RAG
        st.subheader("🔍 Type de RAG")
        rag_type = st.selectbox(
            "Stratégie de recherche",
            options=[
                "Naïf (Basic RAG)",
                "MMR (Diversifié)",
                "Similarity + Score",
            ],
            index=0,
            help="Choisissez la stratégie de récupération des documents",
        )

        # Afficher la description du type de RAG
        rag_config = get_rag_type_config(rag_type)
        st.caption(f"ℹ️ {rag_config['description']}")

        top_k = st.slider(
            "📄 Nombre d'extraits (top-k)",
            min_value=1,
            max_value=10,
            value=4,
            help="Nombre d'extraits récupérés pour construire le contexte",
        )

        score_threshold = 0.5
        if rag_type == "Similarity + Score":
            score_threshold = st.slider(
                "🎯 Seuil de score",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Score minimum pour inclure un extrait dans le contexte (0 = tout inclure, 1 = très strict). Uniquement pour 'Similarity + Score'.",
            )

        st.divider()

        # Découpage du texte
        st.subheader("✂️ Découpage du texte")
        chunk_size = st.slider(
            "Taille des chunks",
            min_value=200,
            max_value=2000,
            value=800,
            step=100,
            help="Nombre de caractères par chunk",
        )
        chunk_overlap = st.slider(
            "Chevauchement",
            min_value=0,
            max_value=400,
            value=100,
            step=25,
            help="Chevauchement entre chunks consécutifs",
        )

        st.divider()

        # Mémoire
        st.subheader("🧠 Mémoire")
        max_memory_turns = st.slider(
            "Tours de conversation conservés",
            min_value=1,
            max_value=20,
            value=5,
            help="Nombre de tours de conversation gardés en mémoire",
        )

        st.divider()

        # Bouton reset
        if st.button("🗑️ Réinitialiser la conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_histories = {}
            st.rerun()

        # Info
        with st.expander("📖 À propos"):
            st.markdown("""
**RAG PDF Chat** — Powered by LangChain + OpenAI

Techniques utilisées :
- 📂 **PyPDFLoader** — Extraction du PDF
- ✂️ **RecursiveCharacterTextSplitter** — Découpage
- 🔢 **OpenAI Embeddings** — Vectorisation
- 🗄️ **FAISS** — Index vectoriel
- 🔍 **Retriever** — Recherche sémantique
- 🤖 **ChatOpenAI** — Génération
            """)

    # ── Contenu principal ─────────────────────────────────────

    # Initialisation de l'état de session
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "pdf_name" not in st.session_state:
        st.session_state.pdf_name = None
    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}
    if "pdf_stats" not in st.session_state:
        st.session_state.pdf_stats = {}

    # ── Upload du PDF ─────────────────────────────────────────
    col_upload, col_stats = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "📂 Téléchargez votre document PDF",
            type=["pdf"],
            help="Formats supportés : PDF (max ~50 pages recommandé)",
        )

    if uploaded_file is not None:
        # Vérification de la clé API
        if not os.getenv("OPENAI_API_KEY"):
            st.error("❌ Veuillez entrer votre clé API OpenAI dans la barre latérale.")
            st.stop()

        # Construire le vectorstore si c'est un nouveau fichier
        if (
            st.session_state.pdf_name != uploaded_file.name
            or st.session_state.vectorstore is None
        ):
            with st.spinner(f"⏳ Traitement de **{uploaded_file.name}** en cours..."):
                try:
                    vectorstore, nb_chunks, nb_pages = build_vectorstore(
                        pdf_bytes=uploaded_file.getvalue(),
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        embedding_model=embedding_model,
                    )
                    st.session_state.vectorstore = vectorstore
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.pdf_stats = {
                        "nb_pages": nb_pages,
                        "nb_chunks": nb_chunks,
                    }
                    # Réinitialiser la conversation
                    st.session_state.messages = []
                    st.session_state.chat_histories = {}
                    st.success(
                        f"✅ **{uploaded_file.name}** indexé avec succès! "
                        f"({nb_pages} pages, {nb_chunks} chunks)"
                    )
                except Exception as e:
                    st.error(f"❌ Erreur lors du traitement du PDF : {e}")
                    st.stop()

        # Afficher les statistiques du document
        with col_stats:
            if st.session_state.pdf_stats:
                st.markdown("**📊 Stats du document**")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Pages", st.session_state.pdf_stats["nb_pages"])
                with c2:
                    st.metric("Chunks", st.session_state.pdf_stats["nb_chunks"])
                with c3:
                    st.metric("Top-k", top_k)

        st.divider()

        # ── Zone de chat ──────────────────────────────────────
        st.subheader(f"💬 Chat — {st.session_state.pdf_name}")

        # Afficher l'historique des messages
        chat_container = st.container()
        with chat_container:
            if not st.session_state.messages:
                st.info(
                    "👋 Le document est prêt ! Posez votre première question ci-dessous."
                )
            else:
                for msg in st.session_state.messages:
                    role = msg["role"]
                    content = msg["content"]
                    avatar = "👤" if role == "user" else "🤖"

                    with st.chat_message(role, avatar=avatar):
                        st.markdown(content)

                        # Afficher les sources si disponibles
                        if role == "assistant" and "sources" in msg:
                            with st.expander("📚 Sources utilisées", expanded=False):
                                for src in msg["sources"]:
                                    page = src.get("page", "?")
                                    snippet = src.get("content", "")[:200]
                                    st.markdown(
                                        f'<span class="source-badge">Page {page}</span>',
                                        unsafe_allow_html=True,
                                    )
                                    st.caption(f"…{snippet}…")

        # ── Input utilisateur ─────────────────────────────────
        question = st.chat_input(
            "Posez votre question sur le document...",
            disabled=st.session_state.vectorstore is None,
        )

        if question:
            # Ajouter le message utilisateur
            st.session_state.messages.append({"role": "user", "content": question})

            with st.chat_message("user", avatar="👤"):
                st.markdown(question)

            # Générer la réponse
            with st.chat_message("assistant", avatar="🤖"):
                response_placeholder = st.empty()

                try:
                    # Construire le pipeline RAG
                    rag_chain, retriever = build_rag_chain(
                        vectorstore=st.session_state.vectorstore,
                        llm_model=llm_model,
                        temperature=temperature,
                        top_k=top_k,
                        rag_type=rag_type,
                        score_threshold=score_threshold,
                        max_memory_turns=max_memory_turns,
                    )

                    # Récupérer les sources
                    sources_docs = retriever.invoke(question)
                    sources = [
                        {
                            "page": doc.metadata.get("page", "?"),
                            "content": doc.page_content,
                        }
                        for doc in sources_docs
                    ]

                    # Streaming de la réponse
                    full_response = ""
                    config = {"configurable": {"session_id": "main_session"}}

                    for chunk in rag_chain.stream(
                        {"question": question}, config=config
                    ):
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")

                    response_placeholder.markdown(full_response)

                    # Afficher les sources
                    with st.expander("📚 Sources utilisées", expanded=False):
                        for src in sources:
                            page = src.get("page", "?")
                            snippet = src.get("content", "")[:200]
                            st.markdown(
                                f'<span class="source-badge">Page {page}</span>',
                                unsafe_allow_html=True,
                            )
                            st.caption(f"…{snippet}…")

                    # Sauvegarder la réponse
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": full_response,
                            "sources": sources,
                        }
                    )

                except Exception as e:
                    error_msg = f"❌ Erreur : {e}"
                    response_placeholder.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )

    else:
        # Pas de PDF uploadé — message d'accueil
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #718096;">
            <div style="font-size: 4rem;">📄</div>
            <h3>Bienvenue dans RAG PDF Chat</h3>
            <p>Téléchargez un document PDF pour commencer à chatter avec son contenu.</p>
            <br>
            <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
                <div>🔍 <strong>Recherche sémantique</strong></div>
                <div>💬 <strong>Chat conversationnel</strong></div>
                <div>📚 <strong>Sources citées</strong></div>
                <div>⚡ <strong>Streaming temps réel</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.info("👈 Configurez vos paramètres dans la barre latérale, puis uploadez votre PDF.")


if __name__ == "__main__":
    main()
