"""
Indexer - Script pour créer l'index FAISS à partir de documents
"""
import argparse
import json
import os
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import pickle


def load_documents(input_path: str) -> List[Dict[str, Any]]:
    """
    Charger les documents depuis un fichier JSON
    
    Args:
        input_path: Chemin vers le fichier JSON
        
    Returns:
        Liste de documents
    """
    print(f"📖 Chargement des documents depuis: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    print(f"✅ {len(documents)} documents chargés")
    return documents


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Découper un texte en chunks avec overlap
    
    Args:
        text: Texte à découper
        chunk_size: Taille des chunks en caractères
        overlap: Overlap entre chunks
        
    Returns:
        Liste de chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Ne pas couper au milieu d'un mot
        if end < len(text) and text[end] not in [' ', '\n', '.', ',', '!', '?']:
            # Chercher le dernier espace
            last_space = chunk.rfind(' ')
            if last_space > chunk_size // 2:  # Au moins 50% du chunk
                chunk = chunk[:last_space]
                end = start + last_space
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks


def process_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = 500,
    overlap: int = 50
) -> pd.DataFrame:
    """
    Traiter les documents: chunking + métadonnées
    
    Args:
        documents: Liste de documents
        chunk_size: Taille des chunks
        overlap: Overlap entre chunks
        
    Returns:
        DataFrame avec chunks et métadonnées
    """
    print(f"🔄 Traitement des documents (chunk_size={chunk_size}, overlap={overlap})")
    
    processed = []
    
    for doc in documents:
        doc_id = doc.get('id', f"doc_{len(processed)}")
        title = doc.get('title', 'Untitled')
        content = doc.get('content', '')
        metadata = doc.get('metadata', {})
        
        # Chunking
        chunks = chunk_text(content, chunk_size, overlap)
        
        # Créer une entrée par chunk
        for i, chunk in enumerate(chunks):
            processed.append({
                'id': doc_id,
                'chunk_id': f"{doc_id}_chunk_{i}",
                'title': title,
                'content': content,  # Garder le contenu complet
                'chunk': chunk,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'metadata': metadata
            })
    
    df = pd.DataFrame(processed)
    print(f"✅ {len(df)} chunks créés depuis {len(documents)} documents")
    
    return df


def create_embeddings(
    texts: List[str],
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
) -> np.ndarray:
    """
    Créer des embeddings pour une liste de textes
    
    Args:
        texts: Liste de textes
        model_name: Nom du modèle d'embeddings
        
    Returns:
        Array numpy d'embeddings
    """
    print(f"🔄 Chargement du modèle: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"🔄 Génération des embeddings pour {len(texts)} textes...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    
    print(f"✅ Embeddings créés: shape = {embeddings.shape}")
    return np.array(embeddings).astype('float32')


def create_index(embeddings: np.ndarray, index_type: str = "flat") -> faiss.Index:
    """
    Créer un index FAISS
    
    Args:
        embeddings: Embeddings à indexer
        index_type: Type d'index ('flat' ou 'ivf')
        
    Returns:
        Index FAISS
    """
    dimension = embeddings.shape[1]
    
    if index_type == "flat":
        print(f"🔄 Création d'un index Flat (dimension={dimension})")
        index = faiss.IndexFlatL2(dimension)
    elif index_type == "ivf":
        nlist = min(100, len(embeddings) // 10)  # Nombre de clusters
        print(f"🔄 Création d'un index IVF (dimension={dimension}, nlist={nlist})")
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        index.train(embeddings)
    else:
        raise ValueError(f"Index type '{index_type}' not supported")
    
    print(f"🔄 Ajout de {len(embeddings)} vecteurs à l'index...")
    index.add(embeddings)
    
    print(f"✅ Index créé avec {index.ntotal} vecteurs")
    return index


def save_index(
    index: faiss.Index,
    documents: pd.DataFrame,
    embeddings: np.ndarray,
    output_dir: str
):
    """
    Sauvegarder l'index et les documents
    
    Args:
        index: Index FAISS
        documents: DataFrame des documents
        embeddings: Embeddings
        output_dir: Répertoire de sortie
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarder l'index FAISS
    index_path = os.path.join(output_dir, "faiss_index.bin")
    faiss.write_index(index, index_path)
    print(f"✅ Index sauvegardé: {index_path}")
    
    # Sauvegarder les documents et embeddings
    data_path = os.path.join(output_dir, "documents.pkl")
    data = {
        "documents": documents,
        "embeddings": embeddings
    }
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"✅ Documents sauvegardés: {data_path}")
    
    # Sauvegarder des stats
    stats_path = os.path.join(output_dir, "stats.json")
    stats = {
        "total_documents": len(documents['id'].unique()),
        "total_chunks": len(documents),
        "embedding_dimension": embeddings.shape[1],
        "index_type": index.__class__.__name__
    }
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Stats sauvegardées: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Indexer des documents pour RAG")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Chemin vers le fichier JSON de documents"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="index/",
        help="Répertoire de sortie pour l'index"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="Modèle d'embeddings"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Taille des chunks en caractères"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Overlap entre chunks"
    )
    parser.add_argument(
        "--index-type",
        type=str,
        default="flat",
        choices=["flat", "ivf"],
        help="Type d'index FAISS"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 RAG Indexer")
    print("="*80)
    
    # Étape 1: Charger les documents
    documents = load_documents(args.input)
    
    # Étape 2: Chunking
    df = process_documents(documents, args.chunk_size, args.chunk_overlap)
    
    # Étape 3: Créer les embeddings
    texts = df['chunk'].tolist()
    embeddings = create_embeddings(texts, args.embedding_model)
    
    # Étape 4: Créer l'index
    index = create_index(embeddings, args.index_type)
    
    # Étape 5: Sauvegarder
    save_index(index, df, embeddings, args.output)
    
    print("="*80)
    print("✅ Indexation terminée avec succès!")
    print("="*80)


if __name__ == "__main__":
    main()
