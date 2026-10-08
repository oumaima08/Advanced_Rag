# =============================================================================
# ingest.py — Script d'indexation avancée des documents
# =============================================================================
# Ce script doit être lancé MANUELLEMENT chaque fois que tu ajoutes
# des fichiers dans le dossier "docs/".
#
# Formats supportés :
#   .pdf   → PyPDFLoader
#   .txt   → TextLoader  (UTF-8)
#   .docx  → Docx2txtLoader  (Word)
#   .pptx  → UnstructuredPowerPointLoader  (PowerPoint)
#   .xlsx  → UnstructuredExcelLoader  (Excel)
#   .csv   → CSVLoader
#   .md    → TextLoader  (Markdown traité comme texte brut)
#   .html  → BSHTMLLoader  (pages web locales)
#   .json  → JSONLoader  (tableau de strings ou champ "text")
#
# Lancement : python ingest.py
# =============================================================================

import os
import sys
import pickle
import logging
from pathlib import Path

# --- Désactive la télémétrie Chroma ---
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"]     = "False"

# --- Chargeurs de documents ---
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    BSHTMLLoader,
    JSONLoader,
)

# --- Découpage du texte ---
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Embeddings locaux ---
from langchain_huggingface import HuggingFaceEmbeddings

# --- Base vectorielle Chroma ---
from langchain_chroma import Chroma

# --- BM25 (sparse retriever) ---
from langchain_community.retrievers import BM25Retriever

# =============================================================================
# Configuration du logger
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# =============================================================================
# Constantes
# =============================================================================
DOCS_DIR        = Path("./docs")
CHROMA_DIR      = "./chroma_db"
BM25_PATH       = "./bm25_index.pkl"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 100

# Mapping extension → (loader_class, kwargs)
# Chaque extension reconnue est associée à son loader LangChain
LOADERS: dict[str, tuple] = {
    ".pdf":  (PyPDFLoader,                    {}),
    ".txt":  (TextLoader,                     {"encoding": "utf-8"}),
    ".md":   (TextLoader,                     {"encoding": "utf-8"}),
    ".docx": (Docx2txtLoader,                 {}),
    ".pptx": (UnstructuredPowerPointLoader,   {"mode": "single"}),
    ".xlsx": (UnstructuredExcelLoader,        {"mode": "single"}),
    ".csv":  (CSVLoader,                      {"encoding": "utf-8"}),
    ".html": (BSHTMLLoader,                   {"open_encoding": "utf-8"}),
    ".htm":  (BSHTMLLoader,                   {"open_encoding": "utf-8"}),
    ".json": (JSONLoader,                     {"jq_schema": ".[].text", "text_content": False}),
}


# =============================================================================
# Étape 1 : Chargement des documents (multi-format)
# =============================================================================
def charger_documents(dossier: Path) -> list:
    """
    Parcourt récursivement le dossier docs/ et charge tous les fichiers
    dont l'extension est reconnue dans le dictionnaire LOADERS.
    Les fichiers illisibles sont ignorés avec un avertissement.
    Retourne une liste de Documents LangChain.
    """
    documents  = []
    extensions = set(LOADERS.keys())

    # iterdir récursif + filtre sur extensions connues, sans doublons (set)
    fichiers = sorted(set(
        f for f in dossier.rglob("*")
        if f.is_file() and f.suffix.lower() in extensions
    ))

    if not fichiers:
        exts_lisibles = "  ".join(sorted(extensions))
        logger.error(
            f"❌  Aucun fichier supporté trouvé dans '{dossier}'.\n"
            f"    Extensions acceptées : {exts_lisibles}\n"
            f"    Ajoute des documents puis relance : python ingest.py"
        )
        sys.exit(1)

    # Résumé par type de fichier
    compteur: dict[str, int] = {}
    for f in fichiers:
        ext = f.suffix.lower()
        compteur[ext] = compteur.get(ext, 0) + 1
    resume = "  ".join(f"{ext}:{n}" for ext, n in sorted(compteur.items()))
    logger.info(f"📂  {len(fichiers)} fichier(s) trouvé(s) — [{resume}]")

    for fichier in fichiers:
        ext = fichier.suffix.lower()
        loader_cls, loader_kwargs = LOADERS[ext]

        try:
            loader = loader_cls(str(fichier), **loader_kwargs)
            docs   = loader.load()

            # Garantit que la métadonnée "source" est bien définie
            for doc in docs:
                doc.metadata.setdefault("source", str(fichier))

            documents.extend(docs)
            logger.info(
                f"  ✅  [{ext.upper():5s}] {fichier.name}"
                f"  ({len(docs)} section(s))"
            )

        except Exception as e:
            logger.warning(
                f"  ⚠️  [{ext.upper():5s}] {fichier.name} — ignoré\n"
                f"       Erreur : {e}"
            )

    return documents


# =============================================================================
# Étape 2 : Découpage en chunks
# =============================================================================
def decouper_en_chunks(documents: list) -> list:
    """
    Découpe les documents en chunks courts.
    Chunks plus courts = meilleure précision pour le reranker CrossEncoder.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"✂️   {len(chunks)} chunk(s) créé(s) (taille={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
    return chunks


# =============================================================================
# Étape 3 : Construction de l'index Chroma (dense)
# =============================================================================
def creer_index_chroma(chunks: list, embeddings) -> None:
    """
    Calcule les embeddings denses et les stocke dans Chroma.
    """
    logger.info(f"💾  Construction de l'index Chroma dans '{CHROMA_DIR}'...")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    logger.info("  ✅  Index Chroma créé.")


# =============================================================================
# Étape 4 : Construction de l'index BM25 (sparse)
# =============================================================================
def creer_index_bm25(chunks: list) -> None:
    """
    Crée un index BM25 sur les chunks et le sérialise dans bm25_index.pkl.
    BM25 est un retriever lexical (mots-clés) complémentaire aux embeddings.
    La combinaison dense + sparse = Hybrid Search.
    """
    logger.info(f"🔍  Construction de l'index BM25...")

    # BM25Retriever.from_documents crée l'index en mémoire
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 10  # On garde plus de candidats pour le reranker

    # Sérialisation sur disque pour être rechargé par app.py
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25_retriever, f)

    logger.info(f"  ✅  Index BM25 sauvegardé dans '{BM25_PATH}'.")


# =============================================================================
# Point d'entrée
# =============================================================================
def main():
    print("=" * 65)
    print("  RAG Avancé — Script d'indexation (Dense + Sparse)")
    print("=" * 65)

    # Vérification du dossier docs/
    if not DOCS_DIR.exists():
        logger.error("❌  Le dossier 'docs/' n'existe pas. Crée-le et ajoute des fichiers.")
        sys.exit(1)

    # --- Étape 1 : Chargement ---
    logger.info("📖  Étape 1/4 — Chargement des documents...")
    documents = charger_documents(DOCS_DIR)

    if not documents:
        logger.error("❌  Aucun document valide chargé. Arrêt.")
        sys.exit(1)

    # --- Étape 2 : Chunking ---
    logger.info("✂️   Étape 2/4 — Découpage en chunks...")
    chunks = decouper_en_chunks(documents)

    # --- Étape 3 : Embeddings ---
    logger.info(f"🧠  Étape 3/4 — Chargement du modèle d'embeddings : {EMBEDDING_MODEL}")
    logger.info("    (Première exécution : téléchargement ~420 Mo, patiente...)")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # --- Étape 4a : Index Chroma (dense) ---
    logger.info("💾  Étape 4a/4 — Index vectoriel Chroma (dense)...")
    creer_index_chroma(chunks, embeddings)

    # --- Étape 4b : Index BM25 (sparse) ---
    logger.info("🔍  Étape 4b/4 — Index BM25 (sparse/lexical)...")
    creer_index_bm25(chunks)

    print("=" * 65)
    print(f"  ✅  Indexation terminée !")
    print(f"  📊  {len(chunks)} chunks indexés dans Chroma ET BM25.")
    print(f"  🧠  Embeddings : {EMBEDDING_MODEL}")
    print(f"  🔍  Index BM25 : {BM25_PATH}")
    print(f"  📄  Formats supportés : {' '.join(sorted(LOADERS.keys()))}")
    print(f"  ▶️   Lance maintenant : streamlit run app.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
