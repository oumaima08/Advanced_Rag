# =============================================================================
# ingest.py — Script d'indexation des documents
# =============================================================================
# Ce script doit être lancé MANUELLEMENT chaque fois que tu ajoutes
# des fichiers dans le dossier "docs/".
#
# Il effectue les étapes suivantes :
#   1. Lecture de tous les fichiers PDF et TXT dans "docs/"
#   2. Découpage en chunks (morceaux de texte)
#   3. Calcul des embeddings (vecteurs numériques) avec sentence-transformers
#   4. Sauvegarde dans la base Chroma (dossier "chroma_db/")
#
# Lancement : python ingest.py
# =============================================================================

import os
import sys
import logging
from pathlib import Path

# --- Désactive la télémétrie Chroma (évite les warnings "capture() takes 1 argument") ---
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

# --- Chargeurs de documents ---
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# --- Découpage du texte ---
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Embeddings locaux via sentence-transformers ---
from langchain_huggingface import HuggingFaceEmbeddings

# --- Base vectorielle Chroma ---
from langchain_chroma import Chroma

# =============================================================================
# Configuration du logger (affiche les avertissements dans la console)
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# =============================================================================
# Constantes — chemins et paramètres
# =============================================================================
DOCS_DIR = Path("./docs")               # Dossier contenant les documents sources
CHROMA_DIR = "./chroma_db"              # Dossier de persistance de la base Chroma
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Modèle d'embeddings

# Paramètres de découpage du texte
CHUNK_SIZE = 800      # Taille maximale d'un chunk (en caractères)
CHUNK_OVERLAP = 150   # Chevauchement entre deux chunks consécutifs


def charger_documents(dossier: Path) -> list:
    """
    Parcourt le dossier "docs/" et charge tous les fichiers PDF et TXT trouvés.
    Les fichiers corrompus ou illisibles sont ignorés avec un avertissement.

    Retourne une liste de documents LangChain.
    """
    documents = []

    # Récupère tous les fichiers PDF et TXT (insensible à la casse)
    fichiers = (
        list(dossier.glob("*.pdf"))
        + list(dossier.glob("*.PDF"))
        + list(dossier.glob("*.txt"))
        + list(dossier.glob("*.TXT"))
    )

    if not fichiers:
        logger.error(
            "❌  Le dossier 'docs/' est vide ou ne contient aucun fichier PDF/TXT.\n"
            "    Ajoute des documents dans ce dossier puis relance : python ingest.py"
        )
        sys.exit(1)

    logger.info(f"📂  {len(fichiers)} fichier(s) trouvé(s) dans '{dossier}'.")

    for fichier in fichiers:
        try:
            extension = fichier.suffix.lower()

            if extension == ".pdf":
                loader = PyPDFLoader(str(fichier))
            elif extension == ".txt":
                # encoding="utf-8" obligatoire sous Windows pour les accents
                loader = TextLoader(str(fichier), encoding="utf-8")
            else:
                continue  # Extension non supportée, on passe

            docs = loader.load()
            documents.extend(docs)
            logger.info(f"  ✅  Chargé : {fichier.name}  ({len(docs)} page(s)/section(s))")

        except Exception as e:
            # On log l'avertissement et on continue avec les autres fichiers
            logger.warning(
                f"  ⚠️   Impossible de lire '{fichier.name}' — fichier ignoré.\n"
                f"       Détail de l'erreur : {e}"
            )

    return documents


def decouper_en_chunks(documents: list) -> list:
    """
    Découpe les documents en morceaux (chunks) pour l'indexation.
    Un chunk de taille raisonnable améliore la précision de la recherche.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],  # Ordre de préférence de découpe
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"✂️   Découpage terminé : {len(chunks)} chunk(s) créé(s).")
    return chunks


def creer_base_vectorielle(chunks: list) -> None:
    """
    Calcule les embeddings et sauvegarde tous les chunks dans Chroma.
    Le dossier 'chroma_db/' est créé automatiquement s'il n'existe pas.
    """
    logger.info(f"🔢  Chargement du modèle d'embeddings : {EMBEDDING_MODEL}")
    logger.info("    (Première exécution : téléchargement automatique ~90 Mo)")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},   # Utilise le CPU (pas besoin de GPU)
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info(f"💾  Indexation et sauvegarde dans '{CHROMA_DIR}' en cours...")

    # Chroma.from_documents() crée la base ET sauvegarde automatiquement
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    logger.info(f"✅  Base vectorielle créée avec succès dans '{CHROMA_DIR}'.")


# =============================================================================
# Point d'entrée principal
# =============================================================================
def main():
    print("=" * 60)
    print("  RAG — Script d'indexation des documents")
    print("=" * 60)

    # Étape 1 : Vérification que le dossier docs/ existe
    if not DOCS_DIR.exists():
        logger.error(
            "❌  Le dossier 'docs/' n'existe pas.\n"
            "    Crée-le et place tes fichiers PDF/TXT dedans."
        )
        sys.exit(1)

    # Étape 2 : Chargement des documents
    logger.info("📖  Étape 1/3 — Chargement des documents...")
    documents = charger_documents(DOCS_DIR)

    if not documents:
        logger.error("❌  Aucun document valide n'a pu être chargé. Arrêt.")
        sys.exit(1)

    # Étape 3 : Découpage en chunks
    logger.info("✂️   Étape 2/3 — Découpage en chunks...")
    chunks = decouper_en_chunks(documents)

    # Étape 4 : Création de la base vectorielle
    logger.info("🔢  Étape 3/3 — Création de la base vectorielle Chroma...")
    creer_base_vectorielle(chunks)

    print("=" * 60)
    print(f"  ✅  Indexation terminée !")
    print(f"  📊  {len(chunks)} chunk(s) indexé(s) dans '{CHROMA_DIR}'.")
    print(f"  ▶️   Tu peux maintenant lancer : streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
