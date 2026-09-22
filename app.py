# =============================================================================
# app.py — Interface Streamlit du RAG
# =============================================================================
# Ce fichier lance l'interface web pour interroger tes documents.
# Il LIT uniquement la base Chroma déjà construite par ingest.py.
# Il ne re-indexe JAMAIS les documents — c'est le rôle de ingest.py.
#
# Connexion au LLM : Ollama exposé sur http://localhost:11434 (API REST locale)
# Aucune installation de client Ollama nécessaire côté Python.
#
# Lancement : streamlit run app.py
# Prérequis  : avoir lancé "python ingest.py" au moins une fois
# =============================================================================

import os
import sys
from pathlib import Path

import requests
import streamlit as st

# --- Désactive la télémétrie Chroma (évite les warnings "capture() takes 1 argument") ---
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

# =============================================================================
# Configuration de la page (doit être le premier appel Streamlit)
# =============================================================================
st.set_page_config(
    page_title="RAG — Assistant Documentaire",
    page_icon="📚",
    layout="centered",
)

# =============================================================================
# Constantes
# =============================================================================
CHROMA_DIR       = "./chroma_db"
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_BASE_URL  = "http://localhost:11434"   # URL de l'API REST Ollama
OLLAMA_MODEL     = "qwen2.5:7b"
NB_SOURCES       = 4  # Nombre de chunks récupérés par la recherche


# =============================================================================
# Fonctions utilitaires
# =============================================================================

def base_chroma_est_valide(chemin: str) -> bool:
    """
    Vérifie que le dossier chroma_db existe et contient au moins un fichier.
    Retourne True si la base est prête, False sinon.
    """
    dossier = Path(chemin)
    if not dossier.exists():
        return False
    # Chroma crée plusieurs fichiers internes — on vérifie qu'il y en a au moins un
    fichiers = list(dossier.rglob("*"))
    return len(fichiers) > 0


@st.cache_resource(show_spinner="Chargement des embeddings et de la base vectorielle…")
def charger_retriever():
    """
    Charge les embeddings et la base Chroma UNE SEULE FOIS grâce au cache Streamlit.
    Retourne un retriever LangChain prêt à l'emploi.
    """
    # Import ici pour éviter les erreurs si les dépendances manquent au démarrage
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Chargement de la base existante — PAS de ré-indexation
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": NB_SOURCES},
    )

    return retriever


def construire_chaine_rag(retriever):
    """
    Construit la chaîne RetrievalQA en utilisant ChatOllama pointé sur
    l'API REST locale http://localhost:11434.
    On n'utilise PAS le client CLI Ollama — uniquement l'API HTTP.
    """
    from langchain_ollama import ChatOllama
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate

    # Prompt neutre : le modèle répond dans la même langue que la question
    prompt_template = """You are a helpful assistant. Answer the question using ONLY \
the context provided below. If the answer is not in the context, say so clearly \
instead of making something up. Always reply in the same language as the question.

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    # ChatOllama se connecte à l'API REST locale — pas besoin du client CLI
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,   # http://localhost:11434
        temperature=0.3,
    )

    chaine = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",            # "stuff" = tous les chunks dans un seul prompt
        retriever=retriever,
        return_source_documents=True,  # Pour afficher les sources à l'utilisateur
        chain_type_kwargs={"prompt": prompt},
    )

    return chaine


def verifier_ollama() -> tuple[bool, str]:
    """
    Vérifie qu'Ollama est joignable sur http://localhost:11434
    et que le modèle qwen2.5:7b est bien disponible.
    Retourne (True, "") si tout va bien, (False, message_erreur) sinon.
    """
    # 1. Vérifier que le serveur répond
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        return False, (
            "Le serveur Ollama ne répond pas sur `http://localhost:11434`.\n\n"
            "Lance Ollama depuis le menu Démarrer (icône dans la barre des tâches) "
            "ou via la commande :\n```\nollama serve\n```"
        )
    except requests.exceptions.Timeout:
        return False, "Ollama met trop de temps à répondre (timeout). Vérifie qu'il n'est pas surchargé."
    except Exception as e:
        return False, f"Erreur inattendue lors de la connexion à Ollama : `{e}`"

    # 2. Vérifier que le modèle est téléchargé
    modeles_disponibles = [m["name"] for m in resp.json().get("models", [])]
    modele_present = any(OLLAMA_MODEL in m for m in modeles_disponibles)

    if not modele_present:
        liste = "\n".join(f"- `{m}`" for m in modeles_disponibles) or "_(aucun modèle trouvé)_"
        return False, (
            f"Le modèle **{OLLAMA_MODEL}** n'est pas téléchargé.\n\n"
            f"Modèles disponibles sur ce serveur :\n{liste}\n\n"
            f"Pour télécharger le modèle, ouvre un terminal et lance :\n"
            f"```\nollama pull {OLLAMA_MODEL}\n```"
        )

    return True, ""


def extraire_noms_sources(source_documents: list) -> list[str]:
    """
    Extrait les noms de fichiers uniques depuis les métadonnées des documents sources.
    """
    sources = set()
    for doc in source_documents:
        source = doc.metadata.get("source", "")
        if source:
            # On affiche uniquement le nom du fichier, pas le chemin complet
            sources.add(Path(source).name)
    return sorted(sources)


# =============================================================================
# Interface principale
# =============================================================================

def main():
    # --- En-tête ---
    st.title("📚 Assistant RAG — Interroge tes documents")
    st.markdown(
        "Pose une question sur tes documents. L'assistant cherche les passages "
        "pertinents et génère une réponse grâce au modèle **Qwen 2.5 7B** "
        "tournant entièrement en local via Ollama (`http://localhost:11434`)."
    )
    st.divider()

    # -------------------------------------------------------------------------
    # Vérification 1 : la base Chroma doit exister
    # -------------------------------------------------------------------------
    if not base_chroma_est_valide(CHROMA_DIR):
        st.error(
            "⚠️ **Base de données introuvable ou vide.**\n\n"
            "La base vectorielle `chroma_db/` n'a pas encore été créée. "
            "Pour l'initialiser, ouvre un terminal dans ce dossier et lance :\n\n"
            "```\npython ingest.py\n```\n\n"
            "Assure-toi d'avoir placé au moins un fichier PDF ou TXT "
            "dans le dossier `docs/` avant de lancer cette commande."
        )
        st.stop()

    # -------------------------------------------------------------------------
    # Vérification 2 : Ollama doit être joignable ET le modèle disponible
    # -------------------------------------------------------------------------
    ollama_ok, ollama_msg = verifier_ollama()
    if not ollama_ok:
        st.error(f"🔴 **Impossible de contacter Ollama.**\n\n{ollama_msg}")
        st.stop()

    st.success(f"✅ Ollama connecté — modèle `{OLLAMA_MODEL}` prêt.")

    # -------------------------------------------------------------------------
    # Chargement du retriever (mis en cache — chargé une seule fois)
    # -------------------------------------------------------------------------
    try:
        retriever = charger_retriever()
    except Exception as e:
        st.error(
            f"❌ **Erreur lors du chargement de la base vectorielle.**\n\n"
            f"Détail : `{e}`\n\n"
            "Essaie de relancer `python ingest.py` pour reconstruire la base."
        )
        st.stop()

    # -------------------------------------------------------------------------
    # Initialisation de l'historique de conversation dans la session Streamlit
    # -------------------------------------------------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Affichage de l'historique des échanges précédents
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # -------------------------------------------------------------------------
    # Zone de saisie de la question
    # -------------------------------------------------------------------------
    question = st.chat_input("Pose ta question sur tes documents…")

    if question:
        # Affiche la question de l'utilisateur dans le chat
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        # --- Génération de la réponse ---
        with st.chat_message("assistant"):
            with st.spinner("Recherche dans les documents et génération de la réponse…"):
                try:
                    chaine = construire_chaine_rag(retriever)
                    resultat = chaine.invoke({"query": question})
                    reponse = resultat.get("result", "Aucune réponse générée.")
                    source_documents = resultat.get("source_documents", [])

                except Exception as e:
                    reponse = None
                    source_documents = []
                    erreur = str(e).lower()

                    if any(m in erreur for m in ["connection", "refused", "connectionerror", "timeout"]):
                        st.error(
                            "🔴 **Ollama s'est déconnecté pendant la génération.**\n\n"
                            "Vérifie qu'Ollama tourne toujours, puis réessaie."
                        )
                    else:
                        st.error(f"❌ **Erreur inattendue.**\n\nDétail : `{e}`")

            # Affichage de la réponse si elle a été générée
            if reponse:
                st.markdown(reponse)
                st.session_state.messages.append(
                    {"role": "assistant", "content": reponse}
                )

                # --- Section Sources (optionnelle) ---
                if source_documents:
                    noms_sources = extraire_noms_sources(source_documents)
                    with st.expander("📎 Sources utilisées pour cette réponse"):
                        for nom in noms_sources:
                            st.markdown(f"- 📄 `{nom}`")


# =============================================================================
# Point d'entrée
# =============================================================================
if __name__ == "__main__":
    main()
