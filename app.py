# =============================================================================
# app.py — Interface Streamlit professionnelle — RAG Avancé
# =============================================================================
# Pipeline :
#   Multi-Query → Hybrid Search (Dense + BM25) → FlashRank Reranking
#   → Compression Contextuelle → Historique persistant → Ollama qwen2.5:7b
#
# Nouveautés :
#   - Persistance des conversations (JSON sur disque via conversations_manager)
#   - Sidebar avec liste des sessions (comme ChatGPT / Qwen)
#   - Multi-Query Retriever : génère 3 reformulations pour mieux retrouver
#   - Nouvelle conversation, chargement, suppression
# =============================================================================

import os
import pickle
import time
from pathlib import Path

import requests
import streamlit as st

from conversations_manager import (
    nouvelle_session_id,
    sauvegarder_session,
    charger_session,
    lister_sessions,
    supprimer_session,
    titre_depuis_question,
)

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"]     = "False"

# =============================================================================
# Config page
# =============================================================================
st.set_page_config(
    page_title="DocMind — RAG Avancé",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CSS — thème sombre professionnel
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li { color: #c9d1d9; font-size: 0.85rem; }

.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(102,126,234,0.3);
}
.main-header h1 {
    color: white !important; font-size: 2rem !important;
    font-weight: 700 !important; margin: 0 !important; letter-spacing: -0.5px;
}
.main-header p {
    color: rgba(255,255,255,0.85) !important; font-size: 0.95rem !important;
    margin: 0.4rem 0 0 0 !important;
}

.badge-container { display:flex; gap:10px; flex-wrap:wrap; margin:1rem 0 1.5rem 0; }
.badge {
    display:inline-flex; align-items:center; gap:6px; padding:6px 14px;
    border-radius:20px; font-size:0.78rem; font-weight:600; letter-spacing:0.3px;
}
.badge-purple { background:rgba(139,92,246,0.2); color:#a78bfa; border:1px solid rgba(139,92,246,0.4); }
.badge-blue   { background:rgba(59,130,246,0.2);  color:#60a5fa; border:1px solid rgba(59,130,246,0.4); }
.badge-green  { background:rgba(16,185,129,0.2);  color:#34d399; border:1px solid rgba(16,185,129,0.4); }
.badge-orange { background:rgba(245,158,11,0.2);  color:#fbbf24; border:1px solid rgba(245,158,11,0.4); }
.badge-pink   { background:rgba(236,72,153,0.2);  color:#f472b6; border:1px solid rgba(236,72,153,0.4); }

[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important; margin-bottom: 0.75rem !important;
    padding: 0.75rem 1rem !important; backdrop-filter: blur(10px);
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(102,126,234,0.12) !important;
    border-color: rgba(102,126,234,0.25) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: rgba(118,75,162,0.1) !important;
    border-color: rgba(118,75,162,0.2) !important;
}

[data-testid="stChatInput"] textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important; color: #000000 !important;
    font-size: 0.95rem !important; transition: border-color 0.2s;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 2px rgba(102,126,234,0.25) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; font-size: 0.85rem !important;
    padding: 0.55rem 1rem !important; transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(102,126,234,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(102,126,234,0.45) !important;
}
.danger-btn button {
    background: linear-gradient(135deg, #ef4444, #b91c1c) !important;
    box-shadow: 0 4px 15px rgba(239,68,68,0.3) !important;
}

/* Bouton session active */
.session-active button {
    background: rgba(102,126,234,0.25) !important;
    border: 1px solid rgba(102,126,234,0.5) !important;
    box-shadow: none !important; text-align: left !important;
}

[data-testid="stExpander"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important; margin-top: 0.5rem !important;
}
[data-testid="stExpander"] summary {
    color: #8b949e !important; font-size: 0.83rem !important; font-weight: 500 !important;
}

.status-card {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px; padding: 12px 14px; margin-bottom: 8px;
}
.status-card .label {
    color: #8b949e; font-size: 0.73rem; text-transform: uppercase;
    letter-spacing: 0.8px; font-weight: 600;
}
.status-card .value { color: #e6edf3; font-size: 0.88rem; font-weight: 500; margin-top: 2px; }

hr { border-color: rgba(255,255,255,0.08) !important; margin: 1rem 0 !important; }
[data-testid="stAlert"] { border-radius: 10px !important; border-width: 1px !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 10px; }
p, li, span { color: #c9d1d9; }
h1, h2, h3  { color: #e6edf3 !important; }

.sidebar-logo {
    text-align: center; padding: 1.2rem 0 0.8rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 1.2rem;
}
.sidebar-logo .logo-text {
    font-size: 1.4rem; font-weight: 700;
    background: linear-gradient(135deg, #667eea, #f472b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.sidebar-logo .logo-sub {
    color: #8b949e; font-size: 0.72rem; letter-spacing: 1.5px;
    text-transform: uppercase; margin-top: 2px;
}

.session-item {
    display: flex; align-items: center; gap: 8px; padding: 8px 10px;
    border-radius: 8px; margin-bottom: 4px; cursor: pointer;
    border: 1px solid transparent; transition: all 0.15s;
}
.session-item:hover { background: rgba(255,255,255,0.06); border-color: rgba(255,255,255,0.1); }
.session-item.active { background: rgba(102,126,234,0.15); border-color: rgba(102,126,234,0.35); }
.session-title { font-size: 0.82rem; color: #c9d1d9; flex: 1; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap; }
.session-date  { font-size: 0.70rem; color: #8b949e; white-space: nowrap; }
.session-count { font-size: 0.68rem; color: #8b949e; }

.fragment-card {
    background: rgba(255,255,255,0.03); border-left: 3px solid #667eea;
    border-radius: 0 8px 8px 0; padding: 10px 14px; margin: 6px 0;
    font-size: 0.83rem; color: #c9d1d9; line-height: 1.6;
}

.stat-row { display: flex; gap: 8px; margin: 0.4rem 0; flex-wrap: wrap; }
.stat-pill {
    background: rgba(102,126,234,0.15); color: #a78bfa;
    border: 1px solid rgba(102,126,234,0.3); border-radius: 6px;
    padding: 3px 10px; font-size: 0.76rem; font-weight: 600;
}

.welcome-box {
    background: rgba(102,126,234,0.08); border: 1px dashed rgba(102,126,234,0.35);
    border-radius: 12px; padding: 1.5rem 2rem; text-align: center; margin: 2rem 0;
}
.welcome-box h3  { color: #a78bfa !important; margin-bottom: 0.4rem !important; }
.welcome-box p   { color: #8b949e !important; font-size: 0.9rem !important; }

.new-chat-btn button {
    background: rgba(102,126,234,0.15) !important;
    border: 1px dashed rgba(102,126,234,0.4) !important;
    color: #a78bfa !important; box-shadow: none !important;
    font-size: 0.82rem !important;
}
.new-chat-btn button:hover {
    background: rgba(102,126,234,0.25) !important; transform: none !important;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Constantes
# =============================================================================
FORMATS_SUPPORTES = {
    ".pdf":  ("📕", "PDF"),   ".docx": ("📘", "Word"),
    ".pptx": ("📙", "PPT"),   ".xlsx": ("📗", "Excel"),
    ".csv":  ("📊", "CSV"),   ".txt":  ("📄", "TXT"),
    ".md":   ("📝", "MD"),    ".html": ("🌐", "HTML"),
    ".json": ("🔧", "JSON"),
}

CHROMA_DIR      = "./chroma_db"
BM25_PATH       = "./bm25_index.pkl"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "qwen2.5:7b"
NB_DENSE        = 10
NB_SPARSE       = 10
NB_RERANK       = 4
DENSE_WEIGHT    = 0.6
SPARSE_WEIGHT   = 0.4
WINDOW_SIZE     = 5
NB_MULTI_QUERY  = 3   # Nombre de reformulations générées par Multi-Query


# =============================================================================
# Utilitaires
# =============================================================================

def base_est_valide() -> bool:
    chroma_ok = Path(CHROMA_DIR).exists() and len(list(Path(CHROMA_DIR).rglob("*"))) > 0
    bm25_ok   = Path(BM25_PATH).exists()
    return chroma_ok and bm25_ok


def verifier_ollama() -> tuple[bool, str]:
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        return False, "Ollama ne répond pas sur `http://localhost:11434`.\n\nLance :\n```\nollama serve\n```"
    except requests.exceptions.Timeout:
        return False, "Ollama ne répond pas (timeout)."
    except Exception as e:
        return False, f"Erreur : `{e}`"

    modeles      = [m["name"] for m in resp.json().get("models", [])]
    modele_dispo = any(OLLAMA_MODEL in m for m in modeles)
    if not modele_dispo:
        liste = "\n".join(f"- `{m}`" for m in modeles) or "_(aucun)_"
        return False, (
            f"Modèle **{OLLAMA_MODEL}** absent.\n\nDisponibles :\n{liste}\n\n"
            f"```\nollama pull {OLLAMA_MODEL}\n```"
        )
    return True, ""


def extraire_sources(source_documents: list) -> list[str]:
    sources = set()
    for doc in source_documents:
        src = doc.metadata.get("source", "")
        if src:
            sources.add(Path(src).name)
    return sorted(sources)


def formater_date(iso: str) -> str:
    """Formate une date ISO en affichage court ex: 'Aujourd'hui 14:35'."""
    try:
        from datetime import datetime, date
        dt   = datetime.fromisoformat(iso)
        auj  = date.today()
        if dt.date() == auj:
            return f"Aujourd'hui {dt.strftime('%H:%M')}"
        return dt.strftime("%d/%m %H:%M")
    except Exception:
        return iso[:10]


# =============================================================================
# Chargement du pipeline RAG (caché — 1 seule fois)
# =============================================================================

@st.cache_resource(show_spinner=False)
def charger_retrievers():
    """
    Pipeline complet :
      Chroma (dense) + BM25 (sparse)
      → EnsembleRetriever (Hybrid, RRF)
      → FlashRank Reranker
      → ContextualCompressionRetriever
    Le Multi-Query wrapper est ajouté dans construire_chaine()
    car il nécessite le LLM.
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import FlashrankRerank

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    dense_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": NB_DENSE},
    )
    with open(BM25_PATH, "rb") as f:
        bm25_retriever = pickle.load(f)
    bm25_retriever.k = NB_SPARSE

    hybrid = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[DENSE_WEIGHT, SPARSE_WEIGHT],
    )
    reranker = FlashrankRerank(top_n=NB_RERANK)
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=hybrid,
    )


# =============================================================================
# Chaîne RAG avec Multi-Query (historique passé manuellement)
# =============================================================================

def construire_chaine(base_retriever):
    """
    Ajoute Multi-Query autour du retriever de base puis construit
    la ConversationalRetrievalChain.

    Multi-Query : le LLM génère NB_MULTI_QUERY reformulations de la
    question → élargit le retrieval → moins de passages manqués.
    """
    from langchain_ollama import ChatOllama
    from langchain.chains import ConversationalRetrievalChain
    from langchain.retrievers.multi_query import MultiQueryRetriever
    from langchain.prompts import PromptTemplate, ChatPromptTemplate
    from langchain.prompts.chat import (
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.3,
    )

    # --- Multi-Query : génère plusieurs reformulations de la question ---
    multi_query_prompt = PromptTemplate(
        input_variables=["question"],
        template=(
            f"You are an AI assistant. Generate {NB_MULTI_QUERY} different versions "
            "of the following question to improve document retrieval. "
            "Output ONLY the questions, one per line, no numbering.\n\n"
            "Original question: {question}"
        ),
    )
    mq_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        prompt=multi_query_prompt,
    )

    # --- Prompt système ---
    system_template = """You are a helpful and precise document assistant.
Answer the question using ONLY the context provided below.
If the answer is not in the context, clearly say so — do not invent anything.
Always reply in the same language as the question.
Structure your answer with bullet points or paragraphs when helpful.

Context:
{context}"""

    # --- Prompt de condensation (reformule question de suivi en standalone) ---
    condense_template = """Given the conversation history and a follow-up question,
rephrase the follow-up as a standalone question capturing all necessary context.
If the follow-up is already standalone or there is no history, return it as-is.

Chat history:
{chat_history}

Follow-up: {question}

Standalone question:"""

    chaine = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=mq_retriever,
        condense_question_prompt=PromptTemplate(
            input_variables=["chat_history", "question"],
            template=condense_template,
        ),
        combine_docs_chain_kwargs={
            "prompt": ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template("{question}"),
            ])
        },
        return_source_documents=True,
        verbose=False,
    )
    return chaine


# =============================================================================
# Gestion de l'état de session Streamlit
# =============================================================================

def init_session_state():
    """Initialise les variables de session au premier chargement."""
    if "session_id"  not in st.session_state:
        st.session_state.session_id  = nouvelle_session_id()
    if "messages"    not in st.session_state:
        st.session_state.messages    = []
    if "historique"  not in st.session_state:
        st.session_state.historique  = []
    if "session_titre" not in st.session_state:
        st.session_state.session_titre = "Nouvelle conversation"


def charger_session_dans_state(session_id: str):
    """Charge une session depuis le disque dans session_state."""
    data = charger_session(session_id)
    if data:
        st.session_state.session_id    = data["id"]
        st.session_state.session_titre = data["titre"]
        st.session_state.messages      = data["messages"]
        # historique : liste de listes dans JSON → convertir en tuples
        st.session_state.historique    = [
            (h[0], h[1]) for h in data.get("historique", [])
        ]


def nouvelle_conversation():
    """Réinitialise l'état pour une nouvelle session."""
    st.session_state.session_id    = nouvelle_session_id()
    st.session_state.session_titre = "Nouvelle conversation"
    st.session_state.messages      = []
    st.session_state.historique    = []


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar(ollama_ok: bool):
    with st.sidebar:
        # Logo
        st.markdown("""
        <div class="sidebar-logo">
            <div class="logo-text">🧠 DocMind</div>
            <div class="logo-sub">RAG Advanced Engine</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Nouvelle conversation ────────────────────────────────────────
        st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
        if st.button("✏️  Nouvelle conversation", use_container_width=True, key="btn_new"):
            nouvelle_conversation()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Liste des conversations passées ──────────────────────────────
        sessions = lister_sessions()
        if sessions:
            st.markdown(
                f"<p style='color:#8b949e;font-size:0.75rem;font-weight:600;"
                f"text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px'>"
                f"💬 Conversations ({len(sessions)})</p>",
                unsafe_allow_html=True,
            )

            for sess in sessions:
                sid        = sess["id"]
                titre      = sess["titre"]
                date_label = formater_date(sess["modifie_le"])
                nb_msgs    = sess["nb_msgs"] // 2
                is_active  = (sid == st.session_state.get("session_id", ""))

                # Ligne session : titre + bouton supprimer
                col_titre, col_del = st.columns([5, 1])

                with col_titre:
                    css_class = "session-active" if is_active else ""
                    st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
                    label = f"{'▶ ' if is_active else ''}{titre[:38]}{'…' if len(titre)>38 else ''}"
                    if st.button(
                        label,
                        key=f"load_{sid}",
                        use_container_width=True,
                        help=f"{date_label} · {nb_msgs} échange(s)",
                    ):
                        if not is_active:
                            charger_session_dans_state(sid)
                            st.rerun()
                    st.markdown(
                        f"<p style='color:#8b949e;font-size:0.68rem;"
                        f"margin:-6px 0 4px 4px'>{date_label} · {nb_msgs} éch.</p>",
                        unsafe_allow_html=True,
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

                with col_del:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("🗑", key=f"del_{sid}", help="Supprimer"):
                        supprimer_session(sid)
                        if is_active:
                            nouvelle_conversation()
                        st.rerun()
        else:
            st.markdown(
                "<p style='color:#8b949e;font-size:0.82rem;text-align:center;"
                "padding:1rem 0'>Aucune conversation sauvegardée</p>",
                unsafe_allow_html=True,
            )

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Statut système ───────────────────────────────────────────────
        st.markdown("**Statut système**")
        if ollama_ok:
            st.markdown("""<div class="status-card">
                <div class="label">🟢 Ollama</div>
                <div class="value">Connecté · qwen2.5:7b</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="status-card" style="border-color:rgba(239,68,68,0.3)">
                <div class="label">🔴 Ollama</div>
                <div class="value" style="color:#f87171">Non connecté</div>
            </div>""", unsafe_allow_html=True)

        index_ok = base_est_valide()
        if index_ok:
            st.markdown("""<div class="status-card">
                <div class="label">🟢 Index</div>
                <div class="value">Chroma + BM25 chargés</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="status-card" style="border-color:rgba(245,158,11,0.3)">
                <div class="label">🟡 Index</div>
                <div class="value" style="color:#fbbf24">Non indexé</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Pipeline ─────────────────────────────────────────────────────
        st.markdown("**Pipeline**")
        st.markdown(f"""
        <div class="stat-row">
            <span class="stat-pill">🔀 Hybrid</span>
            <span class="stat-pill">🎯 Rerank-{NB_RERANK}</span>
            <span class="stat-pill">🔍 MQ-{NB_MULTI_QUERY}</span>
            <span class="stat-pill">💬 W×{WINDOW_SIZE}</span>
        </div>""", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Formats ──────────────────────────────────────────────────────
        st.markdown("**Formats acceptés**")
        pills = "".join([
            f'<span class="stat-pill" style="margin-bottom:5px">{ic} {lb}</span> '
            for ext, (ic, lb) in FORMATS_SUPPORTES.items()
        ])
        st.markdown(f'<div class="stat-row" style="flex-wrap:wrap">{pills}</div>',
                    unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Aide ─────────────────────────────────────────────────────────
        with st.expander("❓ Comment utiliser"):
            st.markdown("""
**1.** Place tes docs dans `docs/`
```
python ingest.py
```
**2.** Lance l'interface :
```
streamlit run app.py
```
**3.** Tes conversations sont sauvegardées automatiquement et accessibles dans la sidebar.

> 💡 Clique sur une conversation pour la reprendre.
""")


# =============================================================================
# INTERFACE PRINCIPALE
# =============================================================================

def main():
    init_session_state()

    # Vérifications
    index_valide          = base_est_valide()
    ollama_ok, ollama_msg = verifier_ollama()

    # Sidebar
    render_sidebar(ollama_ok)

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 DocMind — Assistant RAG Avancé</h1>
        <p>Hybrid Search · Multi-Query · Reranking · Conversations persistantes</p>
    </div>
    """, unsafe_allow_html=True)

    # Badges
    st.markdown("""
    <div class="badge-container">
        <span class="badge badge-purple">🔀 Hybrid Search</span>
        <span class="badge badge-blue">🎯 FlashRank</span>
        <span class="badge badge-green">🔍 Multi-Query</span>
        <span class="badge badge-orange">💬 Mémoire</span>
        <span class="badge badge-pink">💾 Persistance</span>
        <span class="badge badge-blue">📂 Multi-format</span>
    </div>
    """, unsafe_allow_html=True)

    # Blocages
    if not index_valide:
        manque = []
        if not Path(CHROMA_DIR).exists() or not list(Path(CHROMA_DIR).rglob("*")):
            manque.append("`chroma_db/`")
        if not Path(BM25_PATH).exists():
            manque.append("`bm25_index.pkl`")
        st.error(
            f"⚠️ **Index manquant : {', '.join(manque)}**\n\n"
            "Place des fichiers dans `docs/` puis lance :\n```\npython ingest.py\n```"
        )
        st.stop()

    if not ollama_ok:
        st.error(f"🔴 **Ollama non disponible**\n\n{ollama_msg}")
        st.stop()

    # Chargement retriever
    with st.spinner("Chargement des index…"):
        try:
            retriever = charger_retrievers()
        except Exception as e:
            st.error(f"❌ Erreur index.\n\n`{e}`\n\nRelance `python ingest.py`.")
            st.stop()

    # Titre de la session en cours
    if st.session_state.messages:
        st.markdown(
            f"<p style='color:#8b949e;font-size:0.82rem;margin-bottom:1rem'>"
            f"📝 {st.session_state.session_titre}</p>",
            unsafe_allow_html=True,
        )

    # Affichage de l'historique ou écran de bienvenue
    if not st.session_state.messages:
        formats_liste = " · ".join([
            f"{ic} {lb}" for ext, (ic, lb) in FORMATS_SUPPORTES.items()
        ])
        st.markdown(f"""
        <div class="welcome-box">
            <h3>👋 Bienvenue sur DocMind</h3>
            <p>Pose une question sur tes documents indexés.<br>
            Tes conversations sont sauvegardées automatiquement.</p>
            <div style="margin-top:1rem;padding:0.75rem 1rem;
                        background:rgba(102,126,234,0.1);border-radius:8px;
                        font-size:0.82rem;color:#8b949e;line-height:2;">
                <strong style="color:#a78bfa">Formats supportés :</strong><br>
                {formats_liste}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Input
    question = st.chat_input("Pose ta question sur tes documents…")

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        # Génère titre de session depuis la 1ère question
        if len(st.session_state.messages) == 1:
            st.session_state.session_titre = titre_depuis_question(question)

        with st.chat_message("assistant"):
            placeholder    = st.empty()
            reponse_finale = None
            source_docs    = []

            with st.spinner("🔍 Multi-Query · Hybrid · Reranking · Génération…"):
                try:
                    chaine   = construire_chaine(retriever)
                    hist_rec = st.session_state.historique[-WINDOW_SIZE:]

                    resultat = chaine.invoke({
                        "question":     question,
                        "chat_history": hist_rec,
                    })
                    reponse_finale = resultat.get("answer", "Aucune réponse générée.")
                    source_docs    = resultat.get("source_documents", [])

                    st.session_state.historique.append((question, reponse_finale))

                    # Streaming simulé
                    affichage = ""
                    for mot in reponse_finale.split(" "):
                        affichage += mot + " "
                        placeholder.markdown(affichage + "▌")
                        time.sleep(0.012)
                    placeholder.markdown(reponse_finale)

                except Exception as e:
                    erreur = str(e).lower()
                    if any(m in erreur for m in ["connection", "refused", "timeout"]):
                        st.error("🔴 **Ollama s'est déconnecté.** Vérifie qu'il tourne.")
                    else:
                        st.error(f"❌ **Erreur inattendue**\n\n`{e}`")

            if reponse_finale:
                st.session_state.messages.append(
                    {"role": "assistant", "content": reponse_finale}
                )

                # ── Sauvegarde automatique sur disque ──────────────────────
                sauvegarder_session(
                    session_id = st.session_state.session_id,
                    titre      = st.session_state.session_titre,
                    messages   = st.session_state.messages,
                    historique = st.session_state.historique,
                )

                # ── Sources ────────────────────────────────────────────────
                if source_docs:
                    sources = extraire_sources(source_docs)
                    with st.expander(
                        f"📎 {len(sources)} source(s) · {len(source_docs)} fragment(s)"
                    ):
                        cols = st.columns(min(len(sources), 3))
                        for i, s in enumerate(sources):
                            cols[i % len(cols)].markdown(f"📄 `{s}`")
                        st.markdown("---")
                        st.markdown(
                            "<p style='color:#8b949e;font-size:0.8rem;margin-bottom:8px'>"
                            "Extraits après compression contextuelle :</p>",
                            unsafe_allow_html=True,
                        )
                        for i, doc in enumerate(source_docs[:NB_RERANK], 1):
                            fname = Path(doc.metadata.get("source", "?")).name
                            page  = doc.metadata.get("page", "")
                            label = f"**Fragment {i}** — `{fname}`"
                            if page != "":
                                label += f" · p.{page + 1}"
                            st.markdown(label)
                            st.markdown(
                                f'<div class="fragment-card">{doc.page_content}</div>',
                                unsafe_allow_html=True,
                            )
                            if i < len(source_docs[:NB_RERANK]):
                                st.markdown(
                                    "<hr style='margin:6px 0;"
                                    "border-color:rgba(255,255,255,0.06)'>",
                                    unsafe_allow_html=True,
                                )


# =============================================================================
# Point d'entrée
# =============================================================================
if __name__ == "__main__":
    main()
