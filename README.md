# RAG Avancé — Assistant Documentaire Local

Système RAG (*Retrieval-Augmented Generation*) avancé, 100 % open source, tournant entièrement en local sur Windows.

> Architecture production-grade : **Hybrid Search + Reranking + Compression Contextuelle + Mémoire Conversationnelle**

---

## Stack Technique

| Couche | Outil / Librairie | Version |
|---|---|---|
| **LLM local** | Ollama + Qwen2.5:7b | — |
| **Orchestration** | LangChain | 0.3.25 |
| **Embeddings** | sentence-transformers / all-mpnet-base-v2 | 3.4.1 |
| **Vector store** | ChromaDB | 1.0.9 |
| **Recherche sparse** | BM25 (rank-bm25) | 0.2.2 |
| **Reranking** | FlashRank (CrossEncoder) | 0.2.10 |
| **Interface** | Streamlit | 1.45.1 |
| **Chargement docs** | PyPDF, python-docx, python-pptx, openpyxl | — |
| **Langage** | Python | 3.11 |

---

## Améliorations vs RAG Naïf

| Fonctionnalité | RAG Naïf | RAG Avancé |
|---|---|---|
| Formats supportés | PDF, TXT | PDF, DOCX, PPTX, XLSX, CSV, TXT, MD, HTML, JSON |
| Retrieval | Dense seul (Chroma) | Hybrid : Dense (60%) + BM25 (40%) |
| Classement | Ordre cosine | CrossEncoder Reranking (FlashRank) |
| Contexte envoyé au LLM | Chunks bruts | Compression contextuelle |
| Mémoire | Aucune | N derniers échanges |
| Embeddings | all-MiniLM-L6-v2 | all-mpnet-base-v2 (+30% qualité) |
| Streaming | Non | Oui |

---

## Prérequis

### Python 3.11
```powershell
python --version   # doit afficher Python 3.11.x
```

### Ollama
```powershell
# 1. Installer : https://ollama.com/download

# 2. Vérifier qu'Ollama tourne
curl http://localhost:11434
# Réponse attendue : Ollama is running

# 3. Télécharger le modèle (~4,7 Go)
ollama pull qwen2.5:7b
```

---

## Installation

```powershell
# 1. Aller dans le dossier du projet
cd C:\Users\Asus\Downloads\RAG_Advanced\rag_cours

# 2. Créer l'environnement virtuel
python -m venv .venv

# 3. Activer le venv
.\.venv\Scripts\Activate.ps1

# 4. Installer les dépendances
pip install -r requirements.txt
```

---

## Utilisation

### Étape 1 — Indexer les documents

Place tes fichiers dans le dossier `docs/` (les sous-dossiers sont indexés automatiquement), puis lance :

```powershell
python ingest.py
```

Formats supportés : `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.csv`, `.txt`, `.md`, `.html`, `.json`

Cela génère deux index :
- `chroma_db/` — index vectoriel dense
- `bm25_index.pkl` — index lexical sparse

### Étape 2 — Lancer l'interface

```powershell
streamlit run app.py
```

Ouvre **http://localhost:8501** dans ton navigateur.

---

## Structure du projet

```
rag_cours/
├── .venv/              ← Environnement virtuel (non versionné)
├── .streamlit/
│   └── config.toml
├── docs/               ← Documents sources
├── chroma_db/          ← Index dense (généré par ingest.py)
├── bm25_index.pkl      ← Index sparse BM25 (généré par ingest.py)
├── ingest.py           ← Script d'indexation
├── app.py              ← Interface Streamlit
├── requirements.txt
└── README.md
```

---

## Dépannage

**"Index manquant : chroma_db/, bm25_index.pkl"**
→ Lance `python ingest.py` après avoir placé des fichiers dans `docs/`.

**"Impossible de contacter Ollama"**
```powershell
ollama serve
ollama pull qwen2.5:7b
```

**Erreur FlashrankRerank**
```powershell
pip install flashrank==0.2.10
```

**Erreur d'activation du venv PowerShell**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**PDF non lisible**
→ PDF scanné sans couche texte. Utilise `ocrmypdf` pour ajouter une couche texte.
