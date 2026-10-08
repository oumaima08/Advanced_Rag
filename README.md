# 🧠 RAG Avancé — Assistant Documentaire Local

Système RAG (*Retrieval-Augmented Generation*) avancé, 100 % open source,
tournant entièrement en local sur Windows.

> Évolution complète du RAG naïf vers une architecture production-grade :
> **Hybrid Search + Reranking + Compression Contextuelle + Mémoire Conversationnelle**

---

## Ce qui a changé par rapport au RAG naïf

| Fonctionnalité | RAG Naïf | RAG Avancé |
|---|---|---|
| Formats de documents | PDF, TXT | **PDF, DOCX, PPTX, XLSX, CSV, TXT, MD, HTML, JSON** |
| Retrieval | Dense (Chroma seul) | **Hybrid : Dense + BM25** |
| Classement | Aucun (ordre cosine) | **CrossEncoder Reranking (FlashRank)** |
| Contexte envoyé au LLM | Chunks bruts entiers | **Compression contextuelle** |
| Conversation | Sans mémoire | **Mémoire des N derniers échanges** |
| Embeddings | all-MiniLM-L6-v2 (~90 Mo) | **all-mpnet-base-v2 (~420 Mo, +30% qualité)** |
| Streaming | Non | **Oui (mot par mot)** |

---

## Architecture détaillée

```
┌──────────────────────────────────────────────────────────────────┐
│                    INDEXATION  (ingest.py)                       │
│                                                                  │
│  docs/          Chunking        Embeddings       chroma_db/     │
│  (PDF/TXT)  → (512 car.)  →  (mpnet-base-v2) →  [DENSE]        │
│                                                                  │
│                              BM25 index     →  bm25_index.pkl   │
│                              (lexical)          [SPARSE]        │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                  INTERROGATION  (app.py)                         │
│                                                                  │
│  Historique +    Condensation    Question       EnsembleRetriever│
│  Question     →  (standalone) →  embedding  →  Dense (60%)      │
│                                               + BM25  (40%)     │
│                                               = 20 candidats    │
│                                                    ↓            │
│                                          FlashRank Reranker     │
│                                          (CrossEncoder local)   │
│                                          = Top-4 re-classés     │
│                                                    ↓            │
│                                     Compression Contextuelle    │
│                                     (filtre phrases inutiles)   │
│                                                    ↓            │
│                                       Prompt → Ollama API REST  │
│                                                localhost:11434  │
│                                                    ↓            │
│                             Réponse streamée + Sources + Extraits│
└──────────────────────────────────────────────────────────────────┘
```

**Pourquoi Hybrid Search ?**
La recherche dense (embeddings) retrouve les passages *sémantiquement* proches.
La recherche sparse (BM25) retrouve les passages contenant exactement les *mots-clés*.
La combinaison des deux couvre les cas que chacun rate seul.

**Pourquoi le Reranking ?**
Le retriever retourne 20 candidats vite. Le CrossEncoder (plus lent mais précis)
les ré-évalue un par un et ne garde que les 4 vraiment pertinents.

**Pourquoi la Compression Contextuelle ?**
Un chunk de 512 caractères contient souvent des phrases hors-sujet.
Le compresseur ne transmet au LLM que les phrases directement utiles.
Le prompt est plus court, la réponse plus précise.

---

## Prérequis

### 1. Python 3.11

```powershell
python --version   # doit afficher Python 3.11.x
```

### 2. Ollama

**a) Installer Ollama** — [https://ollama.com/download](https://ollama.com/download)

**b) Vérifier qu'Ollama tourne :**
```powershell
curl http://localhost:11434
# Réponse attendue : Ollama is running
```

**c) Télécharger le modèle (~4,7 Go) :**
```powershell
ollama pull qwen2.5:7b
```

**d) Vérifier :**
```powershell
ollama list   # doit afficher qwen2.5:7b
```

---

## Installation

```powershell
# 1. Aller dans le dossier du projet
cd C:\Users\Asus\Downloads\RAG_Basic\rag_cours

# 2. Créer l'environnement virtuel
python -m venv .venv

# 3. Activer le venv (PowerShell)
.\.venv\Scripts\Activate.ps1
# Si bloqué : Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 4. Installer les dépendances
pip install -r requirements.txt
```

> La première installation télécharge ~500 Mo de modèles ML.

---

## Utilisation en deux étapes

### ▶ Étape 1 — Indexer les documents

Place tes fichiers dans le dossier `docs/`. Les formats supportés sont :

| Format | Extension | Prérequis |
|--------|-----------|-----------|
| PDF | `.pdf` | inclus |
| Word | `.docx` | `python-docx` |
| PowerPoint | `.pptx` | `python-pptx` |
| Excel | `.xlsx` | `openpyxl` |
| CSV | `.csv` | inclus |
| Texte brut | `.txt` | inclus |
| Markdown | `.md` | inclus |
| HTML | `.html` `.htm` | inclus |
| JSON | `.json` | inclus (structure `[{"text":"..."}]`) |

> Les fichiers dans des **sous-dossiers** sont également indexés automatiquement.

Lance ensuite l'indexation :

```powershell
python ingest.py
```

Ce script crée **deux index** :
- `chroma_db/` — index vectoriel dense (embeddings)
- `bm25_index.pkl` — index lexical sparse (BM25)

> Relance `python ingest.py` à chaque ajout de nouveaux documents.

Sortie attendue :
```
=================================================================
  RAG Avancé — Script d'indexation (Dense + Sparse)
=================================================================
...
  ✅  Indexation terminée !
  📊  87 chunks indexés dans Chroma ET BM25.
  🧠  Embeddings : sentence-transformers/all-mpnet-base-v2
  🔍  Index BM25 : ./bm25_index.pkl
  ▶️   Lance maintenant : streamlit run app.py
=================================================================
```

### ▶ Étape 2 — Lancer l'interface

```powershell
streamlit run app.py
```

Ouvre **http://localhost:8501** dans ton navigateur.

L'interface affiche :
- ✅ Confirmation de connexion Ollama
- 📂 Liste des formats supportés dans la sidebar et l'écran de bienvenue
- Un panneau latéral avec les paramètres actifs
- Un chat avec mémoire des échanges précédents
- Les sources et extraits compressés utilisés pour chaque réponse

---

## Idées d'améliorations futures

### 1. Multi-query Retrieval
Générer automatiquement 3-5 reformulations de la question avec le LLM,
lancer un retrieval par reformulation, puis dédupliquer les résultats.
Capture les passages que la question originale raterait.

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
```

### 2. RAG-Fusion (Reciprocal Rank Fusion amélioré)
Variante de Multi-query où les résultats de plusieurs requêtes sont fusionnés
par RRF avant le reranking. Amélioration mesurable sur les benchmarks BEIR.

### 3. Parent Document Retriever
Indexer de petits chunks (128 car.) pour la précision du retrieval,
mais envoyer au LLM le document parent plus large (1024 car.) pour le contexte.
Meilleur compromis précision / contexte.

### 4. Self-RAG (Retrieval auto-évalué)
Le LLM décide lui-même si un retrieval est nécessaire, évalue la pertinence
des passages récupérés, et critique sa propre réponse avant de la valider.
Nécessite un modèle capable de suivre des instructions structurées.

### 5. Graphe de connaissances (GraphRAG)
Extraire des entités et relations des documents pour construire un graphe
(Neo4j ou NetworkX). Permet des requêtes relationnelles impossibles avec
la recherche vectorielle seule ("Qui travaille avec qui ?").

### 6. Support de nouveaux formats
- `.docx` via `python-docx`
- Pages web via `WebBaseLoader`
- Vidéos/audio via Whisper (transcription automatique)
- Images avec texte via `pytesseract` (OCR)

---

## Structure du projet

```
rag_cours/
├── .venv/              ← Environnement virtuel (ne pas versionner)
├── .streamlit/
│   └── config.toml     ← Désactive le file watcher torch
├── docs/               ← Tous tes documents sources (PDF, DOCX, PPTX, XLSX, CSV, TXT, MD, HTML, JSON)
│   └── sous-dossier/   ← Les sous-dossiers sont indexés automatiquement
├── chroma_db/          ← Index dense (généré par ingest.py)
├── bm25_index.pkl      ← Index sparse BM25 (généré par ingest.py)
├── ingest.py           ← Script d'indexation multi-format
├── app.py              ← Interface Streamlit avancée
├── requirements.txt
└── README.md
```

---

## Dépannage

### ❌ "Index manquant : chroma_db/, bm25_index.pkl"
Lance `python ingest.py` après avoir placé des fichiers dans `docs/`.

### ❌ "Impossible de contacter Ollama"
```powershell
ollama serve          # démarre le serveur
ollama list           # vérifie que qwen2.5:7b est présent
ollama pull qwen2.5:7b  # si absent
```

### ❌ Erreur à l'import de FlashrankRerank
```powershell
pip install flashrank==0.2.10
```

### ⚠️ "Impossible de lire 'fichier.pdf'" dans ingest.py
PDF scanné (image sans couche texte) ou protégé par mot de passe.
Utilise `ocrmypdf` pour ajouter une couche texte.

### ❌ Erreur sur un fichier Word/PowerPoint/Excel
Ces formats nécessitent des dépendances supplémentaires :
```powershell
pip install python-docx==1.2.0 python-pptx==1.0.2 openpyxl==3.1.5
```

### ❌ Fichier JSON non chargé
Le loader JSON attend ce format : `[{"text": "contenu..."}, ...]`
Si ta structure est différente, modifie `jq_schema` dans le dictionnaire `LOADERS` de `ingest.py`.

### ❌ Réponses hors-sujet
- Vérifie que `ingest.py` s'est terminé sans erreur
- Reformule la question avec des mots présents dans tes documents
- Augmente `NB_DENSE` et `NB_SPARSE` dans `app.py` pour plus de candidats

### ❌ Erreur d'activation du venv PowerShell
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
