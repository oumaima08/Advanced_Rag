# 🧠 DocMind — Architecture Interne Détaillée

> Ce document décrit **de A à Z** tout ce qui se passe en interne dans le projet,
> chaque modèle utilisé, chaque méthode choisie, et **pourquoi** ces choix ont été faits.

---

## Table des matières

1. [Vue d'ensemble du système](#1-vue-densemble-du-système)
2. [Phase 1 — Indexation (ingest.py)](#2-phase-1--indexation-ingestpy)
   - 2.1 Chargement des documents
   - 2.2 Chunking
   - 2.3 Embeddings — all-mpnet-base-v2
   - 2.4 Index dense — ChromaDB
   - 2.5 Index sparse — BM25
3. [Phase 2 — Interrogation (app.py)](#3-phase-2--interrogation-apppy)
   - 3.1 Multi-Query Retriever
   - 3.2 Hybrid Search (EnsembleRetriever)
   - 3.3 Reranking — FlashRank CrossEncoder
   - 3.4 Compression Contextuelle
   - 3.5 Gestion de l'historique (fenêtre glissante)
   - 3.6 Condensation de la question
   - 3.7 Construction du prompt final
   - 3.8 LLM — Qwen 2.5 7B via Ollama
   - 3.9 Streaming de la réponse
4. [Phase 3 — Persistance des conversations](#4-phase-3--persistance-des-conversations)
5. [Pourquoi ces choix techniques ?](#5-pourquoi-ces-choix-techniques-)
6. [Flux de données complet — schéma](#6-flux-de-données-complet--schéma)
7. [Limites actuelles et pistes d'amélioration](#7-limites-actuelles-et-pistes-damélioration)

---

## 1. Vue d'ensemble du système

Le projet est un système **RAG (Retrieval-Augmented Generation)**.

L'idée fondamentale du RAG est simple :
> Au lieu de demander au LLM de répondre uniquement depuis sa mémoire
> (ce qui produit des hallucinations), on lui **donne le contexte exact**
> extrait de tes documents avant qu'il génère sa réponse.

Le système se décompose en **deux phases distinctes** :

```
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 1 — INDEXATION (une seule fois, ou à chaque ajout de docs)  │
│                                                                     │
│  Tes documents  →  Découpage  →  Vectorisation  →  Stockage        │
│  (PDF, DOCX...)     (chunks)     (embeddings)      (Chroma + BM25) │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 2 — INTERROGATION (à chaque question)                       │
│                                                                     │
│  Question  →  Multi-Query  →  Hybrid Search  →  Reranking          │
│            →  Compression  →  Prompt + Contexte  →  LLM  →  Réponse│
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Phase 1 — Indexation (`ingest.py`)

### 2.1 Chargement des documents

**Ce qui se passe :**
Le script parcourt récursivement le dossier `docs/` et charge chaque fichier
selon son extension avec le loader LangChain adapté.

| Format | Loader utilisé | Pourquoi ce loader |
|--------|---------------|-------------------|
| `.pdf` | `PyPDFLoader` | Extrait le texte page par page, gère les métadonnées de page |
| `.docx` | `Docx2txtLoader` | Extrait paragraphes et tableaux Word sans dépendance lourde |
| `.pptx` | `UnstructuredPowerPointLoader` | Extrait le texte de chaque slide |
| `.xlsx` | `UnstructuredExcelLoader` | Convertit les cellules en texte lisible |
| `.csv` | `CSVLoader` | Chaque ligne devient un document séparé |
| `.txt` | `TextLoader` | Lecture directe UTF-8 (encodage explicite pour Windows) |
| `.md` | `TextLoader` | Markdown traité comme texte brut |
| `.html` | `BSHTMLLoader` | BeautifulSoup retire les balises HTML |
| `.json` | `JSONLoader` | Extrait un champ texte via jq_schema |

**Pourquoi cette approche :**
Un seul pipeline unifié gère tous les formats. Si un fichier est illisible
(PDF scanné, fichier corrompu), il est ignoré avec un avertissement et
l'indexation continue — pas de crash total.

**Métadonnée `source` :**
Chaque document chargé reçoit automatiquement la métadonnée `source`
(chemin du fichier). Cela permet d'afficher le nom du fichier source
dans l'interface après chaque réponse.

---

### 2.2 Chunking — RecursiveCharacterTextSplitter

**Ce qui se passe :**
Chaque document est découpé en morceaux (chunks) plus petits.

**Paramètres choisis :**
- `chunk_size = 512` caractères
- `chunk_overlap = 100` caractères
- `separators = ["\n\n", "\n", ".", " ", ""]`

**Pourquoi 512 et pas 800 (valeur du RAG naïf) :**
Plus le chunk est court, plus le CrossEncoder (reranker) peut évaluer
précisément sa pertinence. Un chunk de 512 caractères (~80 mots) est
suffisant pour contenir une idée complète sans trop de bruit autour.

**Pourquoi un chevauchement de 100 :**
Si une réponse se trouve à cheval entre deux chunks (fin de paragraphe),
le chevauchement garantit qu'elle est capturée dans au moins un des deux.
Sans chevauchement, des informations critiques pourraient être coupées.

**Pourquoi RecursiveCharacterTextSplitter :**
Ce splitter essaie de couper d'abord aux paragraphes (`\n\n`), puis
aux lignes (`\n`), puis aux phrases (`.`), puis aux mots (` `).
Il respecte la structure naturelle du texte au lieu de couper brutalement.

---

### 2.3 Modèle d'Embeddings — `all-mpnet-base-v2`

**Ce qui se passe :**
Chaque chunk est transformé en un vecteur numérique de 768 dimensions.
Ce vecteur représente le **sens sémantique** du texte dans un espace mathématique.
Deux textes similaires auront des vecteurs proches (cosine similarity élevée).

**Modèle choisi :** `sentence-transformers/all-mpnet-base-v2`

**Pourquoi ce modèle et pas d'autres :**

| Modèle | Dimensions | Taille | Score SBERT | Choix |
|--------|-----------|--------|------------|-------|
| all-MiniLM-L6-v2 | 384 | ~90 Mo | 68.7 | ❌ (RAG naïf) |
| **all-mpnet-base-v2** | **768** | **~420 Mo** | **69.6** | **✅ (ce projet)** |
| all-MiniLM-L12-v2 | 384 | ~120 Mo | 69.0 | Alternative |
| e5-large | 1024 | ~1.3 Go | 71.2 | Trop lourd pour CPU |

`all-mpnet-base-v2` offre le meilleur compromis **qualité / taille** pour
une exécution CPU locale. Il surpasse MiniLM de ~1 point sur les benchmarks
de recherche sémantique tout en restant téléchargeable en quelques minutes.

**Paramètres importants :**
- `device="cpu"` : pas de GPU requis, tourne sur n'importe quelle machine
- `normalize_embeddings=True` : normalise les vecteurs à longueur 1 pour
  que la similarité cosine soit équivalente au produit scalaire (plus rapide)

---

### 2.4 Index Dense — ChromaDB

**Ce qui se passe :**
Les vecteurs d'embeddings sont stockés dans ChromaDB, une base de données
vectorielle locale persistée dans le dossier `chroma_db/`.

**Pourquoi ChromaDB :**
- 100 % local, pas de serveur externe
- Persistence automatique sur disque (pas besoin de tout recalculer à chaque lancement)
- Interface Python simple, compatible LangChain nativement
- Recherche par similarité cosine en millisecondes même sur CPU

**Ce qui est stocké :**
Pour chaque chunk → vecteur 768 dimensions + texte original + métadonnées
(source, numéro de page). La base Chroma peut être interrogée pour retourner
les N chunks les plus proches sémantiquement d'une question.

**Recherche Dense :**
Quand une question arrive, elle est aussi transformée en vecteur (avec le
même modèle d'embeddings). Chroma retourne les chunks dont le vecteur est
le plus proche = les plus **sémantiquement similaires**.

---

### 2.5 Index Sparse — BM25

**Ce qui se passe :**
En parallèle de Chroma, un index BM25 est construit sur les mêmes chunks
et sauvegardé dans `bm25_index.pkl` (sérialisation Python pickle).

**Qu'est-ce que BM25 :**
BM25 (Best Match 25) est un algorithme de recherche **lexicale** basé sur
la fréquence des mots-clés. C'est l'algorithme derrière les moteurs de
recherche classiques (Google Search avant 2015, Elasticsearch par défaut).

**Pourquoi BM25 en plus des embeddings :**
Les deux approches sont complémentaires et ont des forces différentes :

| Situation | Dense (embeddings) | Sparse (BM25) |
|-----------|-------------------|---------------|
| "Qu'est-ce que le deep learning ?" | ✅ Comprend le sens | ✅ Trouve les mots-clés |
| "Qwen 2.5 7B" (terme technique exact) | ❌ Peut rater | ✅ Trouve exactement |
| Question paraphrasée | ✅ Comprend la paraphrase | ❌ Mots différents = raté |
| Sigle ou acronyme (NLP, RAG, BERT) | ❌ Dépend du training | ✅ Match exact |

Combiner les deux = **Hybrid Search** = couvre les failles de chacun.

---

## 3. Phase 2 — Interrogation (`app.py`)

### 3.1 Multi-Query Retriever

**Ce qui se passe :**
Avant de chercher dans les index, le LLM génère **3 reformulations**
différentes de la question originale.

**Exemple :**
```
Question originale : "c'est quoi PeerRead ?"

Reformulations générées automatiquement :
  1. "What is the PeerRead dataset?"
  2. "Describe the PeerRead peer review collection"
  3. "PeerRead scientific paper review dataset description"
```

Chaque reformulation déclenche une recherche hybride séparée.
Les résultats sont fusionnés et dédupliqués avant le reranking.

**Pourquoi Multi-Query :**
Une question courte ou mal formulée peut rater des passages importants.
En générant plusieurs formulations, on maximise la couverture du retrieval.
C'est particulièrement utile pour :
- Les questions très courtes ("c'est quoi X ?")
- Les questions en langue différente des documents
- Les questions ambiguës

**Coût :** Un appel LLM supplémentaire (~1-2 secondes) avant le retrieval.
Ce surcoût est acceptable au vu du gain de qualité.

---

### 3.2 Hybrid Search — EnsembleRetriever

**Ce qui se passe :**
Les résultats de Chroma (dense) et BM25 (sparse) sont fusionnés par
l'algorithme **Reciprocal Rank Fusion (RRF)**.

**Pondération choisie :**
- Dense (Chroma) : **60%**
- Sparse (BM25) : **40%**

**Pourquoi 60/40 et pas 50/50 :**
Les embeddings capturent mieux le sens général → légèrement prioritaires.
BM25 reste important pour les termes techniques exacts → 40% suffisant.
Cette pondération est empiriquement reconnue dans la littérature RAG.

**Comment fonctionne RRF :**
Chaque retriever classe ses résultats de 1 à N.
RRF calcule un score combiné : `score = 1/(rang_dense + 60) + 1/(rang_sparse + 60)`
Le constante 60 réduit l'impact des rangs très élevés (résultats peu pertinents).
Les chunks avec les meilleurs scores fusionnés sont retenus.

**Résultat :** 20 chunks candidats (10 dense + 10 sparse, dédupliqués).

---

### 3.3 Reranking — FlashRank CrossEncoder

**Ce qui se passe :**
Les 20 chunks candidats sont re-évalués par un **CrossEncoder** pour
ne garder que les **4 meilleurs** (NB_RERANK=4).

**Différence Bi-encoder vs CrossEncoder :**

```
Bi-encoder (embeddings) :
  Question → vecteur Q
  Chunk    → vecteur C
  Score    = cosine(Q, C)   ← rapide, mais approximatif

CrossEncoder :
  [Question + Chunk] → Score de pertinence   ← plus lent, mais précis
```

Le CrossEncoder lit la question ET le chunk **ensemble** et évalue leur
relation directe. Il est beaucoup plus précis qu'un simple score cosine
car il comprend les nuances et la relation question-réponse.

**Pourquoi FlashRank et pas un CrossEncoder HuggingFace classique :**
FlashRank utilise des modèles ONNX optimisés qui tournent ~10x plus vite
qu'un CrossEncoder classique sur CPU, sans perte de qualité significative.
Pour un usage interactif, cette rapidité est essentielle.

**Résultat :** 4 chunks réordonnés par pertinence réelle.

---

### 3.4 Compression Contextuelle — ContextualCompressionRetriever

**Ce qui se passe :**
Même après reranking, un chunk de 512 caractères contient souvent des
phrases hors-sujet. Le compresseur identifie et supprime ces phrases
pour ne garder que les passages directement utiles à la question.

**Exemple :**
```
Chunk brut (512 car.) :
  "...introduction générale sur les datasets NLP. PeerRead est une
   collection de peer reviews scientifiques comportant 14K reviews
   de papiers ICLR, ACL et NeurIPS. Le dataset inclut aussi les
   scores d'acceptation. Par ailleurs, les auteurs ont également
   travaillé sur d'autres projets non liés..."

Après compression (question : "qu'est-ce que PeerRead ?") :
  "PeerRead est une collection de peer reviews scientifiques
   comportant 14K reviews de papiers ICLR, ACL et NeurIPS.
   Le dataset inclut aussi les scores d'acceptation."
```

**Avantage :**
Le prompt envoyé au LLM est plus court et plus précis → réponse de
meilleure qualité, moins d'hallucinations, tokens économisés.

---

### 3.5 Gestion de l'historique — Fenêtre glissante

**Ce qui se passe :**
L'historique de conversation est stocké dans `st.session_state.historique`
sous forme de liste de tuples `(question, réponse)`.

Seuls les **5 derniers échanges** (WINDOW_SIZE=5) sont transmis au LLM
à chaque nouvelle question.

**Pourquoi une fenêtre glissante et pas tout l'historique :**
- Les LLMs ont une limite de contexte (tokens). Envoyer 50 échanges
  pourrait dépasser cette limite.
- Les échanges récents sont plus pertinents pour la question actuelle.
- Réduire le contexte = réponse plus rapide.

**Pourquoi ne pas utiliser ConversationBufferWindowMemory de LangChain :**
Cette classe est dépréciée en LangChain 0.3.x et cause des bugs quand
la chaîne est recréée à chaque question (ce qu'on doit faire dans Streamlit
car Streamlit réexécute le script entier à chaque interaction).
La gestion manuelle dans `session_state` est plus fiable et transparente.

---

### 3.6 Condensation de la question

**Ce qui se passe :**
Avant le retrieval, le LLM reformule la question en tenant compte de
l'historique pour créer une question **standalone** (autonome).

**Exemple :**
```
Historique :
  User : "Qu'est-ce que PeerRead ?"
  Bot  : "C'est un dataset de peer reviews..."

Question de suivi : "Combien de reviews contient-il ?"

Question condensée : "Combien de reviews contient le dataset PeerRead ?"
```

Sans cette étape, la recherche hybride ne comprendrait pas à quoi "il"
fait référence et retournerait des résultats non pertinents.

**Pourquoi c'est fait par le LLM et pas par règle :**
La résolution de coréférences ("il", "ce système", "cette méthode") est
un problème de NLP complexe que les LLMs gèrent très bien naturellement.

---

### 3.7 Construction du prompt final

**Ce qui se passe :**
Un prompt structuré est assemblé avec :
1. Les instructions système (rôle, contraintes, langue)
2. Le contexte (les 4 chunks compressés)
3. La question de l'utilisateur

**Template système :**
```
You are a helpful and precise document assistant.
Answer the question using ONLY the context provided below.
If the answer is not in the context, clearly say so — do not invent anything.
Always reply in the same language as the question.
Structure your answer clearly with bullet points or paragraphs when helpful.

Context:
{context}
```

**Pourquoi "ONLY the context" :**
Sans cette contrainte, le LLM mélange ses connaissances générales avec
le contenu des documents, ce qui produit des réponses invérifiables.
Cette instruction force le LLM à rester dans le périmètre des documents.

**Pourquoi "same language as the question" :**
Qwen 2.5 est multilingue. Cette instruction assure que la réponse est
dans la même langue que la question (français si question en français,
anglais si en anglais, etc.) sans configuration supplémentaire.

---

### 3.8 LLM — Qwen 2.5 7B via Ollama

**Ce qui se passe :**
Le prompt complet est envoyé à Ollama via une requête HTTP POST sur
`http://localhost:11434/api/chat`. Ollama exécute le modèle localement
et retourne la réponse token par token.

**Modèle choisi :** `qwen2.5:7b`

**Pourquoi Qwen 2.5 7B :**

| Modèle | Params | RAM nécessaire | Qualité | Langue FR |
|--------|--------|---------------|---------|-----------|
| llama3.2:3b | 3B | ~3 Go | Basique | Moyenne |
| mistral:7b | 7B | ~5 Go | Bonne | Bonne |
| **qwen2.5:7b** | **7B** | **~5 Go** | **Très bonne** | **Excellente** |
| llama3.1:8b | 8B | ~6 Go | Très bonne | Bonne |
| mixtral:8x7b | 47B | ~32 Go | Excellente | Excellente |

Qwen 2.5 7B a été spécifiquement entraîné sur des données multilingues
incluant beaucoup de français. Il excelle particulièrement dans les tâches
de compréhension de texte et de synthèse documentaire — exactement ce
dont un RAG a besoin.

**Pourquoi Ollama et pas HuggingFace directement :**
Ollama gère le chargement du modèle en mémoire, la quantification
(réduction de taille), et l'API HTTP de manière transparente. Sans Ollama,
charger un LLM 7B en Python nécessiterait 8-16 Go de RAM et des dépendances
complexes (transformers, accelerate, bitsandbytes...).

**Paramètre temperature=0.3 :**
Une temperature basse (0 = déterministe, 1 = créatif) est choisie pour
un assistant documentaire. On veut des réponses précises et factuelles,
pas créatives. 0.3 laisse assez de flexibilité pour formuler des réponses
naturelles sans divaguer.

**Connexion via API REST locale :**
L'utilisation de `ChatOllama(base_url="http://localhost:11434")` avec
`langchain-ollama` est préférée au client CLI Ollama car :
- Plus stable sur Windows
- Pas besoin d'avoir `ollama` dans le PATH système
- La vérification de connectivité (`GET /api/tags`) peut être faite
  indépendamment avant même de charger le modèle

---

### 3.9 Streaming de la réponse

**Ce qui se passe :**
La réponse complète est générée par le LLM, puis affichée mot par mot
dans l'interface avec un délai de 12ms entre chaque mot.

**Pourquoi streaming simulé et pas vrai streaming :**
`ConversationalRetrievalChain` de LangChain ne supporte pas le vrai
streaming natif (les tokens arrivent un par un) car la chaîne a besoin
de la réponse complète pour mettre à jour la mémoire et retourner les
source_documents. Le streaming simulé reproduit l'effet visuel tout en
restant simple à implémenter.

**Vitesse choisie : 12ms par mot (~80 mots/sec) :**
Assez rapide pour paraître fluide, assez lent pour que l'utilisateur
perçoive l'effet de génération en temps réel.

---

## 4. Phase 3 — Persistance des conversations

### Comment ça fonctionne

**Fichier responsable :** `conversations_manager.py`

Chaque conversation est un fichier JSON dans `conversations/` :

```
conversations/
├── 2026-09-20_14-35-12.json
├── 2026-09-21_09-10-44.json
└── 2026-09-22_17-22-05.json
```

**Cycle de vie d'une session :**

```
1. Lancement app.py
      ↓
2. init_session_state() → crée un nouvel ID si aucune session active
      ↓
3. Utilisateur pose une question
      ↓
4. LLM génère la réponse
      ↓
5. sauvegarder_session() → écrit/met à jour le JSON sur disque
      ↓
6. Sidebar : lister_sessions() relit tous les JSONs → affiche la liste
      ↓
7. Clic sur une session → charger_session() relit le JSON
                        → recharge messages + historique dans session_state
      ↓
8. L'utilisateur peut continuer la conversation depuis ce point
```

**Pourquoi JSON et pas SQLite ou autre base de données :**
- Lisible directement avec un éditeur de texte
- Pas de serveur ni de schéma à gérer
- Facilement exportable, sauvegardable, transférable
- Suffisant pour des centaines de conversations

**Titre automatique :**
Le titre de chaque session est généré depuis la **première question**
(tronquée à 50 caractères). Cela reproduit le comportement de ChatGPT
qui nomme les conversations d'après leur premier message.

---

## 5. Pourquoi ces choix techniques ?

### Pourquoi LangChain ?

LangChain fournit des abstractions standardisées pour tous les composants
(loaders, splitters, retrievers, chains, prompts). Sans LangChain, chaque
composant devrait être connecté manuellement avec du code spécifique.
Les versions sont figées (`==`) dans `requirements.txt` pour éviter les
breaking changes des futures versions.

### Pourquoi Streamlit et pas Flask/FastAPI ?

Streamlit permet de créer une interface web interactive en Python pur,
sans HTML/CSS/JavaScript. Pour un projet de démonstration et d'apprentissage,
c'est le bon outil. Flask/FastAPI seraient plus adaptés pour une API REST
ou un déploiement en production.

### Pourquoi tout en local ?

- **Confidentialité** : les documents ne quittent jamais la machine
- **Coût zéro** : pas d'API payante (OpenAI, Anthropic, Cohere...)
- **Disponibilité** : fonctionne sans connexion internet
- **Contrôle** : on sait exactement quel modèle tourne et comment

### Pourquoi deux index séparés (Chroma + BM25) ?

Un seul index dense suffit pour ~80% des questions. Mais pour les 20%
restants (termes techniques, acronymes, noms propres), BM25 est supérieur.
Le surcoût en stockage et en temps d'indexation est négligeable comparé
au gain de qualité sur ces cas difficiles.

---

## 6. Flux de données complet — schéma

```
╔══════════════════════════════════════════════════════════════════╗
║                    INDEXATION (ingest.py)                        ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  docs/                                                           ║
║  ├── paper.pdf    ──┐                                            ║
║  ├── notes.docx   ──┤ PyPDFLoader / Docx2txtLoader / ...        ║
║  └── data.csv     ──┘                                            ║
║           │                                                      ║
║           ▼                                                      ║
║  RecursiveCharacterTextSplitter                                  ║
║  (chunk_size=512, overlap=100)                                   ║
║           │                                                      ║
║           ├──────────────────────────────────┐                  ║
║           ▼                                  ▼                  ║
║  HuggingFaceEmbeddings               BM25Retriever              ║
║  (all-mpnet-base-v2, 768 dims)        (index lexical)            ║
║           │                                  │                  ║
║           ▼                                  ▼                  ║
║     chroma_db/                        bm25_index.pkl            ║
║     (index vectoriel dense)           (index sparse)             ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════╗
║                  INTERROGATION (app.py)                          ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Historique (5 derniers échanges)                                ║
║  + Question utilisateur                                          ║
║           │                                                      ║
║           ▼                                                      ║
║  ┌─────────────────────────────┐                                 ║
║  │  Condensation de question   │ ← LLM reformule si suivi        ║
║  │  (standalone question)      │                                 ║
║  └─────────────────────────────┘                                 ║
║           │                                                      ║
║           ▼                                                      ║
║  ┌─────────────────────────────┐                                 ║
║  │  Multi-Query Retriever      │ ← LLM génère 3 reformulations   ║
║  │  → 3 variantes de question  │                                 ║
║  └─────────────────────────────┘                                 ║
║           │                                                      ║
║     ┌─────┴──────┐                                               ║
║     ▼            ▼                                               ║
║  Chroma        BM25                                              ║
║  (dense)       (sparse)                                          ║
║  k=10          k=10                                              ║
║     └─────┬──────┘                                               ║
║           ▼                                                      ║
║  EnsembleRetriever (RRF)                                         ║
║  Dense 60% + Sparse 40%                                          ║
║  = 20 chunks candidats (dédupliqués)                             ║
║           │                                                      ║
║           ▼                                                      ║
║  FlashRank CrossEncoder                                          ║
║  → Re-classe par pertinence réelle                               ║
║  = Top-4 chunks retenus                                          ║
║           │                                                      ║
║           ▼                                                      ║
║  ContextualCompressionRetriever                                  ║
║  → Filtre les phrases hors-sujet dans chaque chunk               ║
║  = 4 extraits compressés et pertinents                           ║
║           │                                                      ║
║           ▼                                                      ║
║  ┌─────────────────────────────────────────────────┐             ║
║  │  PROMPT FINAL                                   │             ║
║  │                                                 │             ║
║  │  System: "Tu es un assistant documentaire.      │             ║
║  │   Réponds UNIQUEMENT depuis le contexte.        │             ║
║  │   Réponds dans la langue de la question."       │             ║
║  │                                                 │             ║
║  │  Context: [extrait 1] [extrait 2] [extrait 3]   │             ║
║  │           [extrait 4]                           │             ║
║  │                                                 │             ║
║  │  Question: "..."                                │             ║
║  └─────────────────────────────────────────────────┘             ║
║           │                                                      ║
║           ▼                                                      ║
║  Ollama API REST (localhost:11434)                                ║
║  Qwen 2.5 7B — temperature=0.3                                   ║
║           │                                                      ║
║           ▼                                                      ║
║  Réponse générée                                                 ║
║  + sauvegarde JSON dans conversations/                           ║
║  + affichage streaming dans Streamlit                            ║
║  + affichage des sources dans expander                           ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 7. Limites actuelles et pistes d'amélioration

### Limites

| Limite | Impact | Explication |
|--------|--------|-------------|
| PDFs scannés non supportés | Moyen | Images sans couche texte = 0 contenu extrait |
| Streaming simulé | Faible | Effet visuel uniquement, pas de vrai token streaming |
| Index non incrémental | Moyen | Réindexer tout à chaque ajout de document |
| Pas de déduplication des chunks | Faible | Mêmes passages peuvent apparaître plusieurs fois |
| Temperature fixe | Faible | Pas de contrôle depuis l'interface |

### Pistes d'amélioration

**1. Parent Document Retriever**
Indexer de petits chunks (128 car.) pour la précision,
mais envoyer au LLM le chunk parent large (1024 car.) pour le contexte.
Meilleur compromis précision / contexte.

**2. Self-RAG**
Le LLM s'auto-évalue : décide si un retrieval est nécessaire,
critique la qualité des chunks récupérés, valide sa propre réponse.
Nécessite un modèle capable de suivre des instructions structurées.

**3. GraphRAG**
Extraire des entités (personnes, concepts, organisations) et leurs
relations pour construire un graphe de connaissances. Permet des
requêtes relationnelles impossibles avec la recherche vectorielle.

**4. Indexation incrémentale**
Calculer un hash de chaque fichier et n'indexer que ceux qui ont
changé depuis la dernière indexation. Évite de tout réindexer.

**5. Vrai streaming**
Migrer vers LCEL (LangChain Expression Language) avec `.astream()`
pour afficher les tokens au fur et à mesure de leur génération.

**6. OCR pour PDFs scannés**
Intégrer `ocrmypdf` ou `pytesseract` pour extraire le texte
des PDFs composés uniquement d'images scannées.
