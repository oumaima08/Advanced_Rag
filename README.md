# 📚 RAG Naïf Local — Assistant Documentaire

Un système RAG (*Retrieval-Augmented Generation*) 100 % open source qui tourne
entièrement en local sur ta machine Windows. Dépose des PDF ou des fichiers texte,
et interroge-les en langage naturel via une interface web simple.

Le LLM (Qwen 2.5 7B) est appelé via l'**API REST locale d'Ollama** sur
`http://localhost:11434` — aucun accès cloud, tout reste sur ta machine.

---

## Fonctionnalités

- Indexation de fichiers **PDF** et **TXT** depuis un dossier local
- Découpage intelligent du texte en chunks avec chevauchement
- Embeddings calculés localement avec **sentence-transformers** (all-MiniLM-L6-v2)
- Stockage vectoriel persistant avec **ChromaDB** (dossier `chroma_db/`)
- Réponses générées par **Qwen 2.5 7B** via l'API REST Ollama (100 % local)
- Interface web en français avec **Streamlit**
- Vérification automatique de la connexion Ollama au démarrage
- Affichage des sources utilisées pour chaque réponse

---

## Architecture du projet

```
rag_cours/
├── .venv/          ← Environnement virtuel Python (créé à l'étape 2)
├── docs/           ← Tes fichiers PDF et TXT à indexer
├── chroma_db/      ← Base vectorielle créée par ingest.py
├── ingest.py       ← Script d'indexation (à lancer manuellement)
├── app.py          ← Interface Streamlit (lecture seule de chroma_db)
├── requirements.txt
└── README.md
```

---

## ÉTAPE 1 — Installer Ollama et télécharger le modèle

### 1.1 Télécharger et installer Ollama

Rends-toi sur **[https://ollama.com/download](https://ollama.com/download)**
et clique sur **"Download for Windows"**.

Lance l'installateur `.exe` téléchargé. Après installation, Ollama
s'ajoute automatiquement à la barre des tâches Windows (icône en bas à droite).

> Ollama démarre automatiquement au lancement de Windows.
> Si l'icône n'apparaît pas, cherche "Ollama" dans le menu Démarrer et lance-le.

### 1.2 Vérifier qu'Ollama tourne

Ouvre PowerShell et tape :

```powershell
curl http://localhost:11434
```

Tu dois voir la réponse : `Ollama is running`

Si tu vois une erreur de connexion, démarre Ollama manuellement :

```powershell
ollama serve
```

Laisse ce terminal ouvert.

### 1.3 Télécharger le modèle Qwen 2.5 7B

> ⚠️ Le téléchargement pèse environ **4,7 Go**. Assure-toi d'avoir
> une connexion stable et suffisamment d'espace disque.

Ouvre un **nouveau** terminal PowerShell et lance :

```powershell
ollama pull qwen2.5:7b
```

Le téléchargement peut prendre 5 à 20 minutes selon ta connexion.
Tu verras une barre de progression. À la fin tu verras :
`success` ou `pulling manifest`.

### 1.4 Vérifier que le modèle est bien installé

```powershell
ollama list
```

Tu dois voir `qwen2.5:7b` dans la liste. Exemple de sortie :

```
NAME            ID              SIZE    MODIFIED
qwen2.5:7b      ...             4.7 GB  ...
```

### 1.5 Tester le modèle en ligne de commande (optionnel)

```powershell
ollama run qwen2.5:7b "Dis bonjour en français"
```

Si le modèle répond, tout est prêt. Tape `/bye` pour quitter.

---

## ÉTAPE 2 — Installer Python et le projet

### 2.1 Vérifier Python 3.11

```powershell
python --version
```

La version doit afficher `Python 3.11.x`.
Si ce n'est pas le cas, télécharge Python 3.11 sur
[https://www.python.org/downloads/](https://www.python.org/downloads/)
(coche **"Add Python to PATH"** pendant l'installation).

### 2.2 Ouvrir un terminal dans le dossier du projet

```powershell
cd C:\Users\Asus\Downloads\RAG_Basic\rag_cours
```

### 2.3 Créer l'environnement virtuel

```powershell
python -m venv .venv
```

### 2.4 Activer l'environnement virtuel

**PowerShell :**
```powershell
.\.venv\Scripts\Activate.ps1
```

**CMD :**
```cmd
.venv\Scripts\activate.bat
```

> Si PowerShell refuse avec une erreur sur les scripts, exécute d'abord :
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Puis réessaie l'activation.

Quand le venv est actif, tu vois `(.venv)` au début de la ligne dans le terminal.

### 2.5 Installer les dépendances

```powershell
pip install -r requirements.txt
```

> La première installation prend 3 à 10 minutes.
> Le modèle d'embeddings sentence-transformers (~90 Mo) sera téléchargé
> automatiquement lors du premier lancement de `ingest.py`.

---

## ÉTAPE 3 — Connexion à ChromaDB et indexation des documents

ChromaDB est une base vectorielle **locale** — elle s'enregistre simplement
dans le dossier `chroma_db/` sur ton disque. Aucune configuration serveur
n'est nécessaire.

### 3.1 Ajouter tes documents

Place tes fichiers **PDF** et/ou **TXT** dans le dossier `docs/` :

```
rag_cours/
└── docs/
    ├── mon_cours.pdf
    ├── notes.txt
    └── ...
```

### 3.2 Lancer l'indexation

```powershell
python ingest.py
```

Ce script va :
1. Lire tous les PDF et TXT dans `docs/`
2. Découper le texte en chunks de 800 caractères
3. Calculer les embeddings avec sentence-transformers
4. Sauvegarder tout dans `chroma_db/`

**Sortie attendue :**
```
============================================================
  RAG — Script d'indexation des documents
============================================================
2025-... [INFO] 📂  2 fichier(s) trouvé(s) dans 'docs'.
2025-... [INFO]   ✅  Chargé : mon_cours.pdf  (12 page(s))
2025-... [INFO] ✂️   Découpage terminé : 47 chunk(s) créé(s).
2025-... [INFO] 🔢  Chargement du modèle d'embeddings...
2025-... [INFO] 💾  Indexation et sauvegarde dans './chroma_db'...
2025-... [INFO] ✅  Base vectorielle créée avec succès.
============================================================
  ✅  Indexation terminée !
  📊  47 chunk(s) indexé(s) dans './chroma_db'.
  ▶️   Tu peux maintenant lancer : streamlit run app.py
============================================================
```

### 3.3 Vérifier que chroma_db est bien rempli

```powershell
Get-ChildItem .\chroma_db\ -Recurse | Measure-Object | Select-Object -ExpandProperty Count
```

Ce nombre doit être supérieur à 0 (en général 5 à 10 fichiers créés par Chroma).

> **Important :** Relance `python ingest.py` à chaque fois que tu ajoutes
> ou modifies des fichiers dans `docs/`.

---

## ÉTAPE 4 — Lancer l'interface et poser des questions

### 4.1 S'assurer qu'Ollama tourne toujours

```powershell
curl http://localhost:11434
```

Réponse attendue : `Ollama is running`. Sinon, relance `ollama serve`.

### 4.2 Démarrer l'interface Streamlit

```powershell
streamlit run app.py
```

Streamlit ouvre automatiquement ton navigateur sur :
**`http://localhost:8501`**

### 4.3 Utiliser l'interface

1. Au démarrage, l'interface vérifie automatiquement :
   - Que `chroma_db/` existe et contient des données
   - Qu'Ollama répond sur `http://localhost:11434`
   - Que le modèle `qwen2.5:7b` est téléchargé
2. Un message vert ✅ confirme qu'Ollama est connecté
3. Tape ta question dans le champ en bas et appuie sur Entrée
4. La réponse apparaît avec les sources utilisées (clique sur "📎 Sources")

---

## Comment fonctionne le RAG

```
┌─────────────────────────────────────────────────────────────┐
│                  INDEXATION  (ingest.py)                    │
│                                                             │
│  docs/          Chunking      Embeddings     chroma_db/    │
│  (PDF/TXT)  →  (800 car.)  →  (MiniLM-L6) →  (Chroma)    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                 INTERROGATION  (app.py)                     │
│                                                             │
│  Question   →  Embedding    →  Recherche   →  4 chunks    │
│                question        vectorielle    pertinents   │
│                                                             │
│  4 chunks + Question  →  Prompt  →  Ollama API REST        │
│                                    localhost:11434          │
│                       →  Réponse en français + Sources     │
└─────────────────────────────────────────────────────────────┘
```

**Pourquoi "naïf" ?** Ce RAG utilise la stratégie la plus simple : les chunks
récupérés sont insérés directement dans le prompt (*chain_type="stuff"*).
Pas de re-ranking, pas de mémoire conversationnelle. Idéal pour apprendre.

---

## Idées d'améliorations

1. **Mémoire conversationnelle** — `ConversationalRetrievalChain` pour tenir
   compte des échanges précédents dans la même session.
2. **Re-ranking** — Ajouter un `CrossEncoder` pour re-classer les chunks par
   pertinence avant de les envoyer au LLM.
3. **Support d'autres formats** — Loaders LangChain pour `.docx`, Markdown, CSV.
4. **Mise à jour incrémentale** — N'indexer que les nouveaux documents plutôt
   que tout réindexer à chaque lancement d'`ingest.py`.
5. **Streaming de la réponse** — Afficher la réponse mot par mot en temps réel
   via `st.write_stream` pour une meilleure expérience utilisateur.

---

## Dépannage

### ❌ `curl http://localhost:11434` → erreur de connexion

Ollama n'est pas lancé.

**Solution :** Lance Ollama depuis le menu Démarrer (cherche "Ollama")
ou dans un terminal :
```powershell
ollama serve
```

---

### ❌ "Le modèle qwen2.5:7b n'est pas téléchargé"

L'interface affiche les modèles disponibles et demande de faire le pull.

**Solution :**
```powershell
ollama pull qwen2.5:7b
```

---

### ❌ "Base de données introuvable ou vide"

Le dossier `chroma_db/` n'existe pas encore.

**Solution :** Place des fichiers dans `docs/` et lance :
```powershell
python ingest.py
```

---

### ❌ Le script ingest.py s'arrête "dossier docs/ vide"

Aucun fichier PDF ou TXT dans `docs/`.

**Solution :** Ajoute au moins un fichier puis relance `python ingest.py`.

---

### ⚠️ "Impossible de lire 'fichier.pdf'" — fichier ignoré

Le PDF est corrompu, protégé par mot de passe, ou uniquement composé
d'images scannées (sans couche texte extractible).

**Solutions :**
- Vérifie que le PDF s'ouvre et n'est pas protégé
- Les PDFs scannés nécessitent un OCR préalable (outil : `ocrmypdf`)

---

### ❌ Erreur d'activation du venv sous PowerShell

```
Impossible de charger le fichier Activate.ps1, l'exécution de scripts est désactivée
```

**Solution :**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

### ❌ Les réponses ne parlent pas de mes documents

Le modèle répond sur ses connaissances générales.

**Solutions :**
- Vérifie que `ingest.py` s'est terminé sans erreur
- Assure-toi que tes fichiers contiennent du texte (pas des scans)
- Reformule ta question avec des termes présents dans tes documents
