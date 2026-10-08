# =============================================================================
# conversations_manager.py — Gestionnaire de sessions de conversation
# =============================================================================
# Sauvegarde chaque conversation dans un fichier JSON dans le dossier
# "conversations/". Chaque session a un ID unique, un titre auto-généré
# (basé sur la première question) et un horodatage.
#
# Structure d'un fichier JSON de session :
# {
#   "id":         "2026-09-22_14-35-12",
#   "titre":      "What is PeerRead?",
#   "cree_le":    "2026-09-22T14:35:12",
#   "modifie_le": "2026-09-22T14:42:05",
#   "messages":   [{"role": "user"|"assistant", "content": "..."}],
#   "historique": [["question", "reponse"], ...]
# }
# =============================================================================

import json
from datetime import datetime
from pathlib import Path

CONVERSATIONS_DIR = Path("./conversations")


def _ensure_dir():
    """Crée le dossier conversations/ s'il n'existe pas."""
    CONVERSATIONS_DIR.mkdir(exist_ok=True)


def nouvelle_session_id() -> str:
    """Génère un ID unique basé sur l'horodatage."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def sauvegarder_session(session_id: str, titre: str, messages: list, historique: list):
    """
    Sauvegarde ou met à jour une session sur disque.
    Appelée après chaque échange pour garder l'historique à jour.
    """
    _ensure_dir()
    chemin = CONVERSATIONS_DIR / f"{session_id}.json"

    # Si la session existe déjà, on récupère la date de création
    cree_le = datetime.now().isoformat()
    if chemin.exists():
        try:
            with open(chemin, encoding="utf-8") as f:
                existant = json.load(f)
                cree_le = existant.get("cree_le", cree_le)
        except Exception:
            pass

    data = {
        "id":         session_id,
        "titre":      titre,
        "cree_le":    cree_le,
        "modifie_le": datetime.now().isoformat(),
        "messages":   messages,
        "historique": historique,
    }

    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def charger_session(session_id: str) -> dict | None:
    """
    Charge une session depuis le disque.
    Retourne None si le fichier n'existe pas.
    """
    chemin = CONVERSATIONS_DIR / f"{session_id}.json"
    if not chemin.exists():
        return None
    try:
        with open(chemin, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def lister_sessions() -> list[dict]:
    """
    Retourne la liste de toutes les sessions triées par date de modification
    (la plus récente en premier). Chaque élément contient id, titre, modifie_le.
    """
    _ensure_dir()
    sessions = []
    for fichier in CONVERSATIONS_DIR.glob("*.json"):
        try:
            with open(fichier, encoding="utf-8") as f:
                data = json.load(f)
                sessions.append({
                    "id":         data.get("id", fichier.stem),
                    "titre":      data.get("titre", "Sans titre"),
                    "modifie_le": data.get("modifie_le", ""),
                    "nb_msgs":    len(data.get("messages", [])),
                })
        except Exception:
            continue

    # Tri par date de modification décroissante
    sessions.sort(key=lambda s: s["modifie_le"], reverse=True)
    return sessions


def supprimer_session(session_id: str) -> bool:
    """
    Supprime le fichier JSON d'une session.
    Retourne True si supprimé, False si le fichier n'existait pas.
    """
    chemin = CONVERSATIONS_DIR / f"{session_id}.json"
    if chemin.exists():
        chemin.unlink()
        return True
    return False


def titre_depuis_question(question: str, max_len: int = 50) -> str:
    """
    Génère un titre court depuis la première question de la session.
    Tronque à max_len caractères avec '…' si nécessaire.
    """
    titre = question.strip().rstrip("?").strip()
    if len(titre) > max_len:
        titre = titre[:max_len - 1] + "…"
    return titre
