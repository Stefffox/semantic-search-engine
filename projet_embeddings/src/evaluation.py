"""
evaluation.py
-------------
Métriques d'évaluation pour les moteurs de recherche (P@k et MRR)
et tableau comparatif Sémantique vs Lexical.

CONTEXTE
--------
Pour évaluer un moteur de recherche, on dispose d'un "ground truth" :
un ensemble de paires (requête, {chunks_pertinents}).

On compare les résultats retournés par le moteur avec ce ground truth
via deux métriques standard en Information Retrieval.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. PRECISION@k — P@k
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Définition :
    P@k(q) = |{ résultats pertinents parmi le top k }| / k

La précision mesure la PURETÉ du top k : quelle fraction des résultats
retournés est réellement pertinente ?

Exemple :
    top 5 retournés : [A, B, C, D, E]
    pertinents      : {A, C, F}          (F n'est pas dans le top 5)
    P@5 = 2/5 = 0.40  (A et C sont pertinents parmi les 5 retournés)

Interprétation :
    P@1 = 1.0 → le meilleur résultat est toujours correct
    P@5 = 0.4 → en moyenne 2 résultats sur 5 sont pertinents

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. MRR — Mean Reciprocal Rank
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Définition :
    RR(q)  = 1 / rang_du_premier_résultat_pertinent(q)
             0  si aucun résultat pertinent dans le top k

    MRR    = (1 / |Q|) × Σ_{q ∈ Q} RR(q)

où Q est l'ensemble des requêtes de test et "rang" est 1-indexé.

Exemple avec 3 requêtes :
    q1 : premier pertinent en position 1 → RR = 1/1 = 1.000
    q2 : premier pertinent en position 3 → RR = 1/3 = 0.333
    q3 : aucun pertinent dans le top 5   → RR = 0.000
    MRR = (1.000 + 0.333 + 0.000) / 3 = 0.444

Interprétation :
    MRR = 1.0  → le bon résultat est TOUJOURS en première position
    MRR = 0.5  → en moyenne, le bon résultat est en 2ᵉ position
    MRR < 0.2  → le moteur est peu fiable (bon résultat souvent en 5+)

Différence P@k vs MRR :
    P@k compte TOUS les pertinents dans le top k (mesure la couverture)
    MRR s'intéresse au PREMIER pertinent seulement (mesure la précision
    du résultat de tête — plus important pour l'utilisateur final)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. FORMAT DU GROUND TRUTH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Le ground truth est une liste de dicts :
    [
        {
            "query": "Comment fonctionne la photosynthèse ?",
            "relevant": {"doc1_chunk3", "doc1_chunk4"}  # set de chunk IDs
        },
        ...
    ]

Un "chunk ID" est la clé unique d'un chunk. On utilise le format :
    "{source}::{chunk_index}"
    Ex. : "biologie.txt::5" pour le 6ème chunk du fichier biologie.txt

Cette convention est cohérente avec les métadonnées produites par chunker.py.
"""

from __future__ import annotations

from typing import Callable


# ---------------------------------------------------------------------------
# Identifiant unique d'un chunk
# ---------------------------------------------------------------------------

def chunk_id(chunk: dict) -> str:
    """
    Retourne l'identifiant unique d'un chunk : "{source}::{chunk_index}".

    Paramètre
    ---------
    chunk : dict
        Dict de chunk (produit par chunker ou retourné par search).

    Retourne
    --------
    str
        Identifiant unique, ex. "biologie.txt::5".
    """
    source = chunk.get("source", "")
    idx = chunk.get("chunk_index", 0)
    return f"{source}::{idx}"


# ---------------------------------------------------------------------------
# Precision@k
# ---------------------------------------------------------------------------

def precision_at_k(results: list[dict], relevant_ids: set[str], k: int) -> float:
    """
    Calcule la Precision@k pour une requête.

        P@k = |{ résultats pertinents parmi le top k }| / k

    Paramètres
    ----------
    results : list[dict]
        Résultats retournés par le moteur, triés par score décroissant.
        Chaque dict doit contenir "source" et "chunk_index".
    relevant_ids : set[str]
        Ensemble des chunk IDs pertinents (ground truth).
        Format : {"{source}::{chunk_index}", ...}
    k : int
        Seuil de coupure (on ne regarde que les k premiers résultats).

    Retourne
    --------
    float
        Score entre 0.0 et 1.0.
        0.0 si aucun résultat pertinent dans le top k.
        1.0 si tous les k premiers résultats sont pertinents.
    """
    top_k = results[:k]
    if not top_k:
        return 0.0

    hits = sum(1 for r in top_k if chunk_id(r) in relevant_ids)
    return hits / k


def mean_precision_at_k(
    queries_results: list[tuple[list[dict], set[str]]],
    k: int,
) -> float:
    """
    Calcule la Precision@k MOYENNE sur plusieurs requêtes (MAP@k).

        MAP@k = (1/|Q|) × Σ_q P@k(q)

    Paramètres
    ----------
    queries_results : list[tuple[list[dict], set[str]]]
        Liste de (résultats_moteur, relevant_ids) pour chaque requête.
    k : int
        Seuil de coupure.

    Retourne
    --------
    float
        Score moyen entre 0.0 et 1.0.
    """
    if not queries_results:
        return 0.0
    scores = [precision_at_k(res, rel, k) for res, rel in queries_results]
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# MRR — Mean Reciprocal Rank
# ---------------------------------------------------------------------------

def reciprocal_rank(results: list[dict], relevant_ids: set[str]) -> float:
    """
    Calcule le Reciprocal Rank pour une requête.

        RR = 1 / rang_du_premier_pertinent   (0 si aucun pertinent)

    Le rang est 1-indexé (le premier résultat a le rang 1).

    Paramètres
    ----------
    results : list[dict]
        Résultats retournés par le moteur.
    relevant_ids : set[str]
        Ensemble des chunk IDs pertinents.

    Retourne
    --------
    float
        1.0 si le premier résultat est pertinent.
        1/k si le premier pertinent est en position k.
        0.0 si aucun résultat pertinent.
    """
    for rank, result in enumerate(results, start=1):
        if chunk_id(result) in relevant_ids:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    queries_results: list[tuple[list[dict], set[str]]],
) -> float:
    """
    Calcule le MRR sur plusieurs requêtes.

        MRR = (1/|Q|) × Σ_q RR(q)

    Paramètres
    ----------
    queries_results : list[tuple[list[dict], set[str]]]
        Liste de (résultats_moteur, relevant_ids) pour chaque requête.

    Retourne
    --------
    float
        Score entre 0.0 et 1.0.
    """
    if not queries_results:
        return 0.0
    scores = [reciprocal_rank(res, rel) for res, rel in queries_results]
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Évaluation complète : tableau comparatif
# ---------------------------------------------------------------------------

def evaluate_engine(
    engine_fn: Callable[[str, int], list[dict]],
    ground_truth: list[dict],
    top_k: int = 5,
) -> dict:
    """
    Évalue un moteur de recherche sur un ground truth.

    Paramètres
    ----------
    engine_fn : Callable[[str, int], list[dict]]
        Fonction de recherche : engine_fn(query, top_k) → list[dict].
        Compatible avec SemanticSearch.search et TFIDFIndex.search.
    ground_truth : list[dict]
        Liste de {"query": str, "relevant": set[str]}.
    top_k : int
        Seuil de coupure pour P@k.

    Retourne
    --------
    dict avec :
        "map_at_k"  : float — Precision@k moyenne
        "mrr"       : float — Mean Reciprocal Rank
        "details"   : list[dict] — scores par requête
    """
    queries_results: list[tuple[list[dict], set[str]]] = []
    details: list[dict] = []

    for item in ground_truth:
        query = item["query"]
        relevant = set(item["relevant"])

        results = engine_fn(query, top_k)
        queries_results.append((results, relevant))

        pk = precision_at_k(results, relevant, top_k)
        rr = reciprocal_rank(results, relevant)
        details.append({"query": query, f"P@{top_k}": pk, "RR": rr})

    return {
        f"map_at_{top_k}": mean_precision_at_k(queries_results, top_k),
        "mrr": mean_reciprocal_rank(queries_results),
        "details": details,
    }


def print_comparison_table(
    semantic_metrics: dict,
    lexical_metrics: dict,
    top_k: int = 5,
) -> None:
    """
    Affiche un tableau comparatif Sémantique vs Lexical dans la console.

    Paramètres
    ----------
    semantic_metrics : dict
        Résultat de evaluate_engine() pour le moteur sémantique.
    lexical_metrics : dict
        Résultat de evaluate_engine() pour le moteur lexical (TF-IDF).
    top_k : int
        Seuil de coupure utilisé pour les deux évaluations.
    """
    map_key = f"map_at_{top_k}"

    sem_map = semantic_metrics.get(map_key, 0.0)
    sem_mrr = semantic_metrics.get("mrr", 0.0)
    lex_map = lexical_metrics.get(map_key, 0.0)
    lex_mrr = lexical_metrics.get("mrr", 0.0)

    # Symboles de comparaison
    def cmp(a: float, b: float) -> str:
        if a > b + 0.001:
            return "✓ MIEUX"
        elif a < b - 0.001:
            return "✗ moins bien"
        return "= égal"

    print("\n" + "=" * 60)
    print(" TABLEAU COMPARATIF : Sémantique vs Lexical (TF-IDF)")
    print("=" * 60)
    print(f"{'Métrique':<20} {'Sémantique':>12} {'Lexical':>12} {'Verdict':>14}")
    print("-" * 60)
    print(f"{'MAP@' + str(top_k):<20} {sem_map:>12.4f} {lex_map:>12.4f} {cmp(sem_map, lex_map):>14}")
    print(f"{'MRR':<20} {sem_mrr:>12.4f} {lex_mrr:>12.4f} {cmp(sem_mrr, lex_mrr):>14}")
    print("=" * 60)
    print()

    # Détail par requête
    sem_details = semantic_metrics.get("details", [])
    lex_details = lexical_metrics.get("details", [])

    if sem_details:
        print("Détail par requête :")
        print(f"{'Requête':<35} {'P@k Sém':>8} {'P@k Lex':>8} {'RR Sém':>8} {'RR Lex':>8}")
        print("-" * 70)
        for sem_d, lex_d in zip(sem_details, lex_details):
            q = sem_d["query"][:33] + ".." if len(sem_d["query"]) > 35 else sem_d["query"]
            print(
                f"{q:<35} "
                f"{sem_d.get(f'P@{top_k}', 0.0):>8.3f} "
                f"{lex_d.get(f'P@{top_k}', 0.0):>8.3f} "
                f"{sem_d.get('RR', 0.0):>8.3f} "
                f"{lex_d.get('RR', 0.0):>8.3f}"
            )
        print()
