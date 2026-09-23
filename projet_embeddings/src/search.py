"""
search.py
---------
Moteur de recherche sémantique : encode une requête et interroge l'index.

RÔLE MATHÉMATIQUE
-----------------
La recherche sémantique se décompose en deux étapes :

Étape 1 — Projection de la requête dans l'espace des embeddings :
    q_texte ∈ str   →   q ∈ ℝⁿ   via le même provider que l'indexation

Étape 2 — Sélection des k plus proches voisins par similarité cosinus :
    résultats = argmax_k  S(q, M[i])   pour i = 0..K-1

    où S(q, M[i]) = (q · M[i]) / (||q|| · ||M[i]||)

Il est CRUCIAL que la requête soit encodée avec le MÊME modèle que les
chunks indexés. Sinon les vecteurs appartiennent à des espaces différents
(même dimension n, mais bases différentes) et la similarité cosinus ne
mesure plus rien de pertinent.

Analogie géométrique : on ne peut pas mesurer l'angle entre deux vecteurs
si l'un est exprimé en coordonnées cartésiennes et l'autre en coordonnées
polaires. Même dimension ≠ même espace vectoriel.

RÔLE FONCTIONNEL
----------------
La classe SemanticSearch encapsule :
  - un VectorIndex (la base de données vectorielle)
  - un EmbeddingProvider (pour encoder la requête)

Elle expose une méthode search(query, top_k) qui retourne les top_k chunks
les plus sémantiquement proches de la requête.
"""

from __future__ import annotations

from .embedding.base import EmbeddingProvider
from .indexer import VectorIndex


class SemanticSearch:
    """
    Moteur de recherche sémantique combinant un index et un provider.

    Exemple d'utilisation
    ---------------------
        from src.embedding.ollama_provider import OllamaEmbeddingProvider
        from src.indexer import VectorIndex
        from src.search import SemanticSearch

        provider = OllamaEmbeddingProvider()
        index = VectorIndex.load()
        engine = SemanticSearch(index, provider)

        results = engine.search("Comment fonctionne la photosynthèse ?", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] ({r['source']}) {r['text'][:120]}...")
    """

    def __init__(self, index: VectorIndex, provider: EmbeddingProvider):
        """
        Paramètres
        ----------
        index : VectorIndex
            L'index vectoriel pré-construit (chargé depuis le disque
            ou construit avec VectorIndex.build()).
        provider : EmbeddingProvider
            Le fournisseur d'embeddings. DOIT être le même modèle que
            celui utilisé lors de la construction de l'index.
        """
        self._index = index
        self._provider = provider

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Recherche les chunks les plus proches de la requête.

        Étapes internes
        ---------------
        1. Encode la requête : q = provider.embed(query)   → q ∈ ℝⁿ
        2. Délègue à index.search(q, top_k)               → liste de résultats

        Paramètres
        ----------
        query : str
            La requête en langue naturelle. Peut être une question,
            une phrase, ou quelques mots-clés.
        top_k : int
            Nombre de résultats à retourner (les top_k plus similaires).

        Retourne
        --------
        list[dict]
            Liste de dicts triés par score décroissant.
            Chaque dict contient :
              - "text"        : str   — le texte du chunk
              - "score"       : float — similarité cosinus (dans [-1, 1])
              - "source"      : str   — document d'origine
              - "chunk_index" : int   — position du chunk dans le document
              - "start_word"  : int   — position du premier mot
              - "end_word"    : int   — position du dernier mot (exclu)
        """
        query_vector = self._provider.embed(query)
        return self._index.search(query_vector, top_k=top_k)

    def search_and_display(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Recherche et affiche les résultats formatés dans la console.

        Utile pour une démonstration rapide ou main.py.

        Paramètres
        ----------
        query : str
            La requête.
        top_k : int
            Nombre de résultats.

        Retourne
        --------
        list[dict]
            Mêmes résultats que search(), affichés en plus dans la console.
        """
        results = self.search(query, top_k=top_k)

        print(f'\nRequête : "{query}"')
        print(f"Top {top_k} résultats :\n")

        if not results:
            print("  Aucun résultat (index vide ?)")
            return results

        for rank, result in enumerate(results, start=1):
            source = result.get("source", "?")
            score = result.get("score", 0.0)
            text_preview = result.get("text", "")[:200]
            ellipsis = "..." if len(result.get("text", "")) > 200 else ""

            print(f"  #{rank}  score={score:.4f}  [{source}]")
            print(f"       {text_preview}{ellipsis}")
            print()

        return results
