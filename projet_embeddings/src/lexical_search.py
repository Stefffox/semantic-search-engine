"""
lexical_search.py
-----------------
Moteur de recherche LEXICALE basé sur TF-IDF, implémenté de zéro.

RÔLE DANS LE PROJET
--------------------
Ce module fournit une baseline de comparaison pour l'évaluation.
On compare les résultats de la recherche sémantique (cosinus sur embeddings)
avec ceux de la recherche lexicale (TF-IDF) afin de mesurer l'apport
des embeddings sur le corpus choisi.

MATHÉMATIQUES DU TF-IDF
------------------------

1. TF — Term Frequency (fréquence du terme dans un document)
   ----------------------------------------------------------
   Pour un terme t et un document d :

       TF(t, d) = count(t, d) / |d|

   où count(t, d) est le nombre d'occurrences de t dans d,
   et |d| est le nombre total de mots dans d.

   Normaliser par |d| évite de favoriser les documents longs :
   un document de 1000 mots avec 10 occurrences de "chat" (TF=0.01)
   n'est pas plus pertinent qu'un document de 100 mots avec 1 occurrence
   (TF=0.01 aussi).

2. IDF — Inverse Document Frequency (rareté dans le corpus)
   ---------------------------------------------------------
   Pour un terme t dans un corpus de N documents :

       IDF(t) = log( N / (1 + df(t)) )

   où df(t) = nombre de documents contenant t.

   Interprétation :
   - Si t est dans TOUS les documents (df=N) → IDF ≈ log(1) = 0
     → terme trop commun ("le", "est", "de") → pas discriminant → score 0
   - Si t est RARE (df=1) → IDF = log(N/2) → grande valeur → très discriminant
   - Le +1 est une régularisation (smoothing) qui évite la division par 0
     quand un terme n'apparaît que dans la requête (pas dans le corpus)

3. Score TF-IDF d'un document face à une requête
   -----------------------------------------------
   La requête Q = {t₁, t₂, ..., tₘ} est un ensemble de termes.

       score(Q, d) = Σ_{t ∈ Q} TF(t, d) × IDF(t)

   On additionne les scores TF-IDF de chaque terme de la requête
   présent dans le document. Les termes absents contribuent 0.

   Cette formule est une approximation du produit scalaire dans l'espace
   des représentations TF-IDF vectorielles (les vecteurs sont creux —
   sparse — car la plupart des termes du vocabulaire sont absents d'un
   document donné).

LIMITES DU TF-IDF
-----------------
- Synonymie : "voiture" et "automobile" sont traités comme deux mots
  totalement différents, même s'ils signifient la même chose.
- Polysémie : "banque" (institution financière) et "banque" (de poissons)
  ont le même score TF-IDF dans n'importe quel contexte.
- Ordre des mots ignoré : "le chat mange la souris" et "la souris mange
  le chat" produisent exactement le même score.

Ces limites sont précisément ce que la recherche par embeddings résout,
ce qui justifie la comparaison dans evaluation.py.

IMPLÉMENTATION SANS DÉPENDANCES
---------------------------------
Tout est implémenté avec la bibliothèque standard Python (math, collections).
Pas de sklearn, pas de nltk — conforme à l'esprit du projet.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict


def _tokenize(text: str) -> list[str]:
    """
    Tokenisation minimaliste : minuscules + séparation sur les non-alphanumériques.

    Ex : "Le chat, vif, mange!" → ["le", "chat", "vif", "mange"]

    On ne fait pas de stemming (réduction à la racine) pour garder
    l'implémentation simple. En production on utiliserait nltk ou spaCy.
    """
    text = text.lower()
    # Garde lettres, chiffres, apostrophes (contraction française)
    tokens = re.findall(r"[a-zàâäéèêëîïôöùûüÿçœæ0-9]+", text)
    return tokens


class TFIDFIndex:
    """
    Index TF-IDF en mémoire pour la recherche lexicale.

    Construction en deux passes :
    1. Indexation : calcul des TF par document, puis des IDF globaux
    2. Recherche : calcul du score TF-IDF pour chaque document face à Q

    Attributs (après build)
    -----------------------
    _documents : list[dict]
        Liste des chunks (même format que VectorIndex.metadata).
    _tf : list[Counter]
        _tf[i] = Counter {terme: tf(terme, doc_i)}
    _idf : dict[str, float]
        {terme: idf(terme)} calculé sur tout le corpus
    _vocab : set[str]
        Tous les termes vus pendant l'indexation
    """

    def __init__(self):
        self._documents: list[dict] = []
        self._tf: list[dict[str, float]] = []
        self._idf: dict[str, float] = {}
        self._df: Counter = Counter()  # df[terme] = nb docs contenant terme
        self._vocab: set[str] = set()

    def build(self, chunks: list[dict]) -> "TFIDFIndex":
        """
        Indexe une liste de chunks et calcule les TF et IDF.

        Algorithme
        ----------
        Passe 1 : pour chaque chunk i
            - Tokeniser le texte
            - Calculer TF(t, i) = count(t) / len(tokens)
            - Mettre à jour df(t) += 1 si t ∈ chunk_i

        Passe 2 : pour chaque terme t dans le vocabulaire
            - IDF(t) = log(N / (1 + df(t)))

        Paramètres
        ----------
        chunks : list[dict]
            Chunks produits par chunker (même format que pour VectorIndex).

        Retourne
        --------
        TFIDFIndex
            self (pour chaînage : idx = TFIDFIndex().build(chunks))
        """
        self._documents = chunks
        N = len(chunks)

        # Passe 1 : TF et DF
        for chunk in chunks:
            tokens = _tokenize(chunk["text"])
            if not tokens:
                self._tf.append({})
                continue

            count = Counter(tokens)
            n_tokens = len(tokens)

            # TF normalisé par la longueur du document
            tf = {terme: cnt / n_tokens for terme, cnt in count.items()}
            self._tf.append(tf)

            # DF : chaque terme vu dans ce document → +1
            for terme in count:
                self._df[terme] += 1
                self._vocab.add(terme)

        # Passe 2 : IDF
        for terme in self._vocab:
            self._idf[terme] = math.log(N / (1 + self._df[terme]))

        return self

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Retourne les top_k chunks les plus pertinents pour la requête.

        Algorithme
        ----------
        1. Tokeniser la requête → termes Q = {t₁, ..., tₘ}
        2. Pour chaque chunk i :
               score_i = Σ_{t ∈ Q ∩ vocab} TF(t, i) × IDF(t)
           (les termes de Q absents du vocabulaire contribuent 0)
        3. Trier par score décroissant, retourner top_k

        Paramètres
        ----------
        query : str
            La requête en langue naturelle.
        top_k : int
            Nombre de résultats à retourner.

        Retourne
        --------
        list[dict]
            Dicts triés par score, avec clé "score" ajoutée.
            Même format que VectorIndex.search() pour faciliter
            la comparaison dans evaluation.py.
        """
        query_terms = set(_tokenize(query))

        scored: list[tuple[float, int]] = []
        for i, tf_i in enumerate(self._tf):
            score = sum(
                tf_i.get(terme, 0.0) * self._idf.get(terme, 0.0)
                for terme in query_terms
            )
            scored.append((score, i))

        # Tri décroissant, puis alphabétique par index pour la stabilité
        scored.sort(key=lambda t: (-t[0], t[1]))

        results: list[dict] = []
        for score, idx in scored[:top_k]:
            result = dict(self._documents[idx])
            result["score"] = round(score, 6)
            results.append(result)

        return results
