"""
chunker.py
----------
Découpage d'un texte long en segments (chunks) pour l'indexation.

PROBLÈME MATHÉMATIQUE
---------------------
Un modèle d'embedding comme nomic-embed-text a une fenêtre contextuelle
maximale T (≈ 8 192 tokens pour nomic-embed-text). Si un document contient
N > T tokens, le modèle tronque silencieusement — les tokens au-delà de T
sont ignorés, et le vecteur produit ne représente qu'une fraction du texte.

Même sans troncature, un document long pose un autre problème :

    Le vecteur d'un document entier est approximativement le BARYCENTRE
    des vecteurs de ses phrases. Pour un document sur "physique quantique
    et cuisine française", le vecteur résultant est au milieu des deux
    domaines — donc mal positionné pour répondre à une requête précise
    sur l'un ou l'autre sujet.

SOLUTION — CHUNKING AVEC CHEVAUCHEMENT
----------------------------------------
On découpe le document en segments de taille fixe w (en mots), avec un
chevauchement de s mots entre segments consécutifs.

    Mots du document : [w₀, w₁, w₂, ..., wₙ]

    Chunk 0 : mots [ 0,         w - 1     ]
    Chunk 1 : mots [ w-s,       2w-s - 1  ]
    Chunk 2 : mots [ 2(w-s),    3(w-s)+s-1]
    ...
    Chunk k : mots [ k*(w-s),   k*(w-s)+w-1]

    Pas entre chunks : stride = w - s

Le chevauchement garantit qu'une phrase à la frontière entre deux chunks
est intégralement contenue dans au moins l'un d'eux. Sans chevauchement,
une phrase coupée exactement à la frontière produirait deux demi-phrases
aux vecteurs dégradés.

ESPACE DES CHUNKS
-----------------
Soit un document de N mots :

    Nombre de chunks ≈ ceil((N - overlap) / (chunk_size - overlap))
                     = ceil((N - s) / (w - s))

Exemple : document de 1 000 mots, chunk_size=300, overlap=50 :
    stride = 300 - 50 = 250
    nb_chunks ≈ ceil((1000 - 50) / 250) = ceil(950/250) = ceil(3.8) = 4

UNITÉ DE DÉCOUPAGE : LES MOTS (pas les tokens)
-----------------------------------------------
On découpe sur les espaces (mots), pas sur des tokens BPE. Pourquoi ?
  - Simple à implémenter sans dépendance externe
  - Stable entre modèles (les tokens varient selon le tokenizer)
  - La correspondance mots↔tokens est approximative mais prévisible :
    1 mot ≈ 1.3 tokens en français (langue à mots composés moyens)
    Donc 300 mots ≈ 390 tokens, bien sous la limite de 8 192.

STRUCTURE DE RETOUR
-------------------
Chaque chunk est un dict contenant :
    - "text"        : str   — le texte du chunk (pour l'embedding)
    - "start_word"  : int   — index du premier mot dans le document
    - "end_word"    : int   — index du dernier mot (exclu)
    - "chunk_index" : int   — numéro du chunk (0-indexé)
    - "source"      : str   — nom du document source (optionnel)

Cette structure permet de retrouver la position exacte dans le document
original lors de l'affichage des résultats de recherche.
"""

from __future__ import annotations

import re


def chunk_text(
    text: str,
    chunk_size: int = 300,
    overlap: int = 50,
    source: str = "",
) -> list[dict]:
    """
    Découpe un texte en chunks de taille fixe avec chevauchement.

    Paramètres
    ----------
    text : str
        Texte brut à découper. Peut contenir des sauts de ligne, tabulations,
        etc. — ils sont normalisés en espaces simples.

    chunk_size : int
        Taille maximale de chaque chunk en MOTS (pas en caractères, ni
        en tokens). Doit être > overlap.

    overlap : int
        Nombre de mots partagés entre deux chunks consécutifs.
        Un overlap de 0 produit des chunks disjoints (plus rapide,
        mais risque de couper des phrases à la frontière).
        Doit être < chunk_size.

    source : str
        Nom du document source (ex. nom de fichier). Utile pour
        afficher la provenance dans les résultats de recherche.

    Retourne
    --------
    list[dict]
        Liste de chunks, chacun étant un dict avec les clés :
        "text", "start_word", "end_word", "chunk_index", "source".
        La liste est vide si le texte est vide ou ne contient que
        des espaces.

    Lève
    ----
    ValueError : si chunk_size <= overlap (le stride serait ≤ 0,
                 produisant une boucle infinie).

    Exemples
    --------
    >>> chunks = chunk_text("un deux trois quatre cinq", chunk_size=3, overlap=1)
    >>> len(chunks)
    3
    >>> chunks[0]["text"]
    'un deux trois'
    >>> chunks[1]["text"]
    'trois quatre cinq'
    """
    if chunk_size <= overlap:
        raise ValueError(
            f"chunk_size ({chunk_size}) doit être strictement supérieur "
            f"à overlap ({overlap}). Sinon le stride = chunk_size - overlap "
            f"serait ≤ 0 et le découpage ne progresserait jamais."
        )

    # Normalisation : remplace tout espace blanc (tabulation, newline, etc.)
    # par un espace simple, et supprime les espaces en début/fin.
    normalized = re.sub(r"\s+", " ", text).strip()

    if not normalized:
        return []

    # Tokenisation naïve : on sépare sur les espaces
    # Note : "mots" ici inclut la ponctuation collée (ex. "jour," est un
    # "mot"). C'est acceptable — on indexe du texte brut, pas de l'analyse
    # syntaxique.
    words = normalized.split(" ")
    n_words = len(words)

    stride = chunk_size - overlap  # pas entre le début de deux chunks consécutifs
    chunks: list[dict] = []
    chunk_index = 0
    start = 0

    while start < n_words:
        end = min(start + chunk_size, n_words)  # dernier mot exclu

        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)

        chunks.append(
            {
                "text": chunk_text_str,
                "start_word": start,
                "end_word": end,
                "chunk_index": chunk_index,
                "source": source,
            }
        )

        chunk_index += 1

        # Si ce chunk va jusqu'à la fin du texte, on s'arrête
        if end >= n_words:
            break

        start += stride  # avancer d'un stride (= chunk_size - overlap)

    return chunks


def chunk_documents(
    documents: dict[str, str],
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[dict]:
    """
    Découpe un ensemble de documents en chunks, en conservant la source.

    Paramètres
    ----------
    documents : dict[str, str]
        Dictionnaire {nom_document: texte_brut}.

    chunk_size : int
        Taille de chaque chunk en mots (transmis à chunk_text).

    overlap : int
        Chevauchement entre chunks consécutifs (transmis à chunk_text).

    Retourne
    --------
    list[dict]
        Tous les chunks de tous les documents, concaténés dans l'ordre
        d'itération du dictionnaire. Chaque chunk contient "source" pour
        identifier son document d'origine.

    Exemple
    -------
    >>> docs = {"doc1.txt": "...", "doc2.txt": "..."}
    >>> all_chunks = chunk_documents(docs, chunk_size=200, overlap=30)
    >>> # all_chunks[i]["source"] vaut "doc1.txt" ou "doc2.txt"
    """
    all_chunks: list[dict] = []
    for doc_name, text in documents.items():
        doc_chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap, source=doc_name)
        all_chunks.extend(doc_chunks)
    return all_chunks
