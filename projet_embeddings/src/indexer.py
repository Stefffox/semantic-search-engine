"""
indexer.py
----------
Construction, sauvegarde et chargement d'un index vectoriel de chunks.

STRUCTURE MATHÉMATIQUE DE L'INDEX
-----------------------------------
Après découpage en K chunks et calcul des embeddings, on dispose de :

    Matrice des vecteurs  M ∈ ℝ^(K × n)
        M[i] = f(chunk_i) ∈ ℝⁿ
        Chaque ligne est le vecteur d'embedding du chunk i.

    Métadonnées  meta[i] = { "text", "source", "chunk_index", ... }
        Informations pour retrouver et afficher le chunk i.

La recherche consiste ensuite à calculer, pour une requête q ∈ ℝⁿ :

    scores[i] = cosine_similarity(q, M[i])   pour i = 0..K-1

et retourner les k indices avec le score le plus élevé.

Cette recherche exhaustive est en O(K × n). Pour nomic-embed-text (n=768),
un corpus de 10 000 chunks nécessite 10 000 × 768 = 7.68M multiplications
— parfaitement faisable en Python pur.

PERSISTANCE SUR DISQUE
-----------------------
On sauvegarde deux fichiers dans data/index/ :

    vectors.bin    — la matrice M en binaire (float32, big-endian)
                     Format maison : [K: uint32][n: uint32][K*n float32]
                     Compact, lisible sans numpy, portable.

    metadata.json  — liste de K dicts, un par chunk + header JSON
                     Lisible par un humain (débogage, inspection).

IMPLÉMENTATION SANS NUMPY
--------------------------
On utilise le module 'struct' de la bibliothèque standard pour lire/écrire
des float32 en binaire. Pas de dépendance externe — le projet tourne
avec Python pur (compatible Python 3.15 alpha).

Le calcul de similarité cosinus reste dans metrics.py (pur Python).
"""

from __future__ import annotations

import json
import struct
from datetime import datetime
from pathlib import Path

from .chunker import chunk_documents
from .embedding.base import EmbeddingProvider
from .metrics import cosine_similarity

# Chemins par défaut (relatifs à la racine du projet)
_DEFAULT_VECTORS_PATH = Path("data/index/vectors.bin")
_DEFAULT_METADATA_PATH = Path("data/index/metadata.json")


class VectorIndex:
    """
    Index vectoriel en mémoire : matrice de vecteurs + métadonnées.

    Attributs
    ---------
    vectors : list[list[float]]   shape logique (K, n)
        Matrice des embeddings stockée comme liste de listes Python.
        vectors[i] = vecteur du chunk i, list[float] de longueur n.
    metadata : list[dict]
        Liste de K dicts, un par chunk. Contient au minimum "text" et "source".
    provider_info : dict
        Informations sur le provider utilisé pour la construction.

    Utilisation typique
    -------------------
        idx = VectorIndex.build(chunks, provider)
        idx.save()
        # ... plus tard ...
        idx = VectorIndex.load()
        results = idx.search("ma requête", top_k=5)
    """

    def __init__(
        self,
        vectors: list[list[float]],
        metadata: list[dict],
        provider_info: dict | None = None,
    ):
        """
        Paramètres
        ----------
        vectors : list[list[float]]
            Matrice des vecteurs d'embeddings (liste de K vecteurs de dim n).
        metadata : list[dict]
            Liste de K dicts correspondant aux chunks.
        provider_info : dict | None
            Métadonnées optionnelles sur le provider (modèle, dimension...).
        """
        if len(vectors) != len(metadata):
            raise ValueError(
                f"Incohérence : {len(vectors)} vecteurs mais {len(metadata)} "
                f"entrées de métadonnées. Les deux listes doivent avoir la même taille."
            )
        self.vectors: list[list[float]] = vectors
        self.metadata = metadata          # list[dict], len = K
        self.provider_info = provider_info or {}

    @property
    def num_chunks(self) -> int:
        """Nombre de chunks indexés (K)."""
        return len(self.metadata)

    @property
    def embedding_dim(self) -> int:
        """Dimension n des vecteurs (0 si l'index est vide)."""
        if self.vectors:
            return len(self.vectors[0])
        return 0

    # ------------------------------------------------------------------
    # Construction de l'index
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        chunks: list[dict],
        provider: EmbeddingProvider,
        verbose: bool = True,
    ) -> "VectorIndex":
        """
        Construit un index en calculant l'embedding de chaque chunk.

        Algorithme
        ----------
        Pour chaque chunk i (0 ≤ i < K) :
            M[i] = provider.embed(chunk_i["text"])   ← vecteur dans ℝⁿ

        Paramètres
        ----------
        chunks : list[dict]
            Chunks produits par chunker.chunk_text() ou chunk_documents().
        provider : EmbeddingProvider
            Provider d'embeddings (Ollama ou Fake).
        verbose : bool
            Si True, affiche la progression dans la console.

        Retourne
        --------
        VectorIndex
            Index prêt à l'emploi (en mémoire, pas encore sauvegardé).
        """
        if not chunks:
            raise ValueError("La liste de chunks est vide — rien à indexer.")

        vectors_list: list[list[float]] = []

        for i, chunk in enumerate(chunks):
            if verbose and (i % 10 == 0 or i == len(chunks) - 1):
                print(f"  Indexation chunk {i + 1}/{len(chunks)} ...", end="\r")

            vector = provider.embed(chunk["text"])
            vectors_list.append(vector)

        if verbose:
            print(f"\n  ✓ {len(chunks)} chunks indexés.")

        provider_info = {
            "dimension": len(vectors_list[0]) if vectors_list else 0,
            "class": type(provider).__name__,
            "indexed_at": datetime.now().isoformat(),
        }

        return cls(vectors_list, chunks, provider_info)

    @classmethod
    def from_documents(
        cls,
        documents: dict[str, str],
        provider: EmbeddingProvider,
        chunk_size: int = 300,
        overlap: int = 50,
        verbose: bool = True,
    ) -> "VectorIndex":
        """
        Raccourci : découpe les documents en chunks, puis construit l'index.

        Paramètres
        ----------
        documents : dict[str, str]
            Dictionnaire {nom_document: texte_brut}.
        provider : EmbeddingProvider
            Provider d'embeddings.
        chunk_size : int
            Taille des chunks en mots.
        overlap : int
            Chevauchement entre chunks consécutifs.
        verbose : bool
            Affiche la progression.

        Retourne
        --------
        VectorIndex
        """
        if verbose:
            print(f"Découpage de {len(documents)} document(s) en chunks...")

        chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)

        if verbose:
            print(f"  → {len(chunks)} chunks générés (chunk_size={chunk_size}, overlap={overlap})")

        return cls.build(chunks, provider, verbose=verbose)

    # ------------------------------------------------------------------
    # Persistance sur disque
    # ------------------------------------------------------------------

    def save(
        self,
        vectors_path: str | Path = _DEFAULT_VECTORS_PATH,
        metadata_path: str | Path = _DEFAULT_METADATA_PATH,
    ) -> None:
        """
        Sauvegarde l'index sur disque.

        Crée les répertoires parents si nécessaire.

        Fichiers produits
        -----------------
        vectors.bin    : matrice float32 en binaire pur (struct)
                         Format : [K: uint32 LE][n: uint32 LE][K*n float32 LE]
                         Lisible sans numpy, en Python pur.
        metadata.json  : liste de K dicts + header JSON
                         Lisible par un humain pour inspection.

        Paramètres
        ----------
        vectors_path : str | Path
            Chemin du fichier .bin pour les vecteurs.
        metadata_path : str | Path
            Chemin du fichier .json pour les métadonnées.
        """
        vectors_path = Path(vectors_path)
        metadata_path = Path(metadata_path)

        # Crée les dossiers parents si besoin
        vectors_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        # Sauvegarde la matrice en binaire via struct
        # Format : 4 octets K (uint32) + 4 octets n (uint32) + K*n float32
        K = self.num_chunks
        n = self.embedding_dim
        with open(vectors_path, "wb") as f:
            f.write(struct.pack("<II", K, n))  # en-tête : K et n en little-endian
            for vec in self.vectors:
                f.write(struct.pack(f"<{n}f", *vec))

        # Sauvegarde les métadonnées avec un en-tête informatif
        payload = {
            "_header": {
                "num_chunks": self.num_chunks,
                "embedding_dim": self.embedding_dim,
                **self.provider_info,
            },
            "chunks": self.metadata,
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"Index sauvegardé : {vectors_path} ({self.num_chunks} chunks, dim={self.embedding_dim})")

    @classmethod
    def load(
        cls,
        vectors_path: str | Path = _DEFAULT_VECTORS_PATH,
        metadata_path: str | Path = _DEFAULT_METADATA_PATH,
    ) -> "VectorIndex":
        """
        Charge un index depuis le disque.

        Paramètres
        ----------
        vectors_path : str | Path
            Chemin du fichier .bin des vecteurs.
        metadata_path : str | Path
            Chemin du fichier .json des métadonnées.

        Retourne
        --------
        VectorIndex
            Index chargé en mémoire, prêt pour la recherche.

        Lève
        ----
        FileNotFoundError : si l'un des fichiers n'existe pas.
        """
        vectors_path = Path(vectors_path)
        metadata_path = Path(metadata_path)

        if not vectors_path.exists():
            raise FileNotFoundError(
                f"Fichier de vecteurs introuvable : {vectors_path}\n"
                f"Lancez d'abord l'indexation avec VectorIndex.build() ou .from_documents()."
            )
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Fichier de métadonnées introuvable : {metadata_path}"
            )

        # Lecture binaire via struct
        with open(vectors_path, "rb") as f:
            K, n = struct.unpack("<II", f.read(8))  # en-tête : K et n
            vectors: list[list[float]] = []
            float_size = n * 4  # n floats × 4 octets
            for _ in range(K):
                row_bytes = f.read(float_size)
                row = list(struct.unpack(f"<{n}f", row_bytes))
                vectors.append(row)

        with open(metadata_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        provider_info = payload.get("_header", {})
        metadata = payload.get("chunks", [])

        print(f"Index chargé : {K} chunks, dim={n}")
        return cls(vectors, metadata, provider_info)

    # ------------------------------------------------------------------
    # Recherche
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Retourne les top_k chunks les plus similaires au vecteur requête.

        Algorithme — Recherche exhaustive
        ----------------------------------
        Pour chaque chunk i (0 ≤ i < K) :
            score_i = cosine_similarity(query_vector, M[i])

        Puis on trie par score décroissant et on retourne les top_k.

        Complexité : O(K × n) — linéaire en K, la taille du corpus.
        Pour des petits corpus (< 100 000 chunks), c'est très acceptable.
        Pour des grands corpus, on utiliserait FAISS ou HNSW, mais ce
        n'est pas l'objet de ce projet.

        Paramètres
        ----------
        query_vector : list[float]
            Vecteur de la requête, dans le même espace ℝⁿ que les chunks.
            Doit avoir la même dimension que les vecteurs de l'index.
        top_k : int
            Nombre de résultats à retourner.

        Retourne
        --------
        list[dict]
            Liste de top_k dicts, triés par score décroissant.
            Chaque dict contient toutes les métadonnées du chunk plus
            une clé "score" avec la similarité cosinus (float dans [-1, 1]).
        """
        if self.num_chunks == 0:
            return []

        # Calcul des scores pour tous les chunks
        scored: list[tuple[float, int]] = []
        for i in range(self.num_chunks):
            chunk_vector = self.vectors[i]  # déjà list[float]
            score = cosine_similarity(query_vector, chunk_vector)
            scored.append((score, i))

        # Tri décroissant par score
        scored.sort(key=lambda t: t[0], reverse=True)

        # Construction de la liste de résultats
        results: list[dict] = []
        for score, idx in scored[:top_k]:
            result = dict(self.metadata[idx])  # copie défensive
            result["score"] = round(score, 6)
            results.append(result)

        return results
