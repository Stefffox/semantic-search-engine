"""
test_search.py
--------------
Tests d'intégration du pipeline complet (chunker → indexer → search)
en utilisant FakeEmbeddingProvider — sans dépendance vers Ollama.

STRATÉGIE DE TEST
-----------------
On utilise le mode "dictionnaire explicite" du FakeEmbeddingProvider
pour contrôler EXACTEMENT les vecteurs produits. Cela nous permet de
raisonner mathématiquement sur les résultats attendus.

Exemple :
    provider = FakeEmbeddingProvider(
        dim=2,
        explicit_vectors={
            "Le ciel est bleu": [1.0, 0.0],   # axe x
            "Les nuages sont blancs": [0.9, 0.1],  # proche de l'axe x
            "La mer est profonde": [0.0, 1.0], # axe y, orthogonal au premier
        }
    )
    requête "couleur du ciel" → vecteur [1.0, 0.0]
    → score avec "ciel bleu" >> score avec "mer profonde"
    → ordre des résultats garanti mathématiquement

TESTS COUVERTS
--------------
1. Pipeline complet chunker → indexer → search (retourne les bons chunks)
2. Ordre des résultats (le plus similaire en premier)
3. top_k fonctionne (retourne exactement k résultats)
4. Persistance sur disque (save → load → search donne même résultat)
5. Évaluation P@k et MRR avec ground truth artificiel
"""

import math
import sys
import tempfile
from pathlib import Path

# Permet d'importer src/ sans installer le projet comme package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunker import chunk_text
from src.embedding.fake_provider import FakeEmbeddingProvider
from src.evaluation import (
    chunk_id,
    evaluate_engine,
    mean_reciprocal_rank,
    mean_precision_at_k,
    precision_at_k,
    reciprocal_rank,
)
from src.indexer import VectorIndex
from src.lexical_search import TFIDFIndex
from src.search import SemanticSearch


# ---------------------------------------------------------------------------
# Fixture : corpus et provider contrôlés
# ---------------------------------------------------------------------------

def _make_corpus_and_provider():
    """
    Crée un corpus de 4 chunks avec des vecteurs explicites contrôlés.

    Géométrie des vecteurs (dim=2) :
        "astronomie soleil"  → [1.0, 0.0]   axe x positif
        "astronomie lune"    → [0.8, 0.6]   proche de l'axe x (cos = 0.8)
        "cuisine recette"    → [0.0, 1.0]   axe y (orthogonal à x)
        "cuisine ingrédients"→ [0.1, 0.9]   proche de l'axe y
    """
    texts = {
        "astronomie soleil": [1.0, 0.0],
        "astronomie lune": [0.8, 0.6],
        "cuisine recette": [0.0, 1.0],
        "cuisine ingrédients": [0.1, 0.9],
        # Requête test
        "planètes étoiles": [0.95, 0.3],  # proche de l'axe x → astronomie
        "préparation plat": [0.05, 0.95], # proche de l'axe y → cuisine
    }
    provider = FakeEmbeddingProvider(dim=2, explicit_vectors=texts)

    # Chunks manuels (on n'utilise pas le chunker ici pour garder le contrôle)
    chunks = [
        {"text": "astronomie soleil", "source": "astro.txt", "chunk_index": 0,
         "start_word": 0, "end_word": 2},
        {"text": "astronomie lune", "source": "astro.txt", "chunk_index": 1,
         "start_word": 2, "end_word": 4},
        {"text": "cuisine recette", "source": "cuisine.txt", "chunk_index": 0,
         "start_word": 0, "end_word": 2},
        {"text": "cuisine ingrédients", "source": "cuisine.txt", "chunk_index": 1,
         "start_word": 2, "end_word": 4},
    ]
    return chunks, provider


# ---------------------------------------------------------------------------
# Tests de l'indexeur
# ---------------------------------------------------------------------------

def test_build_index_taille():
    """VectorIndex.build() doit créer un index avec K lignes."""
    chunks, provider = _make_corpus_and_provider()
    index = VectorIndex.build(chunks, provider, verbose=False)
    assert index.num_chunks == 4, f"Attendu 4 chunks, obtenu {index.num_chunks}"
    assert index.embedding_dim == 2, f"Attendu dim=2, obtenu {index.embedding_dim}"


def test_search_ordre_correct():
    """
    La requête "planètes étoiles" ([0.95, 0.3]) doit retourner les chunks
    d'astronomie avant les chunks de cuisine.

    Calcul à la main :
        cos([0.95,0.3], [1.0,0.0]) = (0.95×1 + 0.3×0) / (||[0.95,0.3]|| × 1)
                                   = 0.95 / sqrt(0.9025 + 0.09) = 0.95 / 1.0 ≈ 0.95
        cos([0.95,0.3], [0.8,0.6]) = (0.76 + 0.18) / (1.0 × 1.0) = 0.94
        cos([0.95,0.3], [0.0,1.0]) = 0.3 / 1.0 = 0.3
        cos([0.95,0.3], [0.1,0.9]) = (0.095 + 0.27) / 1.0 ≈ 0.37

    → Les 2 premiers résultats doivent être des chunks d'astronomie.
    """
    chunks, provider = _make_corpus_and_provider()
    index = VectorIndex.build(chunks, provider, verbose=False)
    engine = SemanticSearch(index, provider)

    results = engine.search("planètes étoiles", top_k=4)

    assert len(results) == 4
    # Les 2 meilleurs doivent être de source "astro.txt"
    assert results[0]["source"] == "astro.txt", (
        f"Attendu 'astro.txt' en #1, obtenu '{results[0]['source']}' "
        f"(score={results[0]['score']})"
    )
    assert results[1]["source"] == "astro.txt", (
        f"Attendu 'astro.txt' en #2, obtenu '{results[1]['source']}'"
    )
    # Le score doit être décroissant
    assert results[0]["score"] >= results[1]["score"]
    assert results[1]["score"] >= results[2]["score"]


def test_search_top_k():
    """search(top_k=2) doit retourner exactement 2 résultats."""
    chunks, provider = _make_corpus_and_provider()
    index = VectorIndex.build(chunks, provider, verbose=False)
    engine = SemanticSearch(index, provider)

    results = engine.search("planètes étoiles", top_k=2)
    assert len(results) == 2


def test_search_scores_entre_moins1_et_1():
    """Tous les scores doivent être dans [-1, 1] (propriété cosinus)."""
    chunks, provider = _make_corpus_and_provider()
    index = VectorIndex.build(chunks, provider, verbose=False)
    engine = SemanticSearch(index, provider)

    results = engine.search("planètes étoiles", top_k=4)
    for r in results:
        assert -1.0 <= r["score"] <= 1.0, f"Score hors bornes : {r['score']}"


# ---------------------------------------------------------------------------
# Tests de persistance (save → load)
# ---------------------------------------------------------------------------

def test_save_load_round_trip():
    """
    Après save() + load(), les résultats de search() doivent être identiques.
    """
    chunks, provider = _make_corpus_and_provider()
    index = VectorIndex.build(chunks, provider, verbose=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        vec_path = Path(tmpdir) / "vectors.bin"
        meta_path = Path(tmpdir) / "metadata.json"

        index.save(vec_path, meta_path)
        loaded_index = VectorIndex.load(vec_path, meta_path)

    engine_orig = SemanticSearch(index, provider)
    engine_load = SemanticSearch(loaded_index, provider)

    results_orig = engine_orig.search("planètes étoiles", top_k=4)
    results_load = engine_load.search("planètes étoiles", top_k=4)

    assert len(results_orig) == len(results_load)
    for r_o, r_l in zip(results_orig, results_load):
        assert r_o["source"] == r_l["source"]
        assert math.isclose(r_o["score"], r_l["score"], abs_tol=1e-4), (
            f"Scores divergent après round-trip : {r_o['score']} vs {r_l['score']}"
        )


# ---------------------------------------------------------------------------
# Tests du chunker
# ---------------------------------------------------------------------------

def test_chunker_nombre_chunks():
    """
    Texte de 10 mots, chunk_size=4, overlap=1 :
        stride = 4 - 1 = 3
        chunks : [0..3], [3..6], [6..9]  → 3 chunks (le dernier peut être plus court)

    Calcul :
        mots = w0 w1 w2 w3 w4 w5 w6 w7 w8 w9   (10 mots)
        chunk0 : w0 w1 w2 w3
        chunk1 : w3 w4 w5 w6
        chunk2 : w6 w7 w8 w9   → fin atteinte, arrêt
        → 3 chunks
    """
    texte = "un deux trois quatre cinq six sept huit neuf dix"
    chunks = chunk_text(texte, chunk_size=4, overlap=1)
    assert len(chunks) == 3, f"Attendu 3 chunks, obtenu {len(chunks)}: {[c['text'] for c in chunks]}"


def test_chunker_chevauchement():
    """Le dernier mot du chunk k doit être le premier mot du chunk k+1."""
    texte = "un deux trois quatre cinq six sept"
    chunks = chunk_text(texte, chunk_size=3, overlap=1)
    # chunk0 = "un deux trois", chunk1 = "trois quatre cinq"
    mots_chunk0 = chunks[0]["text"].split()
    mots_chunk1 = chunks[1]["text"].split()
    assert mots_chunk0[-1] == mots_chunk1[0], (
        f"Chevauchement attendu : dernier mot de chunk0='{mots_chunk0[-1]}' "
        f"doit être le premier mot de chunk1='{mots_chunk1[0]}'"
    )


def test_chunker_texte_vide():
    """chunk_text sur un texte vide doit retourner une liste vide."""
    chunks = chunk_text("", chunk_size=100, overlap=10)
    assert chunks == []


# ---------------------------------------------------------------------------
# Tests d'évaluation (P@k, MRR)
# ---------------------------------------------------------------------------

def test_precision_at_k_perfect():
    """P@2 = 1.0 si les 2 premiers résultats sont pertinents."""
    results = [
        {"source": "a.txt", "chunk_index": 0},
        {"source": "a.txt", "chunk_index": 1},
        {"source": "b.txt", "chunk_index": 0},
    ]
    relevant = {"a.txt::0", "a.txt::1"}
    assert precision_at_k(results, relevant, k=2) == 1.0


def test_precision_at_k_zero():
    """P@2 = 0.0 si aucun des 2 premiers résultats n'est pertinent."""
    results = [
        {"source": "b.txt", "chunk_index": 0},
        {"source": "b.txt", "chunk_index": 1},
        {"source": "a.txt", "chunk_index": 0},  # pertinent mais en position 3
    ]
    relevant = {"a.txt::0"}
    assert precision_at_k(results, relevant, k=2) == 0.0


def test_reciprocal_rank_premier_pertinent():
    """RR = 1.0 si le premier résultat est pertinent."""
    results = [{"source": "a.txt", "chunk_index": 0}]
    relevant = {"a.txt::0"}
    assert reciprocal_rank(results, relevant) == 1.0


def test_reciprocal_rank_troisieme_position():
    """RR = 1/3 si le premier pertinent est en 3ème position."""
    results = [
        {"source": "b.txt", "chunk_index": 0},
        {"source": "b.txt", "chunk_index": 1},
        {"source": "a.txt", "chunk_index": 0},  # pertinent en position 3
    ]
    relevant = {"a.txt::0"}
    rr = reciprocal_rank(results, relevant)
    assert math.isclose(rr, 1 / 3, abs_tol=1e-9), f"Attendu 1/3 ≈ 0.333, obtenu {rr}"


def test_mrr_calcul_manuel():
    """
    MRR sur 3 requêtes :
        q1 : premier pertinent en position 1 → RR = 1.0
        q2 : premier pertinent en position 2 → RR = 0.5
        q3 : aucun pertinent                 → RR = 0.0
        MRR = (1.0 + 0.5 + 0.0) / 3 = 0.5
    """
    q1_results = [{"source": "a.txt", "chunk_index": 0}]
    q2_results = [
        {"source": "b.txt", "chunk_index": 0},
        {"source": "a.txt", "chunk_index": 0},
    ]
    q3_results = [{"source": "b.txt", "chunk_index": 0}]

    relevant_all = {"a.txt::0"}

    queries_results = [
        (q1_results, relevant_all),
        (q2_results, relevant_all),
        (q3_results, relevant_all),
    ]
    mrr = mean_reciprocal_rank(queries_results)
    assert math.isclose(mrr, 0.5, abs_tol=1e-9), f"Attendu MRR=0.5, obtenu {mrr}"


# ---------------------------------------------------------------------------
# Test du moteur lexical TF-IDF
# ---------------------------------------------------------------------------

def test_tfidf_retourne_chunk_avec_mot_cle():
    """
    Un chunk contenant exactement les mots de la requête doit avoir le score
    le plus élevé.
    """
    chunks = [
        {"text": "le chat mange une souris", "source": "doc.txt", "chunk_index": 0,
         "start_word": 0, "end_word": 5},
        {"text": "le chien aboie dans le jardin", "source": "doc.txt", "chunk_index": 1,
         "start_word": 5, "end_word": 11},
        {"text": "la voiture roule sur la route", "source": "doc.txt", "chunk_index": 2,
         "start_word": 11, "end_word": 17},
    ]
    index = TFIDFIndex().build(chunks)
    results = index.search("chat souris", top_k=3)

    assert results[0]["chunk_index"] == 0, (
        f"Le chunk 0 ('chat souris') doit être premier, obtenu chunk {results[0]['chunk_index']}"
    )


# ---------------------------------------------------------------------------
# Runner manuel
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"OK  - {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL - {t.__name__}: {e}")
        except Exception as e:
            print(f"ERR  - {t.__name__}: {type(e).__name__}: {e}")

    print(f"\n{passed}/{len(tests)} tests passés.")
