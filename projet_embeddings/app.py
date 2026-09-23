"""
app.py
------
Interface interactive en ligne de commande (REPL) pour le moteur de recherche.

Lance une boucle de recherche interactive :
    python app.py

On peut y taper des requêtes en continu sans relancer le script,
l'index est chargé une seule fois en mémoire.
"""

from __future__ import annotations

import sys
from pathlib import Path


def run_app() -> None:
    try:
        from src.embedding.ollama_provider import OllamaEmbeddingProvider
        from src.indexer import VectorIndex
        from src.lexical_search import TFIDFIndex
        from src.search import SemanticSearch
    except ImportError as e:
        print(f"Erreur d'import : {e}")
        sys.exit(1)

    print("=" * 60)
    print(" Moteur de recherche sémantique — projet_embeddings")
    print("=" * 60)

    # Chargement de l'index
    try:
        provider = OllamaEmbeddingProvider()
        index = VectorIndex.load()
    except FileNotFoundError:
        print("\n⚠️  Aucun index trouvé. Lancez d'abord :")
        print("    python main.py index <vos_fichiers.txt>")
        sys.exit(1)
    except ConnectionError as e:
        print(f"\n⚠️  Ollama inaccessible : {e}")
        sys.exit(1)

    sem_engine = SemanticSearch(index, provider)
    tfidf_index = TFIDFIndex().build(index.metadata)

    print(f"\nIndex chargé : {index.num_chunks} chunks, dim={index.embedding_dim}")
    print("Tapez votre requête (ou 'quit' pour quitter, 'mode' pour changer de moteur)\n")

    mode = "semantic"  # "semantic" ou "lexical"

    while True:
        try:
            user_input = input(f"[{mode}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir !")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Au revoir !")
            break
        if user_input.lower() == "mode":
            mode = "lexical" if mode == "semantic" else "semantic"
            print(f"Mode basculé → {mode}\n")
            continue

        # Recherche
        if mode == "semantic":
            results = sem_engine.search(user_input, top_k=5)
        else:
            results = tfidf_index.search(user_input, top_k=5)

        if not results:
            print("  Aucun résultat.\n")
            continue

        print(f'\nTop 5 résultats pour : "{user_input}"\n')
        for rank, r in enumerate(results, 1):
            score = r.get("score", 0.0)
            source = r.get("source", "?")
            text_preview = r.get("text", "")[:200]
            ellipsis = "..." if len(r.get("text", "")) > 200 else ""
            print(f"  #{rank}  score={score:.4f}  [{source}]")
            print(f"       {text_preview}{ellipsis}\n")


if __name__ == "__main__":
    run_app()
