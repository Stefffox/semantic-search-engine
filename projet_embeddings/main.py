"""
main.py
-------
Point d'entrée en ligne de commande pour le projet.

Usage :
    python main.py index   <fichier1.txt> [fichier2.txt ...]
    python main.py search  <"votre requête">
    python main.py eval

Exemples :
    python main.py index data/corpus/biologie.txt data/corpus/histoire.txt
    python main.py search "Comment fonctionne la photosynthèse ?"
    python main.py eval
"""

from __future__ import annotations

import sys
from pathlib import Path


def cmd_index(filepaths: list[str]) -> None:
    """Indexe les fichiers texte donnés et sauvegarde l'index."""
    from src.embedding.ollama_provider import OllamaEmbeddingProvider
    from src.indexer import VectorIndex

    documents: dict[str, str] = {}
    for path_str in filepaths:
        p = Path(path_str)
        if not p.exists():
            print(f"Fichier introuvable : {p}", file=sys.stderr)
            sys.exit(1)
        documents[p.name] = p.read_text(encoding="utf-8")
        print(f"Chargé : {p.name} ({len(documents[p.name].split())} mots)")

    provider = OllamaEmbeddingProvider()
    index = VectorIndex.from_documents(documents, provider, chunk_size=300, overlap=50)
    index.save()


def cmd_search(query: str, top_k: int = 5) -> None:
    """Charge l'index et effectue une recherche sémantique."""
    from src.embedding.ollama_provider import OllamaEmbeddingProvider
    from src.indexer import VectorIndex
    from src.search import SemanticSearch

    provider = OllamaEmbeddingProvider()
    index = VectorIndex.load()
    engine = SemanticSearch(index, provider)
    engine.search_and_display(query, top_k=top_k)


def cmd_eval() -> None:
    """Lance une évaluation comparative Sémantique vs TF-IDF."""
    from src.embedding.ollama_provider import OllamaEmbeddingProvider
    from src.evaluation import evaluate_engine, print_comparison_table
    from src.indexer import VectorIndex
    from src.lexical_search import TFIDFIndex
    from src.search import SemanticSearch

    # Charge l'index existant
    provider = OllamaEmbeddingProvider()
    index = VectorIndex.load()

    # Ground truth minimal d'exemple (à remplacer par le vôtre)
    # Format : {"query": str, "relevant": set de chunk IDs "source::chunk_index"}
    ground_truth = [
        {
            "query": "exemple de requête 1",
            "relevant": set(),  # à remplir avec les vrais IDs pertinents
        },
    ]
    print("⚠️  Définissez votre ground_truth dans main.py cmd_eval() avant de lancer l'évaluation.")
    print("   Format : [{\"query\": \"...\", \"relevant\": {\"source.txt::0\", ...}}, ...]")

    sem_engine = SemanticSearch(index, provider)
    tfidf_index = TFIDFIndex().build(index.metadata)

    sem_metrics = evaluate_engine(sem_engine.search, ground_truth, top_k=5)
    lex_metrics = evaluate_engine(tfidf_index.search, ground_truth, top_k=5)
    print_comparison_table(sem_metrics, lex_metrics, top_k=5)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == "index":
        if len(sys.argv) < 3:
            print("Usage : python main.py index <fichier1.txt> [fichier2.txt ...]")
            sys.exit(1)
        cmd_index(sys.argv[2:])

    elif cmd == "search":
        if len(sys.argv) < 3:
            print('Usage : python main.py search "<votre requête>"')
            sys.exit(1)
        query = " ".join(sys.argv[2:])
        cmd_search(query)

    elif cmd == "eval":
        cmd_eval()

    else:
        print(f"Commande inconnue : '{cmd}'. Choix : index, search, eval")
        sys.exit(1)


if __name__ == "__main__":
    main()
