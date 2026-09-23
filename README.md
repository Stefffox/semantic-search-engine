# semantic-search-engine

Semantic search engine in Python using text embeddings and cosine similarity (implemented from scratch). Includes document indexing with save/reload, top-k search, and a comparison against classic keyword search. Built as a BUT3 university project.

## Features

- Text-to-vector embedding using [Ollama](https://ollama.com) with the `nomic-embed-text` model
- Custom cosine similarity implementation (no external similarity library)
- Document indexing with save and reload support
- Top-k semantic search over an indexed corpus
- Baseline keyword search for comparison
- Test suite covering exact match, synonym, paraphrase, unrelated and ambiguous queries

## Requirements

- Python 3.x
- [Ollama](https://ollama.com) running locally with the `nomic-embed-text` model pulled
- Dependencies listed in `requirements.txt`

## Installation

```bash
git clone https://github.com/Stefffox/semantic-search-engine.git
cd semantic-search-engine
pip install -r requirements.txt
ollama pull nomic-embed-text
```

## Usage

```bash
# Build the index
python index.py

# Run a search
python search.py "your query here"
```

*(commands to update once the CLI is finalized)*

## Project structure

```
semantic-search-engine/
├── embedding.py       # text -> vector
├── similarity.py      # cosine similarity implementation
├── index.py           # indexing (build/save/load)
├── search.py          # search engine (top-k retrieval)
├── keyword_search.py  # baseline keyword search
├── corpus/            # document collection
├── tests/             # test queries and evaluation
└── report/            # project report
```

## Authors

- Nathanaël Daunis ([@Stefffox](https://github.com/Stefffox))
- Santiago GABARRE ([@RuZyoFR](https://https://github.com/RuZyoFR))

## Context

University project (BUT3 Informatique) exploring semantic search and RAG-style retrieval fundamentals.
