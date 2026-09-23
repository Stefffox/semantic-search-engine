"""
ollama_provider.py
------------------
Implémentation concrète d'EmbeddingProvider qui utilise l'API REST d'Ollama.

RÔLE MATHÉMATIQUE
-----------------
Ollama expose localement un modèle nomic-embed-text qui implémente
une fonction entraînée :

    f_θ : str → ℝ⁷⁶⁸

Les paramètres θ (des milliards de poids d'un réseau de neurones) ont été
optimisés pendant l'entraînement pour que des textes sémantiquement proches
produisent des vecteurs proches en cosinus. On n'a pas à comprendre le
réseau de neurones ici — on appelle juste f_θ via HTTP et on récupère le
vecteur de 768 flottants.

RÔLE FONCTIONNEL
----------------
Ollama expose une API compatible OpenAI. Le endpoint qui nous intéresse est :

    POST http://localhost:11434/api/embeddings
    Body : { "model": "nomic-embed-text", "prompt": "votre texte" }
    Réponse : { "embedding": [0.123, -0.456, ..., 0.789] }  ← liste de 768 floats

On utilise urllib (bibliothèque standard Python, sans dépendance externe)
pour faire la requête HTTP. Si requests est disponible, c'est encore plus
simple, mais urllib suffit.

GESTION DES ERREURS
-------------------
Deux types d'erreurs doivent être gérées proprement :
  - Ollama ne tourne pas   → ConnectionError claire, pas une KeyError cryptique
  - Mauvais modèle/réponse → ValueError avec le message d'Ollama
"""

import json
import urllib.error
import urllib.request
from typing import Optional

from .base import EmbeddingProvider

# URL par défaut du serveur Ollama local
_DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Modèle par défaut — nomic-embed-text produit des vecteurs de 768 dimensions
_DEFAULT_MODEL = "nomic-embed-text"


class OllamaEmbeddingProvider(EmbeddingProvider):
    """
    Fournisseur d'embeddings via l'API REST d'Ollama (serveur local).

    Prérequis :
      1. Ollama installé et lancé : `ollama serve`
      2. Modèle téléchargé       : `ollama pull nomic-embed-text`

    Exemple d'utilisation :
        provider = OllamaEmbeddingProvider()
        vecteur = provider.embed("Le chat mange une souris.")
        # vecteur est une liste de 768 floats
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        base_url: str = _DEFAULT_OLLAMA_URL,
        timeout: int = 30,
    ):
        """
        Paramètres
        ----------
        model : str
            Nom du modèle Ollama à utiliser. Doit être un modèle
            d'embeddings (pas un modèle de génération de texte).
        base_url : str
            URL racine du serveur Ollama. "http://localhost:11434" par défaut.
        timeout : int
            Délai maximal (en secondes) pour attendre la réponse d'Ollama.
            Un texte très long peut prendre quelques secondes à encoder.
        """
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._dimension: Optional[int] = None  # découvert au premier appel

    @property
    def dimension(self) -> int:
        """
        Retourne la dimension des vecteurs.

        La dimension est découverte dynamiquement au premier appel embed()
        (lazy initialization) : on ne suppose pas 768, on lit la taille
        réelle retournée par Ollama. Pour nomic-embed-text, ce sera 768.

        Lève RuntimeError si embed() n'a jamais été appelé.
        """
        if self._dimension is None:
            raise RuntimeError(
                "La dimension n'est pas encore connue. "
                "Appelez embed() au moins une fois d'abord."
            )
        return self._dimension

    def embed(self, text: str) -> list[float]:
        """
        Envoie le texte à Ollama et retourne le vecteur d'embedding.

        Protocole HTTP
        --------------
        Requête :
            POST /api/embeddings
            Content-Type: application/json
            {"model": "nomic-embed-text", "prompt": "<text>"}

        Réponse (JSON) :
            {"embedding": [f1, f2, ..., f768]}

        Le vecteur retourné est directement utilisable dans cosine_similarity().

        Paramètre
        ---------
        text : str
            Texte à encoder. Les espaces en début/fin sont nettoyés.

        Retourne
        --------
        list[float]
            Vecteur d'embedding (768 floats pour nomic-embed-text).

        Lève
        ----
        ConnectionError : si Ollama n'est pas joignable (pas lancé).
        ValueError      : si la réponse Ollama est inattendue.
        """
        url = f"{self._base_url}/api/embeddings"
        payload = json.dumps({"model": self._model, "prompt": text.strip()}).encode()

        request = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = json.loads(response.read().decode())
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Impossible de joindre Ollama sur {self._base_url}.\n"
                f"Vérifiez qu'Ollama est lancé (`ollama serve`) et que le modèle "
                f"'{self._model}' est téléchargé (`ollama pull {self._model}`).\n"
                f"Détail : {exc}"
            ) from exc

        if "embedding" not in body:
            raise ValueError(
                f"La réponse d'Ollama ne contient pas de clé 'embedding'.\n"
                f"Réponse reçue : {body}\n"
                f"Vérifiez que '{self._model}' est bien un modèle d'embeddings."
            )

        vector: list[float] = body["embedding"]

        # Mémorise la dimension au premier appel
        if self._dimension is None:
            self._dimension = len(vector)

        return vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Encode une liste de textes en N requêtes séquentielles vers Ollama.

        Note : l'API Ollama /api/embeddings ne supporte pas le batch natif
        (contrairement à OpenAI /v1/embeddings). On boucle donc, mais on
        pourrait paralléliser avec threading si la performance devenait
        un enjeu.

        Retourne
        --------
        list[list[float]]
            Vecteurs dans le même ordre que texts.
        """
        return [self.embed(text) for text in texts]
