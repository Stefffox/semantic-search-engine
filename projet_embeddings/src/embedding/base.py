"""
base.py
-------
Interface abstraite EmbeddingProvider.

RÔLE MATHÉMATIQUE
-----------------
Un "embedding provider" implémente une fonction :

    f : str → ℝⁿ

c'est-à-dire qu'il transforme un texte (chaîne de caractères) en un
vecteur de nombres réels à n dimensions. Tous les providers produisent
des éléments du même espace vectoriel ℝⁿ — donc on peut toujours
calculer une similarité cosinus entre deux vecteurs, peu importe le
provider qui les a générés.

RÔLE FONCTIONNEL
----------------
On utilise une classe abstraite (ABC = Abstract Base Class) pour définir
un "contrat" que toute implémentation concrète doit respecter.

Avantage : le reste du code (indexer.py, search.py, etc.) n'a pas besoin
de savoir si les vecteurs viennent d'Ollama, d'OpenAI, ou d'un fake pour
les tests. Il appelle simplement provider.embed(texte) et récupère un
vecteur de float — c'est le principe de substitution de Liskov (SOLID).

Cela découple :
  - le code mathématique/algorithmique (search, indexer) qui est stable
  - le code d'appel réseau (Ollama) qui peut changer sans tout casser
"""

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """
    Interface abstraite pour tout fournisseur d'embeddings.

    Une sous-classe concrète DOIT implémenter :
      - embed(text)       : texte → vecteur (liste de floats)
      - embed_batch(texts): liste de textes → liste de vecteurs (optimisé)
      - dimension         : propriété retournant la taille des vecteurs

    Les signatures de types garantissent que tous les vecteurs produits
    sont des list[float], donc compatibles avec les fonctions de metrics.py.
    """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """
        Transforme un texte en vecteur d'embedding.

        Paramètre
        ---------
        text : str
            Le texte à encoder. Peut être une phrase, un paragraphe,
            ou un document entier (selon les limites du modèle).

        Retourne
        --------
        list[float]
            Vecteur de dimension self.dimension, chaque composante étant
            un nombre réel. Ce vecteur représente la position du texte
            dans l'espace sémantique ℝⁿ.
        """
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Transforme une liste de textes en une liste de vecteurs.

        Implémentation par défaut : appelle embed() en boucle.
        Les sous-classes peuvent surcharger cette méthode pour envoyer
        une requête batch à l'API (plus efficace que N appels séquentiels).

        Paramètre
        ---------
        texts : list[str]
            Liste de textes à encoder.

        Retourne
        --------
        list[list[float]]
            Liste de vecteurs, dans le même ordre que texts.
            len(résultat) == len(texts) est garanti.
        """
        return [self.embed(text) for text in texts]

    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Retourne la dimension n des vecteurs produits par ce provider.

        Pour nomic-embed-text : 768.
        Pour le FakeEmbeddingProvider de test : configurable.
        """
        ...
