"""
fake_provider.py
----------------
Implémentation "bouchon" d'EmbeddingProvider pour les tests unitaires.

RÔLE MATHÉMATIQUE
-----------------
Le FakeEmbeddingProvider doit produire des vecteurs qui ont les propriétés
mathématiques requises par les tests :

1. DÉTERMINISME : f(texte) retourne TOUJOURS le même vecteur.
   Condition nécessaire pour que les tests soient reproductibles.

2. INJECTIVITÉ (approximative) : des textes DIFFÉRENTS doivent produire
   des vecteurs DIFFÉRENTS, pour que la recherche par similarité ait
   un sens dans les tests.

3. DIMENSION CONFIGURABLE : on veut pouvoir tester en 4D, 8D, etc., sans
   avoir à gérer 768 composantes dans les jeux de données de test.

STRATÉGIE CHOISIE — construction déterministe par hachage
----------------------------------------------------------
On utilise hash() de Python, appliqué au texte, pour obtenir un entier
pseudo-aléatoire déterministe. Puis on distribue cet entier sur `dim`
dimensions en utilisant des décalages de bits :

    Soit h = hash(texte)     ← entier sur 64 bits (en Python 3)
    Pour i de 0 à dim-1 :
        composante_i = sin((h >> (i % 32)) * (i + 1))

Le sinus appliqué sur un grand entier donne des valeurs dans [-1, 1]
qui varient bien d'une dimension à l'autre. Ce n'est pas aléatoire
au sens probabiliste — c'est une bijection déterministe sur ℝᵈⁱᵐ.

LIMITE : hash() n'est pas garanti stable entre processus Python différents
(salage aléatoire par défaut depuis Python 3.3). Pour des tests stables,
on utilise hashlib.md5 à la place, qui est déterministe entre sessions.

    hash_md5(texte) = int(md5(texte.encode()).hexdigest(), 16)

Cela garantit que le même texte → même vecteur, même si on relance Python.

RÔLE FONCTIONNEL
----------------
Avec ce provider, on peut écrire des tests comme :
    - Indexer des textes A, B, C avec des vecteurs connus
    - Vérifier que la recherche de Q retourne A et pas C
    - Sans jamais lancer Ollama

Le provider accepte aussi un dictionnaire explicite text→vector pour
les tests où on veut contrôler précisément les angles entre vecteurs.
"""

import hashlib
import math

from .base import EmbeddingProvider


class FakeEmbeddingProvider(EmbeddingProvider):
    """
    Provider d'embeddings factice pour les tests unitaires.

    Deux modes de fonctionnement :

    Mode 1 — Dictionnaire explicite (contrôle total)
    -------------------------------------------------
    On passe un dict {texte: vecteur} au constructeur. Tout texte présent
    dans ce dict retourne exactement le vecteur spécifié. Utile pour les
    tests qui raisonnent sur des angles précis (ex. cos(45°) = 1/√2).

        provider = FakeEmbeddingProvider(
            explicit_vectors={
                "chien": [1.0, 0.0],
                "chat":  [0.9, 0.1],   # proche de "chien"
                "voiture": [0.0, 1.0], # orthogonal à "chien"
            },
            dim=2
        )

    Mode 2 — Hash déterministe (tests de comportement général)
    -----------------------------------------------------------
    Si le texte n'est pas dans le dictionnaire, on calcule le vecteur par
    hachage MD5. Utile pour tester le pipeline complet (indexer + search)
    sans se préoccuper des valeurs exactes — on vérifie juste que le même
    texte est retrouvé.

        provider = FakeEmbeddingProvider(dim=8)
        v1 = provider.embed("hello")
        v2 = provider.embed("hello")
        assert v1 == v2  # toujours vrai
    """

    def __init__(
        self,
        dim: int = 8,
        explicit_vectors: dict[str, list[float]] | None = None,
    ):
        """
        Paramètres
        ----------
        dim : int
            Dimension des vecteurs générés par hachage (mode 2).
            N'affecte pas les vecteurs du dictionnaire explicite (mode 1).
        explicit_vectors : dict[str, list[float]] | None
            Dictionnaire optionnel texte→vecteur pour le mode 1.
            Si None, seul le mode 2 (hash) est utilisé.
        """
        self._dim = dim
        self._explicit: dict[str, list[float]] = explicit_vectors or {}

    @property
    def dimension(self) -> int:
        """Dimension des vecteurs produits (configurable au constructeur)."""
        return self._dim

    def embed(self, text: str) -> list[float]:
        """
        Retourne un vecteur déterministe pour le texte donné.

        Algorithme (mode 2 — hachage)
        ------------------------------
        Étape 1 : h = MD5(text) converti en entier
                  MD5 produit 128 bits → un entier entre 0 et 2¹²⁸ - 1

        Étape 2 : Pour chaque dimension i (0 à dim-1) :
                  composante_i = sin(h * (i + 1))

                  Pourquoi sin ? La fonction sin est bornée dans [-1, 1]
                  et très sensible aux petits changements de son argument
                  (effet "chaos" quand l'argument est grand). Deux textes
                  différents ont des h très différents → des composantes
                  très différentes.

        Étape 3 : On ne normalise PAS ici. cosine_similarity() se charge
                  de la normalisation — c'est son rôle de gérer les normes.

        Paramètre
        ---------
        text : str
            Texte à encoder.

        Retourne
        --------
        list[float]
            Vecteur de dimension self._dim. Même texte → même vecteur.
        """
        # Mode 1 : vecteur explicitement défini
        if text in self._explicit:
            return list(self._explicit[text])  # copie défensive

        # Mode 2 : génération par hachage MD5
        # MD5 est suffisant ici — on veut du déterminisme, pas de la sécurité.
        md5_hex = hashlib.md5(text.encode("utf-8")).hexdigest()
        h = int(md5_hex, 16)  # entier de 0 à 2^128 - 1

        # Construction des composantes : sin(h * (i+1)) pour i = 0..dim-1
        # La multiplication par (i+1) différencie les dimensions :
        # sans elle, sin(h*1) == sin(h*1) pour tous les i → vecteur uniforme.
        vector = [math.sin(h * (i + 1)) for i in range(self._dim)]
        return vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode tous les textes (boucle simple, synchrone)."""
        return [self.embed(text) for text in texts]
