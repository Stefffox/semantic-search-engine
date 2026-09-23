"""
metrics.py
----------
Implémentation "à la main" de la similarité cosinus entre deux vecteurs.

La similarité cosinus mesure l'angle entre deux vecteurs, indépendamment
de leur norme (longueur). Elle est définie par :

                x . y
    S(x, y) = ---------
              ||x|| ||y||

où x . y est le produit scalaire, et ||x|| la norme euclidienne de x.

Interprétation :
    S = 1  -> vecteurs colinéaires de même sens (angle 0°),  sens identique
    S = 0  -> vecteurs orthogonaux (angle 90°),               aucun rapport
    S = -1 -> vecteurs colinéaires de sens opposé (angle 180°)
"""

import math
from typing import Sequence


def dot_product(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Calcule le produit scalaire de deux vecteurs : sum(x_i * y_i).

    Fonctionnellement : chaque paire de composantes (x_i, y_i) est
    multipliée, puis on additionne tout. Le résultat est grand quand
    les deux vecteurs "varient dans le même sens" composante par
    composante.
    """
    if len(x) != len(y):
        raise ValueError(
            f"Les vecteurs doivent avoir la même dimension "
            f"(reçu {len(x)} et {len(y)})"
        )
    return sum(xi * yi for xi, yi in zip(x, y))


def norm(x: Sequence[float]) -> float:
    """
    Calcule la norme euclidienne (longueur) d'un vecteur : sqrt(sum(x_i^2)).

    C'est la généralisation du théorème de Pythagore à n dimensions :
    la distance entre l'origine (0, ..., 0) et le point x.
    """
    return math.sqrt(sum(xi * xi for xi in x))


def cosine_similarity(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Calcule la similarité cosinus entre deux vecteurs x et y.

    Étapes :
        1. produit scalaire de x et y
        2. norme de x, norme de y
        3. division du produit scalaire par le produit des deux normes

    Cas limite : si l'un des vecteurs est nul (norme = 0), l'angle n'est
    pas défini mathématiquement (division par zéro). On retourne alors 0
    par convention : un vecteur nul n'a pas de direction, donc pas de
    similarité avec quoi que ce soit.

    Retourne un score entre -1 et 1 (en pratique proche de [0, 1] pour
    des embeddings de texte).
    """
    norme_x = norm(x)
    norme_y = norm(y)

    if norme_x == 0 or norme_y == 0:
        return 0.0

    return dot_product(x, y) / (norme_x * norme_y)
