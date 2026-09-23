"""
test_metrics.py
----------------
Tests unitaires de la similarité cosinus, sur des vecteurs dont le
résultat est calculable à la main (voir le rapport pour le détail des
calculs).
"""

import math
import sys
from pathlib import Path

# Permet d'importer src/ sans installer le projet comme package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import cosine_similarity, dot_product, norm


def test_dot_product_simple():
    # (1*4) + (2*5) + (3*6) = 4 + 10 + 18 = 32
    assert dot_product([1, 2, 3], [4, 5, 6]) == 32


def test_dot_product_dimension_mismatch():
    try:
        dot_product([1, 2], [1, 2, 3])
        assert False, "une ValueError aurait dû être levée"
    except ValueError:
        pass


def test_norm_simple():
    # sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5 (triplet 3-4-5)
    assert norm([3, 4]) == 5.0


def test_cosine_vecteurs_colineaires_meme_sens():
    # (1,0) et (2,0) pointent dans la même direction -> angle 0 -> S = 1
    resultat = cosine_similarity([1, 0], [2, 0])
    assert math.isclose(resultat, 1.0, abs_tol=1e-9)


def test_cosine_vecteurs_orthogonaux():
    # (1,0) et (0,1) -> angle 90° -> produit scalaire nul -> S = 0
    resultat = cosine_similarity([1, 0], [0, 1])
    assert math.isclose(resultat, 0.0, abs_tol=1e-9)


def test_cosine_vecteurs_sens_oppose():
    # (1,0) et (-1,0) -> angle 180° -> S = -1
    resultat = cosine_similarity([1, 0], [-1, 0])
    assert math.isclose(resultat, -1.0, abs_tol=1e-9)


def test_cosine_vecteur_nul():
    # Un vecteur nul n'a pas de direction : convention S = 0
    resultat = cosine_similarity([0, 0], [1, 1])
    assert resultat == 0.0


def test_cosine_angle_45_degres():
    # (1,0) et (1,1) : angle de 45°, cos(45°) = 1/sqrt(2) ≈ 0.7071
    resultat = cosine_similarity([1, 0], [1, 1])
    assert math.isclose(resultat, 1 / math.sqrt(2), abs_tol=1e-9)


def test_cosine_insensible_a_la_norme():
    # (1,1) et (10,10) pointent dans la même direction, seule la norme
    # diffère -> la similarité cosinus doit rester 1 (contrairement à
    # une distance euclidienne, qui elle serait très différente)
    resultat = cosine_similarity([1, 1], [10, 10])
    assert math.isclose(resultat, 1.0, abs_tol=1e-9)


def test_cosine_est_symetrique():
    # S(x, y) doit être égal à S(y, x)
    a, b = [1, 2, 3], [4, -1, 2]
    assert math.isclose(cosine_similarity(a, b), cosine_similarity(b, a))


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK  - {t.__name__}")
    print(f"\n{len(tests)} tests passés avec succès.")
