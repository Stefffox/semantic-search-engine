# Roadmap - Projet Embeddings & Recherche Sémantique (BUT3)

Binôme : à compléter
Deadline : à compléter

## Objectif

Construire un moteur de recherche sémantique en Python : transformer des textes en vecteurs (embeddings), comparer ces vecteurs par similarité cosinus, et retrouver les documents les plus pertinents pour une requête donnée.

## Répartition proposée

- Personne A : embedding + similarité cosinus (étapes 1 et 2)
- Personne B : squelette d'indexation (sauvegarde/rechargement de l'index, étape 3)
- Ensemble : moteur de recherche, tests, rapport (étapes 4, 5, 7)

À ajuster selon vos préférences respectives.

## Étape 1 : représenter un texte par un vecteur

- [ ] Lire le document fourni sur les embeddings
- [ ] Mettre en place un modèle d'embedding (Ollama + nomic-embed-text, ou autre)
- [ ] Écrire une fonction `texte -> vecteur`
- [ ] Vérifier la dimension des vecteurs obtenus
- [ ] Vérifier que deux phrases différentes donnent des vecteurs différents
- [ ] Observer le comportement sur deux phrases de sens proche

## Étape 2 : mesurer la proximité entre deux textes

- [ ] Implémenter soi-même la similarité cosinus (pas de bibliothèque toute faite)
- [ ] Tester la fonction sur des vecteurs simples calculables à la main
- [ ] Calculer les embeddings de plusieurs paires de phrases :
  - [ ] deux phrases de sens proche
  - [ ] deux phrases avec des mots communs mais des sujets différents
  - [ ] deux phrases sans rapport
- [ ] Comparer et commenter les scores obtenus

## Étape 3 : construire l'indexation

- [ ] Constituer un corpus de plusieurs dizaines de petits documents
- [ ] Pour chaque document : lire le contenu, calculer l'embedding, stocker texte + id + vecteur
- [ ] Sauvegarder l'index sur disque
- [ ] Recharger l'index sans tout recalculer

## Étape 4 : construire le moteur de recherche

- [ ] Calculer l'embedding d'une question utilisateur
- [ ] Calculer la similarité avec chaque document indexé
- [ ] Classer les documents par score décroissant
- [ ] Afficher les k documents les plus pertinents avec leur score

## Étape 5 : tester la qualité de la recherche

- [ ] Requête reprenant exactement les mots d'un document
- [ ] Reformulation avec synonymes
- [ ] Question exprimant la même idée avec des mots très différents
- [ ] Requête sans rapport avec le corpus
- [ ] Requête ambiguë (plusieurs documents possibles)
- [ ] Comparaison avec une recherche par mots-clés basique
- [ ] Conclusion : dans quels cas les embeddings apportent un vrai plus vs le lexical ?

## Étape 6 (bonus, si le temps le permet)

- [ ] FAISS pour une recherche vectorielle efficace
- [ ] Découpage des documents longs en chunks
- [ ] Visualisation par réduction de dimension (voir si les thèmes forment des groupes)
- [ ] Mini-RAG : passer les documents trouvés à un LLM pour générer une réponse

## Étape 7 : rapport

- [ ] Principe d'un embedding
- [ ] Similarité cosinus et son interprétation
- [ ] Architecture du moteur de recherche
- [ ] Choix d'implémentation
- [ ] Expériences réalisées
- [ ] Exemples de recherches réussies et moins réussies
- [ ] Analyse des limites de la méthode

## Livrables à rendre

- [ ] Fonction d'embedding
- [ ] Implémentation maison de la similarité cosinus
- [ ] Index sauvegardable / rechargeable
- [ ] Moteur retournant les k documents les plus proches
- [ ] Interface simple pour faire plusieurs recherches
- [ ] Tests de pertinence
- [ ] Comparaison avec une recherche par mots-clés
- [ ] Rapport complet
