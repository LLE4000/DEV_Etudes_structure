# La taille du code du site, module par module

*Demande du propriétaire (13/11/2026) : « un petit tableau qui dit combien de lignes de code pour le site, combien par
module ; après chaque lot, on voit la réduction. S'il y a moins de code et que ça donne la même chose, en plus juste,
c'est optimisé. » L'unification commence par la poutre, les dalles et les prédalles.*

**L'outil** : `/usr/bin/python3.12 outils/taille_du_code.py` (le tableau d'aujourd'hui), `--comparer` (aujourd'hui face
à la première mesure, avec l'écart), `--enregistrer "étiquette"` (ajoute la mesure, datée, à
`docs/site/taille_du_code.json` — on en ajoute après chaque lot, on n'en réécrit pas). Son test :
`tests/test_taille_du_code.py`.

**Ce qui est compté** : une *ligne de code* porte une instruction — ni vide, ni commentaire, ni docstring ; « avec
commentaires » compte toutes les lignes. Le *site* est ce que la page charge : le Python du moteur embarqué (la liste
que la construction écrit, `site/web/moteur/manifeste.json`), les générateurs de la note PDF, le JavaScript et les
feuilles de style, la construction des pages. Chaque fichier du site tombe dans UN groupe ; un fichier sans règle fait
échouer le comptage. Hors du site, à part : l'ancien logiciel Streamlit et les tests.

## Mesure du 13/11/2026 — avant l'unification (commit c020cee)

| Module | Lignes de code | Avec commentaires | Fichiers |
|---|---:|---:|---:|
| Poutres béton armé | 3 618 | 5 215 | 13 |
| Dalles et prédalles | 5 019 | 7 647 | 20 |
| Charges sur plancher | 4 608 | 6 120 | 20 |
| Fondations de grues | 5 307 | 6 990 | 19 |
| Taux d'armature | 2 263 | 3 176 | 9 |
| Enrobage | 1 076 | 1 526 | 7 |
| Maturité du béton | 3 007 | 4 214 | 10 |
| Tableau des armatures | 903 | 1 282 | 6 |
| Données béton | 1 151 | 1 617 | 10 |
| Choix de profilé | 1 453 | 1 999 | 7 |
| Catalogue des profilés | 2 085 | 2 918 | 8 |
| Cornières | 2 902 | 4 228 | 8 |
| Flambement | 1 311 | 1 807 | 8 |
| Assemblages métalliques | 18 397 | 25 054 | 97 |
| Soudures | 3 771 | 4 934 | 13 |
| Garde-corps | 3 032 | 3 983 | 11 |
| Poutres bois | 1 828 | 2 712 | 9 |
| Rigidité du sol | 3 228 | 4 404 | 8 |
| Bibliothèques (béton, acier, autres) | 5 648 | 7 307 | 23 |
| Code commun (écran, note PDF, construction) | 14 926 | 20 126 | 68 |
| **Total du site** | **85 533** | **117 259** | **374** |
| Ancien logiciel Streamlit (hors site) | 23 918 | 32 653 | 71 |
| Tests (hors site) | 66 563 | 81 528 | 146 |

À lire avec l'état des lieux UNI0 (`SUITE_DU_CHANTIER.md`) : la dalle et la prédalle ont un seul écran, mais deux
fichiers de site qui partagent 40 fonctions de même nom (dont 15 copies identiques) ; la poutre a sa propre copie de la
lecture des matériaux ; `beton/dalle/fichier.py` (la lecture des fichiers Streamlit des dalles) est encore expédié
alors que l'écran ne l'appelle plus, et `beton/predalle/fichier.py` ne sert plus qu'à un calcul de y_G — à examiner
au lot 2 (rangement).

## Mesure du 13/11/2026 — après le lot 2a (commit c401951) : l'écran de l'ancien module Prédalles retiré

Résultats identiques (les cas de référence des prédalles, figés avant, passent après). L'écart se lit depuis la
mesure de départ.

| Module | Lignes de code | Avec commentaires | Fichiers | Écart depuis le 2026-11-13 (avant l'unification (poutre, dalle, prédalle)) |
|---|---:|---:|---:|---:|
| Poutres béton armé | 3 618 | 5 215 | 13 | — |
| Dalles et prédalles | 4 431 | 6 853 | 20 | −588 |
| Charges sur plancher | 4 608 | 6 120 | 20 | — |
| Fondations de grues | 5 307 | 6 990 | 19 | — |
| Taux d'armature | 2 263 | 3 176 | 9 | — |
| Enrobage | 1 076 | 1 526 | 7 | — |
| Maturité du béton | 3 007 | 4 214 | 10 | — |
| Tableau des armatures | 903 | 1 282 | 6 | — |
| Données béton | 1 151 | 1 617 | 10 | — |
| Choix de profilé | 1 453 | 1 999 | 7 | — |
| Catalogue des profilés | 2 085 | 2 918 | 8 | — |
| Cornières | 2 902 | 4 228 | 8 | — |
| Flambement | 1 311 | 1 807 | 8 | — |
| Assemblages métalliques | 18 397 | 25 054 | 97 | — |
| Soudures | 3 771 | 4 934 | 13 | — |
| Garde-corps | 3 032 | 3 983 | 11 | — |
| Poutres bois | 1 828 | 2 712 | 9 | — |
| Rigidité du sol | 3 228 | 4 404 | 8 | — |
| Bibliothèques (béton, acier, autres) | 5 648 | 7 307 | 23 | — |
| Code commun (écran, note PDF, construction) | 14 912 | 20 115 | 68 | −14 |
| **Total du site** | **84 931** | **116 454** | **374** | **−602** |
| Ancien logiciel Streamlit (hors site) | 23 918 | 32 653 | 71 | — |
| Tests (hors site) | 65 954 | 80 810 | 146 | −609 |

## Mesure du 13/11/2026 — après le lot 3 (commit 85a6a83) : la règle de position des barres

La règle de position est écrite une seule fois (`beton/position.py`, 41 lignes de code, rangé dans le code commun) pour
la poutre, la dalle pleine et la prédalle. Depuis le lot 2a, le site gagne **+161** lignes (dalles et prédalles +30,
poutre +70, code commun +61) : ce que la règle ajoute — le détail de chaque axe dans la fenêtre de calcul, la position
réelle et l'axe de calcul de chaque couche, la relecture des anciennes notes, le cran des flèches, l'import Streamlit de
la poutre corrigé — pèse plus que ce qu'elle retire (les deux comparaisons ancienne/nouvelle règle, `chiffres_changes`,
l'ancienne fonction d'axe). La baisse attendue viendra du rangement des moteurs (une lecture des matériaux, un dessin
commun dalle/prédalle), reporté à un lot suivant (`SUITE_DU_CHANTIER.md`). L'écart se lit depuis la mesure de départ.

| Module | Lignes de code | Avec commentaires | Fichiers | Écart depuis le 2026-11-13 (avant l'unification (poutre, dalle, prédalle)) |
|---|---:|---:|---:|---:|
| Poutres béton armé | 3 688 | 5 324 | 13 | +70 |
| Dalles et prédalles | 4 461 | 6 850 | 20 | −558 |
| Charges sur plancher | 4 608 | 6 120 | 20 | — |
| Fondations de grues | 5 307 | 6 990 | 19 | — |
| Taux d'armature | 2 263 | 3 176 | 9 | — |
| Enrobage | 1 076 | 1 526 | 7 | — |
| Maturité du béton | 3 007 | 4 214 | 10 | — |
| Tableau des armatures | 903 | 1 282 | 6 | — |
| Données béton | 1 151 | 1 617 | 10 | — |
| Choix de profilé | 1 453 | 1 999 | 7 | — |
| Catalogue des profilés | 2 085 | 2 918 | 8 | — |
| Cornières | 2 902 | 4 228 | 8 | — |
| Flambement | 1 311 | 1 807 | 8 | — |
| Assemblages métalliques | 18 397 | 25 054 | 97 | — |
| Soudures | 3 771 | 4 934 | 13 | — |
| Garde-corps | 3 032 | 3 983 | 11 | — |
| Poutres bois | 1 828 | 2 712 | 9 | — |
| Rigidité du sol | 3 228 | 4 404 | 8 | — |
| Bibliothèques (béton, acier, autres) | 5 648 | 7 308 | 23 | — |
| Code commun (écran, note PDF, construction) | 14 973 | 20 272 | 69 | +47 |
| **Total du site** | **85 092** | **116 718** | **375** | **−441** |
| Ancien logiciel Streamlit (hors site) | 23 918 | 32 653 | 71 | — |
| Tests (hors site) | 66 171 | 81 161 | 147 | −392 |
