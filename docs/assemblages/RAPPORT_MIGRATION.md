# Rapport de migration — Assemblages métalliques, doubles cornières d'âme

**Dépôt** `LLE4000/DEV_Etudes_structure` · branche `claude/dalle-module-creation-sjmitr`
· du commit de base `eb1b1d9` à la phase 6 · 21/09/2026.

---

## 1. Ce qui a été fait

L'outil HTML autonome « Assemblages métalliques – doubles cornières d'âme »
est devenu un module de l'application : `Accueil → Acier → Assemblages
métalliques → Poutre – Poutre → Doubles cornières d'âme`.

| Phase | Livrable | Commit |
|---|---|---|
| 0 | `PLAN_INTEGRATION_ASSEMBLAGES.md`, ligne de base (320 contrôles), témoins versionnés, Q1 harnais, Q2 Python 3.12 | `25953d7` |
| 1 | `INVENTAIRE_HTML.md` — matrice de traçabilité, 19 catégories | `5e0a5fe` |
| 2 | moteur Python, bibliothèques, parité 100 % | `5fee855` |
| 3 | schémas (élévation, plan, 3 niveaux, 31 cotes), parité des dessins | `1dd11fb`, `d1fa44e` |
| 4 | interface, registre, page de sélection, composant cliquable, export texte | `655ca4d` |
| 5 | note de calcul PDF (`ndc_pdf`, extension rétro-compatible) | `cd08ec1` |
| 6 | non-régression, recette, ce rapport, `ECARTS_ET_CORRECTIONS.md` | (ce commit) |

Arbitrages du bureau appliqués : Q1 harnais (`filtered_state` → `to_dict()`,
4 lignes) ; Q2 Python 3.12 (`.devcontainer`) ; Q3 vignette `Logo_corniere.png`
sur l'accueil, vignette SVG propre par assemblage dans le registre ;
Q4 palette `01_encre`, même identité graphique que les modules béton.

## 2. Architecture livrée

```
acier/                                   paquet métier, sans Streamlit
  js.py · formats.py · bibliotheques.py · resultats.py · resistances.py
  donnees/bibliotheques.json             source unique (extrait du corrigé)
  reference/                             témoins : HTML de référence, corrigé JSON
  registre.py                            famille → assemblages (id, titre, pitch, vignette, état, entrée)
  composants/svg_cliquable/              composant Streamlit bidirectionnel, sans compilation
  assemblages/ui_commun.py               briques d'écran de la charte béton
  assemblages/poutre_poutre/doubles_cornieres/
    entrees.py    84 clés, 14 groupes, conditions, ◆ prédim
    moteur.py     compute · bolt_table · predim · apply_solution (transcription de DC)
    benchmark.py  BENCH (4 × 37) · VALID (2 × 22) · NOCOVER · run_bench
    parite.py     instantané et comparateur
    schemas.py    Feuille · elevation · plan · SVG (+ style résolu en attributs)
    texte.py      export texte essentiel / complet
    rapport.py    note PDF (peintre ReportLab de la scène + planches)
    interface.py · ecran_saisie.py · ecran_resultats.py · methode.py
modules/assemblages.py                   page de sélection + aiguillage (clé asm_courant)
outils/oracle_double_corniere.js         oracle Node (annexe A) · oracle_dessins.js
lancer_tests.py                          lance toutes les suites, résumé, code de sortie
docs/assemblages/                        plan, inventaire, écarts, ce rapport
```

**Fichiers existants modifiés** (le strict nécessaire) :

| Fichier | Modification |
|---|---|
| `streamlit_app.py` | +1 import, +1 entrée `"Assemblages métalliques"` |
| `modules/accueil.py` | +1 carte (ligne Acier, 5 cartes par ligne) |
| `ndc_pdf/styles.py` | `planche()` : `dessin`, `titre_coupe`, `coupe_w`, `coupe_h`, `n_cols` par section ; item `("p", texte)` — sans effet sur les notes béton (étalon identique au pixel) |
| `.devcontainer/devcontainer.json` | image Python 3.12 (Q2) |
| `tests/test_dalle_bidir.py`, `test_predalle.py`, `test_sol_interface.py` | `filtered_state` → `to_dict()` (Q1, 4 lignes) |

Aucun module béton (`poutre.py`, `dalle.py`, `predalle.py`, exports) modifié.
Aucun renommage. Aucune version de dépendance changée.

Ajouter un assemblage = un dossier sous `acier/assemblages/<famille>/` + une
entrée dans `acier/registre.py`. La page de sélection ne change pas.

## 3. Matrice de traçabilité finale (§4 du cahier des charges)

| Catégorie | Compte HTML | Livré | Où |
|---|---|---|---|
| Données d'entrée | 84 | 84 ✔ | `entrees.py`, colonne « Données » (14 groupes, conditions d'affichage) |
| Paramètres géométriques | tous (`R.*`) | ✔ | moteur ; tableau « Paramètres retenus » |
| Matériaux | 6 aciers + Personnalisé | ✔ | `bibliotheques.json`, sélecteurs |
| Boulons | 8 boulons, 4 classes, résistances, efforts par boulon | ✔ | moteur, `resistances.py`, tableau par boulon |
| Cornières | 18 + Personnalisé, orientation, ailes / épaisseurs | ✔ | bibliothèque, moteur |
| Profilés | 90 + Personnalisé, aire recalculée | ✔ | bibliothèque, croisement catalogue |
| Efforts | VEd, NEd, HEd, MEd et dérivés | ✔ | moteur |
| Hypothèses | 6 lignes + onglet Méthode + notes de champs + hypothèse des cordons | ✔ | `texte.py`, `methode.py`, aides des champs, note PDF |
| Vérifications | 38 (EC 17 · EC+COMP 8 · COMP 12 · INT 1) | 38 ✔ | onglet Vérifications (barre de taux, statut, détail), « sans objet » listées |
| Formules | 38 formules générales + formules de calcul | ✔ | détail dépliable, export texte, planche « développement » |
| Résultats intermédiaires | 190 scalaires, chaînes `vals` | ✔ | parité ≤ 1e-9 ; égalité stricte des chaînes |
| Résultats finaux | statut, taux maximal, gouvernante, benchmark, prédim | ✔ | bandeau permanent, synthèse |
| Avertissements géométriques | 11 lignes Tableau 3.3, `rec_L`, 4 informatives, `val_N` | ✔ | tableau des pinces (NON OK cliquable), alertes |
| Configurations impossibles | 15 bloquantes + `dist_…` | ✔ | alertes cliquables : explication chiffrée, champs en cause modifiables, cotes/éléments rouges |
| Dessins | élévation + plan, 8 éléments nommés, cartouche | ✔ | `schemas.py` (parité 372 vues), écran et PDF |
| Cotations | 31 cotes (18 modifiables / 13 calculées), 3 niveaux, verrouillage prédim | ✔ | idem |
| Interactions avec le dessin | clic → saisie → recalcul ; alerte → rouge ; niveau | ✔ | composant `svg_cliquable` **et** panneau « Cotes » ; « Localiser » |
| Export texte | essentiel / complet, .txt, copie | ✔ | onglet Rapport et export (1 300 lignes identiques au corrigé) |
| Autres fonctions | .json (enregistrer / charger, format du module **et** de l'outil HTML), défauts, Benchmark, Méthode, Prédim (solutions cliquables, tableau complet « Appliquer »), messages | ✔ | barre d'outils, onglets |

Remplacé : « Imprimer (A4) » du HTML → note PDF `ndc_pdf` (bouton
« 📄 Générer PDF », même charte que les notes béton).

## 4. Parité prouvée

| Test | Portée | Résultat |
|---|---|---|
| `tests/test_assemblages_parite.py` | 31 cas du corrigé : scalaires, vérifications, alertes, Tableau 3.3, efforts par boulon, prédimensionnement, statut, gouvernante | **23 977 grandeurs, 0 écart** (≤ 1e-9 ; égalité stricte des booléens, identifiants, chaînes) |
| idem | benchmark : 37 valeurs publiées + 22 manuelles, statuts | identiques, VALIDÉ |
| `tests/test_assemblages_oracle.py` | ≥ 300 cas aléatoires contre le moteur du HTML sous Node (v22) | **400 cas, 692 377 grandeurs, 0 écart** ; rejoué une fois à 1 500 cas : 2 583 647 grandeurs, 0 écart |
| `tests/test_assemblages_schemas.py` | dessins contre `DCDraw` sous Node : 31 cas × 3 niveaux, chaque alerte du corrigé, prédim, 60 cas aléatoires | **372 vues, 0 divergence** (éléments, classes, coordonnées ± 1e-6, textes, clés) |
| `tests/test_assemblages_texte.py` | 5 exports × 2 versions | **1 300 lignes identiques** |

Écart maximal observé sur les nombres : 0 (égalité exacte sur les 31 cas ;
seuls des écarts ≤ 1e-12 apparaissent sur les cas aléatoires, dus à l'ordre
des opérations flottantes).

## 5. Tests — état final

```
python3 lancer_tests.py
```

| Suite | Contrôles | Origine |
|---|---|---|
| `test_dalle_bidir.py` | 56 | existante (inchangée hors Q1) |
| `test_predalle.py` | 32 | existante |
| `test_sol_interface.py` | 40 | existante |
| `test_export_pdf_ndc.py` | 39 | existante — **étalon béton identique au pixel après l'extension de `ndc_pdf`** |
| `test_poutre_audit.py` | 17 | existante |
| `test_sol_import.py` | 50 | existante |
| `test_sol_theorie.py` | 86 | existante |
| `test_assemblages_bibliotheques.py` | 20 | nouvelle |
| `test_assemblages_parite.py` | 44 | nouvelle |
| `test_assemblages_oracle.py` | 3 | nouvelle (ignorée proprement sans `node`) |
| `test_assemblages_schemas.py` | 9 | nouvelle |
| `test_assemblages_texte.py` | 12 | nouvelle |
| `test_assemblages_interface.py` | 34 | nouvelle (AppTest sur l'application réelle) |
| `test_assemblages_rapport.py` | 29 | nouvelle |
| `test_fumee_pages.py` | 18 | nouvelle — les 16 pages + l'écran de l'assemblage |
| **Total** | **489 OK, 0 échec, 15/15 suites vertes** | ligne de base de la phase 0 : 320, tous intacts |

Recette complémentaire, hors dépôt : contrôle du composant cliquable dans un
vrai navigateur (Chromium/Playwright) sur l'application lancée — clic sur la
cote `Lc`, saisie, Entrée → formulaire et dessin synchronisés, calcul relancé ;
saisie dans le formulaire → dessin ; alerte « Localiser » → cotes et cornière
en rouge, explication chiffrée, champ marqué 🔴 ; correction par la chip →
blocage levé.

## 6. Références non vérifiées, corrections

Voir `ECARTS_ET_CORRECTIONS.md` : aucune erreur manifeste, **aucune correction
appliquée** ; références citées de mémoire listées comme non vérifiées ;
une proposition (P1, longueur efficace des cordons `L − 2a`) à trancher.

## 7. Dépendances

**Aucune dépendance ajoutée** à `requirements.txt`. Le moteur n'utilise que la
bibliothèque standard ; les schémas sont du SVG écrit à la main ; le PDF passe
par `ndc_pdf` / ReportLab déjà présent ; le composant est du HTML + JS servi
par Streamlit.

Outils de développement, facultatifs : `node` (deux tests d'oracle,
ignorés proprement s'il est absent) ; `pymupdf` (déjà dans `requirements.txt`,
utilisé par les tests de PDF).

Environnement : **Python 3.12** (le dépôt l'exigeait déjà — f-strings avec
antislash — et `.devcontainer` est aligné). La version de Python servie par la
plateforme de déploiement se règle dans ses paramètres avancés : **à vérifier
par le bureau** (non accessible depuis cette session).

## 8. Commandes

```
# application
streamlit run streamlit_app.py

# toutes les suites (une par une, comme des scripts) + résumé
python3 lancer_tests.py
python3 lancer_tests.py assemblages          # seulement les nouvelles

# une suite
python3 tests/test_assemblages_parite.py
python3 tests/test_assemblages_oracle.py 1500   # nombre de cas aléatoires

# régénérer un corrigé par l'oracle Node (si le HTML de référence change)
node outils/oracle_double_corniere.js acier/reference/assemblage_double_corniere_EC3.html cas.json > resultats.json
```

## 9. Limites connues

- **Note PDF** : la note du bouton « 📄 Générer PDF » tient sur **une page
  A4 paysage** (refonte du 21/09/2026, `note.py`) ; les formules qui n'y
  tiennent pas renvoient au **rapport détaillé** (3 pages : garde,
  synthèse, développement — `rapport.py`, onglet Note). L'export texte
  complet reste le document exhaustif.
- `use_container_width` (boutons) : le module suit la convention du dépôt ;
  Streamlit annonce son retrait (remplacement par `width=`). Tous les
  modules seront concernés en même temps.
- Le composant cliquable repose sur le protocole des composants Streamlit
  (v1, `declare_component(path=…)`). Si un déploiement le bloquait, la case
  « Dessin interactif » des **Paramètres avancés** le désactive : le schéma
  reste affiché et le panneau « Cotes » assure l'édition.
- Vignette d'accueil : `Logo_corniere.png`, servie par l'URL distante des
  images d'accueil (dépôt `LLE4000/Etudes-structure`) — à remplacer là-bas
  pour une image propre.
- Carte « Platine d'about » : entrée **à venir** du registre (grisée, non
  cliquable) pour montrer le regroupement par famille ; à supprimer ou à
  remplir.
- Les tests d'interface et de fumée durent quelques secondes chacun ;
  `test_sol_interface.py` (existant) reste le plus long (~11 s).

## 9 bis. Refonte UX du 21/09/2026 (après livraison)

Conception : `REFONTE_UX.md` (diagnostic, principes, architecture, ce qui a
été réalisé et les écarts assumés). En résumé :

- **une seule entrée par paramètre** : la géométrie (cotes, rangées, files,
  cordons) se modifie sur le dessin — cotes cliquables, poignées + / −,
  fenêtre de groupe — et seulement là ; le reste dans une carte compacte
  (mode et fixations, profilés, cornières, boulons, efforts ; paramètres
  avancés et identification repliés) ;
- **notation Eurocode** partout à l'affichage (hc, c, dc,sup / dc,inf, Δz,
  e1 / p1 / e2 / p2 du Tableau 3.3) — `notation.ec()` est une vue, le
  moteur et l'export texte ne changent pas ;
- **statut sur une ligne**, taux par élément, alertes courtes chiffrées ;
- **vérifications regroupées par élément** (boulons, cornières en matrice
  aile A · aile B · max, poutre portée, poutre porteuse, cordons), formules
  en deux lignes avec **substitution numérique testée** (2 284 réévaluées
  sur les 31 cas, 0 écart) ;
- **note d'une page** A4 paysage (31 cas, 0 débordement, corps ≥ 6,0 pt) ;
  rapport détaillé conservé ;
- **régression calcul : 0 différence** (`tests/test_assemblages_regression.py`,
  23 977 grandeurs et 5 exports texte contre le corrigé figé avant refonte).

Suites après refonte : `test_assemblages_interface.py` 40, `_schemas.py` 18,
`_formules.py` (nouvelle) 21, `_note.py` (nouvelle) 22,
`_regression.py` (nouvelle) 2 — **545 OK, 0 échec, 18/18 suites vertes** ;
étalon béton de `ndc_pdf` toujours identique au pixel.

## 10. Critères d'acceptation

- [x] Les modules existants fonctionnent comme avant ; tests d'origine verts (320/320).
- [x] `Acier → Assemblages métalliques → Poutre – Poutre → Doubles cornières d'âme` accessible ; registre prêt.
- [x] Parité : 31 cas + benchmark + 400 cas aléatoires (Node), écart relatif ≤ 1e-9.
- [x] 84 entrées, 38 vérifications, toutes les alertes, Tableau 3.3, deux modes, solutions de prédimensionnement cliquables, deux dessins à trois niveaux, mise en évidence des erreurs, édition des cotes (composant **et** panneau), export texte, enregistrement / chargement, onglets Benchmark et Méthode.
- [x] Visuel dans la charte de l'application ; note PDF A4 paysage dans le style des notes béton.
- [x] Moteur sans import d'interface ; architecture prête pour les prochains assemblages.
- [x] Aucune correction silencieuse ; `ECARTS_ET_CORRECTIONS.md` et ce rapport livrés.
