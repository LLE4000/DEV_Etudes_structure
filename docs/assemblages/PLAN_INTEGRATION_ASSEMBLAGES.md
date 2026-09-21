# Plan d'intégration — Assemblages métalliques (phase 0, lecture seule)

**Dépôt** : `LLE4000/DEV_Etudes_structure` · **HEAD** `eb1b1d9` sur `main` et sur
`claude/dalle-module-creation-sjmitr` · arbre de travail propre.
**Date** : 21/09/2026. **Rien n'a été écrit dans le dépôt.** Ce document est le
livrable de la phase 0 ; il attend votre validation avant la phase 1.

---

## 0. Ce que j'ai vérifié sur les trois fichiers fournis

Avant de proposer quoi que ce soit, j'ai contrôlé que les fichiers sont
cohérents entre eux — c'est la fondation de toute la migration.

| Contrôle | Méthode | Résultat |
|---|---|---|
| **Le JSON correspond-il au HTML ?** | oracle Node de l'annexe A exécuté sur les **31 cas**, comparaison grandeur par grandeur au JSON | **23 977 grandeurs comparées, 0 écart** |
| Cas par défaut seul | idem | 1 563 grandeurs, 0 écart |
| **Node est-il disponible ?** | `node -v` | **v22.22.2** → le test aléatoire ≥ 300 cas est faisable |
| **Le moteur tourne-t-il hors navigateur ?** | `vm.runInThisContext` du 1er `<script>` | oui, `DC` s'expose, `module.exports` présent |
| **Bibliothèque de profilés saisie de mémoire ?** | croisement des 90 profilés HTML avec `profiles_test.json` du dépôt (48 profilés) | **31 profilés communs, 0 divergence** sur `h, b, tw, tf, r` |
| Aires des profilés | `A = 2·b·tf + (h − 2tf)·tw + (4 − π)·r²` vs `A` du catalogue | écart **max 0,57 %**, médian 0,05 % — compatible avec l'arrondi du catalogue à 0,1 cm² |
| Catalogue des vérifications | comptage par groupe et par nature | **38** : Boulons 6, Cornières 13, Poutre sec. 10, Poutre princ. 3, Soudures 6 ; natures EC 17, EC+COMP 8, COMP 12, INT 1 ; 18 essentielles — **conforme au §4 du cahier des charges** |
| Alertes | dépouillement des 31 cas | **23 identifiants** rencontrés (19 bloquants dont 4 `dist_…`, 4 informatifs), chacun avec message, explication chiffrée, champs, cotes, éléments |
| Cotes du dessin | `id:` du 2ᵉ `<script>` | **31 cotes**, exactement celles listées au §4 ; **18** portent une clé d'entrée (modifiables), 13 sont calculées |

**Conséquence** : la parité est un objectif mesurable et l'oracle Node est
disponible. C'est la meilleure configuration possible pour ce genre de portage.

Contrôles de vraisemblance sur les bibliothèques (aucune modification) :
classes 4.6 / 5.6 / 8.8 / 10.9 avec `fyb/fub` = 240/400, 300/500, 640/800,
900/1000 et `αv` = 0,6 / 0,6 / 0,6 / 0,5 — conformes aux tableaux 3.1 et 3.4 de
l'EN 1993-1-8 ; `As` des boulons M12→M36 = 84,3 / 157 / 245 / 303 / 353 / 459 /
561 / 817 mm² — valeurs ISO usuelles.

---

## 1. La navigation actuelle

**Un seul point d'entrée, un dictionnaire, pas de multipage Streamlit.**

`streamlit_app.py` (64 lignes) :

1. `st.set_page_config(page_title="Études Structure", layout="wide", initial_sidebar_state="collapsed")` ;
2. lit `?page=…` dans `st.query_params` et l'écrit dans `st.session_state.page` ;
3. si `st.session_state.retour_accueil_demande` est vrai → retour à `"Accueil"` ;
4. `pages = {"Accueil": accueil.show, "Poutre": poutre.show, …}` — **15 entrées**,
   la clé est le libellé affiché ;
5. `pages.get(st.session_state.page, accueil.show)()`.

**Enregistrer une page = deux lignes** : l'import dans le tuple `from modules import (…)`
et l'entrée dans `pages`.

`modules/accueil.py` (81 lignes) : trois sections — **Béton** (`#FF6F61`), **Acier**
(`#FFA500`), **Autres** (`#6C63FF`) — rendues par `render_section(titre_html, tools, cols_per_row)`.
Chaque outil est une carte HTML `<a href="?page={page}" target="_self">` avec une image.

> **Point à connaître** : `base_url` pointe sur
> `https://raw.githubusercontent.com/LLE4000/Etudes-structure/main/assets` —
> **un autre dépôt**. Les vignettes d'accueil ne viennent donc pas du dossier
> `assets/` local. Ajouter une image locale ne suffirait pas à l'afficher.
> → Pour la carte d'accueil « Assemblages métalliques » je réutilise
> `Logo_corniere.png`, déjà présent à cette URL. Risque nul, aucun fichier binaire
> à publier ailleurs. Vous pourrez remplacer l'image quand vous voudrez.

> **Défaut existant, hors périmètre, signalé pour mémoire** : le bouton
> « 🏠 Accueil » des modules écrit `retour_accueil_demande` puis relance, mais
> `?page=…` est réappliqué à chaque relance en tête de `streamlit_app.py` et
> écrase le retour. Je n'y touche pas dans cette migration ; le nouveau module
> reproduira le comportement des autres modules, ni mieux ni moins bien.

---

## 2. L'architecture des modules béton

**Tout tient dans un fichier par module**, dans `modules/` :

| Module | Lignes | Rôle |
|---|---|---|
| `poutre.py` | 2 731 | écran Poutre BA + calcul |
| `dalle.py` | 1 886 | écran Dalle BA bidirectionnelle + calcul |
| `predalle.py` | 1 932 | écran Prédalle (fork contrôlé de `dalle.py`) |
| `export_pdf.py` | 1 633 | note Poutre (miroir du moteur d'écran) |
| `export_pdf_dalle.py` | 850 | note Dalle / Prédalle (miroir du moteur d'écran) |
| `treillis.py` | 81 | base des treillis courants |

**Il n'y a pas de couche « moteur » séparée.** Le calcul est écrit dans les
fonctions de l'écran (`_compute_section`, `_layers_geometry`, `_shear_lines`…)
et **ré-écrit** dans le module d'export (`_compute_section` y existe aussi).
Les deux copies sont tenues synchrones à la main, sous le contrôle des tests.

C'est important pour la suite : **le module Assemblages ne doit pas reproduire
ce schéma.** Le cahier des charges impose un moteur séparé (règle 7) et c'est
aussi ce que l'audit du 21/09 recommandait pour le béton. Le nouveau module
sera donc le premier du dépôt à avoir un moteur pur, testé hors interface — un
modèle pour une future extraction côté béton, sans rien y changer aujourd'hui.

**Organisation interne d'un module d'écran** (identique pour les trois) :

1. en-tête de version commenté ;
2. constantes de style (`C_COULEURS = {"ok": "#e6ffe6", "warn": "#fffbe6", "nok": "#ffe6e6"}`,
   `C_ICONES = {"ok": "✅", "warn": "⚠️", "nok": "❌"}`) ;
3. **clés de session préfixées** : `KD(base, id) → f"dal{id}_{base}"`,
   `KS(base, id, sec) → f"dal{id}_sec{sec}_{base}"` ; espaces de noms
   volontairement disjoints (`b…` Poutre, `dal…` Dalle, `pre…` Prédalle) ;
4. `PERSISTED_GLOBAL_KEYS` (partagées : `units_len`, `units_as`, `gamma_s`,
   `jeu_enrobage_cm`, `nom_projet`, `partie`, `date`, `indice`, `chk_infos_projet`) ;
5. deux garde-fous appelés en tête de `show()` :
   `_pin_persistent_state()` (ré-affecte chaque clé persistante pour empêcher
   Streamlit de nettoyer les widgets non rendus) et `_sync_float_raw_keys()`
   (les saisies décimales FR vivent en `clé_raw` texte et sont converties en
   `clé` numérique **avant tout calcul**) ;
6. `_reset_module()` qui ne supprime **que** les clés du module ;
7. `_build_save_payload()` / `_load_from_payload()` pour l'enregistrement JSON,
   avec migration de format idempotente ;
8. `show()`.

**Tests** : `tests/` contient 7 fichiers ; ce sont des **scripts** qui
s'exécutent à l'import et impriment un compte-rendu, pas des modules pytest.
Ils utilisent `streamlit.testing.v1.AppTest` pour piloter l'écran réel.

---

## 3. Les composants d'interface réutilisables

Tout est repris tel quel par le nouveau module — **aucune seconde charte**.

| Élément | Où | Forme exacte |
|---|---|---|
| Titre de page | `show()` | `st.columns([8, 1.6, 0.55])` → `## Titre`, « Version x.y » en gris à droite, bouton `❔` |
| Barre d'outils | `show()` | `st.columns(5)` : `🏠 Accueil` · `🔄 Réinitialiser` · `💾 Enregistrer` (`st.download_button` JSON) · `📂 Ouvrir` (`st.file_uploader` replié) · `📄 Générer PDF` puis `⬇️ Télécharger le rapport PDF` |
| Corps de page | `show()` | `st.columns([2, 3])` — **saisie à gauche, résultats à droite** |
| Paramètres avancés | colonne droite | bouton `⚙️` + `st.container(border=True)` |
| Encadré de résultat | `open_bloc_left_right(left, right, etat, pct)` + `close_bloc()` | fond vert / ocre / rouge, titre à gauche, valeur et **taux en %** à droite, icône |
| Saisie décimale FR | `float_input_fr_simple(label, key, …)` | `st.text_input` sur `clé_raw`, virgule acceptée |
| Regroupement | `st.expander(titre, expanded=True)` | une dalle / une poutre = un expander ; sections à l'intérieur |
| Tableau éditable | `st.columns(COUCHE_COLS)` | largeurs relatives déclarées en constante |
| Statut | `_status_merge()`, `_status_icon_label()` | fusion `ok` / `warn` / `nok` |

La colonne **[2, 3]** tombe bien : le cahier des charges veut le dessin visible
pendant la saisie. Saisie à gauche (2), dessin + synthèse à droite (3) — c'est
la disposition de l'application, pas une invention.

---

## 4. Le système de rapports PDF

**Paquet `ndc_pdf/` — ReportLab, dessin vectoriel, aucune police système, aucun HTML.**

| Fichier | Lignes | Rôle |
|---|---|---|
| `data.py` | 810 | **le seul fichier branché sur le moteur** : construit `DOC` et `SECTIONS` |
| `styles.py` | 435 | mise en page, page de garde, `Style.planche()`, 10 palettes |
| `kit.py` | 247 | `Doc`, `Frame`, texte, `kv_rows`, `formula`, `gauge`, `chip`, `box` |
| `mathx.py` | 553 | analyseur mini-LaTeX + rendu vectoriel des formules |
| `section.py` | 864 | dessin de la coupe béton (`draw_section`, `draw_dalle`, `PALETTE_DIA`) |
| `fonts.py` | 176 | enregistrement des 25 TTF embarquées, repli glyphes |
| `build.py` | 95 | génération en lot + catalogue |
| `fonts/` | 25 TTF | OFL / GUST / Apache |
| `reference/NOTE_DE_CALCUL.pdf` | — | **étalon** : `tests/test_export_pdf_ndc.py` compare texte strict + pixels |

**La mise en page validée est bien celle que vous décriviez** (le dépôt le confirme) :

- **page de garde A4 portrait** : bandeau `panel` de 132 pt, bureau en petites
  capitales espacées, titre 26 pt, cartouche **PROJET / PARTIE / DATE / INDICE**
  (seuls les champs renseignés sont remplis), **SOMMAIRE** avec pastille d'état
  et numéro de page ;
- **une planche A4 paysage par section** : en-tête (bureau, date · indice,
  titre 14 pt, pastille d'état, matériaux), filet 1,0 pt ; **colonne de gauche
  `coupe_w = 292` pt sur fond `panel` arrondi** : titre « COUPE DE SECTION »,
  la coupe cotée (`coupe_h = 320` pt max), puis les blocs
  DIMENSIONS / MATÉRIAUX / SOLLICITATIONS en clé-valeur ; **à droite `n_cols = 2`
  colonnes de calcul en flux continu**, numéro de vérification **dans un carré
  plein**, report « (suite) » quand une vérification change de colonne ;
- marges `margin = 28` pt, corps `s_form 8,0` / `s_kv 7,8` / `s_lab 6,3` /
  `s_verd 7,3` / `s_tit 9,8`, **plancher de police `min_form = 6,0`** ;
- verdicts en encadré : `ok` vert `#2E7D46`, `att` ocre `#8A6B1F`, `ko` rouge `#9C3341` ;
- `doc.warnings` liste ce qui déborderait ; **liste vide = rien ne déborde**.

**Contrat de contenu** (`ndc_pdf/README.md`) : une section est un `dict` avec
`poutre`, `section`, `beton`, `acier`, `etat`, `coupe`, `blocs`, `verifs` ;
les items d'une vérification sont des tuples
`("f", libellé, formule)` · `("v", libellé, symbole, valeur, unité)` ·
`("t", texte)` · `("s", sous-titre)` · `("k", i)`.

**Le seul point de rigidité** : `Style.planche()` appelle `draw_section(…, sec["coupe"], …)`
en dur — la colonne de gauche dessine toujours une coupe béton.

> **Extension rétro-compatible proposée** (≈ 8 lignes dans `styles.py`, zéro
> effet sur les notes béton) : si une section porte `sec["dessin"]`, `planche()`
> appelle cet objet dessinateur ; sinon elle appelle `draw_section` comme
> aujourd'hui. Le titre du panneau vient de `sec.get("titre_coupe", "COUPE DE SECTION")`
> et `n_cols` reste réglable par section. La preuve de non-régression est
> l'étalon : `tests/test_export_pdf_ndc.py` compare le PDF au pixel près.

Les deux modules de branchement (`modules/export_pdf.py`, `modules/export_pdf_dalle.py`)
**ne seront pas touchés** ; le rapport Assemblages aura le sien.

---

## 5. Ligne de base de non-régression — **avant** toute modification

### 5.1 Ce que j'ai trouvé en ouvrant le conteneur

L'environnement d'exécution **a changé depuis la dernière séance** (le conteneur
est recréé à chaque session). Il faut le dire clairement, car cela conditionne la
recette :

| Constat | Commande | Résultat |
|---|---|---|
| Python par défaut = **3.11** | `python3 -V` | `Python 3.11.15` |
| `pytest` absent, `matplotlib` absent | `python3 -c "import pytest"` | `ModuleNotFoundError` |
| **Le dépôt ne s'importe pas sous Python 3.11** | `python3 -c "import modules.poutre"` | `SyntaxError: f-string expression part cannot include a backslash` — `ndc_pdf/data.py:225` et `modules/poutre.py:2294` |
| `.devcontainer/devcontainer.json` épingle **3.11** | `image: …python:1-3.11-bookworm` | **incohérent avec le code** |
| Sous Python **3.12**, tout s'importe | `/usr/bin/python3.12 -c "import modules.poutre"` | OK |

**Diagnostic** : les *f-strings* contenant une barre oblique inverse dans la
partie expression ne sont légales qu'à partir de **Python 3.12** (PEP 701). Le
dépôt exige donc 3.12+, alors que le conteneur de développement déclare 3.11.
**Ce n'est pas causé par la migration Assemblages** et je n'y touche pas sans
votre accord — mais c'est un défaut réel : sur une plateforme qui servirait
Python 3.11, l'application ne démarrerait pas.

### 5.2 La ligne de base, mesurée

Environnement retenu pour la recette : **Python 3.12** +
`streamlit 1.64.0`, `reportlab 5.0.1`, `matplotlib 3.11.2`, `pandas 3.0.5`,
`pymupdf`, `pytest 9.1.1`.

```
/usr/bin/python3.12 tests/test_dalle_bidir.py
/usr/bin/python3.12 tests/test_predalle.py
/usr/bin/python3.12 tests/test_sol_interface.py
/usr/bin/python3.12 tests/test_export_pdf_ndc.py
/usr/bin/python3.12 tests/test_poutre_audit.py
/usr/bin/python3.12 tests/test_sol_import.py
/usr/bin/python3.12 tests/test_sol_theorie.py
```

| Suite | Contrôles | Résultat |
|---|---|---|
| `test_dalle_bidir.py` | **56** | 0 échec *(1)* |
| `test_predalle.py` | **32** | 0 échec *(1)* |
| `test_sol_interface.py` | **40** | 0 échec *(1)* |
| `test_export_pdf_ndc.py` | **39** | 0 échec |
| `test_poutre_audit.py` | **17** | 0 échec |
| `test_sol_import.py` | **50** | 0 échec |
| `test_sol_theorie.py` | **86** | 0 échec |
| **Total** | **320** | **0 échec** |

*(1)* Ces trois suites lisent `at.session_state.filtered_state`. **Streamlit 1.64
ne l'expose plus** sur l'objet rendu par `AppTest.session_state` (les méthodes
`keys()`, `items()`, `to_dict()`, l'itération et `in` le remplacent). Sans
correctif elles s'arrêtent sur `AttributeError: filtered_state not found in
session_state` — **avant** toute modification de ma part. Je les ai rejouées
avec une cale **hors dépôt** (`sitecustomize.py` dans un répertoire temporaire
qui remet la propriété) : les 128 contrôles repassent au vert, à l'identique.

**Deux façons de rendre cette ligne de base durable** — à trancher (§7, Q1) :
- **(A)** adapter les **4 lignes** concernées du harnais
  (`tests/test_dalle_bidir.py:160`, `tests/test_predalle.py:144`,
  `tests/test_sol_interface.py:246` et `:253`) : `…filtered_state` → `…to_dict()`.
  Le harnais **suit** l'écran ; aucun module applicatif touché ;
- **(B)** épingler `streamlit` à une version antérieure dans `requirements.txt`.

Je recommande **(A)**, et **je l'ai vérifiée** : sur des copies patchées des
trois suites, exécutées hors dépôt dans un arbre miroir, **sans aucune cale**,
les comptes reviennent **56 / 32 / 40, 0 échec** — identiques à la ligne de
base. `to_dict()` rend exactement `filtered_state` en 1.64. Cette option ne fige
pas la plateforme et ne touche aucun module applicatif.

### 5.3 Deux constats de lancement, à connaître

- **`python3 -m pytest -q` ne fonctionne pas** à la racine :
  `INTERNALERROR> SystemExit: 0` — `tests/test_sol_import.py` et
  `tests/test_sol_theorie.py` appellent `sys.exit()` au niveau module. Les
  suites se lancent **une par une, comme des scripts**. La recette de la phase 6
  donnera la commande exacte, et je proposerai (sans l'imposer) un
  `lancer_tests.py` qui les enchaîne et rend un code de sortie unique.
- **Test de fumée des 15 pages** : je l'ai écrit et exécuté (hors dépôt) —
  `Accueil, Poutre, Dalle, Prédalle, Cornière, Garde-corps, Poutre bois,
  Tableau armatures, Age béton, Choix profilé, Flambement, Tableau profilés,
  Enrobage, Rigidité du sol, Taux d'armature` → **15 OK, 0 KO**, aucune
  exception. Ce script deviendra `tests/test_fumee_pages.py` en phase 6 et
  inclura la nouvelle page.

---

## 6. Architecture proposée pour `Acier → Assemblages métalliques`

### 6.1 Chemin de l'utilisateur

```
Accueil ──(carte « Assemblages métalliques », ligne Acier)──▶ page « Assemblages métalliques »
          │
          ├─ Famille « Poutre – Poutre »  ─▶ carte « Doubles cornières d'âme »  [disponible]
          │                                  carte « Platine d'about »          [à venir]
          └─ Familles suivantes                                                 [à venir]
```

La page de sélection et le module vivent sous **une seule clé de navigation**
(`"Assemblages métalliques"`) avec un sous-état interne `asm_courant`. Ainsi
`streamlit_app.py` n'est touché qu'**une fois**, aujourd'hui et pour tous les
assemblages à venir.

### 6.2 Fichiers **créés**

```
acier/                                  ← nouveau paquet métier, sans Streamlit
  __init__.py
  donnees/
    bibliotheques.json                  ← extrait du JSON de référence (source unique)
    reference_double_corniere.json      ← le corrigé fourni, versionné tel quel (témoin)
    assemblage_double_corniere_EC3.html ← le HTML fourni, versionné tel quel (témoin)
  bibliotheques.py                      ← chargement : profils, aciers, boulons, classes, cornières, ailes, épaisseurs
  resultats.py                          ← Verification, Alerte, Cote (dataclasses) — communs à la famille
  resistances.py                        ← résistances élémentaires boulons / soudures — communes à la famille
  registre.py                           ← registre déclaratif famille → assemblages (id, titre, pitch, vignette SVG, état, point d'entrée)
  assemblages/
    __init__.py
    poutre_poutre/
      __init__.py
      doubles_cornieres/
        __init__.py
        entrees.py                      ← les 84 clés : libellé, unité, type, défaut, condition d'affichage, groupe
        moteur.py                       ← transcription de DC : defaults, compute, predim, applySolution, solutionLabel
        benchmark.py                    ← BENCH, VALID, runBench (4 exemples publiés + 2 calculs manuels)
        schemas.py                      ← générateur SVG : élévation + plan, 3 niveaux, 31 cotes, cartouche, mise en rouge
        texte.py                        ← export texte (essentiel / complet)
        rapport.py                      ← alimentation de ndc_pdf (DOC + SECTIONS), dessins vectoriels
        interface.py                    ← écran Streamlit de l'assemblage
  composants/
    svg_cliquable/                      ← composant bidirectionnel générique, sans compilation
      __init__.py
      static/index.html                 ← SVG + JS minimal (protocole composant Streamlit)
modules/
  assemblages.py                        ← page Streamlit : sélection par familles + aiguillage (shim mince)
outils/
  oracle_double_corniere.js             ← oracle Node de l'annexe A
tests/
  test_assemblages_parite.py            ← 31 cas + benchmark, ≤ 1e-9
  test_assemblages_oracle.js.py         ← ≥ 300 cas aléatoires Python ↔ Node (ignoré si node absent)
  test_assemblages_bibliotheques.py     ← croisement avec profiles_test.json + contrôle d'aire
  test_assemblages_interface.py         ← AppTest : la page se charge, les 84 champs, les conditions d'affichage
  test_assemblages_exports.py           ← export texte comparé à exports_texte (5 cas × 2 versions) + PDF sans débordement
  test_fumee_pages.py                   ← les 16 pages se chargent sans exception
```

### 6.3 Fichiers **modifiés** (le strict minimum)

| Fichier | Modification | Risque |
|---|---|---|
| `streamlit_app.py` | +1 import, +1 entrée `"Assemblages métalliques": assemblages.show` | nul |
| `modules/accueil.py` | +1 carte dans `acier_tools` (image `Logo_corniere.png`) | nul |
| `ndc_pdf/styles.py` | `planche()` : `sec.get("dessin")` sinon `draw_section` ; titre de panneau paramétrable | **couvert par l'étalon pixel** |
| `requirements.txt` | rien à ajouter *(voir 6.6)* | nul |
| 4 lignes de harnais | `filtered_state` → `to_dict()` — **si vous retenez l'option (A)** du §5.2 | nul |

**Aucun module béton n'est modifié.**

### 6.4 Espace de noms de session

Préfixe **`asm_`** pour tout le module (`asm_V_Ed`, `asm_prof_S`, …), plus
`asm_ui_*` pour l'état d'écran (onglet courant, niveau de cotation, alerte
sélectionnée). Disjoint de `b…`, `dal…`, `pre…` et des clés globales partagées.
Les **noms des 84 entrées sont conservés tels quels** dans le moteur (ce sont
les clés du JSON de référence) ; le préfixe ne vit que dans `session_state`.

### 6.5 Les schémas et l'édition des cotes (§6 du cahier des charges)

**Un seul générateur**, `schemas.py`, en Python pur : il produit du SVG à partir
de l'état et du résultat du moteur, avec l'algorithme de placement du 2ᵉ
`<script>` (rangées T/B/L/R, recherche de créneau libre, libellé déporté quand
la cote est trop courte). Le **même** générateur sert l'écran et le PDF : pour
le rapport, le même modèle de dessin est rendu par les primitives ReportLab de
`kit.py` — **vectoriel**, pas d'image.

**Édition des cotes — ce que je propose, et le compromis :**

- **Cible** : un petit composant bidirectionnel **générique et sans étape de
  compilation** (`acier/composants/svg_cliquable/`), déclaré par
  `components.declare_component(name, path=…)` — l'API existe bien en
  Streamlit 1.64.0 (vérifié). Il reçoit un SVG de Python, rend cliquable tout
  élément portant `data-k`, ouvre une saisie sur place et renvoie `{clé, valeur}`.
  Réutilisable tel quel par les prochains assemblages.
- **Filet de sécurité, livré en même temps** : un **panneau « Cotes »**
  synchronisé à côté du dessin (les 18 cotes modifiables, mêmes clés, même
  verrouillage en prédimensionnement). Si le composant ne se charge pas
  (déploiement restreint, version de Streamlit différente), le module bascule
  sur le panneau — même fonction, même état, ergonomie un cran en dessous.
- **Pourquoi les deux** : nous venons de voir en §5 qu'une montée de version de
  Streamlit casse silencieusement une API. Votre priorité n°1 est « ne rien
  casser » ; un chemin de repli testé la garantit. Les deux chemins écrivent
  dans **le même état unique** — pas de seconde source de vérité.

Le parcours « message rouge → clic → champs fautifs mis en évidence + cotes et
éléments en rouge sur le dessin + explication chiffrée » est conservé dans les
deux cas : les alertes portent déjà `champs`, `cotes` et `elements` (vérifié sur
les 31 cas), et la mise en rouge est faite **dans le SVG généré par Python**.

### 6.6 Dépendances

**Aucune dépendance nouvelle n'est nécessaire.** Le moteur n'utilise que la
bibliothèque standard (`math`) ; les schémas sont du SVG écrit à la main ; le
PDF passe par `ndc_pdf`/ReportLab déjà présent ; le composant est du HTML/JS
servi par Streamlit. `node` n'est utilisé **que** par un test, qui se déclare
« ignoré » s'il est absent.

### 6.7 Déroulé des phases 1 à 6

| Phase | Livrable | Fin de phase |
|---|---|---|
| 1 | `INVENTAIRE_HTML.md` — matrice de traçabilité, 19 catégories, une ligne par élément | commit |
| 2 | `bibliotheques.py`, `moteur.py`, `benchmark.py` + **parité 31 cas & benchmark & ≥ 300 cas Node** | commit, parité 100 % exigée avant la suite |
| 3 | `schemas.py` — 2 vues, 3 niveaux, 31 cotes, cartouche, mise en rouge | commit |
| 4 | `interface.py`, `modules/assemblages.py`, `registre.py`, composant + panneau | commit |
| 5 | `texte.py` (comparé à `exports_texte`), `rapport.py` + extension `ndc_pdf` | commit |
| 6 | non-régression, recette, `RAPPORT_MIGRATION.md`, `ECARTS_ET_CORRECTIONS.md` | commit + pousse |

Branche : `claude/dalle-module-creation-sjmitr` (celle que vous m'avez
assignée). Un commit par phase, message clair, jamais de commit direct ailleurs.

---

## 7. Risques et questions ouvertes

### Questions qui demandent votre décision

**Q1 — Ligne de base du harnais.** Option **(A)** adapter 4 lignes de tests
(`filtered_state` → `to_dict()`), ou **(B)** épingler la version de Streamlit
dans `requirements.txt` ? *Je recommande (A).* Sans l'une des deux, 128 des 320
contrôles ne tournent pas — donc pas de ligne de base opposable.

**Q2 — Python 3.11 vs 3.12.** Le dépôt exige 3.12 (PEP 701) mais
`.devcontainer` déclare 3.11. Je peux : ne rien faire (statu quo, signalé) ;
corriger `.devcontainer` en 3.12 ; ou réécrire les 2 f-strings pour redevenir
compatibles 3.11. **Hors périmètre de la migration** — dites-moi si vous voulez
que je le traite, et dans quel commit séparé.

**Q3 — Vignette d'accueil.** Je réutilise `Logo_corniere.png` (déjà servie par
l'URL distante). Voulez-vous une image propre « assemblage » ? Si oui, il faudra
la publier dans le dépôt `LLE4000/Etudes-structure`, pas ici.

**Q4 — Palette de la note.** Les notes béton utilisent `01_encre` (accent
`#33415C`). Le rapport Assemblages prend la **même** palette par défaut — dites
le contraire si vous voulez distinguer l'acier au premier coup d'œil (une
palette existe déjà : `03_acier`). *Je recommande la même : « pas de deuxième
charte ».*

### Risques identifiés, et comment je les traite

| Risque | Traitement |
|---|---|
| **Pièges de transcription JS → Python** : `Math.round` (JS arrondit −0,5 vers 0, Python fait du « pair le plus proche »), `ceil5` et sa tolérance, division par zéro qui doit donner `Infinity` et non une exception, `min`/`max` avec des valeurs « sans objet », départage du prédimensionnement au **premier** minimum, `|VEd|` et `|MEd|`, chaînes numériques venant du formulaire | fonctions d'appoint isolées et testées une par une **avant** `compute` ; 31 cas + ≥ 300 cas aléatoires Node comme filet |
| Le rapport PDF sur **une page A4 paysage** avec 38 vérifications | la note ne porte que les **18 essentielles** + la synthèse ; `doc.warnings` doit rester vide ; 2 pages permises plutôt que descendre sous 6,0 pt |
| L'extension de `ndc_pdf` casse les notes béton | l'étalon `reference/NOTE_DE_CALCUL.pdf` est comparé **au pixel** ; la phase 5 le rejoue avant et après |
| Le composant bidirectionnel ne marche pas sur votre déploiement | panneau « Cotes » livré en même temps, même état, même test |
| **Références citées de mémoire** (numéros de « Check » SCI P358, sous-alinéas EN 1993-1-8 / 1-1) | **listées comme non vérifiées** dans `RAPPORT_MIGRATION.md`, jamais modifiées |
| `fu` du S355 (510) et coefficients γM | **inchangés** ; regroupés dans un seul endroit du code, faciles à caler sur l'ANB |
| Vérifications `INT` / `COMP` sans exemple publié (`flA`, `m2`, `stD`, `vlP`, `vnP`, traitement de `HEd` et `MEd`) | transcrites à l'identique, **signalées comme telles** dans l'écran (badge de nature) et dans le rapport ; analysées dans `ECARTS_ET_CORRECTIONS.md` sans rien appliquer sans votre accord |
| Soudures : longueur efficace = longueur totale | affiché comme **hypothèse**, dans l'écran et dans la note |
| Le fichier `interface.py` devient trop gros | découpage par responsabilité dès la phase 4 (saisie / dessin / vérifications / prédim / exports), pas en fin de course |

### Ce que je ne ferai pas sans vous le demander

- modifier une formule, une référence normative ou un coefficient du HTML ;
- « corriger » une erreur de modèle : elle est **consignée**, chiffrée sur le cas
  par défaut, et attend votre arbitrage ;
- toucher à un module béton, à `poutre.py`, `dalle.py`, `predalle.py`, ou à leurs
  exports ;
- ouvrir une pull request.

---

## 8. Ce qui se passe à votre « go »

1. je crée la branche de travail et je verse les trois fichiers témoins
   (HTML, JSON, ce plan) ;
2. je lis le HTML en entier et je livre `INVENTAIRE_HTML.md` (phase 1) ;
3. puis le moteur et la parité (phase 2) — **rien ne part vers l'interface tant
   que la parité n'est pas à 100 %.**

**Je m'arrête ici et j'attends votre validation.**

---

## 9. Arbitrages rendus le 21/09/2026 (validation de la phase 0)

| Question | Décision |
|---|---|
| Q1 harnais | **(A)** : `filtered_state` → `to_dict()` sur les 4 lignes, rien d'autre. |
| Q2 Python | **3.12** : `.devcontainer` aligné ; le code n'est pas ramené à 3.11. La version servie par la plateforme de déploiement se règle dans ses paramètres avancés — à vérifier par le bureau. |
| Q3 vignette | `Logo_corniere.png` pour ce premier assemblage ; le registre porte une vignette **par assemblage**. |
| Q4 palette | `01_encre`, même identité graphique que les modules béton et leurs notes. |
| Architecture | validée : paquet `acier/` sans Streamlit, registre déclaratif, intégration minimale dans la navigation, aucune dépendance nouvelle. |
| Ergonomie | **toute la géométrie se modifie à l'écran** : chaque cote modifiable est cliquable sur le dessin (édition bidirectionnelle dessin ↔ champ), **et** le panneau « Cotes » est conservé — les deux, pas l'un ou l'autre. Une alerte géométrique est cliquable et met en évidence, sur le dessin et dans les champs, les paramètres responsables. |
| Parité | complète avec le HTML : aucune vérification, formule, alerte, donnée, dessin ni export ne disparaît. |
