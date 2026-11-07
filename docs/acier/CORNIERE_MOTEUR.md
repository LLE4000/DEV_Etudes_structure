# Cornières ancrées — le moteur sans Streamlit et le module du site

*Livré le 07/11/2026. Moteur `acier/corniere/` : bibliothèque standard, et le catalogue des profilés
`acier/profiles` (numpy, présent dans le navigateur) ; Python 3.12 et 3.13 (donc Pyodide 0.29.5). Extrait de
l'écran Streamlit `modules/corniere.py` (293 lignes), qui reste la **référence** et n'est pas modifié ; la preuve
de parité est `tests/test_corniere_moteur.py`. Le module du site est `api_web/corniere.py` (établi
`corniere.html`), ses dessins `api_web/dessin_corniere.py`, ses tests `tests/test_api_web_corniere.py`.*

> **⚠ Défaut NON CONSERVATIF de l'écran Streamlit, reproduit (parité) et signalé à chaque calcul.** La
> vérification « longitudinale (entre fixations) » de `modules/corniere.py` est provisoire (« TODO » dans son code) :
> sur un panneau de largeur s, M_long = V_loc·e = V·s·e et W_long = s·t²/6, donc **s se simplifie** et σ_long
> redonne exactement le σ du calcul transversal (U_long = U_ELU, au bit près). La flexion de la cornière entre
> fixations (q·s²/8) et la concentration de l'effort autour de chaque fixation **ne sont vérifiées nulle part** :
> une cornière peut être déclarée conforme « entre fixations » quel que soit l'entraxe s (10 cm ou 1,50 m donnent le
> même taux). Côté non sûr dès que les fixations sont espacées. Le site le dit en tête des avertissements du résultat
> (« **Côté non sûr** : … ») et dans le pied de la fenêtre longitudinale ; il ne le corrige pas (règle de parité).
> Les ancrages ne sont pas vérifiés non plus (l'écran le dit, le site aussi). Détail : § 8, points 1 et 2.

## 1 · Ce qui est livré

| Pièce | Rôle |
|---|---|
| `acier/corniere/entree.py` | les défauts et le domaine de chaque champ de l'écran (`CHAMPS`), la section proposée à l'ouverture, le catalogue des cornières (familles L et LI d'`acier/profiles`, triées par masse), la désignation du bureau, la relecture d'une désignation tapée, la normalisation stricte |
| `acier/corniere/moteur.py` | le calcul de l'écran **mot pour mot** (`calculer`), la section, les quatre vérifications, les formules du détail, les avertissements, le sélecteur trié par masse — et le **format complet** dans sa docstring |
| `api_web/corniere.py` | le module du site : note à plusieurs cornières, cadre Données, résultat, fenêtres de calcul, sélecteur |
| `api_web/dessin_corniere.py` | les quatre dessins à l'échelle (coupe, élévation, efforts, section) |
| `tests/test_corniere_moteur.py` | parité avec l'écran, contrôles à la main, couverture, catalogue, sélecteur, format, pureté, **mutations** |
| `tests/test_api_web_corniere.py` | ce que le site affiche = ce que le moteur calcule ; saisies, refus, sélecteur, éléments, échappement, dessins |

## 2 · L'API

```python
from acier.corniere import moteur as M

M.nouvelle_corniere() -> dict          # « Cornière 1 », L 100x100x10, linéaire 5 kN/m, e 2,3 cm, γ 1,35, s 30 cm…
M.normaliser(corniere) -> dict         # complétée et contrôlée ; ValueError hors du domaine de l'écran
M.verifier(corniere) -> dict           # le calcul complet
M.candidats(corniere, familles=None)   # les 233 cornières du catalogue, triées par masse, chacune calculée
M.optimal(corniere) -> dict | None     # la plus légère qui tient
M.nom_catalogue("80x80x8") -> "L 80x80x8"   ;   M.designation("L 45x45x4.5") -> "L 45×45×4,5"
```

**Entrée** (une clé absente ou `null` prend le défaut de l'écran) : `nom`, `standard` (le bouton « Utiliser une
cornière standard »), `profil` (nom au catalogue), `A`, `B`, `t` (saisie libre, mm), `charge` (`"lineaire"` ou
`"ponctuelle"` ; les libellés de l'écran sont admis), `V` (kN/m ou kN), `L` (m, charge ponctuelle), `e` (cm),
`gamma_ELU`, `s` (cm), `bord` (cm), `sig_lim_ELS`, `tau_lim_ELS`, `sig_lim_ELU` (N/mm²).

**Domaine** = les bornes des `st.number_input` de l'écran (lues dans l'écran par le test) : A et B de 10 à 300 mm,
t de 2 à 20 mm, V ≥ 0, L de 0,1 à 5 m, e ≥ 0, γ ≥ 1, s ≥ 10 cm, bord ≥ 0, σ_lim,ELS ≥ 10, τ_lim,ELS ≥ 0,1,
σ_eq,lim,ELU ≥ 10 N/mm². Hors domaine : **ValueError** avec un message (« excentricité e = -1 cm hors du domaine de
l'écran Streamlit (au moins 0 cm) ») — jamais de valeur ramenée en silence. Une cornière du catalogue peut avoir
t > 20 mm (38 sur 233, jusqu'à 35 mm) : c'est le même calcul, la borne de 20 mm n'est que celle de la saisie libre.

**Deux refus de plus que l'écran** (ajoutés à la relecture, chacun avec son message ; aucun chiffre admis ne change) :

- **t ≥ A ou t ≥ B** en saisie libre (« épaisseur t = 12 mm ≥ aile A = 12 mm : ce n'est pas une cornière ») :
  l'écran l'accepte parce qu'il ne lit que t, mais il n'y a plus ni aile B où poser la charge ni contour à dessiner
  (le dessin tombait sur une division par zéro) ;
- **un calcul qui déborde** (V, e, s ou γ démesurés : V = 10²⁰⁰ kN/m, s = 10³⁰⁸ cm…) : une contrainte devient infinie
  ou indéterminée, l'écran Streamlit s'arrête sur une `OverflowError` ; le moteur refuse (« valeurs trop grandes : le
  calcul déborde »). Le contrôle est mené à t = 2 mm, l'épaisseur la plus faible du domaine : aucune cornière du
  sélecteur ne déborde donc.

« -0 » est relu 0 (le champ n'écrit jamais « -0 »).

**Sortie de `verifier`** (JSON strict, nombres non arrondis) : `corniere` (normalisée), `section` (`source`,
`designation`, `famille`, `A`, `B`, `t`, `r1`, `r2`, `masse` — `None` en saisie libre, comme l'écran —, `aire`,
`yc`, `zc`, `norme`), `grandeurs` (§ 4), `verifications` (`els_sigma`, `els_tau`, `elu`, `long_elu` : valeur,
limite, taux = valeur / limite, `ok` = valeur ≤ limite), `verdicts` (les ✅ / ❌ de l'écran), `taux_max`, `statut`,
`gouvernant`, `calcul` (les formules, groupées : efforts, section, ELS, ELU ; panneau, efforts, contraintes — chaque
résultat se retrouve à partir de ses seuls termes), `avertissements` (`ancrages_non_verifies`,
`longitudinal_provisoire`, et `excentricite_hors_aile` quand e > B).

## 3 · La méthode de l'écran

Modèle **lame** : la cornière est vérifiée comme une plaque d'épaisseur t ; seules t, les charges et les critères
entrent dans le calcul (A et B n'y entrent pas).

1. **Efforts par mètre courant.** Charge linéaire : V_ELS = V, V_ELU = γ_ELU·V (kN/m). Charge ponctuelle (champ
   L de la barre latérale) : V_ELS = V / L, V_ELU = γ_ELU·V / L. M = V·e (kN·m/m), e en m = e / 100.
2. **Section** (b = 1 000 mm) : I = b·t³/12, W = b·t²/6, A_z = b·t.
3. **Transversal.** σ = M·10⁶ / W, τ = V·10³ / A_z, σ_eq = √(σ² + 3·τ²), à l'ELS et à l'ELU.
4. **Longitudinal** (« entre fixations », provisoire dans l'écran) : panneau b = 10·s (mm), V_loc = V·s/100 (kN),
   M_long = V_loc·e (kN·m), σ_long = M_long·10⁶ / (b·t²/6), τ_long = V_loc·10³ / (b·t), σ_eq,long.
5. **Taux et verdicts.** U_σ = σ_eq,ELS / σ_lim,ELS ; U_τ = τ_ELS / τ_lim,ELS ; U_ELU = σ_eq,ELU / σ_eq,lim,ELU ;
   U_long = σ_eq,long,ELU / σ_eq,lim,ELU. ✅ ELS si σ_eq,ELS ≤ σ_lim,ELS **et** τ_ELS ≤ τ_lim,ELS ; ✅ ELU si
   σ_eq,ELU ≤ σ_eq,lim,ELU ; ✅ longitudinal si σ_eq,long,ELU ≤ σ_eq,lim,ELU. Comparaisons larges (≤).

Chaque expression de `calculer` est celle de l'écran, **dans le même ordre d'opérations** : les flottants sont
identiques au bit près (100 % des comparaisons).

## 4 · D'où vient chaque grandeur

`grandeurs` → variable locale de `show()` dans `modules/corniere.py`.

| Grandeur | Sens | Dans l'écran |
|---|---|---|
| `b_trans`, `t` | largeur de lame (1 000 mm), épaisseur (mm) | `b_trans_mm`, `t_mm` |
| `I_trans`, `W_trans`, `Az_trans` | b·t³/12, b·t²/6, b·t | `I_trans`, `W_trans`, `Az_trans` |
| `V_ELS`, `V_ELU` | efforts par mètre (kN/m) | `V_ELS`, `V_ELU` |
| `e_m` | e en m | `e_m` |
| `M_ELS`, `M_ELU` | V·e (kN·m/m), affichés « M (ELS) » / « M (ELU) » | `V_ELS*e_m`, `V_ELU*e_m` (dans le markdown) |
| `M_ELS_Nmm`, `M_ELU_Nmm` | les mêmes en N·mm par mètre | `M_ELS_Nmm`, `M_ELU_Nmm` |
| `sigma_ELS`, `sigma_ELU`, `tau_ELS`, `tau_ELU` | contraintes transversales | mêmes noms |
| `sigma_eq_ELS`, `sigma_eq_ELU` | √(σ² + 3τ²) | mêmes noms |
| `b_long`, `I_long`, `W_long`, `Az_long` | le panneau de largeur s | `b_long_mm`, `I_long`, `W_long`, `Az_long` |
| `Vloc_ELS`, `Vloc_ELU` | V·s (kN) | mêmes noms |
| `Mlong_ELS`, `Mlong_ELU` | V_loc·e (kN·m) | `Mlong_ELS_kNm`, `Mlong_ELU_kNm` |
| `sigma_long_ELS`, `sigma_long_ELU`, `tau_long_ELS`, `tau_long_ELU` | contraintes du panneau | mêmes noms |
| `sigma_eq_long_ELS`, `sigma_eq_long_ELU` | σ_eq du panneau | mêmes noms |
| `U_sigma`, `U_tau`, `U_ELU`, `U_long_ELU` | les taux des bandeaux (÷ 100) | mêmes noms |
| `ok_els`, `ok_elu`, `ok_long` | les ✅ / ❌ | `ok_els`, `ok_elu`, `ok_long` (« ✅ » / « ❌ ») |

`I_trans`, `I_long`, `sigma_eq_long_ELS` et `tau_long_ELS` sont calculées par l'écran sans entrer dans un verdict ;
le moteur les rend pour la parité (et le détail du calcul).

## 5 · La preuve de parité (`tests/test_corniere_moteur.py`, 158 contrôles, ~3 s)

- **Les défauts et les bornes** du moteur sont LUS dans l'écran : ses `st.number_input`, `st.radio`, `st.toggle`,
  `st.selectbox` sont enregistrés par un faux `st` (12 champs numériques, le type de charge, la section proposée,
  la liste de 41 sections).
- **L'écran est exécuté** (`show()`) avec un faux `st` qui sert les saisies de chaque cas et enregistre ce qui
  s'affiche ; une trace (`sys.settrace`) relève **toutes** les variables locales de `show()` à son retour. Une
  variable ajoutée à l'écran fait rougir le test tant qu'elle n'est pas comparée. **99 cornières** : les 41
  sections de la liste de l'écran (au catalogue ou en saisie libre), la saisie libre aux bornes, les charges
  linéaires et ponctuelles (L = 0,1 et 5 m), V = 0, e = 0, γ = 1 et 1,5, s = 10 et 150 cm, chaque verdict ✅ et ❌
  (ELS par σ seul, par τ seul, par les deux ; ELU seul ; tout), les critères au minimum de l'écran, des valeurs non
  rondes, t = 25 et 35 mm (catalogue), 30 tirages à graine fixe et 3 **bornes exactes** construites (τ = τ_lim,
  σ_eq = σ_lim, σ_eq,ELU = σ_eq,lim : ✅, donc « ≤ » et non « < »). **4 719 comparaisons numériques, toutes
  identiques au bit près**, plus les textes affichés (bandeaux ELS / ELU / longitudinal avec leurs pourcentages et
  leurs icônes, épaisseur, largeur de lame, moments et contraintes du markdown).
- **Couverture** : les cas atteignent 30 situations exigées (chaque verdict, chaque vérification ok / nok, chaque
  charge, catalogue / saisie / liste, familles L et LI, V = 0, e = 0, t > 20, bornes exactes).
- **À la main** : deux cornières recalculées en N et mm dans le test.
- **Sélecteur** : les 233 cornières du catalogue repassées une à une par l'écran (saisie libre, même t) pour deux
  jeux de charges : taux et verdicts identiques, tri par masse, optimal = la première qui tient **et dont l'aile B
  reçoit la charge** (e ≤ B ; avec e = 80 mm, L 45×45×7 « tient » à l'écran mais son aile fait 45 mm : l'optimal est
  L 80×80×7).
- **Format** : JSON strict, normalisation idempotente, 30 saisies refusées (dont t ≥ aile et les calculs qui
  débordent), 12 bornes admises, les 3 069 formules
  recalculées à partir de leurs seuls termes, gouvernant et statut, désignations relues, avertissement e > B.
- **Pureté** : un sous-processus dont `sys.meta_path` refuse streamlit, pandas, reportlab, matplotlib, scipy,
  PIL, fitz charge le moteur et calcule les 99 cornières et le sélecteur ; seuls la bibliothèque standard et numpy
  sont chargés — sous Python 3.12 et 3.13, résultats identiques.
- **Mutations** : le moteur est rechargé depuis ses sources, modifié d'**une** ligne, et repassé par les contrôles
  ci-dessus (bornes lues dans l'écran, parité avec les écrans rejoués, refus, formules, sélecteur). Les **30
  mutations** sont toutes rejetées : I, W, A_z, largeur de bande, γ oublié, charge ponctuelle non répartie, e lu en
  dm, √(σ² + τ²), τ de l'ELS dans l'ELU, panneau 100·s, V·s sans /100, σ_eq,long avec τ_ELS, U_τ avec τ_ELU,
  « ≤ » devenu « < », ELS sans τ, longitudinal comparé à σ_lim,ELS, épaisseur ou ailes du catalogue faussées,
  valeur de la vérification τ, terme faux d'une formule, gouvernant au plus petit taux, sélecteur trié par nom,
  optimal le plus lourd, optimal qui oublie l'aile B, t ≥ aile admise, calcul qui déborde admis, défaut de e, borne
  de t, section proposée, saisie négative ramenée à la borne. Une
  mutation **témoin** (un commentaire) n'est rejetée par aucun contrôle.

## 6 · L'écran Streamlit, champ par champ, et le site

| Écran Streamlit | Site |
|---|---|
| bouton « Utiliser une cornière standard » | section 1 : segments **Catalogue / Saisie libre** |
| menu « Section » (41 sections, ordre alphabétique) | champ **Désignation** (« 80x80x8 », « L 80×80×8 »…) et **sélecteur** de la ligne « Cornière » du résultat : les 233 cornières EN 10056-1 triées par masse, ✓ la retenue, ★ la plus légère qui tient, filtres ailes égales / inégales, propriétés au survol |
| bandeau « Dimensions : A × B × t mm — kg/m » | fiche de la cornière (ailes, t, congés, masse, aire, norme) |
| largeurs A, B et épaisseur t (saisie libre) | section 1, en saisie libre (mêmes bornes) |
| « Type de charge » linéaire / ponctuelle | section 2 : segments |
| V (kN/m ou kN), e, coefficient global ELU, légende M = V × e | section 2 |
| L, longueur d'influence (barre latérale) | section 2, sous V, quand la charge est ponctuelle |
| s, distance de l'âme au bord du mur | section 3 (le bord est dit sans effet sur le calcul) |
| σ_lim,ELS, τ_lim,ELS, σ_eq,lim,ELU, légende S235 | section 4 |
| métriques t, b = 1 000 mm, poids | hypothèses de la fenêtre de calcul ; masse au pied du résultat |
| efforts, section, contraintes ELS / ELU (markdown) | fenêtre « transversal » : tableau + détail des formules substituées |
| bandeaux ELS (σ_eq/lim, τ/lim) et ELU | lignes du résultat (une par vérification, colorées), verdict gouvernant |
| panneau, efforts locaux, moments, contraintes, bandeau longitudinal | groupe « Longitudinal » du résultat et sa fenêtre de calcul |
| « 3) Vérification des ancrages : à compléter » | avertissement du résultat, aide de la section 3, renvoi « fixation non vérifiée » du dessin |
| « Notes & hypothèses » | hypothèses des fenêtres, pied de la fenêtre longitudinale, avertissement « provisoire » |
| — (aucun dessin) | **quatre dessins à l'échelle** : coupe (schéma statique), élévation entre fixations, efforts (diagrammes V et M), section cotée |
| — (une cornière à la fois) | **plusieurs cornières par note** (ajouter, copier, supprimer, ouvrir) |

**Hors du site** : le bouton « Accueil » de l'écran (la navigation du site le remplace). Aucun calcul de l'écran
n'est omis.

## 7 · Ce qui n'est pas couvert

- les **ancrages** (non vérifiés dans l'écran : le site le dit, sans rien calculer de neuf) ;
- la **flexion de la cornière entre fixations** (le « TODO » de l'écran : M = q·s²/8 n'est pas calculé) ;
- la **position et le diamètre de la fixation** sur le dessin sont une convention de dessin (M6 à M20 selon l'aile),
  sans effet sur le calcul ;
- la note de calcul PDF (hors périmètre du site : le bouton est grisé).

## 8 · Défauts constatés dans l'écran Streamlit (reproduits par le moteur, non corrigés)

1. **Le modèle lame ignore la géométrie de la cornière** : A et B n'entrent pas dans le calcul, seule t compte ;
   deux cornières de même épaisseur ont les mêmes taux, quelles que soient leurs ailes. L'écran ne contrôle pas non plus que
   la charge tombe sur l'aile : avec e > B, il calcule comme si de rien n'était. Le moteur le **signale**
   (avertissement `excentricite_hors_aile`, sans changer un chiffre).
2. **NON CONSERVATIF — la vérification longitudinale répète la transversale** : M_long = V_loc·e sur un panneau de
   largeur s donne σ_long = σ transversal (s se simplifie : `sigma_long_ELU` = `sigma_ELU`, U_long = U_ELU). La
   flexion entre fixations, annoncée « TODO » dans le code, n'est pas calculée, et le modèle « par mètre courant »
   suppose l'aile fixée tenue tout du long alors qu'elle ne l'est qu'aux fixations : **côté non sûr** dès que les
   fixations sont espacées (q·s²/8 n'est vérifié nulle part, s ne change aucun taux). Voir l'encadré en tête.
3. **τ_lim,ELS = 0,8 N/mm² par défaut** : un ordre de grandeur cent fois plus faible qu'un cisaillement admissible
   d'acier (S235 : ≈ 90 N/mm² à l'ELS). Côté sûr, mais c'est ce critère qui gouverne la cornière neuve (62,5 %) et
   qui écarte au sélecteur toute cornière de moins de 6,25 mm d'épaisseur pour 5 kN/m (τ = V/t ≤ 0,8). Probable confusion d'unité ou de grandeur.
4. **Les bandeaux sont verts même quand ❌** : les verdicts ELS et ELU sont écrits dans `st.success` (vert), le
   longitudinal dans `st.info` (bleu), quel que soit le résultat ; et le pourcentage est arrondi à l'unité
   (`:.0f`) : 100,4 % s'affiche « 100% ❌ ». Le site colore selon le taux et le verdict, et arrondit sans changer
   de bande (101 %). L'écran arrondit aussi « au pair » (format Python) : la cornière neuve, U_τ = 62,5 %, s'y lit
   « 62% » ; le site arrondit au plus proche (63 %) et sa fenêtre de calcul écrit 62,5 %. Relevé à la relecture sur
   24 cas rejoués (écran Streamlit face aux fragments du site) : aucun autre écart que ces arrondis d'un point.
5. **La longueur d'influence L est dans la barre latérale** (`st.sidebar`), loin de la charge, et n'apparaît que
   pour une charge ponctuelle ; sur téléphone elle est repliée.
6. **La distance au bord du mur est demandée et jamais utilisée.**
7. **La liste des cornières** (`ANGLES_STD`, 41 sections) est triée par ordre **alphabétique** : la section
   proposée à l'ouverture est « 100x100x10 » (et non la plus petite) ; le commentaire « inégales et autres
   tailles » est placé avant des cornières égales. Voir § 9 pour ses écarts au catalogue.
8. Le code de l'écran porte des séquences `\(` `\)` dans des chaînes non brutes (LaTeX du markdown) : Python 3.12
   émet `SyntaxWarning: invalid escape sequence` au chargement (une erreur dans une version future).

9. **t ≥ A accepté** : une saisie libre 10 × 10 × 20 est calculée (seul t compte) ; ce n'est pas une cornière.
10. **Débordement** : V, e ou s démesurés font tomber l'écran sur une `OverflowError` Python.

Le site **refuse** ce que l'écran refuse (bornes de ses champs), avec un message, et les deux cas 9 et 10 (§ 2) ;
aucune saisie n'est ramenée en silence.

## 9 · La liste de l'écran face au catalogue `acier/profiles`

Le site propose le catalogue de l'application (`acier/donnees/profiles.json`, EN 10056-1 : 233 cornières, familles
L et LI), source unique des profilés, au lieu de la liste recopiée dans l'écran. Constat (le test exige qu'il soit
écrit ici) :

- **10 sections de la liste de l'écran sont absentes du catalogue** : `15x15x3`, `30x30x5`, `80x80x12`,
  `40x20x4`, `40x25x4`, `50x30x5`, `60x40x6`, `70x50x6`, `80x40x6`, `80x60x7`. Elles restent calculables en
  saisie libre (même t, donc mêmes taux : le modèle lame ne lit que t), sans masse.
- **Les 31 autres** ont des masses supérieures de +1,7 à +2,5 % à celles du catalogue (100x100x10 : 15,32 kg/m à
  l'écran, 15,04 kg/m au catalogue).
- **`90x90x10` est fausse** : 18,54 kg/m à l'écran, 13,45 kg/m au catalogue (+38 %) — la masse d'une 120×120×10.

La masse n'entre pas dans le calcul (elle n'est qu'affichée) : ces écarts ne changent aucun taux.

## 10 · Le module du site (`api_web/corniere.py`, `api_web/dessin_corniere.py`)

- **La note** : `{"version": 1, "cornieres": [<cornière du moteur>, …], "actif": 0}` ; ajouter (« Cornière n »,
  le premier nom libre), copier (« (copie) »), ouvrir, supprimer (jamais la dernière) ; le résumé de la note est sa
  cornière la plus chargée.
- **Données** : 1 Cornière (nom, origine catalogue / saisie libre, désignation tapée relue au catalogue — « 80x80x8 »,
  « L 80×80×8 », ailes dans un ordre ou dans l'autre —, fiche du catalogue ou trio A, B, t) ; 2 Charges (type, V,
  L si ponctuelle, e, γ_ELU) ; 3 Ancrages (s, bord) ; 4 Critères (trio σ_lim,ELS, τ_lim,ELS, σ_eq,lim,ELU). Toute
  saisie passe par `normaliser` du moteur : hors du domaine de l'écran, elle est refusée avec son message.
- **Résultat** : verdict (taux gouvernant, comparaison « valeur ≤ limite »), ligne « Cornière » (sélecteur, ★ quand
  une plus légère tient), groupes Transversal et Longitudinal (une ligne par vérification, colorée par `C.niveau`,
  rouge dès qu'elle ne tient pas), avertissements titrés en gras (le grave — charge au-delà de l'aile — en tête, puis
  « Côté non sûr » pour la vérification longitudinale et « Ancrages non vérifiés »), masse.
- **Sélecteur** : les 233 cornières triées par masse, ✓ la retenue, ★ la plus légère qui tient et dont l'aile B
  reçoit la charge ; une cornière qui tient mais dont l'aile B est plus courte que e est **grise** (l'en-tête et la
  ligne disent pourquoi), une qui ne tient pas est rouge.
- **Fenêtres de calcul** : transversal (efforts, section, contraintes ELS et ELU) et longitudinal (panneau, efforts,
  contraintes, hypothèse provisoire) ; chaque ligne « symbole = expression = valeurs = résultat » est écrite à partir
  des termes du moteur, et le test la réévalue : les valeurs écrites redonnent le résultat à l'arrondi près.
- **Dessins** : quatre vues à l'échelle, reconstruites jusqu'à ce que leur texte se lise à 13 px dans la toile de
  l'établi (626 × 682 px au format PC) ; aucun indice « _ » brut. Une vue qu'on ne sait pas dessiner (proportions
  absurdes : s ou e de plusieurs kilomètres face à une aile de 20 mm) est remplacée par un cadre « Dessin impossible à
  cette échelle » ; le calcul est rendu normalement.
- **Saisie** : les champs réécrivent la valeur saisie sans l'arrondir (6 décimales) ; un nom de plus de 60 caractères
  est refusé ; une note dont l'indice de la cornière ouverte est abîmé s'ouvre sur la première.
- **Vérification** : `tests/test_api_web_corniere.py` (68 contrôles, dont un **bout à bout** : l'écran Streamlit rejoué (faux `st`) face aux chiffres que le site AFFICHE, sur 7 cas — taux à ±1 point, contraintes à ½ unité affichée, rouge ⇔ ❌), le contrat commun
  `tests/test_api_web_modules.py corniere` (17), et `site/web/verifier_site.js … module:corniere` (19, dans
  Chromium, moteur dans le navigateur) ; les captures des quatre vues, des fenêtres, du sélecteur, de la saisie libre,
  de la charge ponctuelle, de deux cornières et des formats tablette et téléphone ont été relues (aucun texte coupé,
  aucun débordement mesuré, toutes sections ouvertes).
