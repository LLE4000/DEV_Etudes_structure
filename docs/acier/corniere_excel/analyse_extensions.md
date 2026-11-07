# Classeur REALEX « Dimensionnement cornières » — le modèle étendu

*Lecture du 07/11/2026. Cette partie prolonge l'analyse mécanique du classeur (`analyse_mecanique.md`, qui traite l'inventaire des cellules, la reproduction et les jeux) : elle dit **comment étendre** la méthode sans la trahir — diagrammes continus, cornière retournée, ailes inégales, cornière longue à plusieurs ancrages, ELS/ELU et efforts pour Hilti, maillage de la coupe, architecture du futur module. C'est une proposition chiffrée ; aucun fichier du site n'a été modifié.*

**Comment lire ce rapport.** Chaque chiffre sort d'un script lancé avec `/usr/bin/python3.12` (ou de Pyodide sous Node pour les temps navigateur) ; la sortie brute de chaque script est à côté de lui. Trois étiquettes :

- **CERTAIN** : lu dans une formule ou une valeur du classeur (cellule citée, par ex. `P8!C57`) ;
- **INTERPRÉTÉ** : sens physique qu'on en déduit — lecture d'ingénieur, à confirmer par l'auteur ;
- **HYPOTHÈSE** : valeur choisie pour chiffrer un exemple (raideur d'ancrage, répartition G/Q…), dite comme telle.

Repère de la coupe (configuration A, celle du classeur) : **u** = distance horizontale depuis le **nu du mur** (dos de l'aile verticale) ; **y** = hauteur depuis le **dessous** de la cornière. X' = X + Xj, Y' = Y + Yj.

| Script (dossier `extensions/`) | Rôle | Sortie |
|---|---|---|
| `ext_commun.py` | cas de contrôle, méthode du classeur en symboles, épaisseur du congé t(s), corps libre | — |
| `e01_diagrammes.py` | diagrammes continus M, N, V ; balayage du congé ; maxima | `e01_sortie.txt` |
| `e02_retournee_inegales.py` | configuration B, position de l'ancrage, ailes inégales, catalogue du classeur | `e02_sortie.txt` |
| `e03_longitudinal.py` | cornière de 2,00 m, 3 ancrages : poutre + torsion sur appuis élastiques | `e03_sortie.txt` |
| `e04_els_elu_ancrages.py` | ELS/ELU, feuille E10A BD, efforts par ancrage pour Hilti | `e04_sortie.txt` |
| `ef_l.py` | **prototype d'éléments finis de la coupe, numpy seul** (Q9, contact unilatéral, solveur bande) | — |
| `e05_ef_coupe.py`, `e05b_sensibilite_levier.py` | cas maillés, comparaison aux sections 1–4, effet de levier | `e05_sortie.txt`, `e05b_sortie.txt` |
| `e06_pyodide.mjs`, `e06b_temps_cpython.py` | le même calcul dans Pyodide 0.29.5 + numpy 2.2.5 (ceux du site) ; temps CPython | `e06_sortie.json`, `e06b_sortie.txt` |
| `e07_architecture.py` | moteur actuel du site sur le cas de contrôle, géométrie partagée, cornière PRS | `e07_sortie.txt` |

Figures (`figures/`) : `e01_schema_moments.png`, `e01_diagrammes.png`, `e01_efforts_NV.png`, `e02_A_contre_B.png`, `e03_longitudinal.png`, `e05_ef_vonmises.png`, `e05_ef_zoom_angle.png`, `e05_surface_interieure.png`, `e05_effet_levier.png`.

---

## 0. L'essentiel

1. **La méthode est reproduite au bit près** : M1 = 2,009720 ; M2 = 2,565600 ; M3 = 2,358697 ; M4 = 2,896645 kN·m ; taux maximal 88,8621 % en section 3 — écarts au cache ≤ 4,4·10⁻¹⁶ [e01 §1].
2. **Les diagrammes continus se déduisent des mêmes formules, sans interpolation** : M(u) = R·(X' − u) dans l'aile horizontale (statique exacte), M(y) = Mmax·(Y' − y)/(Y' − e/2) dans l'aile verticale (P8!C55·C56). Dans le congé, l'épaisseur croît **au second ordre** pendant que le moment croît linéairement : le maximum n'est pas au pied du congé mais **≈ 2 mm à l'intérieur** — +2,11 % sur la section 1 (77,40 %), +1,45 % sur la section 3 (**90,15 %** au lieu de 88,86 %) [e01 §3].
3. **Les formules de l'aile verticale ne sont pas l'équilibre d'un seul corps libre** : l'effort du couple H = Mmax/(Y' − e/2) = 41,38 kN suppose le cisaillement d'ancrage au nu du mur, le diagramme M = H·(Y' − y) le suppose sur l'axe de l'aile. Les deux choix sont du côté sûr. Le maillage place le pivot à l'**arête inférieure** (y ≈ 0 à 1,9 mm) : H = 33,9 à 34,7 kN et M3 = 1,94 à 1,98 kN·m : le classeur est **au-dessus de 19 à 22 %** sur M3 et de 23 à 26 % sur la traction d'ancrage (42,76 kN) [e01 §2, e05 §3–4].
4. **Retourner la cornière (aile horizontale en haut) change le schéma** : le contact passe au **bout libre** de l'aile verticale, le moment entre l'ancrage et l'angle devient **constant et maximal**, R·(X' − e/2) = 2,886 kN·m, avec une **compression** R. Même cornière, même charge : section 3 à **112,6 %** (contre 77,0 % en A, même équilibre ; 88,9 % au classeur), traction d'ancrage **×2,43** si l'ancrage reste à 85 mm sous la face supérieure. Le maillage le confirme à 0,14 % près [e02 §1, e05 §6]. Appliquer les formules du classeur à B serait **non conservatif** : le vrai M3 dépasse celui de la formule Mmax·Ye/YH de 22 % (V sur l'axe) à 36 % (convention du classeur).
5. **Cornière longue (2,00 m, 3 ancrages, 42,76 kN sur 300 mm centrés à 0,40 m)** : la répartition « R/n » est fausse. Ancrage le plus chargé : traction **1,84 à 2,22 fois** H/3, cisaillement **2,10 fois** R/3 ; un ancrage reçoit un cisaillement **inversé** (−1,9 à −2,4 kN) ; la cornière travaille en **torsion** (τ = 92 à 112 N/mm², 67 à 82 % de τ_adm) et en flexion longitudinale (σ ≈ 85 N/mm²) — rien de cela n'existe dans le classeur [e03 §3–4].
6. **Effet de levier** : avec un ancrage très raide (au-delà de ≈ 250 kN/mm par cheville dans le cas de contrôle), le haut de l'aile verticale vient s'appuyer sur le mur au-dessus de l'ancrage ; la traction totale passe de 33,9 kN à **95,8 kN** (ancrage infiniment raide) — **2,2 fois** les 42,76 kN du classeur [e05 §3, §7]. Avec des raideurs courantes de chevilles (≤ 100 kN/mm), il n'apparaît pas.
7. **ELU** : le classeur multiplie N et V par 68,9/42,76 = **1,6113**. Avec γG = 1,35 et γQ = 1,50 sur les mêmes charges, ce rapport ne peut pas dépasser 1,50 : **les deux réactions ne viennent pas de la même combinaison** (CERTAIN, arithmétique). Coïncidence à vérifier : 1,35·G + 1,5·Q de la volée 12 de la feuille E10A BD vaut **42,768 kN/m**, à 0,018 % de la « réaction ELS » 42,76 [e04 §1–2].
8. **Le maillage tourne dans le navigateur** : 1 469 nœuds en **0,79 s**, 4 725 nœuds (9 450 inconnues) en **4,55 s**, dans Pyodide 0.29.5 + numpy 2.2.5 — ceux que le site embarque — sous Node 22 ; chargement de Pyodide et numpy 1,71 s ; résultats identiques à CPython [e06].
9. **Ce que le maillage confirme** : le moment de l'aile horizontale (0,03 %), la forme des diagrammes, la configuration B. **Ce qu'il contredit** : la distribution linéaire des contraintes dans les sections « e + r » (2 et 4) — en haut de ces coupes la contrainte est presque nulle (1,7 N/mm²), en bas elle vaut 1,9 fois la valeur poutre ; elles ne gouvernent pas. **Ce qu'il ajoute** : un pic **local** de 236 N/mm² à 3 mm dans l'arc du congé (facteur de concentration ≈ 1,30 sur la contrainte nominale au même point) — une contrainte de pointe, pas une contrainte nominale [e05 §4–5].
10. **Le module actuel du site** (modèle « lame » : M = V·e au nu du mur, épaisseur e) donne **120,65 %** sur le cas de contrôle (+35,8 % sur le classeur) : il ne connaît ni le congé, ni l'aile verticale, ni l'ancrage [e07 §1].

---

## 1. Diagrammes continus exacts (configuration A)

### 1.1 Ce que les formules imposent — CERTAIN

| Grandeur | Formule du classeur | Cellules | Diagramme continu qui en découle |
|---|---|---|---|
| Moment, aile horizontale | M1 = R·(Xe + Xj), M2 = R·(Xe+r + Xj) | `P8!C35`, `C45` (Xe = `C19` = X − r − e, Xe+r = `C20` = X − e) | **M(u) = R·(X' − u)**, nul sous la charge, R·X' au nu du mur |
| Moment au nu du mur | Mmax = (X + Xj)·R | `C55` = `C63` = `C78` | M(0) = R·X' = 3,207 kN·m |
| Moment, aile verticale | M3 = Mmax·(Ye + Yj)/(YH + Yj), M4 = Mmax·(Ye+r + Yj)/(YH + Yj) | `C56`·`C55` = `C57`, `C64`·`C63` = `C65` (Ye = `C23`, Ye+r = `C24`, YH = `C25` = Y − e/2) | **M(y) = Mmax·(Y' − y)/(Y' − e/2)**, nul à l'ancrage, Mmax à y = e/2 |

Le diagramme continu n'est donc pas « tracé entre quatre points » : il est la droite dont les quatre sections sont des lectures. La reproduction est exacte (écarts ≤ 4,4·10⁻¹⁶ sur `C35`, `C45`, `C57`, `C65`, `I37` à `I67`, `D3`) [e01 §1].

### 1.2 Lire ces formules comme un équilibre — INTERPRÉTÉ, CALCULÉ

Corps libre de la cornière isolée (frottement sur le mur négligé, comme dans le classeur) : R vers le bas à u = X' ; ancrage à y = Y' : traction H (vers le mur) et cisaillement V = R (vers le haut), de ligne d'action u_V ; compression du mur C = H au pivot (u = 0, y = y_P).

> Moments autour du pivot : **H·(Y' − y_P) = R·(X' − u_V)** ; aile verticale (coupe à la hauteur y, au-dessus du congé) : **M(y) = H·(Y' − y) − R·(e/2 − u_V)**.

- **Le classeur prend H = Mmax/(Y' − e/2) = 41,381 kN** : c'est l'équilibre avec u_V = 0 (cisaillement au **nu du mur**) et y_P = e/2.
- **Mais il prend M(y) = H·(Y' − y)** : c'est le moment avec u_V = e/2 (cisaillement sur **l'axe** de l'aile). Avec u_V = 0, le même corps libre donnerait M3 = 2,038 kN·m (variante Q1).
- **Au nœud filaire** (intersection des lignes moyennes, u = y = e/2), l'aile horizontale délivre R·(X' − e/2) = 2,8863 kN·m quand l'aile verticale du classeur reçoit 3,2070 kN·m : saut de **R·e/2 = 0,3207 kN·m (+11,1 %)**. Le classeur n'est donc pas un modèle filaire ; c'est un modèle filaire **majoré** des deux côtés.
- **Équivalence exacte** : avec V sur l'axe, le pivot qui redonne M3 et M4 au bit près vaut y_P* = Y' − (X' − e/2)(Y' − e/2)/X' = **15,25 mm** (≈ e). Ce n'est pas une constante physique : y_P* dépend de X et Y.

| Corps libre (V sur l'axe sauf Q1) | H (kN) | M3 | M4 (kN·m) | taux 3 (convention du classeur) | N par ancrage (kN) |
|---|---|---|---|---|---|
| Classeur (modèle E) | 41,381 | 2,35870 | 2,89665 | 88,86 % | 14,253 (C82, bras Y − 1 cm) |
| P0 — pivot à l'arête (y = 0) | 33,956 | 1,93552 | 2,37695 | 73,03 % | 11,319 |
| P1 — pivot à e/2 (nœud filaire) | 37,243 | 2,12283 | 2,60698 | 80,03 % | 12,414 |
| P10 — pivot à 10 mm (C76) | 38,484 | 2,19359 | 2,69388 | 82,68 % | 12,828 |
| P15 — pivot à e | 41,233 | 2,35027 | 2,88630 | 88,55 % | 13,744 |
| Q1 — pivot à e/2, V au nu du mur | 41,381 | 2,03800 | 2,57595 | 76,86 % | 13,794 |

Le maillage (§5) tranche la position du pivot pour un mur raide : **l'arête inférieure** (y = 0,00 mm ; 1,86 mm avec un mur souple), donc la variante **P0** (H maillé 33,897 kN contre 33,956 kN). Le classeur dépasse le maillage de **19 à 22 %** sur M3 et de **23 à 26 %** sur la traction d'ancrage (42,76 kN contre 33,9 à 34,7) — **tant qu'il n'y a pas d'effet de levier** (§5.4).

### 1.3 Effort normal et tranchant — CALCULÉ

| Aile | Zone | N | V | Remarque |
|---|---|---|---|---|
| horizontale | e ≤ u < X' | 0 | R = 42,76 kN | porte-à-faux pur |
| horizontale | u > X' | 0 | 0 | bout libre |
| verticale | e ≤ y < Y' | **+R = 42,76 kN (traction)** | **H** (41,38 kN modèle E ; 33,96 P0) | l'angle « pend » à l'ancrage |
| verticale | y > Y' | 0 | 0 | au-dessus de l'ancrage |

Le « τ » des sections 3 et 4 (`G54` = `G64` = R/(L·t)) est en réalité l'effort **normal** de l'aile verticale ; son vrai tranchant est H. En section 3 : σ_N = 9,502 N/mm², τ_H = 9,196 N/mm² (modèle E) — voisins par coïncidence (X ≈ YH). Ajouter σ_N à la flexion porte la section 3 à **92,85 %** (convention E) ou **83,96 %** (P1) [e01 §3–4]. Figure `e01_efforts_NV.png`.

### 1.4 Propriétés variables dans le congé, contrainte et taux le long des ailes — CALCULÉ

Épaisseur à la distance s du dos de l'autre aile (congé de rayon r centré en (e + r ; e + r)) :

> **t(s) = e + r − √(r² − (e + r − s)²)** pour e ≤ s ≤ e + r ; t = e au-delà ; s < e : bloc d'angle (pas de théorie des poutres).

W(s) = L·t(s)²/6, A(s) = L·t(s) ; σ(s) = M(s)/W(s) (+ N/A dans l'aile verticale si l'on retient l'effort normal) ; taux(s) = σ_éq(s)/σ_adm. Balayage au pas de 0,02 mm [e01 §3] :

| Aile | Section ou point | s (mm) | t (mm) | M (kN·m) | σ_éq (N/mm², convention du classeur) | taux |
|---|---|---|---|---|---|---|
| horizontale | section 2 (nu intérieur de l'aile verticale) | 15,00 | 28,000 | 2,56560 | 66,04 | 27,90 % |
| horizontale | milieu du congé | 21,50 | 16,745 | 2,28787 | 163,86 | 69,24 % |
| horizontale | **maximum** | **26,03** | 15,151 | 2,09417 | 183,19 | **77,40 %** |
| horizontale | section 1 (pied du congé) | 28,00 | 15,000 | 2,00972 | 179,40 | 75,80 % |
| verticale (modèle E) | section 4 | 15,00 | 28,000 | 2,89665 | 74,42 | 31,44 % |
| verticale (modèle E) | **maximum** | **26,34** | 15,106 | 2,42739 | 213,37 | **90,15 %** |
| verticale (modèle E) | section 3 | 28,00 | 15,000 | 2,35870 | 210,31 | 88,86 % |

Figures : `e01_schema_moments.png` (cornière à l'échelle, diagrammes portés sur les ailes, quatre sections, bloc d'angle hachuré) et `e01_diagrammes.png` (M, puis taux en fonction de la position, congés grisés, maximum repéré).

**« e + r » : conservatif, admissible ou à remplacer ?** L'épaisseur e + r est géométriquement juste à l'endroit des coupes 2 et 4 (CERTAIN : t(e) = e + r). Mais (i) ce ne sont pas les points les plus sollicités du congé, (ii) le maillage montre que la contrainte n'y est pas linéaire (§5.3). **Proposition** : garder les sections 2 et 4 comme repères historiques, et **vérifier le maximum du balayage** t(s), qui reste dans la même théorie des poutres et ne coûte rien.

### 1.5 Ce que le module calcule — proposition

- **Mode « méthode historique »** : le modèle E, mot pour mot (M1..M4, σ, τ, σ_éq, taux, ancrages C82/C83), avec le cas de contrôle comme test de parité.
- **Mode « équilibre »** : le corps libre paramétré (pivot y_P, ligne d'action u_V, frottement nul), N et V vrais dans chaque aile, même balayage. Valeurs par défaut proposées : u_V = e/2, y_P = 0 (maillage) — ou y_P = e/2 si l'on veut rester proche du classeur.
- Dans les deux modes : diagrammes **continus** M, N, V par aile ; **diagramme séparé** des contraintes et du taux ; quatre sections marquées ; maximum et position affichés ; congés et bloc d'angle identifiés. Le moment (équilibre) et la résistance (géométrie locale) restent deux courbes distinctes.

---

## 2. Cornière retournée et ailes inégales

### 2.1 Configuration B (aile horizontale en haut) : le corps libre — CALCULÉ

Géométrie : l'aile horizontale est en haut, l'aile verticale descend le long du mur depuis l'angle ; Y_B = distance de la face supérieure à l'axe de l'ancrage ; y_a = H_aile − Y_B = hauteur de l'ancrage au-dessus du bout libre.

- R fait tourner la cornière vers le mur sous l'ancrage : **le contact est au BOUT LIBRE** de l'aile verticale (y ≈ 0), sous l'ancrage ; au-dessus de l'ancrage, l'angle s'écarte du mur.
- Moments autour du bout comprimé : **H·(y_a − c) = R·(X' − u_V)** — le bras est la distance de l'ancrage au **bout libre**, pas à l'aile horizontale.
- Aile verticale, **entre le bout et l'ancrage** : M = H·(y − c), V = H, N = 0.
- **Entre l'ancrage et l'angle** : **M = R·(X' − u_V) constant**, V = 0, **N = −R (compression)**.
- Aile horizontale : M(u) = R·(X' − u) inchangé, mais **la face côté congé est comprimée** (en A elle est tendue).
- **Section nette** : en B, le trou de l'ancrage est dans la zone de moment maximal (en A, le moment y est nul).

| Cas de contrôle (L120×120×15, 42,76 kN à 75 mm) | M3 / M4 (kN·m) | taux 3 | traction totale (kN) |
|---|---|---|---|
| A, méthode du classeur | 2,3587 / 2,8966 | 88,86 % | 42,76 |
| A, corps libre P0 (+ effort normal) | 1,9355 / 2,3770 | 77,03 % | 33,96 |
| B, Y_B = 85 mm (y_a = 35), V sur l'axe | **2,8863 / 2,8863** | **112,64 %** | **82,47** |
| B, Y_B = 85 mm, convention du classeur (u_V = 0) | 3,2070 / 3,2070 | 124,66 % | 91,63 |
| B, Y_B = 45 mm (y_a = 75), V sur l'axe | 2,8863 / 2,8863 | 112,64 % | 38,48 |

- Section nette au droit de l'ancrage en B (3 trous Ø18 supposés sur 300 mm) : σ = 324,5 N/mm², **137,1 %** [e02 §1].
- Compression du mur au bout libre : 274,9 N/mm (Y_B = 85) — à vérifier sur le béton (pression locale), ce que le classeur n'a jamais à faire en A.
- **Maillage** : B-Y85 H = 82,580 kN (corps libre 82,466 : +0,14 %), contact à y = 0,00 ; M en section 3 = 2,8867 kN·m (corps libre 2,8863), N = −42,89 kN ; B-Y45 H = 38,520 kN (+0,09 %) avec **le même** moment en section 3 [e05 §6]. La contrainte maximale de surface est en **compression**, 327,5 N/mm² (local).
- Figure `e02_A_contre_B.png` : en A, le moment de l'aile verticale est nul à l'ancrage et croît vers l'angle ; en B, il est maximal et constant de l'ancrage à l'angle.

### 2.2 Ce qui change, et ce qui doit le dire à l'écran — INTERPRÉTÉ

| Hypothèse | Configuration A (classeur) | Configuration B |
|---|---|---|
| Contact avec le mur | arête inférieure, sous l'angle | **bout libre** de l'aile verticale |
| Bras de l'ancrage | Y (vers le dessous) | **distance ancrage – bout libre** |
| Meilleure position de l'ancrage | le plus haut possible | le plus **près de l'angle** possible (Y_B = 45 → 38,5 kN ; 85 → 82,5 kN ; 95 → 115,5 kN) |
| Moment, aile verticale | nul à l'ancrage, croissant vers l'angle | **constant maximal** de l'ancrage à l'angle |
| Effort normal, aile verticale | traction R | **compression R** |
| Face du congé | tendue | **comprimée** |
| Trou d'ancrage | hors moment | **dans le moment maximal** (section nette) |
| Formule M3 = Mmax·Ye/YH | applicable | **inapplicable** : le vrai M3 la dépasse de 22 % (V sur l'axe) à 36 % (convention du classeur) |

Il ne suffit donc pas de retourner le dessin : B demande son propre corps libre, sa propre position de contact et une vérification de la section nette et du béton au bout libre.

### 2.3 Ailes inégales — CERTAIN pour le classeur, CALCULÉ pour l'exemple

- Les formules de P8 ne lisent que e, r, X, Y (et larg., haut. pour le dessin) : elles s'appliquent telles quelles à une cornière inégale en configuration A. L'orientation (grande aile contre le mur ou en console) change Y possible, donc le partage entre traction d'ancrage et moment.
- **Le catalogue du classeur ne permet pas de choisir l'orientation** : 15 noms y figurent deux fois avec largeur et hauteur échangées (L150x100x12 lignes 6 et 7, L120x80x11 lignes 10 et 11, …) ; la recherche `VLOOKUP(…;FALSE())` de `P8!C4:C7` ne lit que la **première** ligne : la seconde orientation est inaccessible [e02 §4].
- **Rayon** : pour L150x100x12, le classeur porte r = 6 mm (`CORNIERES!E6`, `E7`) ; le catalogue du site (EN 10056-1) donne r1 = 12 et r2 = 6 — le classeur a pris le **rayon de bout d'aile**. Effet sur un exemple (même charge que le cas de contrôle, pour comparer ; cette cornière ne convient pas) : taux 1 = 143,3 % avec r = 6 contre 128,3 % avec r = 12 (côté sûr, mais géométrie fausse). Deux autres lignes diffèrent (L25x25x3 et L20x20x3 : 4 mm contre r1 = 3,5) ; 31 lignes (17 désignations) n'existent pas au catalogue du site.
- **Exemple L150×100×12 (r1 = 12), X = 75, ancrage à 35 mm du bord libre** :

| Orientation (configuration A) | taux 1 | taux 3 (classeur) | traction (classeur) | taux 3 (P0) | traction (P0) |
|---|---|---|---|---|---|
| grande aile horizontale (150 × 100), Y = 65 | 128,3 % | 131,1 % | 58,31 kN | 114,6 % | 45,39 kN |
| grande aile verticale (100 × 150), Y = 115 | 128,3 % | 157,4 % | 30,54 kN | 142,3 % | 25,66 kN |

Grande aile verticale : ancrage plus haut → **traction presque divisée par deux**, mais moment en section 3 plus proche de R·X (rapport Ye/YH → 1). L'orientation doit être un **paramètre explicite** du calcul, pas une ligne en double du catalogue.

---

## 3. Cornière longue : 2,00 m, trois ancrages, charges localisées

### 3.1 Local et longitudinal

- **Comportement local** (méthode du classeur) : une tranche de cornière, les deux ailes en flexion transversale, l'équilibre R / ancrage / compression du mur. Il suppose que la charge, l'ancrage et la compression agissent **dans la même tranche**, sur la même longueur L.
- **Comportement longitudinal** : dès que les ancrages sont espacés et que la charge n'est pas au droit d'un ancrage, la cornière transporte la charge le long de z : **flexion** dans le plan vertical (vers les cisaillements d'ancrage) et **torsion** (le couple R·(X' − u_S) doit rejoindre les couples ancrage + contact, qui n'existent qu'aux ancrages). Une cornière est une section ouverte : **très raide en flexion, très souple en torsion** (G·It = 21,4 kN·m² contre E·I = 934 kN·m² pour la L120×120×15) [e03 §1].

### 3.2 La méthode proposée — poutre sur appuis élastiques avec torsion

Modèle `e03_longitudinal.py` (numpy, quelques centaines d'éléments) :

- **Inconnues** le long de z : U, V (translations du centre de cisaillement S, à l'intersection des lignes moyennes), φ (rotation) ; flexion biaxiale avec le produit d'inertie (Iy = Iz = 444,9 cm⁴, Iyz = −260,2 cm⁴) et torsion de Saint-Venant (It = 26,43 cm⁴), **lus dans le catalogue du site** ; gauchissement d'une cornière négligé.
- **Ancrage i** : tirant horizontal à la hauteur Y, raideur k_N en **série** avec la flexion locale de l'aile verticale k_pl ; appui vertical k_V.
- **Mur** : contact unilatéral continu au pivot (arête inférieure), et en tête d'aile (pour l'effet de levier) ; itérations jusqu'à stabilité des zones de contact.
- **Charges** : chaque charge = effort vertical + couple R·(X' − u_S), ponctuelle ou répartie sur sa largeur réelle.
- **Raideur locale** : longueur efficace d'un tronçon en T (EN 1993-1-8 §6.2.4) avec m = (Y − e) − 0,8·r = 59,6 mm : min(2πm ; 4m + 1,25·e_bord) = **282,1 mm** ; k_pl = 3EI/(Y − y_P)³ = 81,4 kN/mm [e03 §2].

**Hypothèses simplificatrices indispensables** (à afficher dans la note) :

1. section **indéformable** dans la poutre longitudinale ; la déformation de l'aile autour de l'ancrage est remplacée par k_pl en série ;
2. pivot sur l'arête inférieure, frottement mur–cornière **nul** (tout le cisaillement va aux ancrages) ;
3. raideurs d'ancrage **déclarées** (déplacements sous charge de service de l'ETA de la cheville retenue) — le résultat en dépend ;
4. **jeu des trous de passage** ignoré : avec du jeu, le cisaillement va aux ancrages dont le trou porte d'abord ; l'EN 1992-4 (§6.2.2, à vérifier dans le texte) limite les ancrages réputés travailler en cisaillement selon ce jeu ;
5. charge appliquée à position fixe (X') : le report de l'appui vers le talon quand la cornière tourne n'est pas compté (côté sûr) ;
6. matériau élastique, petits déplacements ; la combinaison ELU se calcule **chargée** (le contact n'est pas linéaire, §4.3).

Quand ces hypothèses ne suffisent pas (charge loin des ancrages, aile mince, torsion > ≈ 50 % de l'admissible), **une plaque (coque) des deux ailes** est la vérification de référence — hors du module dans un premier temps.

### 3.3 L'exemple chiffré — CALCULÉ (raideurs = HYPOTHÈSES)

Cornière L120×120×15, L = 2 000 mm, ancrages à z = 200, 1 000, 1 800 mm (bords 200, entraxe 800 — HYPOTHÈSE), Y = 85, X = 75, R = 42,76 kN répartis sur 300 mm centrés à z = 400 (la largeur du classeur). Référence « R/n » : **V = 14,25 kN** et **N = H/3 = 11,32 kN** par ancrage (H = 33,96 kN, corps libre P0) ; le classeur donnerait N = V = 14,25 kN [e03 §3].

| Ancrages | N1 | N2 | N3 (kN) | V1 | V2 | V3 (kN) | N1/(H/3) | V1/(R/3) | couple max (kN·m) | τ torsion max | σz max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rigides | 25,16 | 8,80 | 0,00 | 29,93 | 14,97 | **−2,14** | **2,22** | 2,10 | 1,970 | 111,8 | 85,0 |
| raides (k_N = 100, k_V = 30 kN/mm) + aile | 24,04 | 9,55 | 0,37 | 29,99 | 14,85 | −2,08 | 2,12 | 2,10 | 1,879 | 106,7 | 84,5 |
| souples (10 et 5 kN/mm) + aile | 20,87 | 10,28 | 2,81 | 30,16 | 14,51 | −1,91 | 1,84 | 2,12 | 1,621 | 92,0 | 84,7 |
| cornière 1 000 fois plus raide, ancrages souples | 11,35 | 11,32 | 11,29 | 30,29 | 14,25 | −1,78 | **1,00** | 2,12 | — | — | — |
| rigides, **avec effet de levier** (contact en tête 20,5 kN) | **41,52** | 21,38 | 0,00 | 29,69 | 15,45 | −2,38 | **3,67** | 2,08 | 1,965 | 111,5 | 81,1 |

(τ et σz en N/mm² ; contrôles d'équilibre : ΣV = 42,760 kN, ΣN = ΣC à 10⁻⁴ kN.) Figure `e03_longitudinal.png` : tractions et cisaillements par ancrage, couple de torsion et moment longitudinal le long de la cornière.

**Lecture.**

- **Cisaillement** : la cornière est beaucoup plus raide en flexion que les chevilles en cisaillement ; la répartition suit la **règle du levier d'une poutre rigide** : V_i = R/n + R·e_z·z_i/Σz_i² = 30,3 / 14,25 / −1,78 kN — quelle que soit la raideur des ancrages (V1 = 29,7 à 30,3 kN). « R/n » n'est juste que si la charge est **au centre de gravité du groupe d'ancrages**.
- **Traction** : elle dépend du rapport entre la raideur en rotation d'un ancrage, k_θ = k_N,eff·(Y − y_P)², et la raideur de torsion d'une travée, G·It/s. Ici k_θ·s/(G·It) = **2,4 à 12,1** : la cornière est trop souple en torsion pour répartir ; le couple va surtout à l'ancrage voisin (74 % / 26 % / 0 avec ancrages rigides, partage G·It/0,2 m contre G·It/0,6 m). « R/n » sur la traction suppose une cornière **infiniment raide en torsion** (dernière ligne sans levier : 11,35 / 11,32 / 11,29).
- **Torsion** : τ = T·e/It = 92 à 112 N/mm², soit **67 à 82 % de τ_adm** (136,6 N/mm²) entre la charge et l'ancrage voisin. Le classeur ne le voit pas.
- **Flexion longitudinale** : ≈ 85 N/mm² aux fibres extrêmes, à cumuler avec la flexion transversale locale (biaxial).
- **Un ancrage au droit de la charge** (variante 400 / 1 000 / 1 600 mm) ne supprime pas la torsion (73 à 74 N/mm² : la charge de 300 mm de large rejoint un ancrage ponctuel) ; il concentre l'effort : N = 23,2 à 32,1 kN et V = 36,4 à 42,7 kN sur cet ancrage [e03 §5].

### 3.4 Quand « R/n » est-il faux ?

| Effort | R/n est acceptable si… | sinon |
|---|---|---|
| cisaillement V | la charge est au centre du groupe d'ancrages | règle du levier (poutre rigide), jeu des trous à déclarer |
| traction N | G·It/s ≫ k_θ (cornière raide en torsion devant les ancrages) — rarement vrai pour une cornière | poutre + torsion sur appuis élastiques (§3.2) |
| vérification locale | la largeur L de la tranche est celle qui porte réellement | **largeur efficace autour de chaque ancrage** (≈ 282 mm ici), effort de l'ancrage le plus chargé |

Conséquence locale [e03 §4] : aile verticale au droit de l'ancrage le plus chargé, sur l_eff = 282 mm : σ ≈ 120 (souples) à 143 N/mm² (rigides), soit 51 à 60 % ; **231 N/mm² (97 %) avec effet de levier**. Le cas de contrôle (3 ancrages sur 300 mm, charge au droit) donnait 88,86 %.

---

## 4. ELS et ELU : la cornière, les ancrages, Hilti

### 4.1 La cornière : méthode historique conservée — CERTAIN

Le classeur compare σ_éq(ELS non majoré) à σ_adm = f_y/1,50 (`H31` = `C3/C10`) et τ à σ_adm/√3 (`H34`). Le « gM » de `C10` est un **coefficient de sécurité global** de méthode aux contraintes admissibles, pas le γ_M0 de l'EN 1993-1-1. Le module doit l'afficher ainsi (« contraintes admissibles, coefficient global 1,50 sur f_y, sollicitations ELS ») et ne pas l'appeler vérification Eurocode 3.

### 4.2 Le passage ELS → ELU du classeur — CERTAIN puis INTERPRÉTÉ

- `C85` = (`C71`/`C70`)·`C82` et `C86` = (`C71`/`C70`)·`C83` : N et V multipliés par **68,9/42,76 = 1,61132** → 22,967 kN chacun. `G85` = 24,68 et `G86` = 22,97 sont des valeurs **tapées** (probablement des résultats PROFIS).
- Avec γG = 1,35 et γQ = 1,50 sur les **mêmes** charges caractéristiques, le rapport ELU/ELS varie de 1,35 (tout permanent) à 1,50 (tout variable). **1,611 est impossible** : 68,9 kN ne vient pas de 42,76 kN par l'EN 1990 (6.10). Il faudrait que 42,76 soit une combinaison fréquente (G/Q = 2,66) ou quasi permanente (G/Q = 3,89), ou que les deux réactions viennent de modèles différents [e04 §1].
- **Feuille E10A BD** (G = `D4:D11`, Q = `G4:G11`, kN par appui et par mètre ; diviseur `D3` = 1,345 m, `F2` = 4 kN/m²) : pour la volée 12, **1,35·G + 1,5·Q = 42,768 kN/m**, à **+0,018 %** de la « réaction max (ELS) » `C12` = 42,76 [e04 §2]. Coïncidence ou réaction ELU par mètre prise pour une ELS ? **Le classeur ne permet pas de trancher** (`A12` renvoie à « p. 15 » d'un autre document). À poser à l'auteur.
- `G85` = 24,68 kN correspond, avec R_ELU = 68,9 kN, X = 75 mm et trois ancrages, à un bras z = 69,79 mm : compression à **15,21 mm** au-dessus du dessous de la cornière, ≈ e (INTERPRÉTÉ : PROFIS aurait modélisé la platine comme l'aile verticale **au-dessus** de l'aile horizontale).

### 4.3 Proposition : des cas de charge, des combinaisons, un calcul par combinaison

- Les charges sont saisies **par cas** (G, Q, éventuellement d'autres), chacun avec sa position X, sa position le long de la cornière et sa largeur ; les combinaisons sont **définies** (ELS caractéristique pour la méthode historique ; ELU EN 1990 (6.10), ou (6.10a/b) selon l'annexe nationale belge retenue au projet — coefficients affichés).
- Chaque combinaison est **calculée chargée** : le contact unilatéral rend le modèle non linéaire. Écart de superposition mesuré sur l'exemple long : jusqu'à 0,22 kN par ancrage [e04 §4].
- Exemple long (HYPOTHÈSE d'illustration : G = 30 kN répartis sur les 2,00 m, Q = 30 kN sur 300 mm centrés à 0,40 m, ancrages raides) :

| Ancrage | N_ELS (G+Q) | N_ELU (1,35G + 1,5Q) | coef. global 1,425 × N_ELS | coef. du classeur 1,611 × N_ELS | V_ELS | V_ELU |
|---|---|---|---|---|---|---|
| 1 (z = 200) | 24,26 | **35,30** | 34,58 (−2,0 %) | 39,10 (+10,8 %) | 29,72 | 43,26 |
| 2 (z = 1 000) | 15,74 | 22,23 | 22,43 (+0,9 %) | 25,36 | 23,07 | 32,72 |
| 3 (z = 1 800) | 7,64 | 10,37 | 10,89 (+5,0 %) | 12,31 | 7,22 | 9,51 |

Une **platine rigide de 2 m** (répartition égale des tractions, hypothèse d'une platine unique supposée rigide) donnerait 22,63 kN par ancrage : l'ancrage le plus chargé est **56 % au-dessus**.

### 4.4 Ce que le module donne à Hilti PROFIS

- **Par ancrage**, ELS et ELU séparés, par combinaison : **N** (traction, kN) et **V** (cisaillement vertical, signe et sens écrits — un ancrage peut être cisaillé vers le haut), position (z le long de la cornière, hauteur Y au-dessus du dessous de la cornière, distance au bord libre de l'aile), et les hypothèses de répartition (pivot, raideurs, jeu des trous, frottement nul, effet de levier exclu ou compté).
- **Ou les actions sur la platine** pour une tranche courte où PROFIS peut répartir lui-même : cas de contrôle à l'ELU, V = 68,9 kN vers le bas, M = V·X = 5,168 kN·m autour de l'axe horizontal du mur (bras depuis la face du béton), N = 0 [e04 §3]. PROFIS calcule alors sa propre zone comprimée (hypothèse de platine rigide de l'EN 1992-4 ; à vérifier dans la version utilisée).
- **Cornière longue** : ne pas la modéliser comme **une seule platine rigide** de 2 m (répartition égale des tractions, −36 % sur l'ancrage le plus chargé dans l'exemple) ; vérifier l'ancrage le plus sollicité avec les efforts du modèle longitudinal.
- Le module **ne vérifie pas** les chevilles (cône, bord, effet de levier côté béton, épaisseur du support) : il le dit dans le résultat et dans la note.

| Cas de contrôle (coupe de 300 mm, 3 ancrages) | N_ELS | V_ELS | N_ELU | V_ELU (kN par ancrage) | ELU |
|---|---|---|---|---|---|
| classeur (bras Y − 1 cm) | 14,253 | 14,253 | 22,967 | 22,967 | ×68,9/42,76 |
| corps libre P0 (maillage), V sur l'axe | 11,319 | 14,253 | 18,238 | 22,967 | ×68,9/42,76 |
| idem, 1,35·G + 1,5·Q avec le G/Q de la volée 12 (HYPOTHÈSE : G = 30,03 ; Q = 12,73 kN) | 11,319 | 14,253 | 15,786 | 19,879 | R_ELU = 59,64 kN |

---

## 5. Maillage de la coupe : prototype et comparaison

### 5.1 Le prototype — `ef_l.py`, numpy seul

- Tranche de 1 mm, **déformation plane** (cornière longue chargée uniformément), E = 210 000, ν = 0,3.
- **Maillage structuré par blocs** (interpolation transfinie) : deux ailes, deux blocs d'angle séparés par la diagonale, **congé exact**, éléments quadrilatères à 9 nœuds, Gauss 3×3. Maillage fin : 4 725 nœuds, 1 120 éléments, 9 450 inconnues.
- **Appuis** : mur = ressorts **unilatéraux** sur toute la face extérieure de l'aile verticale (raide 10⁵ N/mm³ ou souple 10³) ; ancrage = ressorts unilatéraux sous l'écrou (± 3 mm, ou rondelle Ø30) de raideur déclarée + appui vertical au nœud de mi-épaisseur ; charge q = 42 760/300 N/mm répartie sur 10 mm autour de X.
- **Solveur** : numérotation de Cuthill–McKee inverse (demi-largeur de bande 169), Cholesky bande **par panneaux** (numpy.linalg sur des blocs de 64), itérations de contact au départ de la zone d'arête (5 à 6 itérations).
- **Convergence** : pic de surface 238,56 (1 469 nœuds) contre 236,41 N/mm² (4 725 nœuds), **0,90 %** [e05 §2].

### 5.2 Faisable dans le navigateur ? — MESURÉ

| Environnement | 1 469 nœuds | 4 725 nœuds | Remarque |
|---|---|---|---|
| CPython 3.12, un fil | 0,34 s | 1,84 à 1,97 s | 5 et 6 itérations de contact ; sans le départ « arête » : 13–14 itérations, 0,73 et 4,14 s |
| **Pyodide 0.29.5 + numpy 2.2.5 (vendus par le site), Node 22** | **0,79 s** | **4,55 s** | chargement Pyodide + numpy 1,71 s ; H = 33,897 kN, identique à CPython |

Mesuré sous Node (même moteur V8 que Chromium), pas dans un onglet. **Conclusion** : quelques milliers de nœuds tiennent dans le navigateur sans scipy. Un maillage de ≈ 1 500 nœuds répond en moins d'une seconde (convergé à 1 %) ; le maillage fin doit tourner dans un *worker* pour ne pas figer l'écran. Pistes chiffrables plus tard : une seule factorisation et la formule de Woodbury pour les changements de contact.

### 5.3 Ce que le maillage confirme ou contredit — CALCULÉ

Cas de référence **A-ref** : ancrage 30 kN/mm par cheville (3 sur 300 mm), mur raide.

| Section | Classeur M (kN·m) | Classeur σ | Maillage M | Maillage N (kN) | σ poutre avec M, N du maillage | **Maillage, fibre intérieure** | fibre extérieure | fibre / poutre |
|---|---|---|---|---|---|---|---|---|
| 1 | 2,0097 | 178,64 | 2,0099 | 0,12 | 178,69 | **200,73** | −174,88 | 1,123 |
| 2 | 2,5656 | 65,45 | 2,5828 | 1,20 | 66,03 | **1,71** | −124,70 | 0,026 |
| 3 | 2,3587 | 209,66 | 1,9374 | 42,87 | 181,74 | **203,64** | −159,12 | 1,121 |
| 4 | 2,8966 | 73,89 | 2,1166 | 43,93 | 59,22 | **1,64** | −97,47 | 0,028 |

**Confirmé**

- Moment de l'aile horizontale : statique exacte (section 1 : +0,034 % ; section 2 : +0,69 %, coupe à travers l'angle).
- Effort normal de l'aile verticale = R en traction (42,87 kN) ; tranchant = H.
- Pivot à l'arête : contact à y = 0,00 mm (mur raide), 1,86 mm (mur souple) ; H = 33,897 / 34,654 kN, conforme au corps libre P0. M3 maillé = 1,937 / 1,981 kN·m, soit 18 et 16 % sous le classeur.
- Configuration B : §2.1.

**Contredit**

- La distribution **linéaire** des contraintes dans les coupes « e + r » (2 et 4) : le haut de ces coupes est le pied du congé de l'AUTRE aile, presque sans contrainte (1,7 N/mm²) ; le bas atteint 1,9 fois la valeur poutre (−124,7 contre ±66). Elles restent loin de gouverner.
- La contrainte de fibre aux sections 1 et 3 dépasse de **12 %** la contrainte poutre (pied du congé).

**Ajouté**

- Pic **local** à la surface du congé : **236,42 N/mm² à 2,92 mm dans l'arc** côté aile horizontale (la poutre donne 181,66 au même point : rapport **1,30**) ; 234,94 N/mm² côté aile verticale (1,28 sur la poutre avec les efforts du maillage). Von Mises (déformation plane) maximal : 207,14 N/mm², au même endroit. Figures `e05_surface_interieure.png`, `e05_ef_zoom_angle.png`.
- **Lecture** : c'est une **contrainte de pointe** (concentration au raccord), pas une contrainte nominale. La méthode historique, comme l'EN 1993 en statique, travaille en contraintes **nominales** ; le pic n'a pas à être comparé à σ_adm en statique (l'acier s'adapte localement), mais il compterait en **fatigue**. Le module peut l'afficher comme information, sans en faire un critère.

### 5.4 L'effet de levier — découverte du maillage

| Raideur axiale d'une cheville (3 sur 300 mm, mur raide) | 30 | 100 | 200 | **300** | 500 | 1 000 | 10 000 | 100 000 kN/mm |
|---|---|---|---|---|---|---|---|---|
| traction totale H (kN) | 33,53 | 33,54 | 33,54 | **36,27** | 44,30 | 53,43 | 76,17 | 92,67 |
| contact en tête d'aile (kN) | 0 | 0 | 0 | **1,95** | 7,71 | 14,30 | 32,92 | 48,41 |

(maillage de 1 469 nœuds [e05 §7] ; maillage fin, ancrage infiniment raide : **H = 95,78 kN**, dont 50,20 kN de contact au-dessus de l'ancrage.) Avec une rondelle Ø30 (mur raide), le seuil reste entre 100 et 300 kN/mm par cheville ; avec un mur souple (10³ N/mm³), il monte entre 300 et 1 000 kN/mm [e05b]. Figure `e05_effet_levier.png`.

- **INTERPRÉTÉ** : si l'ancrage est plus raide que l'aile, la partie d'aile au-dessus de l'ancrage bascule contre le mur et fait levier ; la traction croît sans que la charge change. Le classeur (comme le corps libre) l'ignore ; ses 1 cm de « jeu » (`C76`) ne le couvrent pas. Les chevilles courantes (déplacements de service de l'ordre du demi-millimètre, raideurs ≤ 100 kN/mm) en sont loin ; une cheville **précontrainte** reste très raide tant que l'assemblage n'est pas décomprimé — le cas mérite un avertissement et, si la raideur est inconnue, un calcul avec et sans levier.

### 5.5 Ce que le maillage doit être dans le module — proposition

Une **vue de contrôle** (déformée et couleurs : `e05_ef_vonmises.png`), avec ses hypothèses écrites (raideur d'ancrage, mur, déformation plane) et ses grandeurs de synthèse (pivot trouvé, H, contact en tête), à côté de la méthode historique — pas un remplaçant. Couleurs : une seule teinte claire → foncée pour von Mises ; deux teintes (bleu compression, rouge traction) autour d'un gris neutre pour la contrainte principale.

---

## 6. Architecture du module

### 6.1 Le point de départ

`acier/corniere/moteur.py` reproduit l'écran Streamlit : modèle « lame », M = V·e au nu du mur, épaisseur e, ni congé, ni aile verticale, ni ancrage (`docs/acier/CORNIERE_MOTEUR.md` §3). Sur le cas de contrôle : σ = 285,07 N/mm², **U_σ = 120,65 %**, contre 88,86 % au classeur [e07 §1]. Le nouveau calcul est un autre modèle ; l'ancien reste disponible tant que la parité Streamlit l'exige.

### 6.2 Découpage proposé (une responsabilité par fichier, numpy seul dans le navigateur)

| Fichier | Responsabilité |
|---|---|
| `acier/corniere/section.py` | la section : catalogue (`acier/profiles`, familles L et LI, r1/r2, It, Iyz) ou PRS ; **orientation** (aile contre le mur) ; fonction t(s) de chaque aile (congé ou cordon) |
| `acier/corniere/coupe.py` | corps libre A et B (pivot, u_V, jeux Xj, Yj, position réelle de l'ancrage et du contact) ; M, N, V continus ; 4 sections ; balayage et maxima |
| `acier/corniere/historique.py` | la méthode du classeur mot pour mot (M1..M4, C82/C83) ; test de parité sur le cas de contrôle et les 46 000 comparaisons de la reproduction |
| `acier/corniere/longitudinal.py` | poutre + torsion sur appuis élastiques (§3.2), charges par cas, combinaisons calculées chargées |
| `acier/corniere/ancrages.py` | N et V par ancrage, ELS et ELU, positions, hypothèses ; export pour PROFIS |
| `acier/corniere/ef.py` | le maillage de contrôle (`ef_l.py`), lancé à la demande, dans un *worker* côté site |
| `acier/corniere/note.py` | la note A4 paysage sur `ndc_pdf` |

Un même dictionnaire de résultats nourrit l'écran PC, tablette et téléphone (composants partagés), la fenêtre de calcul détaillé (formule symbolique, valeurs substituées, résultat, admissible, taux à droite) et la note.

### 6.3 Réutiliser le catalogue — vérifié

Les fonctions partagées `acier/profiles/geometrie.py` redonnent la fiche de la L120×120×15 : `contour_L(120, 120, 15, 13, 6.5)` → A = 33,932 cm², Iy = 444,88 cm⁴, Iyz = −260,22 cm⁴, yc = 35,15 mm, `torsion_saint_venant` → It = 26,43 cm⁴ (fiche : 33,93 ; 444,88 ; −260,22 ; 35,15 ; 26,43) [e07 §2]. Pas de seconde base : le catalogue du classeur (rayons de bout d'aile pris pour des rayons de congé, orientations en double) est remplacé par celui du site, l'orientation devient un paramètre.

### 6.4 Préparer les cornières PRS (deux plats soudés)

- **Géométrie** : plat horizontal (largeur, t_h), plat vertical (hauteur, t_v), cordon(s) intérieur et/ou extérieur de gorge a ; propriétés par les mêmes fonctions partagées. Exemple 120×120, t_h = 15, t_v = 12, cordon intérieur a = 6 : A = 30,960 cm², Iy = 386,71, Iz = 435,53, Iyz = −238,98 cm⁴, It = 20,19 cm⁴ (deux plats minces séparés : 19,55) [e07 §2].
- **t(s)** : le congé devient un **triangle** de côté a·√2 (t passe de t_h + a√2 à t_h sur la longueur du cordon) [e07 §3].
- **Les quatre sections se transposent** : 1 = pied du cordon sur l'aile horizontale, 2 = nu du plat vertical, 3 = pied du cordon sur le plat vertical, **4 = interface des deux plats**, qui n'est plus une section d'acier continue : elle se vérifie comme un **assemblage** (cordon + contact), avec le moteur `acier/soudures` du site. En configuration A le cordon intérieur est tendu par le moment d'angle ; en B il est comprimé et le contact extérieur tendu — à traiter avant d'ouvrir la PRS.

### 6.5 Les données d'entrée (proposées)

Section (catalogue / libre / PRS), orientation (A ou B, grande aile), acier ; longueur totale, ancrages (positions z, hauteur, diamètre et trou, raideurs N et V si connues), distances aux extrémités ; charges par cas (valeur, X, Xj, position z, largeur) ; jeux Xj (position de la charge), Yj (position de l'ancrage) et hauteur du pivot **distincts** ; combinaisons ELS et ELU ; mode (historique / équilibre / longitudinal / maillage).

### 6.6 Plan de la note A4 paysage (kit `ndc_pdf`, comme Garde-corps et Soudures)

| Page | Contenu | Sections demandées |
|---|---|---|
| 1 | cartouche (chantier, cornière, indice) ; hypothèses et méthode (mode, coefficient global 1,50, pivot, raideurs, frottement nul) ; coupe cotée à l'échelle (mur, ailes, congé, charge, jeux, ancrage, contact, bras de levier, sections 1–4) ; tableau des charges par cas et combinaisons ELS/ELU | 1, 2, 3, 4 |
| 2 | corps libre dessiné et réactions (H, V, C, position du contact) ; diagrammes M, N, V des deux ailes avec sections et maxima ; contraintes et taux le long des ailes ; tableau des 4 sections et du maximum du congé (formule, substitution, résultat, admissible, taux à droite) | 5, 6, 7 |
| 3 | élévation (longueur, ancrages, entraxes, bords, charges et largeurs) ; diagrammes longitudinaux (répartition N et V, torsion, flexion) ; tableau des efforts par ancrage ELS/ELU, N et V séparés, positions, hypothèses ; synthèse des taux ; conclusion et limites (chevilles non vérifiées, effet de levier, nominal/pointe) | 8, 9, 10 |
| 4 (option) | annexe maillage : déformée, carte de contraintes, pivot et levier trouvés | — |

---

## 7. Certain, interprété, à trancher

**CERTAIN** (formules et valeurs du classeur) : M(u) et M(y) linéaires (`C35`, `C45`, `C55`·`C56`, `C63`·`C64`) ; H implicite 41,381 kN ; deux pivots différents (e/2 pour les sections, 1 cm pour les ancrages `C76`) ; τ des sections 3–4 calculé avec R ; ELU = ×1,61132 (`C85`, `C86`) ; 15 noms en double et VLOOKUP sur la première ligne ; r = 6 pour L150x100x12.

**INTERPRÉTÉ** : le classeur majore deux fois l'équilibre de l'aile verticale (V au nu du mur pour H, sur l'axe pour M) ; « τ » des sections 3–4 = effort normal ; le pivot réel est à l'arête (maillage) ; 24,68 kN de PROFIS ↔ compression à 15,2 mm ; 42,76 kN pourrait être une valeur ELU par mètre.

**CALCULÉ avec des HYPOTHÈSES** : raideurs d'ancrage (10 à 100 kN/mm, rigides), positions des ancrages de l'exemple long, partage G/Q, l_eff du tronçon en T, mur raide ou souple, déformation plane.

---

## 8. Doutes et points ouverts

1. Nature de `C12` = 42,76 kN : ELS (libellé) ou ELU par mètre (coïncidence à 0,018 % avec la volée 12 de E10A BD) ? Et origine de `C71` = 68,9 kN (rapport 1,611 > 1,50).
2. Raideur réelle des chevilles (et précontrainte) : elle décide de l'effet de levier (seuil ≈ 200 à 300 kN/mm par cheville dans le cas de contrôle) et de la répartition longitudinale.
3. Jeu des trous de passage : non modélisé ; la règle de l'EN 1992-4 sur les ancrages réputés cisaillés est à relire avant de l'écrire dans le moteur.
4. Torsion de la cornière longue : le modèle poutre (Saint-Venant, section indéformable) est une première approche ; une coque des deux ailes devrait confirmer les 92 à 112 N/mm² avant d'en faire un critère.
5. Le pic de 236 N/mm² au congé est une contrainte locale : à afficher, à ne pas vérifier en statique — sauf décision contraire du bureau (fatigue, chocs).
6. Temps navigateur mesurés sous Node, pas dans Chromium ; à confirmer dans un onglet avec un *worker*.
7. Le report de l'appui de l'escalier vers le talon quand la cornière tourne (charge à position variable) n'est pas compté : côté sûr, mais il peut changer beaucoup la torsion d'une cornière longue.
