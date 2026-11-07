# Le classeur « Dimensionnement des cornières » — analyse technique critique

*Bureau d'études — analyse du 07/11/2026, faite **avant** toute modification du module Cornières de Stab Méthodes.*
*Classeur : « REALEX Dimensionnement corniereS E10A#0 02-04-26.xlsx » (en-tête imprimé : Escaliers E10A, paliers 02/05/08, indice 0), feuilles `P8`, `E10A BD` et `CORNIERES`.*

**Cas de contrôle** : L120×120×15, r = 13 mm, S355, longueur reprise 300 mm, R = 42,76 kN (ELS), X = 75 mm, Y = 85 mm, Xj = Yj = 0, coefficient 1,50.

> **Comment lire ce document**
>
> - Chaque chiffre sort d'un script Python lancé sur le classeur ; aucun n'est estimé. Le renvoi entre crochets, par exemple **[s02 §4]**, donne le script et la rubrique de sa sortie (liste complète en annexe D).
> - **Certain** : lu dans une formule, une valeur ou la structure du classeur ; la cellule est citée (`P8!C35`).
> - **Interprété** : sens physique qu'on en déduit ; c'est une lecture d'ingénieur, à confirmer par l'auteur de la note.
> - **Hypothèse** : valeur choisie pour chiffrer un exemple (raideur de cheville, positions d'ancrages…), dite comme telle.
> - Repère : **u** = distance horizontale depuis le nu du mur (dos de l'aile verticale) ; **y** = hauteur depuis le dessous de la cornière ; X' = X + Xj ; Y' = Y + Yj ; L = longueur reprise = 300 mm.
> - Ce document réunit et départage trois analyses faites séparément (mécanique, critique, extensions). Là où elles divergeaient, les chiffres ont été recalculés ici et l'écart est expliqué (annexe D.3).

---

## 0. L'essentiel en une page

**Ce que fait le classeur (certain).** La feuille `P8` étudie une tranche de cornière de 300 mm (`C8` = 30 cm) qui reçoit la réaction ELS d'un escalier (`C12` = 42,76 kN), posée sur l'aile horizontale à X = 75 mm du mur. Elle calcule quatre moments en quatre sections — deux par aile, au pied du congé (épaisseur e) et au nu de l'autre aile (épaisseur e + r) —, puis σ = M/W, τ = R/(L·t), σ_éq = √(σ² + 3τ²), et compare à σ_adm = f_y/1,5. Elle donne ensuite la traction et le cisaillement de chaque ancrage à l'ELS, puis à l'ELU en multipliant par 68,9/42,76.

**Ce qui est juste.**

- Le cas de contrôle est **reproduit exactement** : M1 = 2,009720 ; M2 = 2,565600 ; M3 = 2,358697 ; M4 = 2,896645 kN·m ; taux maximal **88,8621 %** en section 3. Les 93 formules numériques de `P8`, recalculées telles qu'elles sont écrites, redonnent les valeurs enregistrées sans écart ; la méthode réécrite à partir des seuls symboles les redonne à 2,5·10⁻¹⁶ près, et sur 592 saisies tirées au hasard (54 464 comparaisons) à 2,0·10⁻¹⁵ près [s01 §2–4].
- L'**aile horizontale** suit la statique exacte d'un porte-à-faux : le maillage retrouve M1 à 0,01 % près [s07].
- La **contrainte équivalente** de von Mises et la méthode aux **contraintes admissibles** sont cohérentes : f_y/1,5 sur des efforts ELS équivaut exactement à l'EC3 élastique avec un coefficient 1,50 sur toutes les actions [s02 §8].
- Dans la configuration du classeur, l'**aile verticale est du côté sûr** : 88,86 % contre 76,91 % à 83,96 % pour un équilibre cohérent, et M3 dépasse de 22 % le moment trouvé par le maillage [s02 §5, s07].

**Ce qui doit être vérifié ou corrigé.**

1. **Le jeu par défaut de 20 mm fait échouer le cas validé** : appliqué à Xj, le taux passe de 88,86 % à **112,43 %** ; 100 % est atteint dès Xj = 9,455 mm. Les trois « jeux » du classeur (`C14`, `C17`, `C76`) n'agissent pas sur les mêmes grandeurs et ne doivent pas être fusionnés [s02 §4].
2. **Deux pivots différents dans la même feuille** : e/2 = 7,5 mm pour les sections (H = 41,38 kN), 1 cm pour les ancrages (H = 42,76 kN). La valeur Hilti tapée en `G85` suppose 15,21 mm. Un seul équilibre doit être écrit [s02 §5].
3. **Dans l'aile verticale, le « τ » du classeur est un effort normal** (traction R) ; le vrai tranchant est H. Les deux erreurs de l'aile verticale se compensent ; le classeur cesse d'être du côté sûr si l'ancrage est bas (Y < 33 à 43 mm) ou si le support est très souple : sa marge sur la section 3 tombe de 13,3 % (mur raide) à 3,3 % (module de réaction 30 N/mm³) et s'annule vers 16 N/mm³ [s02 §5, s07].
4. **Le maximum n'est pas exactement aux sections 1 et 3** : il est ~2 mm dans le congé (+2,11 % et +1,45 % ici, mais jusqu'à ×1,27 quand la charge est à 20 mm du pied du congé). Les sections « e + r » ne gouvernent que si X + Xj < 30,57 mm (aile horizontale) ou Y + Yj < 33,02 mm (aile verticale, à X = 75 mm) avec le critère σ_éq du classeur — 33,23 mm en flexion seule —, et le maillage montre que leur contrainte n'est pas linéaire [s03, s07].
5. **L'ELU n'est pas une combinaison** : 68,9/42,76 = 1,611 > 1,50, ce qui est impossible avec 1,35·G + 1,5·Q appliqués aux mêmes charges et K_FI = 1 (avec K_FI = 1,1, classe CC3, le plafond monte à 1,650, mais la volée 12 donnerait 1,534, pas 1,611). Coïncidence troublante : 1,35·G + 1,5·Q de la volée 12 vaut 42,768 kN/m, soit `C12` à 0,018 % près [s06].
6. **Aucun contrôle de domaine** : une charge hors de l'aile est calculée à 155,9 %, un ancrage dans le congé donne M3 < 0, « 20 » tapé en `C76` (case en cm) donne une traction de −9,30 kN, sans alerte [s02 §4, §7].
7. **Catalogue** : 15 orientations de cornières inégales sont inaccessibles (RECHERCHEV lit la première ligne) ; la L150×100×12 porte le rayon de bout d'aile (6) au lieu du congé (12) [s08].
8. **La cornière retournée ne relève pas de ces formules** : M3 = 2,886 kN·m constant de l'ancrage à l'angle, taux 112,42 %, traction ×2,43 [s04].
9. **Sur une cornière longue, « R/n » est faux** : l'ancrage le plus chargé reçoit 2,22 fois H/3 en traction et 2,10 fois R/3 en cisaillement ; la torsion atteint 82 % de τ_adm [s05].
10. **Le module actuel du site ne suit pas le classeur** : 120,65 % sur le cas de contrôle si l'on y saisit σ_lim,ELS = f_y/1,5 = 236,67 N/mm² ; avec ses valeurs par défaut (σ_lim,ELS = 155, τ_lim,ELS = 0,8, σ_eq,lim,ELU = 186 N/mm², γ_ELU = 1,35), 184,22 % à l'ELS, 207,2 % à l'ELU et 1 187,8 % en cisaillement (modèle de bande d'acier, ni congé, ni aile verticale, ni ancrage), et sa vérification « longitudinale » répète la transversale [s02 §9].

**Ce qu'on propose** (détail au §8).

- Un mode **« méthode historique »** qui reproduit le classeur au bit près, cellule par cellule, avec le cas de contrôle comme test.
- Un mode **« équilibre »** : un seul corps libre, pivot et efforts nommés et dessinés, efforts normal et tranchant exacts, configurations A et B.
- Des **diagrammes continus** M, N, V, épaisseur, contrainte et taux, les quatre sections restant des repères, le maximum étant cherché partout.
- Quatre grandeurs de **jeu et de position** séparées, en mm, dessinées et appliquées du côté défavorable de chaque vérification.
- Un **modèle longitudinal** (poutre + torsion sur appuis élastiques) pour les cornières longues, des efforts par ancrage ELS et ELU **calculés par combinaison**, prêts pour Hilti PROFIS.
- Des contrôles de domaine, le **catalogue du site** (orientation séparée de la désignation), un maillage de contrôle affiché à titre d'information.

---

## 1. Le cheminement des efforts dans la cornière

![Figure a — coupe cotée du cas de contrôle](figures/fig_a_coupe_A.png)

*Figure a — Cas de contrôle à l'échelle : bras de levier (bleu), jeux, sections 1 à 4 (orange), ancrage, charge, résultante de compression (vert). Fichier source : `figures/fig_a_coupe_A.svg`.*

### 1.1 Le chemin, pas à pas

1. **La charge.** L'escalier pose sa réaction R = 42,76 kN sur la face supérieure de l'aile horizontale, à X = 75 mm du nu du mur, répartie sur la longueur reprise L = 300 mm (`C12`, `C13`, `C8`, certain).
2. **L'aile horizontale** travaille en **porte-à-faux** encastré dans l'angle. Effort tranchant V = R, effort normal nul ; le moment croît linéairement de 0 sous la charge à R·X' = 3,207 kN·m au nu du mur (`C35`, `C45`, `C55`, certain). La face côté congé est tendue.
3. **L'angle** fait tourner le moment d'une aile à l'autre. Les deux diagrammes du classeur valent Mmax = 3,207 kN·m au même point (`C55` = `C63` = `C78`) : ce sont les pastilles vertes du schéma de la note.
4. **L'aile verticale**, entre l'angle et l'ancrage, est **tendue** : l'angle « pend » à l'ancrage, qui reprend R vers le haut. Son effort tranchant est l'effort horizontal **H** du couple ancrage–mur, et son moment décroît linéairement jusqu'à **0 à l'ancrage** (`C56`·`C55`, certain pour la forme ; interprété pour les efforts N et V, que le classeur ne nomme pas).
5. **L'ancrage** reprend une **traction** H (vers le mur) et un **cisaillement** V = R (vers le haut), partagés entre n = 3 ancrages (`C82`, `C83`).
6. **Le mur** reprend une **compression** C = H au pied de l'aile verticale (le « pivot »). Le frottement contre le mur n'est pas compté : tout le cisaillement va aux ancrages (côté sûr pour eux).

### 1.2 L'équilibre d'ensemble — et ce que le classeur en fait

En isolant la cornière (frottement nul), avec le pivot à la hauteur y_P et la ligne d'action du cisaillement d'ancrage à la distance u_V du nu du mur :

- équilibre vertical : V = R ;
- équilibre horizontal : C = H ;
- moments autour du pivot : **H·(Y' − y_P) = R·(X' − u_V)** ;
- aile verticale, coupe à la hauteur y : **M(y) = H·(Y' − y) − R·(e/2 − u_V)**, **N = +R** (traction), **V = H**.

**Ce que font les formules (certain).** `C25` = Y − e/2 et `C56` = (Ye + Yj)/(YH + Yj) reviennent à prendre **H = R·X'/(Y' − e/2)** : pivot à **y_P = e/2** et cisaillement au **nu du mur** (u_V = 0). Mais le moment M3 = H·Ye correspond à u_V = **e/2** (cisaillement sur l'axe de l'aile). Les deux choix sont du côté sûr, mais ils ne décrivent pas le même équilibre. Au nœud des lignes moyennes (u = y = e/2), l'aile horizontale délivre R·(X − e/2) = 2,88630 kN·m quand l'aile verticale du classeur en reçoit 3,20700 : un saut de R·e/2 = 0,32070 kN·m (+11,1 %) [s02 §3].

**Pour les ancrages, le classeur prend un autre pivot (certain)** : `C82` = Mmax/((Y − jeu)·n), avec `C76` = 1 cm, soit **y_P = 10 mm**. La même cornière a donc deux bras de levier : 77,5 mm pour les sections, 75 mm pour les ancrages.

| Variante de l'équilibre | y_P (mm) | u_V (mm) | H (kN) | N par ancrage (kN) | M3 (kN·m) | Taux 3 |
|---|---|---|---|---|---|---|
| Classeur, sections (`C25`, `C57`) | 7,5 | 0 / 7,5 | 41,381 | 13,794 | 2,35870 | 88,86 % |
| Classeur, ancrages (`C82`) | 10 | 0 | 42,760 | 14,253 | 2,43732 | 91,81 % ¹ |
| Hilti (`G85` = 24,68 kN) | 15,21 | 0 | 45,950 | 15,317 | 2,61915 | 98,62 % ¹ |
| **P0** — pivot à l'arête, V sur l'axe | 0 | 7,5 | 33,956 | 11,319 | 1,93552 | **76,91 %** ² |
| P1 — nœud des lignes moyennes | 7,5 | 7,5 | 37,243 | 12,414 | 2,12283 | 83,96 % ² |
| Q1 — pivot e/2, V au nu du mur | 7,5 | 0 | 41,381 | 13,794 | 2,03800 | 80,84 % ² |
| P15 — pivot remonté à 15 mm | 15 | 7,5 | 41,233 | 13,744 | 2,35027 | 92,53 % ² |

¹ moment du classeur (M = H·Ye) avec le pivot indiqué, contraintes du classeur ; ² équilibre cohérent : σ = M/W + N/A, τ = H/A [s02 §5].

- Le pivot qui redonne **exactement** M3 et M4 du classeur avec le cisaillement sur l'axe vaut y_P* = Y − (X − e/2)(Y − e/2)/X = **15,25 mm** (≈ e). Ce n'est pas une constante physique : il dépend de X et de Y [s02 §5].
- La valeur Hilti `G85` = 24,68 kN (tapée, lue par aucune formule) dépasse `C85` de **7,46 %** ; elle correspond, avec 68,9 kN, X = 75 mm et trois ancrages, à un bras de 69,79 mm, donc à un pivot à 15,21 mm (interprété : PROFIS aurait mis la compression au bord inférieur d'une platine située au-dessus de l'aile horizontale).

### 1.3 Ce que dit le maillage de la coupe

Le prototype d'éléments finis de la coupe (numpy seul, 4 725 nœuds, contact unilatéral avec le mur ; figure e au §4.4) tranche la position du pivot pour un mur raide [s07] :

- la compression du mur est centrée à **y = 0,00 mm**, sur l'**arête inférieure** ; avec un mur souple (module de réaction 10³ N/mm³), à 1,86 mm ;
- la traction totale vaut **33,90 kN** (équilibre P0 : 33,96 kN) ; le classeur la dépasse de **26 %** pour les ancrages (42,76 kN) et de 22 % pour les sections (41,38 kN) ; M3 maillé = 1,937 kN·m, soit **−18 %** sous le classeur (M3 classeur / M3 maillé = 1,217) ;
- l'effort normal de l'aile verticale vaut 42,87 kN en traction (R = 42,76) : la lecture « l'angle pend à l'ancrage » est confirmée.

**Deux réserves** qui rendent la marge du classeur fragile :

- **Support souple** : quand le mur s'écrase, le pivot remonte et la section 3 se charge. Avec le modèle d'ancrage de ce document (effort de l'écrou sur ± 3 mm autour de l'axe), la contrainte nominale de la section 3 (|M|/W + N/A, résultantes du maillage) vaut [s07] :

| Module de réaction du mur (N/mm³) | raide (10⁵) | 1 000 | 300 | 100 | 30 | 20 | 10 | 5 |
|---|---|---|---|---|---|---|---|---|
| Pivot trouvé (mm) | 0,00 | 1,86 | 3,30 | 5,47 | 9,28 | 10,99 | 14,38 | 18,06 |
| σ3 nominale (N/mm²) | 181,7 | 185,6 | 188,7 | 193,6 | 202,8 | 207,1 | 214,3 | 220,1 |
| Écart au classeur (209,66) | −13,3 % | −11,5 % | −10,0 % | −7,7 % | −3,3 % | −1,2 % | +2,2 % | +5,0 % |

  **La marge diminue** : de mur raide à 30 N/mm³, la section 3 prend 11,6 % (8,9 points de σ_adm) et reste 3,3 % sous le classeur (2,9 % avec une rondelle élastique de ± 15 mm). Elle **s'annule vers 16 N/mm³** (interpolation), ce qui correspond, en poutre, à un pivot remonté à 11,94 mm. Un module aussi bas (maçonnerie tendre, cale compressible) doit donc être déclaré, et le pivot calculé en conséquence.

  *Correction (relecture adverse du 07/11/2026)* : la version précédente annonçait « pivot à 11,8 mm, section 3 à **+3,8 %** sur le classeur dès 30 N/mm³ ». Ce chiffre sortait d'un autre modèle (`scripts/c03c_mur.py` : rondelle Ø30 **rigide**, cheville de 0,1 mm), où la résultante de l'ancrage **saute d'un bord de la rondelle à l'autre** : y_T = 70,8 mm sur mur rigide (bord bas, 70), 99,0 mm à 30 N/mm³ (bord haut, 100) [s09]. C'est ce saut, non la seule montée du pivot, qui donnait +3,8 %. Le classeur n'est dépassé à 30 N/mm³ que si la résultante d'ancrage monte au-dessus de l'axe ; cette hypothèse de rondelle doit être déclarée.
- **Effet de levier** : avec un ancrage très raide, le haut de l'aile vient s'appuyer sur le mur au-dessus de l'ancrage et la traction augmente sans que la charge change. Toutes les colonnes sont calculées au même maillage, celui de la figure e (4 725 nœuds) [s07] :

| Raideur axiale d'un ancrage (kN/mm) | 30 | 100 | 200 | 250 | 300 | 1 000 | 100 000 | infinie |
|---|---|---|---|---|---|---|---|---|
| Traction totale H (kN) | 33,90 | 33,98 | 34,03 | 34,04 | 36,54 | 53,73 | 93,24 | 95,78 |
| Contact en tête d'aile (kN) | 0 | 0 | 0 | 0 | 1,76 | 13,95 | 47,98 | 50,20 |

(Statique de référence, contact au talon et cisaillement sur l'axe : H = 33,96 kN. La version précédente de ce tableau mêlait un maillage gros pour les six premières colonnes et le maillage fin pour la dernière ; à 30 kN/mm, le maillage gros donne 33,53 kN, 1,1 % de moins.)

Le seuil se situe entre 250 et 300 kN/mm par cheville ; les chevilles courantes restent en dessous ; une cheville précontrainte peut le dépasser. Le « jeu » de 1 cm de `C76` ne couvre pas ce cas (95,78 kN = 2,24 fois les 42,76 kN du classeur).

---

## 2. La signification de chaque bras de levier et de chaque jeu

### 2.1 Le tableau

| Cellule | Symbole = formule (certain) | Cas | Ce que c'est (interprété sauf mention) |
|---|---|---|---|
| `C13` | X (saisie) | 75 mm | distance du nu du mur à la ligne d'action de R |
| `C14` | Xj (saisie) | 0 mm | **déplacement de la charge vers l'extérieur** : s'ajoute à tous les bras de l'aile horizontale et à Mmax (certain) ; tolérance sur la position de l'appui de l'escalier |
| `C16` | Y (saisie) | 85 mm | hauteur de l'axe de l'ancrage au-dessus du dessous de la cornière |
| `C17` | Yj (saisie) | 0 mm | **ancrage plus haut que prévu** : n'entre que dans `C56` et `C64` (certain), pas dans les ancrages ; tolérance de perçage vers le haut (flèche du schéma) |
| `C19` | Xe = X − r − e | 47 mm | bras de R jusqu'à la **section 1** (pied du congé, à e + r du dos) |
| `C20` | Xe+r = X − e | 60 mm | bras de R jusqu'à la **section 2** (nu intérieur de l'aile verticale) ; « Xe+r » se lit « Xe plus r » |
| `C23` | Ye = Y − e − r | 57 mm | distance de l'ancrage à la **section 3** (pied du congé de l'aile verticale) : bras de H |
| `C24` | Ye+r = Y − e | 70 mm | distance de l'ancrage à la **section 4** (niveau de la face supérieure de l'aile horizontale) |
| `C25` | YH = Y − e/2 | 77,5 mm | distance de l'ancrage au **pivot des sections** (plan moyen de l'aile horizontale) ; le dessin l'affiche « 78 » (format `0" mm"`, certain) |
| `C76` | jeu (saisie, **en cm**) | 1 cm | n'entre que dans `C82` (certain) : **position de la résultante de compression** 1 cm au-dessus du dessous, **ou** tolérance de pose de l'ancrage 1 cm plus bas — la formule ne permet pas de trancher |
| `C75`−`C76` | Y − jeu (en cm) | 7,5 cm | bras du couple pour la traction d'ancrage (non dessiné) |
| `A20` | larg. − X | 45 mm | de la charge au bord libre ; dessin seulement, aucun contrôle |
| `A21` | haut. − Y | 35 mm | de l'ancrage au bord supérieur ; dessin seulement : **aucune formule ne la lit** (certain) — `C90` recalcule `C5` − `C16` |
| `C6`, `C7` | e, r (catalogue ou `K6`, `K7`) | 15 ; 13 mm | épaisseur ; **congé intérieur r1** (pour 22 lignes du catalogue sur 26 vérifiables) |
| `C8` | L (saisie, **en cm**) | 30 cm | longueur reprise : largeur de toutes les sections et des τ |

### 2.2 Les trois jeux ne sont pas interchangeables

Chaque jeu agit sur des grandeurs différentes. Effet d'un jeu de 20 mm placé dans chaque case [s02 §4] :

| Saisie | M1 | M3 (kN·m) | Taux 1 | Taux 3 | Traction / ancrage |
|---|---|---|---|---|---|
| Cas de contrôle | 2,00972 | 2,35870 | 75,80 % | 88,86 % | 14,253 kN |
| Xj = 20 mm | 2,86492 | 2,98768 | **107,83 %** | **112,43 %** | 18,054 kN |
| Yj = 20 mm | 2,00972 | 2,53271 | 75,80 % | 95,38 % | 14,253 kN (inchangée) |
| Xj = Yj = 20 mm | 2,86492 | 3,20810 | 107,83 % | **120,69 %** | 18,054 kN |
| `C76` = 2 cm | 2,00972 | 2,35870 | 75,80 % | 88,86 % (inchangé) | 16,446 kN |
| Yj = −20 mm | 2,00972 | 2,06363 | 75,80 % | 77,82 % | 14,253 kN |

- **Xj** pèse lourd : +1 mm ajoute 1,18 point au taux maximal ; le cas validé atteint 100 % pour **Xj = 9,455 mm** (94,75 % à 5 mm, 100,64 % à 10 mm).
- **Yj** pèse moins : +1 mm ajoute 0,40 point ; 100 % est atteint pour Yj = 41,83 mm ; à l'infini, la section 3 tend vers 120,65 % (tout Mmax sur l'épaisseur e).
- **Yj ne touche pas aux ancrages** : un ancrage plus haut allonge le bras et réduit la traction, ce qui est du côté sûr ; mais un ancrage plus **bas** (défavorable pour l'ancrage) n'est couvert que par `C76`.
- **`C76` ne touche pas à la cornière** : il ne change que la traction d'ancrage.
- **Pièges d'unités (certain)** : `C8` et `C76` sont en cm, `C13` à `C17` en mm. « 20 » tapé en `C76` donne un bras de −115 mm et une traction de **−9,30 kN** ; « 300 » tapé en `C8` fait tomber le taux à **8,89 %**. Aucune alerte.

### 2.3 Le jeu par défaut de 20 mm demandé

Un jeu par défaut de 20 mm, appliqué au déplacement de la charge (Xj), **fait passer le cas historique validé de 88,86 % à 112,43 %**. Il ne peut donc pas être la valeur par défaut du mode « méthode historique », qui doit reproduire la note. Il peut l'être dans le mode « équilibre », à condition que l'écran et la note disent clairement qu'il ne figurait pas dans la note validée. **À trancher avec le maître de stage** : sur quelle grandeur porte ce jeu, et faut-il l'appliquer dans les deux sens ?

### 2.4 Proposition : quatre grandeurs nommées, en mm, dessinées

| Grandeur proposée | Remplace | Appliquée… | Agit sur |
|---|---|---|---|
| **Δx — décalage de la charge** | `C14` (Xj) | vers l'extérieur (défavorable partout) | moments des deux ailes, traction d'ancrage |
| **±Δy — tolérance de position de l'ancrage** | `C17` (Yj) et la lecture « tolérance » de `C76` | vers le haut pour la cornière, vers le bas pour l'ancrage (configuration A) ; vers le bout libre pour l'ancrage en configuration B | rapport Ye/YH ; bras de la traction |
| **y_c — position de la résultante de compression** | `C25` (e/2) et la lecture « bras » de `C76` | une seule valeur pour tout le calcul (0 sur support raide d'après le maillage ; e/2 pour rester proche du classeur) | H et donc M3, M4 et la traction |
| **g — cale ou jeu entre cornière et support** | (absent) | distance de montage | ne change pas les moments de la cornière ; donne à l'ancrage un **cisaillement avec bras de levier**, à transmettre à Hilti ; fixe l'endroit réel de la compression (sous la cale) |

Ainsi, une **distance de montage** (g, Δy) n'est jamais confondue avec un **bras de levier** (y_c).

---

## 3. Pourquoi les quatre moments sont différents

### 3.1 Le cas de contrôle reproduit

| Section | t (mm) | W (mm³) | Bras (mm) | M (kN·m) | σ | τ | σ_éq (N/mm²) | Taux |
|---|---|---|---|---|---|---|---|---|
| 1 — aile horiz., e | 15 | 11 250 | Xe = 47 | 2,00972 | 178,642 | 9,502 | 179,398 | 75,80 % |
| 2 — aile horiz., e + r | 28 | 39 200 | Xe+r = 60 | 2,56560 | 65,449 | 5,090 | 66,040 | 27,90 % |
| 3 — aile vert., e | 15 | 11 250 | Ye = 57 | 2,35870 | 209,662 | 9,502 | 210,307 | **88,86 %** |
| 4 — aile vert., e + r | 28 | 39 200 | Ye+r = 70 | 2,89665 | 73,894 | 5,090 | 74,418 | 31,44 % |

σ_adm = f_y/1,5 = 236,667 N/mm² ; τ_adm = σ_adm/√3 = 136,640 N/mm² ; Mmax = 3,2070 kN·m [s02 §1].

**Écart au classeur.** Valeurs de l'ingénieur : M1 = 2,00972, M2 = 2,56560, M3 = 2,35870, M4 = 2,89665 kN·m, taux 88,86 % en section 3 ; valeurs recalculées : 2,009720 ; 2,565600 ; 2,358697 ; 2,896645 ; 88,8621 %. Écart nul aux arrondis près ; écart relatif au cache du fichier ≤ 2,5·10⁻¹⁶ sur les 92 cellules numériques modélisées (la 93ᵉ, `L22` = 3,7 × 0,19 × 1 230, est isolée et sans lien avec la cornière) [s01 §3].

### 3.2 Deux diagrammes, quatre lectures

Un moment est un effort multiplié par un bras. Les quatre moments sont **quatre ordonnées de deux diagrammes différents** (voir figure b au §4) :

- **M1 et M2** : même effort **R**, deux bras (47 et 60 mm). M2 − M1 = R·r = **0,55588 kN·m**.
- **M3 et M4** : l'effort n'est pas R mais **H = Mmax/YH = 41,381 kN**, et les bras se comptent depuis l'ancrage (57 et 70 mm). M4 − M3 = H·r = **0,53795 kN·m**.
- **M3 ≠ M1**, bien que les deux sections soient au pied d'un congé : M3/M1 = (H/R)·(Ye/Xe) = 0,96774 × 1,21277 = **1,17364**. L'ancrage est plus loin de la section 3 que la charge ne l'est de la section 1. À épaisseur égale, la section 3 porte le plus grand moment : **c'est pourquoi elle gouverne**.
- **Ordre** : M1 = 2,00972 < M3 = 2,35870 < M2 = 2,56560 < M4 = 2,89665 < Mmax = 3,20700 kN·m. Les sections 2 et 4 portent les plus grands moments (elles sont plus près de l'angle) mais leur module est ((e + r)/e)² = **3,4844 fois** plus grand : σ2/σ1 = 0,3664, σ4/σ3 = 0,3524.
- **Le cisaillement ne pèse presque rien** : σ_éq dépasse σ de +0,42 % (S1), +0,90 % (S2), +0,31 % (S3), +0,71 % (S4).

### 3.3 Les ancrages

| Grandeur | Cellule | Formule | Valeur |
|---|---|---|---|
| Traction ELS par ancrage | `C82` | Mmax/((Y − jeu)·n) = 3,207/((85 − 10)·3) | 14,2533 kN |
| Cisaillement ELS par ancrage | `C83` | R/n | 14,2533 kN |
| Rapport ELU/ELS | `C71`/`C70` | 68,9/42,76 | 1,61132 |
| Traction et cisaillement ELU | `C85`, `C86` | × 1,61132 | 22,9667 kN |
| Valeurs tapées (Hilti ?) | `G85`, `G86` | — | 24,68 kN (×1,0746) ; 22,97 kN |
| Pression diamétrale (ELS) | `C88` | V_ELS/(d·e) | 59,389 N/mm² |
| Arrachement (ELS) | `C90` | V_ELS/(2·(haut. − Y)·e) | 13,575 N/mm² |

- N = V ici parce que X = Y − jeu = 75 mm : **coïncidence**, pas une règle.
- `C88` et `C90` **ne sont comparées à aucune limite** : leurs mises en forme conditionnelles visent `#REF!` (certain, comme celles des contraintes G31 à G67) [s01 §1]. L'arrachement compte la longueur jusqu'à l'axe du trou ; en retirant un demi-trou Ø18 (hypothèse), il monte à 18,27 N/mm² (+34,6 %) [s02 §2].
- Le « OK » affiché devant 24,68 kN fait partie du format de la cellule : ce n'est pas un contrôle.

### 3.4 Pour situer la méthode

| Rapport ELU/ELS | Admissible (classeur) | EC3 élastique | EC3 plastique |
|---|---|---|---|
| 1,35 (tout permanent) | 88,86 % | 79,98 % | 53,15 % |
| 1,50 (tout variable) | 88,86 % | 88,86 % | 59,06 % |
| 1,6113 (`C71`/`C70`) | 88,86 % | 95,46 % | 63,44 % |

Section 3, statique du classeur, γ_M0 = 1,0 ; V/V_pl = 0,075 à l'ELU, sans réduction [s02 §8]. **Lecture** : la méthode est aussi sévère ou plus sévère que l'EC3 élastique tant que le rapport ELU/ELS réel reste sous 1,50, et 1,50 à 1,67 fois plus sévère que l'EC3 plastique. Ce n'est pas une justification aux états limites ; le module l'affichera pour ce qu'elle est : « contraintes admissibles, efforts ELS non majorés, coefficient global 1,50 sur f_y ».

**Le module actuel du site** (bande d'acier, M = V·e au nu du mur, épaisseur e) donne **120,65 %** sur ce cas (+35,8 %) **si l'on y saisit** σ_lim,ELS = f_y/1,5 = 236,67 N/mm² ; selon le bras saisi, de 75,80 % (il manque la section 3 : non conservatif) à 120,65 %. **Avec ses valeurs par défaut** (`acier/corniere/entree.py` : σ_lim,ELS = 155, τ_lim,ELS = 0,8, σ_eq,lim,ELU = 186 N/mm², γ_ELU = 1,35), le même cas donne 184,22 % à l'ELS, 207,2 % à l'ELU et 1 187,8 % en cisaillement [s02 §9].

---

## 4. Comment obtenir des diagrammes continus exacts

### 4.1 Les équations — aucune interpolation

Les quatre sections sont des lectures de deux droites ; les diagrammes continus en découlent directement :

- **Aile horizontale** (statique exacte, certain) : M(u) = R·(X' − u), V = R, N = 0, pour e ≤ u ≤ X' ; nuls au-delà de la charge.
- **Aile verticale, méthode du classeur** (certain) : M(y) = H·(Y' − y), avec H = Mmax/(YH + Yj) ; nul à l'ancrage.
- **Aile verticale, équilibre cohérent** : M(y) = H·(Y' − y) − R·(e/2 − u_V), N = +R, V = H (§1.2).
- **Épaisseur dans le congé** (rayon r centré en (e + r ; e + r)) : **t(s) = e + r − √(r² − (e + r − s)²)** pour e ≤ s ≤ e + r, s étant la distance au dos de l'autre aile ; t = e au-delà. Pour s < e on est dans le **bloc d'angle**, où la théorie des poutres ne s'applique pas.
- **Contrainte** : σ(s) = M(s)/(L·t(s)²/6) [+ N/(L·t)], τ = V/(L·t), taux(s) = σ_éq(s)/σ_adm.

Le **moment** dépend de l'équilibre ; la **contrainte** dépend en plus de la géométrie locale. Les deux restent des courbes distinctes.

![Figure b — diagrammes continus](figures/fig_b_diagrammes.png)

*Figure b — Moment, épaisseur locale, contrainte équivalente et taux le long des deux ailes ; sections 1 à 4, congés et bloc d'angle repérés ; en vert, l'équilibre cohérent P0. Fichier source : `figures/fig_b_diagrammes.svg`.*

### 4.2 Le résultat sur le cas de contrôle

En entrant dans le congé, le moment croît linéairement alors que l'épaisseur ne croît qu'au second ordre (le congé est tangent à l'aile) : le maximum se trouve **juste avant le pied du congé** [s03 §1].

| Aile | Point | Distance au dos (mm) | t (mm) | M (kN·m) | σ_éq (N/mm²) | Taux |
|---|---|---|---|---|---|---|
| horizontale | section 2 | 15,00 | 28,000 | 2,56560 | 66,040 | 27,90 % |
| horizontale | **maximum** | **26,02** | 15,152 | 2,09438 | 183,185 | **77,40 %** |
| horizontale | section 1 | 28,00 | 15,000 | 2,00972 | 179,398 | 75,80 % |
| verticale (classeur) | section 4 | 15,00 | 28,000 | 2,89665 | 74,418 | 31,44 % |
| verticale (classeur) | **maximum** | **26,34** | 15,106 | 2,42739 | 213,365 | **90,15 %** |
| verticale (classeur) | section 3 | 28,00 | 15,000 | 2,35870 | 210,307 | 88,86 % |
| verticale (P0) | maximum | 26,39 | 15,100 | 1,99019 | 184,465 | 77,94 % |

Le maximum est à 1,98 mm du pied du congé dans l'aile horizontale (+2,11 % sur la section 1) et à 1,66 mm dans l'aile verticale (+1,45 % sur la section 3).

**Généralisation** : l'écart entre le maximum et la section au pied du congé dépend de la distance s entre la charge (ou l'ancrage) et le pied du congé [s03 §4] :

| s | Catalogue du site (197 cornières égales, r1) | Catalogue du classeur (42 désignations) |
|---|---|---|
| 40 mm | jusqu'à ×1,086 | jusqu'à ×1,052 |
| 20 mm | jusqu'à ×1,271 | jusqu'à ×1,178 |
| 10 mm | jusqu'à ×1,720 | jusqu'à ×1,513 |
| 5 mm | jusqu'à ×2,684 | jusqu'à ×2,269 |

(Le maximum est atteint par les plus grosses cornières, L300×300×35 — cornière complémentaire du catalogue du site, hors séries courantes — et L200×200×20.) Pour le cas de contrôle (s = 47 et 57 mm), ×1,0214 et ×1,0147.

### 4.3 L'approximation « e + r » : conservatrice, admissible ou à remplacer ?

- **L'épaisseur est juste** (certain, géométrie) : à la coupe x = e, la hauteur de matière vaut exactement e + r = 28 mm.
- **La répartition linéaire de contrainte ne l'est pas** : le haut de cette coupe est le pied du congé de l'autre aile, un bord presque libre. Le maillage y trouve **1,71 N/mm²** en fibre côté congé (section 2) et **124,70 N/mm²** en fibre opposée, soit **1,905 fois** σ2 du classeur ; section 4 : 1,64 et 97,47 N/mm² (×1,319) [s07]. Comme estimation de la contrainte à l'endroit qu'elle désigne, la section « e + r » est donc **non conservatrice**.
- **Elle ne gouverne pourtant presque jamais** : avec le critère du classeur (σ_éq, cisaillement compris), la section 2 ne dépasse la section 1 que si X + Xj < **30,57 mm**, et la section 4 ne dépasse la section 3 que si Y + Yj < **33,02 mm** (à X = 75 mm) ; en flexion seule, le seuil commun vaut 33,23 mm (charge à 5,23 mm du pied du congé), qui reste une condition nécessaire [s03 §2].
- **Verdict : à remplacer comme vérification du congé, à conserver comme repère historique.** La vérification du congé devient la recherche du maximum le long de t(s), qui reste dans la même théorie des poutres. (Pour mémoire, une coupe à e + 0,8·r avec l'épaisseur e, par analogie avec l'EN 1993-1-8, donnerait +5,5 % sur S1 et +4,6 % sur S3 ; analogie à confirmer dans le texte, non retenue ici [s03 §3].)

### 4.4 Ce que montre le maillage

![Figure e — maillage de la coupe](figures/fig_e_maillage.png)

*Figure e — Déformée amplifiée et contrainte de von Mises du cas de contrôle ; zoom sur l'angle avec les sections 1 à 4. Fichier source : `figures/fig_e_maillage.svg`.*

- **Confirmé** : le moment de l'aile horizontale (M1 maillé / statique exacte = 1,0001), l'effort normal de traction de l'aile verticale, le pivot à l'arête.
- **Concentration au pied du congé** : en sections 1 et 3, la fibre côté congé dépasse la contrainte de poutre de 12 % (rapports 1,123 et 1,121). En section 3, elle reste sous la valeur du classeur (rapport 0,971) parce que le classeur surestime M3.
- **Pic local** : **236,4 N/mm²** à la surface du congé, à 2,95 mm du pied côté aile horizontale, soit 1,30 fois la contrainte de poutre au même point et 99,9 % de σ_adm. Un second maillage indépendant (analyse critique, `scripts/c03b_convergence.py`) trouve 236,5 N/mm² au pas de 0,25 mm [s09]. **Interprété** : c'est une contrainte de pointe ; l'acier ductile la redistribue en statique. Elle compte en fatigue et pour situer la première plastification, pas comme critère ELS. Le module peut l'afficher à titre d'information.
- **Faisable dans le navigateur** : le même calcul tourne dans Pyodide avec numpy (ceux du site) en 0,79 s pour 1 469 nœuds et 4,55 s pour 4 725 nœuds, plus 1,71 s de chargement (mesuré sous Node 22 par `extensions/e06_pyodide.mjs`, pas encore dans un onglet) [s09].

### 4.5 Ce que le module affichera

- Par aile : diagrammes **M, N, V** continus ; bande **épaisseur t(s)** ; diagramme **σ_éq** et **taux** ; congés et bloc d'angle grisés ; sections 1 à 4 marquées avec leurs valeurs ; **maximum** et sa position.
- Le mode historique trace les courbes du classeur ; le mode « équilibre » trace les siennes, l'ancienne restant superposable pour comparaison.

---

## 5. Comment traiter une cornière de 2 m avec plusieurs ancrages et des charges localisées

![Figure d — cornière longue](figures/fig_d_longue.png)

*Figure d — Cornière de 2,00 m en élévation (à l'échelle), efforts par ancrage selon la raideur des ancrages, couple de torsion et moment longitudinal. Fichier source : `figures/fig_d_longue.svg`.*

### 5.1 Local et longitudinal

- **Comportement local** (la méthode du classeur) : une tranche de cornière dont les deux ailes fléchissent ; il suppose que charge, ancrage et compression du mur agissent **dans la même tranche** de longueur L.
- **Comportement longitudinal** : dès que les ancrages sont espacés et que la charge n'est pas au droit d'un ancrage, la cornière transporte la charge le long de son axe, par **flexion** (vers les cisaillements d'ancrage) et par **torsion** (le couple R·(X' − e/2) doit rejoindre les couples ancrage–mur, qui n'existent qu'aux ancrages). Une cornière est une section ouverte : très raide en flexion, très souple en torsion — **E·I = 934,3 kN·m² contre G·It = 21,41 kN·m²** pour la L120×120×15, rapport 44 (propriétés du catalogue du site) [s05 §1].

### 5.2 Pourquoi « R/n » ne tient pas

- **Cisaillement** : il dépend peu de la raideur des ancrages, mais la **règle du levier**, V_i = R/n + R·e_z·(z_i − z_G)/Σ(z_i − z_G)² (**30,29 / 14,25 / −1,78 kN** ici), n'en est que la limite des ancrages **souples** devant la cornière. Avec des ancrages raides, la cornière est une poutre continue sur appuis (hyperstatique) : la raideur de flexion d'une travée, 48·E·I/s³ = 87,6 kN/mm, est du même ordre que celle d'un ancrage raide. Dans le modèle (flexion biaxiale + torsion), V va de 29,84 / 15,15 / −2,23 kN (appuis verticaux très raides) à 30,29 / 14,26 / −1,78 kN (très souples) ; pour les ancrages rigides du tableau, V3 vaut 1,20 fois la règle du levier ; une poutre continue en flexion verticale seule sur appuis rigides donne 29,66 / 15,51 / −2,41 kN [s05 §2c]. « R/n » n'est juste que si la charge est au centre du groupe d'ancrages.
- **Traction** : elle dépend du rapport entre la raideur en rotation d'un ancrage et la raideur en torsion d'une travée, k_θ·s/(G·It) = **2,40 à 12,12** ici : la cornière est trop souple en torsion pour répartir, et le couple va surtout à l'ancrage voisin de la charge. « R/n » suppose une cornière raide en torsion **par rapport aux ancrages**, k_θ·s/(G·It) → 0 : avec G·It × 1 000, on retrouve 11,45 / 11,27 / 11,23 kN (ancrages souples) et 11,88 / 11,12 / 10,96 kN (raides), mais avec des ancrages rigides la répartition reste 23,46 / 10,50 / 0,00 kN [s05 §2b].

### 5.3 L'exemple chiffré

**Hypothèses d'illustration** : L120×120×15 de 2 000 mm ; ancrages à z = 200, 1 000 et 1 800 mm, Y = 85 ; R = 42,76 kN répartis sur 300 mm centrés à z = 400 mm, X = 75. Référence « R/n » : V = 14,25 kN et N = H/3 = 11,32 kN (équilibre P0) ; le classeur donnerait 14,25 kN [s05 §2].

| Ancrages | N1 / N2 / N3 (kN) | V1 / V2 / V3 (kN) | N1 / (H/3) | Torsion (N/mm²) |
|---|---|---|---|---|
| rigides | 25,16 / 8,80 / 0,00 | 29,93 / 14,97 / −2,14 | 2,22 | 112 (82 %) |
| raides (100 / 30 kN/mm) + aile | 24,04 / 9,55 / 0,37 | 29,99 / 14,85 / −2,08 | 2,12 | 107 (78 %) |
| souples (10 / 5 kN/mm) + aile | 20,87 / 10,28 / 2,81 | 30,16 / 14,51 / −1,91 | 1,84 | 92 (67 %) |

(Les pourcentages de torsion se rapportent à τ_adm = 136,6 N/mm² ; contrôles d'équilibre : ΣV = 42,760 kN, ΣN − ΣC = 0,0000 kN.)

- **Un ancrage est cisaillé dans l'autre sens** (A3, −1,91 à −2,14 kN) : à donner à Hilti avec son signe.
- **Flexion longitudinale** : |M| max = 4,64 à 4,70 kN·m, σz ≈ 85 N/mm², à cumuler avec la flexion locale. Le classeur ne la calcule pas, et le module actuel du site non plus : sa vérification « longitudinale » redonne le taux transversal à 4,4·10⁻¹⁶ près sur 200 entraxes [s02 §9].
- **Conséquence locale** : au droit de l'ancrage le plus chargé, par analogie avec le tronçon en T de l'EN 1993-1-8 (à confirmer) : m = (Y − e) − 0,8·r = 59,6 mm, l_eff = min(2π·m ; 4·m + 1,25·(h − Y)) = min(374,5 ; 282,1) = 282,1 mm, et σ = 6·N1·m/(l_eff·e²) + V1/(l_eff·e) (V1 traverse l'aile verticale en traction). L'aile travaille à **124,7 à 148,8 N/mm² (53 à 63 %)** ; avec le bras Ye = 57 mm de la section 3 au lieu de m, 119,6 à 142,6 N/mm² [s05 §3]. L'effet de levier ne se lit pas dans ce modèle (§5.4) ; s'il existe (§1.3), N1 augmente d'autant.
- **Un ancrage au droit de la charge** (variante 400 / 1 000 / 1 600 mm) ne supprime pas la torsion (73 à 74 N/mm²) et concentre l'effort : N = 23,19 à 32,14 kN, V = 36,43 à 42,74 kN sur cet ancrage [s05 §4].
- **Largeur d'appui** : la longueur reprise L n'est entièrement efficace que si la charge est appliquée sur au moins **206 mm** (diffusion à 45° jusqu'à la section 1), **253 mm** en bout de cornière ; avec un appui de 100 mm, σ1 est multipliée par 1,55 (taux 117,2 %) [s02 §6].

### 5.4 La méthode proposée et ses hypothèses

Modèle de **poutre à section indéformable sur appuis élastiques, avec torsion** (quelques centaines d'éléments, numpy seul) :

- inconnues le long de la cornière : deux translations et la rotation ; flexion biaxiale avec le produit d'inertie et torsion de Saint-Venant, **lus dans le catalogue du site** (Iy = Iz = 444,881 cm⁴, Iyz = −260,219 cm⁴, It = 26,4262 cm⁴) ;
- chaque ancrage : un tirant horizontal à la hauteur Y (raideur de l'ancrage en série avec la flexion locale de l'aile, k_pl = 3·E·(l_eff·e³/12)/Y³ = 81,4 kN/mm ici — **hypothèse** : console de longueur Y encastrée au talon, le bloc d'angle tournant avec la section indéformable ; encastrée en section 3, sur Ye = 57 mm, elle vaudrait 270,0 kN/mm) et un appui vertical ;
- le mur : un contact unilatéral continu à l'arête inférieure. **Pas de contact en tête** : la section étant indéformable, ce modèle ne contient pas la flexion de l'aile au-dessus de l'ancrage qui crée l'effet de levier ; un contact en tête y donne un effort qui suit la pénalité choisie (N1/(H/3) = 2,55 ; 3,67 ; 8,54 ; 25,77 pour une pénalité de 10⁴ à 10⁷ N/mm par mm, contre 2,22 sans contact) [s05 §2d]. L'effet de levier se juge sur le maillage de la coupe (§1.3) ;
- chaque charge : un effort vertical et un couple, ponctuel ou réparti sur sa **largeur réelle** ;
- **les combinaisons sont calculées chargées** : le contact rend le modèle non linéaire (écart de superposition jusqu'à 0,221 kN par ancrage dans l'exemple ELU ci-dessous) [s06 §4].

**Hypothèses à afficher dans la note** : section indéformable ; frottement mur–cornière nul ; raideurs d'ancrage **déclarées** (déplacements de service de l'ETA de la cheville) — le résultat en dépend ; jeu des trous de passage ignoré (avec du jeu, le cisaillement va d'abord aux ancrages dont le trou porte ; la règle correspondante de l'EN 1992-4 est à relire) ; charge à position fixe ; petits déplacements. Quand ces hypothèses ne suffisent pas (torsion au-delà d'environ la moitié de l'admissible, aile mince, charge loin des ancrages), un modèle en coque des deux ailes est la vérification de référence, hors du module dans un premier temps.

### 5.5 ELS, ELU et efforts à fournir à Hilti

- **Le passage ELS → ELU du classeur n'est pas une combinaison** : avec γG = 1,35 et γQ = 1,50 sur les mêmes charges, le rapport ELU/ELS reste entre 1,35 et 1,50 ; 1,611 est impossible avec K_FI = 1. (Avec K_FI = 1,1, classe de conséquences CC3 de l'EN 1990 annexe B, si l'annexe nationale l'applique, le plafond monte à 1,650 ; la volée 12 donnerait 1,534 : la conclusion tient pour cet escalier [s06 §1].) Il faudrait que 42,76 kN soit une combinaison fréquente (G/Q = 2,657) ou quasi permanente (G/Q = 3,890), ou que les deux réactions viennent de calculs différents [s06 §1].
- **Sur une cornière longue, un coefficient global ne donne pas l'ELU par ancrage** (hypothèse d'illustration : G = 30 kN répartis sur 2 m, Q = 30 kN sur 300 mm centrés à 0,40 m, ancrages raides) [s06 §4] :

| Ancrage | N_ELS (G + Q) | N_ELU calculé | 1,425 × N_ELS | 1,611 × N_ELS | V_ELU calculé |
|---|---|---|---|---|---|
| A1 (z = 200) | 24,26 | **35,30** | 34,58 (−2,0 %) | 39,10 | 43,26 |
| A2 (z = 1 000) | 15,74 | 22,23 | 22,43 (+0,9 %) | 25,36 | 32,72 |
| A3 (z = 1 800) | 7,64 | 10,37 | 10,89 (+5,0 %) | 12,31 | 9,51 |

Une **platine unique supposée rigide** de 2 m (tractions égales) donnerait 22,63 kN par ancrage : elle sous-estime l'ancrage le plus chargé de **36 %**.

- **Ce que le module donne à Hilti PROFIS**, par ancrage et par combinaison, ELS et ELU séparés : **N** (traction) et **V** (cisaillement, avec son sens), positions (z, hauteur Y, distance au bord libre), et les hypothèses de répartition (pivot, raideurs, frottement nul, levier compté ou non). Pour une tranche courte où PROFIS répartit lui-même : V = 68,9 kN et M = V·X = 5,168 kN·m sur la platine (cas de contrôle à l'ELU) [s06 §3].
- **Ce que le module ne fait pas** : vérifier les chevilles (cône de béton, bords, épaisseur du support) ; il le dit dans le résultat et dans la note.

| Cas de contrôle, par ancrage | N_ELS | V_ELS | N_ELU | V_ELU |
|---|---|---|---|---|
| Classeur (bras Y − 1 cm, ×1,6113) | 14,253 | 14,253 | 22,967 | 22,967 |
| Équilibre P0, même R_ELU | 11,319 | 14,253 | 18,238 | 22,967 |
| P0, ELU = 1,35·G + 1,5·Q avec le partage G/Q de la volée 12 (hypothèse) | 11,319 | 14,253 | 15,786 | 19,879 |

---

## 6. Ce qui change lorsqu'on retourne la cornière

![Figure c — cornière retournée](figures/fig_c_retournee.png)

*Figure c — Même cornière, même charge, même trou d'ancrage : configuration A (classeur) et configuration B (aile horizontale en haut). Fichier source : `figures/fig_c_retournee.svg`.*

### 6.1 La configuration B a son propre équilibre

Aile horizontale en haut, aile verticale qui descend le long du mur ; Y_B = distance de la face supérieure à l'ancrage (85 mm pour la même cornière retournée), soit un ancrage à y_a = 35 mm au-dessus du bout libre [s04 §1].

- **Le contact passe au bout libre** de l'aile verticale : R fait tourner la cornière vers le mur sous l'ancrage. Le maillage le confirme (contact à y = 0,00 mm) [s07].
- **Le bras de la traction est la distance de l'ancrage au bout libre** : H = R·(X' − e/2)/y_a = **82,47 kN** pour Y_B = 85 mm (maillage : 82,58 kN, +0,14 %), soit **2,43 fois** la configuration A au même équilibre. En remontant l'ancrage près de l'angle (Y_B = 45 mm), H = 38,48 kN.
- **Entre l'ancrage et l'angle, le moment est constant et maximal** : M3 = M4 = R·(X' − e/2) = **2,886 kN·m**, **indépendant de la position de l'ancrage** (statique de la partie située au-dessus de la coupe, où seule la charge agit — certain, sans hypothèse de pivot). La formule du classeur Mmax·Ye/YH donnerait 2,359 kN·m, soit **18,3 % de moins** : elle serait non conservatrice.
- **L'aile verticale est comprimée** (N = −R) et la face du congé est comprimée : σ = M/W + R/A = 256,56 + 9,502 = 266,06 N/mm², **taux 112,42 %** (τ nul entre l'ancrage et l'angle).
- **Le trou d'ancrage est dans le moment maximal** : section nette au droit des trous (3 trous Ø18 sur 300 mm, hypothèse) **137,1 %**. En A, le moment y est nul.
- **La compression du mur au bout libre** vaut 82,47 kN, soit 274,9 N/mm de cornière : une pression locale sur le support à vérifier, ce que A n'impose jamais.
- **Les jeux changent de sens** : un ancrage décalé de 10 mm vers le bout libre fait passer H à 115,45 kN ; vers l'angle, à 64,14 kN ; Xj = 20 mm donne 144,54 %.

| | Configuration A (classeur) | Configuration B (retournée) |
|---|---|---|
| Contact avec le mur | arête inférieure, sous l'angle | bout libre de l'aile verticale |
| Bras de la traction | Y (vers le dessous) | distance ancrage – bout libre |
| Meilleure position d'ancrage | le plus **haut** possible pour la traction, mais la section 3 se charge | le plus **près de l'angle** possible |
| Moment, aile verticale | nul à l'ancrage, croissant vers l'angle | constant et maximal de l'ancrage à l'angle |
| Effort normal, aile verticale | traction R | compression R |
| Face du congé | tendue | comprimée |
| Trou d'ancrage | hors moment | dans le moment maximal |
| Formule M3 = Mmax·Ye/YH | applicable | **inapplicable** (−18,3 %) |

En A, monter l'ancrage soulage l'ancrage mais charge la section 3 (Ye/YH → 1) : pour Y = 65 / 85 / 105 mm, taux 3 du classeur 77,82 / 88,86 / 95,38 %, en P0 66,12 / 76,91 / 83,63 %, traction P0 44,40 / 33,96 / 27,49 kN [s04 §1b].

**Conclusion** : il ne suffit pas de retourner le dessin. B demande son corps libre, sa position de contact, la vérification de la section nette et celle de la pression sur le support.

### 6.2 Ailes inégales

- Les formules de **moment** de `P8` ne lisent que e, r, X, Y (et Xj, Yj) : elles s'appliquent telles quelles à une cornière inégale en configuration A (certain). L'arrachement `C90` lit en plus la hauteur `C5` (haut. − Y) : il dépend donc de l'orientation d'une cornière inégale (certain) [s01 §1]. L'orientation (grande aile contre le mur ou en console) change la hauteur d'ancrage possible, donc le partage entre traction et moment.
- **Le catalogue du classeur ne permet pas de choisir l'orientation** : 15 désignations y figurent deux fois avec les ailes échangées, et `VLOOKUP` (`C4` à `C7`) lit toujours la première ligne (certain) [s08].
- **Exemple** (même charge que le cas de contrôle, pour comparer ; cette cornière ne convient pas) : L150×100×12, ancrage à 35 mm du bord libre [s04 §3].

| Orientation | Rayon | Taux 1 | Taux 3 (classeur) | Traction (classeur) | Taux 3 (P0) | Traction (P0) |
|---|---|---|---|---|---|---|
| 150 horizontal × 100 vertical, Y = 65 | r1 = 12 | 128,3 % | 131,1 % | 58,31 kN | 114,6 % | 45,39 kN |
| 100 horizontal × 150 vertical, Y = 115 | r1 = 12 | 128,3 % | 157,4 % | 30,54 kN | 142,1 % | 25,66 kN |
| 150 × 100, rayon du classeur | r = 6 | 143,3 % | 150,2 % | 58,31 kN | 130,5 % | 45,39 kN |

Grande aile verticale : la traction est presque divisée par deux, mais la section 3 se charge (Ye/YH → 1). **L'orientation doit être un paramètre explicite** du calcul, pas une ligne en double du catalogue.

---

## 7. Les hypothèses : à conserver, à vérifier, à corriger

| Hypothèse (cellules) | Verdict | Preuve chiffrée |
|---|---|---|
| Statique de l'aile horizontale, M = R·bras (`C35`, `C45`) | **À conserver** (exacte) | maillage / statique = 1,0001 [s07] |
| σ_éq = √(σ² + 3τ²), σ en fibre et τ moyen (`G37`…`G67`) | **Conservatrice** (léger) | σ_éq/σ − 1 de +0,31 % à +0,90 % [s02 §1] |
| Contraintes admissibles, f_y/1,5 sur ELS (`H31`, `H34`) | **À conserver, en la nommant** | = EC3 élastique avec 1,50 sur tout ; 79,98 % (1,35) à 95,46 % (1,611) [s02 §8] |
| Moment de l'aile verticale Mmax·ρ (`C56`, `C57`, `C64`, `C65`) | **Conservatrice ici, à corriger dans la forme** | 88,86 % contre 76,91 % (P0) à 83,96 % (P1) ; M3 classeur / maillage = 1,217 [s02 §5, s07] |
| « τ » = R dans l'aile verticale (`G54`, `G64`) | **À corriger** (c'est un effort normal) | N maillé = 42,87 kN ; le classeur n'est plus sûr si Y < 33,2 mm (P0) à 43,5 mm (P1) [s02 §5] |
| Pivot e/2 pour les sections, 1 cm pour les ancrages (`C25`, `C82`) | **À corriger** (un seul équilibre) | 41,38 contre 42,76 kN ; Hilti 15,21 mm ; maillage 0,00 mm (raide), 1,86 mm (souple) [s02 §5, s07] |
| Support indéformable (implicite) | **À vérifier** | 30 N/mm³ : pivot à 9,28 mm, S3 −3,3 % sous le classeur (−13,3 % sur mur raide) ; classeur atteint vers 16 N/mm³ ; rondelle rigide : résultante d'ancrage déplacée, à déclarer [s07, s09] |
| Pas d'effet de levier (implicite) | **À vérifier** | ancrage > 250 à 300 kN/mm par cheville : H jusqu'à 95,78 kN, ×2,24 [s07] |
| Cisaillement R/n (`C83`) | **Admissible** pour une tranche courte ; **à corriger** pour une cornière longue | V1 = 29,93 kN = 2,10 × R/3 [s05] |
| Traction partagée également (`C82`) | **Admissible** pour une tranche courte ; **à corriger** pour une cornière longue | N1 = 2,22 × H/3 (ancrages rigides) ; 1,84 (souples) [s05] |
| Frottement nul au mur (implicite) | **Conservatrice** pour les ancrages | tout le cisaillement va aux ancrages |
| Sections au pied du congé (S1, S3) | **Admissible ici, à corriger pour généraliser** | +2,11 % / +1,45 % ici ; jusqu'à ×1,271 à 20 mm, ×1,720 à 10 mm [s03] |
| Sections « e + r » (S2, S4) | **À remplacer** comme vérification du congé, à garder comme repère | fibre opposée 1,905 × σ2 ; gouvernent seulement si X + Xj < 30,57 mm ou Y + Yj < 33,02 mm (σ_éq) [s03, s07] |
| Pic local au congé (non vérifié) | **Admissible en statique** (interprété), à afficher | 236,4 N/mm² = 1,30 × poutre ; 99,9 % de σ_adm [s07] |
| Largeur L = longueur reprise pour toutes les sections (`C32`…`C62`) | **Admissible sous conditions** | appui ≥ 206 mm ; 100 mm → 117,2 % ; L = 250 mm → 106,63 % [s02 §6] |
| Rayon du catalogue (`C7`) | **À corriger** pour L150×100×12 (côté sûr) | r = 13 → 88,86 % ; 6,5 → 98,94 % ; 0 → 109,02 % [s02 §6] |
| Jeux Xj, Yj, `C76` | **À conserver séparés, à clarifier** | 20 mm : 112,43 % (Xj) ; 95,38 % (Yj) ; 16,45 kN (`C76`) [s02 §4] |
| ELU = ELS × 68,9/42,76 (`C85`, `C86`) | **À corriger** | 1,611 > 1,50 (K_FI = 1) ; cornière longue : −2,0 % à +5,0 % (coefficient global), platine rigide −36 % [s06] |
| Pas de contrôle de domaine ni d'unités | **À corriger** | charge hors de l'aile 155,9 % ; Y = 25 → M3 = −0,550 kN·m ; `C76` = 20 → −9,30 kN [s02 §4, §7] |
| `C88`, `C90` sans limite | **À corriger ou retirer** | 59,39 et 13,57 N/mm² ; demi-trou déduit : +34,6 % [s02 §2] |
| Configuration B avec les mêmes formules | **Inapplicable** | M3 −18,3 % ; vrai taux 112,42 % ; section nette 137,1 % [s04] |

---

## 8. Les améliorations proposées au module, par étapes

### 8.1 Ce qu'on garde de l'existant

**Du classeur** : la statique de l'aile horizontale, les quatre sections comme repères, la contrainte équivalente, la méthode admissible avec son coefficient 1,50 affiché, les efforts par ancrage ELS et ELU, et l'esprit du schéma de la note (sections orange, pastilles vertes, cotes bleues).

**Du module actuel du site** (`acier/corniere`, `api_web/corniere.py`) : le **catalogue partagé** (233 cornières : EN 10056-1 et gamme complémentaire, dont 138 hors séries courantes ; r1 et r2 distincts, propriétés et It — ses fonctions redonnent la fiche de la L120×120×15, analyse extensions `e07`), le sélecteur trié par masse, la lecture des désignations tapées, les **refus explicites** hors domaine, plusieurs cornières par note, les fenêtres de calcul (formule, valeurs substituées, résultat). **Ce qu'on remplace** : le modèle de bande d'acier (120,65 % sur le cas de contrôle avec σ_lim,ELS = f_y/1,5 saisi, 184,22 % avec ses valeurs par défaut ; aucun congé, aucune aile verticale), la vérification « longitudinale » qui répète la transversale (côté non sûr, déjà signalé), la limite τ_lim = 0,8 N/mm² par défaut.

### 8.2 Étape 2 — reproduction exacte du cas historique

- Un mode **« méthode Excel historique »** qui reprend les formules de `P8` mot pour mot, chaque grandeur portant le nom de sa cellule (`C35`, `C57`…) dans la fenêtre de calcul.
- **Tests de parité** : le cas de contrôle (M1 à M4, taux 88,8621 %) et les 592 saisies tirées au hasard de [s01 §4], comparés aux formules du classeur recalculées.
- Les anomalies de forme sont corrigées **dans les libellés seulement** (B43 « Xe = » pour Xe+r ; X et Xj affichés en section 3 sans être relus ; YH affiché « 78 ») ; les chiffres restent ceux du classeur.
- Les contrôles de domaine (charge sur l'aile, ancrage hors du congé et dans l'aile, unités) **avertissent sans changer un chiffre**.

### 8.3 Étape 3 — dessins, deux orientations, diagrammes continus, quatre sections

- **Coupe cotée à l'échelle** sur le modèle de la figure a : mur, ailes, congé, charge, les quatre grandeurs de jeu et de position du §2.4, ancrage, résultante de compression, bras de levier, sections 1 à 4. Toute cote modifiée met à jour les efforts.
- **Mode « équilibre »** : un seul corps libre (pivot y_c et ligne d'action u_V choisis et affichés ; par défaut le pivot à l'arête trouvé par le maillage, ou e/2 pour rester près du classeur), N et V exacts dans chaque aile.
- **Configurations A et B** avec leur propre équilibre (§6), y compris section nette et pression au bout libre en B ; **ailes inégales** avec l'orientation comme paramètre.
- **Diagrammes continus** du §4.5 et recherche du maximum dans le congé.
- **Maillage de contrôle** (option) : déformée et couleurs, avec ses hypothèses écrites et ses grandeurs de synthèse (pivot trouvé, traction, contact en tête) ; une vue d'information, jamais un critère.

### 8.4 Étape 4 — généralisation

- Longueur totale, ancrages (positions, entraxes, distances aux extrémités, diamètre et trou, raideurs si connues), charges par **cas** (valeur, X, position le long de la cornière, largeur d'appui), combinaisons ELS et ELU **définies** (coefficients affichés).
- Modèle longitudinal du §5.4 ; diagramme longitudinal distinct (efforts par ancrage, torsion, flexion) ; vérification locale au droit de l'ancrage le plus chargé ; avertissement sur l'effet de levier.
- Efforts par ancrage ELS et ELU, N et V séparés, avec positions et hypothèses, **prêts pour Hilti** ; aucune prétention de vérifier les chevilles.
- Architecture prête pour les **cornières PRS** (deux plats soudés) : même géométrie partagée, le congé devenant le cordon ; la section 4 devient l'interface des plats et se vérifie comme un assemblage avec le moteur des soudures du site (analyse extensions `e07` : propriétés d'une PRS 120×120, t = 15 et 12 mm, cordon de 6 mm, déjà calculées par les fonctions partagées).

### 8.5 Étape 5 — la note finale

Note A4 paysage, homogène avec les autres modules, dans l'ordre demandé : identification ; hypothèses et méthode (mode, coefficient 1,50, pivot, raideurs, frottement nul) ; géométrie et coupes cotées ; charges ELS et ELU ; schéma mécanique et réactions ; diagrammes ; vérifications des ailes et des sections critiques ; efforts par ancrage ; synthèse des taux ; conclusion et limites (chevilles non vérifiées, effet de levier, contrainte nominale et contrainte de pointe). Chaque formule est consultable : formule symbolique, valeurs substituées, résultat, admissible, taux à droite.

### 8.6 Questions à trancher avec l'auteur de la note et le maître de stage

1. `C12` = 42,76 kN est-il bien une réaction **ELS caractéristique** ? (1,35·G + 1,5·Q de la volée 12 de `E10A BD` vaut 42,768 kN/m.)
2. D'où vient `C71` = 68,9 kN, dont le rapport à 42,76 kN (1,611) dépasse 1,50 ?
3. `C76` : position de la compression contre le mur, ou tolérance de pose de l'ancrage ?
4. `C17` (Yj) : un jeu vers le haut seulement, ou dans les deux sens ?
5. Sur quelle longueur l'escalier s'appuie-t-il réellement dans les 300 mm repris ?
6. Le jeu par défaut de 20 mm : sur quelle grandeur, dans quel sens, et dans quel mode ?
7. Quel pivot retenir par défaut dans le mode « équilibre » : l'arête (maillage) ou e/2 (proche du classeur) ?
8. Le pic au congé (contrainte de pointe) : afficher seulement, ou vérifier (fatigue, chocs) ?
9. Le rapport PROFIS qui a donné `G85` = 24,68 kN est-il disponible ?
10. Quelles chevilles (ETA) : leur raideur décide de l'effet de levier et de la répartition sur une cornière longue.

---

## Annexe A — Inventaire des formules de `P8`

Les 93 formules numériques de la feuille `P8`, dans l'ordre des lignes ; valeur enregistrée dans le fichier. Les 8 formules de texte (`F35`, `F37`, `F45`, `F47`, `F55`, `F57`, `F65`, `F67`) ne font qu'afficher des libellés comme « 300 x 15 » ou « √(178,6² + 3 × 9,5²) = » [s01 §5].

| Cellule | Formule | Symbole | Valeur |
|---|---|---|---|
| `D3` | `=MAX(I37,I47,I57,I67)` | taux max = MAX(I37 ; I47 ; I57 ; I67) | 88,86 % |
| `C4` | `=IF($B$3="NON STANDARD",K4,VLOOKUP('P8 '!$B$3,CORNIERES!$A$2:$E$59,2,FALSE()))` | larg. (aile horizontale) — catalogue col. 2 ou K4 | 120 |
| `C5` | `=IF($B$3="NON STANDARD",K5,VLOOKUP('P8 '!$B$3,CORNIERES!$A$2:$E$59,3,FALSE()))` | haut. (aile verticale) — col. 3 ou K5 | 120 |
| `C6` | `=IF($B$3="NON STANDARD",K6,VLOOKUP('P8 '!$B$3,CORNIERES!$A$2:$E$59,4,FALSE()))` | e — col. 4 ou K6 | 15 |
| `C7` | `=IF($B$3="NON STANDARD",K7,VLOOKUP('P8 '!$B$3,CORNIERES!$A$2:$E$59,5,FALSE()))` | r (congé) — col. 5 ou K7 | 13 |
| `C19` | `=$C$13-$C$7-$C$6` | Xe = X − r − e | 47 |
| `A20` | `=C4-C13` | larg. − X (dessin) | 45 |
| `C20` | `=$C$13-$C$6` | Xe+r = X − e | 60 |
| `A21` | `=C5-C16` | haut. − Y (dessin) | 35 |
| `L22` | `=3.7*0.19*1230` | 3,7 × 0,19 × 1 230 (isolée, sens inconnu) | 864,69 |
| `C23` | `=$C$16-$C$6-$C$7` | Ye = Y − e − r | 57 |
| `C24` | `=C16-$C$6` | Ye+r = Y − e | 70 |
| `C25` | `=$C$16-$C$6/2` | YH = Y − e/2 | 77,5 |
| `C31` | `=$C$6` | t₁ = e | 15 |
| `F31` | `=C35` | copie de M1 | 2,00972 kN·m |
| `G31` | `=F31*10^6/F32` | σ₁ = M1/W₁ | 178,642 |
| `H31` | `=C3/$C$10` | σ_adm = f_y/γ | 236,667 |
| `I31` | `=G31/H31` | σ₁/σ_adm | 75,48 % |
| `C32` | `=$C$8*10*C31^2/6` | W₁ = L·e²/6 | 11 250 |
| `F32` | `=C32` | copie de W₁ | 11 250 |
| `C33` | `=$C$19` | Xe | 47 |
| `C34` | `=C14` | Xj | 0 |
| `F34` | `=$C$12` | copie de R | 42,76 kN |
| `G34` | `=($C$12)*1000/($C$8*10*$C$31)` | τ₁ = R/(L·e) | 9,502 |
| `H34` | `=C3/$C$10/SQRT(3)` | τ_adm = f_y/(γ·√3) | 136,64 |
| `I34` | `=G34/H34` | τ₁/τ_adm | 6,95 % |
| `C35` | `=$C$12*(($C$33+$C$34)/1000)` | M1 = R·(Xe + Xj) | 2,00972 kN·m |
| `G37` | `=(G31^2+3*G34^2)^0.5` | σ_éq,1 = √(σ₁² + 3τ₁²) | 179,398 |
| `H37` | `=C3/$C$10` | σ_adm | 236,667 |
| `I37` | `=G37/H37` | taux 1 | 75,80 % |
| `C41` | `=$C$6+$C$7` | t₂ = e + r | 28 |
| `F41` | `=C45` | copie de M2 | 2,5656 kN·m |
| `G41` | `=F41*10^6/F42` | σ₂ = M2/W₂ | 65,449 |
| `H41` | `=$C$3/$C$10` | σ_adm | 236,667 |
| `I41` | `=G41/H41` | σ₂/σ_adm | 27,65 % |
| `C42` | `=$C$8*10*C41^2/6` | W₂ = L·(e+r)²/6 | 39 200 |
| `F42` | `=C42` | copie de W₂ | 39 200 |
| `C43` | `=$C$20` | Xe+r (libellé B43 « Xe = ») | 60 |
| `C44` | `=C14` | Xj | 0 |
| `F44` | `=$C$12` | copie de R | 42,76 kN |
| `G44` | `=(F44)*1000/(C8*10*C41)` | τ₂ = R/(L·(e+r)) | 5,09 |
| `H44` | `=$C$3/$C$10/SQRT(3)` | τ_adm | 136,64 |
| `I44` | `=G44/H44` | τ₂/τ_adm | 3,73 % |
| `C45` | `=$C$12*(($C$43+$C$44)/1000)` | M2 = R·(Xe+r + Xj) | 2,5656 kN·m |
| `G47` | `=(G41^2+3*G44^2)^0.5` | σ_éq,2 | 66,04 |
| `H47` | `=$C$3/$C$10` | σ_adm | 236,667 |
| `I47` | `=G47/H47` | taux 2 | 27,90 % |
| `C51` | `=$C$6` | t₃ = e | 15 |
| `F51` | `=C57` | copie de M3 | 2,3587 kN·m |
| `G51` | `=F51*10^6/F52` | σ₃ = M3/W₃ | 209,662 |
| `H51` | `=$C$3/$C$10` | σ_adm | 236,667 |
| `I51` | `=G51/H51` | σ₃/σ_adm | 88,59 % |
| `C52` | `=$C$8*10*C51^2/6` | W₃ = L·e²/6 | 11 250 |
| `F52` | `=C52` | copie de W₃ | 11 250 |
| `C53` | `=$C$13` | X (affiché, non relu) | 75 |
| `C54` | `=C14` | Xj (affiché, non relu) | 0 |
| `F54` | `=$C$12` | copie de R | 42,76 kN |
| `G54` | `=($F$54)*1000/($C$8*10*$C$51)` | τ₃ = R/(L·e) | 9,502 |
| `H54` | `=$C$3/$C$10/SQRT(3)` | τ_adm | 136,64 |
| `I54` | `=G54/H54` | τ₃/τ_adm | 6,95 % |
| `C55` | `=($C$13+$C$14)/1000*$C$12` | Mmax = R·(X + Xj) | 3,207 kN·m |
| `C56` | `=($C$23+$C$17)/($C$25+$C$17)` | ρ₃ = (Ye + Yj)/(YH + Yj) | 0,73548 |
| `C57` | `=C56*C55` | M3 = ρ₃·Mmax | 2,3587 kN·m |
| `G57` | `=(G51^2+3*G54^2)^0.5` | σ_éq,3 | 210,307 |
| `H57` | `=$C$3/$C$10` | σ_adm | 236,667 |
| `I57` | `=G57/H57` | taux 3 | 88,86 % |
| `C61` | `=$C$6+$C$7` | t₄ = e + r | 28 |
| `F61` | `=C65` | copie de M4 | 2,89665 kN·m |
| `G61` | `=F61*10^6/F62` | σ₄ = M4/W₄ | 73,894 |
| `H61` | `=$C$3/$C$10` | σ_adm | 236,667 |
| `I61` | `=G61/H61` | σ₄/σ_adm | 31,22 % |
| `C62` | `=$C$8*10*C61^2/6` | W₄ = L·(e+r)²/6 | 39 200 |
| `F62` | `=C62` | copie de W₄ | 39 200 |
| `C63` | `=($C$13+$C$14)/1000*$C$12` | Mmax (2ᵉ copie) | 3,207 kN·m |
| `C64` | `=($C$24+$C$17)/($C$25+$C$17)` | ρ₄ = (Ye+r + Yj)/(YH + Yj) | 0,90323 |
| `F64` | `=$C$12` | copie de R | 42,76 kN |
| `G64` | `=($F$64)*1000/($C$8*10*$C$61)` | τ₄ = R/(L·(e+r)) | 5,09 |
| `H64` | `=$C$3/$C$10/SQRT(3)` | τ_adm | 136,64 |
| `I64` | `=G64/H64` | τ₄/τ_adm | 3,73 % |
| `C65` | `=C64*C63` | M4 = ρ₄·Mmax | 2,89665 kN·m |
| `G67` | `=(G61^2+3*G64^2)^0.5` | σ_éq,4 | 74,418 |
| `H67` | `=$C$3/$C$10` | σ_adm | 236,667 |
| `I67` | `=G67/H67` | taux 4 | 31,44 % |
| `C70` | `=C12` | R_ELS | 42,76 kN |
| `D73` | `=C8` | L en cm (affichage) | 30 cm |
| `C75` | `=C16/10` | Y en cm | 8,5 cm |
| `C78` | `=($C$13+$C$14)/1000*$C$12` | Mmax (3ᵉ copie) | 3,207 kN·m |
| `C82` | `=C78/(C75*0.01-C76*0.01)/C73` | N_ELS = Mmax/((Y − jeu)·n) | 14,253 kN |
| `C83` | `=C70/C73` | V_ELS = R/n | 14,253 kN |
| `C85` | `=(C71/C70)*C82` | N_ELU = N_ELS·R_ELU/R_ELS | 22,967 kN |
| `C86` | `=(C71/C70)*C83` | V_ELU = V_ELS·R_ELU/R_ELS | 22,967 kN |
| `C88` | `=C83*10^3/(C74*C6)` | pression diamétrale V_ELS/(d·e) | 59,389 N/mm² |
| `C90` | `=C83*1000/(2*(C5-C16)*C6)` | arrachement V_ELS/(2·(haut. − Y)·e) | 13,575 N/mm² |

**Structure du classeur (certain)** [s01 §1] :

- `P8` : 101 formules (93 numériques, 8 de texte), 137 constantes ; zone d'impression A1:I104, saut de page après la ligne 67 (page 1 : sections ; page 2 : ancrages).
- `E10A BD` : 37 formules, 38 constantes ; **aucune formule hors de cette feuille ne la lit**.
- `CORNIERES` : 352 constantes, aucune formule.
- Dessin : 24 zones de texte liées à des cellules (dont `C21`, vide : elle n'affiche rien) et 4 étiquettes fixes (les numéros des sections).
- Listes de validation : `B3` ← `CORNIERES!A2:A59` ; `C3` ← `CORNIERES!G2:G3` = {235 ; 355}, déclarée deux fois. Le nom défini `ListeCorniere` (`CORNIERES!A2:A56`) n'est utilisé par aucune formule.
- Mises en forme conditionnelles : celles des taux (`D3`, `I37`, `I47`, `I57`, `I67`) fonctionnent ; celles des contraintes et de `C88`, `C90` comparent à `#REF!`.
- Unités affichées : `C8` et `C76` en « cm », `C13` à `C17` et `C25` en « mm », `C12` en « kN », `C3` en « S355 ».

## Annexe B — Comparaison des catalogues

Feuille `CORNIERES` face au catalogue du site (`acier/donnees/profiles.json`, familles L et LI, 233 cornières) [s08] :

| Constat | Nombre | Détail |
|---|---|---|
| Lignes / désignations | 57 / 42 | plus « NON STANDARD » |
| Désignations en double (ailes échangées) | 15 | toutes inégales ; la seconde ligne n'est jamais lue |
| Présentes au site | 25 désignations (26 lignes) | ailes et épaisseur identiques sur les 26 lignes |
| Absentes du site | 17 désignations (31 lignes) | L100x75x9, L120x80x11, L30x20x3, L30x30x5, L35x20x3.5, L35x35x3, L35x35x3.5, L40x25x4, L45x30x4, L50x30x5, L50x40x5, L60x40x5, L70x50x6, L80x50x7, L80x60x7, L80x60x8, L90x60x8 |
| Rayon = congé r1 | 22 lignes | ex. L120x120x15 : 13 |
| Rayon = bout d'aile r2 | 2 lignes | **L150x100x12 : 6 au lieu de r1 = 12** (côté sûr : 143,3 % au lieu de 128,3 % en section 1, §6.2) |
| Rayon différent des deux | 2 lignes | L25x25x3 et L20x20x3 : 4 au lieu de r1 = 3,5 |
| Masse (non utilisée) | 26 lignes | de −4,17 % (L150x100x12 : 21,6 contre 22,541 kg/m) à +0,54 % (L25x25x3) |

Lignes jamais atteintes par `VLOOKUP` : 7, 11, 15, 18, 21, 23, 25, 28, 32, 36, 38, 42, 46, 51, 56 (L150x100x12, L120x80x11, L100x75x9, L90x60x8, L80x60x8, L80x60x7, L80x50x7, L70x50x6, L60x40x5, L50x40x5, L50x30x5, L45x30x4, L40x25x4, L35x20x3.5, L30x20x3). Les rayons des inégales absentes du site (5,5 pour L120x80x11, 4 pour L90x60x8…) ont l'allure de rayons de bout d'aile ; ils n'ont pas été vérifiés un par un. La colonne G (« ACIER ») n'est pas une propriété des cornières : c'est la liste {235 ; 355} de la validation de `C3`.

**Proposition** : une seule source, le catalogue du site, avec r1 pour le congé, et l'orientation comme donnée séparée de la désignation.

## Annexe C — Feuille `E10A BD` et origine des réactions

La feuille donne, par volée, le poids (masse/100) et la charge d'exploitation (surface × 4 kN/m²), puis leur moitié par mètre de largeur (diviseur 1,345 m, interprété comme la largeur de la volée) [s06 §2] :

| Volée | G (kN/m) | Q (kN/m) | G + Q | 1,35·G + 1,5·Q | Rapport |
|---|---|---|---|---|---|
| 1 | 4,403 | 2,662 | 7,065 | 9,937 | 1,4065 |
| 2 | 10,310 | 6,156 | 16,466 | 23,152 | 1,4061 |
| 4 | 4,125 | 2,647 | 6,772 | 9,539 | 1,4086 |
| 6 | 10,215 | 6,156 | 16,371 | 23,025 | 1,4064 |
| 7 | 8,494 | 5,115 | 13,610 | 19,140 | 1,4064 |
| 9 | 10,323 | 6,156 | 16,480 | 23,171 | 1,4060 |
| 10 | 13,488 | 7,197 | 20,685 | 29,004 | 1,4022 |
| 12 | 21,535 | 9,130 | 30,665 | **42,768** | 1,3947 |

- **Aucune formule de `P8` ne lit cette feuille** (certain). Le libellé `A12` (« Réaction max (ELS) : p. 15 ») renvoie à une page d'un autre document ; `C12` ne contient que la valeur 42,76.
- 1,35·G + 1,5·Q de la volée 12 = **42,7678 kN/m**, à +0,018 % de `C12` = 42,76 : coïncidence ou réaction ELU par mètre prise pour une ELS ? Le classeur ne permet pas de trancher.
- La moitié du total de la volée 12, (poids + charge)/2 = 41,245 kN (−3,543 %) ; celle des volées 10 et 12 réunies, 69,066 kN, est à +0,24 % de 68,9 kN, mais c'est une somme non pondérée.
- Rapporté à 0,30 m, 42,76 kN fait 142,5 kN/m, près de cinq fois la plus grande réaction linéaire de la feuille (30,67 kN/m) : si `C12` est une réaction concentrée d'un modèle d'escalier, la largeur d'appui réelle devient la vraie question (§5.3).

## Annexe D — Scripts, sorties, accords et désaccords

### D.1 Les scripts de ce document

Dossier : `/tmp/claude-0/-home-user/092afadd-796a-54d4-b802-037ae1cad731/scratchpad/corniere_excel/synthese/` (sorties dans `synthese/sorties/`, figures dans `figures/`). Relance complète : `sh synthese/lancer_tout.sh` (chaque script est lancé depuis `/tmp` par `/usr/bin/python3.12 -I -B`).

| Script | Contenu | Sortie |
|---|---|---|
| `s_commun.py` | cas de contrôle, méthode en symboles, interpréteur des formules de `P8`, congé t(s), corps libre | — |
| `s01_classeur.py` | structure, formules recalculées, symboles face au cache, 592 saisies, annexe A | `s01_classeur.txt` / `.json` |
| `s02_controle.py` | sections, ancrages, moments, jeux, pivots, aile verticale, sensibilités, largeur d'appui, domaine, EC3, moteur du site | `s02_controle.*` |
| `s03_continus.py` | diagrammes continus, maxima, seuil « e + r », généralisation aux catalogues | `s03_continus.*`, `s03_courbes.json` |
| `s04_retournee.py` | configuration B, jeux en B, ailes inégales | `s04_retournee.*` |
| `s_longi.py`, `s05_longue.py` | modèle longitudinal (repris de `extensions/e03_longitudinal.py`), cornière de 2 m, torsion ×1 000, cisaillement et poutre continue, contact en tête (pénalité) | `s05_longue.*`, `s05_courbes.json` |
| `s06_elu.py` | ELS/ELU, feuille `E10A BD`, efforts pour Hilti, ELU de la cornière longue | `s06_elu.*` |
| `s07_maillage.py` | maillage de la coupe (prototype `extensions/ef_l.py`), sections, pic, B, mur souple (balayage 10⁵ à 5 N/mm³), levier (maillage fin) | `s07_maillage.*`, `s07_figure.json` |
| `s08_catalogue.py` | catalogue `CORNIERES` face au catalogue du site | `s08_catalogue.*` |
| `svg.py`, `f_a_coupe.py` … `f_e_maillage.py`, `rendre_png.cjs` | figures a à e (SVG, puis PNG par Chromium) | `figures/fig_*.svg`, `.png` |
| `s09_reprises.py` | chiffres repris des autres analyses, relus dans leurs sorties (support souple du modèle à rondelle rigide, position de la résultante d'ancrage, second maillage, seuil sans cisaillement, temps Pyodide, cornière PRS) | `s09_reprises.*` |
| `assembler.py`, `verifier_chiffres.py` | assemblage de ce document (annexe A) ; contrôle que chaque nombre cité figure dans une sortie de script | — |

Chiffres repris d'autres scripts du dossier, relus dans leurs sorties par `s09_reprises.py` : support souple (`scripts/c03c_mur.py`), convergence du second maillage (`scripts/c03b_convergence.py`), seuil sans cisaillement (`scripts/c05_cisaillement.py`), temps dans Pyodide (`extensions/e06_pyodide.mjs`), propriétés d'une cornière PRS (`extensions/e07_architecture.py`). Le contrôle `verifier_chiffres.py` retrouve chacun des nombres du document dans une sortie imprimée de script (à deux décimales, un nombre pris au hasard n'y est retrouvé que 2 fois sur 100).

### D.2 Les trois analyses de départ

- `analyse_mecanique.md` (scripts `inventaire.py`, `catalogue.py`, `reproduction.py`, `complements.py`) ;
- `analyse_critique.md` (dossier `scripts/`, sorties `sorties/`) ;
- `analyse_extensions.md` (dossier `extensions/`).

### D.3 Accords et désaccords

- **Accord complet** sur la reproduction (M1 à M4, 88,8621 %), les trois jeux, les deux pivots, le maximum à ~2 mm dans le congé (+2,11 % / +1,45 %), le catalogue (15 doublons, r2 de la L150×100×12), l'ELU (1,611 > 1,50), le moteur actuel du site (120,65 %).
- **« 92,85 % » (analyse mécanique) contre « 80,8 à 84,0 % » (analyse critique)** : le premier ajoute l'effort normal au moment du classeur, qui suppose déjà le cisaillement sur l'axe ; c'est un **double compte** (« correction » partielle à ne pas faire). Un équilibre cohérent donne 76,91 % (pivot à l'arête) à 83,96 % (pivot à e/2), et 92,53 % si le pivot remonte à 15 mm [s02 §5].
- **Seuil de Y sous lequel le classeur n'est plus sûr** : 38,25 mm dans l'analyse critique, qui compare flexion et effort normal sans le cisaillement (formule 1,25·e + 1,5·r, retrouvée ici) ; avec le cisaillement τ = H/A, il passe à 43,5 mm (P1), et il vaut 33,2 mm (P0) ou 33,6 mm (Q1) selon l'équilibre retenu [s02 §5].
- **Généralisation du congé** : les rapports diffèrent parce que les bases diffèrent (catalogue du classeur ou du site, rapport au pied du congé ou au maximum de S1 et S2) ; les deux sont donnés au §4.2.
- **Traction en configuration B** : 82,47 kN (contact au bout libre, confirmé par le maillage) ; 128,3 kN dans l'analyse critique, qui plaçait le contact 10 mm au-dessus du bout libre avec le cisaillement au nu du mur — c'est une sensibilité (115,45 à 128,28 kN selon ces choix [s04 §1]), pas le cas de référence.
- **Traction maillée** : 33,90 kN (ancrage sous l'écrou, extensions) contre 40,8 kN (rondelle Ø30 rigide et cheville de 0,1 mm, critique) ; les deux restent sous les 42,76 kN du classeur, l'écart vient de la modélisation de l'ancrage, qui devra être déclarée. La rondelle rigide place la résultante d'ancrage au bord bas de la rondelle sur mur rigide (y_T = 70,8 mm) et au bord haut à 30 N/mm³ (99,0 mm) : c'est ce saut qui faisait dépasser le classeur de 3,8 % dans l'analyse critique ; avec l'écrou, la section 3 reste 3,3 % sous le classeur [s07, s09].
- **Cornière longue** : géométries différentes (ancrages à 100/1 000/1 900 mm dans l'analyse critique, 200/1 000/1 800 mm dans les extensions) ; seule la seconde, qui compte la torsion, est retenue ici.

### D.4 Relecture adverse du 07/11/2026 : ce qui a été corrigé, ce qui est discuté

Chaque remarque du relecteur a été recalculée indépendamment (dossier `verification_relecture/`, scripts `v01` à `v07`, lancés par `/usr/bin/python3.12 -I -B` depuis `/tmp`), puis le document, les scripts de synthèse et les figures ont été corrigés.

| N° | Remarque | Verdict | Correction portée |
|---|---|---|---|
| 1 | Support souple : le « +3,8 % » | **Fondée**, une réserve (ci-dessous) | §0, §1.3, §7, D.3 ; `s07` (balayage du mur), `s09` (résultante d'ancrage) |
| 2 | « Effet de levier permis » dans le modèle longitudinal | **Fondée** | ligne retirée du §5.3, §5.4, §7 ; `s05` §2d (pénalité) |
| 3 | Cisaillement « quelle que soit la raideur » | **Fondée** | §5.2 ; `s05` §2c ; légende de la figure d |
| 4 | 120,65 % du module du site | **Fondée** | §0, §3.4, §8.1 ; `s02` §9 |
| 5 | « Le plus haut possible » en configuration A | **Fondée** | §6.1 ; `s04` §1b |
| 6 | Torsion ×1 000 vérifiée avec des ancrages souples | **Fondée** | §5.2 ; `s05` §2b |
| 7 | Tableau de levier sur deux maillages | **Fondée** | §1.3 refait au maillage fin ; `s07` |
| 8 | Conséquence locale sans formules, bras incohérent | **Fondée** | §5.3, §5.4 ; `s05` §1, §3 |
| 9 | Seuil « e + r » unique de 33,23 mm | **Fondée** | §0, §4.3, §7 ; `s03` §2 |
| 10 | « 1,611 impossible » sans K_FI | **Fondée** | §0, §5.5, §7 ; `s06` §1 |
| 11 | « Ne lisent que e, r, X, Y » ; `A21` et `C90` | **Fondée** | §2.1, §6.2 ; `s01` §1 |
| 12 | « 233 cornières EN 10056-1 » | **Fondée** | §4.2, §8.1 ; `s08` |
| 13 | Repères 1 et 4 de la figure e | **Fondée** | figure e redessinée : chaque numéro au bout de sa coupe, comme la figure a |
| 14 | « `C12` porte p. 15 » | **Fondée** | annexe C ; `s01` §1 |

**Point discuté (remarque 1).** Le relecteur écrit : « le classeur n'est dépassé que si la résultante d'ancrage monte au-dessus de l'axe (rondelle rigide) ». C'est vrai à 30 N/mm³, mais pas en général : avec l'effort d'ancrage sur son axe (écrou ± 3 mm), la section 3 nominale atteint le classeur vers **16 N/mm³** et le dépasse de 2,2 % à 10 N/mm³ et de 5,0 % à 5 N/mm³ (pivot à 14,38 et 18,06 mm) ; en poutre, il suffit que le pivot monte à 11,94 mm [s07]. La conclusion du §0 (« le classeur cesse d'être du côté sûr si le support est très souple ») est donc maintenue, avec ce seuil chiffré au lieu du « +3,8 % » retiré.

**Écarts de chiffres sans effet sur les conclusions.** Remarque 1 : la section 3 prend 8,9 points de σ_adm entre mur raide et 30 N/mm³ (le relecteur : « environ 10 »). Remarque 3 : le document cite la plage du modèle à flexion biaxiale et torsion (V1 = 29,84 à 30,29 kN), non la poutre en flexion verticale seule ni la ligne retirée. Remarque 6 : avec ce modèle, G·It × 1 000 et ancrages rigides donnent 23,46 / 10,50 / 0,00 kN (le relecteur, avec son propre modèle de poutre, trouve une répartition voisine, tout aussi éloignée de « R/n »). Remarque 11 : `D5` est vide dans le classeur ; `C5` n'est lue que par `A21` et `C90`.
