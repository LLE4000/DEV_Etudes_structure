# Classeur REALEX « Dimensionnement cornières » : analyse mécanique et reproduction

*Lecture du 07/11/2026. Classeur : « REALEX Dimensionnement corniereS E10A#0 02-04-26.xlsx » (en-tête imprimé : « Escaliers E10A, Paliers 02/05/08, REALEX, Dimensionnement des cornières support escaliers, 03-04-2026, IND 0 »). Cette partie de l'analyse porte sur la **mécanique** et sur la **reproduction** du calcul. Elle ne traite pas la cornière retournée, la cornière longue à plusieurs ancrages ni la note finale.*

**Comment lire ce rapport.** Chaque chiffre vient d'un script Python lancé sur le classeur. Aucun n'est estimé. Les sorties brutes sont dans `sorties/` (le fichier et la rubrique sont cités entre crochets, par exemple [reproduction §A]). Deux niveaux de certitude sont distingués partout :

- **CERTAIN** : ce qui se lit dans une formule, un format de cellule ou la structure du fichier ;
- **INTERPRÉTÉ** : le sens physique qu'on en déduit. C'est une lecture d'ingénieur, à confirmer par l'auteur de la note.

La « valeur en cache » d'une cellule est la valeur qu'Excel a enregistrée dans le fichier lors de son dernier calcul.

| Script | Rôle | Sortie |
|---|---|---|
| `inventaire.py` | toutes les cellules, leurs formules, qui lit quoi, cellules isolées, liens du dessin, validations, feuille E10A BD | `sorties/inventaire.txt` |
| `catalogue.py` | catalogue CORNIERES face au catalogue du site (`acier/donnees/profiles.json`) | `sorties/catalogue.txt` |
| `reproduction.py` | modèle recalculé à partir des symboles, comparé à chaque valeur en cache ; formules du classeur réévaluées ; effet des jeux | `sorties/reproduction.txt` |
| `complements.py` | ce que le classeur ne calcule pas : pivots, effort normal, pic dans le congé, ELU, ancrages, domaine, E10A BD, moteur actuel du site | `sorties/complements.txt` |

Pour tout relancer : `cd /tmp && /usr/bin/python3.12 -I -B <dossier>/<script>.py`. Le classeur et les fichiers du site sont seulement lus. `complements.py` vérifie lui-même qu'il n'a écrit aucun fichier sous `/home/user/integ/acier` (résultat : aucun).

---

## 0. L'essentiel

1. **La reproduction est exacte.** Le modèle a été réécrit à partir des seuls symboles du schéma, en N et en mm. Il redonne les 93 valeurs numériques de P8 avec un écart relatif d'au plus 2,7·10⁻¹⁶ (cellule G51), soit le bruit d'arrondi de l'ordinateur. Il redonne aussi M1 = 2,00972, M2 = 2,56560, M3 = 2,35870 et M4 = 2,89665 kN·m, et un taux maximal de 88,8621 % en section 3. Sur 500 cas tirés au hasard (cornière, nuance, longueur, charge, X, Xj, Y, Yj, ancrages), il retrouve le classeur réévalué sur 46 000 comparaisons, avec un écart d'au plus 9,6·10⁻¹⁶ [reproduction §A, §B2].
2. **Le modèle du classeur est un portique en L rigide.** La charge R est placée à X + Xj du nu du mur. Le nœud est au nu du mur, au plan moyen de l'aile horizontale. L'ancrage est à Y + Yj au-dessus du dessous de la cornière. L'équilibre donne l'effort horizontal H = R·(X + Xj)/(YH + Yj) = 41,38 kN. Les deux diagrammes de moment sont linéaires et se rejoignent au nœud à la même valeur, Mmax = 3,207 kN·m [complements §1].
3. **Les quatre moments diffèrent parce qu'ils sont quatre ordonnées de DEUX diagrammes différents.** Dans l'aile horizontale, l'effort est R et les bras sont 47 et 60 mm. Dans l'aile verticale, l'effort est H, différent de R (H/R = X/YH = 0,968), et les bras sont 57 et 70 mm. On obtient M2 − M1 = R·r = 0,556 kN·m, M4 − M3 = H·r = 0,538 kN·m et M3/M1 = 1,174. La section 3 gouverne parce qu'elle porte le plus grand moment des deux sections d'épaisseur e [reproduction §A].
4. **Le classeur utilise trois positions différentes du point de rotation (le « pivot »).** Pour les sections 3 et 4, il est à e/2 = 7,5 mm du dessous : traction de 13,79 kN par ancrage. Pour les ancrages, il est à 1 cm (C76) : 14,25 kN par ancrage. Les 24,68 kN tapés depuis Hilti correspondent, si l'excentrement est le même, à un pivot à 15,2 mm, soit 15,32 kN par ancrage à l'ELS. Avec un pivot unique à 1 cm, la section 3 passe de 88,86 % à 91,81 % [complements §1].
5. **Dans l'aile verticale, le « τ » du classeur est en réalité l'effort normal.** Entre l'ancrage et l'angle, l'aile est tendue par R (σ_N = 9,50 N/mm²). L'effort tranchant y vaut H (τ = 9,20 N/mm²). En ajoutant σ_N à la flexion, le taux de la section 3 passe à 92,85 % (+4,5 %). En prenant en plus le pivot des ancrages, il passe à 95,81 % [complements §2].
6. **Les sections au pied du congé ne sont pas exactement le maximum.** Avec l'épaisseur réelle du congé, le pic de contrainte se trouve 1,7 à 2,0 mm avant le pied, côté angle. Il dépasse la section 1 de 2,1 % et la section 3 de 1,5 %. L'écart grandit quand la charge ou l'ancrage se rapproche du congé : jusqu'à +5 % à 40 mm du pied, +18 % à 20 mm et +51 % à 10 mm sur le catalogue du classeur. Les sections « e + r » (2 et 4) ont une épaisseur géométriquement exacte, mais elles ne gouvernent que si X ou Y est inférieur à 33,2 mm pour la L120×120×15 [complements §3].
7. **Le jeu de 20 mm demandé par défaut fait échouer le cas historique.** Avec Xj = 20 mm, le taux maximal passe à 112,4 % (section 3) et la section 1 passe à 107,8 %. Avec Yj = 20 mm, il passe à 95,4 %. Avec les deux, il passe à 120,7 %. Le cas historique atteint 100 % pour Xj = 9,46 mm ou pour Yj = 41,8 mm. Yj ne change pas les efforts d'ancrage, et Xj ne change pas leur cisaillement [reproduction §C].
8. **Le catalogue CORNIERES compte 42 désignations.** 15 cornières inégales y figurent deux fois sous le même nom, avec les ailes inversées : la seconde orientation est inaccessible, car RECHERCHEV (VLOOKUP) rend toujours la première ligne. 25 désignations existent au catalogue du site, avec des dimensions identiques, et 17 n'y sont pas. Le rayon est le congé intérieur r1 du site, sauf pour la L150x100x12 (6 mm au lieu de 12, du côté sûr). La méthode est très sensible au rayon : le cas de contrôle donne 88,9 % avec r = 13, 98,9 % avec r = 6,5 et 109,0 % avec r = 0 [catalogue].
9. **La feuille E10A BD n'est reliée à rien.** Aucune formule de P8 ne la lit, et Rmax = 42,76 kN ne s'en déduit pas : la combinaison simple la plus proche est 41,245 kN (−3,5 %). Le rapport ELU/ELS de la note, 68,9/42,76 = 1,611, dépasse 1,5, le plus grand coefficient d'EN 1990. L'ELU ne se déduit donc pas de l'ELS par des coefficients de combinaison [complements §8].
10. **Le module actuel du site ne suit pas la méthode du classeur.** Son modèle de bande d'acier de 1 m vérifie Mmax/W_e au nu du mur. Sur le cas de contrôle, il donne 120,65 % au lieu de 88,86 %, sans congé ni rapport Ye/YH [complements §9].

---

## 1. Le classeur en bref

| Feuille | Formules | Constantes (textes compris) | Rôle |
|---|---|---|---|
| `P8 ` (nom terminé par une espace) | 101 : 93 numériques et 8 de texte | 137 | la note : saisies, 4 sections, ancrages |
| `E10A BD` | 37 | 38 | descente de charges des volées d'escalier, sans lien avec P8 |
| `CORNIERES` | 0 | 352 | le catalogue (58 lignes) |

Éléments attachés à P8 [inventaire §6, §7], **CERTAIN** :

- **Le dessin.** C'est une image (le schéma fourni) avec 24 zones de texte **liées aux cellules** : elles affichent C4, C5, C6 (deux fois), C7, C12, C13, C14, C16, C17, C19, C20, C23, C24, C25, C35, C45, C57, C63, C65, C78, A20, A21 et C21. Quatre étiquettes fixes portent les numéros 1 à 4 des sections. Le dessin se met donc à jour avec les saisies.
- **Les listes de validation.** B3 (la cornière) propose CORNIERES!A2:A59, « NON STANDARD » compris. C3 (la nuance) propose CORNIERES!G2:G3, soit {235 ; 355}. Cette dernière validation est déclarée deux fois.
- **Le nom défini `ListeCorniere`** couvre CORNIERES!A2:A56. Aucune formule ne s'en sert, et il oublie L25x25x3, L20x20x3 et NON STANDARD.
- **Les mises en forme conditionnelles.** Celles de D3 et de I37, I47, I57 et I67 (taux ≤ 1 / > 1) fonctionnent. Celles des contraintes (G31…G67), de la pression diamétrale (C88) et de l'arrachement (C90) comparent à `#REF!` : la cellule de référence a été supprimée, et ces cellules ne sont plus colorées.
- **L'impression.** La zone imprimée est P8!A1:I104, avec un saut de page après la ligne 67 : page 1 pour les sections, page 2 pour les ancrages. La colonne M est masquée et vide.

---

## 2. Les cellules de saisie

**CERTAIN** : la liste des constantes lues par au moins une formule [inventaire §2] et les unités tirées des formats de cellule.

| Cellule | Libellé dans la feuille | Symbole | Unité | Cas | Lue par |
|---|---|---|---|---|---|
| P8!B3 | Choix de la cornière | désignation | — | L120x120x15 | C4:C7 (VLOOKUP) |
| P8!C3 | (format « S »0) | f_y | N/mm² | 355 | les 12 cellules d'admissible H31…H67 |
| P8!K4:K7 | CORNIERES NON STANDARD | larg., haut., e, r | mm | 120 / 120 / 15 / 13 | C4:C7 si B3 = « NON STANDARD » |
| P8!C8 | Longeur reprise | L | **cm** (format « 0" cm" ») | 30 | 13 formules, toujours sous la forme C8·10 → mm |
| P8!C10 | Coef sécurité « gM » | γ | — | 1,5 | les 12 admissibles |
| P8!C12 | Réaction max (ELS) : p. 15 | R | kN | 42,76 | 11 formules et le dessin |
| P8!C13 | Excentrement de la charge | X | mm | 75 | C19, C20, A20, C53, C55, C63, C78 |
| P8!C14 | Jeu horizontal | Xj | mm | 0 | C34, C44, C54, C55, C63, C78 |
| P8!C16 | Hauteur ancrage | Y | mm | 85 | C23, C24, C25, A21, C75, C90 |
| P8!C17 | Jeu vertical | Yj | mm | 0 | **seulement** C56 et C64 |
| P8!C71 | ELU (logiciel HILTI) | R_ELU | kN | 68,9 | C85, C86 |
| P8!C73 | Nombre d'ancrages pour reprendre la réaction | n | — | 3 | C82, C83 |
| P8!C74 | Diamètre ancrage | d | mm | 16 | C88 |
| P8!C76 | Jeu : « pour dimensionner les ancrages » | jeu | **cm** | 1 | **seulement** C82 |

**Valeurs tapées et lues par rien** [inventaire §3, §4] :

- **G85 = 24,68** et **G86 = 22,97**. Leur format les affiche « OK, on a dans Hilti : 24,68 kN ». Le « OK » fait partie du format : il s'affiche quelle que soit la valeur, ce n'est pas un contrôle.
- **L22 = 3,7 × 0,19 × 1 230 = 864,69**. Cette cellule est isolée, hors de la zone imprimée, et n'a aucun lien avec la cornière. Son sens est inconnu.

**Trois jeux distincts (CERTAIN)** : Xj (C14) n'entre que dans les moments, Yj (C17) que dans les rapports ρ3 et ρ4, et le jeu des ancrages (C76) que dans la traction d'ancrage. Ces trois grandeurs ne sont jamais combinées entre elles (§6.3).

---

## 3. Inventaire des formules de P8

Notations : L = 10·C8 = 300 mm (longueur reprise, en mm) ; R en kN dans le classeur ; moments en kN·m ; σ et τ en N/mm². Les valeurs sont celles en cache, retrouvées par le modèle [reproduction §A].

### 3.1 Géométrie

| Cellule | Formule Excel | En symboles | Unité | Valeur |
|---|---|---|---|---|
| C4 | `=IF($B$3="NON STANDARD",K4,VLOOKUP('P8 '!$B$3,CORNIERES!$A$2:$E$59,2,FALSE()))` | larg. = aile horizontale (colonne B du catalogue, ou K4) | mm | 120 |
| C5 | idem, colonne 3, ou K5 | haut. = aile verticale | mm | 120 |
| C6 | idem, colonne 4, ou K6 | e | mm | 15 |
| C7 | idem, colonne 5, ou K7 | r (congé) | mm | 13 |
| C19 | `=$C$13-$C$7-$C$6` | Xe = X − r − e | mm | 47 |
| C20 | `=$C$13-$C$6` | Xe+r = X − e | mm | 60 |
| A20 | `=C4-C13` | larg. − X (de la charge au bord libre ; dessin seulement) | mm | 45 |
| A21 | `=C5-C16` | haut. − Y (de l'ancrage au bord supérieur ; dessin seulement) | mm | 35 |
| C23 | `=$C$16-$C$6-$C$7` | Ye = Y − e − r | mm | 57 |
| C24 | `=C16-$C$6` | Ye+r = Y − e | mm | 70 |
| C25 | `=$C$16-$C$6/2` | YH = Y − e/2 | mm | 77,5 |

### 3.2 Section 1 : aile horizontale, épaisseur e (pied du congé)

| Cellule | Formule | En symboles | Unité | Valeur |
|---|---|---|---|---|
| C31 | `=$C$6` | t₁ = e | mm | 15 |
| C32 | `=$C$8*10*C31^2/6` | W₁ = L·e²/6 | mm³ | 11 250 |
| C33 | `=$C$19` | Xe | mm | 47 |
| C34 | `=C14` | Xj | mm | 0 |
| C35 | `=$C$12*(($C$33+$C$34)/1000)` | **M1 = R·(Xe + Xj)** | kN·m | 2,00972 |
| F31, F32, F34 | `=C35`, `=C32`, `=$C$12` | copies d'affichage de M1, W₁, R | | |
| G31 | `=F31*10^6/F32` | σ₁ = M1/W₁ | N/mm² | 178,642 |
| H31 | `=C3/$C$10` | σ_adm = f_y/γ | N/mm² | 236,667 |
| I31 | `=G31/H31` | σ₁/σ_adm (affiché, hors synthèse) | — | 75,48 % |
| G34 | `=($C$12)*1000/($C$8*10*$C$31)` | τ₁ = R/(L·e) | N/mm² | 9,502 |
| H34 | `=C3/$C$10/SQRT(3)` | τ_adm = f_y/(γ·√3) | N/mm² | 136,640 |
| I34 | `=G34/H34` | τ₁/τ_adm (affiché, hors synthèse) | — | 6,95 % |
| G37 | `=(G31^2+3*G34^2)^0.5` | σ_éq,1 = √(σ₁² + 3·τ₁²) | N/mm² | 179,398 |
| H37, I37 | `=C3/$C$10`, `=G37/H37` | σ_adm ; **taux 1** | | 75,80 % |
| F35, F37 | texte | « 300 x 15 » ; « √(178,6² + 3 × 9,5²) = » | | |

### 3.3 Section 2 : aile horizontale, épaisseur e + r (nu intérieur de l'aile verticale)

| Cellule | Formule | En symboles | Unité | Valeur |
|---|---|---|---|---|
| C41 | `=$C$6+$C$7` | t₂ = e + r | mm | 28 |
| C42 | `=$C$8*10*C41^2/6` | W₂ = L·(e+r)²/6 | mm³ | 39 200 |
| C43 | `=$C$20` | Xe+r (**le libellé B43 dit « Xe = »**) | mm | 60 |
| C44 | `=C14` | Xj | mm | 0 |
| C45 | `=$C$12*(($C$43+$C$44)/1000)` | **M2 = R·(Xe+r + Xj)** | kN·m | 2,56560 |
| G41 | `=F41*10^6/F42` | σ₂ = M2/W₂ | N/mm² | 65,449 |
| G44 | `=(F44)*1000/(C8*10*C41)` | τ₂ = R/(L·(e+r)) | N/mm² | 5,090 |
| G47, I47 | `=(G41^2+3*G44^2)^0.5`, `=G47/H47` | σ_éq,2 ; **taux 2** | | 66,040 ; 27,90 % |
| H41, H44, H47, I41, I44 | comme pour la section 1 | | | 236,667 ; 136,640 ; 236,667 ; 27,65 % ; 3,73 % |

### 3.4 Section 3 : aile verticale, épaisseur e (pied du congé)

| Cellule | Formule | En symboles | Unité | Valeur |
|---|---|---|---|---|
| C51 | `=$C$6` | t₃ = e | mm | 15 |
| C52 | `=$C$8*10*C51^2/6` | W₃ = L·e²/6 | mm³ | 11 250 |
| C53, C54 | `=$C$13`, `=C14` | X et Xj **affichés seulement** : C55 relit C13 et C14 directement | mm | 75 ; 0 |
| C55 | `=($C$13+$C$14)/1000*$C$12` | **Mmax = R·(X + Xj)** | kN·m | 3,207 |
| C56 | `=($C$23+$C$17)/($C$25+$C$17)` | ρ₃ = (Ye + Yj)/(YH + Yj) | — | 0,73548 |
| C57 | `=C56*C55` | **M3 = ρ₃·Mmax** | kN·m | 2,35870 |
| G51 | `=F51*10^6/F52` | σ₃ = M3/W₃ | N/mm² | 209,662 |
| G54 | `=($F$54)*1000/($C$8*10*$C$51)` | τ₃ = R/(L·e) | N/mm² | 9,502 |
| G57, I57 | `=(G51^2+3*G54^2)^0.5`, `=G57/H57` | σ_éq,3 ; **taux 3** | | 210,307 ; **88,86 %** |
| H51, H54, H57, I51, I54 | comme pour la section 1 | | | 236,667 ; 136,640 ; 236,667 ; 88,59 % ; 6,95 % |

### 3.5 Section 4 : aile verticale, épaisseur e + r (niveau de la face supérieure de l'aile horizontale)

| Cellule | Formule | En symboles | Unité | Valeur |
|---|---|---|---|---|
| C61 | `=$C$6+$C$7` | t₄ = e + r | mm | 28 |
| C62 | `=$C$8*10*C61^2/6` | W₄ = L·(e+r)²/6 | mm³ | 39 200 |
| C63 | `=($C$13+$C$14)/1000*$C$12` | Mmax (deuxième copie) | kN·m | 3,207 |
| C64 | `=($C$24+$C$17)/($C$25+$C$17)` | ρ₄ = (Ye+r + Yj)/(YH + Yj) | — | 0,90323 |
| C65 | `=C64*C63` | **M4 = ρ₄·Mmax** | kN·m | 2,89665 |
| G61 | `=F61*10^6/F62` | σ₄ = M4/W₄ | N/mm² | 73,894 |
| G64 | `=($F$64)*1000/($C$8*10*$C$61)` | τ₄ = R/(L·(e+r)) | N/mm² | 5,090 |
| G67, I67 | `=(G61^2+3*G64^2)^0.5`, `=G67/H67` | σ_éq,4 ; **taux 4** | | 74,418 ; 31,44 % |

### 3.6 Synthèse

| Cellule | Formule | En symboles | Valeur |
|---|---|---|---|
| D3 | `=MAX(I37,I47,I57,I67)` | taux max = max des quatre σ_éq/σ_adm | **88,86 %** |

**CERTAIN** : D3 ne retient que les quatre taux en contrainte équivalente. Les taux de flexion seule (I31…I61) et de cisaillement seul (I34…I64) sont affichés mais n'entrent pas dans la synthèse. Comme σ_éq ≥ σ et que σ_éq ≥ √3·τ, rien n'est perdu.

### 3.7 Ancrages

| Cellule | Formule | En symboles | Unité | Valeur |
|---|---|---|---|---|
| C70 | `=C12` | R_ELS | kN | 42,76 |
| D73 | `=C8` | « pour une longueur de 30 cm » (affichage) | cm | 30 |
| C75 | `=C16/10` | Y en cm | cm | 8,5 |
| C78 | `=($C$13+$C$14)/1000*$C$12` | Mmax (troisième copie) | kN·m | 3,207 |
| C82 | `=C78/(C75*0.01-C76*0.01)/C73` | **N_ELS = Mmax/((Y − jeu)·n)** | kN | 14,2533 |
| C83 | `=C70/C73` | **V_ELS = R/n** | kN | 14,2533 |
| C85 | `=(C71/C70)*C82` | N_ELU = N_ELS·R_ELU/R_ELS | kN | 22,9667 |
| C86 | `=(C71/C70)*C83` | V_ELU = V_ELS·R_ELU/R_ELS | kN | 22,9667 |
| C88 | `=C83*10^3/(C74*C6)` | pression diamétrale p = V_ELS/(d·e) | N/mm² | 59,389 |
| C90 | `=C83*1000/(2*(C5-C16)*C6)` | arrachement τ = V_ELS/(2·(haut. − Y)·e) | N/mm² | 13,575 |

### 3.8 Anomalies de forme relevées (CERTAIN)

- **B43** affiche « Xe = » à côté de la valeur Xe+r (60 mm).
- **C53 et C54** affichent X et Xj dans le bloc de la section 3, mais aucune formule ne les lit. Le bloc de la section 3 ne montre pas Yj (C17), qui intervient pourtant dans C56.
- La zone de texte **« ZoneTexte 4 » du dessin est liée à C21, une cellule vide** : elle n'affiche rien.
- Le dessin affiche **YH = 78 mm** (format sans décimale) alors que le calcul utilise 77,5 mm. Un vérificateur qui relit sur le dessin trouvera un écart.
- **C88 et C90 ne sont comparées à aucune limite** : leurs mises en forme conditionnelles visent `#REF!`.
- **Mmax est calculé trois fois** (C55, C63, C78) par la même formule. Les trois copies sont identiques, mais rien ne les lie.

---

## 4. La feuille E10A BD

**Ce qu'elle calcule (CERTAIN)** [inventaire §8]. Une ligne par volée d'escalier : 1, 2, 4, 6, 7, 9, 10 et 12.

| Colonne | Formule | Lecture |
|---|---|---|
| B | saisie | « Masse » (kg) |
| C | `=B/100` | « Poids » : kg/100, soit des kN avec g pris à 10 m/s² (**interprété**) |
| D | `=C/2/$D$3`, avec D3 = 1,345 sous l'en-tête « SCIA » | demi-poids divisé par 1,345 : réaction permanente par mètre (**interprété** : 1,345 m serait la largeur de la volée) |
| E | saisie | « Surface » (m²) |
| F | `=E*$F$2`, avec F2 = 4 | « Ch TOT expl » : surface × 4 (**interprété** : 4 kN/m² de charge d'exploitation) |
| G | `=F/2/$G$3`, avec G3 = 1,345 | demi-charge d'exploitation par mètre |

La volée 12 est la plus chargée : D + G = 30,67 kN/m, et (C + F)/2 = 41,245 kN par extrémité.

**Cellules isolées** : I10 = 3,29·4 = 13,16 ; I13 = 6,59·2 = 13,18 ; H15 = 36,28/2 + 8,7 = 26,84 ; I15 = 10,73 + 11,54 = 22,27 ; H16 = 5,58·4 = 22,32. Elles vont **par paires presque égales** : I13/I10 − 1 = +0,15 % et I15/H16 − 1 = −0,22 % (**interprété** : recoupements à la main de réactions SCIA). Les libellés A13, A31 et A48 (« avec appuis à 30 cm et 70 cm / 60 cm / 80 cm ») n'ont rien en dessous, et la feuille ne contient aucun dessin. Il y avait sans doute des captures d'écran SCIA, qui ont été retirées (**interprété**).

**Lien avec P8 : aucun (CERTAIN).** Aucune formule hors de E10A BD ne la lit [inventaire §5]. Rmax (P8!C12 = 42,76 kN) porte la mention « p. 15 », une page d'une autre note. Les combinaisons simples des valeurs de la feuille ne redonnent pas 42,76 kN : la plus proche est la volée 12, avec (G + Q)/2 = 41,245 kN, soit −3,54 % [complements §8]. Rapporté à 0,30 m, 42,76 kN fait 142,5 kN/m, près de cinq fois la plus grande réaction linéaire de la feuille (30,67 kN/m). Rmax est donc vraisemblablement une **réaction concentrée** d'un modèle SCIA, et non une réaction répartie (**interprété**).

**L'ELU de la note.** 68,9 kN (C71, « logiciel HILTI ») divisé par 42,76 kN donne **1,611**. C'est plus que 1,5, le plus grand coefficient partiel d'EN 1990 : cet ELU ne peut pas venir de l'ELS par une combinaison 1,35 G + 1,5 Q. Pour la volée 12, cette combinaison donnerait un rapport de 1,395. La somme (G + Q)/2 des volées 10 et 12 vaut 69,07 kN (+0,24 %), mais c'est une somme non pondérée, donc probablement une coïncidence. L'origine de 68,9 kN reste hors du classeur.

---

## 5. Le catalogue CORNIERES et le catalogue du site

**Colonnes (CERTAIN).** A Nom ; B « Largeur » (aile horizontale, lue en C4) ; C « Hauteur » (aile verticale, C5) ; D « Épaisseur » (e, C6) ; E « rayon » (r, C7) ; F « Poids kg/m », **non utilisé** ; G « ACIER », qui n'est **pas une propriété de la cornière** mais la liste {235 ; 355} de la validation de C3. Les en-têtes ne donnent pas d'unité : les millimètres se déduisent des formats de C4:C7 (« 0" mm" »).

**Contenu** [catalogue] : 57 cornières et « NON STANDARD », soit **42 désignations distinctes**.

- **15 désignations figurent deux fois**, avec les ailes inversées. Ce sont toutes des inégales : L150x100x12, L120x80x11, L100x75x9, L90x60x8, L80x60x8, L80x60x7, L80x50x7, L70x50x6, L60x40x5, L50x40x5, L50x30x5, L45x30x4, L40x25x4, L35x20x3.5 et L30x20x3. VLOOKUP en correspondance exacte rend **toujours la première ligne** : la L150x100x12 est toujours prise avec 150 mm à l'horizontale et 100 mm à la verticale. **L'orientation inverse est inaccessible par le nom (CERTAIN).** Seule la saisie « NON STANDARD » permet de la calculer.

**Face au catalogue du site** (`acier/donnees/profiles.json`, familles L et LI, 233 cornières) [catalogue] :

| Constat | Nombre | Détail |
|---|---|---|
| désignations présentes au site | 25 (26 lignes) | ailes et épaisseur **identiques** pour toutes |
| désignations absentes du site | 17 (31 lignes) | L100x75x9, L120x80x11, L30x20x3, L30x30x5, L35x20x3.5, L35x35x3, L35x35x3.5, L40x25x4, L45x30x4, L50x30x5, L50x40x5, L60x40x5, L70x50x6, L80x50x7, L80x60x7, L80x60x8, L90x60x8 |
| rayon = congé intérieur r1 du site | 22 lignes | par exemple L120x120x15 : 13 = r1 (r2 = 6,5) |
| rayon = rayon d'extrémité r2 du site | 2 lignes | **L150x100x12 : 6 au lieu de r1 = 12** |
| rayon différent des deux | 2 lignes | L25x25x3 et L20x20x3 : 4 au lieu de r1 = 3,5 (+0,5 mm) |
| masse (colonne F, non utilisée) | 26 lignes | de −4,17 % (L150x100x12 : 21,6 contre 22,541 kg/m) à +0,54 % (L25x25x3) |

**La méthode est sensible au rayon (CERTAIN, calculé).** Le rayon déplace les sections 1 et 3 vers le congé : il raccourcit Xe et Ye de r. Pour le cas de contrôle, le taux de la section 3 vaut 88,86 % avec r = 13 (r1), **98,94 %** avec r = 6,5 (r2) et **109,02 %** avec r = 0. Le rayon « rapporte » donc 20 points de taux : prendre le bon congé r1 est indispensable. Pour la L150x100x12 sous la charge du cas de contrôle, le rayon du classeur (6) donne 159,85 % contre 145,58 % avec r1 = 12. L'écart va **du côté sûr**. Pour les L25 et L20, le +0,5 mm va du côté non sûr, mais de façon négligeable.

---

## 6. Le cheminement des efforts

### 6.1 Le modèle reconstitué

Repère : u est la distance horizontale depuis le **nu du mur** (le dos de l'aile verticale), y la hauteur depuis le **dessous de la cornière**.

```
          mur                    aile verticale (épaisseur e)
          │▓▓▓│
   N ◄────┤   ├══●══ axe d'ancrage, à y = Y + Yj (sections) ou y = Y (ancrages)
   V ▲    │▓▓▓│  ║   M(y) = H·(Y + Yj − y), nul à l'ancrage
          │▓▓▓│  ║
          │▓▓▓│  ║                         R (charge ELS)
          │▓▓▓│  ╚═╗ congé r                │
   C ────►●▓▓▓│    ╚════════════════════════▼═══════╗ aile horizontale (épaisseur e)
  pivot P │▓▓▓│ ── plan moyen, y = e/2      M(u) = R·(X + Xj − u)
  (u = 0) └───┴──────────────────────────────────────╝ dessous de la cornière (y = 0)
              |<──────────── X + Xj ───────────>|
```

**Ce que les formules imposent (CERTAIN).**

- **Aile horizontale.** M1 et M2 valent R multiplié par la distance de la charge à la section, la charge étant à X + Xj du nu du mur. C'est un porte-à-faux chargé par R. Son moment croît linéairement, de 0 sous la charge à **R·(X + Xj) = Mmax** au nu du mur (C55 = C63 = C78).
- **Aile verticale.** M3 = Mmax·(Ye + Yj)/(YH + Yj) et M4 = Mmax·(Ye+r + Yj)/(YH + Yj). Cela revient à écrire M(y) = H·(Y + Yj − y), avec **H = Mmax/(YH + Yj)**. Le moment est nul à l'axe d'ancrage, croît linéairement vers le bas et vaut Mmax au niveau y = e/2.
- **Le nœud.** Les deux diagrammes valent Mmax au même point. Sur le dessin, c'est la ligne verte en L (nu du mur et plan moyen de l'aile horizontale) et ses deux pastilles vertes. Les zones de texte 18 et 19 y affichent 3,21 kN·m (C78 et C63).

**L'équilibre qui en découle (INTERPRÉTÉ, et vérifié sur les chiffres)** [complements §1]. On considère la cornière entière comme un corps rigide, chargée par :

- R vers le bas, à u = X + Xj ;
- l'ancrage, qui reprend une traction N horizontale vers le mur et un cisaillement V vertical vers le haut, à y = Y + Yj ;
- la compression C du mur sur le talon, concentrée au pivot P (u = 0, y = e/2).

Les trois équations d'équilibre donnent :

- ΣF_vertical : **V = R** ;
- ΣF_horizontal : **N = C = H** ;
- ΣM autour de P : **R·(X + Xj) = H·(YH + Yj)**. Le résidu calculé est −4,7·10⁻¹⁰ N·mm, soit zéro. On obtient H = 41,38 kN au total, 13,79 kN par ancrage.

Ce modèle explique pourquoi le moment de l'aile verticale est nul à l'ancrage : au-dessus de la section considérée, seuls agissent N et V, et V n'a pas de bras par rapport à l'aile. Il explique aussi pourquoi le moment croît en H·s, où s est la distance à l'ancrage. La charge descend ainsi : **R → aile horizontale (porte-à-faux) → congé → nœud d'angle → aile verticale (traction R et flexion H·s) → ancrage (V = R, N = H) et talon (C = H contre le mur)**.

### 6.2 Les bras de levier

| Symbole | Cellule | Définition exacte | Valeur | Sur le dessin | Sens physique |
|---|---|---|---|---|---|
| X | C13 | saisie | 75 mm | du dos de l'aile verticale (nu du mur) à la verticale de Rmax | position de l'appui de l'escalier sur l'aile |
| Xe | C19 | X − e − r | 47 mm | de la section 1 (pied du congé, à e + r du dos) à la charge | bras de R par rapport à la section 1 |
| Xe+r | C20 | X − e = Xe + r | 60 mm | de la section 2 (nu intérieur de l'aile verticale) à la charge | bras de R par rapport à la section 2 |
| Y | C16 | saisie | 85 mm | du dessous de la cornière à l'axe de l'ancrage | hauteur de l'ancrage |
| Ye | C23 | Y − e − r | 57 mm | de l'axe d'ancrage à la section 3 (pied du congé, à e + r au-dessus du dessous) | bras de H par rapport à la section 3 |
| Ye+r | C24 | Y − e = Ye + r | 70 mm | de l'axe d'ancrage à la section 4 (niveau de la face supérieure de l'aile horizontale) | bras de H par rapport à la section 4 |
| YH | C25 | Y − e/2 | 77,5 mm | de l'axe d'ancrage à la ligne verte (plan moyen de l'aile horizontale) | bras du couple ancrage–compression dans le modèle des sections |
| Y − jeu | C75 − C76 | (8,5 − 1) cm | 75 mm | non dessiné | bras du couple utilisé pour la traction d'ancrage |
| larg. − X | A20 | larg. − X | 45 mm | de la charge au bord libre | dessin seulement : aucun contrôle ne vérifie que la charge reste sur l'aile |
| haut. − Y | A21 | haut. − Y | 35 mm | de l'axe d'ancrage au bord supérieur | longueur des deux plans d'arrachement (C90) |

« Xe+r » et « Ye+r » se lisent « Xe plus r » et « Ye plus r » : ce sont des sommes, pas des indices (**CERTAIN** : C20 = C19 + C7 et C24 = C23 + C7).

### 6.3 Les jeux : trois grandeurs différentes

**Xj (C14), le jeu horizontal.**

- **CERTAIN** : il s'ajoute à TOUS les bras de l'aile horizontale (M1, M2) et à Mmax, donc à M3, M4 et à la traction d'ancrage. Il ne touche ni au cisaillement ni aux contraintes τ.
- **INTERPRÉTÉ, mais imposé par la formule** : Xj **déplace la charge vers l'extérieur**, et non la cornière. Une cale entre la cornière et le mur ne changerait pas le bras de R par rapport à la section 1, qui est intérieur à la cornière. Or le classeur l'allonge. Le dessin le confirme : la flèche Xj prolonge X au-delà de la ligne d'action de Rmax. Xj est donc une **tolérance sur la position de l'appui de l'escalier** (« jeu entre la position de référence de la charge et son application réelle »).

**Yj (C17), le jeu vertical.**

- **CERTAIN** : il n'entre que dans ρ3 = (Ye + Yj)/(YH + Yj) et ρ4. Il allonge d'autant la distance de l'ancrage à la section et au pivot. Il n'entre **pas** dans la traction d'ancrage (C82 lit Y sans Yj).
- **Sur le dessin**, la flèche Yj part de l'axe d'ancrage **vers le haut**.
- **INTERPRÉTÉ** : l'ancrage peut se trouver Yj **plus haut** que prévu par rapport à la cornière (tolérance de perçage, trou oblong). Un ancrage plus haut augmente M3 et M4, car ρ tend vers 1. Il diminue en revanche la traction, car le bras s'allonge. Ignorer Yj pour les ancrages est donc du côté sûr **si** le jeu ne joue que vers le haut. S'il joue dans les deux sens, l'ancrage plus bas n'est couvert que par C76.

**C76, le « jeu pour dimensionner les ancrages » (1 cm).**

- **CERTAIN** : il ne sert qu'à réduire le bras de la traction d'ancrage, de Y à Y − 1 cm.
- **INTERPRÉTÉ, avec deux lectures que la formule ne permet pas de départager** :
  - (a) la résultante de compression contre le mur est à 1 cm au-dessus du dessous de la cornière : c'est une **hypothèse de bras de levier** ;
  - (b) l'ancrage peut être posé 1 cm plus bas que Y : c'est une **tolérance de montage**, symétrique de Yj.
- Les deux lectures donnent la même formule. C'est exactement la confusion contre laquelle l'ingénieur met en garde (« une distance de montage ne doit pas être confondue avec un bras de levier d'ancrage »). À trancher avec l'auteur.

**Pourquoi il ne faut pas les fusionner (CERTAIN, calculé)** [reproduction §C]. Pour 20 mm, chacun agit sur des grandeurs différentes :

| Grandeur | Cas | Xj = 20 | Yj = 20 | Xj = Yj = 20 |
|---|---|---|---|---|
| M1 (C35) | 2,00972 | 2,86492 (+42,55 %) | inchangé | +42,55 % |
| M2 (C45) | 2,56560 | 3,42080 (+33,33 %) | inchangé | +33,33 % |
| Mmax (C55) | 3,20700 | 4,06220 (+26,67 %) | inchangé | +26,67 % |
| M3 (C57) | 2,35870 | 2,98768 (+26,67 %) | 2,53271 (+7,38 %) | 3,20810 (+36,01 %) |
| M4 (C65) | 2,89665 | 3,66908 (+26,67 %) | 2,96031 (+2,20 %) | 3,74972 (+29,45 %) |
| H (implicite) | 41,381 kN | 52,415 (+26,67 %) | 32,892 (−20,51 %) | 41,664 (+0,68 %) |
| taux 1 (I37) | 75,80 % | **107,83 %** | 75,80 % | 107,83 % |
| taux 3 (I57) | 88,86 % | **112,43 %** | 95,38 % | **120,69 %** |
| taux 4 (I67) | 31,44 % | 39,72 % | 32,13 % | 40,59 % |
| taux max (D3) | 88,86 % | 112,43 % | 95,38 % | 120,69 % |
| N_ELS (C82) | 14,253 kN | 18,054 (+26,67 %) | **inchangé** | 18,054 |
| V_ELS (C83) | 14,253 kN | inchangé | inchangé | inchangé |
| N_ELU (C85) | 22,967 kN | 29,091 | inchangé | 29,091 |

- Xj = +1 mm ajoute 1,18 point au taux maximal, Yj = +1 mm en ajoute 0,40.
- Le cas historique atteint 100 % pour **Xj = 9,455 mm** ou pour **Yj = 41,83 mm** (vérifié sur le classeur réévalué : D3 = 1,000000).
- Quand Yj grandit sans limite, ρ3 tend vers 1 et la section 3 tend vers 120,65 % : c'est la flexion entière Mmax sur l'épaisseur e.
- **Conséquence pour le jeu par défaut de 20 mm demandé** : appliqué en Xj, il fait passer le cas historique, validé à 88,86 %, à 112,43 %. Il faudra le signaler à l'ingénieur avant d'en faire une valeur par défaut.

### 6.4 Pourquoi M1, M2, M3 et M4 sont différents

Un moment est un effort multiplié par la distance de sa ligne d'action à la section, M = F·bras [reproduction §A].

- **M1 et M2** : même effort **R = 42,76 kN** (la charge), deux bras différents. Xe = 47 mm pour la section 1, au pied du congé, et Xe+r = 60 mm pour la section 2, au nu intérieur de l'aile verticale. On a M2 − M1 = R·r = 0,55588 kN·m.
- **M3 et M4** : l'effort n'est **pas R** mais **H = R·X/YH = 41,38 kN**, l'effort horizontal du couple ancrage–compression. Les bras se mesurent depuis l'ancrage : Ye = 57 mm et Ye+r = 70 mm. On a M4 − M3 = H·r = 0,53795 kN·m.
- **M3 et M1 sont tous deux au pied d'un congé, mais diffèrent** : M3/M1 = (H/R)·(Ye/Xe) = 0,96774 × 1,21277 = **1,17364**. L'ancrage (Ye = 57) est plus loin de la section 3 que la charge (Xe = 47) ne l'est de la section 1. C'est pourquoi la section 3 gouverne : à épaisseur égale (W₁ = W₃), elle porte le plus grand moment.
- **Ordre des moments** : M1 = 2,00972 < M3 = 2,35870 < M2 = 2,56560 < M4 = 2,89665 < Mmax = 3,20700 kN·m. Les sections 2 et 4 portent les plus grands moments, car elles sont plus près du nœud, mais leur module est (28/15)² = **3,484 fois** plus grand. Leurs contraintes ne valent que 0,366 (σ₂/σ₁) et 0,352 (σ₄/σ₃) de celles des sections minces.

---

## 7. Les contraintes

**Ce que fait le classeur (CERTAIN).**

- **Flexion** : σ = M/W, avec **W = L·t²/6**. C'est le module élastique d'un rectangle de largeur L = 300 mm (la « longueur reprise », C8 en cm × 10) et d'épaisseur t = e pour les sections 1 et 3, t = e + r pour les sections 2 et 4. Les quatre sections ont la même largeur L.
- **Cisaillement** : **τ = R/(L·t)**, un cisaillement **moyen** sur toute la section, avec le même effort R dans les quatre sections.
- **Contrainte équivalente** : **σ_éq = √(σ² + 3τ²)**, le critère de von Mises pour une contrainte normale et un cisaillement.
- **Admissibles** : **σ_adm = f_y/γ = 355/1,5 = 236,67 N/mm²** et **τ_adm = σ_adm/√3 = 136,64 N/mm²**. Le coefficient C10 est libellé « gM », mais il s'applique à des efforts **non majorés** (ELS) : c'est un coefficient global de méthode aux contraintes admissibles, pas le γ_M0 de l'EC3.

| Section | t (mm) | W (mm³) | M (kN·m) | σ | τ | σ_éq | taux |
|---|---|---|---|---|---|---|---|
| 1 | 15 | 11 250 | 2,00972 | 178,642 | 9,502 | 179,398 | 75,80 % |
| 2 | 28 | 39 200 | 2,56560 | 65,449 | 5,090 | 66,040 | 27,90 % |
| 3 | 15 | 11 250 | 2,35870 | 209,662 | 9,502 | 210,307 | **88,86 %** |
| 4 | 28 | 39 200 | 2,89665 | 73,894 | 5,090 | 74,418 | 31,44 % |

**Observations mécaniques (calculées)** [complements §2, §4] :

- **Le cisaillement ne pèse presque rien.** σ_éq/σ − 1 vaut +0,42 % (section 1), +0,90 % (section 2), +0,31 % (section 3) et +0,71 % (section 4). Combiner σ maximal (fibre extrême) avec τ moyen est une convention : à la fibre extrême, le cisaillement est nul. Au plan neutre, le cisaillement maximal vaut 1,5·τ = 14,25 N/mm², d'où σ_éq = 24,69 N/mm². Cela ne gouverne jamais.
- **Dans l'aile verticale, « τ » est en réalité un effort normal (INTERPRÉTÉ, d'après l'équilibre du §6.1).** Entre l'ancrage et l'angle, l'aile verticale transmet R **dans son axe**, en traction : σ_N = R/(L·e) = 9,502 N/mm². C'est le nombre que le classeur appelle τ (G54). Son effort tranchant vaut **H**, d'où τ_H = H/(L·e) = 9,196 N/mm². Les deux nombres sont voisins parce que X ≈ YH (75 contre 77,5 mm), ce qui est une coïncidence du cas. Avec σ = σ_M + σ_N et τ = τ_H, la section 3 donne 219,74 N/mm², soit **92,85 % (+4,49 %)**. Cela suppose que le frottement au contact du mur ne reprend rien de R.
- **Toutes les contraintes sont inversement proportionnelles à L** : le taux maximal vaut 106,63 % pour L = 250 mm, 88,86 % pour 300 mm et 76,17 % pour 350 mm. **INTERPRÉTÉ** : la note suppose la réaction répartie uniformément sur 300 mm de cornière. Si l'appui réel de l'escalier est plus étroit, la diffusion de l'effort dans les ailes devient la vraie question (ce sujet relève de la lecture « cornière longue »).

---

## 8. Les efforts d'ancrage

**Ce que fait le classeur (CERTAIN).**

- **Traction par ancrage** : N_ELS = Mmax/((Y − jeu)·n) = 3,207 / (0,075 × 3) = **14,2533 kN**. C'est l'équilibre des moments autour d'un point du nu du mur situé à **jeu = 1 cm au-dessus du dessous** : R·(X + Xj) = n·N·(Y − jeu). Avec la lecture (b) du §6.3, c'est le même point, mais compté autour de l'arête inférieure avec l'ancrage 1 cm plus bas.
- **Cisaillement par ancrage** : V_ELS = R/n = **14,2533 kN**, partagé également entre les ancrages, sans frottement.
- **ELU** : les deux efforts sont multipliés par R_ELU/R_ELS = 68,9/42,76 = 1,61132, d'où N_ELU = V_ELU = **22,9667 kN**. Ce passage suppose une répartition identique à l'ELS et à l'ELU.
- **N = V dans ce cas** parce que X = Y − jeu = 75 mm. C'est une **coïncidence**, pas une règle.

**Trois pivots dans le même classeur (CERTAIN pour les deux premiers, INTERPRÉTÉ pour le troisième)** [complements §1].

| Pivot (hauteur au-dessus du dessous) | Où | H total (kN) | N par ancrage ELS (kN) | M3 (kN·m) | taux 3 |
|---|---|---|---|---|---|
| 0 (arête inférieure) | aucune cellule, pour comparaison | 37,729 | 12,577 | 2,15058 | 81,07 % |
| **e/2 = 7,5 mm** | sections 3 et 4 (YH) | 41,381 | 13,794 | 2,35870 | **88,86 %** |
| **1 cm** | ancrages (C76) | 42,760 | **14,253** | 2,43732 | 91,81 % |
| 15,2 mm | ce que donnent les 24,68 kN de Hilti si l'excentrement vaut 75 mm et que la traction est également partagée | 45,950 | 15,317 | 2,61915 | 98,62 % |

- Le classeur prend un pivot plus haut pour les ancrages (bras plus court, traction plus forte) que pour l'aile, donc chaque vérification de son côté le plus défavorable. Mais **le même couple ne peut pas avoir deux bras**.
- La valeur Hilti (G85 = 24,68 kN) dépasse celle du classeur (C85 = 22,97 kN) de **6,94 %**. Pour le cisaillement, G86 = 22,97 et C86 = 22,9667 coïncident à 0,01 % près.
- Le « OK » qui précède 24,68 kN est du texte de format, pas un contrôle (§2).

**Vérifications locales de l'aile percée (CERTAIN)** :

- **Pression diamétrale** C88 = V_ELS/(d·e) = 59,39 N/mm² à l'ELS, soit 95,69 N/mm² une fois multipliée par 1,611 ;
- **Arrachement** C90 = V_ELS/(2·(haut. − Y)·e) = 13,57 N/mm², soit 21,87 N/mm² à l'ELU. Ce sont deux plans de cisaillement au-dessus du trou : V pousse l'aile vers le haut contre l'ancrage.
- **Ni l'une ni l'autre n'est comparée à une limite.**
- La longueur d'arrachement est prise jusqu'à l'**axe** du trou (35 mm). En retranchant le demi-trou (d0 = 18 mm, valeur supposée), elle tombe à 26 mm et la contrainte monte à 18,27 N/mm² (+34,6 %).

---

## 9. La reproduction (`reproduction.py`)

**Méthode.** Le modèle (fonction `modele`) est écrit en N et en mm à partir des seuls symboles : géométrie des sections, H, M1 à M4, σ, τ, σ_éq, taux, N, V et vérifications locales. Il n'emprunte aucune formule au classeur. Il est ensuite confronté de trois façons.

1. **À chaque valeur en cache de P8.** Les 93 cellules numériques correspondent toutes à une grandeur du modèle (les copies d'affichage pointent vers la même), sauf L22 qui est hors modèle. Le plus grand écart relatif vaut **2,71·10⁻¹⁶** (G51). Valeurs attendues par l'ingénieur : M1 2,009720, M2 2,565600, M3 2,358697, M4 2,896645 et taux 88,8621 %, en section 3.
2. **Aux formules du classeur, réévaluées telles qu'elles sont écrites.** Un petit programme relit chaque formule (références, MAX, IF, VLOOKUP, SQRT, puissances) et la recalcule. Il redonne les **130** formules numériques des trois feuilles **sans aucun écart** au cache. Il sert ensuite d'arbitre pour les cas qu'Excel n'a jamais calculés.
3. **Sur 500 cas tirés au hasard** (graine fixe) : cornière du catalogue (première ligne du nom, comme VLOOKUP) ou « NON STANDARD », f_y ∈ {235 ; 355}, γ ∈ {1,0 ; 1,1 ; 1,5}, L, R, X, Xj, Y, Yj, n, d et jeu variés. Sur **46 000 comparaisons**, l'écart maximal vaut **9,6·10⁻¹⁶**. La lecture en symboles est donc exactement celle du classeur, pour toute saisie qui respecte la géométrie.

**Variantes Xj et Yj** : voir le tableau du §6.3. Pour chacune, le modèle et le classeur réévalué s'accordent à 2,7·10⁻¹⁶ près.

---

## 10. Diagrammes continus et approximation « e + r »

### 10.1 Les diagrammes exacts du modèle

Ils découlent des mêmes équilibres que les quatre sections, sans aucune interpolation entre les quatre points [complements §3 bis] :

- **aile horizontale** : M(u) = R·(X + Xj − u), de 0 sous la charge à Mmax au nu du mur ;
- **aile verticale** : M(y) = H·(Y + Yj − y), de 0 à l'ancrage à Mmax au pivot.

**La contrainte, elle, dépend en plus de l'épaisseur locale.** Dans le congé, l'épaisseur réelle à la distance u du dos vaut t(u) = e + r − √(r² − (u − e − r)²). Elle vaut e + r au nu intérieur de l'autre aile et e au pied du congé.

| Aile horizontale, u (mm) | Repère | M (kN·m) | t (mm) | σ (N/mm²) | taux σ_éq |
|---|---|---|---|---|---|
| 0 | nu du mur (nœud) | 3,20700 | — | — | — |
| 15 | section 2 | 2,56560 | 28,000 | 65,449 | 27,90 % |
| 21,5 | milieu du congé | 2,28766 | 16,742 | 163,239 | 69,26 % |
| 26 | 2 mm avant le pied | 2,09524 | 15,155 | 182,459 | **77,40 %** |
| 28 | section 1 | 2,00972 | 15,000 | 178,642 | 75,80 % |
| 75 | charge | 0 | 15,000 | 0 | 6,95 % (τ seul) |

| Aile verticale, y (mm) | Repère | M (kN·m) | t (mm) | σ (N/mm²) | taux σ_éq |
|---|---|---|---|---|---|
| 7,5 | pivot (nœud) | 3,20700 | — | — | — |
| 15 | section 4 | 2,89665 | 28,000 | 73,894 | 31,44 % |
| 21,5 | milieu du congé | 2,62767 | 16,742 | 187,501 | 79,47 % |
| 26 | 2 mm avant le pied | 2,44146 | 15,155 | 212,609 | **90,10 %** |
| 28 | section 3 | 2,35870 | 15,000 | 209,662 | 88,86 % |
| 85 | ancrage | 0 | 15,000 | 0 | 6,95 % |

Entre le nœud et le nu intérieur de l'autre aile, la coupe traverse l'angle plein. La théorie des poutres n'y donne pas de contrainte représentative : c'est le domaine d'un calcul aux éléments finis.

### 10.2 L'approximation « e + r » : conservatrice, admissible ou à remplacer ?

- **Sections 2 et 4 (épaisseur e + r)** : l'épaisseur est **géométriquement exacte** à l'endroit de la coupe, car t(e) = e + r au nu intérieur de l'autre aile (CERTAIN). Ce n'est donc pas une approximation d'épaisseur. En revanche, ce sont les sections les moins sollicitées du congé. Pour la L120×120×15, la section 2 ne dépasse la section 1 que si X − e < r·(e+r)²/((e+r)² − e²) = 18,23 mm, c'est-à-dire **X < 33,2 mm**, une charge posée à 5 mm du pied du congé. Le même seuil vaut pour Y entre les sections 4 et 3.
- **Sections 1 et 3 (épaisseur e au pied du congé)** : ce sont les bonnes sections de référence, mais **pas exactement le maximum**. En entrant dans le congé, le moment croît linéairement tandis que l'épaisseur ne croît qu'au second ordre (le congé est tangent à l'aile). Le pic se trouve donc juste avant le pied :
  - à 1,98 mm, avec **+2,11 %** sur la section 1 (77,40 % au lieu de 75,80 %) ;
  - à 1,66 mm, avec **+1,45 %** sur la section 3 (**90,15 %** au lieu de 88,86 %).
- **Généralisation (calculée sur les 42 cornières du classeur).** Le rapport entre le pic et la contrainte au pied du congé dépend de la distance s entre la charge (ou l'ancrage) et le pied du congé :

| s | Pic / pied, au plus | Cornière |
|---|---|---|
| 40 mm | 1,052 | L200x200x20 |
| 20 mm | 1,178 | L200x200x20 |
| 10 mm | 1,513 | L200x200x20 |
| 5 mm | 2,270 | L200x200x20 |

- **Conclusion.** Les quatre sections du classeur sont une approximation **admissible** quand la charge et l'ancrage restent loin du congé : jusqu'à 5 % sous le pic pour s ≥ 40 mm. Elles deviennent **non conservatrices** quand l'un ou l'autre s'en approche. La recherche du maximum le long de t(u) est simple et reste dans la même théorie des poutres : elle peut remplacer la seule lecture aux quatre sections, en gardant ces dernières comme repères.

---

## 11. Ce que le classeur ne contrôle pas : le domaine

**CERTAIN, calculé** [complements §7]. Aucune formule ne vérifie la géométrie. Le classeur rend des nombres sans avertissement dans les cas suivants :

| Situation | Ce que rend le classeur |
|---|---|
| X = 20 mm < e + r : charge posée sur le congé | M1 = −0,342 kN·m (négatif), taux 24,6 % : faussement rassurant |
| Y = 25 mm < e + r : ancrage dans le congé | M3 = −0,550 kN·m, ρ3 = −0,171 |
| X + Xj = 125 mm > larg. = 120 mm : charge hors de l'aile | calcul mené normalement, 155,9 % |
| Y = jeu : bras d'ancrage nul | division par zéro (#DIV/0!) ; avec Y = jeu + 10⁻⁹ mm, N_ELS = 1,07·10¹² kN, et M3 = −23,09 kN·m car ρ3 = −7,2 |

---

## 12. Pour situer : ELU et module actuel du site

**L'ELU de la note** [complements §5]. Le classeur vérifie 1,5·σ_éq(ELS) ≤ f_y.

- Avec la réaction ELU de la même note (68,9 kN, soit ×1,611) et f_y/1,0 en élastique, la section 3 donne **95,46 %**. C'est le taux du classeur multiplié par 1,611/1,5 = 1,074.
- En plastique (W_pl = L·e²/4, effort tranchant négligeable : V_Ed/V_pl = 7,5 %), elle donne 63,44 %.
- La méthode historique se situe entre les deux. Ces chiffres ne sont donnés que pour situer : le choix de la méthode relève de l'ingénieur.

**Le moteur actuel du site** (`acier/corniere/moteur.py`, importé en lecture seule) [complements §9]. Sur le même cas (charge ponctuelle de 42,76 kN répartie sur 0,30 m, e = 7,5 cm, mêmes admissibles), il donne σ = Mmax/W_e = 285,07 N/mm², soit U_σ = **120,65 %** contre 88,86 % pour le classeur.

- Il vérifie la section **au nu du mur**, avec l'épaisseur e et le moment entier R·X. Il n'y a ni congé, ni rapport Ye/YH, ni aile verticale, ni ancrage.
- Avec ses valeurs par défaut (τ_lim = 0,8 N/mm²), le cisaillement donnerait 1 188 %.
- Le moteur actuel n'est donc pas la méthode du classeur : la reprise « méthode Excel historique » est à écrire, elle ne se règle pas par un paramètre.

---

## 13. Certain ou interprété : récapitulatif

| Sujet | CERTAIN (formule, format, structure) | INTERPRÉTÉ (sens physique) |
|---|---|---|
| Modèle | M1, M2 = R × bras ; M3, M4 = Mmax × rapport ; Mmax commun aux deux ailes | portique en L rigide, pivot au nu du mur et au plan moyen de l'aile horizontale, compression concentrée au talon |
| Xj | s'ajoute à M1, M2, Mmax, N | déplacement de la charge vers l'extérieur (tolérance d'appui) |
| Yj | seulement dans ρ3 et ρ4 ; pas dans N | ancrage plus haut que prévu (tolérance de perçage ou trou oblong) |
| C76 | seulement dans le bras de N (Y − 1 cm) | soit position de la compression, soit tolérance de pose vers le bas : formule identique, sens à trancher |
| τ | R/(L·t) moyen, avec R dans les quatre sections | dans l'aile verticale, c'est l'effort normal ; le tranchant y vaut H |
| L | même largeur de 300 mm pour les quatre sections | réaction supposée répartie sur toute la longueur reprise |
| γ = 1,5 | divise f_y ; efforts ELS | coefficient global de méthode aux contraintes admissibles, pas γ_M0 |
| ELU | multiplication par 68,9/42,76 | répartition identique à l'ELS ; 68,9 kN vient d'ailleurs (Hilti) |
| E10A BD | aucun lien avec P8 | descente de charges des volées, recoupements SCIA |
| G85, G86 | tapés, lus par rien | résultats Hilti recopiés pour comparaison |

---

## 14. Doutes et points ouverts

1. **Le sens de C76** (position de la résultante de compression, ou tolérance de pose de l'ancrage vers le bas) : les deux lectures donnent la même formule.
2. **Le sens de Yj** : un jeu vers le haut seulement, ou dans les deux sens ? Le classeur ne l'applique que vers le haut, et pas aux ancrages.
3. **Les 24,68 kN de Hilti** : le pivot implicite de 15,2 mm suppose le même excentrement (75 mm) et un partage égal entre les trois ancrages. Le rapport PROFIS n'est pas dans le classeur.
4. **L'origine de Rmax = 42,76 kN** (« p. 15 ») et de R_ELU = 68,9 kN : hors du classeur. Le rapport 1,611 > 1,5 n'est pas explicable par EN 1990 à partir de ce seul ELS.
5. **L'effort normal R dans l'aile verticale** : il suppose que le frottement au contact du mur ne reprend aucune part de R.
6. **La largeur réelle d'application** de la réaction sur les 300 mm de longueur reprise : inconnue, alors que le taux varie en 1/L.
7. **Les cellules isolées** L22 (3,7 × 0,19 × 1 230) et I10, I13, H15, I15, H16 de E10A BD : leur sens n'est pas connu.
8. **La valeur 1,345** (D3, G3 de E10A BD) : largeur de volée supposée.
9. **Les unités du catalogue CORNIERES** ne sont pas écrites : les mm se déduisent des formats de P8.
10. **Le fichier de demande** `demande_utilisateur.md` était absent au premier contrôle du dossier et y est apparu en cours d'analyse. Il a été lu en entier ensuite. Rien de ce qui précède n'en dépend, sinon le choix de commenter le jeu par défaut de 20 mm (§6.3) et l'approximation « e + r » (§10.2).
