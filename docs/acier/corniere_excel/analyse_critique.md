# Classeur REALEX « Cornières » (E10A, 02-04-26) — critique des hypothèses

*Lecture du 07/11/2026. Objet : pour chaque hypothèse du classeur, dire si elle est **CONSERVATIVE**, **ADMISSIBLE**
ou **À CORRIGER**, et le prouver par un calcul. Aucun fichier du site n'a été modifié : le moteur du site a été
exécuté en lecture seule (sans écrire de fichier compilé dans son dossier). Chaque chiffre de ce rapport sort d'un
script de `scripts/` (liste au § 6) ; aucun n'est estimé.*

**Marquage utilisé**

- **[Lu]** : certain, lu dans une formule ou une valeur du classeur (la cellule est citée) ;
- **[Calculé]** : obtenu par un script, à partir des formules ou d'un modèle mécanique explicite ;
- **[Interprété]** : sens physique que je prête à une cellule ou à une hypothèse ; c'est une lecture, pas un fait.

**Repère** (celui du schéma de la note) : origine au talon extérieur, contre le mur ; x horizontal le long de l'aile
horizontale, y vertical le long de l'aile verticale ; le mur est le plan x = 0. Cas de contrôle : L120×120×15, r = 13,
S355, b = 300 mm, R = 42,76 kN (ELS), X = 75, Y = 85, Xj = Yj = 0, coefficient 1,50.

---

## 0. En bref

| # | Hypothèse du classeur | Cellules | Verdict | La preuve, en un chiffre |
|---|---|---|---|---|
| H1 | Section 2 (et 4) d'épaisseur « e + r » | C41–C47, C61–C67 | **À CORRIGER dans son principe**, sans effet sur ce cas | la hauteur de matière vaut bien e + r = 28 mm à x = e, mais la contrainte y vaut 125,9 MPa (éléments finis) contre 65,4 au classeur (×1,92) ; la section 2 ne gouverne que si X + Xj < 33,2 mm |
| H2 | Les quatre sections suffisent | D3 | **ADMISSIBLE ici, À CORRIGER pour généraliser** | maximum réel dans le congé à ~2 mm du pied : +2,1 % (aile horizontale), +1,5 % (aile verticale) ; jusqu'à +27 % si la charge est à 20 mm du pied du congé, ×2,1 à 5 mm |
| H3 | Largeur b = « longueur reprise » | C8 | **ADMISSIBLE sous conditions** | légitime si l'appui de la charge fait au moins 206 mm (centré) ; avec 100 mm, σ1 ×1,55 (taux 117 %) |
| H4 | Jeux Xj, Yj et « jeu » des ancrages | C14, C17, C76 | **CONSERVATIVE si positifs, À CLARIFIER** | 20 mm en Xj : taux 88,9 → 112,4 % ; en Yj : 95,4 % sans changer les chevilles ; en C76 : chevilles +15 % sans changer la cornière ; « 20 » tapé en C76 (cm) : traction négative, sans alerte |
| H5 | Modèle d'ancrage (pivot, bras, R/n) | C25, C56, C75–C86 | **ADMISSIBLE en ordre de grandeur, NON CONSERVATIF dans deux cas** | deux pivots différents dans la même feuille (7,5 et 10 mm) ; cheville raide : T +19 % ; support très souple : section 3 +3,8 % en contrainte nominale ; HILTI (G85) = C85 × 1,075 |
| H6 | τ = R/(b·t) partout, σ_eq = √(σ² + 3τ²) | G34–G67 | **À CORRIGER dans la forme, côté sûr ici** | dans l'aile verticale R est un effort NORMAL, pas un tranchant ; statique cohérente : 80,8 à 84,0 % au lieu de 88,9 % ; non conservatif si Y < 38,3 mm |
| H7 | Contraintes admissibles ELS, f_y/1,5 | H31, H34 | **À GARDER, en la nommant** | identique à l'EC3 élastique avec γ = 1,50 sur tout ; plus sévère que l'EC3 plastique d'un facteur 1,50 à 1,67 ; MOINS sévère que l'EC3 élastique si le rapport ELU/ELS dépasse 1,50 — celui du classeur (C71/C70) vaut 1,611 → 95,5 % |
| H8 | Catalogue et saisies | CORNIERES, C4–C7, C8, C76 | **À CORRIGER** | 15 lignes jamais atteintes par RECHERCHEV ; rayons des cornières inégales = r2 au lieu de r1 ; trois unités différentes |
| H9 | Généralisation | toute la feuille | **NON GÉNÉRALISABLE tel quel** | cornière retournée : M3 ×1,36 (taux 120,7 %) et, selon la cote d'ancrage, T ×3 ; charge hors de l'aile ou ancrage hors de l'aile calculés sans alerte |
| Site | Module actuel `acier/corniere` | — | **ne reproduit pas la méthode** | modèle « lame » : seul t compte, un seul bras e à choisir (taux de 75,8 à 120,7 % selon le choix) ; « longitudinal » = transversal à 4,4·10⁻¹⁶ près |

---

## 1. Le classeur reproduit, base de la critique

**[Calculé, c01]** Les 45 cellules calculées de P8 ont été recalculées formule par formule (`commun.classeur`) :
écart relatif maximal au cache du fichier **0,0**. Le cas de contrôle de l'ingénieur est retrouvé exactement :
M1 = 2,009720 · M2 = 2,565600 · M3 = 2,358697 · M4 = 2,896645 kN·m · taux maximal 88,862 % en section 3.

**Le chemin des efforts que le classeur suppose** [Lu dans les formules ; Interprété pour la lecture physique] :

1. **Aile horizontale** — une console encastrée dans le coin, chargée par R à la distance X du nu du mur. Moment
   M(x) = R·(X + Xj − x) : c'est la **statique exacte** du corps libre à droite de la coupe (la charge est la seule force
   qui s'y applique). Section 1 au pied du congé, x = e + r (bras Xe = X − e − r, P8!C19) ; section 2 à l'aplomb de la
   face intérieure de l'aile verticale, x = e (bras Xe+r = X − e, P8!C20). C'est pourquoi **M2 > M1** : la section 2
   est plus près du coin.
2. **Aile verticale** — tenue par la cheville (traction T vers le mur à la hauteur Y) et par le mur (compression C au
   pied). Le classeur place cette compression à **y = e/2** (YH = Y − e/2, P8!C25) : le moment y varie linéairement
   de R·X (au pivot) à 0 (à la cheville), d'où M3 = R·X·Ye/YH (P8!C56·C55) et M4 = R·X·(Ye+r)/YH (P8!C64·C63).
   **M4 > M3** pour la même raison que M2 > M1 ; **M3 > M1** parce que le moment de l'aile verticale part du moment
   total R·X, non du bras Xe.
3. **Ancrages** — T = R·X/(Y − 10 mm) répartie sur n chevilles (P8!C82), cisaillement R/n (P8!C83).

---

## 2. Les hypothèses, une à une

### H1 · L'épaisseur « e + r » des sections 2 et 4

**[Lu]** P8!C41 = `$C$6+$C$7` (e + r = 28 mm) ; P8!C42 = `$C$8*10*C41^2/6` = 39 200 mm³ ; le bras de la section 2 est
P8!C43 = `$C$20` = X − e = 60 mm, donc la section 2 est à **x = e = 15 mm** du nu du mur, à l'aplomb de la face
intérieure de l'aile verticale. Même construction pour la section 4 (P8!C61, C62, C64), à **y = e**.

**[Calculé, c02] Ce qui est juste.** Avec le congé exact d'un L laminé (arc de rayon r tangent aux deux faces
intérieures, centre en (e + r, e + r)), la hauteur de matière sur la verticale x = e vaut **exactement e + r**
(28,000 mm). Sur ce point, le classeur ne se trompe pas.

**[Calculé, c02] Ce qui ne l'est pas.** À cet endroit précis, la face du congé est **verticale** (pente de la face :
−51, −510, −5 099 pour des pas de 10⁻², 10⁻⁴, 10⁻⁶ mm). La fibre haute de la coupe 2 est donc un bord libre dont la
normale est horizontale : la contrainte de flexion σ_xx y est **nulle par équilibre**. La répartition linéaire de
contrainte qu'implique W = b·(e+r)²/6 y est impossible. La coupe x = e est en outre un point singulier : 0,5 mm plus
loin la matière ne fait plus que 24,43 mm, 2 mm plus loin 21,07 mm, 5 mm plus loin 17,75 mm.

**[Calculé, c03 — éléments finis plans de la coupe, congé exact, contact unilatéral avec le mur ; convergé au pas de
0,25 mm]** Sur la coupe x = e : σ_xx en fibre haute = **4,6 MPa** (≈ 0, comme prévu) ; σ maximale = **125,9 MPa**,
en sous-face (compression, côté talon), contre **65,4 MPa** au classeur → **×1,92**. Le module « vu » par cette coupe
vaut M/σ_max = 68 à 69 mm³ par mm (selon qu'on prend le moment de la statique ou celui lu sur la coupe) au lieu de
(e+r)²/6 = 130,7 : c'est celui d'une épaisseur d'environ **20 mm**, pas de 28. Section 4 : 91,4 MPa contre 73,9 au
classeur (**×1,24**), épaisseur équivalente ≈ 21 mm.
Voir `sorties/c03_sections.png` (la courbe EF de S2 s'écarte franchement de la droite du classeur).

**[Calculé, c02] Autres coupes possibles.** Coupe radiale à 45° (bissectrice, du centre du congé au talon) :
épaisseur √2·(e + r) − r = **26,60 mm**, W = 0,902 × W(e+r) = 3,144 × W(e) ; contrainte de poutre 83,3 MPa — loin du
critique. Coupes normales aux ailes (verticales dans l'aile horizontale, horizontales dans l'aile verticale) :
voir H2.

**[Calculé, c02] Quand la section 2 gouvernerait-elle ?** σ2 > σ1 ⇔ X + Xj < e + (e+r)²/(2e + r) = **33,2 mm**, soit
une charge à moins de 5,2 mm du pied du congé. Même seuil pour la section 4 face à la section 3 (Y + Yj < 33,2 mm).

**Verdict H1 : À CORRIGER dans son principe, SANS EFFET sur le verdict du cas.** Comme estimation de la contrainte
à l'endroit qu'elle désigne, la section « e + r » est **non conservative** (−48 % en S2, −19 % en S4). Elle ne
gouverne pourtant presque jamais (ici 27,9 % et 31,4 % quand S1 et S3 sont à 75,8 et 88,9 %), parce que le vrai
point dur est ailleurs (H2). **Proposition** : garder S2 et S4 affichées comme sections HISTORIQUES (traçabilité),
mais ne plus les présenter comme une vérification du congé ; vérifier le congé par le balayage continu de H2.

### H2 · Où se trouve vraiment la section la plus sollicitée

Le moment croît vers le coin pendant que la section s'épaissit dans le congé : les deux effets se disputent.

**[Calculé, c02 — théorie des poutres avec l'épaisseur exacte du congé, pas de 0,01 mm]**

- Aile horizontale (statique exacte) : σ_eq max = **183,19 MPa à x = 26,02 mm**, soit **1,98 mm à l'intérieur du
  congé**, contre 179,40 en section 1 → **+2,11 %**. (Au pied du congé la pente de l'épaisseur est nulle : on gagne
  du moment sans perdre d'épaisseur, le maximum est forcément un peu en deçà.)
- Aile verticale (statique du classeur) : σ_eq max = **213,37 MPa à y = 26,34 mm** contre 210,31 en section 3 →
  **+1,45 %**.

**[Calculé, c03 — éléments finis]** Pic de contrainte principale sur la surface du congé : **236,5 MPa** (pas de
0,25 mm ; 237,5 à 0,5 mm ; 238,4 à 1 mm), à ~3 mm du pied du congé côté aile horizontale, soit **1,32 × σ1 du
classeur** et 99,9 % de σ_adm. En section 1, la fibre haute vaut 202,6 MPa (×1,134) alors que la contrainte
linéarisée retrouve celle du classeur (178,5). Maximum sur les coupes horizontales de l'aile verticale : 210,4 MPa —
il égale la section 3 du classeur (209,7) **par compensation** (le classeur surestime le moment de l'aile verticale,
voir H5, et ignore la concentration). Figures : `sorties/c03_carte_vm.png`, `sorties/c03_balayage.png`.

**[Interprété]** Ce pic est une concentration de contrainte élastique locale (coin chargé en ouverture, le congé est
du côté tendu). La méthode du classeur comme l'EC3 raisonnent en contraintes nominales ; un acier ductile redistribue
un tel pic en statique. Il compte en fatigue et pour situer la première plastification, pas comme critère ELS.

**[Calculé, c02] Généralisation sur le catalogue du site** (197 à 198 cornières à ailes égales, rayon r1 exact,
théorie des poutres) : maximum dans le congé ÷ max(S1, S2) =

| distance de la charge au pied du congé | rapport, du meilleur au pire |
|---|---|
| milieu de l'aile restante | 1,004 à 1,024 |
| 20 mm | 1,009 à **1,271** (L 250×250×35) |
| 10 mm | 1,025 à **1,607** (L 200×200×27) |
| 5 mm | 1,093 à **2,082** (L 180×180×16) |

**[Calculé, c02] Convention 0,8 r** — usage EN 1993-1-8 pour les profils laminés [Interprété : par analogie avec le
tronçon en T, où la ligne de rotule est prise à 0,8 r de la face de l'âme, avec l'épaisseur courante] : section à
x = e + 0,8 r = 25,4 mm, épaisseur e → σ = 188,5 MPa (**+5,5 %** sur S1) ; aile verticale 219,2 MPa (**+4,6 %** sur S3).

**Verdict H2 : ADMISSIBLE pour le cas de contrôle (écart nominal ≤ 2,1 %), À CORRIGER pour généraliser** : dès que
la charge (ou la cheville) s'approche du congé, les quatre sections ratent le maximum. **Proposition** : diagramme
continu σ(x) = M(x)/W(t(x)) avec l'épaisseur exacte, maximum cherché sur toute l'aile (congé compris) ; S1 à S4
marquées dessus ; le pic élastique EF montré à titre d'information, avec son facteur (≈ 1,3 ici).

### H3 · La largeur b = « longueur reprise » (P8!C8)

**[Lu]** Toutes les sections prennent la même largeur b = C8 × 10 = 300 mm (C32, C42, C52, C62, et les τ). Le
classeur ne donne ni la largeur d'appui de la charge, ni la position des chevilles le long de b.

**[Calculé, c07] Diffusion à 45° depuis un appui de largeur c** (centré, loin des extrémités) jusqu'à la section 1
(47 mm) et à la section 2 (60 mm) :

| c (mm) | b_eff S1 (mm) | σ1 × | taux S1 |
|---|---|---|---|
| 50 | 144 | 2,08 | 157,9 % |
| 100 | 194 | 1,55 | 117,2 % |
| 150 | 244 | 1,23 | 93,2 % |
| ≥ 206 | 300 | 1,00 | 75,8 % |

Il faut c ≥ **206 mm** (S1) et ≥ 180 mm (S2) pour que b = 300 mm soit atteint ; **253 mm** si la charge est à une
extrémité de la cornière (diffusion d'un seul côté).

**[Calculé, c07] Côté chevilles.** Avec 3 chevilles sur 300 mm (entraxe 100 mm si régulières — **hypothèse**,
le classeur ne le dit pas), la diffusion à 45° depuis une rondelle Ø30 jusqu'à la section 3 (57 mm plus bas) couvre
144 mm : le moment est uniforme sur b à la section 3 si l'entraxe ≤ **144 mm**. Pour mémoire, l'analogie du tronçon en
T équivalent (EN 1993-1-8, rangée isolée) donne m = (Y − e) − 0,8 r = 59,6 mm, l_eff = 2πm = 374 mm ou
4m + 1,25e = 282 mm [Interprété : analogie, à confirmer dans le texte].

**[Calculé, c07] Répartition entre chevilles (poutre continue, appuis rigides ou élastiques).** Dans le cas du
classeur (charge répartie sur 300 mm, chevilles supposées à 50/150/250 mm) : appuis rigides 15,14 / 12,47 / 15,14 kN ;
chevilles réalistes (0,1 mm sous 14,25 kN) **14,27 / 14,22 / 14,27 kN** : R/n = 14,25 kN est juste, parce que la
cornière est très raide devant les chevilles. Mais sur la **cornière de 2,00 m de la demande** (chevilles à 100,
1 000 et 1 900 mm, 42,76 kN sur 400–700 mm) : **18,1 / 28,0 / 3,3 kN** (chevilles 0,1 mm), 17,5 / 29,1 / 3,9 kN
(rigides), 19,7 / 24,8 / 1,7 kN (0,5 mm) — la cheville centrale prend **deux fois** R/n — et la flexion longitudinale
vaut **6,3 à 7,3 kN·m**, soit σ = 121 à 138 MPa sur W_el,y = 52,43 cm³ (51 à 58 % de σ_adm à elle seule), que le
classeur ne calcule pas.

**Verdict H3 : ADMISSIBLE SOUS CONDITIONS** — charge appliquée sur au moins ~206 mm (ou répartie sur b), chevilles
régulières à entraxe ≤ ~144 mm, cornière courte et raide devant les chevilles. Hors de ces conditions : **non
conservatif** (×1,55 sur σ1 pour un appui de 100 mm ; ×2 sur la cheville la plus chargée d'une cornière longue).

### H4 · Les jeux Xj (C14), Yj (C17) et le « jeu » des ancrages (C76)

**[Lu]** Xj s'ajoute à TOUS les bras horizontaux (C34, C44, C54 → M1, M2, et C55/C63/C78 → M3, M4 et les chevilles).
Yj s'ajoute au numérateur et au dénominateur des rapports C56 et C64 (sections 3 et 4) et **n'entre pas** dans C82
(chevilles). C76 (« Jeu : 1 — pour dimensionner les ancrages », **en cm**) ne sert qu'à C82 : bras = Y − 10 mm.
Unités : C14, C17 en mm ; C76 et C8 en cm (C75 = C16/10 convertit Y en cm).

**[Interprété]** D'après le schéma : Xj = la charge peut tomber plus loin du mur que prévu ; Yj = la cheville peut
être plus haut que prévu (flèche vers le haut au-dessus de l'ancrage). C76 est ambigu : soit la position de la
résultante de compression (10 mm au-dessus de la sous-face), soit une tolérance « cheville plus basse de 10 mm ». Dans
les deux lectures, c'est **un bras de levier**, pas une distance de montage.

**[Calculé, c04] Un jeu de 20 mm (la valeur par défaut demandée) dans chaque case :**

| saisie | taux S1 | taux S3 | N_ELS par cheville |
|---|---|---|---|
| cas de contrôle | 75,80 % | 88,86 % | 14,25 kN (bras 75 mm) |
| Xj = 20 mm | 107,83 % | **112,43 %** | 18,05 kN |
| Yj = 20 mm | 75,80 % | 95,38 % | 14,25 kN (inchangé) |
| Yj = −20 mm | 75,80 % | 77,82 % | 14,25 kN (inchangé) |
| C76 = 2 cm | 75,80 % | 88,86 % (inchangé) | 16,45 kN (bras 65 mm) |
| C76 = 20 (« 20 mm » tapé en cm) | 75,80 % | 88,86 % | **−9,30 kN** (bras −115 mm), sans alerte |
| C8 = 300 (« 300 mm » tapé en cm) | 7,58 % | **8,89 %** | 14,25 kN |
| Xj = −20 mm | 43,92 % | 65,34 % | 10,45 kN |

**Verdict H4 : CONSERVATIVE tant que les jeux sont positifs et placés du côté défavorable de chaque vérification ;
À CLARIFIER.** Yj est défavorable pour l'aile (cheville plus haute → Ye/YH augmente) mais la cheville n'en profite
pas : une cheville plus BASSE (Yj < 0) serait défavorable pour les chevilles et favorable pour l'aile — le classeur ne
peut pas le dire avec une seule case. Les trois « jeux » ont trois effets, deux unités, et C76/C8 acceptent une valeur
dix fois trop grande sans alerte. **Proposition** : trois grandeurs nommées et dessinées — décalage de la charge
(± Xj, appliqué dans le sens défavorable), tolérance de position de la cheville (± Yj, appliquée vers le haut pour
l'aile et vers le bas pour la cheville), position de la résultante de compression (bras de levier) — toutes en mm.

### H5 · Le modèle d'ancrage : pivot, levier, raideurs

**[Lu]** Deux pivots dans la même feuille : sections 3-4 → pivot à e/2 = 7,5 mm (YH, C25) ; chevilles → bras
Y − 10 mm (C82). **[Calculé, c04]** T = R·X/YH = **41,38 kN** pour les sections, R·X/75 = **42,76 kN** pour les
chevilles (×1,033). Les deux ne décrivent pas le même équilibre.

**[Calculé, c04] Équilibre complet du corps libre** — moment autour du point de compression :
T·(Y − y_c) = R·(X − x_V), x_V étant l'abscisse où la cheville reprend l'effort vertical (0 au nu du mur, e/2 au
milieu de l'appui de la tige). Selon (x_V, y_c) : T = 33,96 kN (e/2 ; 0) … 37,24 (e/2 ; 7,5) … 41,38 (0 ; 7,5) …
42,76 (0 ; 10) … **45,81 kN** (0 ; 15).

**[Calculé, c03 — éléments finis]** Mur rigide, cheville réaliste (0,1 mm sous 14,25 kN) : **T = 40,8 kN**, compression
concentrée au talon (y_c ≈ 0) — le classeur est juste à 5 % près. Mais :

- **la rondelle porte sur un bord**, pas sur l'axe : avec une cheville raide la résultante est à y ≈ 70 mm (bord
  bas de la rondelle Ø30), avec une cheville souple à y = 100 mm (bord haut). Le bras réel varie de ±15 mm autour de Y ;
- **raideur de la cheville** (déplacement sous l'effort de service) : 0,01 mm → T = **56,4 kN** (+32 %) avec un effet
  de levier en tête d'aile de 8,9 kN ; 0,02 mm → **50,7 kN** (+19 %), levier 5,6 kN ; 0,05 → 41,2 ; 0,1 → 40,8 ;
  0,2 → 34,8 ; 0,5 → 28,9 kN. Cheville indéformable : 93,8 kN (aile pincée entre rondelle et mur — borne théorique) ;
- **support souple** (module de réaction du mur k, N/mm³) : le pivot remonte et la section 3 se charge. Comparaison
  À BASE ÉGALE [Calculé, c03c] — contrainte linéarisée EF (nominale) face à la contrainte nominale du classeur
  (209,7 MPa) : mur rigide, y_c = 0, **164,3 MPa (−21,6 %)** ; k = 1 000 : y_c = 2,8 mm, 174,6 (−16,7 %) ; k = 300 :
  4,7 mm, 184,4 (−12,0 %) ; k = 100 : 7,3 mm, 198,6 (−5,3 %) ; **k = 30 : 11,8 mm (contact jusqu'à 37,5 mm),
  217,6 MPa (+3,8 %)**, et +4,6 % au maximum nominal dans le congé. Avec la concentration au pied du congé (×1,13),
  le pic EF rejoint la valeur du classeur dès k ≈ 300 (209,1 MPa) et la dépasse de 7,5 % (k = 100) à 17,8 % (k = 30).
  La marge du classeur sur mur rigide (−21,6 %) vient de trois approximations qui jouent dans le même sens : pivot réel
  au talon (0 au lieu de 7,5 mm), rondelle portant sur son bord bas (≈ 70 mm au lieu de 85), réaction verticale
  au milieu de l'épaisseur. [Interprété : un béton sain se situe plutôt dans les valeurs hautes de k, une maçonnerie
  ou une cale souple dans les basses ; ces correspondances ne sont pas calculées.]

**[Lu, Calculé] HILTI.** P8!G85 = 24,68 kN en face de C85 = 22,97 kN (×**1,0746**) ; G86 = 22,97 = C86.
[Interprété : G85 et G86 sont les efforts du logiciel ; ils ne sont pas titrés.] Le bras implicite de G85 vaut
69,79 mm (pivot à 15,2 mm) : le logiciel met **7,5 % de traction de plus** que la formule C82.

**[Calculé, c04] Ce que la feuille rend sans prévenir** : Y = 25 mm (< e + r) → M3 < 0 et 71,3 kN par cheville ;
Y = 28 → M3 = 0 ; Y = 130 (> haut. = 120) → calculé, C90 négatif. C88 (pression diamétrale, 59,4 MPa) et C90
(arrachement, 13,6 MPa) ne sont **comparés à aucune limite** [Lu] ; C90 ne déduit pas le trou : avec Ø18 (hypothèse
M16, jeu de 2 mm) il vaut 18,3 MPa (×1,35).

**Verdict H5 : ADMISSIBLE en ordre de grandeur pour un mur en béton et une cheville courante ; NON CONSERVATIF**
pour une cheville très raide (T +19 % à +32 %) et pour un support très souple (section 3 : +3,8 % en contrainte
nominale à k = 30 N/mm³ ; la marge de 21,6 % du mur rigide fond quand le pivot remonte) ; les deux pivots sont
**incohérents** entre eux. Le cisaillement R/n est conservatif (aucun frottement au mur n'est compté).
**Proposition** : un seul équilibre, écrit, avec la résultante de compression nommée et dessinée ; T calculé avec le
bras le plus court plausible (bord bas de rondelle, pivot remonté) pour les chevilles, et le moment de l'aile avec le
bras le plus long ; efforts ELU fournis à HILTI à partir des combinaisons, jamais par un rapport ELU/ELS global.

### H6 · Cisaillement et contrainte équivalente

**[Lu]** τ = R·1000/(b·t) dans les quatre sections (G34, G44, G54, G64) ; σ_eq = √(σ² + 3τ²) (G37…G67) ; aucun
effort normal.

**Aile horizontale [Calculé, c05].** τ réel parabolique, maximal 14,25 MPa au plan moyen, nul en fibre extrême. La
combinaison du classeur (σ en fibre extrême avec τ moyen) donne 179,40 MPa contre 178,64 pour le vrai maximum point
par point : **+0,42 %, côté sûr**. Le τ seul vaut 10,4 % de τ_adm.

**Aile verticale [Calculé, c05] — l'erreur de nature.** Au-dessus de la section 3, le corps libre ne contient que la
cheville : **R y est un EFFORT NORMAL (traction)**, l'effort tranchant est **T** (horizontal). Les éléments finis le
confirment : N = 141,6 N/mm sur la section 3 pour q = R/b = 142,5. Le classeur prend R comme tranchant et oublie N/A
(9,5 MPa). Il calcule aussi le moment autour du nu du mur (R·X) et non au centre de la section. Les deux erreurs sont
de sens contraires ; avec x_V = 0 on a exactement **σ_cohérent − σ_classeur = −2R/(b·e) = −19,0 MPa**.

| statique de la section 3 | σ_eq (MPa) | taux |
|---|---|---|
| classeur | 210,31 | 88,86 % |
| cohérente, réaction verticale au nu du mur | 191,32 | 80,84 % |
| cohérente, réaction au milieu de l'épaisseur | 198,72 | 83,96 % |
| cohérente, pivot remonté à 15 mm (support souple) | 218,99 | **92,53 %** |
| « correction » partielle : moment du classeur + N/A | 219,74 | 92,85 % (à NE PAS faire : double compte) |

Le classeur cesse d'être du côté sûr si **Y < 1,25 e + 1,5 r = 38,3 mm** (Y = 30 mm : +27,5 %) ou si le pivot remonte.

**Verdict H6 : À CORRIGER dans la forme, CÔTÉ SÛR ici.** La combinaison √(σ² + 3τ²) est la bonne (critère de von
Mises plan, EN 1993-1-1 6.2.1(5)) ; ce sont les efforts qu'on y met dans l'aile verticale qui sont faux.

### H7 · Contraintes admissibles ELS (σ_adm = f_y/1,5) face à l'EN 1993-1-1

**Ce que ce n'est pas.** Ce n'est pas une vérification aux états limites : pas de combinaison d'actions, pas de
coefficients partiels propres à chaque action, γ = 1,50 appliqué au matériau et non aux charges, résistance
élastique seulement, aucun critère pour les chevilles (renvoyées à HILTI).

**[Calculé, c06]** Taux admissible = σ_eq/(f_y/1,5) = 1,5·σ_eq/f_y : **c'est exactement l'EC3 élastique avec un
coefficient 1,50 sur toutes les actions** (88,8621 % = 88,8621 %). Pour la section 3 (statique du classeur),
γ_M0 = 1,0 :

| γ_eq = ELU/ELS | admissible | EC3 élastique | EC3 plastique |
|---|---|---|---|
| 1,35 (tout permanent, 6.10) | 88,86 % | 79,98 % | 53,15 % |
| 1,3947 à 1,4086 (volées de la feuille E10A BD) | 88,86 % | 82,62 à 83,45 % | 54,91 à 55,46 % |
| 1,50 (tout variable) | 88,86 % | 88,86 % | 59,06 % |
| **1,6113 (C71/C70, le rapport du classeur lui-même)** | 88,86 % | **95,46 %** | 63,44 % |

(Section rectangulaire pleine : classe 1, M_pl = b·t²·f_y/4 ; interaction effort normal M_N = M_pl·(1 − n²) quand N est
compté ; V/V_pl < 0,5 partout, pas de réduction.)

**Lecture.** Avec des actions caractéristiques combinées en 6.10 (γ_eq entre 1,35 et 1,50), la méthode est **aussi
sévère ou plus sévère** que l'EC3 élastique (jusqu'à 11 %) et **1,50 à 1,67 fois plus sévère** que l'EC3 plastique.
Elle devient **moins sévère** que l'EC3 élastique dès que le rapport ELU/ELS réel dépasse 1,50 — ce qui est le cas du
rapport porté par le classeur pour les ancrages, C71/C70 = 68,9/42,76 = **1,611** [Calculé]. [Interprété : un tel
rapport signifie que R_max ELS et R_ELU HILTI ne viennent pas d'une même combinaison caractéristique — autre charge,
autre combinaison, ou ELS non caractéristique. À éclaircir.]

**Verdict H7 : À GARDER comme méthode du bureau, en l'affichant pour ce qu'elle est** (« contraintes admissibles, ELS
non majorés, coefficient global 1,50 ») ; montrer à côté le rapport ELU/ELS réel et prévenir quand il dépasse 1,50.

### H8 · Données : catalogue, unités, saisies

**[Calculé, c01]**

- **RECHERCHEV sur des noms en double** (C4 à C7 : `VLOOKUP(B3; CORNIERES!A2:E59; …; FALSE)`) : 15 désignations
  figurent deux fois (les cornières inégales, une ligne par orientation) ; RECHERCHEV rend toujours la première. Les
  lignes **7, 11, 15, 18, 21, 23, 25, 28, 32, 36, 38, 42, 46, 51, 56** ne sont jamais atteintes : l'orientation
  « grande aile verticale » est inaccessible hors « NON STANDARD ».
- **Rayons** : sur 26 lignes retrouvées au catalogue du site, 22 ont r = r1 (rayon de congé) ; **L150×100×12 (les
  deux lignes) porte r = 6 = r2** (arrondi de bout d'aile) au lieu de r1 = 12 ; L25×25×3 et L20×20×3 portent 4 au lieu
  de 3,5. Les autres inégales du classeur (r = 5,5 pour L120×80, 4 pour L90×60…) ont l'allure de r2 ; elles sont
  absentes du catalogue du site (31 lignes absentes, surtout des petites inégales), donc non vérifiées une à une.
  Effet sur L150×100×12 (X = 75, Y = 70) : taux 153,2 % avec r = 6 contre 135,6 % avec r1 = 12 — **côté sûr**, mais
  les sections sont placées au mauvais endroit.
- **larg. et haut. ne servent presque à rien** : les passer de 120 à 60 et 400 ne change aucun taux ; ils n'entrent
  que dans C90 (et A20, A21, calculés mais jamais lus). Aucune cellule ne vérifie que la charge est sur l'aile ou
  la cheville dans l'aile.
- **Saisies à la main** : R_max (C12, « p. 15 ») et R_ELU (C71, « logiciel HILTI ») ne sont liés à rien ; aucune
  formule de P8 ne lit la feuille E10A BD. P8!L22 = 3,7·0,19·1 230 est un calcul isolé sans lien.
- **Unités mélangées** : mm (géométrie, jeux Xj et Yj), cm (C8, C75, C76). Pièges chiffrés en H4.
- Masses : écart maximal −4,2 % (L150×100×12 : 21,6 contre 22,54 kg/m) ; sans effet sur le calcul.

**Verdict H8 : À CORRIGER** (source unique : le catalogue du site, avec r1 et r2 distincts et l'orientation comme
donnée séparée de la désignation).

### H9 · Ce qui empêche de généraliser

**[Calculé, c08] Cornière retournée (aile horizontale en haut).** Le pivot passe au **bout libre** de l'aile
verticale ; tout le couple cheville/compression se trouve sous les sections du coin, si bien que **le moment de
l'aile verticale vaut R·X d'un bout à l'autre entre la cheville et le coin** (3,207 kN·m) au lieu de R·X·Ye/YH
(2,359) : **×1,36, taux S3 = 120,7 %** au lieu de 88,86 %. Le bras des chevilles se compte depuis le bout : si l'on
garde « Y = 85 mm depuis l'aile horizontale » sur une aile de 120, il reste 25 mm de bras et **T = 128,3 kN, ×3,0**
la valeur du classeur. L'aile verticale entre coin et cheville passe en **compression**. Les formules du classeur ne
s'appliquent pas, même en retournant le dessin.

**[Calculé, c04, c02, c07, c08] Autres limites** :

- charge **au-delà de l'aile** (X = 140 > larg. = 120) : taux 180,0 % rendu sans alerte ; charge **dans le congé**
  (X = 20) : M1 = −0,342 kN·m ;
- cheville **dans le congé** (Y < e + r) ou **au-dessus de l'aile** (Y > haut.) : calculé sans alerte (H5) ;
- charge ou cheville **proche du congé** : maximum manqué de +27 % à ×2,1 (H2) ;
- **ailes inégales** : orientation inaccessible, rayon r2 (H8) ;
- **cornière longue, chevilles espacées, charge localisée** : R/n faux (×2 sur la cheville centrale) et flexion
  longitudinale non vérifiée (H3) ;
- une seule rangée de chevilles, une seule charge verticale descendante (ni soulèvement, ni effort horizontal) ;
- f_y saisi à la main (C3) sans lien avec l'épaisseur [Interprété : sans effet pour e = 15 mm ; à vérifier au-delà de
  16 mm selon l'option retenue par l'annexe nationale pour f_y] ;
- cornière **PRS** (deux plats soudés) : pas de congé, la logique « e + r » n'a pas d'objet ; c'est la soudure qui se
  vérifie.

**Verdict H9 : NON GÉNÉRALISABLE tel quel.** La configuration B demande son propre équilibre ; les autres cas
demandent des contrôles de domaine (charge sur l'aile, cheville hors du congé et dans l'aile).

---

## 3. Le module actuel du site, face au classeur

**[Lu]** `acier/corniere/moteur.py` (`calculer`) et `docs/acier/CORNIERE_MOTEUR.md` : modèle « lame » par mètre
courant, b = 1 000 mm, **seul t entre dans le calcul** (ni les ailes, ni r, ni Y) ; M = V·e avec un unique bras e
(« distance charge – cornière ») ; σ = M/(b·t²/6), τ = V/(b·t), σ_eq = √(σ² + 3τ²) ; ELU = coefficient global
γ_ELU × ELS ; limites par défaut σ_lim,ELS = 155, **τ_lim,ELS = 0,8**, σ_eq,lim,ELU = 186 N/mm² ; ancrages non vérifiés.

**[Calculé, c09 — le moteur du site exécuté sur le cas de contrôle, charge ponctuelle 42,76 kN sur 0,30 m, t = 15]**

| bras e saisi | σ_ELS | taux avec les limites S355/1,5 | taux avec les limites par défaut du site |
|---|---|---|---|
| e = X = 7,5 cm (moment au nu du mur) | 285,07 MPa | 120,7 % | σ 184,2 %, **τ 1 187,8 %**, ELU 207,2 % |
| e = Xe = 4,7 cm (bras de la section 1) | 178,64 MPa | 75,8 % — **manque la section 3 qui gouverne** | σ 115,7 % |
| e = X·Ye/YH = 5,516 cm | 209,66 MPa | 88,9 % (= classeur) | σ 135,7 % |

Le résultat du site dépend donc entièrement du bras que l'utilisateur choisit : selon le choix, il est plus sévère
(×1,36) ou **non conservatif** (75,8 % au lieu de 88,86 %) face au classeur.

**Le défaut non conservatif connu, vérifié [Calculé, c09].** La vérification « longitudinale entre fixations » pose
M_long = V·s·e sur un panneau de largeur s : s se simplifie. Sur 200 entraxes de 10 à 1 463 cm, U_long/U_ELU − 1
≤ **4,4·10⁻¹⁶** (égalité au bit près dans 154 cas sur 200) : le taux « longitudinal » est le taux transversal. Ce qui
manque : la flexion de la cornière entre fixations — pour le cas par défaut du site (L100×100×10, 5 kN/m), q·s²/8
donne 2,3 MPa à s = 30 cm mais **57,1 MPa à s = 150 cm**, et le taux affiché ne bouge pas ; sur la cornière de 2 m de
la demande, 121 à 138 MPa (H3).

**Ce que le site n'a pas et que le classeur a** : le rayon et le congé, les quatre sections, l'aile verticale et sa
statique (Y, YH), les efforts par cheville (N, V, ELS et ELU), les jeux, la largeur reprise, des limites tirées de
f_y. **Ce que le site a et que le classeur n'a pas** : le catalogue EN 10056-1 (233 cornières, r1 et r2 distincts),
un sélecteur, l'alerte « charge au-delà de l'aile ».

---

## 4. Ce qu'il faut garder, corriger, ajouter

**Garder** (méthode historique, reproduite à l'identique dans un mode « Excel historique ») : la statique de l'aile
horizontale (exacte), les quatre sections comme repères, la combinaison √(σ² + 3τ²), la méthode admissible ELS avec
son coefficient 1,50 affiché.

**Corriger ou compléter, chacun mesuré ci-dessus** :

1. un seul équilibre de la cornière, pivot et résultante de compression nommés et dessinés (H5) ;
2. dans l'aile verticale : N = R (traction en configuration A, compression en B), tranchant T, moment au centre de la
   section (H6) ;
3. diagramme continu σ(x) avec l'épaisseur exacte du congé et le maximum réel ; S2/S4 gardées comme repères
   historiques, pas comme vérification du congé (H1, H2) ;
4. trois jeux distincts, en mm, appliqués du côté défavorable de chaque vérification (H4) ;
5. largeur efficace par diffusion, répartition entre chevilles par une poutre sur appuis élastiques, flexion
   longitudinale (H3) ;
6. configuration B avec son propre équilibre (H9) ;
7. contrôles de domaine : charge sur l'aile, cheville hors du congé et dans l'aile, valeurs d'unité suspectes (H4, H5,
   H9) ;
8. catalogue unique du site, r1 pour le congé, orientation séparée de la désignation (H8) ;
9. efforts ELU des chevilles tirés des combinaisons, le rapport ELU/ELS réel affiché (H7) ;
10. la carte de contraintes par éléments finis de la coupe (c03) est faisable et rapide (13 s au pas de 0,5 mm) :
    elle peut illustrer la note, à titre d'information, sans remplacer les contraintes nominales.

---

## 5. Limites de cette analyse

- **Modèle éléments finis** : coupe plane (contrainte plane, 1 mm de longueur), mur sans frottement, rondelle rigide
  Ø30 mm, raideur de cheville supposée (déplacement sous l'effort de service, de 0,01 à 1 mm), ni précontrainte, ni
  trou. T et la position de la résultante sous la rondelle dépendent de ces choix ; la sensibilité est donnée, pas une
  valeur unique. Équilibre global tenu à 10⁻⁸ N/mm près ; contrôle poutre loin du coin +0,29 % ; convergence entre 1 ; 0,5
  et 0,25 mm meilleure que 1 % sur tous les chiffres cités. Les résultantes lues exactement sur les coupes x = e et
  y = e sont moins précises (point de tangence singulier : effort normal parasite de 15 N/mm au pas de 0,5 mm) ; les
  contraintes maximales citées y sont stables d'un pas à l'autre (125,9 / 126,0 / 125,9 MPa).
- **Correspondance k (N/mm³) ↔ nature du support** et **déplacements des chevilles** : non calculés, à prendre dans
  les données du support et l'ETA de la cheville.
- **Analogies EN 1993-1-8** (0,8 r, longueurs efficaces du tronçon en T) : citées de mémoire, à confirmer dans le texte.
- **Partage G/Q** de R_max inconnu : les volées de la feuille E10A BD ne servent que d'illustration.
- Position des chevilles le long de b, largeur d'appui de la charge, diamètre de trou (Ø18 supposé pour C90) et de
  rondelle : non donnés par le classeur.
- Configuration B : statique de poutre seulement (pivot au bout libre décalé de 10 mm), sans éléments finis.

---

## 6. Scripts et sorties

Tous dans `scripts/` (lancés par `/usr/bin/python3.12 -I`, depuis un autre dossier que `source/`) ; relance complète :
`scripts/lancer_tout.sh`. Sorties chiffrées dans `sorties/`.

| script | objet | sortie |
|---|---|---|
| `commun.py` | la feuille P8 recopiée formule par formule ; géométrie exacte du congé | — |
| `c01_classeur.py` | relecture des formules, reproduction (45 cellules), RECHERCHEV, rayons face au catalogue du site | `c01_classeur.json` |
| `c02_conge.py` | hauteur de matière, coupes verticales / horizontales / radiales, balayage continu, catalogue, convention 0,8 r | `c02_conge.json` |
| `ef_coupe.py` | éléments finis plans de la coupe (maillage du congé exact, contact unilatéral, cheville) | — |
| `c03_ef.py` | cas de base, quatre sections, pic du congé, variantes cheville / mur / x_V, figures | `c03_ef_h0.5.json`, `c03_h0.5.txt`, trois `.png` |
| `c03b_convergence.py` | convergence au pas de 1 ; 0,5 ; 0,25 mm | `c03b_convergence.json`, `.txt` |
| `c03c_mur.py` | support souple comparé à base égale (nominal contre nominal), résidu d'équilibre | `c03c_mur.json` |
| `diag_coupes.py` | contrôle des résultantes de coupe face à la statique exacte | (écran) |
| `c04_ancrage.py` | pivots, équilibre, HILTI, jeux de 20 mm, sensibilité à Y et X, C88/C90 | `c04_ancrage.json` |
| `c05_cisaillement.py` | τ réel, nature des efforts dans l'aile verticale, seuil Y | `c05_cisaillement.json` |
| `c06_ec3.py` | admissible face à l'EC3 élastique et plastique | `c06_ec3.json` |
| `c07_largeur.py` | diffusion, chevilles, poutre continue (classeur et cornière de 2 m) | `c07_largeur.json` |
| `c08_generalisation.py` | cornière retournée, ailes inégales, effet de larg./haut. | `c08_generalisation.json` |
| `c09_site.py` | le moteur du site sur le cas de contrôle, défaut longitudinal | `c09_site.json` |
