# Platine sur béton — le CALCUL MANUEL (lot D)

Le module « Platine de pied de poteau sur béton » a deux modes. Le mode par
défaut est le **calcul manuel de la platine** : des boulons posés où l'on
veut, un ou plusieurs éléments soudés (profilé en I ou H, tube rectangulaire
ou circulaire, plat sur chant, rond plein, rectangle plein), des raidisseurs,
des efforts N, Mx, My, Vx, Vy, T appliqués en un point — et à la sortie
l'épaisseur que la platine doit avoir, lue comme sur une note à la main :
quel moment, quel bras de levier, quelle largeur de plat, quelle contrainte.
Le second mode est la **vérification complète** des lots A à C (Table 6.7,
tronçons en T par rangée, ancrages EN 1992-4) ; il reste disponible quand la
disposition est régulière (grille centrée autour d'un seul profilé en I ou
RHS, sans raidisseur), et l'outil le dit.

Les logiciels d'ancrage (Spit, Hilti) donnent les efforts des tiges et
vérifient le béton ; ils ne vérifient pas la platine, ou la cachent dans un
calcul par éléments finis. Ici le verdict porte sur la **platine** ; les tiges
(acier) et l'appui (fjd) sont donnés en information, pour comparer.

Code : `acier/assemblages/pied_poteau/platine_beton/manuel/` — `elements.py`
(géométrie des éléments et raidisseurs), `console.py` (bandes comprimées),
`moteur.py` (le calcul), `conversion.py` (passage entre les deux modes).
Tests : `tests/test_platine_manuelle.py` — chaque nombre y est recalculé à la
main.

## 1 · Ce que l'on saisit

| Objet | Saisies | Où |
|---|---|---|
| Platine | bp × lp × tp, nuance | cotes du dessin, carte |
| Boulons | d, classe ; liste des positions (x ; y) — un boulon se déplace sur le dessin ou se tape dans la table | carte, dessin |
| Éléments soudés | type, profilé du catalogue ou dimensions, position (x ; y), rotation 0 / 90° | carte, dessin |
| Raidisseurs | point de départ (x ; y), direction (±x, ±y), longueur L, épaisseur t, hauteur h (dessin) | carte, dessin |
| Appui | béton, massif (fjd par EN 1992-1-1 §6.7 et EN 1993-1-8 §6.2.5), mortier ; aire d'appui : bande c ou toute la platine | carte |
| Efforts | N (compression > 0), Mx (comprime y > 0), My (comprime x > 0), Vx, Vy, T ; point d'application (xN ; yN) | carte, dessin |
| Critère | EN 1993-1-8 (tronçon en T plastique, console élastique) ou tout élastique (Von Mises) | carte |

Repère : origine au centre de la platine, x et y du dessin en plan. Un
élément excentré par rapport aux boulons, ou des efforts appliqués hors du
centre, créent leur moment N·e tout seuls : `Mx = Mx,Ed + N·yN`,
`My = My,Ed + N·xN`.

## 2 · Le modèle, étape par étape

### 2.1 Les efforts des tiges — platine rigide

Même solveur que la vérification complète (`equilibre.elastique`) : la
platine est rigide, le béton est un lit de ressorts SANS traction sur l'aire
d'appui, les tiges sont des ressorts EN TRACTION seulement, le plan de
pénétration (trois inconnues) équilibre N, Mx, My. C'est le modèle que
l'EN 1992-4 (section 6) impose pour dériver les efforts des fixations, et
c'est celui des logiciels d'ancrage.

- **Aire d'appui** : par défaut la bande c de l'EN 1993-1-8 §6.2.5 autour de
  chaque paroi et de chaque raidisseur, `c = tp·√(fy/(3·fjd·γM0))`, découpée
  par la platine — un rond est découpé en 48 lamelles d'aire exacte ; en
  option toute la platine (platine rigide, comme les logiciels d'ancrage :
  moins de pression, plus de bras, tiges un peu moins tendues).
- **Raideurs** : `kc = Ecm/(1,275·√A)` (k13 d'EN 1993-1-8 §6.3.4, A = la plus
  grande bande, ou la platine), `kb = (1,6 ou 2,0)·E·As/Lb` (k16 ; 1,6 avec
  effet de levier, 2,0 sans), `Lb = 8d + tg + tp + 0,6d`.
- **Cisaillement** : V partagé entre les boulons, torsion T autour du centre
  du groupe (V appliqué hors du centre du groupe en ajoute), frottement
  `Cf,d·Nc` déduit (EN 1993-1-8 §6.2.2).

Ce que l'on voit : l'effort T de chaque boulon, la résultante C, la pression
maximale, la zone comprimée sur le dessin. Les efforts des tiges se comparent
à ceux du logiciel d'ancrage.

### 2.2 Côté tendu, boulon par boulon — le tronçon en T

Pour chaque boulon tendu :

1. **l'appui le plus proche** : le point le plus proche sur le bord d'une
   paroi (aile, âme, paroi de tube, plat, raidisseur) ou sur le cercle d'un
   rond. Si la projection du boulon tombe sur une face, le bras est
   perpendiculaire à la face ; sinon le boulon est **dans un angle** et le
   bras est la diagonale jusqu'à l'angle ;
2. **le bras** `m = d − 0,8·a·√2` (cordon déduit, EN 1993-1-8 Fig. 6.8),
   **le moment pris** `M = T·m` ;
3. **la longueur efficace** des lignes de rupture, par boulon (convention
   « demi-longueur » du Tableau 6.6, celle de la vérification complète : la
   somme sur une rangée de deux boulons redonne le tableau) :
   circulaire `min(π·m ; π·m/2 + s/2 ; π·m/2 + e_lat)`, non circulaire
   `min(2·m + 0,625·e ; e_lat/2 + m + 0,3125·e ; s/4 + m + 0,3125·e)` —
   `e` la pince dans la direction du bras, `e_lat` la pince le long de la
   face, `s` l'entraxe avec le boulon le plus proche — **plafonnée à ce que
   la face offre** : `L/2 + m` (la face de longueur L, plus la diffusion à
   45° sur le bras) ; `m` dans un angle (mécanisme en éventail autour de
   l'angle — modèle d'ingénieur, signalé) ; `R + m` sur un rond ;
4. **la résistance** du Tableau 6.2, modes 1, 2, 3, avec ou sans effet de
   levier selon `Lb ≤ Lb* = 8,8·m³·As/(leff·tp³)` — `FT,Rd` par boulon, et le
   taux `T/FT,Rd` ; ou, au critère élastique, la contrainte de la bande de
   charnière `2·leff` : `σ = 6·T·m/(2·leff·tp²)`, `τ = T/(2·leff·tp)`,
   `√(σ² + 3τ²) ≤ fy/γM0`.

Épaisseur demandée par ce boulon : mode 1 avec effet de levier
`tp = √(T·m·γM0/(leff·fy))`, sans effet de levier `√(2·T·m·γM0/(leff·fy))`,
élastique `√(3·T·m·γM0/(leff·fy))`.

### 2.3 Côté tendu, face par face — la section au nu

C'est le calcul classique : la platine fléchit autour du **nu de la paroi**
sous tous les boulons qui tirent devant elle.

- un boulon tire sur la face de son appui le plus proche — sur les DEUX
  faces de l'angle quand il est dans un angle, chacune avec son bras
  perpendiculaire (la platine fléchit autour de chacune des deux lignes) ;
- moment `M = Σ T·m⊥`, largeur `b = min(platine ; L + 2·m)` (diffusion à
  45° depuis les bouts de la face) ;
- `M ≤ Mpl = 0,25·b·tp²·fy/γM0`, ou élastique `σ = 6·M/(b·tp²)` composée
  avec `τ = ΣT/(b·tp)`.

C'est ce calcul qui **explose quand l'élément est étroit** : un rond fin à
l'axe donne un bras de 264 mm au lieu de 62 sur un RHS 300, et une largeur
de plat réduite au diamètre plus la diffusion. Sur un rond, chaque boulon a
sa section tangente (`L = 2R`). Même logique pour un raidisseur : un
raidisseur qui court le long du boulon raccourcit le bras et offre sa
longueur ; un boulon posé juste au-delà du BOUT d'un raidisseur de 10 mm
n'est tenu que par ce bout (`b = 10 + 2·m`), et l'épaisseur requise monte —
le modèle le dit, c'est voulu (tests, § 4).

### 2.4 Côté comprimé, bande par bande

Au nu de chaque face (enveloppe de l'élément, faces intérieures des ailes et
des parois d'un tube, quatre faces d'un raidisseur, cercle d'un rond), une
bande de largeur unité part en **console**, perpendiculaire à la face, et
porte la pression `p(x, y)` de la répartition sur ce qu'elle traverse de la
zone de contact : `m = ∫ p·t dt`. Autour d'un rond la bande est radiale et
s'élargit : `m = (1/R)·∫ p·t·(R + t) dt`.

**Règle de partage** : chaque paroi reprend la pression qui lui est plus
proche qu'à toute autre paroi (cellule de Voronoï). La bande s'arrête donc
au bord de la platine ou là où un autre appui devient plus proche : entre
deux ailes, chacune porte sa moitié en console ; près d'un raidisseur, c'est
lui qui porte ; dans le vide d'un tube, chaque côté porte jusqu'au centre.
Cette règle est du côté de la sécurité (une bande tenue aux deux bouts porte
moins : p·L²/12 contre p·L²/8), et elle est cohérente avec la bande c : en
mode bande c, le moment d'une bande ne dépasse jamais `p·c²/2`, et
l'épaisseur requise côté comprimé est celle pour laquelle la bande c porte
N à fjd — exactement la dérivation de c dans l'EN 1993-1-8.

On échantillonne la face (41 bandes) ou le cercle (72 rayons) et on garde la
bande la plus sollicitée — la bande d'un mètre « au point le plus comprimé »
du calcul à la main, la pression linéaire en plus. Contrôle élastique
`m ≤ tp²·fy/(6·γM0)` (§6.2.5).

### 2.5 Les trous — pression diamétrale

Tableau 3.4 de l'EN 1993-1-8 sur la platine, boulon par boulon, dans la
direction de son cisaillement : `Fb,Rd = k1·αb·fu·d·tp/γM2`,
`αb = min(e1/(3d0) ; p1/(3d0) − 1/4 ; fub/fu ; 1)`,
`k1 = min(2,8·e2/d0 − 1,7 ; 1,4·p2/d0 − 1,7 ; 2,5)` — e1, p1 dans la
direction de l'effort, e2, p2 perpendiculaires, lus sur la disposition libre
(un boulon « dans la même ligne » est à moins de d0 de côté). Tableau 3.3 :
`e ≥ 1,2·d0` au bord de la platine, `p ≥ 2,2·d0` entre boulons.

### 2.6 L'épaisseur requise

Pour chaque critère — tronçon en T de chaque boulon, section au nu de chaque
paroi, bande comprimée, pression diamétrale — la plus petite épaisseur qui
le satisfait, par balayage (tous les 2 mm jusqu'à 60, puis tous les 10, et
dichotomie) ; **la répartition est recalculée à chaque épaisseur essayée**
(bande c, raideurs, efforts des tiges, pressions) : une platine plus mince a
une bande c plus étroite, une résultante plus près du profilé et des tiges
un peu plus tendues. L'épaisseur requise est la plus grande des quatre ; le
verdict est « t_req ≤ tp », cohérent avec les vérifications à l'épaisseur
retenue (le balayage descend depuis tp tant que le critère tient).

## 3 · Ce que donnent les deux critères

- **EN 1993-1-8** (défaut) : côté tendu le tronçon en T plastique du
  Tableau 6.2 — avec effet de levier quand `Lb ≤ Lb*`, ce qui est le cas des
  platines minces sur tiges longues : la platine s'appuie sur le béton
  au-delà du boulon et le boulon reprend l'effort de levier Q, d'où
  `FT,Rd = 4·Mpl/m` ; côté comprimé la console élastique du §6.2.5.
- **Élastique** : partout `σ de Von Mises ≤ fy/γM0` sur la bande, sans
  effet de levier. Sans effet de levier l'épaisseur vaut √1,5 = 1,22 fois
  la plastique (rapport des modules élastique et plastique) ; avec effet de
  levier, jusqu'à √3 = 1,73 fois. Les deux sont affichées.

## 4 · Exemples vérifiés à la main (tests/test_platine_manuelle.py)

Platine 500 × 500 S355, HEA 300 à l'axe, 4 × M24 8.8, cordon a = 8.

| Cas | À la main | Résultat |
|---|---|---|
| Suspente N = −100 kN, boulons à (±200 ; ±200) | T = 25 kN ; angle : m = √(50² + 55²) − 9,05 = 65,3 mm, leff = m, Lb* = 847 > Lb = 261 → effet de levier, mode 1 : tp = √(T/fy) = 8,39 mm ; about d'aile : b = 14 + 2·40,9 = 96 mm, M = 25 × 40,9 → tp = √(4·M/(b·fy)) = 10,97 mm | t_T = 8,41 ; t_S = 10,97 |
| Suspente, boulons devant l'aile (±100 ; ±220) | m = 66,0 ; leff = 125,3 ; section b = 300 + 2·66 = 432 mm, M = 2·25·66 = 3,30 kNm → tp = 9,27 mm ; tronçon en T tp = 6,09 mm | t_S = 9,27 gouverne |
| Compression centrée N = 300 kN | p = N/Aeff = 2,32 MPa, bande m = p·c²/2 | exact |
| N = 2 000 kN | la bande c porte N à fjd pour c = 45,0 → tp = 18,9 mm | t_c = 18,9 |
| Vy = 200 kN, N = 0 | 50 kN par boulon ; boulon vers le bord : e1 = 50, p1 = ∞ (le boulon arrière : e1 = 450, p1 = 400), e2 = 50, p2 = 400 → αb = 50/78 = 0,64, k1 = 2,5 ; fu = 510 MPa (S355, EN 1993-1-1 Tab. 3.1) ; Fb,Rd = 2,5 × 0,641 × 510 × 24 × 25 / 1,25 = 392,3 kN ; tb = 50 000 × 1,25 / (2,5 × 0,641 × 510 × 24) = 3,19 mm | exact |
| Rond Ø20 à l'axe | bras 264 mm, pression ≫ fjd | aucune épaisseur ne suffit |

Le cas par défaut (N = 300, Mx = 50, My = 30, Vy = 40) donne une épaisseur
requise de 15,6 mm (about d'aile du boulon d'angle) contre 14,0 mm dans la
vérification complète (console moyenne à l'épaisseur retenue) : deux
conventions voisines, la manuelle recalculant tout à chaque épaisseur.

## 5 · Ce que le calcul manuel ne couvre pas

- les cordons de soudure, la résistance des raidisseurs eux-mêmes, le
  poteau : hors verdict ;
- le béton d'ancrage (cône, arrachement, bord, pry-out) : le logiciel
  d'ancrage ou la vérification complète ;
- les boulons d'angle : le bras diagonal et le plafond `leff = m` sont un
  modèle d'ingénieur (éventail autour de l'angle), signalé en alerte, comme
  dans la vérification complète (bras perpendiculaire, plafond 0,25·b) ;
- le mode « toute la platine » entre les ailes, dans le vide d'un tube et
  au-delà des abouts : la règle de Voronoï y est une approximation du côté
  de la sécurité (platine tenue sur trois côtés en réalité) ;
- une platine en pure traction (suspente) n'a pas de bande comprimée : le
  côté tendu décide seul.
