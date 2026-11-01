# Attaches articulées POUTRE – POTEAU — étude de méthode et plan

**Étude du 01/11/2026.** Dépôt `LLE4000/DEV_Etudes_structure`, branche
`claude/dalle-module-creation-sjmitr`, HEAD `383297b`, arbre propre.
Python de recette : `/usr/bin/python3.12`.

**Ce document ne contient aucun code d'application.** Il dit, pour TROIS
attaches articulées neuves — plat d'âme soudé, doubles cornières, platine
d'about de hauteur partielle, cette fois sur un **POTEAU** — ce qui se
vérifie, avec quelle référence, avec quelle nature de modèle, quelles clés,
quels dessins, quels cas de benchmark et dans quel ordre on les écrit. Il est
écrit pour piloter trois agents de réalisation : **ils ne doivent rien avoir à
inventer.**

Tout nombre écrit ici est **recalculé à la main, hors moteur**, par cinq
scripts jetables, dans
`/tmp/claude-0/-home-user/092afadd-796a-54d4-b802-037ae1cad731/scratchpad/pp/` :

| script | ce qu'il contrôle | résultat |
|---|---|---|
| `base.py` | la boîte à outils (formules, bibliothèques recopiées) | — |
| `c00_bibliotheques_et_heritage.py` | les bibliothèques recopiées **et** l'HÉRITAGE : les valeurs publiées du côté pièce d'attache / boulons / âme portée ne dépendent pas du support | 19 OK |
| `c01_fpc.py` | plat d'âme sur poteau, BM1 à BM4 + tying | 80 OK |
| `c02_pac.py` | platine d'about sur poteau, gauge, BM2 à BM4 + tying | 78 OK |
| `c03_dcc.py` | doubles cornières sur poteau, BM2 à BM4 + tying | 79 OK |
| `c04_taux_et_rapports.py` | les taux `η`, les rapports et les grandeurs citées sans être imprimées ailleurs | 35 OK |

**291 contrôles, 0 écart.** Chaque grandeur y est calculée **deux fois** : par
une fonction (celle que le moteur reproduira), puis par une expression
littérale indépendante. Aucun nombre de mémoire — deux erreurs de calcul
mental ont d'ailleurs été attrapées par ces scripts pendant la rédaction
(`λLT` du cas b1, `F,Ed` du cas b1 redimensionné).

---

## 1. Objet et périmètre

### 1.1 Les trois attaches

| id du registre | nom | grand frère poutre–poutre |
|---|---|---|
| `plat_ame_poteau` | Plat d'âme soudé sur poteau (fin plate) | `plat_ame` |
| `doubles_cornieres_poteau` | Doubles cornières d'âme sur poteau | `doubles_cornieres` |
| `platine_about_poteau` | Platine d'about de hauteur partielle sur poteau | `platine_about` |

Les trois sont des assemblages **articulés** qui ne transmettent qu'un effort
tranchant vertical `VEd` (plus, le cas échéant, un effort normal `NEd` — ELU
fondamental ou force de chaînage, et un moment parasite `MEd` pris en valeur
absolue). Famille neuve au registre : **« Poutre – Poteau »**.

### 1.2 Les deux cas d'appui, tous les deux traités

Repère commun à toute l'étude, poteau en I ou H, âme verticale :

- **X** : horizontal, perpendiculaire aux ailes du poteau ;
- **Y** : horizontal, parallèle aux ailes (largeur d'aile `bP`) ;
- **Z** : vertical (axe du poteau).

En coupe horizontale, l'âme du poteau est un segment de longueur
`hP − 2·tfP` **selon X**, d'épaisseur `twP` selon Y ; les deux ailes sont des
segments de longueur `bP` **selon Y**, d'épaisseur `tfP` selon X, aux
abscisses `X = ±hP/2`.

| | **cas (a) — sur l'AILE** | **cas (b) — dans l'ÂME** |
|---|---|---|
| axe du poteau sollicité | **fort** (majeur) | **faible** (mineur) |
| axe de la poutre portée | **X** (⊥ aux ailes) | **Y** (∥ aux ailes) |
| face d'appui | face extérieure d'une aile, `X = hP/2` | face de l'âme, `X = ±twP/2` |
| plan de l'âme de la poutre portée | **parallèle** à l'âme du poteau | **perpendiculaire** à l'âme du poteau |
| plan de la pièce d'attache | **dans le plan de l'âme du poteau** | **perpendiculaire** à l'âme du poteau |
| obstacle géométrique | aucun | les AILES du poteau, en Y |

**C'est la ligne « plan de la pièce d'attache » qui commande toute la
mécanique**, et c'est le résultat central de cette étude :

> **En cas (a), l'attache (plat, dos des cornières, platine) est centrée sur
> l'âme du poteau et vit DANS SON PLAN. L'effort tranchant `VEd` et le moment
> d'excentricité `VEd·z` y cheminent EN PLAN : il n'y a pas de flexion hors
> plan de l'aile sous l'effort tranchant seul, et aucune vérification locale
> de l'aile n'est demandée par les guides au-delà du cordon (attache soudée)
> ou de la pression diamétrale (attache boulonnée). La flexion locale de
> l'aile n'apparaît QUE lorsque les boulons sont tendus — donc en **tying**,
> et à l'ELU seulement si `NEd` de traction existe.
>
> En cas (b), l'attache est perpendiculaire à l'âme du poteau. `VEd` reste en
> plan dans l'âme (d'où le cisaillement local du modèle SCI P358, chargé d'un
> seul côté), mais la composante `qx` du moment `VEd·z` et toute traction des
> boulons sollicitent l'âme **hors de son plan** — ce que l'EN 1993-1-8 ne
> couvre pas pour une âme non raidie. C'est la limite la plus importante du
> cas (b), et elle est écrite à l'écran.

### 1.3 Hors périmètre, et pourquoi

| Hors périmètre | Pourquoi |
|---|---|
| Attache **rigide** ou semi-rigide (platine pleine hauteur, moment) | Un autre modèle (EN 1993-1-8 §6 complet, panneau d'âme, raidisseurs) ; ce ne sont pas des attaches simples. |
| **Panneau d'âme du poteau en cisaillement** sous les moments d'étage | L'attache simple n'apporte pas de moment ; le panneau se vérifie avec l'ensemble des attaches de l'étage, pas avec une attache (§6.2.6.1 et son coefficient `β`). |
| Poteau sous **charges combinées** (N + M + V), flambement, déversement | Calcul d'ÉLÉMENT (EN 1993-1-1 §6.2 et §6.3) ; l'attache lui apporte `VEd` et le moment `VEd·e` à son axe, qui s'y traitent. |
| **Attaches des deux côtés** du poteau partageant les mêmes boulons ou la même âme | Interaction d'efforts opposés : la pression diamétrale, le cisaillement local et le tying changent de modèle. Déclaré non traité. |
| **Torsion** induite par une attache d'un seul côté | Même raison : équilibre d'élément. |
| Poteau **creux** (RHS / SHS / CHS) | Autre composant (plaque sur paroi, EN 1993-1-8 §7) ; le registre a déjà un module « pied de poteau » pour les tubes, ce n'en est pas un. |
| **Goussets, jarrets, raidisseurs** de l'aile ou de l'âme du poteau | Le périmètre est l'attache NON RAIDIE : les longueurs efficaces du Tableau 6.4 (aile non raidie) et les limites de l'âme en supposent l'absence. |
| **ELS**, flèches, vibrations | Comme les trois modules existants. |

---

## 2. Sources, et ce qu'elles couvrent vraiment

| Source | Ce qu'on en tire | Lu / de mémoire |
|---|---|---|
| **EN 1993-1-8:2005** | Tab. 3.3 (pinces, entraxes), Tab. 3.4 (boulons, pression diamétrale, poinçonnement), Tab. 3.6 (ks), §3.4.1, §3.7 (groupe de fixations), §3.8 (βLf), §3.9.1 et §3.9.2 (glissement), §3.10.2(2) et (3) (rupture de bloc), §4.5.1(2), §4.5.2(2), §4.5.3.3 (cordons), §5.2 (classification), §6.2.4 + **Tab. 6.2** (tronçon en T, modes 1 à 3), **Tab. 6.4** (longueurs efficaces d'une **aile de poteau non raidie**), §6.2.6.3 (âme de poteau en traction transversale) | numéros déjà cités par les trois modules existants ; **texte NBN payant, non consulté ici** |
| **MSB Part 5** (*Steel Buildings in Europe — Multi-Storey Steel Buildings, Part 5: Joint Design*, ArcelorMittal / Peiner Träger / Corus, contenu CTICM et SCI, 06/2009) | §2 platine d'about partielle (§2.1 détails, §2.2 cisaillement, §2.2.1, §2.2.4, §2.3 **tying**, §2.4 exemple), §3 fin plate (§3.2, §3.2.3, §3.2.5.1, §3.4 exemple), §4 doubles cornières (§4.2.x, §4.3.x, §4.4 exemple) | sous-clauses reprises **telles quelles des modules existants**, dont les benchmarks reproduisent numériquement les exemples §2.4, §3.4 et §4.4 |
| **SCI P358** | modèle « élément support chargé d'un seul côté » (cisaillement local), pression diamétrale de l'élément support, plats LONGS (déversement), numéros de « Check » | **cités de mémoire** dans le dépôt depuis le 21/09/2026, listés non vérifiés dans `ECARTS_ET_CORRECTIONS.md` §2 — l'étude ne les corrige pas et n'en ajoute aucun |
| **ECCS n°126** | facteur 1,27 du cisaillement brut, sections nettes, ductilité des platines minces | idem |
| **EN 1993-1-1:2005** | Tab. 3.1, §6.2.3(2), §6.2.5, §6.2.6 | idem |
| **BS 5950-1:2000 Annexe B.2.1** (courbe de la Table 17) | déversement d'un plat long, `E = 205 000 MPa` | déjà dans le module `plat_ame`, benchmark VAL-A |

**Règle de citation, reprise du dépôt** : un `§` sans indication est de
l'EN 1993-1-8 ; un `§` du guide porte TOUJOURS « MSB P5 ». **Aucun numéro de
sous-clause n'est inventé** : là où la sous-clause n'a pas été lue, la
référence reste au chapitre (« MSB P5 §3.2 »).

---

## 3. Côté POUTRE PORTÉE et PIÈCE D'ATTACHE : ce qui est repris, ce qui change

### 3.1 Repris TEL QUEL des trois modules existants

**C'est la majeure partie du travail, et c'est un acquis mesuré.** Le script
`c00_bibliotheques_et_heritage.py` le prouve : en rejouant les trois exemples
publiés avec un support dont on ne change QUE l'épaisseur et la nuance, les
valeurs publiées du côté pièce d'attache, boulons et âme portée sortent
**inchangées au 1e-9** — **9** vérifications contrôlées pour le plat d'âme,
**8** pour la platine d'about, **10** pour les cornières. Seules dépendent du
support : le `fu` du cordon (`min(fu pièce ; fu support)`) et les lignes du
groupe « élément support ».

Sont donc repris **sans aucune modification de formule** :

1. **Boulons** côté poutre portée : `Fv,Rd = αv·fub·A/γM2` (Tab. 3.4), un plan
   (plat, platine) ou deux (cornières) ; groupe excentré en **répartition
   élastique sur l'inertie polaire** `Ip = n·[p1²(n1² − 1) + p2²(n2² − 1)]/12`
   (MSB P5 §3.2) ; `βLf` en option (§3.8) ; glissement §3.9.1/§3.9.2 (cat. B
   et C) ; `k_rot = 0,8` du groupe P des cornières (MSB P5 §4.2.1.2) ; la
   règle de groupe en trois branches de la platine d'about (MSB P5 §2.2.4 sur
   §3.7).
2. **Pinces et entraxes du Tableau 3.3** : `e ≥ 1,2·d0`, `p1 ≥ 2,2·d0`,
   `p2 ≥ 2,4·d0`, `p ≤ min(14t ; 200 mm)`, `e ≤ 4t + 40 mm` si exposé.
3. **Pièce d'attache** : pression diamétrale dans les deux directions avec
   interaction quadratique `η = √[(Fz/Fb,ver)² + (Fx/Fb,hor)²]` (Tab. 3.4 +
   MSB P5) ; cisaillement brut avec le facteur **1,27** (ECCS n°126) ; net ;
   rupture de bloc §3.10.2 ; flexion dans le plan (sans objet si
   `hp ≥ 2,73·z`) ; **déversement du plat LONG** `z > tp/0,15` par
   `Mb,Rd = Wel·pb(λLT)/γM1`, `λLT = 2,8·√(z·hp/(1,5·tp²))`, courbe BS 5950-1 ;
   tronçon en T de la platine et des ailes A des cornières (§6.2.4, Tab. 6.2,
   méthode 2) ; poinçonnement.
4. **Règles de ductilité et de détail** : `tp ≤ 0,5·d` (plat, SCI P358 /
   MSB P5), `tp ≤ 0,36·d·√(fub/fy)` (platine, §6.4.2(2) / ECCS n°126),
   `tp ≥ 8 mm` recommandé (MSB P5 §2.1), `hp ≥ 0,6·h` de la portée (maintien
   en torsion), gorge de **pleine résistance** recommandée
   (`a ≥ tp·fy·βw·γM2/(2·fu·γM0)` pour un plat soudé sur le support ;
   `a ≥ βw·γM2·fy·tw/(√2·fu·γM0)` = 0,46 / 0,48 / 0,55·tw en S235 / S275 / S355
   pour une platine soudée sur l'âme portée), trou de boulon hors du congé de
   la cornière (`gA − d0/2 ≥ tc + rc`), la rondelle dégage le cordon
   (`p3 ≥ tw + 2·a·√2 + dw`).
5. **Âme de la poutre portée** : pression diamétrale ; cisaillement brut sur
   l'aire `Av = max(A − 2b·tf + (tw + 2r)·tf ; (h − 2tf)·tw)` (§6.2.6(3)) ; net ;
   rupture de bloc avec `(n1 − 1)·d0` si la poutre n'est **pas** grugée et
   `(n1 − 0,5)·d0` sinon (MSB P5 §3.2.5.1) ; traction section nette §6.2.3(2) ;
   rupture de bloc en traction §3.10.2(2).
6. **Grugeage** : flexion + cisaillement à l'extrémité du grugeage, réduction
   `ρ = 1 − (2V/Vpl − 1)²`, stabilité locale en longueur (`c ≤ h` si
   `h/tw ≤ 54,3` (S275) / `48` (S355), sinon `c ≤ K·h/(h/tw)³` avec
   `K = 160 000 / 110 000`) et en profondeur (`dc ≤ h/2` une semelle,
   `h/5` deux semelles) — MSB P5 §4.2.4 / §4.2.5.
7. **Soudures** : méthode simplifiée §4.5.3.3, `fvw,d = fu/(√3·βw·γM2)`,
   gorge minimale §4.5.2(2), longueur minimale §4.5.1(2).
8. **Prédimensionnement**, **note 2 pages**, **export texte**, **kit de
   dessin**, **carte de saisie**, **onglets** : repris dans leur forme.

### 3.2 Ce qui CHANGE côté poutre portée

#### 3.2.1 Grugeage : nécessaire ou non

Dans les modules poutre–poutre, le grugeage existe pour **dégager la semelle
et le congé de la poutre porteuse**. Un poteau n'a pas de semelle au-dessus
ni en dessous de l'attache.

| | grugeage en profondeur (`dc,sup`, `dc,inf`) |
|---|---|
| **cas (a) aile** | **inutile** — rien n'obstrue au-dessus ni en dessous. Défaut `dc,sup = dc,inf = 0`. |
| **cas (b) âme** | **inutile aussi** : l'obstacle est latéral (en Y), pas vertical. |

**Décision** : les clés `d_nt`, `d_nb`, `l_n`, `r_n` sont **conservées**, à
**0 par défaut**, et descendues dans « Paramètres avancés › GRUGEAGE » — un
projeteur grugera encore pour passer une platine de tête, un gousset ou un
raidisseur. Quand elles sont non nulles, **tout le bloc de la poutre grugée
s'active à l'identique** (MSB P5 §4.2.4 / §4.2.5, `cas_g`, `Av`, `W_N`, `ρ`,
`ln_max`, `val_N`). Les alertes `dnt_min`, `dnb_min`, `ln_min`, `gap_neg`,
**disparaissent** : elles contrôlaient le dégagement de la porteuse.

#### 3.2.2 L'obstacle du cas (b) : les ailes du poteau, et la RECOUPE

En cas (b) la poutre entre **entre les ailes du poteau**. Ses propres semelles
occupent `Y ∈ [−bS/2 ; +bS/2]` ; les faces intérieures des ailes du poteau
sont en `Y = ±(hP/2 − tfP)`. Deux grandeurs commandent :

- **largeur droite entre congés** `hP − 2·tfP − 2·rP` — c'est elle qui limite
  la **pièce d'attache** posée sur l'âme (largeur `bp` d'une platine, entraxe
  `p3`, longueur d'aile A d'une cornière) ;
- **largeur libre entre ailes** `hP − 2·tfP` — c'est elle qui limite la
  **semelle de la poutre portée**.

Mesuré sur le poteau de référence **HEB 300** (`h 300, b 300, tw 11, tf 19,
r 27`) : entre congés **208 mm**, demi-largeur droite **104 mm**, entre ailes
262 mm.

Trois configurations, et c'est une **décision d'étude** :

| variante | condition | conséquence |
|---|---|---|
| **b0 — la poutre passe** | `bS ≤ hP − 2·tfP − 2·j` (`j` = jeu latéral, 5 mm par défaut) | rien de plus. Exemple : IPE 400 (`bS = 180`) dans un HEB 300 → `180 ≤ 252` ✔ : **c'est le cas courant des IPE dans un HEB**. |
| **b1 — la poutre reste au-delà des bouts d'aile** | `gh ≥ (bP − twP)/2 + j` | la pièce d'attache doit traverser tout le demi-poteau : **`gh ≥ 154,5 mm` pour un HEB 300**. Possible UNIQUEMENT pour un plat d'âme (la seule pièce qui dépasse). `z` explose, le plat devient LONG. |
| **b2 — ailes de la poutre RECOUPÉES** | recoupe `≥ (bP − twP)/2 + j − gh` sur la LARGEUR des semelles, ramenée à `≤ hP − 2·tfP − 2·j` | **`recoupe ≥ 144,5 mm` pour un HEB 300 avec `gh = 10`**. Obligatoire pour une platine d'about ou des cornières dès que la poutre ne passe pas (par ex. HEA 300 portée, `bS = 300 > 252`). |

**La recoupe des semelles en LARGEUR n'est pas un grugeage.** Les règles de
stabilité locale de MSB P5 §4.2.5 portent sur une entaille en HAUTEUR (`dc`,
`c`) et sur le Té résiduel ; elles ne s'appliquent pas et **ne sont pas
activées** par la recoupe. Ce qui reste vrai et déjà vérifié : l'âme seule
transmet `VEd` (cisaillement brut, net, rupture de bloc de l'âme), et la
recoupe se fait au voisinage de la rotule, là où le moment est nul. **Point
tranché sans source franche, écrit en clair** (§9, P-3).

#### 3.2.3 Longueur / hauteur de la pièce d'attache

| | plat d'âme `bp` | cornières `bA`, `bB` | platine `bp` |
|---|---|---|---|
| **cas (a)** | `bp ≥ gh + e2,b + (n2−1)·p2 + e2`, inchangé | `bB` inchangé ; `bA ≥ gA + e2,A` avec `gA` sur l'aile du poteau | `bp = p3 + 2·e2`, limité par `bP` |
| **cas (b)** | b1 : `bp ≥ gh + e2,b + e2` avec `gh ≥ 154,5` → `bp ≈ 245 mm` ; b2 : inchangé | `tw,S/2 + bA ≤ (hP − 2tfP)/2 − rP` → **`bA ≤ 99,7 mm` pour un HEB 300 avec une âme portée de 8,6** (une L100 ne passe pas : 104,3 > 104 ; une **L90 passe**) | `bp ≤ hP − 2·tfP − 2·rP − 2·j` → **`bp ≤ 198 mm`**, donc `p3 + 2·e2 ≤ 198` |

La **hauteur** (`hp`, `hc`) reste commandée par la poutre portée
(`hp ≥ 0,6·h`, `zp + hp ≤ h − (tf + r)` ou `h − dc,inf`) : le poteau ne la
limite pas, il n'a pas de semelles horizontales. **La contrainte « la pièce
d'attache reste dans la partie droite de l'âme porteuse » (alerte `pl_web`
des modules existants) DISPARAÎT** — en cas (a) comme en cas (b), la pièce est
libre sur toute la hauteur du poteau.

---

## 4. Côté POTEAU, cas (a) : sur l'AILE

### 4.1 Ce qui se vérifie vraiment, et ce qui ne se vérifie pas

| | plat d'âme (SOUDÉ sur l'aile) | cornières / platine (BOULONNÉES sur l'aile) |
|---|---|---|
| **cordons sur l'aile** | OUI — §4.5.3.3 avec `fu = min(fu pièce ; fu aile)`, `βw` de l'acier le plus faible | sans objet (sauf cornières à ailes A soudées : §4.5.3.3 sur l'aile) |
| **pression diamétrale sur l'aile** | sans objet | **OUI** — Tab. 3.4 avec `t = tfP` (§4.2) |
| **aile en flexion locale (tronçon en T)** | **NON** (voir 4.4) | **NON à l'ELU** sous `VEd` seul ; **OUI en tying** et à l'ELU si `NEd > 0` (§7) |
| **poinçonnement de l'aile** | sans objet | OUI dès que les boulons sont tendus (Tab. 3.4) |
| **cisaillement local de l'aile** | **NON** — aucun modèle publié, voir 4.4 | **NON** |
| **âme du poteau en traction transversale** | NON (l'effort est dans le plan de l'âme) | **OUI en tying** (§6.2.6.3, `ω = 1`) |
| **panneau d'âme en cisaillement** | NON — hors périmètre (§1.3) | NON — hors périmètre |

### 4.2 Pression diamétrale sur l'aile : deux particularités

**Il n'y a PAS de pince d'extrémité verticale** : le poteau continue au-dessus
et en dessous de l'attache. Donc, pour TOUTES les rangées :

```
αb   = min( p1/(3·d0) − 1/4 ;  fub/fu ;  1 )          ← aucun terme e1/(3·d0)
k1   = min( 2,8·e/d0 − 1,7 ;  1,4·p3/d0 − 1,7 ;  2,5 )
Fb,Rd = k1 · αb · fu · d · tfP / γM2
```

avec `e = (bP − p3)/2` la pince au **bout d'aile**. C'est exactement le motif
`bearing(..., e1=None, ...)` déjà écrit dans le moteur des doubles cornières
pour l'âme porteuse — il se transpose sans modification.

Contrôlé (HEB 300 S355, M20, `p1 = 70`, `p3 = 140`) : `αb = 0,810606`,
`k1 = 2,5`, **`Fb,Rd = 314,19 kN par boulon`** ; avec `p3 = 108,6`
(cornières, `gA = 50`) : `e = 95,7 mm`, mêmes `αb` et `k1`, même `Fb,Rd`.

### 4.3 La PINCE et l'ENTRAXE sur l'aile — le « gauge »

Quatre conditions, et la première est NEUVE :

1. **L'écrou dégage le congé de raccordement du poteau** (il est du côté
   intérieur de l'aile, au droit du congé) :
   ```
   p3 / 2 − dw / 2  ≥  twP / 2 + rP        soit    p3 ≥ twP + 2·rP + dw
   ```
   **HEB 300 + M20 : `p3 ≥ 11 + 54 + 37 = 102 mm`.** Conséquence directe et
   importante : **le `p3 = 90 mm` usuel des ÂMES de poutre NE PASSE PAS sur
   l'aile d'un HEB 300** (26,5 mm disponibles pour 32,5 nécessaires). C'est
   la raison des « gauges » usuels d'aile de poteau : **90 / 100 / 120 /
   140 mm selon la série**. Nature **INT** (règle géométrique de l'outil,
   même mécanisme que l'alerte `fillet` des cornières et `p3_cordon` de la
   platine) — alerte **bloquante**.
2. **Pince au bout d'aile** : `e = (bP − p3)/2 ≥ 1,2·d0` (Tab. 3.3) et
   `≤ 4·tfP + 40` si exposé. HEB 300, `p3 = 140` → `e = 80 ≥ 26,4` ✔.
3. **Entraxe** : `p3 ≥ 2,4·d0` (Tab. 3.3, `p2` du tableau) et
   `p3 ≤ min(14·t ; 200)`.
4. **Largeur de la pièce** : `bp = p3 + 2·e2` avec `e2 ≥ 1,2·d0`, et
   `bp ≤ bP` (la platine ne dépasse pas l'aile).

**Valeur proposée par le prédimensionnement** :
`p3 = ceil5( max( 90 ; 2,4·d0 ; twP + 2·rP + dw ) )`, plafonnée par
`bP − 2·1,2·d0`. HEB 300 + M20 → `max(90 ; 52,8 ; 102) = 102 → 105 mm`
(le projeteur mettra 140, l'outil ne l'impose pas).

### 4.4 Épaisseur minimale de l'aile, position par rapport à l'âme

**Épaisseur minimale : AUCUNE règle normative n'est appliquée, et c'est
écrit.** Ce sont la pression diamétrale (ELU) et le tronçon en T (tying) qui
décident de l'épaisseur nécessaire de l'aile ; ni l'EN 1993-1-8 ni les guides
ne posent de minimum d'épaisseur d'aile pour une attache simple. L'outil n'en
pose donc pas. **Point tranché sans source franche** (§9, P-4).

**Position : l'attache est CENTRÉE sur l'âme du poteau.** C'est l'hypothèse
qui autorise tout le §1.2 :

- plat d'âme : la ligne de cordon est au droit de l'âme ; `VEd` et le couple
  `qx = 3·MS/hp²` normal à l'aile sont repris **en plan** par l'âme du
  poteau ;
- cornières / platine : les deux files sont **symétriques** par construction
  (`±p3/2`), et la résultante passe par l'âme.

Aucune clé d'excentricité n'est offerte : une attache décentrée sortirait du
domaine (flexion locale de l'aile sous `VEd`, torsion du poteau). C'est écrit
dans « Méthode › Limites » et dans le bloc HYPOTHÈSES de la note.

**Aucune vérification locale de l'aile sous l'effort tranchant seul.** Pour
l'attache SOUDÉE (plat d'âme, cas a), le groupe « Poteau porteur » est donc
**vide**, et l'écran le dit dans le repli « Périmètre de la vérification » :

> « Aile du poteau : l'attache est dans le plan de l'âme du poteau ; VEd et
> VEd·z y cheminent en plan. Aucune vérification locale de l'aile n'est
> demandée par MSB P5 §3.2 ni par SCI P358 ; la section du poteau se vérifie
> dans son propre calcul (EN 1993-1-1 §6.2). »

Le kit traite déjà ce cas sans rien casser : `synthese.taux_par_element` ne
rend un élément que s'il porte une vérification active (`if on:`), et
`synthese.lignes` vide fait que `table_simple` n'imprime rien.

---

## 5. Côté POTEAU, cas (b) : dans l'ÂME

### 5.1 Cisaillement local de l'âme du poteau — QUEL modèle, et pourquoi

**Le modèle est celui de `vlP` des trois modules existants, repris sans
changement de formule** : SCI P358, « élément support chargé d'un seul côté »,

```
VEd / 2   ≤   Av · fy,P / (√3 · γM0)
```

**Pourquoi celui-là et pas un autre** : en cas (b), l'âme du poteau joue
exactement le rôle que joue l'âme de la poutre porteuse dans les modules
poutre–poutre — une âme verticale, plane, non raidie, recevant l'effort
tranchant d'une attache posée sur UNE de ses deux faces, et devant l'évacuer
en cisaillement **dans son plan**, vers le haut et vers le bas. La géométrie
est la même, le cheminement est le même, le `VEd/2` a la même origine (un seul
côté chargé). Transposer, c'est reconnaître le même composant ; en changer,
ce serait inventer.

L'aire `Av` suit la pièce, comme dans les grands frères :

| module | `Av` | source de la forme |
|---|---|---|
| plat d'âme, platine d'about | `twP · hp` | `plat_ame` / `platine_about` (`Av_P = tw_P·h_p`) |
| doubles cornières | `twP · (et + (n1P − 1)·p1 + eb)` | `doubles_cornieres` (`Av_P`) |

Pour les cornières, les bornes `et` et `eb` du grand frère valent
`et = min(zt ; 5·d)` et `eb = min(zb ; p3/2 ; 5·d)` où `zt`, `zb` sont les
distances aux faces de la poutre porteuse. **Un poteau n'a pas de face
horizontale** : `zt` et `zb` sont sans objet, les bornes se réduisent à
`et = 5·d` et `eb = min(p3/2 ; 5·d)`. Contrôlé (HEB 300, M20, `n1P = 4`,
`p1 = 70`, `p3 = 108,6`) : `et = 100`, `eb = 54,3`, **`Av = 4 007,3 mm²`**,
`VRd brut = 821,33 kN`, `VRd net = 715,93 kN`, pour `Ed = VEd/2 = 100 kN`.

Section **nette** (`vnP`) : active **seulement si l'attache est boulonnée dans
l'âme** (des trous y existent). Pour un plat d'âme SOUDÉ sur l'âme, `vnP` est
sans objet.

### 5.2 Pression diamétrale sur l'âme du poteau

Même particularité que l'aile, en plus franche : **aucun bord libre, ni
vertical ni horizontal** (l'âme court entre les deux ailes et sur toute la
hauteur du poteau) :

```
αb = min( p1/(3·d0) − 1/4 ;  fub/fu ;  1 )
k1 = min( 1,4·p3/d0 − 1,7 ;  2,5 )
Fb,Rd = k1 · αb · fu,P · d · twP / γM2
```

C'est **mot pour mot** l'appel `bearing("P", tw_P, fu_P, None, p1, n1, None,
p2, n2, …, exV = 1,4·p3/d0 − 1,7, exH = p3/(3·d0) − 1/4)` du moteur des
doubles cornières. Contrôlé (HEB 300, M20, `p1 = 70`, `p3 = 110`) :
`αb = 0,810606`, `k1 = 2,5`, **`Fb,Rd = 181,90 kN par boulon`**.

### 5.3 Hauteur libre entre congés — les trois conditions du cas (b)

```
largeur droite de l'âme :  hP − 2·tfP − 2·rP            (HEB 300 : 208 mm)
demi-largeur droite     : (hP − 2·tfP)/2 − rP           (HEB 300 : 104 mm)
```

1. **La pièce d'attache tient** : `bp ≤ hP − 2·tfP − 2·rP − 2·j` → **198 mm**
   pour un HEB 300 (`j = 5`). Alerte bloquante sinon.
   → la platine 290 × **220** du cas (a) **ne passe pas** en cas (b) ; il faut
   `p3 = 110` et `e2 = 40`, soit `bp = 190 ≤ 198` ✔.
2. **L'écrou dégage le congé** : `p3/2 + dw/2 ≤ (hP − 2·tfP)/2 − rP`.
   Contrôlé avec `p3 = 110`, M20 : `55 + 18,5 = 73,5 ≤ 104` ✔, **marge
   30,5 mm**.
3. **Le bout de l'aile A d'une cornière dégage le congé** :
   `tw,S/2 + bA ≤ (hP − 2·tfP)/2 − rP` → **`bA ≤ 99,7 mm`** (HEB 300, âme
   portée 8,6). Une **L100 ne passe pas** (`104,3 > 104`), une **L90 passe**
   (`94,3`). Alerte bloquante, et le prédimensionnement plafonne l'aile A en
   conséquence.

### 5.4 Poinçonnement et flexion de l'âme en tying

- **Poinçonnement** : `Bp,Rd = 0,6·π·dm·twP·fu,P/γMu` (Tab. 3.4), avec
  `t·fu` minimal des pièces sous la tête ou l'écrou — c'est exactement ce que
  `platine_about` calcule déjà. Contrôlé (HEB 300, M20) :
  **`Bp,Rd = 305,70 kN`**.
- **Flexion hors plan de l'âme du poteau sous la traction des boulons :
  NON VÉRIFIÉE, et c'est écrit.** Raison, en clair : le tronçon en T de
  l'EN 1993-1-8 §6.2.4 modélise une aile fléchie **encastrée sur une âme
  centrale** (`m` se mesure depuis le pied de l'âme). Une âme de poteau
  chargée hors plan entre ses deux ailes est une plaque appuyée sur **deux
  bords**, à `(hP − 2·tfP)/2` des files de boulons : le modèle ne transpose
  pas, les longueurs efficaces du Tab. 6.4 ne s'appliquent pas, et
  l'EN 1993-1-8 ne donne rien pour ce composant non raidi. Conséquence
  pratique écrite à l'écran :

  > « Cas (b) : en tying, seuls les boulons, le poinçonnement, la pièce
  > d'attache et l'âme de la poutre portée sont vérifiés. La **flexion hors
  > plan de l'âme du poteau** n'est pas couverte par l'EN 1993-1-8 pour une
  > âme non raidie : elle est à justifier séparément, ou l'attache se raidit
  > (plat traversant, raidisseurs entre ailes). »

  Même avertissement pour la composante `qx = 3·MS/hp²` du moment
  d'excentricité en cas (b) — avec, pour le plat d'âme en variante **b1**, une
  alerte supplémentaire (`z` y vaut 3 à 4 fois sa valeur courante).

---

## 6. Excentricités et bras de levier

### 6.1 Où est la face d'appui

| module | cas (a) | cas (b) |
|---|---|---|
| plat d'âme | face EXTÉRIEURE de l'aile (`X = hP/2`) | face de l'ÂME (`X = twP/2`) |
| doubles cornières | face extérieure de l'aile (les ailes A y portent) | face de l'âme |
| platine d'about | face extérieure de l'aile (le dos de la platine y porte) | face de l'âme |

La **rotule est à la face d'appui**, comme dans les trois grands frères
(MSB P5 §2.2 / §3.2 / §4.2.1.1) — c'est l'origine de `z`.

### 6.2 Le bras `z` du groupe de boulons

| module | `z` | cas (a) | cas (b), b0/b2 | cas (b), b1 |
|---|---|---|---|---|
| **plat d'âme** | `z = gh + e2,b + (n2 − 1)·p2/2` | `50 mm` (gh 10, e2,b 40) | `50 mm` | **`195 mm`** (`gh ≥ 154,5`) |
| **cornières** | `z_S = gh + e2,b + (n2S − 1)·p2/2` (groupe S) ; groupe P **centré** | `50 mm` | `50 mm` | sans objet (b1 impossible) |
| **platine d'about** | **`z = 0`** : groupe CENTRÉ, deux files symétriques, aucun moment d'excentricité | — | — | sans objet |

Moment repris par la pièce et le groupe : `MS = VEd·z + |MEd|`.

- plat d'âme, cas (a) par défaut : `MS = 200 × 0,050 = 10,0 kNm` ;
- plat d'âme, cas (b) variante b1 : `MS = 200 × 0,195 = 39,0 kNm` — **× 3,9**.
- cornières : chaque cornière reprend `MEd = (VEd/2)·z` dans son plan
  (`flB`) ; le groupe P reste centré, la traction parasite de rotation est
  couverte par `k_rot = 0,8` sur `Fv,Rd` (MSB P5 §4.2.1.2).
- platine d'about : la flexion nominale de la platine dans son plan est
  couverte par le facteur **1,27** du cisaillement brut (MSB P5 §2.2).

### 6.3 Le moment que l'attache apporte au POTEAU

`MEd,poteau = VEd · e` avec `e` mesuré de l'**axe du poteau** au point
d'application :

| | `e` |
|---|---|
| cas (a) | `hP/2 + z` (plat, cornières) ou `hP/2 + tp` (platine) |
| cas (b) | `z` (plat, cornières) ou `tp` (platine) |

**Ce moment n'est PAS une vérification de l'attache** : il appartient au
calcul d'élément du poteau (EN 1993-1-1 §6.2, §6.3). L'outil l'AFFICHE dans
« Paramètres retenus » et dans le bloc HYPOTHÈSES de la note, pour que le
projeteur le reporte — il ne le vérifie pas. C'est écrit dans les limites.

---

## 7. TYING (robustesse) — MSB P5 §2.3, SCI P358

### 7.1 Principe commun aux trois modules

La force de chaînage `NEd` est une **situation accidentelle** : elle se
vérifie **SEULE** (sans `VEd`), avec les résistances **ULTIMES** et
`γMu = 1,1` (valeur du guide ; l'EN 1993-1-8 ne la fixe pas — à confirmer pour
la Belgique). C'est le mécanisme `nat_N ∈ {ELU fondamental ; Tying
(accidentel)}` déjà écrit dans `platine_about` : **il est repris tel quel, et
étendu aux deux autres modules** (qui ne l'avaient pas).

Ancre publiée disponible : le benchmark BM2/BM3 du module doubles cornières
reproduit les valeurs de tying du MSB P5 §4.4 (`Ft,Rd,u = 160 kN`,
`Σleff = 430 mm`, `Mpl = 4,2 kNm`, modes `696 / 1 190 / 1 920 kN`), et BM1 du
module platine d'about celles du §2.4 (`493 kN` pour la platine en flexion,
`1 513 kN` pour l'âme en traction). **Les formules des modes 1 / 2 / 3 sont
donc déjà validées sur l'exemple publié** ; ce qui s'ajoute ici n'est pas une
formule mais un **composant de plus**.

### 7.2 Les composantes, module par module

| composante | plat d'âme | cornières | platine d'about |
|---|---|---|---|
| boulons en **CISAILLEMENT** (`αv·fub·A/γMu`) | **OUI** — la force est dans le plan du plat | OUI pour le groupe S | non |
| boulons en **TRACTION** (`0,9·fub·As/γMu`) | non | OUI pour le groupe P | OUI |
| pièce d'attache : traction section nette (`0,9·Anet·fu/γMu`) | OUI | OUI (ailes B) | couvert par le tronçon en T |
| pièce d'attache : rupture de bloc en traction §3.10.2(2) | OUI | OUI (ailes B) | non (pas de bord libre en traction) |
| pièce d'attache : **tronçon en T** §6.2.4 Tab. 6.2 | non (pas de boulon tendu) | OUI (ailes A) — déjà écrit (`tsA`) | OUI — déjà écrit (`ptT`) |
| **aile du poteau : tronçon en T** (cas a) | **non** (voir 4.4) | **NOUVEAU** | **NOUVEAU** |
| **âme du poteau en traction transversale** §6.2.6.3 (cas a) | non | **NOUVEAU** | **NOUVEAU** |
| poinçonnement (Tab. 3.4) | non | **NOUVEAU** (aile / âme du poteau) | OUI — déjà écrit, `t` du poteau à ajouter |
| âme de la poutre portée : traction nette, rupture de bloc | OUI | OUI | OUI (`tw·hp·fu/γMu`) |
| cordons (`fu/(√3·βw·γMu)·a`) | OUI (`qx = N/(2·hp)`) | OUI si soudée | OUI — déjà écrit (`wT`) |
| **flexion hors plan de l'âme du poteau** (cas b) | **NON VÉRIFIÉE** (§5.4) | **NON VÉRIFIÉE** | **NON VÉRIFIÉE** |

### 7.3 Le tronçon en T de l'AILE DU POTEAU — la pièce normative neuve

C'est **le** composant que les trois grands frères n'ont pas, et il est
entièrement normatif.

```
m      = (p3 − twP)/2 − 0,8·rP           ← Figure 6.8, aile de poteau laminé
e      = (bP − p3)/2
n      = min( e ; 1,25·m )
ew     = dw / 4
Σleff  : EN 1993-1-8 Tableau 6.4 — AILE DE POTEAU NON RAIDIE.
         Aucune rangée d'extrémité : le poteau continue au-dessus et en
         dessous, TOUTES les rangées sont INTÉRIEURES.
         rangées isolées :  leff,cp = 2·π·m           leff,nc = 4·m + 1,25·e
         rangées en groupe : leff,cp = 2·p1           leff,nc = p1
         Σleff,1 = min[ min(Σcp ; Σnc) isolées ; min(Σcp ; Σnc) en groupe ]
         Σleff,2 = min( Σnc isolées ; Σnc en groupe )
Mpl,i  = 0,25 · Σleff,i · tfP² · f / γ        (f/γ = fu/γMu en tying, fy/γM0 à l'ELU)
FT,1   = (8·n − 2·ew)·Mpl,1 / [2·m·n − ew·(m + n)]        ← mode 1, MÉTHODE 2
FT,2   = (2·Mpl,2 + n·ΣFt,Rd) / (m + n)
FT,3   = ΣFt,Rd
FT,Rd  = min( FT,1 ; FT,2 ; FT,3 )
```

`FT,1` retombe sur la méthode 1 (`4·Mpl,1/m`) si le dénominateur dégénère —
même garde-fou que `platine_about.tronçon_en_T`.

**Contrôlé, platine d'about sur HEB 300 S355, `p3 = 140`, 4 rangées de
`p1 = 70`, M20 8.8, tying `γMu = 1,1`** :

| | platine (tp 10) | aile du poteau (tf 19) |
|---|---|---|
| `m` | 60,04 mm | **42,90 mm** |
| `n` | 40,00 mm | **53,625 mm** |
| `Σleff` | 290 mm (= `hp`) | **280 mm (= n1·p1)** |
| `Mpl,1` | 3,3614 kNm | **11,7161 kNm** |
| `FT,1` | **261,33 kN** | 1 296,99 kN |
| `FT,2` | 580,14 kN | **955,49 kN** |
| `FT,3` | 1 282,91 kN | 1 282,91 kN |
| `FT,Rd` | **261,33 kN** | **955,49 kN** |

→ **l'aile du poteau est 3,66 × plus forte que la platine** : avec une aile
laminée épaisse, le composant neuf ne gouverne pas. Il gouverne quand l'aile
est mince, le `p3` grand ou la platine épaisse — et c'est une composante de
l'Eurocode : **elle doit être là.**

Idem pour les cornières sur HEB 300, `p3 = 108,6`, 4 rangées :

| | ailes A des cornières (tc 10) | aile du poteau (tf 19) |
|---|---|---|
| `m` | 30,40 mm | **27,20 mm** |
| `n` | 38,00 mm | **34,00 mm** |
| `Σleff` | 290 mm | **280 mm** |
| `FT,1 / FT,2 / FT,3` | 572,02 / 811,01 / 1 282,91 | 2 314,01 / **1 095,61** / 1 282,91 |
| `FT,Rd` | **572,02 kN** | **1 095,61 kN** |

→ **les cornières gouvernent** (rapport 1,915).

### 7.4 Âme du poteau en traction transversale (cas a)

Pour une attache BOULONNÉE sur l'aile, la force de chaînage passe de l'aile à
l'âme. Composante de l'EN 1993-1-8 **§6.2.6.3** :

```
Ft,wc,Rd = ω · beff,t,wc · twP · f / γ        avec  beff,t,wc = Σleff de l'aile
```

Le coefficient `ω` réduit la résistance en fonction du **cisaillement du
panneau d'âme** ; une attache simple n'apporte pas de moment d'étage, donc
**`ω = 1`** dans le périmètre de l'outil (§1.3). Contrôlé (HEB 300,
`Σleff = 280`, tying) : **`Ft,wc,Rd = 1 428,0 kN`** — jamais déterminant sur
le cas de référence, mais la ligne existe.

**Nature : `EC + INT`.** La formule est celle de l'Eurocode ; l'INTERPRÉTATION
est de l'appliquer à une attache SIMPLE en tying avec `ω = 1` et
`beff = Σleff` de l'aile. Les guides ne listent pas explicitement ce
composant pour les attaches simples : **point tranché sans source franche**
(§9, P-5).

### 7.5 Ce qui S'AJOUTE au tying du module `platine_about`

Pour répondre précisément : le module existant a déjà le tronçon en T de la
platine (`ptT`), le poinçonnement (`pp`), l'âme portée en traction (`tnS`), le
cordon (`wT`) et la bascule `nat_N` / `γMu`. **S'ajoutent, et seulement :**

1. le **tronçon en T de l'aile du poteau** (`tfP`), cas (a) ;
2. l'**âme du poteau en traction transversale** (`twtP`), cas (a) ;
3. le poinçonnement prend `t = tfP` (cas a) ou `t = twP` (cas b) au lieu de
   `tw` de la poutre porteuse — une ligne de `min()`, pas une clé ;
4. en cas (b), la **limite écrite** sur la flexion hors plan de l'âme (§5.4) ;
5. la bascule `nat_N` / `γMu` est **portée** aux deux autres modules (ils ne
   l'avaient pas), avec, pour le plat d'âme, un composant de plus : les
   **boulons en cisaillement** sous la force de chaînage (`bvT`), parce que
   dans un plat d'âme la force de chaînage est **dans le plan du plat**.

---

## 8. Les trois modules — liste ORDONNÉE des vérifications

**Convention des colonnes** : `clé` = clé courte du moteur (`R.ck[clé]`) ;
`Ed`, `Rd` = ce qui est comparé ; `réf.` = référence du champ `ref` ;
`nat` ∈ {`EC`, `EC + COMP`, `COMP`, `EC + INT`, `INT`} ; `(a)`/`(b)` = cas
d'appui où la ligne est active ; `★` = vérification **essentielle**
(`ess = 1` : imprimée dans la note courte et l'export texte court).

Groupes d'affichage de la famille, dans cet ordre :
**Boulons · Pièce d'attache · Poutre portée · Poteau porteur · Soudure.**

### 8.1 `plat_ame_poteau` (fpc_) — plat d'âme soudé sur poteau

| # | clé | libellé | Ed | Rd | réf. | nat | cas | ★ |
|---|---|---|---|---|---|---|---|---|
| **Boulons** ||||||||
| 1 | `bv` | Cisaillement des boulons (1 plan), boulon le plus sollicité | `F,Ed = √(Fz² + Fx²)`, `Fz = VEd/n + MS·xmax/Ip`, `Fx = \|NEd,ELU\|/n + MS·ymax/Ip` | `Fv,Rd·βLf` | Tab. 3.4 ; groupe excentré : MSB P5 §3.2 | EC + COMP | a, b | ★ |
| 2 | `gl` | Glissement (1 surface de frottement) | `F,Ed·kser` | `ks·μ·Fp,C/γM3` | §3.9.1 | EC | a, b | |
| 3 | `bvT` | **Tying — boulons en cisaillement (force de chaînage seule)** | `NEd/n` | `αv·fub·A/γMu` | Tab. 3.4 avec γMu ; MSB P5 §2.3 | EC + COMP | a, b | ★ |
| **Plat d'âme** ||||||||
| 4 | `pdL` | Pression diamétrale – plat | `F,Ed` | interaction `√[(Fz/Fb,ver)² + (Fx/Fb,hor)²]` | Tab. 3.4 ; interaction : MSB P5 §3.2 | EC + COMP | a, b | ★ |
| 5 | `pcg` | Cisaillement section brute | `VEd` | `hp·tp·fy/(1,27·√3·γM0)` | MSB P5 §3.2 (d'après ECCS n°126) | COMP | a, b | ★ |
| 6 | `pcn` | Cisaillement section nette | `VEd` | `(hp − n1·d0)·tp·fu/(√3·γM2)` | MSB P5 §3.2 | COMP | a, b | ★ |
| 7 | `pcb` | Rupture de bloc | `VEd` | `0,5·fu·Ant/γM2 + fy·Anv/(√3·γM0)` | §3.10.2(3) ; aires : MSB P5 §3.2 | EC | a, b | ★ |
| 8 | `pfl` | Flexion dans son plan (sans objet si `hp ≥ 2,73·z`) | `MS` | `Wel·fy/γM0` | MSB P5 §3.2.3 ; EC3-1-1 §6.2.5 | COMP | a, b | ★ |
| 9 | `plt` | **Déversement du plat LONG** (`z > tp/0,15`) | `MS` | `Wel·pb(λLT)/γM1` | SCI P358 ; courbe BS 5950-1 Ann. B.2.1 ; γM1 ANB | COMP | a, b | ★ |
| 10 | `ptn` | Traction section nette (ELU ou tying) | `NEd` | `0,9·Anet·fu/γ` (`γM2` ou `γMu`) | EC3-1-1 §6.2.3(2) | EC | a, b | |
| 11 | `pbt` | **Rupture de bloc en traction** | `NEd` | `fu·Ant/γ + fy·Anv/(√3·γM0)` | §3.10.2(2) ; aires : MSB P5 §4.3.1.4 | EC | a, b | |
| **Poutre portée** ||||||||
| 12 | `pdS` | Pression diamétrale – âme de la poutre portée | `F,Ed` | interaction | Tab. 3.4 ; interaction : MSB P5 §3.2 | EC + COMP | a, b | ★ |
| 13 | `vgS` | Âme – cisaillement brut (section réduite si grugée) | `VEd` | `Av·fy/(√3·γM0)` | EC3-1-1 §6.2.6 ; MSB P5 §3.2.5.1 | EC | a, b | ★ |
| 14 | `vnS` | Âme – cisaillement net au droit des boulons | `VEd` | `(Av − n1·d0·tw)·fu/(√3·γM2)` | MSB P5 §3.2.5.1 (ECCS n°126) | COMP | a, b | ★ |
| 15 | `vbS` | Âme – rupture de bloc | `VEd` | `0,5·fu·Ant/γM2 + fy·Anv/(√3·γM0)` | §3.10.2(3) ; aires : MSB P5 §3.2.5.1 | EC | a, b | ★ |
| 16 | `mN` | Poutre grugée – flexion + cisaillement à l'extrémité du grugeage | `VEd·(gh + c) + \|MEd\|` | `Mv,N,Rd` | MSB P5 §4.2.4 | COMP | grugée | ★ |
| 17 | `m2` | Poutre grugée – interaction à la 2ᵉ file (`n2 = 2`) | `VEd·(gh + e2,b + p2)` | `Mc,2,Rd` | MSB P5 §3.2.5 | COMP | grugée | |
| 18 | `stN` | Stabilité locale du grugeage – longueur `c` | `c` | `c,max` | MSB P5 §4.2.5 | COMP | grugée | ★ |
| 19 | `stD` | Stabilité locale du grugeage – profondeur `dc` | `max(dc,sup ; dc,inf)` | `h/2` ou `h/5` | MSB P5 §4.2.5 | COMP | grugée | |
| 20 | `tnS` | Âme – traction section nette (ELU ou tying) | `NEd` | `0,9·Anet·fu/γ` | EC3-1-1 §6.2.3(2) ; hauteur participante = `hp` : hypothèse de l'outil | INT | a, b | |
| 21 | `tbS` | Âme – rupture de bloc en traction | `NEd` | `fu·Ant/γ + fy·Anv/(√3·γM0)` | §3.10.2(2) ; aires : MSB P5 §4.3.2.3 | EC | a, b | |
| **Poteau porteur** ||||||||
| 22 | `vlP` | **Âme du poteau – cisaillement local sur la hauteur du plat (un seul côté chargé)** | `VEd/2` | `twP·hp·fy,P/(√3·γM0)` | SCI P358 (élément support chargé d'un seul côté) | COMP | **b** | ★ |
| | | *cas (a) : groupe VIDE — voir §4.4, écrit dans « Périmètre de la vérification »* | | | | | a | |
| **Soudure** ||||||||
| 23 | `w` | Cordons plat / poteau – effort par unité de longueur au point critique | `√(qz² + qx²)`, `qz = VEd/(2hp)`, `qx = 3·MS/hp² + \|NEd,ELU\|/(2hp)` | `fvw,d·a` | §4.5.3.3 (méthode simplifiée) | EC | a, b | ★ |
| 24 | `wT` | **Cordons – tying (force de chaînage seule)** | `qx = NEd/(2hp)` | `fu/(√3·βw·γMu)·a` | §4.5.3.3 avec γMu ; MSB P5 §2.3 | EC + INT | a, b | |
| 25 | `wa` | Gorge minimale | `3 mm` | `a` | §4.5.2(2) | EC | a, b | |

**Alertes** (`block` = bloquante) : reprises — `e1bot`, `e2p`, `zc_top`,
`h_dispo`, `one_bolt`, `d0_manq`, `tp_duct`, `hp_rec`, `long_nr`, `a_rec`,
`MEd`, `NEd`, `cat_cl` ; **retirées** — `pl_web`, `dnt_min`, `dnb_min`,
`ln_min`, `gap_neg` ; **neuves** —
`aile_poutre` (bloquante, cas b : `bS` ne passe pas entre les ailes et la
recoupe est insuffisante),
`b1_long` (information, cas b variante b1 : `z` triplé, flexion hors plan de
l'âme du poteau non vérifiée, préférer b2),
`ame_hp` (information, cas b : flexion hors plan de l'âme du poteau sous
`qx` et sous tying non vérifiée).

### 8.2 `doubles_cornieres_poteau` (dcc_) — doubles cornières sur poteau

| # | clé | libellé | Ed | Rd | réf. | nat | cas | ★ |
|---|---|---|---|---|---|---|---|---|
| **Boulons** ||||||||
| 1 | `bv_S` | Cisaillement – groupe S (2 plans), boulon le plus sollicité | `F_S` | `2·Fv,Rd·βLf` | Tab. 3.4 ; groupe excentré : MSB P5 §4.2.1.1 | EC + COMP | a, b | ★ |
| 2 | `bv_P` | Cisaillement – groupe P sur le poteau (1 plan par cornière) | `F_P = (VEd/2)/nP` | `k_rot·Fv,Rd·βLf` | Tab. 3.4 ; facteur 0,80 : MSB P5 §4.2.1.2 | EC + COMP | a, b | ★ |
| 3 | `bt_P` | Traction des boulons – groupe P (ELU ou tying) | `Ft,Ed` | `min(Ft,Rd ; Bp,Rd)` | Tab. 3.4 | EC | a, b | |
| 4 | `bi_P` | Interaction cisaillement + traction – groupe P | `Fv,Ed/Fv,Rd + Ft,Ed/(1,4·Ft,Rd)` | `1` | Tab. 3.4 | EC | a, b | |
| 5 | `gl_S` | Glissement – groupe S (2 surfaces) | `F_S·kser` | `ks·2·μ·Fp,C/γM3` | §3.9.1 | EC | a, b | |
| 6 | `gl_P` | Glissement – groupe P (1 surface) | `F_P·kser` | `ks·μ·(Fp,C − 0,8·Ft,Ed)/γM3` | §3.9.1 et §3.9.2 | EC | a, b | |
| **Cornières** ||||||||
| 7 | `pdB` | Pression diamétrale – ailes B (groupe S) | `F_S/2` | interaction | Tab. 3.4 ; interaction : MSB P5 §4.2.1.1 | EC + COMP | a, b | ★ |
| 8 | `pdA` | Pression diamétrale – ailes A (groupe P) | `F_P` | interaction | Tab. 3.4 | EC + COMP | a, b | ★ |
| 9 | `cgB` | Ailes B – cisaillement brut | `VEd` | `2·hc·tc·fy/(1,27·√3·γM0)` | MSB P5 §4.2.2 (d'après ECCS n°126) | COMP | a, b | ★ |
| 10 | `cnB` | Ailes B – cisaillement net | `VEd` | `2·tc·(hc − n1S·d0)·fu/(√3·γM2)` | MSB P5 §4.2.2 | COMP | a, b | ★ |
| 11 | `cbB` | Ailes B – rupture de bloc | `VEd` | `2·[0,5·fu·Ant/γM2 + fy·Anv/(√3·γM0)]` | §3.10.2(3) ; aires : MSB P5 §4.2.2.1 | EC | a, b | ★ |
| 12 | `flB` | Ailes B – flexion dans leur plan | `(VEd/2)·z + \|MEd\|/2` | `Wel·fy/γM0` | MSB P5 §3.2.3 ; EC3-1-1 §6.2.5 | COMP | a, b | ★ |
| 13 | `cgA` | Ailes A – cisaillement brut | `VEd` | idem `cgB` | MSB P5 §4.2.2.2 | COMP | a, b | |
| 14 | `cnA` | Ailes A – cisaillement net | `VEd` | `2·tc·(hc − n1P·d0)·fu/(√3·γM2)` | MSB P5 §4.2.2.2 | COMP | a, b | |
| 15 | `cbA` | Ailes A – rupture de bloc | `VEd` | `2·[0,5·fu·Ant/γM2 + fy·Anv/(√3·γM0)]` | §3.10.2(3) ; aires : MSB P5 §4.2.2.2 | EC | a, b | |
| 16 | `flA` | Ailes A – flexion entre talon et file de boulons | `(VEd/2)·(gA − tc/2)` | `Wel·fy/γM0` | EC3-1-1 §6.2.5 ; bras de levier : hypothèse de l'outil | INT | a, b | |
| 17 | `tsA` | Ailes A en flexion + boulons tendus – tronçon en T (ELU ou tying) | `NEd` | `min(FT,1 ; FT,2 ; FT,3)` | §6.2.4 Tab. 6.2 (méthode 2) ; `leff` et `m` : MSB P5 §4.3.1.1 | EC + COMP | a, b | ★ |
| 18 | `tnB` | Ailes B – traction section nette | `NEd` | `2·0,9·Anet·fu/γ` | EC3-1-1 §6.2.3(2) | EC | a, b | |
| 19 | `tbB` | Ailes B – rupture de bloc en traction | `NEd` | `min(cas 1 ; cas 2)` | §3.10.2(2) ; aires : MSB P5 §4.3.1.4 | EC | a, b | |
| **Poutre portée** ||||||||
| 20–30 | `pdS`, `vgS`, `vnS`, `vbS`, `mN`, `m2`, `stN`, `stD`, `tnS`, `tbS` | *identiques au §8.1, lignes 12 à 21* | | | | | | |
| **Poteau porteur** ||||||||
| 31 | `pdP` | **Pression diamétrale – aile (a) ou âme (b) du poteau** | `F_P` | `k1·αb·fu,P·d·t/γM2`, `t = tfP` (a) ou `twP` (b), `αb` sans pince, `k1` du gauge | Tab. 3.4 ; SCI P358 (élément porteur) | EC | a, b | ★ |
| 32 | `vlP` | **Âme du poteau – cisaillement local brut (un seul côté chargé)** | `VEd/2` | `Av·fy,P/(√3·γM0)`, `Av = twP·(5d + (n1P−1)·p1 + min(p3/2 ; 5d))` | SCI P358 | COMP | **b** | ★ |
| 33 | `vnP` | Âme du poteau – cisaillement local net | `VEd/2` | `(Av − n1P·d0·twP)·fu,P/(√3·γM2)` | SCI P358 | COMP | **b** | |
| 34 | `tfP` | **Aile du poteau en flexion – tronçon en T (tying ; ELU si NEd > 0)** | `NEd` | `min(FT,1 ; FT,2 ; FT,3)` | §6.2.4 Tab. 6.2 (méthode 2) ; `Σleff` : **Tab. 6.4 (aile de poteau non raidie)** ; MSB P5 §2.3 | EC + COMP | **a** | ★ |
| 35 | `twtP` | **Âme du poteau en traction transversale** | `NEd` | `ω·beff,t,wc·twP·f/γ`, `ω = 1`, `beff = Σleff` de l'aile | §6.2.6.3 ; `ω = 1` et `beff` : interprétation de l'outil | EC + INT | **a** | |
| 36 | `ppP` | **Poinçonnement de l'aile / de l'âme du poteau** | `Ft,Ed` | `0,6·π·dm·t·fu,P/γ` | Tab. 3.4 | EC | a, b | |
| **Soudures** (ailes A ou B soudées) ||||||||
| 37–42 | `wS`, `waS`, `wlS`, `wP`, `waP`, `wlP` | *identiques au module existant* ; `wP` sur l'aile/l'âme du poteau avec `fu = min(fu cornière ; fu poteau)` | | | §4.5.3.3 ; §4.5.2(2) ; §4.5.1(2) | EC + COMP | a, b | ★ |

**Alertes neuves** : `p3_congeP` (bloquante, cas a : `p3 < twP + 2·rP + dw` —
l'écrou chevauche le congé du poteau) ; `bA_congeP` (bloquante, cas b :
`tw,S/2 + bA > (hP − 2·tfP)/2 − rP`) ; `p3_libreP` (bloquante, cas b :
`p3/2 + dw/2 > (hP − 2·tfP)/2 − rP`) ; `aile_poutre` ; `ame_hp`.
**Alerte retirée** : `P_web` (boulons P hors de la partie droite de l'âme
porteuse) — un poteau n'a pas de partie droite limitée.

### 8.3 `platine_about_poteau` (pac_) — platine d'about partielle sur poteau

| # | clé | libellé | Ed | Rd | réf. | nat | cas | ★ |
|---|---|---|---|---|---|---|---|---|
| **Boulons** ||||||||
| 1 | `bg` | Groupe de boulons – cisaillement et pression diamétrale sur la platine | `VEd` | 3 branches : `ΣFb,Rd` / `ns·(Fb,Rd)min` / `0,8·ns·Fv,Rd` | Tab. 3.4 ; MSB P5 §2.2.4 sur §3.7 | EC + COMP | a, b | ★ |
| 2 | `gl` | Glissement (1 surface) | `Fv,Ed·kser` | `ks·μ·(Fp,C − 0,8·Ft,Ed)/γM3` | §3.9.1 ; §3.9.2 | EC | a, b | |
| 3 | `bvt` | Cisaillement + traction ELU (interaction) | `Fv,Ed/Fv,Rd + Ft,Ed/(1,4·Ft,Rd)` | `1` | Tab. 3.4 | EC | a, b | |
| **Platine d'about** ||||||||
| 4 | `pcg` | Cisaillement brut (2 plans) | `VEd` | `2·(hp·tp/1,27)·fy/(√3·γM0)` | MSB P5 §2.2 | COMP | a, b | ★ |
| 5 | `pcn` | Cisaillement net (2 plans) | `VEd` | `2·tp·(hp − n1·d0)·fu/(√3·γM2)` | MSB P5 §2.2 | COMP | a, b | ★ |
| 6 | `pcb` | Rupture de bloc (2 blocs) | `VEd` | `2·[k·fu·Ant/γM2 + fy·Anv/(√3·γM0)]`, `k = 1` si `hp ≥ 1,36·p3` | §3.10.2 ; `k` et aires : MSB P5 §2.2 | EC + COMP | a, b | ★ |
| 7 | `ptT` | Flexion de la platine en traction – tronçon en T (ELU ou tying) | `NEd` | `min(FT,1 ; FT,2 ; FT,3)` | §6.2.4 Tab. 6.2 (méthode 2) ; MSB P5 §2.3 | EC + COMP | a, b | ★ |
| 8 | `pp` | Poinçonnement sous la tête ou l'écrou | `Ft,Ed` | `0,6·π·dm·min(tp·fu ; t_poteau·fu,P)/γ` | Tab. 3.4 | EC | a, b | |
| **Poutre portée** ||||||||
| 9 | `vwS` | Âme – cisaillement au droit de la platine (`Av = hp·tw`) | `VEd` | `hp·tw·fy/(√3·γM0)` | EC3-1-1 §6.2.6(1) ; `Av` : MSB P5 §2.2.1 | EC + COMP | a, b | ★ |
| 10 | `vgS` | Grugeage – cisaillement de la section réduite | `VEd` | `Av,N·fy/(√3·γM0)` | EC3-1-1 §6.2.6 | EC | grugée | |
| 11 | `mN` | Poutre grugée – flexion + cisaillement à l'extrémité du grugeage | `VEd·(tp + c)` | `Mv,N,Rd` | MSB P5 §4.2.4 | COMP | grugée | ★ |
| 12 | `stN` | Stabilité locale du grugeage – longueur `c` | `c` | `c,max` | MSB P5 §4.2.5 | COMP | grugée | ★ |
| 13 | `stD` | Stabilité locale du grugeage – profondeur `dc` | `max(dc,sup ; dc,inf)` | `h/2` ou `h/5` | MSB P5 §4.2.5 | COMP | grugée | |
| 14 | `tnS` | Âme – traction au droit de la platine (ELU ou tying) | `NEd` | `tw·hp·fy/γM0` ou `tw·hp·fu/γMu` | EC3-1-1 §6.2.3 ; MSB P5 §2.3 | EC + COMP | a, b | |
| **Poteau porteur** ||||||||
| 15 | `pdP` | **Pression diamétrale – aile (a) ou âme (b) du poteau** | `VEd` | `ns·Fb,Rd`, `t = tfP` (a) ou `twP` (b) | Tab. 3.4 ; SCI P358 (élément porteur) | EC | a, b | ★ |
| 16 | `vlP` | **Âme du poteau – cisaillement local brut** | `VEd/2` | `twP·hp·fy,P/(√3·γM0)` | SCI P358 | COMP | **b** | ★ |
| 17 | `vnP` | Âme du poteau – cisaillement local net | `VEd/2` | `twP·(hp − n1·d0)·fu,P/(√3·γM2)` | SCI P358 | COMP | **b** | |
| 18 | `tfP` | **Aile du poteau en flexion – tronçon en T** | `NEd` | `min(FT,1 ; FT,2 ; FT,3)` | §6.2.4 Tab. 6.2 ; `Σleff` : Tab. 6.4 ; MSB P5 §2.3 | EC + COMP | **a** | ★ |
| 19 | `twtP` | **Âme du poteau en traction transversale** | `NEd` | `ω·beff·twP·f/γ`, `ω = 1` | §6.2.6.3 ; `ω = 1` et `beff` : interprétation de l'outil | EC + INT | **a** | |
| **Soudure** ||||||||
| 20 | `w` | Cordons âme portée / platine – effort par unité de longueur | `√(qz² + qx²)` | `fvw,d·a` | §4.5.3.3 | EC | a, b | ★ |
| 21 | `wT` | Cordons – tying | `qx = NEd/(2hp)` | `fu/(√3·βw·γMu)·a` | §4.5.3.3 avec γMu ; MSB P5 §2.3 | EC + INT | a, b | |
| 22 | `wa` | Gorge minimale | `3 mm` | `a` | §4.5.2(2) | EC | a, b | |

`wa` reste dans `DETAIL_SEUL` : un minimum de détail n'est jamais
« dimensionnant » tant qu'il passe (mécanisme déjà écrit).

### 8.4 Limites de validité à écrire dans l'écran (« non revérifié ici »)

Texte exact du tuple `NON_REVERIFIES` de `ecran_resultats.py`, commun aux
trois modules (le dernier élément ne vaut que pour le cas (b)) :

1. « panneau d'âme du poteau en cisaillement sous les moments d'étage
   (§6.2.6.1) — une attache simple n'y contribue pas, il se vérifie avec
   l'ensemble des attaches de l'étage » ;
2. « poteau sous charges combinées : résistance de section, flambement,
   déversement, et le moment `VEd·e` que cette attache lui apporte
   (EC3-1-1 §6.2 et §6.3) » ;
3. « torsion du poteau induite par une attache d'un seul côté » ;
4. « attaches des DEUX côtés du poteau : efforts opposés, boulons ou âme
   partagés — pression diamétrale, cisaillement local et tying non traités » ;
5. « capacité de ROTATION au-delà des règles de détail : l'articulation est
   supposée (§5.2), le contact de la semelle inférieure de la poutre sur le
   poteau n'est pas vérifié » ;
6. « cisaillement global, flexion et déversement de la poutre portée » ;
7. « flexion HORS PLAN de l'âme du poteau sous `qx = 3·MS/hp²` et sous la
   force de chaînage — non couverte par l'EN 1993-1-8 pour une âme non
   raidie » *(cas (b) seulement)* ;
8. « poteau raidi (goussets, raidisseurs entre ailes) : les longueurs
   efficaces du Tab. 6.4 et les limites de l'âme supposent le poteau NON
   raidi » ;
9. « ELS : flèches, vibrations, confort ».

---

## 9. Points TRANCHÉS SANS SOURCE FRANCHE — à relire par le bureau

Ils sont listés ici **et** doivent l'être dans l'onglet Méthode et dans
`ECARTS_ET_CORRECTIONS.md`.

| N° | Point | Ce que l'étude a tranché | Pourquoi c'est fragile |
|---|---|---|---|
| **P-1** | **Numéro du tableau des longueurs efficaces** | `Σleff` de l'aile de poteau non raidie : **EN 1993-1-8 Tableau 6.4**, avec les formules `2πm`, `4m + 1,25e`, `2p`, `p`. | La consigne de l'étude citait le **Tableau 6.6**. Le texte NBN est payant et n'a pas été consulté. Ce qui est sûr : ce sont bien les formules de l'**aile de poteau non raidie**, et ce sont **celles que le module `platine_about` applique déjà** (en les déclarant « Tab. 6.4 appliqué à la platine (interprétation) ») — et son benchmark reproduit l'exemple publié §2.4 à 0,23 %. **C'est le NUMÉRO, pas la formule, qui est en cause.** Les deux numéros sont à recontrôler sur le texte. Remarque utile : ce qui était une *interprétation* pour une platine devient une **application directe** pour une aile de poteau. |
| **P-2** | **Pas de vérification locale de l'aile sous VEd seul** | Groupe « Poteau porteur » VIDE pour un plat d'âme soudé en cas (a). | Aucun modèle publié n'a été trouvé pour une aile de poteau chargée par un plat soudé sur chant, au droit de l'âme. Le raisonnement (§4.4) est mécaniquement net mais c'est un raisonnement, pas une clause. L'alternative serait d'inventer un `Av = f(tfP, hp)` : refusé. |
| **P-3** | **La recoupe en LARGEUR des semelles n'est pas un grugeage** | Les règles MSB P5 §4.2.4 / §4.2.5 ne sont PAS activées par la recoupe. | Ces règles portent sur une entaille en hauteur et sur le Té résiduel ; rien n'est publié pour une réduction de largeur d'aile à l'about. Le raisonnement : l'âme seule transmet `VEd` (déjà vérifié) et la recoupe est au voisinage de la rotule. |
| **P-4** | **Aucune épaisseur minimale d'aile de poteau** | L'outil n'en pose pas. | Ni l'EN 1993-1-8 ni les guides n'en donnent pour une attache simple ; ce sont la pression diamétrale et le tronçon en T qui décident. Un bureau pourrait vouloir `tfP ≥ tp` pour un cordon de pleine résistance : à trancher. |
| **P-5** | **Âme du poteau en traction transversale avec ω = 1** | `ω = 1` et `beff,t,wc = Σleff` de l'aile, en tying, nature `EC + INT`. | `ω` dépend du cisaillement du panneau d'âme, hors périmètre ici ; `ω = 1` est la valeur en l'absence de moment d'étage, donc la plus favorable. Les guides ne listent pas explicitement ce composant pour les attaches simples. **Si le bureau veut être conservatif, il impose `ω` en saisie.** |
| **P-6** | **Flexion hors plan de l'âme du poteau en tying (cas b)** | NON vérifiée, écrite en limite. | Le tronçon en T suppose une aile encastrée sur une âme centrale ; une âme entre deux ailes ne s'y ramène pas. Choix : déclarer, plutôt que transposer un modèle qui ne tient pas. |
| **P-7** | **Gauge minimal `p3 ≥ twP + 2·rP + dw`** | Alerte bloquante, nature INT. | Règle géométrique de dégagement du congé (l'écrou est côté intérieur de l'aile), pas une clause normative. Elle explique les gauges usuels (90 / 100 / 120 / 140) mais ne les remplace pas ; SCI P358 tabule des gauges minimaux par série, non consultés. Un bureau peut préférer une table. |
| **P-8** | **`et = 5·d` pour le cisaillement local de l'âme du poteau** | Les bornes `zt`, `zb` du grand frère disparaissent, `et = 5·d`, `eb = min(p3/2 ; 5d)`. | Le plafond `5·d` vient de SCI P358 (cité de mémoire dans le dépôt depuis le 21/09/2026). Supprimer les bornes `zt`/`zb` est correct (le poteau continue) mais rend la règle entièrement dépendante du `5·d`. |
| **P-9** | **`γMu = 1,1`** | Valeur du guide (MSB P5 §2.3, SCI P358), modifiable en saisie. | L'EN 1993-1-8 ne la fixe pas ; l'ANB belge n'a pas été consultée. Déjà signalé dans le module `platine_about`. |

---

## 10. ARCHITECTURE — décision et justification

### 10.1 Copie adaptée, ou import du grand frère ?

**Décision : COPIE ADAPTÉE du moteur, et IMPORT des briques PURES** mises en
commun dans un fichier NEUF.

**Pourquoi pas l'import du `compute` du grand frère.** Les trois moteurs
existants sont des `compute(u0)` monolithiques (597, 677 et 664 lignes) qui
construisent `R`, les alertes, le Tableau 3.3 et les vérifications **en une
passe**, à partir de LEUR `defaults()`. Il n'y a aucun point de composition.
« Importer et étendre » voudrait dire : appeler `compute(u_traduit)`, puis
**retirer** des alertes (`pl_web`, `dnt_min`, `ln_min`, `gap_neg`, `P_web` —
toutes fausses pour un poteau), **retirer** des vérifications (`vlP` en cas
(a)), **ajouter** les composantes du poteau, et **réécrire** `R.u` (dont les
clés ne sont pas celles du nouveau module). Un post-traitement de ce genre :

- casse `acier/projet.py:description`, le dessin, la note et l'écran, qui
  lisent `R.u` ;
- fait de toute évolution du grand frère une évolution silencieuse des trois
  neufs — et réciproquement crée la tentation d'ouvrir un « hook » dans un
  moteur dont les 31 + 12 + 13 cas figés sont opposables.

**Pourquoi pas la copie pure et simple non plus.** Recopier trois fois des
formules déjà écrites crée des secondes vérités. On les met donc en commun.

**La règle appliquée** : *une information a une source unique ; quand elle ne
peut pas en avoir, les copies sont NOMMÉES et un test les lie.*

### 10.2 Le fichier neuf partagé : `acier/assemblages/mecanique.py`

Sans Streamlit, pur, bibliothèque standard seulement. Il reçoit les briques
**déjà écrites deux ou trois fois** dans les moteurs existants :

| fonction | ce qu'elle porte | aujourd'hui recopiée dans |
|---|---|---|
| `bearing(t, fu, fub, d, d0, e1, p1, n1, e2, p2, n2, gM2, ktrou, exV, exH)` | les deux `Fb,Rd` directionnels + `αb`, `k1`, et l'interaction quadratique | les 3 moteurs |
| `groupe_excentre(n1, n2, p1, p2, V, N, M)` | `Ip`, `Fz`, `Fx`, `F` | `plat_ame`, `doubles_cornieres` |
| `bolt_table(...)` | efforts par boulon | `plat_ame`, `doubles_cornieres` |
| `section_grugee(profil, fy, dnt, dnb, gM0)` | `cas_g`, `A`, `h_T`, `A_Tee`, `Av`, `Vpl`, `zb`, `I_N`, `W_N`, `ρ`, `el_N`, `lim_N`, `ln_max`, `val_N` | **les 3 moteurs, à l'identique** (le plus gros bloc dupliqué du dépôt) |
| `pb_bs5950(lam, py)`, `lam_LT(z, hp, tp)` | déversement du plat long | `plat_ame` |
| `tp_ductile(d, fub, fy)`, `coef_pleine_resistance(...)`, `groupe_boulons(...)` | règles de la platine | `platine_about` |
| `troncon_T(leff1, leff2, t, m, n, ew, f, gamma, SFt)` | modes 1 (méthode 2), 2, 3 — **génériques**, sans géométrie en dur | `platine_about` (lié à la platine), `doubles_cornieres` (lié aux cornières) |
| `leff_aile_poteau(n1, p1, m, e)` | **NOUVEAU** : Tab. 6.4, rangées intérieures | — |
| `leff_platine_bords_libres(n1, p1, e1h, e1b, m, e)` | `Σleff` d'une platine à bords libres | `platine_about` |

**Les trois modules EXISTANTS ne sont PAS touchés.** Ils gardent leurs copies
locales. Le lien entre les copies est un **test neuf**,
`tests/test_assemblages_mecanique.py`, qui, pour chacun des **56 cas figés**
(`ref_plat_ame.json` 12, `ref_platine_about.json` 13, les 31 cas de parité des
cornières), recalcule le bloc par `mecanique` et compare au moteur existant à
**1e-12** :

```
pour chaque cas figé :
    R = moteur_existant.compute(cas)
    assert mecanique.section_grugee(profil(R), R.fy_S, dnt, dnb) == (R.cas_g, R.Asec_S, R.h_T,
           R.A_Tee, R.Av_S, R.VRd_gS, R.W_N)        # à 1e-12
    assert mecanique.pb_bs5950(R.lam_LT, R.fy_L)    == R.pb_LT
    assert mecanique.troncon_T(...)                 == (R.FT1, R.FT2, R.FT3)
```

Ce test est la **preuve que la copie est une VUE** : si quelqu'un modifie une
formule d'un côté, il rougit et nomme le cas divergent. Il rougit aussi si on
le retire.

### 10.3 Arborescence des fichiers créés

```
acier/assemblages/
  mecanique.py                              ← NOUVEAU, partagé, pur
  poutre_poteau/
    __init__.py
    plat_ame_poteau/
      __init__.py
      entrees.py        ← defaults(), NUM, GROUPES, PILOTEES_PAR_PREDIM, UNITE, CHAMPS,
                          groupes_visibles, champ_visible, charger_json, APPUI_AILE/APPUI_AME,
                          NAT_ELU/NAT_TYING, VISSERIE_DEFAUT
      moteur.py         ← compute(), bolt_table(), predim(), apply_solution(), solution_label()
      schemas.py        ← elevation(), plan(), coupe_aa(), dessins(), cotes_modifiables(),
                          CLE_PAR_COTE, options_ecran/_rapport/_fabrication,
                          FABRICATION_CALC, EXCLURE_*
      formules.py       ← from ..poutre_poutre.plat_ame.formules import textes
      notation.py       ← ref_courte(), NOTATIONS, LEGENDE_REFERENCES, TITRES_ALERTES, ligne_alerte()
      synthese.py       ← TABLES, COURT, ELEMENTS, court(), taux_par_element(), lignes(), titre_table()
      texte.py          ← nom_fichier(), construire_texte()
      methode.py        ← METHODE (+ _notations)
      benchmark.py      ← SRC, BENCH, NOCOVER, VALID, run_bench()
      note.py           ← class Note(plat_ame.note.Note) ; class PlanPrincipePP(PlanPrincipe) ;
                          generer_note()
      ecran_saisie.py   ← PREFIXE, PAS, LIBELLE, AIDES, AFFICHAGE, GEOMETRIE_DESSIN, EN_TETE,
                          SEGMENTS, BLOCS, AVANCES, CARTOUCHE, DIMENSIONS, D, valeur_widget, carte
      ecran_resultats.py← ONGLETS, bandeau, alertes, dessins, panneau_cotes, onglet_*, onglets
      interface.py      ← MODULE_VERSION, PREFIXE, TITRE_COURT/LONG, CLES, compute, charger_payload,
                          show()
    doubles_cornieres_poteau/   ← mêmes 13 fichiers
    platine_about_poteau/       ← mêmes 13 fichiers
tests/
  test_assemblages_mecanique.py       ← NOUVEAU : lie mecanique.py aux 3 moteurs existants
  test_fpc_moteur.py / test_fpc_ecran.py
  test_dcc_moteur.py / test_dcc_ecran.py
  test_pac_moteur.py / test_pac_ecran.py
  ref_fpc.json / ref_dcc.json / ref_pac.json        ← régressions figées
docs/assemblages/
  POUTRE_POTEAU.md                     ← ce document
  PLAT_AME_POTEAU.md / DOUBLES_CORNIERES_POTEAU.md / PLATINE_ABOUT_POTEAU.md   ← en fin de lot
```

### 10.4 Fichiers EXISTANTS modifiés — la liste complète, et rien d'autre

| Fichier | Modification | Risque |
|---|---|---|
| `acier/registre.py` | +1 `Famille("poutre_poteau", "Poutre – Poteau", …)`, +3 `Assemblage`, +3 fonctions `vignette_*` | nul (déclaratif ; `modules/assemblages.py` lit le registre et n'est pas touché) |
| `tests/test_assemblages_kit.py` §10 | le dict `courts` passe de 4 à 7 entrées | **le harnais SUIT** : il décrivait l'ancien registre |
| `tests/test_fumee_pages.py` | +3 ids dans la boucle « Écran de chaque assemblage ouvert » | **le harnais SUIT** |

**Aucun autre fichier du dépôt n'est modifié.** En particulier :
`modules/assemblages.py`, `streamlit_app.py`, `acier/projet.py`,
`acier/projet_ui.py`, `acier/rapport_projet.py`, `acier/assemblages/carte.py`,
`ui_commun.py`, `note_kit.py`, `ndc_pdf/`, et **les trois modules
poutre–poutre et le pied de poteau**. Leurs suites de non-régression
(`test_plat_ame_moteur/ecran`, `test_platine_about_moteur/ecran`,
`test_assemblages_*`, `test_pied_poteau_*`, `test_platine_manuelle*`) doivent
rester **identiques à l'octet** et au vert : c'est le critère de recette de
chaque lot.

`acier/projet.py:description` fonctionne **sans changement** parce que les
clés `prof_S` et `prof_P` sont conservées : la ligne du sommaire et du
récapitulatif dira « IPE 400 sur HEB 300 · V = 200 kN ». `rapport_projet.py`
fonctionne sans changement parce que chaque module expose le contrat habituel
(`CLES`, `ecran_saisie.valeur_widget`, `compute`, `note.generer_note`,
`charger_payload`, `MODULE_VERSION`).

### 10.5 Import du kit de dessin

Les nouveaux `schemas.py` importent le kit comme `plat_ame` le fait déjà :

```python
from acier.assemblages.poutre_poutre.doubles_cornieres.schemas import (
    Options, Feuille, PALETTE, INF_MM, SEGMENTS_ARC,
    _rect, _poly, _path, _circle, _text, _group, _arc, _hachures,
    _rect_rompu, _rupture, _renvoi, _section_I, _section_portee)
```

(import **absolu**, pas `...poutre_poutre`, pour rester lisible depuis une
autre famille). **Chantier à ouvrir séparément, pas dans ces lots** : déplacer
le kit dans `acier/assemblages/dessin_kit.py`. Le faire ici obligerait à
toucher `doubles_cornieres/schemas.py`, `plat_ame/schemas.py`,
`platine_about/schemas.py`, `rapport.py` et `test_assemblages_schemas.py` —
exactement ce que la règle « leurs suites restent identiques » interdit.

### 10.6 Clés de session

**Préfixes retenus** — contrôlés libres (`grep -rn "fpc_\|dcc_\|pac_"
--include=*.py .` → aucun résultat) et disjoints de `asm_`, `fpl_`, `pab_`,
`ppb_`, `prj_`, `b…`, `dal…`, `pre…`, `rs_`, `ab_`, `chk_`, `gamma_`, `pdf_`,
`show_` :

| module | préfixe | `court` du registre (≤ 18) |
|---|---|---|
| `plat_ame_poteau` | **`fpc_`** | « Plat d'âme/poteau » (17) |
| `doubles_cornieres_poteau` | **`dcc_`** | « Cornières/poteau » (16) |
| `platine_about_poteau` | **`pac_`** | « Platine/poteau » (14) |

#### 10.6.1 Clés communes aux trois modules — reprises, retirées, ajoutées

**Reprises à l'identique du grand frère** (même nom, même sens, même défaut,
sauf indication) :

```
id_projet, id_rep, id_red, id_date, mode_calc
prof_P, hP_u, bP_u, twP_u, tfP_u, rP_u, nu_P        ← le POTEAU (libellé « Poteau »)
prof_S, hS_u, bS_u, twS_u, tfS_u, rS_u, nu_S        ← la poutre PORTÉE
d_nt, d_nb, l_n, r_n, lt_ok                          ← défauts 0, 0, 0, 10, « Oui »
g_h                                                  ← jeu : face d'appui → about de la poutre
z_C                                                  ← dessus de la poutre → dessus de la pièce
boulon_u, classe, trou, d0_u, filet, cat, mu_s, k_s, k_ser
V_Ed, N_Ed, M_Ed  (M_Ed : plat d'âme et cornières seulement)
g_M0, g_M2n, g_M2, g_M3, g_M3s   (+ g_M1 : plat d'âme seulement)
opt_blf, expo, fy_u, fu_u, bw_u
eta_c, k_e1, k_p1, k_e2, pd_dmin, pd_dmax
```

**RETIRÉE dans les trois** : `d_top` (« décalage du dessus de la secondaire
sous le dessus de la principale ») — un poteau n'a pas de dessus. La position
verticale de la poutre sur le poteau n'intervient dans aucune résistance ; le
dessin trace un **tronçon de poteau rompu** en haut et en bas.

**AJOUTÉES dans les trois** :

| clé | type | défaut | libellé (`LIBELLE`, ≤ 22) | rôle |
|---|---|---|---|---|
| `appui` | choix | `APPUI_AILE` | « Appui » | `APPUI_AILE = "Aile du poteau"` / `APPUI_AME = "Âme du poteau"`. **La clé centrale.** Affichage court : « Aile » / « Âme ». En-tête de carte, en `segmented_control` à côté du mode. |
| `nat_N` | choix | `NAT_ELU` | « Nature de NEd » | `NAT_ELU = "ELU fondamental"` / `NAT_TYING = "Tying (accidentel)"`. Visible dès `NEd > 0` (condition `traction`, déjà écrite dans `platine_about/entrees.py`). |
| `g_Mu` | nombre | `1.1` | « γMu (tying) » | visible si `nat_N == Tying` **et** `NEd > 0`. Avancés › COEFF. γ. |
| `rec_S` | nombre | `0` | « Recoupe des ailes » | longueur de recoupe des semelles de la poutre, depuis l'about (mm). 0 = aucune. Visible si `appui == APPUI_AME`. |
| `j_lat` | nombre | `5` | « Jeu latéral » | jeu entre la semelle de la poutre et l'aile du poteau (mm). Avancés › GÉOMÉTRIE. Visible si `appui == APPUI_AME`. |

#### 10.6.2 Clés propres à chaque module

**`fpc_` — plat d'âme sur poteau** : les 67 clés du grand frère `plat_ame`,
moins `d_top`, plus les 5 communes → **71 clés**.
Reprises en propre : `hp_u, tp_u, bp_u, nu_pl, a_w, n1_u, n2_u, p1_u, p2_u,
e1_u, e2b_u`.

**`dcc_` — doubles cornières sur poteau** : les 84 clés du grand frère
`doubles_cornieres`, moins `d_top`, plus les 5 communes → **88 clés**.
Reprises en propre : `fix_P, fix_S, corn_u, k1_u, k2_u, kt_u, kr_u, orient,
LC_u, nu_C, n1S_u, n2S_u, p1S_u, p2_S, e1S_u, e2b_u, n1P_u, n2P_u, p1P_u,
p2_P, e1P_u, gA_u, a_P, lh_P, a_S, lh_S, H_Ed, opt_exc, k_rot`.
`fix_P` garde son sens : les ailes A **boulonnées** (défaut) ou **soudées**
sur l'aile / l'âme du poteau. `gA_u` est le trusquinage sur le poteau, d'où
`p3 = tw_S + 2·gA`.

**`pac_` — platine d'about sur poteau** : les **64** clés du grand frère
`platine_about`, moins `d_top`, plus `appui`, `rec_S`, `j_lat` (`nat_N` et
`g_Mu` y sont **déjà**) → **66 clés**.
Reprises en propre : `hp_u, tp_u, bp_u, nu_pl, a_w, n1_u, p1_u, e1_u, p3_u`.
`p3_u` est le **gauge sur le poteau** ; son aide devient : « Entraxe des deux
files, symétrique par rapport à l'âme du poteau. L'écrou doit dégager le
congé : p3 ≥ twP + 2·rP + dw. Gauges usuels d'aile : 90 / 100 / 120 / 140. »

#### 10.6.3 Conditions d'affichage neuves (`entrees.py`)

```python
def sur_aile(u):  return u.get("appui") == APPUI_AILE
def sur_ame(u):   return u.get("appui") == APPUI_AME
def tying(u):     return u.get("nat_N") == NAT_TYING
def traction(u):  ...                       # repris de platine_about/entrees.py
def _tying_N(u):  return tying(u) and traction(u)
```

### 10.7 Les DESSINS

**Trois vues par module**, comme les grands frères, mais **la vue en plan
devient la vue porteuse de l'information neuve** :

| vue | direction du regard | ce qu'on voit — cas (a) AILE | ce qu'on voit — cas (b) ÂME |
|---|---|---|---|
| **ÉLÉVATION** | horizontal, ⊥ à l'axe de la poutre | **le poteau vu de côté** : bande verticale de largeur `hP`, **rompue en haut et en bas** (`_rect_rompu`), les deux ailes en bandes de `tfP`, les congés en trait fin ; la poutre portée vue de côté (âme, semelles, congés, recoupe/grugeage) ; la pièce d'attache de face ; le groupe de boulons ; le symbole de soudure EN 22553 | **le poteau vu de côté** : bande de largeur `bP` rompue en haut et en bas, l'âme repérée par deux traits fins à `±twP/2` ; le reste identique |
| **VUE EN PLAN** (coupe horizontale au niveau des boulons) | vertical, de dessus | **LE POTEAU EN COUPE** (`_section_I`, hachuré `_hachures`) ; l'âme de la poutre portée en coupe, hachurée ; la pièce d'attache (coupée → hachurée) ; les files de boulons ; les cordons à `a·√2` | **LE POTEAU EN COUPE**, la poutre arrivant **entre les ailes** ; la recoupe des semelles ; les jeux `j_lat` ; `gh` depuis la face de l'âme |
| **COUPE A-A** (regard le long de la poutre portée) | horizontal, suivant l'axe de la poutre | la face extérieure de l'aile (rectangle `bP` × hauteur, rompu en haut et en bas) ; la pièce **par la tranche** avec ses cordons ; les rangées de boulons de profil ; la coupe de la poutre portée par-devant (`_section_portee`) | **l'intérieur du poteau** : l'âme en face, les deux ailes **vues par la tranche** à `±hP/2` avec leurs congés — c'est la vue qui montre la **largeur droite entre congés** |

**Cotes NEUVES** (symboles ; `calc = 1` = grandeur calculée, non modifiable) :

| id | symbole | sens | modifiable ? |
|---|---|---|---|
| `hP` | `h` | hauteur (profondeur) du poteau | non (`prof_P`) |
| `bP` | `b` | largeur d'aile du poteau | non |
| `tfP` | `tf` | épaisseur d'aile | non |
| `twP` | `tw` | épaisseur d'âme du poteau | non |
| `p3` | `p3` | entraxe des deux files sur le poteau | **oui** → `p3_u` (pac) ; calculée pour dcc (`tw_S + 2·gA`) |
| `eP` | `e` | pince au bout d'aile `(bP − p3)/2` | non, `calc` |
| `libreP` | `ℓ` | largeur droite entre congés `hP − 2tfP − 2rP` | non, `calc`, cas (b) |
| `rec` | `r_c` | recoupe des ailes de la poutre | **oui** → `rec_S`, cas (b) |
| `jlat` | `j` | jeu latéral | **oui** → `j_lat`, cas (b) |
| `gh` | `gh` | jeu face d'appui → about | **oui** (sens redéfini) |
| `ecol` | `e_col` | axe du poteau → point d'application (information) | non, `calc` |

`CLE_PAR_COTE` reçoit donc, en plus des cotes du grand frère :
`{"p3": "p3_u", "rec": "rec_S", "jlat": "j_lat"}` ; `d_top`/`dz` disparaît.
`FABRICATION_CALC` reçoit `{"eP", "libreP", "ecol"}` en plus.
`EXCLURE_ELEVATION / EXCLURE_PLAN / EXCLURE_COUPE` se règlent pour que
**chaque cote ne soit portée qu'une fois** sur la planche de la page 2 :
`p3`, `eP`, `libreP`, `jlat` → VUE EN PLAN ; `hp`, `zp`, `e1`, `p1`,
`e1'` → ÉLÉVATION ; `tp`, `a`, `bP` → COUPE A-A.

**Pièces nommées** (`piece(grp, …)`, pour les fenêtres d'objet et la mise en
rouge) : `col` (poteau), `colFl` / `colWeb` (aile / âme sollicitée),
`beamS`, `plate` / `cleat`, `bolts`, `weld`, `efforts`, `notchT`, `notchB`,
`recut`.

### 10.8 La NOTE PDF — identique

**2 pages A4 paysage, aucune exception.**

- **Page 1** : bandeau (statut, `η max`, dimensionnante, taux par élément,
  alertes), en-tête compact en 3 lignes, trois colonnes (dessins +
  HYPOTHÈSES · tableaux par élément · tableaux + Tab. 3.3 + conclusion),
  formules et substitutions, pied en 2 lignes (`LEGENDE_REFERENCES`,
  `NOTATIONS`).
- **Page 2** : « PLAN DE PRINCIPE » — **trois vues à la même échelle
  normalisée** + cartouche.

**Mise en œuvre : SOUS-CLASSEMENT, pas recopie.** `plat_ame/note.py` a été
écrit pour cela (attributs de classe `TITRE_NOTE`, `SYN`, `SCH`, `COL2`,
`COL3`, `NORME`, `LEGENDE`, `NOTATIONS`, `REF` et méthodes `entete`,
`hypotheses`, `alerte`, `ref`) et `platine_about` le prouve déjà. Donc :

```python
from ..poutre_poutre.plat_ame.note import Note as NoteFamille
from ..poutre_poutre.doubles_cornieres.note import PlanPrincipe, TEXTE_MM, CART_H, _fmt_echelle

class Note(NoteFamille):
    TITRE_NOTE = "Assemblage poutre–poteau — plat d'âme soudé (fin plate)"
    SYN, SCH   = synthese, schemas
    COL2       = ("boulons", "plat", "soudure")
    COL3       = ("portee", "poteau")
    NORME      = "(γM1 : ANB belge)"
    LEGENDE, NOTATIONS, REF = LEGENDE_REFERENCES, NOTATIONS_COURTES, ref_courte
```

`colonne_dessins` est redéfinie pour empiler **ÉLÉVATION + VUE EN PLAN** (la
COUPE A-A ne va qu'en page 2), et `PlanPrincipePP._vues` rend les trois vues.
**Cartouche de la page 2** : la case « POUTRES » devient
**« POTEAU · POUTRE »** (`P : HEB 300 S355` / `S : IPE 400 S355`) et la case
« ASSEMBLAGE » porte « Poutre–poteau – plat d'âme soudé · **appui : aile du
poteau** ». Les autres cases (`PIÈCE`, `FIXATIONS`, `DATE · INDICE`,
`ÉCHELLE`) sont inchangées.

**Bloc HYPOTHÈSES** (≤ 3 lignes de ≤ 110 caractères, `note_kit.empaqueter`,
« MSB P5 » une fois par ligne). Segments à écrire, pour le plat d'âme sur
l'aile, cas par défaut (longueurs mesurées à la mise en œuvre, le test rougit
si ça ne tient pas) :

1. « Articulé : rotule à la face de l'aile du poteau (MSB P5 §3.2) ;
   z = 50,0 mm ; le groupe reprend VEd·z + |MEd|. »
2. « Appui sur l'aile, attache dans le plan de l'âme du poteau ; plat court
   (z ≤ tp/0,15). »
3. « Cordons pleine hauteur, un par face (§4.5.3.3) ; γM0 1,00 · γM1 1,10
   (ANB) · γM2 1,25. »

Variante cas (b) : la ligne 2 devient « Appui dans l'âme du poteau ;
cisaillement local un seul côté ; flexion hors plan de l'âme NON vérifiée. »

---

## 11. CAS DE BENCHMARK — les nombres

### 11.1 Géométrie et matériaux de référence (communs aux trois modules)

- **poutre portée** : **IPE 400 S355** — `h 400, b 180, tw 8,6, tf 13,5,
  r 21` ; `fy 355, fu 510, βw 0,90`. Aire recalculée
  `A = 2b·tf + (h − 2tf)·tw + (4 − π)·r² = 8 446,36 mm²` ;
  `Av (non grugée) = 4 269,46 mm²` ; partie droite de l'âme
  `h − (tf + r) = 365,5 mm`.
- **poteau** : **HEB 300 S355** — `h 300, b 300, tw 11, tf 19, r 27`.
  Largeur droite entre congés `300 − 38 − 54 = 208 mm` ; demi-largeur
  `104 mm` ; `p3` mini sur l'aile avec M20 : `11 + 54 + 37 = 102 mm`.
- **boulons** : M20 8.8, filet dans le plan de cisaillement —
  `d 20, d0 22, As 245, dm 31,8, dw 37` ; `fub 800`, `αv 0,6` →
  **`Fv,Rd = 94,08 kN`**, `Ft,Rd(γM2) = 141,12 kN`,
  **`Ft,Rd,u(γMu) = 160,364 kN`**.
- **coefficients** : `γM0 = 1,00` ; `γM1 = 1,10` (ANB) ; `γM2 = γM2n = 1,25` ;
  `γMu = 1,10`.
- **`VEd = 200 kN`** sur tous les cas (a) et (b).

### 11.2 `fpc_` — plat d'âme sur poteau : 4 cas

#### BM1 — HÉRITAGE : l'exemple publié MSB P5 §3.4, appui sur l'AILE

Le Worked Example « Fin Plate » du guide, rejoué avec `appui = Aile du
poteau` et un poteau `PERSO` d'aile `tfP = 15 mm` S275 (l'épaisseur de l'âme
support de l'exemple). **Les dix valeurs publiées doivent sortir
inchangées** — c'est démontré : elles ne dépendent pas du support (script 00).

| page | grandeur | publié | outil | écart |
|---|---|---|---|---|
| p. 5-40 | inertie polaire `Ip` | 107 000 mm² | 107 000,00 | 0,000 % |
| p. 5-40 | groupe excentré, boulons `VRd` | 584 kN | 584,072 | +0,012 % |
| p. 5-41 | pression diamétrale du plat `VRd` | 605 kN | 600,403 | −0,760 % |
| p. 5-42 | pression diamétrale de l'âme `VRd` | 624 kN | 623,679 | −0,051 % |
| p. 5-43 | plat, cisaillement brut `VRd,g` | 450 kN | 450,060 | +0,013 % |
| p. 5-43 | plat, cisaillement net `VRd,n` | 497 kN | 496,521 | −0,096 % |
| p. 5-43 | plat, rupture de bloc `VRd,b` | 483 kN | 483,325 | +0,067 % |
| p. 5-45 | âme, cisaillement brut | 953 kN | 957,350 | +0,456 % |
| p. 5-45 | âme, cisaillement net | 995 kN | 1 000,936 | +0,597 % |
| p. 5-45 | âme, rupture de bloc (non grugée : `(n1−1)·d0`) | 507 kN | 506,678 | −0,064 % |

Écarts expliqués, textes à reprendre **mot pour mot** de
`plat_ame/benchmark.py` : `E_ARR` (arrondis intermédiaires de la source,
`αb = 0,61` pour 0,606 ; `Fv,Rd = 94` pour 94,08) et `E_AIRE` (la source
utilise l'aire catalogue 11 700 mm², l'outil recalcule 11 729 mm², +0,25 %).
**Pire écart 0,76 %** ; déversement du plat long : `λLT = 38,798`,
`pb = 264,648 MPa`, `Mb,Rd = 57,164 kNm` (`γM1 = 1,0`, comme la source).

Données de l'exemple, à reprendre **telles quelles** du dict `WE` de
`plat_ame/benchmark.py`, en remplaçant seulement le support :
IPE A 550 S275 portée (`h 547, b 210, tw 9, tf 15,7, r 24`), **poteau `PERSO`
S275 `tfP = 15`**, plat **360 × 160 × 10 S275**, M20 8.8, `n1 = 5`, `n2 = 2`,
`p1 = 70`, `p2 = 60`, `e1 = 40`, `e2,b = 40`, `gh = 10`, `zp = 50`,
poutre **non grugée**, `VEd = 350 kN`, `γM1 = 1,0`
(`z = 80 mm`, `e2 plat = 50`, `e1,b âme = 90`).

#### BM2 — cas (a) AILE, cas par défaut du module

IPE 400 S355 / HEB 300 S355 · plat **290 × 100 × 10 S355** · 4 M20 8.8,
`p1 = 70`, `e1 = 40` · `gh = 10`, `e2,b = 40`, `zp = 40` · `a = 5` ·
`VEd = 200 kN`.

| grandeur | valeur |
|---|---|
| `z = gh + e2,b` | **50,0 mm** |
| `MS = VEd·z` | **10,0 kNm** |
| pince `e2` du plat (bord libre) | 50,0 mm |
| pince basse `e1' = hp − e1 − 3·p1` | 40,0 mm |
| `Ip = n·p1²(n1² − 1)/12` | **24 500 mm²** |
| `Fz = VEd/n` | 50,000 kN |
| `Fx = MS·ymax/Ip` | 42,857 kN |
| `F,Ed = √(Fz² + Fx²)` | **65,854 kN** |
| `η` boulons (`Fv,Rd = 94,08`) | **0,700** |
| plat : `αb,ver / k1,ver` | 0,606061 / 2,5 |
| plat : `Fb,ver,Rd / Fb,hor,Rd` | 123,636 / 154,545 kN |
| plat : `η` pression diamétrale (interaction) | **0,4904** |
| plat : `VRd,g` (1/1,27) | **468,017 kN** |
| plat : `VRd,n` | 475,829 kN |
| plat : `Ant / Anv` | 390 / 1 730 mm² |
| plat : `Veff,2,Rd` | **434,140 kN** |
| plat : court (`hp ≥ 2,73·z` et `z ≤ tp/0,15 = 66,7`) | flexion et déversement **sans objet** |
| gorge de pleine résistance recommandée | **3,915 mm** → `a = 5` ✔ |
| âme : `e1,b = zp + e1 / he` | 80 / 110 mm |
| âme : `Fb,ver,Rd / Fb,hor,Rd` | 142,213 / 106,327 kN |
| âme : `η` pression diamétrale | **0,5349** |
| âme : `VRd,g` | 875,065 kN |
| cordons : `fvw,d` | **261,732 MPa** |
| cordons : `Fw,Rd = fvw,d·a` | 1 308,66 N/mm |
| cordons : `qz / qx / Fw,Ed` | 344,828 / 356,718 / **496,139 N/mm** → `η = 0,379` |
| **poteau** | **aucune pression diamétrale (attache SOUDÉE) ; groupe « Poteau porteur » vide** |

→ **dimensionnante : les boulons, `η = 0,700`.** Assemblage VÉRIFIÉ.

#### BM3 — cas (b) ÂME, variante **b1** : l'attache ne passe plus

Même pièce, même poutre, même poteau ; `appui = Âme du poteau`, la poutre
reste au-delà des bouts d'aile.

| grandeur | valeur |
|---|---|
| `gh` minimal `= (bP − twP)/2 + jeu` | **154,5 mm** → `gh = 155` |
| `z = gh + e2,b` | **195,0 mm** (× 3,9) |
| `MS` | **39,0 kNm** |
| `F,Ed` du boulon le plus sollicité | 174,461 kN |
| **`η` boulons** | **1,854** ✗ |
| plat **LONG** (`z > tp/0,15 = 66,7`) | oui |
| `λLT = 2,8·√(z·hp/(1,5·tp²))` | **54,366** |
| `pb` (Perry-Robertson, `E = 205 000`) | **276,516 MPa** |
| `Wel = tp·hp²/6` | 140 166,67 mm³ |
| `Mb,Rd = Wel·pb/γM1` | **35,235 kNm** |
| **`η` déversement** | **1,107** ✗ |
| âme du poteau : `VRd = twP·hp·fy/(√3·γM0)` | **653,820 kN** pour `Ed = VEd/2 = 100` → `η = 0,153` |

→ **NON VÉRIFIÉ, deux taux au-dessus de 1.** C'est le résultat utile : la
variante b1 d'un plat d'âme en âme de poteau coûte cher. **Le même cas
redimensionné** (5 × 2 **M24** 8.8, `p1 = p2 = 60`, plat **320 × 12**,
`z = 225 mm`, `MS = 45 kNm`) :

| grandeur | valeur |
|---|---|
| `Ip = n·[p1²(n1²−1) + p2²(n2²−1)]/12` | **81 000 mm²** |
| `Fz / Fx / F,Ed` | 36,667 / 66,667 / **76,085 kN** |
| `Fv,Rd` (M24, filet) | 135,552 kN → **`η = 0,561`** ✔ |
| `λLT / pb / Wel / Mb,Rd` | 51,121 / **287,879 MPa** / 204 800 mm³ / **53,598 kNm** |
| **`η` déversement** | **0,840** ✔ |
| ductilité `tp ≤ 0,5·d = 12` | ✔ |

#### BM4 — cas (b) ÂME, variante **b2** : ailes de la poutre recoupées

| grandeur | valeur |
|---|---|
| recoupe minimale `= (bP − twP)/2 + jeu − gh` | **144,5 mm** (`gh = 10`) |
| la poutre passe sans recoupe si `bS ≤ hP − 2tfP − 2j = 252` | IPE 400 (`bS = 180`) ✔ ; **HEA 300 (`bS = 300`) ✗ → recoupe obligatoire** |
| `z` retombe à `gh + e2,b` | **50,0 mm** → tous les taux de BM2 |
| âme du poteau : `VRd = twP·hp·fy/(√3·γM0)` | **653,820 kN**, `Ed = 100` → `η = 0,153` |

#### Tying du plat d'âme (`NEd = 250 kN`, `γMu = 1,1`)

| grandeur | valeur |
|---|---|
| **boulons en CISAILLEMENT** `Fv,Rd,u = αv·fub·A/γMu` | **106,909 kN** → `n·Fv,Rd,u = 427,636 kN` |
| plat, traction section nette `0,9·Anet·fu/γMu` | **842,891 kN** |
| âme portée, traction section nette (hauteur participante `hp`) | **724,886 kN** |
| cordon : `qx = N/(2hp)` | 431,034 N/mm |
| cordon : `Fw,Rd,u = fu/(√3·βw·γMu)·a` | **1 487,11 N/mm** |
| **côté poteau, cas (a)** | rien de plus (l'attache est dans le plan de l'âme) |
| **côté poteau, cas (b)** | flexion hors plan de l'âme **non vérifiée** — limite écrite |

### 11.3 `pac_` — platine d'about sur poteau : 4 cas

#### BM1 — HÉRITAGE : l'exemple publié MSB P5 §2.4, appui sur l'AILE

IPE A 550 S275, platine 430 × 200 × 12 S275, 6 × 2 M20 8.8, `p3 = 140`,
`a = 5,6`, `VEd = 550 kN`, tying 275 kN ; poteau `PERSO` d'aile `tfP = 15`
S275.

| grandeur | publié | outil | écart |
|---|---|---|---|
| âme de la poutre, cisaillement sur `hp` (`Av = 3 870 mm²`) | 614 kN | 614,445 | +0,072 % |
| un boulon en cisaillement `Fv,Rd` | 94 kN | 94,080 | +0,085 % |
| pression diamétrale platine, boulons d'extrémité | 107 kN | 105,986 | −0,948 % |
| pression diamétrale platine, boulons intérieurs | 142 kN | 141,756 | −0,172 % |
| groupe `FRd = 0,8·ns·Fv,Rd` | 902 kN | 903,168 | +0,129 % |
| platine, `min(brut ; net ; bloc)` | 1 182 kN | 1 181,892 | −0,009 % |
| **tying** : platine en flexion (tronçon en T, mode 1) | 493 kN | 491,858 | −0,232 % |
| **tying** : âme de la poutre `tw·hp·fu/γMu` | 1 513 kN | 1 512,818 | −0,012 % |
| cordon de pleine résistance S275 : `a ≥ 0,48·tw` | 0,48 | 0,4805 | +0,101 % |

Tronçon en T de l'exemple (ancre VAL-A à reprendre) : `m = 59,164`, `n = 30`,
`ew = 9,25`, `Σleff = 430 mm` (= `hp`), modes **491,86 / 783,20 /
1 924,36 kN**. Pire écart **0,95 %** (`αb` arrondi à 0,61 par la source).

#### BM2 — cas (a) AILE, cas par défaut

IPE 400 S355 / HEB 300 S355 · platine **290 × 220 × 10 S355** · 4 × 2 M20
8.8, `p1 = 70`, `e1 = 40`, **`p3 = 140`** · `a = 5` · `VEd = 200 kN`.

| grandeur | valeur |
|---|---|
| `p3` minimal (dégagement du congé du poteau) | **102 mm** → **`p3 = 90` REFUSÉ**, `p3 = 140` ✔ |
| pince au bout d'aile `e = (bP − p3)/2` | 80,0 mm (`≥ 1,2·d0 = 26,4`) ✔ |
| pince latérale de la platine `e2 = (bp − p3)/2` | 40,0 mm |
| `ns = 2·n1` ; `Fv,Ed = VEd/ns` | 8 ; 25,0 kN |
| platine : `k1 / αb(extrémité) / αb(intérieur)` | 2,5 / 0,606061 / 0,810606 |
| platine : `Fb,Rd` extrémité / intérieur | **123,636 / 165,364 kN** |
| branche du groupe (`Fv,Rd ≤ (Fb,Rd)min`) | `FRd = 0,8·ns·Fv,Rd` = **602,112 kN** → `η = 0,332` |
| platine : `VRd,g` (2 plans, 1/1,27) | **936,035 kN** |
| platine : `VRd,n` (2 plans) | 951,658 kN |
| platine : `k = 1` (`hp = 290 ≥ 1,36·p3 = 190,4`) ; `Ant / Anv` | 290 / 1 730 mm² |
| platine : `VRd,b` (2 blocs) | **945,799 kN** |
| âme portée : `Av = hp·tw = 2 494 mm²` ; `VRd` | **511,169 kN** → `η = 0,391` |
| cordons : `qz` ; pleine résistance `a/tw = 0,553727` → `a ≥ 4,762 mm` | 344,828 N/mm ; `a = 5` ✔ |
| **aile du poteau** : `αb = 0,810606`, `k1 = 2,5`, `t = tfP = 19` | **`Fb,Rd = 314,191 kN/boulon`** ; `VRd = ns·Fb,Rd = 2 513,53 kN` → `η = 0,0796` |

→ **dimensionnante : l'âme de la poutre portée sur `hp`, `η = 0,391`.**

#### BM3 — cas (a) TYING, `NEd = 250 kN` : le tronçon en T de l'AILE

`Ft,Rd,u = 160,364 kN` ; `ΣFt,Rd,u = 8 × = 1 282,91 kN` ; `ew = dw/4 = 9,25`.

| | **platine** (`tp 10`) | **aile du poteau** (`tfP 19`) |
|---|---|---|
| `m` | `(p3 − tw,S)/2 − 0,8·a·√2` = **60,043 mm** | `(p3 − twP)/2 − 0,8·rP` = **42,900 mm** |
| `e` | `e2` = 40,0 mm | `(bP − p3)/2` = 80,0 mm |
| `n = min(e ; 1,25·m)` | **40,000 mm** | **53,625 mm** |
| `Σleff,1` | **290 mm** (= `hp`, la valeur du guide) | **280 mm** (= `n1·p1`, Tab. 6.4 en groupe) |
| `Mpl,1` | **3,3614 kNm** | **11,7161 kNm** |
| `FT,1` | **261,330 kN** | 1 296,99 kN |
| `FT,2` | 580,141 kN | **955,485 kN** |
| `FT,3` | 1 282,91 kN | 1 282,91 kN |
| **`FT,Rd`** | **261,330 kN** → `η = 0,9566` | **955,485 kN** → `η = 0,262` |

Rapport aile / platine : **3,656**. Autres composantes :

| grandeur | valeur |
|---|---|
| `Ft,Ed = N/ns` | 31,25 kN |
| poinçonnement de la platine (`t = 10`) | **277,911 kN** |
| poinçonnement de l'aile (`t = 19`) | 528,031 kN |
| âme du poteau en traction transversale (`ω = 1`, `beff = 280`) | **1 428,00 kN** |
| âme portée en traction sur `hp` | **1 156,31 kN** |
| cordon : `qx = 431,034` / `Fw,Rd,u = 1 487,11 N/mm` | `η = 0,290` |

→ **dimensionnante en tying : la platine, `η = 0,957`.** L'aile du poteau ne
gouverne pas ici — elle gouvernerait avec une aile mince ou un `p3` plus
grand, et c'est pour cela que la ligne existe.

#### BM4 — cas (b) ÂME DU POTEAU

Platine **290 × 190 × 10**, **`p3 = 110`**, `e2 = 40`.

| grandeur | valeur |
|---|---|
| `bp = 220` du cas (a) **ne tient pas** (`≤ 208 − 2·5 = 198`) | → `bp = 190` ✔ |
| l'écrou dégage le congé : `p3/2 + dw/2 = 73,5 ≤ 104` | marge **30,5 mm** ✔ |
| âme du poteau : `αb = 0,810606`, `k1 = min(1,4·p3/d0 − 1,7 ; 2,5) = 2,5` | **`Fb,Rd = 181,900 kN/boulon`** ; `VRd = 1 455,20 kN` |
| âme du poteau : cisaillement local **brut** (`Av = twP·hp`) | **653,820 kN** pour `Ed = 100` → `η = 0,153` |
| âme du poteau : cisaillement local **net** | **523,412 kN** → `η = 0,191` |
| poinçonnement de l'âme (`t = 11`) | **305,702 kN** |
| recoupe des ailes de la poutre | **non nécessaire** — critère de la SEMELLE : `bS = 180 ≤ hP − 2tfP − 2j = 252` ✔ (la limite 198 est celle de la PIÈCE d'attache, entre congés) |
| flexion hors plan de l'âme du poteau | **NON VÉRIFIÉE** — limite écrite |

### 11.4 `dcc_` — doubles cornières sur poteau : 4 cas

#### BM1 — HÉRITAGE : l'exemple publié MSB P5 §4.4, appui sur l'AILE

IPE A 550 S275, 2 L90 × 90 × 10 S275, `hc = 430`, 6 + 6 M20 8.8, `gA = 50`,
`VEd = 450 kN` ; poteau `PERSO` d'aile `tfP = 15` S275. **Seize valeurs
publiées**, dont **dix** contrôlées indépendantes du support au 1e-9
(script 00) :

| page | grandeur | publié | outil | écart |
|---|---|---|---|---|
| p. 5-70 | `Fv,Rd` (1 plan) | 94 kN | 94,080 | +0,085 % |
| p. 5-70 | groupe S excentré `VRd` | 962 kN | 962,835 | +0,087 % |
| p. 5-71 | `Fb,ver,Rd` cornière | 105 kN | 104,242 | −0,722 % |
| p. 5-71 | `Fb,hor,Rd` cornière | 105 kN | 104,242 | −0,722 % |
| p. 5-71 | groupe S, pression diamétrale cornières `VRd` | 1 075 kN | 1 066,839 | −0,759 % |
| p. 5-72 | `Fb,ver,Rd` âme | 125 kN | 125,482 | +0,385 % |
| p. 5-73 | `Fb,hor,Rd` âme | 94 kN | 93,818 | −0,193 % |
| p. 5-73 | groupe S, pression diamétrale âme `VRd` | 583 kN | 582,507 | −0,085 % |
| p. 5-74 | côté support, `FRd = 0,8·ns·Fv,Rd` | 902 kN | 903,168 | +0,129 % |
| p. 5-74 | cornières, cisaillement brut | 1 076 kN | 1 075,144 | −0,080 % |
| p. 5-75 | cornières, cisaillement net | 1 184 kN | 1 183,707 | −0,025 % |
| p. 5-75 | cornières, rupture de bloc (côté portée) | 954 kN | 953,950 | −0,005 % |
| p. 5-76 | cornières, rupture de bloc (côté support) | 954 kN | 953,950 | −0,005 % |
| p. 5-77 | âme, brut / net / bloc | 953 / 956 / 501 | 957,350 / 961,611 / 500,725 | +0,456 / +0,587 / −0,055 % |

**Tying de l'exemple publié (p. 5-78/79), ancre des MODES 1/2/3** : avec un
acier de cornière `PERSO fy = fu = 430`, `γM0 = γM2 = 1,1` (le guide applique
`fu/γMu`) — `Ft,Rd,u` 160 → 160,364 ; `Σleff` 430 → 430,0 ; `Mpl` 4,2 →
4,2023 kNm ; modes **696 → 691,25** ; **1 190 → 1 188,81** ; **1 920 →
1 924,36 kN** (`m = 31,2`, `n = 39,0`, `ew = 9,25`).
**C'est l'ancre publiée de la formule que l'aile du poteau réutilise.**

#### BM2 — cas (a) AILE, cas par défaut

IPE 400 S355 / HEB 300 S355 · 2 **L100 × 100 × 10 S355** · `hc = 290` ·
4 + 4 M20 8.8, `p1 = 70`, `e1 = 40` · `gA = 50`, `gh = 10`, `e2,b = 40`,
`zp = 40` · `VEd = 200 kN`.

| grandeur | valeur |
|---|---|
| `p3 = tw,S + 2·gA` | **108,6 mm** (`≥ 102` ✔, de justesse) |
| congé de la cornière : `gA − d0/2 = 39 ≥ tc + rc = 22` | ✔ |
| pince au bout d'aile du poteau `e = (bP − p3)/2` | **95,7 mm** |
| pinces `e2` des ailes A et B | 50,0 / 50,0 mm |
| `z` (face de l'aile → centre du groupe S) ; `MS` | **50,0 mm** ; **10,0 kNm** |
| `Ip` ; `F_S` | 24 500 mm² ; **65,854 kN** |
| `η` groupe S (2 plans : `2·Fv,Rd = 188,16`) | **0,350** |
| `F_P = (VEd/2)/n1P` ; `FRd = k_rot·Fv,Rd = 0,8 × 94,08` | 25,0 kN ; 75,264 kN → **`η = 0,332`** |
| cornières : `VRd,g` / `VRd,n` | **936,035 / 951,658 kN** |
| ailes B : `Ant / Anv` ; `VRd,b` | 390 / 1 730 mm² ; **868,279 kN** |
| ailes B : flexion `MEd = (VEd/2)·z` ; `MRd = Wel·fy/γM0` | 5,0 ; **49,759 kNm** → `η = 0,100` |
| pression diamétrale ailes B : `Fb,ver,Rd = 123,636` ; `η` (effort `F/2`) | **0,2452** |
| pression diamétrale ailes A : `Fb,ver,Rd` | 123,636 kN |
| âme portée : `e1,b = 80` ; `Fb,ver / Fb,hor` | 142,213 / 106,327 kN → **`η = 0,5349`** |
| **aile du poteau** : `αb = 0,810606`, `k1 = 2,5`, `t = 19` | **`Fb,Rd = 314,191 kN`** → `η = 0,0796` |

→ **dimensionnante : la pression diamétrale de l'âme portée, `η = 0,535`.**

#### BM3 — cas (a) TYING, `NEd = 300 kN`

`Ft,Rd,u = 160,364` ; `ΣFt,Rd,u = 2·n1P·Ft,Rd,u = 1 282,91 kN` ; `ew = 9,25`.

| | **ailes A des cornières** (`tc 10`) | **aile du poteau** (`tfP 19`) |
|---|---|---|
| `m` | `(p3 − tw,S − 2tc − 1,6·rc)/2` = **30,400 mm** | `(p3 − twP)/2 − 0,8·rP` = **27,200 mm** |
| `n` | **38,000 mm** | **34,000 mm** |
| `Σleff` | `2·e1 + (n1P−1)·p1` = **290 mm** | `n1·p1` = **280 mm** (Tab. 6.4) |
| `Mpl,1` | **3,3614 kNm** | **11,7161 kNm** |
| `FT,1 / FT,2 / FT,3` | **572,015** / 811,013 / 1 282,91 kN | 2 314,01 / **1 095,61** / 1 282,91 kN |
| **`FT,Rd`** | **572,015 kN** → `η = 0,5245` | **1 095,61 kN** → `η = 0,274` |

Rapport aile / cornières : **1,915** — **les cornières gouvernent.** Autres :

| grandeur | valeur |
|---|---|
| poinçonnement de l'aile A (`t = 10`) / de l'aile du poteau (`t = 19`) | **277,911 / 528,031 kN** |
| âme du poteau en traction transversale (`ω = 1`, `beff = 280`) | **1 428,00 kN** |
| ailes B, traction section nette (`2·0,9·Anet·fu/γMu`) | **1 685,78 kN** |
| âme portée, traction section nette (hauteur participante `hc`) | **724,886 kN** |

#### BM4 — cas (b) ÂME DU POTEAU

2 **L90 × 90 × 9 S355**, `gA = 50` — **une L100 ne passe pas.**

| grandeur | valeur |
|---|---|
| `p3 = tw,S + 2·gA` | 108,6 mm |
| demi-largeur droite de l'âme `(hP − 2tfP)/2 − rP` | **104,0 mm** |
| l'écrou dégage le congé : `p3/2 + dw/2 = 72,8 ≤ 104` | ✔ |
| le bout de l'aile A dégage le congé : `tw,S/2 + bA = 94,3 ≤ 104` (L90) | ✔ |
| … avec une **L100** : `4,3 + 100 = 104,3 > 104` | **✗ bloquant** |
| longueur d'aile A maximale admise | **99,7 mm** |
| `gA` minimal (congé de la cornière) `= tc + rc + d0/2` | **31,0 mm** |
| pince `e2` de l'aile A `= bA − gA` | 40,0 mm |
| âme du poteau : `αb = 0,810606`, `k1 = 2,5`, `t = 11` | **`Fb,Rd = 181,900 kN`** → `η = 0,1374` |
| `et = 5·d` ; `eb = min(p3/2 ; 5d)` | 100,0 ; **54,3 mm** |
| `Av = twP·(et + (n1P−1)·p1 + eb)` | **4 007,3 mm²** |
| cisaillement local **brut** / **net** | **821,334 / 715,934 kN** pour `Ed = 100` → `η = 0,122 / 0,140` |
| recoupe des ailes de la poutre | non nécessaire (`bS = 180 ≤ 252`) |

### 11.5 « Vérifications sans exemple publié consulté » (`NOCOVER`)

À écrire dans `benchmark.py`, un item par ligne :

1. **Pression diamétrale sur l'aile / l'âme du poteau** — aucun exemple
   publié : formule directe du Tab. 3.4 avec `αb` sans pince d'extrémité et
   `k1` du gauge ; calcul manuel indépendant (BM2, BM4).
2. **Tronçon en T de l'AILE DU POTEAU** — aucun exemple publié consulté pour
   une attache simple ; les formules des modes 1/2/3 sont ancrées par
   l'exemple §4.4 (BM1 des cornières) et §2.4 (BM1 de la platine) ; les
   longueurs efficaces viennent du Tab. 6.4 (aile non raidie) ; calcul manuel
   indépendant (BM3).
3. **Âme du poteau en traction transversale** — §6.2.6.3 avec `ω = 1` :
   interprétation de l'outil, calcul manuel indépendant (BM3).
4. **Cisaillement local de l'âme du poteau** — même modèle que les trois
   modules existants (SCI P358, un seul côté chargé), `et = 5·d` ; calcul
   manuel indépendant (BM4).
5. **Déversement du plat LONG en âme de poteau** — la courbe est celle du
   module `plat_ame` (BS 5950-1 Ann. B.2.1), ancrée par son VAL-A ;
   l'application à `z ≈ 200 mm` est nouvelle : calcul manuel indépendant
   (BM3 de `fpc_`).
6. **Règles géométriques du cas (b)** (largeur droite entre congés, recoupe
   des ailes, longueur d'aile A, gauge minimal) — géométrie, pas de
   résistance : calculs manuels (BM4).
7. **Glissement (catégories B et C)** — formule directe §3.9, aucun exemple
   publié.

---

## 12. PLAN DE MISE EN ŒUVRE

Un **lot = un commit**, le harnais dans le MÊME commit, aucun lot commencé
avant que le précédent soit vert et relu. Les trois modules se font **en
série pour le lot 1 puis en parallèle** (lots 3, 4, 5), en worktrees séparés.

### Lot 0 — préparation (chef de projet)

1. Relire ce document et trancher les **9 points P-1 à P-9** (§9).
2. `acier/registre.py` : la famille « Poutre – Poteau » et les trois
   `Assemblage` à l'état **« à venir »** (`etat="à venir"`, `module=""`) +
   les trois `vignette_*` (élévation schématique : poteau en coupe, pièce,
   boulons).
3. Faire suivre `tests/test_assemblages_kit.py` §10 (7 entrées) et
   `tests/test_fumee_pages.py` (les ids « à venir » ne s'ouvrent pas : la
   boucle ne les prend pas encore).
4. **Recette** : `test_assemblages_kit`, `test_fumee_pages`,
   `test_projet_assemblages`, `test_rapport_projet` au vert ; les suites des
   quatre modules existants **inchangées**.

### Lot 1 — `acier/assemblages/mecanique.py` et son test de liaison

1. Écrire `mecanique.py` (§10.2), docstrings en français, noms en anglais,
   **bibliothèque standard seulement**.
2. Écrire **`tests/test_assemblages_mecanique.py` ROUGE D'ABORD** : pour les
   **56 cas figés** (12 + 13 + 31), comparer à 1e-12 `section_grugee`,
   `bearing`, `groupe_excentre`, `bolt_table`, `pb_bs5950`, `lam_LT`,
   `tp_ductile`, `coef_pleine_resistance`, `groupe_boulons`, `troncon_T`,
   `leff_platine_bords_libres` aux grandeurs homonymes des trois moteurs
   existants. Ajouter un contrôle de `leff_aile_poteau` contre les nombres du
   §7.3 : pour `n1 = 4`, `p1 = 70`, `m = 42,9`, `e = 80` →
   `Σcp isolées = 1 078,19`, `Σnc isolées = 1 086,40`, `Σcp groupe = 560`,
   `Σnc groupe = 280` mm, donc `Σleff,1 = Σleff,2 = 280 mm`.
3. **Aucun fichier existant modifié.**
4. **Recette** : le test neuf au vert ; les suites des modules existants
   **identiques à l'octet**.

### Lot 2 — `fpc_` (plat d'âme sur poteau), le GABARIT

L'ordre interne vaut pour les trois modules.

| étape | fichier | contenu | contrôle de fin d'étape |
|---|---|---|---|
| 2.1 | `entrees.py` | 71 clés, `NUM`, `APPUI_*`, `NAT_*`, conditions (`sur_aile`, `sur_ame`, `tying`, `traction`, `_tying_N`), `GROUPES`, `PILOTEES_PAR_PREDIM`, `UNITE`, `CHAMPS`, `charger_json` | `defaults()` a 71 clés ; `charger_json` ignore les inconnues ; un fichier sans `appui` reçoit `APPUI_AILE` |
| 2.2 | `moteur.py` | `compute()` : géométrie des deux cas, Tab. 3.3, alertes (§8.1), les 25 vérifications, `tying`, `predim`, `apply_solution` | **`test_fpc_moteur.py`** : BM1 héritage (10 valeurs), BM2 / BM3 / BM3' / BM4 (§11.2) à 1e-9 ; alertes neuves ; 200 tirages aléatoires sans exception ; `compute({})` vérifié |
| 2.3 | `schemas.py` | `elevation`, `plan`, `coupe_aa`, cotes neuves, `CLE_PAR_COTE`, `FABRICATION_CALC`, `EXCLURE_*` | chaque cote de `CLE_PAR_COTE` est émise au moins une fois ; chaque cote de la planche est émise **exactement une fois** ; les trois vues rendent du SVG non vide dans les deux cas d'appui ; **contrôle de GÉOMÉTRIE du canevas** (une vue peut être complète dans le DOM et vide à l'écran) |
| 2.4 | `notation.py`, `synthese.py`, `formules.py`, `texte.py`, `methode.py` | références courtes, `NOTATIONS` (≤ 798 pt à 5,6 pt), `TABLES` / `ELEMENTS` dans l'ordre du §8.1, `METHODE` avec les limites du §8.4 et les points de §9 | `ref_courte` de chaque `ref` actif est non vide et sans commentaire interne ; `synthese.lignes` couvre **toutes** les clés de `ck` actives ; `METHODE` contient les 9 limites |
| 2.5 | `benchmark.py` | `SRC`, `BENCH` (BM1 à BM4), `NOCOVER` (7 items), `VALID` (VAL-A : le déversement du cas b1 ; VAL-B : le cas (a) par défaut), `run_bench()` | `run_bench()["statut"] == "VALIDÉ"`, `worst ≤ 0,01` hors écarts expliqués |
| 2.6 | `ecran_saisie.py` | `PREFIXE = "fpc_"`, `LIBELLE` (≤ 16 / 20 / 22), `AIDES` (≤ 160, **4 « ? » visibles au défaut** : `mode_calc`, `appui`, `trou`, `cat`… à mesurer), `AFFICHAGE`, `BLOCS`, `AVANCES`, `CARTOUCHE`, `GEOMETRIE_DESSIN`, `DIMENSIONS`, `D` | `test_assemblages_kit` étendu : rien ne se perd (`CHAMPS ∪ GEOMETRIE_DESSIN ⊂ carte ∪ avancés ∪ cartouche ∪ cotes ∪ fenêtres`) ; aucune ligne d'objet > 4 widgets ; étanchéité du préfixe |
| 2.7 | `ecran_resultats.py` | bandeau, alertes, les trois dessins, panneau Cotes, 5 onglets, `NON_REVERIFIES`, `CAPTION_USAGE` | **`test_fpc_ecran.py`** : l'écran s'ouvre dans les deux cas d'appui, la carte porte les bonnes clés, les onglets sont dans `md(at)`, la bascule `appui` change le groupe « Poteau porteur » |
| 2.8 | `note.py` | `Note(plat_ame.note.Note)`, `PlanPrincipePP(PlanPrincipe)`, `generer_note()` | `derniers_avertissements == []` (rien ne déborde) sur les 4 cas de benchmark **et** sur les 12 cas figés ; 2 pages ; `empaqueter` ≤ 3 × 110 |
| 2.9 | `interface.py` | `MODULE_VERSION = "1.0"`, barre d'outils partagée, `charger_payload`, `_payload`, `show()` | `test_projet_assemblages` et `test_rapport_projet` étendus : création, duplication, verrou, rapport de projet mixte (un `fpl_` + un `fpc_`) |
| 2.10 | `registre.py` | passer `plat_ame_poteau` à **« disponible »** avec son `module` | `test_fumee_pages` : l'écran s'ouvre |
| 2.11 | `ref_fpc.json` | **régression figée** : ≥ 12 cas (défaut aile, défaut âme b0, b1, b1 redimensionné, b2, grugé, `NEd` ELU, tying, cat. B, cat. C, trou surdimensionné, profil `PERSO`), `checks` (Ed, Rd, η), scalaires, `gov`, `verified`, `alerts` | « RÉGRESSION CALCUL : 0 différence » |
| 2.12 | `docs/assemblages/PLAT_AME_POTEAU.md` | la fiche du module, sur le modèle de `PLAT_AME.md` | — |

**Recette du lot 2** : `test_fpc_moteur`, `test_fpc_ecran`,
`test_assemblages_kit`, `test_fumee_pages`, `test_projet_assemblages`,
`test_rapport_projet`, `test_assemblages_mecanique` au vert ; **les suites des
quatre modules existants identiques** ; `frontend`-équivalent du dépôt : les
captures des **trois formats** (PC 1440, tablette 1024, téléphone 390) relues
à l'œil avec `tools/captures_assemblages.js`, dans les **deux** cas d'appui.

### Lot 3 — `pac_` (platine d'about sur poteau)

Même découpage 3.1 à 3.12. Spécificités :

- `moteur.py` : `tronçon_T` appelé **deux fois** (platine, puis aile du
  poteau) ; `twtP` ; `pp` avec `t` du poteau selon `appui` ; `vlP`/`vnP`
  seulement en cas (b) ; alerte `p3_congeP` **bloquante**.
- `predim` : `p3 = ceil5(max(90 ; 2,4·d0 ; twP + 2·rP + dw))` plafonné par
  `bP − 2·1,2·d0` (cas a) ou par `hP − 2tfP − 2rP − 2j − 2·1,2·d0` (cas b).
- `benchmark.py` : BM1 héritage (9 valeurs), BM2, BM3 (tying, le tableau du
  §7.3), BM4 (cas b) ; VAL-A = le tronçon en T de l'exemple publié ;
  VAL-B = le tronçon en T de l'**aile** (les 6 nombres du §11.3 BM3).
- `ref_pac.json` : ≥ 13 cas.

### Lot 4 — `dcc_` (doubles cornières sur poteau)

Même découpage. Spécificités :

- c'est le module le plus lourd (88 clés, 42 vérifications) ; commencer par
  `entrees.py` et `moteur.py`, et **ne pas toucher au modèle du groupe S** ;
- alertes neuves `p3_congeP`, `bA_congeP`, `p3_libreP` ; alerte `P_web`
  retirée ;
- `predim` : plafonner l'aile A par `(hP − 2tfP)/2 − rP − tw_S/2` en cas (b),
  et `gA` par le dégagement des deux congés (cornière **et** poteau) ;
- `benchmark.py` : BM1 héritage (14 + 6 valeurs de tying), BM2, BM3, BM4 ;
- `ref_dcc.json` : ≥ 14 cas (les deux fixations × les deux appuis).

### Lot 5 — docs, écarts, recette d'ensemble

1. `docs/assemblages/POUTRE_POTEAU.md` (ce document) **daté et révisé** des
   arbitrages rendus ;
2. les trois fiches de module ;
3. `ECARTS_ET_CORRECTIONS.md` : une section « Attaches poutre–poteau » avec
   **P-1 à P-9** et les écarts assumés de chaque module ;
4. `PROJET_ASSEMBLAGES.md` : la famille neuve dans le tableau « Où vivent les
   choses » ;
5. **recette d'ensemble** : toutes les suites du dépôt, une par une
   (`lancer_tests.py`), captures des trois formats pour les trois modules
   dans les deux cas d'appui, note PDF de chacun relue à l'œil, et un
   **rapport de projet mixte** (un assemblage de chaque type, 7 notes
   assemblées, signets, récapitulatif).

### Ce que chaque agent de réalisation doit respecter

- **Noms en anglais, commentaires et docstrings en français.**
- **Aucun fichier existant modifié** hors `acier/registre.py`,
  `tests/test_assemblages_kit.py` et `tests/test_fumee_pages.py`. Si une
  fonction du kit manque, l'écrire **localement** (préfixe `_kit_`) et le
  signaler dans son rapport — le chef de projet consolide après la fusion.
- **La géométrie se règle sur le DESSIN, jamais dans la carte** (règle P6 de
  `ERGONOMIE.md`).
- **Aucun nombre de mémoire** : toute valeur écrite dans un test, une fiche ou
  une note est recalculée à la main dans le même commit.
- **Aucun numéro de sous-clause inventé** : à défaut, la référence reste au
  chapitre.
- **Le harnais suit dans le même commit**, et chaque rouge est classé : le
  harnais **suit** (il décrivait l'ancien écran) ou le harnais a **trouvé**
  (il avait raison).
