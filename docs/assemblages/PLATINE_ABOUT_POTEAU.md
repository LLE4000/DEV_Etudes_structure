# Assemblage poutre–poteau par platine d'about de hauteur partielle — v1.0

Troisième module de la famille **Poutre – Poteau** (lot de la platine d'about
du plan `POUTRE_POTEAU.md` §12), écrit le 02/11/2026 sur le GABARIT du module
`plat_ame_poteau` : mêmes treize fichiers, même clé `appui`, mêmes trois vues,
même note de 2 pages, même charte (`ERGONOMIE.md`). Le côté PLATINE, BOULONS
et ÂME PORTÉE vient du grand frère poutre–poutre `platine_about` **sans
aucune modification de formule**.

## Ce qui change par rapport au grand frère `platine_about`

| | `platine_about` (poutre–poutre) | `platine_about_poteau` |
|---|---|---|
| support | âme d'une poutre porteuse | **aile** (cas a) ou **âme** (cas b) d'un poteau en I/H |
| clé centrale | — | **`appui`** : `APPUI_AILE` / `APPUI_AME`, en segmented_control d'en-tête |
| rotule | face de l'âme porteuse | face d'appui du poteau : `X = hP/2` (a) ou `twP/2` (b) |
| pression diamétrale du support | âme porteuse, boulons « intérieurs » | **aile** (a) ou **âme** (b), `αb` SANS pince d'extrémité verticale, `k1` du gauge et — en (a) — du bout d'aile |
| cisaillement local du support | toujours (brut et net) | **cas (b) seulement** |
| traction des boulons | platine (tronçon en T) seule | **+ tronçon en T de l'AILE** (Tab. 6.4) **+ âme du poteau en traction transversale** (§6.2.6.3, ω = 1) — cas (a) |
| flexion du support hors plan | « non vérifiée » (note d'alerte) | **limite écrite** en cas (b), avec son raisonnement (§5.4 de l'étude) |
| décalage `d_top` (Δz) | oui | **retirée** — un poteau n'a pas de dessus ; le dessin trace un tronçon rompu |
| grugeage | 50 mm par défaut | **0 par défaut** (rien n'obstrue au-dessus ni en dessous) ; tout le bloc s'active à l'identique dès qu'on grugue |
| gauge `p3` | « la rondelle dégage le cordon » | **+ l'écrou dégage le CONGÉ du poteau** : `p3 ≥ twP + 2·rP + dw` (a) ou `p3/2 + dw/2 ≤ (hP − 2tfP)/2 − rP` (b) |
| largeur `bp` | pince `e2` seule | **+ `bp ≤ hP − 2tfP − 2rP − 2j`** en cas (b) ; `bp ≤ bP` recommandé en cas (a) |
| clés neuves | — | `appui`, `rec_S` (recoupe des semelles), `j_lat` (jeu latéral) |
| alertes retirées | — | `pl_web`, `dnt_min`, `dnb_min`, `ln_min`, `gap_neg` (toutes fausses pour un poteau) |
| alertes neuves | — | `p3_congeP`, `p3_libreP`, `bp_libreP` (bloquantes), `bp_aile`, `aile_poutre`, `ame_hp`, `N_tying` |
| clés de session | 64 (`pab_`) | **66** (`pac_`) |

**Les briques de calcul communes ne sont pas recopiées** : pression
diamétrale (`bearing`, `portance_bout_interieur`), groupe de fixations
(`groupe_boulons`), poutre grugée (`section_grugee`), ductilité
(`tp_ductile`), gorge de pleine résistance (`coef_pleine_resistance`),
tronçon en T (`troncon_T`) et les deux jeux de longueurs efficaces
(`leff_platine_bords_libres`, `leff_aile_poteau`) viennent de
`acier/assemblages/mecanique.py` (lot 1), et la suite du moteur vérifie
qu'elles rendent bien les valeurs du moteur (§7 de
`test_platine_about_poteau_moteur`). La page 1 de la note est une
**sous-classe** de `plat_ame.note.Note`, la page 2 une sous-classe de
`doubles_cornieres.note.PlanPrincipe`, et `formules.py` importe la brique de
la famille.

## Les deux cas d'appui — le résultat central

> **Cas (a), sur l'AILE** : les deux files sont symétriques par rapport à
> l'ÂME du poteau ; l'attache vit **dans le plan de cette âme**. À l'ELU le
> côté poteau n'apporte que la **pression diamétrale de l'aile** — il n'y a
> **aucune pince d'extrémité verticale** (le poteau continue au-dessus et en
> dessous) : `αb = min(p1/3d0 − 1/4 ; fub/fu ; 1)` pour TOUTES les rangées, et
> `k1 = min(2,8·e/d0 − 1,7 ; 1,4·p3/d0 − 1,7 ; 2,5)` avec `e = (bP − p3)/2`.
> Dès que les boulons sont tendus s'ajoutent **deux composants normatifs que
> les trois modules poutre–poutre n'ont pas** : le tronçon en T de l'aile non
> raidie (§6.2.4 Tab. 6.2 méthode 2, longueurs efficaces du **Tab. 6.4** —
> toutes les rangées sont INTÉRIEURES) et l'âme du poteau en **traction
> transversale** (§6.2.6.3, `ω = 1`).
>
> **Cas (b), dans l'ÂME** : la platine porte sur l'âme, entre les ailes.
> `VEd` reste en plan — d'où le **cisaillement local** d'un élément support
> chargé d'un seul côté, brut et net (SCI P358). Mais la traction des boulons
> sollicite l'âme **hors de son plan** : l'EN 1993-1-8 ne couvre pas ce
> composant pour une âme non raidie. **Déclaré, pas transposé** — et les deux
> composants du cas (a) sont alors sans objet.

Géométrie du cas (b), mesurée sur le poteau de référence HEB 300
(`h 300, b 300, tw 11, tf 19, r 27`) :

| grandeur | formule | HEB 300 |
|---|---|---|
| largeur droite de l'âme entre congés | `hP − 2tfP − 2rP` | **208 mm** |
| largeur maximale de la PLATINE | `ℓ − 2·j` (j = 5) | **198 mm** |
| demi-largeur droite | `(hP − 2tfP)/2 − rP` | 104 mm |
| entraxe maximal des files (l'écrou dégage le congé) | `2·(demi-largeur − dw/2)` | **171 mm** |
| passage de la semelle portée | `hP − 2tfP − 2j` | **252 mm** |
| recoupe minimale des semelles | `(bP − twP)/2 + j − tp` | **139,5 mm** (tp = 10) |

→ la platine **290 × 220** du cas (a) **ne tient pas** en cas (b) : il faut
`p3 = 110` et `bp = 190`. Un IPE 400 (`bS = 180`) passe entre les ailes sans
recoupe ; une HEA 300 (`bS = 300`) l'exige. **La recoupe en LARGEUR n'est pas
un grugeage** : les règles de stabilité locale de MSB P5 §4.2.5 ne sont pas
activées par elle (point tranché P-3 de l'étude).

## Le GAUGE : la règle qui refuse le `p3 = 90` usuel

En cas (a), l'écrou est du côté **intérieur** de l'aile, au droit du congé de
raccordement : `p3 ≥ twP + 2·rP + dw`. Sur un **HEB 300 avec des M20, cela
fait 102 mm** — le `p3 = 90 mm` usuel des ÂMES de poutre **ne passe pas**
(26,5 mm disponibles pour 32,5 nécessaires). C'est la raison des gauges usuels
d'aile de poteau : **90 / 100 / 120 / 140 mm** selon la série. L'alerte
`p3_congeP` est **bloquante**, elle donne la règle, le chiffre et **propose le
minimum** (105 mm ici), et le prédimensionnement part de
`ceil5(max(90 ; 2,4·d0 ; twP + 2·rP + dw))`. Nature **INT** : règle
géométrique de dégagement, pas une clause normative (point P-7).

## Modèle et vérifications (22 lignes, §8.3 de l'étude)

Rotule à la face d'appui du poteau ; groupe **centré**, `VEd/n` par boulon,
**aucun moment d'excentricité** — c'est ce qui distingue cette attache du plat
d'âme et des cornières.

| Élément | Vérifications | Sources |
|---|---|---|
| Boulons | groupe en cisaillement et pression diamétrale sur la platine (3 branches) ; glissement (cat. B/C) ; interaction cisaillement + traction | Tab. 3.4 ; MSB P5 §2.2.4 sur §3.7 ; §3.9 |
| Platine | cisaillement brut (2 plans, ·1/1,27) / net / rupture de bloc (`k = 1` si `hp ≥ 1,36·p3`) ; **tronçon en T** en traction ; poinçonnement (`t·fu` minimal platine ou poteau) | MSB P5 §2.2 ; §3.10.2 ; §6.2.4 Tab. 6.2 ; Tab. 3.4 |
| Poutre portée | cisaillement sur `hp` (`Av = hp·tw`) ; traction au droit de la platine ; grugeage (facultatif) : cisaillement, flexion + V (bras `tp + c`), stabilité `c` et `dc` | EC3-1-1 §6.2.6 ; MSB P5 §2.2.1, §4.2.4 / §4.2.5 |
| **Poteau porteur** | **pression diamétrale** de l'aile (a) ou de l'âme (b), sans pince d'extrémité ; **cas (b)** : cisaillement local brut et net (`VEd/2`) ; **cas (a) sous traction** : tronçon en T de l'aile (Tab. 6.4) et âme en traction transversale (ω = 1) | Tab. 3.4 ; SCI P358 ; §6.2.4 + Tab. 6.4 ; §6.2.6.3 |
| Soudure | double cordon sur `hp` sous `qz = V/2hp` et `qx = NEd,ELU/2hp` ; **tying** (`fu/(√3·βw·γMu)`) ; gorge min. ; pleine résistance recommandée (alerte) | §4.5.3.3 ; §4.5.2(2) ; MSB P5 §2.2, §2.3 |

Le moment que l'attache apporte au poteau, `VEd·e_col` (`e_col = hP/2 + tp` en
cas (a), `tp` en cas (b)), est **affiché** dans « Paramètres retenus », dans
l'export texte et au bloc HYPOTHÈSES de la note : il appartient au calcul
d'ÉLÉMENT du poteau (EC3-1-1 §6.2 et §6.3), l'outil ne le vérifie pas.

Limites écrites (onglet Vérifications → « Périmètre de la vérification », et
onglet Méthode) : les **neuf** du §8.4 de l'étude, dont la flexion hors plan
de l'âme du poteau (cas b seulement — huit limites en cas a).

## Annexe nationale belge

Défauts : NBN EN 1993-1-1 ANB — γM0 = 1,00 ; valeurs recommandées de
l'EN 1993-1-8 (γM2 = 1,25 ; γM3 = 1,25 ; γM3,ser = 1,1). **Aucune
vérification de cet assemblage n'utilise γM1** (pas d'instabilité de plat) :
il n'est pas demandé. `γMu = 1,1` (tying) est la valeur des guides
(MSB P5 §2.3, SCI P358) : l'EN 1993-1-8 ne la fixe pas et l'ANB n'a pas été
consultée — **à confirmer** (point P-9). Le texte NBN, payant, n'a pas pu être
consulté ; toutes ces valeurs sont modifiables dans les paramètres avancés.

## Validation

- **BM1 — HÉRITAGE** : l'exemple publié (MSB P5 §2.4, *Partial depth end
  plate*, VEd = 550 kN et tying 275 kN) rejoué avec `appui = Aile du poteau`
  et un poteau PERSO d'aile 15 mm S275 : les **neuf** valeurs publiées sortent
  inchangées — âme sur `hp` 614 (+0,072 %) ; `Fv,Rd` 94 (+0,085 %) ; pression
  diamétrale 107 (−0,948 %) et 142 (−0,172 %) ; groupe 902 (+0,129 %) ;
  platine 1 182 (−0,009 %) ; **tying** platine 493 (−0,232 %) et âme portée
  1 513 (−0,012 %) ; cordon de pleine résistance 0,48 (+0,101 %). **Pire écart
  0,95 %**, tous expliqués (arrondis de la source : `αb = 0,61`,
  `Fv,Rd = 94`, `m = 59`). C'est la preuve que rien du côté platine ne dépend
  du support. Le tronçon en T de l'exemple (`m = 59,164`, `Σleff = 430`, modes
  **491,86 / 783,20 / 1 924,36 kN**) est l'**ancre de la formule que l'aile du
  poteau réutilise**.
- **BM2 — cas (a), le défaut du module** : IPE 400 S355 sur HEB 300 S355,
  platine 290 × 220 × 10, 4 × 2 M20 8.8, `p3 = 140`, `VEd = 200 kN`.
  `Fv,Ed = 25 kN` ; platine `Fb,Rd` 123,636 / 165,364 kN → groupe
  **602,112 kN** (η 0,332) ; platine 936,035 / 951,658 / 945,799 kN ; âme
  portée `Av = 2 494 mm²` → **511,169 kN**, η **0,391** (dimensionnante) ;
  **aile du poteau** `αb = 0,810606`, `k1 = 2,5`, `t = 19` →
  **`Fb,Rd = 314,191 kN/boulon`**, `VRd = 2 513,53 kN` (η 0,0796) ; cordons
  `qz = 344,83 N/mm` ; gorge de pleine résistance 4,762 mm. **VÉRIFIÉ.**
- **BM3 — cas (a) TYING** (`NEd = 250 kN`, `γMu = 1,1`) : le tableau du §7.3
  de l'étude, reproduit à 1e-9 —

  | | platine (tp 10) | **aile du poteau** (tfP 19) |
  |---|---|---|
  | `m` | 60,043 mm | **42,900 mm** |
  | `n` | 40,000 mm | **53,625 mm** |
  | `Σleff` | 290 mm (= hp) | **280 mm (= n1·p1, Tab. 6.4)** |
  | `Mpl,1` | 3,3614 kNm | **11,7161 kNm** |
  | `FT,1 / FT,2 / FT,3` | **261,330** / 580,141 / 1 282,91 | 1 296,99 / **955,485** / 1 282,91 |
  | `FT,Rd` | **261,330 kN** (η 0,957) | **955,485 kN** (η 0,262) |

  → **l'aile du poteau est 3,656 × plus forte que la platine** : avec une aile
  laminée épaisse le composant neuf ne gouverne pas — il gouvernerait avec une
  aile mince ou un `p3` plus grand, et **c'est pour cela que la ligne existe**.
  Autres composantes : `Ft,Ed = 31,25 kN` ; poinçonnement **277,911** (platine)
  et 528,031 kN (aile) ; **âme du poteau en traction transversale 1 428,0 kN**
  (ω = 1, `beff = 280`) ; âme portée 1 156,31 kN ; cordons `qx = 431,03` pour
  1 487,11 N/mm. **Dimensionnante en tying : la platine, η = 0,957.**
- **BM4 — cas (b), dans l'âme** : platine **290 × 190 × 10**, `p3 = 110`.
  L'écrou dégage le congé (73,5 ≤ 104, **marge 30,5 mm**) ; `αb = 0,810606`,
  `k1 = 2,5`, `t = 11` → **`Fb,Rd = 181,900 kN`**, `VRd = 1 455,20 kN` ;
  cisaillement local **brut 653,820 kN** (η 0,153) et **net 523,412 kN**
  (η 0,191) ; poinçonnement de l'âme en tying **305,702 kN** ; recoupe non
  nécessaire pour un IPE 400. Flexion hors plan de l'âme : **NON VÉRIFIÉE**,
  limite écrite.
- **Calculs manuels indépendants** (VAL-A, VAL-B, VAL-C) : le tronçon en T de
  l'exemple publié, celui de l'**aile du poteau** avec l'âme en traction
  transversale, et la géométrie + le cisaillement local du cas (b). Chaque
  valeur de référence de `benchmark.py` est une **expression littérale**
  indépendante, jamais une constante recopiée.
- **Régression figée** : `tests/ref_pac.json` — **19 cas, 190 vérifications**
  (Ed, Rd, η) + 80 scalaires par cas + alertes + verdict, à 1e-9 près. Les cas
  couvrent les deux appuis, le gauge refusé, la platine qui ne tient pas entre
  les congés, la recoupe suffisante et insuffisante, `p3` au-delà du maximum du
  Tableau 3.3, le grugeage double, `NEd` ELU, le tying (aile et âme), les
  catégories B et C, le trou surdimensionné, un profil PERSO, l'exemple publié,
  `hp < 1,36·p3`, un cas NON VÉRIFIÉ et le prédimensionnement.
- Suites : `tests/test_platine_about_poteau_moteur.py` (96 contrôles) et
  `tests/test_platine_about_poteau_ecran.py` (98 contrôles, dont le contrat du
  PROJET : registre, `acier/projet.py:description` et un rapport de projet
  MIXTE — un `pab_` + deux `pac_`, huit pages).

## Écarts assumés (v1.0)

- **La recoupe minimale s'écrit `(bP − twP)/2 + j − tp`**, et non
  `… + j − gh` comme au §3.2.2 de l'étude : une platine d'about n'a pas de clé
  `gh`, l'about de la poutre est à `tp` de la face d'appui. Le nombre tombe sur
  **139,5 mm** pour un HEB 300 avec `tp = 10` et `j = 5` — l'étude imprime
  144,5 mm, valeur qui correspond à `j = 10` alors qu'elle déclare `j = 5` par
  défaut (même écart que le module `plat_ame_poteau`). **Le module applique la
  FORMULE** et le dit ici plutôt que d'ajuster un réglage pour coller au
  nombre ; aucune résistance n'en dépend.
- **La limite du cas (b) ne cite pas `qx = 3·MS/hp²`** (texte du §8.4, item 7
  de l'étude) : le groupe est CENTRÉ, il n'y a pas de moment d'excentricité
  dans cette attache — seule la force de chaînage sollicite l'âme hors de son
  plan. La limite est écrite pour ce qui existe ici, et son raisonnement (le
  tronçon en T suppose une aile encastrée sur une âme centrale) est rappelé.
- **`rec_S` et `j_lat` sont des cotes du DESSIN, pas des champs des paramètres
  avancés** : l'étude §10.6.1 les plaçait dans « Avancés › GÉOMÉTRIE », mais la
  règle P6 de la charte — que l'étude impose elle-même au §12 — veut que la
  géométrie se règle sur le dessin. Elles sont donc dans `CLE_PAR_COTE` (vue en
  plan), dans la fenêtre « Poutre portée » et dans le panneau Cotes. Même
  raison pour `d_nt`, `d_nb` et `l_n`.
- **`bp ≤ bP` (cas a) est une alerte d'INFORMATION, pas un blocage** : une
  platine plus large que l'aile se fabrique, elle est seulement inhabituelle —
  alors que `bp > hP − 2tfP − 2rP − 2j` en cas (b) est une impossibilité
  physique (elle ne rentre pas), donc bloquante. La règle de l'étude §4.3,
  point 4, est donc gardée comme recommandation nommée.
- **En cas (b), l'élévation regarde l'attache depuis la face de l'âme** :
  l'aile proche est retirée du dessin (la platine et la poutre se lisent en
  entier sur la bande du poteau). C'est la convention du GABARIT
  `plat_ame_poteau`, gardée pour que les trois modules de la famille se
  dessinent pareil ; la vue en plan et la coupe A-A portent la géométrie
  exacte (poutre entre les ailes, largeur droite entre congés).
- **Le jeu latéral `j` se cote contre la FACE INTÉRIEURE de l'aile** : c'est
  la valeur SAISIE, celle qui commande le passage `hP − 2tfP − 2j`, et non
  l'écart résiduel entre la semelle et l'aile (convention du gabarit).
- Pas de rapport détaillé 3 pages : la note 2 pages (synthèse + plan de
  principe) et l'export texte couvrent la livraison.
- Aucun numéro de sous-clause n'est inventé : là où il n'est pas établi, la
  référence reste au chapitre (« MSB P5 §2.2 »).
- Le nom de disposition de la page 2 (`derniere_disposition`) est celui de la
  famille (« … plan et droite à droite ») : la troisième vue de ce module
  s'appelle COUPE A-A. Texte hérité, non repris pour ne pas toucher au module
  poutre–poutre.

## Briques écrites LOCALEMENT (à verser au socle)

- `schemas.visserie()` et `schemas._boulon_profil()` : les proportions de
  dessin de la boulonnerie (tête, écrou, rondelles) et le boulon vu de profil.
  Ils sont **copiés de `platine_about/schemas.py`** (le kit de dessin partagé,
  `doubles_cornieres/schemas.py`, ne les expose pas) et adaptés : l'écrou est
  du côté INTÉRIEUR du poteau, à `xf − t_appui`.
- `tools/captures_assemblages.js` : sa table `PREFIXE` ne connaît pas
  `platine_about_poteau → pac_`. Le script VERSIONNÉ n'a pas été modifié (il
  est gelé) ; les captures ont été prises avec une COPIE de scratchpad qui
  ajoute cette ligne (et deux états, « ame »/« ameok » et « tying »). **Une
  ligne à verser au socle.**
