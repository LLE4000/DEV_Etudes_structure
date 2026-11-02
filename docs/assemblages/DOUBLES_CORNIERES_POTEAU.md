# Assemblage poutre–poteau par doubles cornières d'âme — v1.0

Deuxième assemblage de la famille **Poutre – Poteau** (lot 3 du plan
`POUTRE_POTEAU.md` §12), écrit le 02/11/2026 sur le GABARIT du lot 2
(`plat_ame_poteau`) : même découpage en treize fichiers, même clé `appui` en
segmented_control d'en-tête, mêmes trois vues, même sous-classement de la
note, même charte (`ERGONOMIE.md`).

C'est le module le plus lourd de la famille : **88 clés** (préfixe `dcc_`) et
**40 vérifications nommées**, parce qu'il porte DEUX groupes de boulons (S sur
l'âme portée, P sur le poteau), chacun boulonné ou soudé, et les DEUX
composants normatifs neufs du poteau — le tronçon en T de l'AILE NON RAIDIE
et l'âme en traction transversale.

## Ce qui change par rapport au grand frère `doubles_cornieres`

Le côté CORNIÈRES, BOULONS et ÂME PORTÉE est repris **sans aucune modification
de formule** — et c'est mesuré : le benchmark BM1 rejoue l'exemple publié du
guide sur un POTEAU et retrouve les **seize** valeurs publiées aux écarts
exacts du grand frère (pire écart 0,76 %), BM1' retrouvant en outre l'ANCRE
publiée du tronçon en T (modes 1 / 2 / 3). Ce qui change :

| | `doubles_cornieres` (poutre–poutre) | `doubles_cornieres_poteau` |
|---|---|---|
| support | âme d'une poutre porteuse | **aile** (cas a) ou **âme** (cas b) d'un poteau en I/H |
| clé centrale | — | **`appui`** : `APPUI_AILE` / `APPUI_AME`, en segmented_control d'en-tête |
| rotule | face de l'âme porteuse | face d'appui du poteau : `X = hP/2` (a) ou `twP/2` (b) |
| côté support, ELU | pression diamétrale de l'âme + cisaillement local | **aile** (`t = tfP`) ou **âme** (`t = twP`) ; cisaillement local **cas (b) seulement** |
| côté support, boulons tendus | rien | **tronçon en T de l'AILE non raidie** (Tab. 6.4) + **âme en traction transversale** (§6.2.6.3, ω = 1) + **poinçonnement** — cas (a) |
| bornes du cisaillement local | `et = min(zt ; 5d)`, `eb = min(zb ; p3/2 ; 5d)` | **`et = 5d`**, `eb = min(p3/2 ; 5d)` — un poteau n'a pas de face horizontale |
| poinçonnement | `min(tc·fu,C ; twP·fu,P)` dans `bt_P` | DEUX lignes nommées : `bt_P` (cornière) et **`ppP`** (poteau) |
| décalage `d_top` (Δz) | oui | **retirée** — un poteau n'a pas de dessus ; le dessin trace un tronçon rompu |
| grugeage | 50 mm par défaut | **0 par défaut** (rien n'obstrue au-dessus ni en dessous) ; tout le bloc s'active à l'identique dès qu'on grugue |
| tying | absent | **présent** : `nat_N`, `γMu`, et les boulons du groupe S en **cisaillement** (la force de chaînage est dans le plan de l'âme portée) |
| clés neuves | — | `appui`, `nat_N`, `g_Mu`, `rec_S` (recoupe des semelles), `j_lat` (jeu latéral) |
| alertes retirées | — | `P_web`, `dnt_min`, `dnb_min`, `ln_min`, `gap_neg` (toutes fausses pour un poteau) |
| alertes neuves | — | `p3_congeP`, `p3_libreP`, `bA_congeP`, `aile_poutre` (bloquantes), `ame_hp`, `n2P_T`, `N_tying` |
| clés de session | 84 (`asm_`) | **88** (`dcc_`) |

**Les briques de calcul communes ne sont pas recopiées** : pression
diamétrale, groupe excentré, table des boulons, poutre grugée, tronçon en T et
les longueurs efficaces de l'aile de poteau non raidie viennent de
`acier/assemblages/mecanique.py` (lot 1), et la suite du moteur vérifie
qu'elles rendent bien les valeurs du moteur (§6 de
`test_doubles_cornieres_poteau_moteur`). La page 1 de la note est une
**sous-classe** de `plat_ame.note.Note`, la page 2 une sous-classe de
`doubles_cornieres.note.PlanPrincipe`, et `formules.py` importe la brique de
la famille.

## Les deux cas d'appui — le résultat central

> **Cas (a), sur l'AILE** : les deux files sont symétriques par construction
> (`±p3/2`), la résultante passe par l'âme du poteau, et l'attache vit DANS LE
> PLAN de cette âme. `VEd` et `VEd·z` y cheminent en plan : sous l'effort
> tranchant seul, le côté poteau n'apporte que la **pression diamétrale** de
> l'aile (ou les cordons si les ailes A sont soudées). Deux particularités :
> aucune pince d'extrémité verticale (`αb = min(p1/(3d0) − ¼ ; fub/fu ; 1)`)
> et un seul bord libre, le **bout d'aile** `e = (bP − p3)/2`. La flexion
> locale de l'aile n'apparaît QUE sous boulons tendus : **tronçon en T** de
> l'aile non raidie et **âme en traction transversale**.
>
> **Cas (b), dans l'ÂME** : l'attache est perpendiculaire à l'âme. `VEd` reste
> en plan (cisaillement local d'un élément support chargé d'un seul côté,
> `VEd/2 ≤ Av·fy,P/(√3·γM0)` avec `Av = twP·(5d + (n1 − 1)p1 + min(p3/2 ; 5d))`,
> SCI P358), mais toute traction sollicite l'âme **hors de son plan** — ce que
> l'EN 1993-1-8 ne couvre pas pour une âme non raidie. **Déclaré, pas
> transposé.** La variante « about au-delà des bouts d'aile » (b1 du plat
> d'âme) est **sans objet** : l'aile B d'une cornière devrait traverser tout le
> demi-poteau.

## Les quatre conditions GÉOMÉTRIQUES du poteau

Mesurées sur le poteau de référence HEB 300 (`h 300, b 300, tw 11, tf 19,
r 27`), une âme portée de 8,6 mm (IPE 400) et des M20 (`dw 37`) :

| condition | formule | HEB 300 | alerte |
|---|---|---|---|
| l'écrou dégage le congé du poteau (cas a) | `p3 ≥ twP + 2rP + dw` | **102 mm** | `p3_congeP`, bloquante |
| l'écrou dégage le congé (cas b) | `p3/2 + dw/2 ≤ (hP − 2tfP)/2 − rP` | marge 31,2 mm pour `p3 = 108,6` | `p3_libreP`, bloquante |
| le bout de l'aile A dégage le congé (cas b) | `tw,S/2 + bA ≤ (hP − 2tfP)/2 − rP` | **`bA ≤ 99,7 mm`** | `bA_congeP`, bloquante |
| la semelle passe entre les ailes (cas b) | `bS ≤ hP − 2tfP − 2j` | **252 mm** | `aile_poutre`, bloquante |

Conséquences pratiques, et ce sont les résultats utiles du module :

- le `p3 = 90 mm` usuel des ÂMES de poutre **NE PASSE PAS** sur l'aile d'un
  HEB 300 (26,5 mm disponibles pour 32,5 nécessaires) : c'est la raison des
  gauges usuels d'aile de poteau, **90 / 100 / 120 / 140 mm** selon la série ;
- en cas (b), une **L100 ne passe pas** (`4,3 + 100 = 104,3 > 104`), une
  **L90×90×9 passe** (`94,3`) — l'alerte le dit avec la longueur maximale ;
- un IPE 400 (`bS = 180`) passe entre les ailes d'un HEB 300 sans recoupe ;
  une HEA 300 (`bS = 300`) l'exige (`rc ≥ 139,5 mm` avec `gh = 10`, `j = 5`).
  **La recoupe en LARGEUR n'est pas un grugeage** : les règles de stabilité
  locale de MSB P5 §4.2.5 ne sont pas activées par elle (point tranché P-3).

Le gauge est en outre **encadré par le haut** : `p3 ≤ min(14·tc ; 200 mm)`
(Tableau 3.3, `t` = la plus mince des pièces extérieures, c'est-à-dire l'aile A
de la cornière). Une cornière mince ferme donc la fenêtre — le
prédimensionnement relève l'épaisseur jusqu'à ce qu'elle existe, et dit par un
message quand aucune ne tient.

## Modèle et vérifications (40 lignes, §8.2 de l'étude)

Rotule à la face d'appui du poteau ; le groupe S reprend `VEd` et le moment
d'excentricité `MS = VEd·z + |MEd|` (répartition élastique, inertie polaire) ;
le groupe P reste **centré**, la traction parasite de rotation étant couverte
par le facteur **0,8** sur `Fv,Rd`.

| Élément | Vérifications | Sources |
|---|---|---|
| Boulons | groupe S en double cisaillement (groupe excentré) ; groupe P en simple cisaillement (`k_rot`) ; traction et interaction V + T du groupe P ; glissement (cat. B/C) ; **tying : groupe S en cisaillement** | Tab. 3.4 ; MSB P5 §4.2.1.1 et §4.2.1.2 ; §3.9.1 et §3.9.2 ; MSB P5 §2.3 |
| Cornières | pression diamétrale des ailes A et B (ver/hor, interaction) ; cisaillement brut (·1/1,27) / net / bloc ; flexion des ailes B dans leur plan ; flexion des ailes A entre talon et file ; **tronçon en T** des ailes A ; traction nette et bloc en traction des ailes B | Tab. 3.4 ; MSB P5 §4.2.2, §4.2.2.1, §4.2.2.2 ; §3.10.2(3) et (2) ; §6.2.4 Tab. 6.2 (méthode 2) + MSB P5 §4.3.1.1 ; EC3-1-1 §6.2.3(2), §6.2.5 |
| Poutre portée | pression diamétrale ; brut / net / bloc ; grugeage (flexion + V, 2ᵉ file, stabilité `c` et `dc`) ; traction nette (hauteur participante `hc`) ; bloc en traction | Tab. 3.4 ; MSB P5 §4.2.3.1, §4.2.3.2, §4.2.4, §4.2.5, §4.3.2.2, §4.3.2.3 |
| **Poteau porteur** | pression diamétrale de l'**aile** (a) ou de l'**âme** (b) ; cisaillement local brut et net de l'âme — **cas (b)** ; **tronçon en T de l'aile non raidie** et **âme en traction transversale** (ω = 1) — **cas (a)**, sous boulons tendus ; poinçonnement | Tab. 3.4 ; SCI P358 (élément support chargé d'un seul côté) ; §6.2.4 Tab. 6.2 + **Tab. 6.4** ; §6.2.6.3 |
| Cordons | groupe de cordons excentré (cordon vertical `hc` + retours `ℓh`), `fu = min(fu cornière ; fu support)` ; gorge et longueur minimales ; **tying** | EN 1993-1-8 §4.5.3.3 ; §4.5.2(2) ; §4.5.1(2) ; MSB P5 §2.3 |

Le moment que l'attache apporte au poteau, `VEd·e_col` (`e_col = hP/2 + z` en
cas (a), `z` en cas (b)), est **affiché** dans « Paramètres retenus », dans
l'export texte et au bloc HYPOTHÈSES de la note : il appartient au calcul
d'ÉLÉMENT du poteau (EC3-1-1 §6.2 et §6.3), l'outil ne le vérifie pas.

Limites écrites (onglet Vérifications → « Périmètre de la vérification », et
onglet Méthode) : les **neuf** du §8.4 de l'étude, dont la flexion hors plan de
l'âme du poteau (cas b seulement — huit limites en cas a).

## Annexe nationale belge

Défauts : NBN EN 1993-1-1 ANB — γM0 = 1,00, γM2 = 1,25. `γMu = 1,1` (tying) est
la valeur des guides (MSB P5 §2.3, SCI P358) : l'EN 1993-1-8 ne la fixe pas et
l'ANB n'a pas été consultée — **à confirmer** (point P-9). Sources croisées :
SCIA « Theory NA EN 1993 » et Bentley STAAD « Belgian NA to EC3 » ; le texte
NBN (payant) n'a pas pu être consulté depuis cet environnement. Toutes ces
valeurs sont modifiables dans les paramètres avancés. Le benchmark rejoue
l'exemple publié avec les coefficients de sa source.

## Validation

- **BM1 — HÉRITAGE** : l'exemple publié (MSB P5 §4.4, *Angle Web Cleats*,
  VEd = 450 kN, 2 L90×90×10, 6 + 6 M20, `hc = 430`) rejoué avec
  `appui = Aile du poteau` et un poteau PERSO d'aile 15 mm S275 : les **seize**
  valeurs publiées sortent inchangées — `Fv,Rd` 94 (+0,085 %) ; groupe S
  962 (+0,087 %) ; cornières 105 / 105 (−0,722 %) et 1 075 (−0,759 %) ; âme
  125 (+0,385 %) / 94 (−0,193 %) / 583 (−0,085 %) ; support 902 (+0,129 %) ;
  cornières 1 076 (−0,080 %) / 1 184 (−0,025 %) / 954 / 954 (−0,005 %) ; âme
  953 (+0,456 %) / 956 (+0,587 %) / 501 (−0,055 %). **Pire écart 0,76 %**, tous
  expliqués (arrondis de la source, aire catalogue). C'est la preuve que rien
  du côté cornières ne dépend du support.
- **BM1' — l'ANCRE du tronçon en T** (p. 5-78/79) : `Ft,Rd,u` 160 → 160,364 ;
  `Σleff` 430 → 430,0 mm ; `Mpl` 4,2 → 4,2023 kNm ; modes **696 → 691,25**,
  **1 190 → 1 188,81**, **1 920 → 1 924,36 kN**. C'est la formule MÊME que
  l'aile du poteau réutilise : elle est donc ancrée sur un exemple publié.
- **BM2 — cas (a), le défaut du module** : IPE 400 S355 sur HEB 300 S355,
  2 L100×100×10, `hc = 290`, 4 + 4 M20 8.8, `gA = 50`. `p3 = 108,6 mm` (≥ 102,
  **de justesse**) ; pince au bout d'aile 95,7 ; `z = 50` ; `MS = 10,0 kNm` ;
  `Ip = 24 500 mm²` ; `F_S = 65,854 kN` → η groupe S 0,350 ; η groupe P 0,332 ;
  cornières 936,04 / 951,66 / 868,28 kN ; flexion des ailes B 5,0 / 49,76 kNm ;
  âme portée η **0,5349** (dimensionnante) ; **aile du poteau
  `Fb,Rd = 314,191 kN` → η 0,0796**. **VÉRIFIÉ.**
- **BM3 — cas (a) TYING, `NEd = 300 kN`** : les deux tronçons en T côte à
  côte — ailes A (`m = 30,400`, `n = 38,000`, `Σleff = 290`,
  `Mpl,1 = 3,3614 kNm`, modes **572,015** / 811,013 / 1 282,91) et aile du
  poteau (`m = 27,200`, `n = 34,000`, `Σleff = 280` = `n1·p1` par le
  Tableau 6.4, `Mpl,1 = 11,7161 kNm`, modes 2 314,03 / **1 095,61** /
  1 282,91). **Rapport 1,915 : les CORNIÈRES gouvernent** — l'aile laminée
  épaisse ne gouverne pas, et c'est pour cela que la ligne existe (elle
  gouvernerait avec une aile mince ou un `p3` plus grand). Âme du poteau en
  traction transversale **1 428,0 kN** ; poinçonnement 277,911 (cornière) /
  528,031 kN (aile) ; ailes B en traction nette 1 685,78 ; âme portée 724,886.
- **BM4 — cas (b) ÂME** : 2 L90×90×9 (une L100 est refusée) ; largeur droite
  208, demi-largeur 104, aile A maximale **99,7 mm** ; `gA` minimal du congé de
  la cornière **31,0 mm** ; **`Fb,Rd = 181,900 kN`** sur l'âme (η 0,1374) ;
  `et = 100`, `eb = 54,3`, `Av = 4 007,3 mm²` ; cisaillement local **brut
  821,334** / **net 715,934 kN** (η 0,122 / 0,140) ; poinçonnement de l'âme
  305,702 kN. **VÉRIFIÉ.**
- **Calculs manuels indépendants** (VAL-A, VAL-B, VAL-C, 50 valeurs) : les deux
  tronçons en T et la force de chaînage, le modèle du cas (a) par défaut avec
  la pression diamétrale de l'aile, et toute la géométrie du cas (b). Chaque
  valeur de référence du fichier `benchmark.py` est une **expression
  littérale** indépendante, jamais une constante recopiée ; les modes 1 / 2 / 3
  y sont réécrits in extenso, hors moteur.
- **Régression figée** : `tests/ref_dcc.json` — **25 cas, 473 vérifications**
  (Ed, Rd, η) + 1 875 scalaires + alertes + verdict, à 1e-9 près. Les cas
  couvrent les DEUX appuis × les DEUX fixations (dont soudée des deux côtés),
  la L100 refusée en cas (b), la recoupe suffisante et insuffisante, le
  grugeage double, deux files par groupe, `NEd` ELU, le tying (aile et âme),
  les catégories B et C, le trou surdimensionné, le gauge insuffisant, un
  profil et une cornière PERSO, le prédimensionnement, une configuration
  refusée et les deux cas de l'exemple publié.
- Suites : `tests/test_doubles_cornieres_poteau_moteur.py` (**98 contrôles**) et
  `tests/test_doubles_cornieres_poteau_ecran.py` (**98 contrôles**, dont le
  contrat du PROJET : registre, `acier/projet.py:description` et un rapport de
  projet MIXTE — un `asm_` + deux `dcc_`, huit pages).

## Écarts assumés (v1.0)

- **La ligne `bvT_S` est AJOUTÉE à la liste du §8.2.** Cette liste ne portait
  pas « boulons du groupe S en cisaillement sous la force de chaînage », que le
  §7.2 de la même étude demande pourtant (« boulons en CISAILLEMENT : OUI pour
  le groupe S »), et que le §7.5 n'attribue qu'au plat d'âme. Laisser le groupe
  S sans vérification en tying serait un **défaut non conservatif** : la force
  de chaînage est dans le plan de l'âme portée, elle cisaille ces boulons.
  L'outil l'écrit (`NEd/n ≤ 2·αv·fub·A/γMu`), l'onglet Méthode le dit, et un
  contrôle de la suite du moteur le fige.
- **Le poinçonnement est coupé en DEUX lignes nommées** là où le grand frère
  prenait `min(tc·fu,C ; twP·fu,P)` dans une seule : `bt_P` garde la cornière,
  **`ppP`** (ligne 36 du §8.2) prend le poteau. C'est strictement équivalent en
  sévérité (un min de deux = deux vérifications), et chaque pièce est nommée.
- **Le gauge `p3` est aussi plafonné par le Tableau 3.3** (`p ≤ min(14t ; 200)`,
  `t` = aile A de la cornière), ce que l'étude n'écrit pas : avec une cornière
  de 8 mm le gauge ne peut pas dépasser 112 mm, alors que le congé d'un HEB 300
  en exige 102 (109 avec des M24). La fenêtre est donc étroite, et c'est le
  prédimensionnement qui le découvre : il relève l'épaisseur de la cornière
  jusqu'à ce qu'elle existe, et le dit par un message quand aucune ne tient.
  Le Tableau 3.3 du module porte la ligne `P_p3` (le gauge) et la ligne `P_eP`
  (la pince au bout d'aile, cas a).
- **Le tronçon en T de l'AILE DU POTEAU est écrit pour UNE file par cornière.**
  `Σleff` et `m` viennent de la file à `±p3/2` (la plus proche de l'âme) et
  `ΣFt,Rd = 2·n1P·Ft,Rd`, comme l'étude §11.4 BM3 le calcule. Avec deux files
  par cornière, la seconde — plus éloignée — n'est pas modélisée : une alerte
  d'information (`n2P_T`) le dit dès que des boulons sont tendus en cas (a).
- **La variante b1 du plat d'âme (about au-delà des bouts d'aile) n'est pas
  offerte** : l'étude §6.2 la déclare « sans objet (b1 impossible) » pour les
  cornières. Le moteur garde néanmoins le scalaire `gh_b1` (fait géométrique) :
  si l'about est au-delà des bouts d'aile, l'alerte `aile_poutre` ne se
  déclenche pas à tort — mais l'aile B, trop courte, est alors refusée par
  `S_e2`.
- **`rec_S` et `j_lat` sont des cotes du DESSIN, pas des champs des paramètres
  avancés** : l'étude §10.6.1 les plaçait dans « Avancés › GÉOMÉTRIE », mais la
  règle P6 de la charte — que l'étude impose elle-même au §12 — veut que la
  géométrie se règle sur le dessin. Elles sont donc dans `CLE_PAR_COTE` (vue en
  plan), dans la fenêtre « Poutre portée » et dans le panneau Cotes. Même
  raison pour `d_nt`, `d_nb`, `l_n`, `gA_u`, `a_P`, `a_S`, `lh_P` et `lh_S`.
- **Répartition des cotes sur la planche de la page 2** (chacune une seule
  fois, dans les deux cas d'appui) : l'ÉLÉVATION porte `hc`, `zc`, `e1`, `p1`,
  `e2,b`, `p2`, `gh`, `c`, `dc` et la largeur de la bande du poteau (`hP` en
  cas a / `bP` en cas b) ; la VUE EN PLAN porte `gA`, `p2 (P)`, `p3`, `bA`,
  `bB`, `tc`, `tw`, `e_col`, `twP`, `tfP`, la pince au bout d'aile (cas a) ou
  la largeur droite entre congés `ℓ` (cas b), la recoupe `rc` et le jeu `j`, et
  l'AUTRE largeur du poteau ; la COUPE A-A porte `bS` et montre les deux ailes
  A en vraie grandeur avec le perçage du groupe P.
- **`waS` / `waP` (gorge minimale) peuvent être « dimensionnantes »** quand une
  fixation est soudée et l'attache peu chargée (`Ed = 3 mm` pour `Rd = a`) :
  comportement **identique au grand frère** poutre–poutre, vérifié en relançant
  son moteur. Le mécanisme `DETAIL_SEUL` du module platine d'about n'est pas
  porté ici (il aurait fallu toucher la synthèse de la famille).
- **Briques écrites LOCALEMENT, à verser au socle** : aucune. Le module
  n'appelle que `mecanique.py`, `carte.py`, `ui_commun.py`, `note_kit.py`, le
  kit de dessin de `doubles_cornieres.schemas` et les deux classes de note de
  la famille — tous inchangés.
- Aucun numéro de sous-clause n'est inventé : là où il n'est pas établi, la
  référence reste au chapitre (« MSB P5 §4.2 »).
- Le nom de disposition de la page 2 (`derniere_disposition`) est celui de la
  famille (« … plan et droite à droite ») : la troisième vue de ce module
  s'appelle COUPE A-A. Texte hérité, non repris pour ne pas toucher au module
  poutre–poutre.
