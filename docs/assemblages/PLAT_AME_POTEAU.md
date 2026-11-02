# Assemblage poutre–poteau par plat d'âme soudé (fin plate) — v1.0

Premier assemblage de la famille **Poutre – Poteau** (lot 2 du plan
`POUTRE_POTEAU.md` §12), écrit le 01/11/2026 avec la même identité graphique,
la même ergonomie (charte `ERGONOMIE.md`) et la même qualité de note que les
trois modules poutre–poutre. **C'est le GABARIT des deux autres attaches
poutre–poteau** (doubles cornières, platine d'about) : son découpage en
treize fichiers, sa clé `appui`, ses trois vues et son benchmark sont le
modèle à reprendre.

## Ce qui change par rapport au grand frère `plat_ame`

Le côté PIÈCE D'ATTACHE, BOULONS et ÂME PORTÉE est repris **sans aucune
modification de formule** — et c'est mesuré : le benchmark BM1 rejoue
l'exemple publié du guide sur un POTEAU et retrouve les **dix** valeurs
publiées aux écarts exacts du grand frère (pire écart 0,76 %). Ce qui change :

| | `plat_ame` (poutre–poutre) | `plat_ame_poteau` |
|---|---|---|
| support | âme d'une poutre porteuse | **aile** (cas a) ou **âme** (cas b) d'un poteau en I/H |
| clé centrale | — | **`appui`** : `APPUI_AILE` / `APPUI_AME`, en segmented_control d'en-tête |
| rotule | face de l'âme porteuse | face d'appui du poteau : `X = hP/2` (a) ou `twP/2` (b) |
| cisaillement local du support | toujours | **cas (b) seulement** ; en cas (a) le groupe « Poteau porteur » est **VIDE** |
| décalage `d_top` (Δz) | oui | **retirée** — un poteau n'a pas de dessus ; le dessin trace un tronçon rompu |
| grugeage | 50 mm par défaut | **0 par défaut** (rien n'obstrue au-dessus ni en dessous) ; tout le bloc s'active à l'identique dès qu'on grugue |
| tying | absent | **présent** : `nat_N`, `γMu`, et les boulons en **cisaillement** (la force de chaînage est dans le plan du plat) |
| clés neuves | — | `appui`, `nat_N`, `g_Mu`, `rec_S` (recoupe des semelles), `j_lat` (jeu latéral) |
| alertes retirées | — | `pl_web`, `dnt_min`, `dnb_min`, `ln_min`, `gap_neg` (toutes fausses pour un poteau) |
| alertes neuves | — | `aile_poutre` (bloquante), `b1_long`, `ame_hp`, `N_tying` |
| clés de session | 67 (`fpl_`) | **71** (`fpc_`) |

**Les briques de calcul communes ne sont pas recopiées** : pression
diamétrale, groupe excentré, table des boulons, poutre grugée, déversement du
plat long et gorge de pleine résistance viennent de
`acier/assemblages/mecanique.py` (lot 1), et la suite du moteur vérifie
qu'elles rendent bien les valeurs du moteur (§6 de `test_plat_ame_poteau_moteur`).
La page 1 de la note est une **sous-classe** de `plat_ame.note.Note`, la page 2
une sous-classe de `doubles_cornieres.note.PlanPrincipe`, et `formules.py`
importe la brique de la famille.

## Les deux cas d'appui — le résultat central

> **Cas (a), sur l'AILE** : l'attache est centrée sur l'âme du poteau et vit
> DANS SON PLAN. `VEd` et `VEd·z` y cheminent en plan : aucune vérification
> locale de l'aile n'est demandée par MSB P5 §3.2 ni par SCI P358 au-delà du
> cordon. Le groupe « Poteau porteur » est **vide**, et l'écran le dit dans le
> repli « Périmètre de la vérification ».
>
> **Cas (b), dans l'ÂME** : l'attache est perpendiculaire à l'âme. `VEd` reste
> en plan (cisaillement local d'un élément support chargé d'un seul côté,
> `VEd/2 ≤ twP·hp·fy,P/(√3·γM0)`, SCI P358), mais `qx = 3·MS/hp²` et la force
> de chaînage sollicitent l'âme **hors de son plan** — ce que l'EN 1993-1-8 ne
> couvre pas pour une âme non raidie. **Déclaré, pas transposé.**

Géométrie du cas (b), mesurée sur le poteau de référence HEB 300
(`h 300, b 300, tw 11, tf 19, r 27`) :

| grandeur | formule | HEB 300 |
|---|---|---|
| largeur droite de l'âme entre congés | `hP − 2tfP − 2rP` | **208 mm** |
| demi-largeur droite | `(hP − 2tfP)/2 − rP` | 104 mm |
| passage de la semelle portée | `hP − 2tfP − 2j` (j = 5) | **252 mm** |
| variante b1 (about au-delà des bouts d'aile) | `gh ≥ (bP − twP)/2 + j` | **149,5 mm** |
| recoupe minimale (variante b2) | `(bP − twP)/2 + j − gh` | **139,5 mm** (gh = 10) |

Un IPE 400 (`bS = 180`) passe donc entre les ailes d'un HEB 300 sans recoupe ;
une HEA 300 (`bS = 300`) l'exige. **La recoupe en LARGEUR n'est pas un
grugeage** : les règles de stabilité locale de MSB P5 §4.2.5 ne sont pas
activées par elle (point tranché P-3 de l'étude).

## Modèle et vérifications (25 lignes, §8.1 de l'étude)

Rotule à la face d'appui du poteau ; le groupe de boulons reprend `VEd` et le
moment d'excentricité `MS = VEd·z + |MEd|` (répartition élastique, inertie
polaire).

| Élément | Vérifications | Sources |
|---|---|---|
| Boulons | cisaillement du groupe excentré ; glissement (cat. B/C) ; **tying : boulons en cisaillement** | Tab. 3.4 ; MSB P5 §3.2 ; §3.9 ; MSB P5 §2.3 |
| Plat | pression diamétrale (ver/hor, interaction) ; cisaillement brut (·1/1,27) / net / bloc ; flexion (sans objet si `hp ≥ 2,73·z`) ; DÉVERSEMENT du plat long (`z > tp/0,15`) ; traction nette ; bloc en traction | Tab. 3.4 ; MSB P5 §3.2, §3.2.3 ; §3.10.2(3) et (2) ; SCI P358 + BS 5950-1 Annexe B.2.1 (Table 17), E = 205 000 ; EC3-1-1 §6.2.3(2) |
| Poutre portée | pression diamétrale ; brut / net / bloc (`(n1 − 1)·d0` non grugée, `(n1 − 0,5)·d0` grugée) ; grugeage (flexion + V, stabilité c et dc) ; traction nette ; bloc en traction | Tab. 3.4 ; MSB P5 §3.2.5.1 ; §4.2.4 / §4.2.5 ; §3.10.2(2) |
| **Poteau porteur** | **cas (b) seulement** : cisaillement local de l'âme, `Av = twP·hp`. Cas (a) : rien, et c'est écrit | SCI P358 (élément support chargé d'un seul côté) |
| Soudure | double cordon pleine hauteur sous `qz = V/2hp` et `qx = 3·MS/hp² + |NEd|/2hp` ; **tying** (`fu/(√3·βw·γMu)`) ; gorge min. ; pleine résistance recommandée (alerte) | EN 1993-1-8 §4.5.3.3 ; §4.5.2(2) ; MSB P5 §2.3 |

Le moment que l'attache apporte au poteau, `VEd·e_col` (`e_col = hP/2 + z` en
cas (a), `z` en cas (b)), est **affiché** dans « Paramètres retenus », dans
l'export texte et au bloc HYPOTHÈSES de la note : il appartient au calcul
d'ÉLÉMENT du poteau (EC3-1-1 §6.2 et §6.3), l'outil ne le vérifie pas.

Limites écrites (onglet Vérifications → « Périmètre de la vérification », et
onglet Méthode) : les **neuf** du §8.4 de l'étude, dont la flexion hors plan
de l'âme du poteau (cas b seulement — huit limites en cas a).

## Annexe nationale belge

Défauts : NBN EN 1993-1-1 ANB — γM0 = 1,00, **γM1 = 1,10** (≠ 1,00 recommandé ;
ne sert qu'au déversement du plat long), γM2 = 1,25. `γMu = 1,1` (tying) est la
valeur des guides (MSB P5 §2.3, SCI P358) : l'EN 1993-1-8 ne la fixe pas et
l'ANB n'a pas été consultée — **à confirmer** (point P-9). Sources croisées :
SCIA « Theory NA EN 1993 » et Bentley STAAD « Belgian NA to EC3 » ; le texte
NBN (payant) n'a pas pu être consulté depuis cet environnement. Toutes ces
valeurs sont modifiables dans les paramètres avancés. Le benchmark rejoue
l'exemple publié avec γM1 = 1,0 (comme sa source).

## Validation

- **BM1 — HÉRITAGE** : l'exemple publié (MSB P5 §3.4, fin plate, VEd = 350 kN,
  2 files de M20, z = 80) rejoué avec `appui = Aile du poteau` et un poteau
  PERSO d'aile 15 mm S275 : les **dix** valeurs publiées sortent inchangées —
  `Ip` 107 000 mm² (0,000 %) ; boulons 584 (+0,012 %) ; plat 605 (−0,760 %) /
  450 (+0,013 %) / 497 (−0,096 %) / 483 (+0,067 %) ; âme 624 (−0,051 %) /
  953 (+0,456 %) / 995 (+0,597 %) / 507 (−0,064 %). **Pire écart 0,76 %**, tous
  expliqués (arrondis de la source, aire catalogue). C'est la preuve que rien
  du côté pièce d'attache ne dépend du support.
- **BM2 — cas (a), le défaut du module** : IPE 400 S355 sur HEB 300 S355, plat
  290 × 100 × 10, 4 M20 8.8. `z = 50` ; `MS = 10,0 kNm` ; `Ip = 24 500 mm²` ;
  `F,Ed = 65,854 kN` → **η boulons 0,700** (dimensionnante) ; plat 468,02 /
  475,83 / 434,14 kN ; âme portée η 0,5349 ; cordons 496,14 / 1 308,66 N/mm ;
  gorge de pleine résistance 3,915 mm. **VÉRIFIÉ.**
- **BM3 — cas (b) variante b1** (`gh = 155`) : `z = 195 mm` (× 3,9),
  `MS = 39 kNm`, `F,Ed = 174,461 kN` → **η boulons 1,854** ; plat LONG,
  `λLT = 54,366`, `pb = 276,516 MPa`, `Mb,Rd = 35,235 kNm` → **η 1,107**.
  **NON VÉRIFIÉ** : c'est le résultat utile — la variante b1 coûte cher.
  Âme du poteau : `VRd = 653,820 kN` pour `Ed = 100` → η 0,153.
- **BM3' — le même, redimensionné** (5 × 2 M24, p1 = p2 = 60, plat 320 × 12) :
  `Ip = 81 000` ; `F,Ed = 76,085` → η 0,561 ; `λLT = 51,121` ; `pb = 287,879` ;
  `Mb,Rd = 53,598` → η 0,840. Les deux taux que l'étude annonce reviennent
  sous 1 — **mais l'outil trouve deux défauts de plus** (voir « Écarts »).
- **BM4 — cas (b), about au contact** : `z` retombe à 50 mm, tous les taux de
  BM2 reviennent, et le cisaillement local de l'âme du poteau s'ajoute
  (653,820 kN, η 0,153).
- **Tying** (`NEd = 250 kN`, `γMu = 1,1`) : `Fv,Rd,u = 106,909 kN/boulon`
  (427,636 kN pour le groupe) ; plat 842,891 kN ; âme portée 724,886 kN ;
  cordons `qx = 431,03` pour `Fw,Rd,u = 1 487,11 N/mm`. Côté poteau : **rien de
  plus** en cas (a) — le plat est soudé, la force est dans son plan.
- **Calculs manuels indépendants** (VAL-A, VAL-B, VAL-C) : le déversement du
  cas b1 (Perry-Robertson réécrit hors moteur), le modèle du cas (a) par défaut
  et le tying. Chaque valeur de référence du fichier `benchmark.py` est une
  **expression littérale** indépendante, jamais une constante recopiée.
- **Régression figée** : `tests/ref_fpc.json` — **18 cas, 232 vérifications**
  (Ed, Rd, η) + 32 scalaires + alertes + verdict, à 1e-9 près. Les cas couvrent
  les deux appuis, les trois variantes du cas (b), la recoupe suffisante et
  insuffisante, le grugeage double, deux files, `NEd` ELU, le tying (aile et
  âme), les catégories B et C, le trou surdimensionné, un profil PERSO, le plat
  long, une configuration refusée et le prédimensionnement.
- Suites : `tests/test_plat_ame_poteau_moteur.py` (88 contrôles) et
  `tests/test_plat_ame_poteau_ecran.py` (94 contrôles, dont le contrat du PROJET :
  registre, `acier/projet.py:description` et un rapport de projet MIXTE
  — un `fpl_` + deux `fpc_`, huit pages).

## Écarts assumés (v1.0)

- **L'étude donne `gh ≥ 154,5 mm` (variante b1) et `recoupe ≥ 144,5 mm`
  (variante b2) pour un HEB 300** (§3.2.2 et §11.2). La formule qu'elle écrit,
  `(bP − twP)/2 + j`, avec le jeu latéral de 5 mm qu'elle déclare par défaut,
  donne **149,5** et **139,5 mm** : les deux valeurs imprimées correspondent à
  `j = 10 mm`. Le module applique la FORMULE (149,5 / 139,5 avec `j_lat = 5`) et
  le dit ici plutôt que d'ajuster un réglage pour coller au nombre. Aucune autre
  grandeur n'en dépend (`z`, `MS` et tous les taux de BM3 et BM4 viennent de
  `gh` saisi, non de ce minimum).
- **BM3' (cas b1 redimensionné) n'est pas complet dans l'étude** : l'outil
  trouve deux non-conformités de plus, et il faut qu'il les trouve —
  (1) l'entraxe `p2 = 60 mm` reste sous le minimum `2,4·d0 = 62,4 mm` du
  Tableau 3.3 pour un M24, (2) la pression diamétrale de l'âme portée
  (IPE 400, `tw 8,6`) atteint **η = 1,148**. Les six valeurs que l'étude
  annonce sont reproduites à 1e-9 ; les deux défauts sont inscrits dans le
  benchmark (deux lignes de BM3') et dans la suite du moteur.
- **`rec_S` et `j_lat` sont des cotes du DESSIN, pas des champs des paramètres
  avancés** : l'étude §10.6.1 les plaçait dans « Avancés › GÉOMÉTRIE », mais la
  règle P6 de la charte — que l'étude impose elle-même au §12 — veut que la
  géométrie se règle sur le dessin. Elles sont donc dans `CLE_PAR_COTE`
  (vue en plan), dans la fenêtre « Poutre portée » et dans le panneau Cotes.
  Même raison pour `d_nt`, `d_nb` et `l_n` : les paramètres avancés gardent le
  rayon du grugeage et le maintien au déversement, les profondeurs sont des
  cotes.
- **La gorge `a` est cotée sur l'ÉLÉVATION** (étiquette du symbole EN 22553),
  pas sur la coupe A-A comme le suggère la liste du §10.7 : le symbole de
  soudure est le porteur naturel de cette valeur, et elle n'est posée qu'une
  fois sur la planche. Même logique pour la largeur de la bande du poteau :
  l'ÉLÉVATION porte `hP` en cas (a) / `bP` en cas (b), la VUE EN PLAN porte
  l'autre — chacune une seule fois, dans les deux cas d'appui.
- Les composants du tronçon en T de l'aile du poteau et de l'âme en traction
  transversale (§7.3, §7.4 de l'étude) n'existent PAS dans ce module : le plat
  est soudé, aucun boulon ne traverse le poteau. Ils arrivent avec les deux
  modules boulonnés (lots 3 et 4), avec le gauge `p3 ≥ twP + 2·rP + dw`
  (l'écart `p3_congeP` est calculé et exposé par le moteur, sans alerte ici).
- Pas de rapport détaillé 3 pages : la note 2 pages (synthèse + plan de
  principe) et l'export texte couvrent la livraison.
- Aucun numéro de sous-clause n'est inventé : là où il n'est pas établi, la
  référence reste au chapitre (« MSB P5 §3.2 »).
- Le nom de disposition de la page 2 (`derniere_disposition`) est celui de la
  famille (« … plan et droite à droite ») : la troisième vue de ce module
  s'appelle COUPE A-A. Texte hérité, non repris pour ne pas toucher au module
  poutre–poutre.
