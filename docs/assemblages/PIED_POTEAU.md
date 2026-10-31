# Pied de poteau sur béton — module `acier/assemblages/pied_poteau/platine_beton`

*Lot A (31/10/2026) : le MOTEUR. Il applique la méthode de l'étude
`PIED_POTEAU_ETUDE.md`. Lot B (31/10/2026) : l'ÉCRAN, les dessins, la note
de deux pages et l'intégration au projet d'assemblages — section 6. Lot C :
validation sur exemples publiés, CHS, tiges post-installées. **Lot D
(31/10/2026) : le CALCUL MANUEL de la platine est devenu le mode par défaut
du module** (boulons posés librement, éléments soudés, raidisseurs,
épaisseur requise lue comme sur une note à la main) — `PLATINE_MANUELLE.md` ;
la vérification complète décrite ici reste disponible pour une disposition
régulière (section 9).*

## 1 · Ce que le moteur calcule

Entrée (`entrees.py`) : poteau (I/H de la base de profilés, ou RHS/SHS par ses
dimensions), platine bp × lp × tp centrée, tiges en grille (2 files × 2 à 4
rangées, entraxes wx, wy), tige (diamètre, classe, tête ou crosse, hef, dh),
béton (classe, fissuré, αcc), massif bc × lc × hc centré, mortier (tg, fg),
cordons (a), efforts N (compression > 0), Mx (comprime y > 0), My (comprime
x > 0), Vx, Vy, T, hauteur du poteau et contreventement (classification),
coefficients partiels.

| Étape | Module | Modèle | Référence |
|---|---|---|---|
| fjd, bande c, aire efficace | `composants.beton_comprime`, `c_bande`, `geometrie.aire_efficace` | Ac0 = platine, Ac1 concentrique limitée par le massif, 3·Ac0 et la hauteur (2:1) ; union exacte de rectangles | EN 1993-1-8 §6.2.5 (6.4)–(6.6) ; EN 1992-1-1 §6.7 (6.63) |
| Répartition | `equilibre.elastique` | platine rigide, lit sans traction sur Aeff (kc = E·k13/(beff·leff)), tiges tendues (kb = E·k16), Newton + intégrales de Green | EN 1992-4 section 6 ; EN 1993-1-8 §6.3.4 |
| Résistance, niveau 1 | `equilibre.capacite_niveau1` | résultante au centre de la bande comprimée, tiges opposées linéaires plafonnées, interaction linéaire des deux axes | EN 1993-1-8 §6.2.8.3 (Table 6.7) ; convexité |
| Résistance, niveau 2 | `equilibre.domaine`, `capacite_direction` | bloc fjd, tiges du côté tendu à leur résistance (champ extrémal → frontière convexe), balayage de l'axe neutre | théorème statique — **modèle d'ingénieur** |
| Platine comprimée | moteur, `pc` | moment élastique du débord sous la répartition réelle, ≤ tp²·fy/(6·γM0) | §6.2.5 (élastique par construction) |
| Platine tendue | `composants.tstub_rangee` | longueurs efficaces par tige (convention demi-longueur des tableaux), modes 1-2-3, Lb* | §6.2.4, Tableau 6.2, 6.4, 6.6 |
| Tiges acier | `tige_traction_EC3`, `tige_traction_EN1992_4`, `tige_cisaillement_*` | min(EC3, EN 1992-4) ; cisaillement EC3 avec frottement (Cf,d·N), EN 1992-4 sans, bras de levier si tg > d/2 ou fg < 30 | Tableau 3.4, §6.2.2 ; EN 1992-4 §7.2.1.3, §7.2.2.3 |
| Tiges béton | `cone_beton`, `arrachement_tete`, `adherence_crosse`, `bord_beton`, `pryout` | tiges coulées en place (k1 = 7,7/11 ; k2 = 7,5/10,5 ; k9 = 1,7/2,4 ; k8) ; ψM,N = 1 | EN 1992-4 §7.2.1, §7.2.2, §7.2.3 ; EN 1993-1-8 §6.2.6.12 |
| Poteau | `inertie_wpl` | aile/paroi comprimée Fc,fc,Rd = Mc,Rd/(h − tf) contre l'effort de la bande | §6.2.6.7 par analogie |
| Cordons | `cordons` | groupe « en ligne », point le plus sollicité, σ⊥ = τ⊥ = σn/√2 | §4.5.3.2 |
| Rigidité | `rigidite` | Sj,ini = E·z²/Σ(1/ki), seuil 30 (ou 48) E·Ic/Lc | §6.3.4, §5.2.2.5 |

Verdict : toutes les vérifications actives hors « information » (`fc2`,
`bet`) et hors minimum de détail (`wa`) ; géométrie valide ; pinces du
Tableau 3.3. Les vérifications de nature `INT` (niveau 2, tiges de coin ou
latérales) portent la mention « modèle d'ingénieur ».

## 2 · Position des tiges et tronçon en T

| Classe | Où | Modèle | Nature |
|---|---|---|---|
| `ext` | au-delà d'une paroi, dans sa largeur | motifs du Tableau 6.6 par tige (la somme sur deux tiges redonne la rangée) | EC |
| `coin` | au-delà de l'aile et de son about | mêmes motifs, plafond 0,25·b par tige (demi-aile disponible) | INT, alerte |
| `lat` | au-delà de l'about, entre les niveaux d'aile | appui réduit : plafond (tf + 4m)/2 | INT, alerte |
| `int` | entre les ailes, contre l'âme | Tableau 6.4 non raidi (conservatif) | EC, alerte |

Sans effet de levier si Lb > Lb* (Lb = 8d + tg + tp + 0,6d). La résistance
par tige est celle de sa rangée divisée par le nombre de tiges ; la demande
est l'effort élastique de la tige.

## 3 · Ce qui est volontairement conservatif

- Ac0 = platine pour √(Ac1/Ac0) ; ψM,N = 1 ; η1 = 0,7 pour l'adhérence ; la
  crosse n'est pas comptée ; rupture de bord calculée pour la résultante de
  cisaillement sur le bord le plus défavorable (trous normaux : rangée
  voisine seule) ; cordons supposés transmettre aussi la compression ; aile
  comprimée vérifiée avec tout l'effort de sa bande.
- Niveau 1 : le modèle de la Table 6.7 ne compte que la bande de la paroi
  comprimée ; au-delà de N = 2·FC,Rd il ne s'applique plus (alerte), le
  niveau 2 renseigne.
- Niveau 2 : l'étude envisageait une répartition linéaire des tiges ; le lot
  A retient le champ extrémal (tiges tendues à leur résistance), seul à
  tracer la frontière convexe du domaine — c'est le niveau 1, conservatif,
  qui porte la prudence vis-à-vis des modes fragiles du béton d'ancrage, et
  les efforts d'ancrage sont toujours ceux de la répartition élastique.

## 4 · Validation (état au lot A)

- `benchmark.py` : VAL-A — chaque composant recalculé à la main sur le cas
  par défaut ; VAL-B — répartition élastique du script autonome de l'étude
  (massif large). **Aucun exemple publié rejoué** : MSB Part 5 (pied
  articulé) et SCI P398 (pied encastré) au lot B, document sous les yeux.
- `tests/test_pied_poteau_moteur.py` : géométrie exacte, équilibre
  (résidus, compression pure, soulèvement), convexité du domaine et
  niveau 1 ≤ niveau 2 (cas tirés au sort), composants à la main,
  configurations (RHS six tiges, int, géométries invalides), benchmark, temps.
- Lot C (section 8) : les exemples publiés restent **à compléter** (documents
  inaccessibles depuis l'environnement) ; à leur place, une validation
  croisée indépendante du moteur (grille de ressorts + programme linéaire,
  rapport daté) ; tiges post-installées (valeurs d'une ETA) ; les coins et le
  biaxial par éléments finis restent hors outil, le niveau 1 garde le verdict.

## 5 · Résultat sur l'exemple du bureau

HEA 300, platine 500 × 500 × 25 S355, 4 × M24 8.8 en coin (400 × 400),
C30/37 fissuré, massif 900 × 900 × 600, N = 300 kN, Mx = 50, My = 30 kNm,
Vy = 40 kN : fjd = 20,4 MPa, c = 59 mm, une seule tige tendue (43 kN),
p_max = 11,7 MPa, niveau 1 = 0,53, niveau 2 = 0,38 — mais **le massif de
900 mm fait gouverner le bord (EN 1992-4, 0,75) et l'interaction béton
(1,08)** : non vérifié ; avec un massif de 1 400 mm, vérifié. C'est le
résultat annoncé par l'étude : ce sont le béton d'ancrage et la longueur
efficace des coins qui décident, pas EC3-1-8.

## 6 · L'écran, les dessins, la note (lot B)

Le pied est le quatrième type du projet d'assemblages (`acier/registre.py`,
famille « Pied de poteau », `etat="disponible"`) : « Créer » dans le
catalogue, repère A1…, sélecteur, 💾 / 📂 du projet, verrou 🔒, rapport du
projet (`acier/rapport_projet.py` assemble sa note de deux pages). Même
présentation que les trois assemblages poutre–poutre — préfixe de session
`ppb_`, colonne du dessin figée, carte par objet, onglets.

| Pièce | Fichier | Ce qu'elle porte |
|---|---|---|
| Dessins | `schemas.py` | **vue en plan** = la répartition : massif, platine, section du poteau hachurée, aire efficace (bande c, classe `ae`), zone comprimée de la répartition élastique (`ct`), axe neutre (`an`), résultante C, effort T de chaque tige tendue, poignées files / rangées ; **coupe A-A** (plan x–z : b, files, platine, mortier, tiges et têtes ou crosses, massif rompu, symbole de soudure) ; **coupe B-B** (plan y–z) ; **domaine (Mx, My)** à N donné (niveau 2 en polygone, niveau 1 en losange, point MEd). Cotes modifiables sur le dessin : bp, lp, tp, wx, wy, hef, dh, tg, bc, lc, a. |
| Kit partagé | `poutre_poutre/doubles_cornieres/schemas.py` | classes ajoutées pour les pieds : `be` béton, `gr` mortier, `tg` tige, `ae` / `ael` aire efficace, `ct` zone comprimée, `an` axe neutre, `d1` / `d2` / `pt` / `axe` / `grd` du domaine — `style_de` + `_css`, donc écran ET peintre PDF ; la parité des doubles cornières ne bouge pas (372 vues comparées). |
| Carte | `ecran_saisie.py` | poteau (forme, profilé ou dimensions, nuance), nuance de la platine, tiges (diamètre, classe, ancrage, files, rangées), visserie, béton (classe, fissuré, hc), efforts ; replis : coefficients (αcc, γc, γMc, γM0, γM2, Cf,d), mortier fg, Lc et contreventement, acier personnalisé. **Jamais de cote dans la carte.** |
| Résultats | `ecran_resultats.py` | bandeau (taux par élément : Pied, Béton, Platine, Tiges, Poteau, Soudure — les lignes d'information `fc2`, `bet` n'y comptent pas), alertes localisables (cotes et pièces en rouge), onglets **Vérifications** (six tableaux, formules, pince Tab. 3.3, paramètres retenus), **Répartition** (domaine, tiges, rangées), Note, Benchmark (VAL-A, VAL-B chargeables), Méthode. |
| Note | `note.py` | page 1 : colonne des dessins = vue en plan (répartition), coupe A-A, domaine, hypothèses ; tableaux Pied · Béton · Cordons / Platine · Tiges · Poteau ; conclusion « selon EN 1993-1-8 §6.2.8, EN 1992-1-1 §6.7 et EN 1992-4 (γM0 : ANB belge ; αcc = 0,85) » ; page 2 : plan de principe (coupe A-A, vue en plan, coupe B-B, échelle normalisée, cartouche TIGES + visserie). Les références courtes (`notation.ref_courte`) sont propres au module (trois normes) : la classe de note de la famille les reçoit par l'attribut `REF`. |
| Tests | `tests/test_pied_poteau_ecran.py` | 45 contrôles : navigation, écran, source unique (message du dessin, chip d'alerte, poignée, crosse, RHS), dessins (SVG, répartition, poignées, cotes de fabrication posées une fois), note (défaut + six cas figés, 2 pages sans débordement), benchmark, exports, projet, étanchéité. |

Les alertes du moteur portent désormais `dims` / `elems` (identifiants du
dessin) pour « Localiser », et une explication chiffrée `why` par alerte.

## 7 · Périmètre, et l'épaisseur de la platine « à la main » (lot B bis)

Le besoin d'origine du bureau est **l'épaisseur de la platine, avec le
moment qu'on prend pour la calculer** — pas le béton. Et la partie béton
d'ancrage (EN 1992-4) n'est validée que par les recalculs manuels de
l'outil (VAL-A), sans exemple publié rejoué. D'où deux décisions :

- **Périmètre** (entrée `perim`, premier bloc de la carte). Par défaut
  **« EN 1993-1-8 seule »** : platine, tiges (acier), poteau, cordons et
  béton d'appui (`fjd`, `N ≤ fjd·Aeff`) décident ; les cinq modes de ruine du
  béton d'ancrage (`cone`, `pull`, `edge`, `pry`, `cv`, groupe « Ancrages »)
  sont calculés, affichés **en information, hors verdict** (`R.info_keys`),
  avec une alerte `ancr_info` quand l'un dépasse 100 %. « Complet » les fait
  entrer dans le verdict (comportement du lot A). Le statut le dit :
  « VÉRIFIÉ (EC3 — béton d'ancrage en information) ». Ce que le périmètre
  EC3 ne garantit donc PAS : cône, arrachement, bord, pry-out, fendage.
- **Épaisseur requise** (`R.plaque`, vérification `tpq`, en tête du tableau
  Platine et du bloc « Épaisseur de la platine — le calcul à la main ») :
  `tp ≥ max(tc ; tN ; tM ; tT)` —
  `tc = √(6·mEd·γM0/fy)` pour le **moment pris côté comprimé** (console au
  nu de l'aile sous la pression réelle de la répartition, sur la bande `c`) ;
  `tN` : `N ≤ fjd·Aeff(c(tp))` ; `tM` : niveau 1 ≤ 1 ; `tT` : `FT,Rd(rangée,
  tp) ≥ T` pour le **moment pris côté tendu `M = T·m`** (rangée de la tige
  la plus sollicitée, bras au nu de l'aile cordon déduit, `Σleff,1`,
  `Mpl,1`, effet de levier). Chaque épaisseur vient d'un balayage de `tp`
  (demi-millimètre jusqu'à 60 mm, puis 2 mm) à répartition élastique figée,
  cohérent avec le verdict par construction (`_epaisseurs_requises` : on
  descend depuis `tp` tant que le critère tient, ou on monte jusqu'à ce
  qu'il tienne — le niveau 1 n'est pas monotone en `tp` : une bande `c`
  plus large que le débord ramène la résultante vers l'axe, alerte
  `c_ecret`). Un **repère manuel** est donné à côté : Navier sur toute la
  platine (`σ = N/A ± 6Mx/W ± 6My/W`, console sur le débord géométrique),
  quand il n'y a pas de soulèvement.

Sur l'exemple du bureau (périmètre EC3) : console comprimée `mEd = 11,5
kNm/m` sur `c = 59 mm` → `tc = 14,0 mm` ; tronçon en T `M = 43,2 kN × 45,9
mm = 1,98 kNm`, `Σleff,1 = 150 mm` → `tT = 8,6 mm` ; appui `tN = 4,0` ;
niveau 1 `tM = 11,1` ; **épaisseur requise 14,0 mm pour tp = 25 (56 %)**,
VÉRIFIÉ — le béton d'ancrage à 108 % reste dit en information.

## 8 · Lot C — validation, tiges post-installées, ce qui reste ouvert

**Les exemples publiés n'ont pas pu être rejoués** : MSB Part 5 §6.5 et SCI
P398 sont des téléchargements gratuits (steelconstruction.info, page *The
Green Books* ; constructalia.arcelormittal.com), mais ces hôtes sont refusés
par la politique réseau de l'environnement de calcul, et la règle du
module est de ne reconstituer aucun nombre de mémoire. Deux **fiches** sont
prêtes dans `docs/validation/exemples_publies.json` (entrées, pages,
grandeurs du moteur à comparer) : une fiche marquée `"complet": true` est
rejouée automatiquement par l'onglet Benchmark (`benchmark.BENCH`). Il
suffit d'y copier les valeurs du document.

**Validation croisée indépendante** (`tools/validation_pied_poteau.py`,
numpy + scipy hors chemin de production ; rapport daté
`docs/validation/pied_poteau_validation_2026-10-31.md`, trois figures) :

| Contrôle | Méthode indépendante | Écart maximal (40 configurations, 468 directions) |
|---|---|---|
| Répartition élastique (C, position, p max, T de chaque tige) | grille de ressorts au pas de 4 mm, Newton analytique | C 0,012 % · position 0,03 mm · p max 0,020 % · T 0,29 ‰ |
| Domaine résistant du niveau 2 | programme linéaire (HiGHS) sur cellules de 5 mm — l'exact du théorème statique | fonction d'appui 0,042 % · capacité sur le rayon 0,042 % |
| Niveau 1 ≤ niveau 2, rayon du moteur intérieur | idem | 0 direction en défaut sur 468 |

Ce que cette validation prouve : le moteur calcule exactement ce que son
modèle dit (découpage des polygones, intégrales de Green, Newton, balayage
de l'axe neutre). Ce qu'elle ne prouve pas : le modèle lui-même (bande c,
raideurs k13 / k16, champ extrémal du niveau 2) et la longueur efficace des
tiges de coin ou latérales, qui demandent un calcul de plaque
élasto-plastique hors outil — c'est pourquoi le niveau 1 garde le verdict
et les coins restent signalés. Le script a trouvé un défaut : une tige déjà
à sa résistance en traction avec bras de levier (`MRk,s = 0`, `VRd,s = 0`)
faisait diviser par zéro l'interaction traction / cisaillement — corrigé,
test figé.

**Tiges post-installées** (`typ_a` = « Tige post-installée (valeurs de
l'ETA) ») : `τRk`, `k1`, `NRk,s`, `VRk,s` saisis depuis l'ETA ; arrachement
combiné du groupe (EN 1992-4 §7.2.1.6, `scr,Np = 7,3·d·√τRk ≤ 3·hef`,
`ψg,Np = 1` conservatif, `τRk` aussi pour `scr,Np`), cône avec le `k1` de
l'ETA, tige seule bornée par `π·d·hef·τRk/γMc` ; coupe dessinée avec la
gaine de scellement. `k9` et `k8` restent ceux des tiges coulées en place.

**CHS** : toujours hors V1. L'aire efficace d'un tube circulaire est un
anneau — le kit de géométrie travaille en unions de rectangles — et aucun
tableau de l'EN 1993-1-8 ne donne la longueur efficace d'un tronçon en T
autour d'un tube ; à traiter quand une référence sera sous les yeux.

## 9 · Lot D — deux modes dans un seul écran

Le module a deux modes, choisis dans le bloc MODE de la carte ; l'entrée du
projet garde le mode avec ses données (`mode`), et un fichier d'avant le lot D
(sans clé `mode`) s'ouvre en vérification complète.

| | Calcul manuel de la platine (défaut) | Vérification complète (sections 1 à 8) |
|---|---|---|
| Disposition | boulons libres (`tiges`), éléments soudés (`elems`), raidisseurs (`raid`) | grille centrée n × n (wx, wy), un profilé en I ou RHS à l'axe |
| Efforts des tiges | platine rigide, même solveur (`equilibre.elastique`), bande c ou toute la platine | platine rigide sur la bande c |
| Verdict | la PLATINE : tronçon en T par boulon, section au nu, bande comprimée, trous — épaisseur requise recalculée à chaque épaisseur | pied complet : Table 6.7 (niveau 1), platine, tiges, poteau, cordons, béton d'ancrage selon le périmètre |
| Information | tiges (acier), appui (fjd) | niveau 2, pression, béton d'ancrage en mode EC3 |
| Code | `manuel/` (elements, console, moteur, conversion, schemas, ecran, note, texte, benchmark, synthese) | les fichiers du module |
| Tests | `tests/test_platine_manuelle.py`, `tests/test_platine_manuelle_ecran.py` | `tests/test_pied_poteau_moteur.py`, `tests/test_pied_poteau_ecran.py` |

Le passage d'un mode à l'autre reprend la disposition (`manuel/conversion.py`) :
vers le manuel, toujours (la grille devient une liste de boulons, le profilé
un élément à l'axe) ; vers la vérification complète, seulement si la
disposition est régulière — sinon la bascule est refusée et la raison est
dite. `calcul.compute` appelle le moteur du mode ; `note.generer_note` et
`texte.construire_texte` aiguillent de même ; l'écran, le projet et le
rapport n'ont qu'un point d'entrée.
