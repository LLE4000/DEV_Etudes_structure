# Pied de poteau sur béton — module `acier/assemblages/pied_poteau/platine_beton`

*Lot A (31/10/2026) : le MOTEUR, sans écran. Il applique la méthode de
l'étude `PIED_POTEAU_ETUDE.md`. Lot B : écran, note, projet. Lot C :
validation, CHS, tiges post-installées.*

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
- Lot C : reproduction de la Table 6.7 sur exemples publiés, éléments finis
  hors outil pour les coins et le biaxial, promotion éventuelle du niveau 2.

## 5 · Résultat sur l'exemple du bureau

HEA 300, platine 500 × 500 × 25 S355, 4 × M24 8.8 en coin (400 × 400),
C30/37 fissuré, massif 900 × 900 × 600, N = 300 kN, Mx = 50, My = 30 kNm,
Vy = 40 kN : fjd = 20,4 MPa, c = 59 mm, une seule tige tendue (43 kN),
p_max = 11,7 MPa, niveau 1 = 0,53, niveau 2 = 0,38 — mais **le massif de
900 mm fait gouverner le bord (EN 1992-4, 0,75) et l'interaction béton
(1,08)** : non vérifié ; avec un massif de 1 400 mm, vérifié. C'est le
résultat annoncé par l'étude : ce sont le béton d'ancrage et la longueur
efficace des coins qui décident, pas EC3-1-8.
