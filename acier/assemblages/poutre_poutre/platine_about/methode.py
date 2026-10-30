# -*- coding: utf-8 -*-
"""Onglet Méthode : le modèle mécanique de la platine d'about, ses sources
et ses limites — en toutes lettres."""

METHODE = """
#### Modèle mécanique

Assemblage **articulé** : la rotule est à la **face de l'âme porteuse**. La
platine, soudée en about de la poutre portée par deux cordons d'angle sur
la hauteur `hp` de l'âme, est boulonnée à l'âme porteuse par **deux files
verticales** symétriques (entraxe `p3`), dans **un seul plan de
cisaillement**. Le groupe est centré : chaque boulon reprend `VEd/n`, sans
moment d'excentricité. La rotation de l'articulation vient de la
**déformation de la platine mince** — d'où les règles de ductilité.
Modèle du guide *Steel Buildings in Europe – Part 5: Joint Design* §2
(platine d'about de hauteur partielle), identique à SCI P358 et ECCS n°126.

#### Vérifications et sources

- **Groupe de boulons** (MSB P5 §2.2.4 sur EN 1993-1-8 §3.7) : pression
  diamétrale sur la platine, boulons d'extrémité (`αb = e1/3d0`) et
  intérieurs (`αb = p1/3d0 − 1/4`), puis résistance du groupe en trois
  branches — `ΣFb,Rd` si `(Fb,Rd)max ≤ Fv,Rd`, `ns·(Fb,Rd)min` entre les
  deux, `0,8·ns·Fv,Rd` si le cisaillement des boulons gouverne (le 0,8
  couvre la traction parasite due à la rotation). Pas de βLf : la platine
  transmet l'effort uniformément (§3.8(2)) — option conservatrice dans les
  paramètres avancés. Glissement §3.9 (catégories B et C).
- **Platine** (MSB P5 §2.2, deux plans de cisaillement, un par côté de
  l'âme) : brut `2·(hp·tp/1,27)·fy/(√3·γM0)` — le 1,27 couvre la flexion
  nominale dans son plan ; net `2·tp·(hp − n1·d0)·fu/(√3·γM2)` ; rupture de
  bloc §3.10.2 avec `k = 1` si `hp ≥ 1,36·p3` (flexion négligeable), `0,5`
  sinon.
- **Âme portée** : cisaillement sur la hauteur de la platine,
  `Av = hp·tw` (EN 1993-1-1 §6.2.6, aire du guide §2.2.1) ; grugeage :
  cisaillement de la section réduite, flexion + cisaillement à l'extrémité
  du grugeage avec le bras `tp + c`, stabilité locale — mêmes règles que les
  deux autres assemblages (MSB P5 §4.2.4 / §4.2.5).
- **Âme porteuse** : pression diamétrale (boulons intérieurs, pas de bord
  libre) et cisaillement local brut / net sous `VEd/2` (un seul côté
  chargé — modèle SCI P358, le même que les deux autres assemblages). Le
  guide MSB P5 §2.2 ne vérifie pas l'élément porteur : ces lignes viennent
  de SCI P358.
- **Soudure** : deux cordons sur toute la hauteur `hp`, méthode simplifiée
  §4.5.3.3 ; gorge minimale §4.5.2 ; **pleine résistance** recommandée par
  le guide : `a ≥ βw·γM2·fy·tw/(√2·fu)` soit **0,46 / 0,48 / 0,55·tw** en
  S235 / S275 / S355 (MSB P5 §2.2) — signalée en alerte.

#### Traction NEd : deux natures

- **ELU fondamental** : combinée avec VEd. Boulons : interaction
  `Fv,Ed/Fv,Rd + Ft,Ed/(1,4·Ft,Rd) ≤ 1` (Tableau 3.4) ; platine : **tronçon
  en T** (§6.2.4, Tableau 6.2, mode 1 par la méthode 2), fy/γM0 ;
  poinçonnement ; âme portée `tw·hp·fy/γM0` ; cordons sous la résultante.
- **Tying** (force de chaînage, situation accidentelle — MSB P5 §2.3) :
  vérifiée **seule**, avec **fu/γMu** (γMu = 1,1, valeur du guide) :
  tronçon en T, âme portée `tw·hp·fu/γMu`, cordons. L'EN 1993-1-8 ne donne
  pas de méthode de tying ; la force vient de l'EN 1991-1-7 annexe A.

Tronçon en T : `m = (p3 − tw)/2 − 0,8·a·√2`, `n = min(e2 ; 1,25·m)`,
`ew = dw/4` ; Σleff des rangées en groupe ou isolées, platine à bords
libres (formules du Tableau 6.4) — pour les détails courants **Σleff = hp**,
la valeur du guide.

#### Règles de détail (alertes)

- `hp ≥ 0,6·h` de la portée : maintien en torsion (MSB P5 §2.1) ;
- **ductilité** : `tp ≤ 0,36·d·√(fub/fy)` (EN 1993-1-8 §6.4.2(2), règle
  d'ECCS n°126 pour les platines d'about) ; 10 ou 12 mm usuels, moins de
  8 mm déconseillé (MSB P5 §2.1) ;
- la rondelle dégage le cordon : `p3 ≥ tw + 2·a·√2 + dw` ; entraxe usuel
  du guide : 90 ou 140 mm.

#### Annexe nationale belge

Les coefficients par défaut suivent la **NBN EN 1993-1-1 ANB** (γM0 = 1,00)
et les valeurs recommandées de l'EN 1993-1-8 (γM2 = 1,25 ; γM3 = 1,25 ;
γM3,ser = 1,1) — aucune divergence belge relevée pour la partie 1-8 ; le
texte NBN, payant, n'a pas été consulté ici : à confirmer, valeurs
modifiables. Aucune vérification de cet assemblage n'utilise γM1. **γMu =
1,1** (tying) est la valeur du guide ; l'EN 1993-1-8 ne la fixe pas — à
confirmer pour un projet belge.

#### Limites du modèle (à justifier séparément le cas échéant)

- flexion **transversale** de l'âme porteuse sous la traction des boulons
  (NEd, tying) — non couverte par l'EN 1993-1-8 pour une âme de poutre ;
- capacité de rotation au-delà des règles de détail (contact de la semelle
  inférieure avec l'âme porteuse) ;
- cisaillement global, flexion et déversement de la poutre porteuse ;
  torsion induite par une attache d'un seul côté ;
- deux poutres en vis-à-vis sur la même âme (interaction non traitée).

#### Sources

*Steel Buildings in Europe – Multi-Storey Steel Buildings, Part 5: Joint
Design* (ArcelorMittal / Peiner Träger / Corus, contenu CTICM et SCI) — §2
platine d'about de hauteur partielle et §2.4 Worked Example (benchmark du
module) ; *SCI P358, Joints in Steel Construction: Simple Joints to
Eurocode 3* ; *ECCS n°126* ; EN 1993-1-8:2005 ; EN 1993-1-1:2005.
"""
