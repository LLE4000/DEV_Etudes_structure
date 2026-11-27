# -*- coding: utf-8 -*-
"""Onglet Méthode : le modèle mécanique de la platine d'about, ses sources
et ses limites — en toutes lettres."""
from .notation import NOTATIONS
from .entrees import PLEINE_RESISTANCE, GUIDE_PLEINE_RESISTANCE
from acier.assemblages import materiau as MT

METHODE = """
#### Ce que l'outil vérifie

Vérification ELU selon EN 1993-1-8 et EN 1993-1-1 (γM0 : ANB belge) ; modèles
MSB Part 5 §2 (platine d'about partielle, tying §2.3), SCI P358, ECCS n°126.
Benchmark : Worked Example publié du guide (§2.4), rejoué à chaque ouverture.

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

- **Groupe de boulons** (SCI P358, Check 8, sur EN 1993-1-8 §3.7) : `Fb,Rd`
  est la plus PETITE pression diamétrale d'un boulon — sur la platine
  (extrémité `αb = e1/3d0`, intérieur `αb = p1/3d0 − 1/4`) ET sur l'âme
  porteuse ; puis `FRd = n·Fb,Rd` si `Fb,Rd ≤ 0,8·Fv,Rd`, sinon
  `FRd = 0,8·n·Fv,Rd` (le 0,8 couvre la traction parasite due à la rotation).
  Avant le lot de clôture, une troisième branche additionnait les pressions
  diamétrales (`ΣFb,Rd`) quand la plus grande restait sous `Fv,Rd` : elle
  dépassait le guide de 4 à 14 % dans quatre cas figés. βLf (§3.8) : « Non »
  par défaut, comme les exemples publiés — l'effort est introduit sur toute
  la longueur de l'attache (transfert uniforme : l'exception du §3.8(2)) ; un avis le rappelle dès que `(n1 − 1)·p1 > 15·d`
  et l'option est dans les paramètres avancés. Glissement §3.9 (catégories
  B et C).
- **Platine** (MSB P5 §2.2, deux plans de cisaillement, un par côté de
  l'âme) : brut `2·(hp·tp/1,27)·fy/(√3·γM0)` — le 1,27 couvre la flexion
  nominale dans son plan ; net `2·tp·(hp − n1·d0)·fu/(√3·γM2)` ; rupture de
  bloc §3.10.2 avec `k = 1` si `hp ≥ 1,36·p3` (flexion négligeable), `0,5`
  sinon ; **platine courte** (`hp < 1,36·p3`) : flexion dans son plan,
  `VRd = 2·tp·hp²·fy/(3·(p3 − tw)·γM0)` (SCI P358, Check 9).
- **Âme portée** : cisaillement sur la hauteur de la platine,
  `Av = 0,9·hp·tw` (SCI P358, Check 4 — avant le lot : `hp·tw`, soit +11 %) ;
  double grugeage : `Av = 0,9·tw·hT` ; grugeage :
  cisaillement de la section réduite, flexion + cisaillement à l'extrémité
  du grugeage avec le bras `tp + c`, stabilité locale — mêmes règles que les
  deux autres assemblages (MSB P5 §4.2.4 / §4.2.5).
- **Âme porteuse** : pression diamétrale (boulons intérieurs, pas de bord
  libre) et cisaillement local brut / net sous `VEd/2` (un seul côté
  chargé — SCI P358, Check 10) sur `Av = t2·(et + (n1 − 1)·p1 + eb)`, avec
  `et = min(e1,t ; 5·d)` et `eb = min(e1,b ; p3/2 ; 5·d)` (avant le lot : la
  hauteur `hp` de la platine). Le guide MSB P5 §2.2 ne vérifie pas l'élément
  porteur : ces lignes viennent de SCI P358.
- **Soudure** : deux cordons de longueur efficace `hp − 2·a` chacun
  (EN 1993-1-8 §4.5.1(1)), méthode simplifiée §4.5.3.3 ; gorge minimale
  §4.5.2 ; une longueur efficace sous `max(30 mm ; 6·a)` est refusée.
  **Pleine résistance** de l'âme : `a ≥ βw·γM2·fy·tw/(√2·fu)`, avec la fy de
  l'ÂME à son épaisseur tw (MSB P5 §2.2) ; le site obtient **{PLEINE}**.
  {GUIDE}. Le **détail éprouvé** du SCI P358 (Check 2, annexe C.2) demande
  0,8 fois cette gorge (la platine plastifie avant le cordon) : sous ce
  seuil, alerte et statut « À VÉRIFIER (articulation à justifier) ».

#### Traction NEd : deux natures

Une **compression** (NEd < 0) est transmise par contact de la platine sur
l'âme porteuse, sans effet sur les boulons ni sur la platine : l'outil
n'en tient pas compte et la nature de NEd n'apparaît que pour une traction.

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
la valeur du guide. En **tying**, Σleff est de plus plafonnée par la règle
des deux guides (SCI P358, Check 11 ; MSB P5 §2.3.1) :
`e1A = min(e1 ; 0,5·(p3 − tw − 2·a·√2) + d0/2)`,
`p1A = min(p1 ; p3 − tw − 2·a·√2 + d0)`. L'exemple 2 du SCI (381 kN) est
retrouvé avec sa tête de 33 mm sur angles ; le site compte une rondelle de
37 mm (393 kN).

#### Ce que le statut dit

« ASSEMBLAGE VÉRIFIÉ » : toutes les vérifications passent et l'attache est
dans le détail éprouvé. « ASSEMBLAGE À VÉRIFIER (raison) » : elles passent,
mais une raison nommée reste à justifier — **articulation** (platine plus
épaisse que la limite de ductilité, gorge sous le détail éprouvé) ; **poutre
grugée non maintenue** au déversement ; **effort tranchant inversé**
(`VEd < 0` : les ruptures de bloc sont celles d'un effort descendant) ;
**traction** : la flexion hors plan de l'âme porteuse n'est pas calculée.
L'écran, la note PDF et les propositions disent le même statut, en orange.

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

#### Données de bibliothèque

Nuances : {NUANCES} Profilés : catalogue HEA / HEB /
HEM / IPE ; « Personnalisé » = dimensions h, b, tw, tf, r saisies.

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
""".replace("{NUANCES}", MT.texte_nuances()).replace("{PLEINE}", PLEINE_RESISTANCE).replace("{GUIDE}", GUIDE_PLEINE_RESISTANCE)
# la règle belge des nuances (materiau.py) et la gorge de pleine résistance (entrees.py) : une source chacune


def _notations(texte):
    """Le paragraphe « Notations » de l'onglet : la ligne « Notations » du pied
    de la note, mot pour mot (une source : notation.py) — les définitions que
    les cotes, réduites à leur symbole, ne portent plus (consolidation du
    01/11/2026, point 4)."""
    return "\n#### Notations\n\n" + texte.split(" : ", 1)[1] + "\n"


METHODE += _notations(NOTATIONS)
