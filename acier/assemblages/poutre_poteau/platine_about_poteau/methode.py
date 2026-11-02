# -*- coding: utf-8 -*-
"""Onglet Méthode : le modèle mécanique de la platine d'about sur poteau, ses
sources et ses limites — en toutes lettres.

Les LIMITES de validité ont une seule source, ``..textes`` pour les huit
communes à la famille et ce fichier pour la septième : l'onglet Méthode
les écrit en liste, et l'onglet Vérifications les reprend mot pour mot dans
le repli « Périmètre de la vérification » (``ecran_resultats.NON_REVERIFIES``
les lit ici). Elles sont celles de l'étude, `POUTRE_POTEAU.md` §8.4 — la
septième ne vaut que pour le cas (b), et sa rédaction est celle de CE module :
le groupe de boulons est CENTRÉ, il n'y a donc pas de composante
``qx = 3·MS/hp²`` du moment d'excentricité, seulement la force de chaînage."""
from .. import textes
from .notation import NOTATIONS

# Les neuf limites du §8.4 ont UNE SOURCE, ``..textes`` (consolidation du
# 02/11/2026) : les six premières et les deux dernières sont celles de la
# FAMILLE, mot pour mot. Seule la septième est rédigée ici — ``LIMITE_AME``
# ne vaut qu'en cas (b), et chaque attache la dit à sa façon : ce qui
# sollicite l'âme hors de son plan n'est pas le même (docstring de
# ``..textes``).
LIMITE_AME = ("flexion HORS PLAN de l'âme du poteau sous la traction des boulons (NEd, tying) — non "
              "couverte par l'EN 1993-1-8 pour une âme non raidie : le tronçon en T suppose une aile "
              "encastrée sur une âme centrale, une âme entre deux ailes ne s'y ramène pas")


def limites(ame=True):
    """Les limites de validité pour le cas d'appui courant (``ame`` : cas b) —
    les neuf du §8.4, assemblées par la source unique de la famille."""
    return textes.limites(LIMITE_AME, ame)


def _liste(items):
    return "\n".join("- " + t + " ;" for t in items)


METHODE = """
#### Ce que l'outil vérifie

Vérification ELU selon EN 1993-1-8 et EN 1993-1-1 (γM0 : ANB belge) ; modèles
MSB Part 5 §2 (platine d'about partielle, **tying** §2.3), SCI P358,
ECCS n°126. Benchmark : le Worked Example publié du guide (§2.4) rejoué à
chaque ouverture **sur un poteau**, et trois cas calculés à la main (étude du
01/11/2026, `docs/assemblages/POUTRE_POTEAU.md` §11.3).

#### Modèle mécanique

Assemblage **articulé** : la rotule est à la **face d'appui du poteau**. La
platine, soudée en about de la poutre portée par deux cordons d'angle sur la
hauteur `hp` de l'âme, est boulonnée au poteau par **deux files verticales**
symétriques (entraxe `p3`, le « gauge »), dans **un seul plan de
cisaillement**. Le groupe est **centré** : chaque boulon reprend `VEd/n`,
sans moment d'excentricité — c'est ce qui distingue cette attache du plat
d'âme et des cornières. La rotation de l'articulation vient de la
**déformation de la platine mince** : d'où les règles de ductilité.

#### Les deux cas d'appui — c'est le choix qui commande tout

- **Cas (a), sur l'AILE** : les deux files sont symétriques par rapport à
  l'ÂME du poteau, et l'attache vit **dans le plan de cette âme**. À l'ELU le
  côté poteau n'apporte que la **pression diamétrale de l'aile** : il n'y a
  **aucune pince d'extrémité verticale** (le poteau continue au-dessus et en
  dessous), donc `αb = min(p1/3d0 − 1/4 ; fub/fu ; 1)` pour TOUTES les
  rangées, et `k1 = min(2,8·e/d0 − 1,7 ; 1,4·p3/d0 − 1,7 ; 2,5)` avec
  `e = (bP − p3)/2`, la pince au bout d'aile. Dès que les boulons sont tendus
  (NEd à l'ELU, ou **tying**) s'ajoutent deux composants de l'Eurocode que les
  trois modules poutre–poutre n'ont pas : le **tronçon en T de l'aile non
  raidie** (§6.2.4 Tab. 6.2 méthode 2, longueurs efficaces du **Tab. 6.4** —
  toutes les rangées sont INTÉRIEURES) et l'**âme du poteau en traction
  transversale** (§6.2.6.3, `ω = 1`).
- **Cas (b), dans l'ÂME** : la platine porte sur l'âme, entre les ailes.
  `VEd` reste en plan dans l'âme — d'où le **cisaillement local** d'un élément
  support chargé d'un seul côté, `VEd/2 ≤ twP·hp·fy,P/(√3·γM0)` et sa section
  nette (SCI P358, le modèle même de l'âme porteuse des modules
  poutre–poutre). Mais la traction des boulons sollicite l'âme **hors de son
  plan** : l'EN 1993-1-8 ne couvre pas ce composant pour une âme NON raidie,
  et l'outil le **déclare** au lieu de transposer un modèle qui ne tient pas —
  le tronçon en T de l'aile et l'âme en traction transversale sont alors sans
  objet. Géométrie du cas (b) : la platine tient entre les congés
  (`bp ≤ hP − 2·tfP − 2·rP − 2·j`), l'écrou dégage le congé
  (`p3/2 + dw/2 ≤ (hP − 2·tfP)/2 − rP`), la semelle passe entre les ailes
  (`bS ≤ hP − 2·tfP − 2·j`) ou elle est **recoupée** en largeur depuis
  l'about, `rc ≥ (bP − twP)/2 + j − tp`. La recoupe en LARGEUR n'est **pas**
  un grugeage : les règles de stabilité locale de MSB P5 §4.2.5 portent sur
  une entaille en hauteur et ne sont pas activées (point tranché, voir plus
  bas).

#### Le GAUGE : la règle qui refuse le p3 usuel des âmes

En cas (a), l'écrou est du côté **intérieur** de l'aile, au droit du congé de
raccordement : il faut `p3 ≥ twP + 2·rP + dw`. Sur un HEB 300 avec des M20,
cela fait **102 mm** — le `p3 = 90 mm` usuel des ÂMES de poutre **ne passe
pas** (26,5 mm disponibles pour 32,5 nécessaires). C'est la raison des gauges
usuels d'aile de poteau : **90 / 100 / 120 / 140 mm** selon la série.
L'alerte est **bloquante** et propose le minimum. Nature **INT** : c'est une
règle géométrique de dégagement, pas une clause normative (SCI P358 tabule
des gauges minimaux par série, non consultés — point P-7).

#### Vérifications et sources

- **Groupe de boulons** (MSB P5 §2.2.4 sur EN 1993-1-8 §3.7) : pression
  diamétrale sur la platine, boulons d'extrémité (`αb = e1/3d0`) et
  intérieurs (`αb = p1/3d0 − 1/4`), puis résistance du groupe en trois
  branches — `ΣFb,Rd` si `(Fb,Rd)max ≤ Fv,Rd`, `ns·(Fb,Rd)min` entre les
  deux, `0,8·ns·Fv,Rd` si le cisaillement des boulons gouverne (le 0,8 couvre
  la traction parasite due à la rotation). Glissement §3.9 (catégories B et
  C) ; interaction cisaillement + traction (Tab. 3.4) à l'ELU.
- **Platine** (MSB P5 §2.2, deux plans de cisaillement, un par côté de
  l'âme) : brut `2·(hp·tp/1,27)·fy/(√3·γM0)` — le 1,27 couvre la flexion
  nominale dans son plan ; net ; rupture de bloc §3.10.2 avec `k = 1` si
  `hp ≥ 1,36·p3`, `0,5` sinon ; **tronçon en T** en traction (§6.2.4) et
  poinçonnement, `t·fu` minimal de la platine ou du poteau.
- **Âme portée** : cisaillement sur la hauteur de la platine, `Av = hp·tw`
  (EN 1993-1-1 §6.2.6, aire du guide §2.2.1) ; traction au droit de la
  platine ; grugeage (facultatif sur un poteau, **0 par défaut**) :
  cisaillement de la section réduite, flexion + cisaillement à l'extrémité du
  grugeage avec le bras `tp + c`, stabilité locale — mêmes règles que les
  modules existants (MSB P5 §4.2.4 / §4.2.5).
- **Poteau porteur** : pression diamétrale de l'aile (a) ou de l'âme (b) ;
  en cas (b) cisaillement local brut et net sous `VEd/2` (SCI P358) ; en
  cas (a) et sous traction, tronçon en T de l'aile (Tab. 6.4) et âme en
  traction transversale (§6.2.6.3, `ω = 1`).
- **Soudure** : deux cordons sur toute la hauteur `hp`, méthode simplifiée
  §4.5.3.3 ; en tying `qx = NEd/2hp` avec `fu/(√3·βw·γMu)` ; gorge minimale
  §4.5.2 ; **pleine résistance** recommandée par le guide :
  `a ≥ βw·γM2·fy·tw/(√2·fu)` soit **0,46 / 0,48 / 0,55·tw** en
  S235 / S275 / S355 (MSB P5 §2.2) — signalée en alerte.

Le moment que l'attache apporte au poteau, `VEd·e_col` (`e_col = hP/2 + tp` en
cas (a), `tp` en cas (b)), est **affiché** en information dans « Paramètres
retenus », dans l'export texte et au bloc HYPOTHÈSES de la note : il
appartient au calcul d'ÉLÉMENT du poteau (EC3-1-1 §6.2 et §6.3), l'outil ne
le vérifie pas.

#### Traction NEd : deux natures

Une **compression** (NEd < 0) est transmise par contact de la platine sur le
poteau, sans effet sur les boulons ni sur la platine : l'outil n'en tient pas
compte et la nature de NEd n'apparaît que pour une traction.

- **ELU fondamental** : combinée avec VEd. Boulons : interaction
  `Fv,Ed/Fv,Rd + Ft,Ed/(1,4·Ft,Rd) ≤ 1` ; platine et aile du poteau :
  tronçons en T avec `fy/γM0` ; poinçonnement ; âme portée `tw·hp·fy/γM0` ;
  âme du poteau en traction transversale ; cordons sous la résultante.
- **Tying** (force de chaînage, situation accidentelle — MSB P5 §2.3) :
  vérifiée **seule**, avec **fu/γMu** (γMu = 1,1, valeur du guide) : les deux
  tronçons en T, le poinçonnement, l'âme portée `tw·hp·fu/γMu`, l'âme du
  poteau et les cordons. L'EN 1993-1-8 ne donne pas de méthode de tying ; la
  force vient de l'EN 1991-1-7 annexe A.

Tronçons en T, `ew = dw/4`, mode 1 par la **méthode 2** du Tableau 6.2 :
*platine* `m = (p3 − tw)/2 − 0,8·a·√2`, `n = min(e2 ; 1,25·m)`, Σleff d'une
platine à bords libres (pour les détails courants **Σleff = hp**, la valeur
du guide) ; *aile du poteau* `m = (p3 − twP)/2 − 0,8·rP` (poteau LAMINÉ : le
congé prend la place du cordon), `e = (bP − p3)/2`, `n = min(e ; 1,25·m)`,
Σleff du **Tab. 6.4** — aucune rangée d'extrémité, donc `Σleff = n1·p1` dès
que les rangées se prennent en groupe.

#### Règles de détail (alertes)

- `hp ≥ 0,6·h` de la portée : maintien en torsion (MSB P5 §2.1) ;
- **ductilité** : `tp ≤ 0,36·d·√(fub/fy)` (EN 1993-1-8 §6.4.2(2), règle
  d'ECCS n°126 pour les platines d'about) ; 10 ou 12 mm usuels, moins de
  8 mm déconseillé (MSB P5 §2.1) ;
- la rondelle dégage le cordon : `p3 ≥ tw + 2·a·√2 + dw` ;
- l'écrou dégage le congé du poteau : `p3 ≥ twP + 2·rP + dw` (cas a) ou
  `p3/2 + dw/2 ≤ (hP − 2·tfP)/2 − rP` (cas b) ;
- `bp ≤ bP` en cas (a) ; `bp ≤ hP − 2·tfP − 2·rP − 2·j` en cas (b).

#### Annexe nationale belge

Les coefficients par défaut suivent la **NBN EN 1993-1-1 ANB** (γM0 = 1,00)
et les valeurs recommandées de l'EN 1993-1-8 (γM2 = 1,25 ; γM3 = 1,25 ;
γM3,ser = 1,1) — aucune divergence belge relevée pour la partie 1-8 ; le
texte NBN, payant, n'a pas été consulté ici : à confirmer, valeurs
modifiables. Aucune vérification de cet assemblage n'utilise γM1.
**γMu = 1,1** (tying) est la valeur des guides (MSB P5 §2.3, SCI P358) ;
l'EN 1993-1-8 ne la fixe pas et l'ANB n'a pas été consultée — à confirmer
(point P-9).

#### Données de bibliothèque

Nuances : S355 : fu = 510 MPa (EN 1993-1-1 Tab. 3.1) ; « S355 (fu 490) » :
valeur d'autres références, à caler sur l'ANB. Profilés : catalogue HEA / HEB /
HEM / IPE ; « Personnalisé » = dimensions h, b, tw, tf, r saisies. Le poteau
est un profil en I ou H, âme verticale : un poteau CREUX (RHS / SHS / CHS)
sort du périmètre (autre composant, EN 1993-1-8 §7).

#### Limites du modèle (à justifier séparément le cas échéant)

""" + _liste(limites(True)) + """

#### Points TRANCHÉS SANS SOURCE FRANCHE — à relire par le bureau

Ils sont listés dans `docs/assemblages/POUTRE_POTEAU.md` §9 et
`docs/assemblages/ECARTS_ET_CORRECTIONS.md`. Ceux qui portent sur CE module :

- **P-1** : le NUMÉRO du tableau des longueurs efficaces de l'aile de poteau
  non raidie — **Tableau 6.4** retenu (la consigne de l'étude citait le
  Tableau 6.6) ; les FORMULES, elles, sont bien celles de l'aile non raidie,
  et ce sont celles que le module poutre–poutre applique déjà à la platine
  (benchmark à 0,23 % de l'exemple publié) ;
- **P-4** : aucune épaisseur minimale d'aile de poteau n'est imposée — ni
  l'EN 1993-1-8 ni les guides n'en donnent pour une attache simple ; ce sont
  la pression diamétrale et le tronçon en T qui décident ;
- **P-5** : l'âme du poteau en traction transversale est appliquée avec
  `ω = 1` et `beff,t,wc = Σleff` de l'aile (nature `EC + INT`) : `ω` dépend du
  cisaillement du panneau d'âme, hors périmètre ici, et `ω = 1` est la valeur
  en l'absence de moment d'étage — donc la plus favorable. Un bureau qui veut
  être conservatif impose `ω` en saisie ;
- **P-6** : la flexion hors plan de l'âme du poteau en cas (b) n'est pas
  vérifiée, elle est déclarée ;
- **P-7** : le gauge minimal `p3 ≥ twP + 2·rP + dw` est une règle géométrique
  de dégagement du congé, pas une clause normative ;
- **P-3** : la recoupe en LARGEUR des semelles n'est **pas** un grugeage ; les
  règles MSB P5 §4.2.4 / §4.2.5 ne sont pas activées par elle ;
- **P-9** : `γMu = 1,1` est la valeur du guide, modifiable en saisie.

#### Sources

*Steel Buildings in Europe – Multi-Storey Steel Buildings, Part 5: Joint
Design* (ArcelorMittal / Peiner Träger / Corus, contenu CTICM et SCI,
06/2009) — §2 platine d'about de hauteur partielle et §2.4 Worked Example
(benchmark du module), §2.3 tying ; *SCI P358, Joints in Steel Construction:
Simple Joints to Eurocode 3* ; *ECCS n°126* ; EN 1993-1-8:2005 ;
EN 1993-1-1:2005.
"""


def _notations(texte):
    """Le paragraphe « Notations » de l'onglet : la ligne « Notations » du pied
    de la note, mot pour mot (une source : notation.py) — les définitions que
    les cotes, réduites à leur symbole, ne portent plus."""
    return "\n#### Notations\n\n" + texte.split(" : ", 1)[1] + "\n"


METHODE += _notations(NOTATIONS)
