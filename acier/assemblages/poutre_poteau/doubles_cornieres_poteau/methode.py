# -*- coding: utf-8 -*-
"""Onglet Méthode : le modèle mécanique des doubles cornières sur poteau, ses
sources et ses limites — en toutes lettres.

Les LIMITES de validité ont une seule source (ce fichier) : l'onglet Méthode
les écrit en liste, et l'onglet Vérifications les reprend mot pour mot dans
le repli « Périmètre de la vérification » (``ecran_resultats.NON_REVERIFIES``
les lit ici). Elles sont celles de l'étude, `POUTRE_POTEAU.md` §8.4."""
from .notation import NOTATIONS

# Les neuf limites du §8.4 : les six premières et les deux dernières valent
# dans les deux cas d'appui ; ``LIMITE_AME`` ne vaut qu'en cas (b).
LIMITES_AVANT = (
    "panneau d'âme du poteau en cisaillement sous les moments d'étage (§6.2.6.1) — une attache simple "
    "n'y contribue pas, il se vérifie avec l'ensemble des attaches de l'étage",
    "poteau sous charges combinées : résistance de section, flambement, déversement, et le moment "
    "VEd·e que cette attache lui apporte (EC3-1-1 §6.2 et §6.3)",
    "torsion du poteau induite par une attache d'un seul côté",
    "attaches des DEUX côtés du poteau : efforts opposés, boulons ou âme partagés — pression "
    "diamétrale, cisaillement local et tying non traités",
    "capacité de ROTATION au-delà des règles de détail : l'articulation est supposée (§5.2), le "
    "contact de la semelle inférieure de la poutre sur le poteau n'est pas vérifié",
    "cisaillement global, flexion et déversement de la poutre portée",
)
LIMITE_AME = ("flexion HORS PLAN de l'âme du poteau sous la force de chaînage et sous la traction des "
              "boulons — non couverte par l'EN 1993-1-8 pour une âme non raidie")
LIMITES_APRES = (
    "poteau raidi (goussets, raidisseurs entre ailes) : les longueurs efficaces du Tab. 6.4 et les "
    "limites de l'âme supposent le poteau NON raidi",
    "ELS : flèches, vibrations, confort",
)


def limites(ame=True):
    """Les limites de validité pour le cas d'appui courant (``ame`` : cas b)."""
    return LIMITES_AVANT + ((LIMITE_AME,) if ame else ()) + LIMITES_APRES


def _liste(items):
    return "\n".join("- " + t + " ;" for t in items)


METHODE = """
#### Ce que l'outil vérifie

Vérification ELU selon EN 1993-1-8 et EN 1993-1-1 ; modèles complémentaires
MSB Part 5 §4 (doubles cornières), SCI P358, ECCS n°126 ; **tying** (force de
chaînage, situation accidentelle) selon MSB Part 5 §2.3. Benchmark : le
Worked Example publié du guide (§4.4), rejoué à chaque ouverture — y compris
son **ancre du tronçon en T** (modes 1 / 2 / 3) —, et trois cas calculés à la
main (étude du 01/11/2026, `docs/assemblages/POUTRE_POTEAU.md` §11.4).

#### Modèle mécanique

Assemblage **articulé** : la rotule est à la **face d'appui du poteau**. Deux
cornières identiques transmettent l'effort tranchant de l'âme de la poutre
portée au poteau :

- âme portée → **groupe S** (boulons en DOUBLE cisaillement, ou cordons) →
  ailes B, `VEd/2` par cornière ;
- talon de chaque cornière → aile A → **groupe P** (boulons en SIMPLE
  cisaillement, un plan par cornière, ou cordons) → poteau.

Le groupe S reprend l'effort tranchant **et** le moment d'excentricité
`MS = VEd·z + |MEd|` (z = face d'appui → centre du groupe) en répartition
élastique (inertie polaire) : `VEd` n'est jamais simplement divisé par le
nombre de boulons. Le groupe P reste **centré** ; la traction parasite due à
la rotation est couverte par le facteur **0,8** sur `Fv,Rd`
(MSB P5 §4.2.1.2, SCI P358). L'entraxe des deux files sur le poteau est
`p3 = tw,S + 2·gA`, **symétrique par rapport à l'âme du poteau**.

#### Les deux cas d'appui — c'est le choix qui commande tout

- **Cas (a), sur l'AILE** : les deux files sont symétriques par construction,
  la résultante passe par l'âme du poteau, et l'attache vit **dans le plan**
  de cette âme. `VEd` et `VEd·z` y cheminent EN PLAN : le côté poteau
  n'apporte, sous l'effort tranchant seul, que la **pression diamétrale** de
  l'aile (`t = tfP`, Tab. 3.4) ou les **cordons** si les ailes A sont
  soudées. Deux particularités : il n'y a **aucune pince d'extrémité
  verticale** (le poteau continue au-dessus et en dessous — `αb` se réduit à
  `min(p1/(3d0) − 1/4 ; fub/fu ; 1)`), et le seul bord libre est le **bout
  d'aile**, `e = (bP − p3)/2`. La flexion locale de l'aile n'apparaît QUE
  lorsque les boulons sont tendus : **tronçon en T de l'aile NON RAIDIE**
  (§6.2.4, Tab. 6.2 méthode 2 ; `Σleff` du **Tableau 6.4**, toutes les
  rangées INTÉRIEURES) et **âme du poteau en traction transversale**
  (§6.2.6.3 avec `ω = 1`), actifs en tying et à l'ELU si `NEd > 0`.
- **Cas (b), dans l'ÂME** : l'attache est **perpendiculaire** à l'âme du
  poteau. `VEd` reste en plan dans l'âme — d'où le **cisaillement local**
  d'un élément support chargé d'un seul côté,
  `VEd/2 ≤ Av·fy,P/(√3·γM0)` avec
  `Av = twP·(5d + (n1 − 1)·p1 + min(p3/2 ; 5d))` (SCI P358, le modèle même de
  l'âme porteuse des modules poutre–poutre ; un poteau n'ayant pas de face
  horizontale, les bornes `zt` / `zb` du grand frère se réduisent à `5·d`),
  et la section nette au droit des trous. Mais toute traction sollicite l'âme
  **hors de son plan** : l'EN 1993-1-8 ne couvre pas ce composant pour une
  âme NON raidie, et l'outil le **déclare** au lieu de transposer un modèle
  qui ne tient pas.

#### Les quatre conditions GÉOMÉTRIQUES du poteau

1. **Le gauge dégage le congé du poteau** (cas a) : l'écrou est du côté
   intérieur de l'aile, au droit du congé, donc
   `p3 ≥ twP + 2·rP + dw` — pour un HEB 300 avec des M20 : `11 + 54 + 37 =
   102 mm`. C'est la raison des gauges usuels d'aile de poteau, **90 / 100 /
   120 / 140 mm** selon la série : le `p3 = 90 mm` usuel des ÂMES de poutre
   NE PASSE PAS sur l'aile d'un HEB 300. Alerte bloquante.
2. **L'écrou dégage le congé** (cas b) : `p3/2 + dw/2 ≤ (hP − 2·tfP)/2 − rP`.
3. **Le bout de l'aile A dégage le congé** (cas b) :
   `tw,S/2 + bA ≤ (hP − 2·tfP)/2 − rP` — pour un HEB 300 avec une âme portée
   de 8,6 mm, **`bA ≤ 99,7 mm` : une L100 ne passe pas, une L90 passe.**
4. **La semelle de la poutre passe entre les ailes** (cas b) :
   `bS ≤ hP − 2·tfP − 2·j` ; sinon il faut **recouper** les semelles en
   largeur depuis l'about, `rc ≥ (bP − twP)/2 + j − gh`. La recoupe en
   LARGEUR n'est **pas** un grugeage : les règles de stabilité locale de
   MSB P5 §4.2.5 portent sur une entaille en hauteur et ne sont pas activées
   (point tranché, voir plus bas). La variante « la poutre reste au-delà des
   bouts d'aile » du plat d'âme est **sans objet ici** : l'aile B d'une
   cornière devrait traverser tout le demi-poteau.

#### Vérifications et sources

- **Boulons** : groupe S en double cisaillement (Tab. 3.4) avec groupe
  excentré (MSB P5 §4.2.1.1) ; groupe P en simple cisaillement avec le
  facteur 0,8 ; traction et interaction V + T du groupe P (Tab. 3.4) ;
  glissement §3.9 (catégories B et C). En **tying**, la force de chaînage est
  dans le plan de l'âme portée : elle **cisaille** le groupe S,
  `NEd/n ≤ 2·αv·fub·A/γMu`.
- **Cornières** : pression diamétrale des ailes A et B (Tab. 3.4, deux
  directions et interaction quadratique) ; cisaillement brut avec le facteur
  **1,27** (MSB P5 §4.2.2, d'après ECCS n°126) ; net ; rupture de bloc
  §3.10.2(3) ; flexion des ailes B dans leur plan ; flexion des ailes A entre
  talon et file de boulons (bras de levier : hypothèse de l'outil) ;
  **tronçon en T** des ailes A (§6.2.4, Tab. 6.2 méthode 2 ; `leff` et `m` :
  MSB P5 §4.3.1.1) ; traction nette et rupture de bloc en traction des
  ailes B.
- **Âme portée** : pression diamétrale ; cisaillement brut / net ; rupture de
  bloc ; traction nette (hauteur participante `hc`) et bloc en traction ;
  grugeage (facultatif sur un poteau, 0 par défaut) : flexion + cisaillement
  et stabilité locale (MSB P5 §4.2.4 / §4.2.5).
- **Poteau porteur** : pression diamétrale de l'aile ou de l'âme ;
  cisaillement local brut et net de l'âme en cas (b) ; tronçon en T de l'aile
  et âme en traction transversale en cas (a), sous boulons tendus ;
  poinçonnement de l'aile ou de l'âme (Tab. 3.4).
- **Cordons** (ailes A ou B soudées) : méthode simplifiée §4.5.3.3 sur le
  groupe de cordons (cordon vertical sur toute la hauteur `hc` + retours
  `ℓh`), `fu = min(fu cornière ; fu support)` ; gorge minimale §4.5.2(2) ;
  longueur minimale §4.5.1(2) ; en tying, `fu/(√3·βw·γMu)`.

Le moment que l'attache apporte au poteau, `VEd·e_col` (`e_col = hP/2 + z` en
cas (a), `z` en cas (b)), est **affiché** en information dans « Paramètres
retenus » et au bloc HYPOTHÈSES de la note : il appartient au calcul
d'ÉLÉMENT du poteau (EC3-1-1 §6.2 et §6.3), l'outil ne le vérifie pas.

#### Annexe nationale belge

Les coefficients par défaut suivent la **NBN EN 1993-1-1 ANB** : γM0 = 1,00 ;
γM2 = 1,25. Sources croisées : documentation des annexes nationales de deux
logiciels (SCIA « Theory NA EN 1993 », Bentley STAAD « Belgian NA to EC3 ») —
le texte NBN, payant, n'a pas été consulté ici : à confirmer, valeurs
modifiables dans les paramètres avancés. Pour l'EN 1993-1-8, aucune
divergence belge relevée : valeurs recommandées (γM2 = 1,25 ; γM3 = 1,25 ;
γM3,ser = 1,1). **γMu = 1,1** (tying) est la valeur des guides
(MSB P5 §2.3, SCI P358) : l'EN 1993-1-8 ne la fixe pas et l'ANB n'a pas été
consultée — à confirmer.

#### Données de bibliothèque

Nuances : S355 : fu = 510 MPa (EN 1993-1-1 Tab. 3.1) ; « S355 (fu 490) » :
valeur d'autres références, à caler sur l'ANB. Profilés : catalogue HEA / HEB /
HEM / IPE ; « Personnalisé » = dimensions h, b, tw, tf, r saisies. Cornières :
dimensions usuelles EN 10056-1, à contrôler avec le catalogue du fournisseur ;
« Orientation » ne s'affiche que pour une cornière à ailes inégales (ou
Personnalisé) en vérification — la cornière proposée par le
prédimensionnement est à ailes égales. Boulons : As selon ISO 898, trous
normaux selon EN 1090-2, dm et dw usuels ISO 4014 et ISO 7089. Le poteau est
un profil en I ou H, âme verticale : un poteau CREUX (RHS / SHS / CHS) sort
du périmètre (autre composant, EN 1993-1-8 §7).

#### Limites du modèle (à justifier séparément le cas échéant)

""" + _liste(limites(True)) + """

#### Points TRANCHÉS SANS SOURCE FRANCHE — à relire par le bureau

Ils sont listés dans `docs/assemblages/POUTRE_POTEAU.md` §9 et
`docs/assemblages/ECARTS_ET_CORRECTIONS.md`. Ceux qui portent sur CE module :

- **P-1** : le NUMÉRO du tableau des longueurs efficaces de l'aile de poteau
  non raidie. L'outil cite le **Tableau 6.4** ; la consigne de l'étude citait
  le Tableau 6.6. Le texte NBN est payant et n'a pas été consulté. Ce qui est
  sûr : ce sont bien les formules de l'aile de poteau NON raidie (`2πm`,
  `4m + 1,25e`, `2p`, `p`), et ce sont celles que le module platine d'about
  applique déjà — c'est le numéro, pas la formule, qui est en cause ;
- **P-3** : la recoupe en LARGEUR des semelles n'est **pas** un grugeage ; les
  règles MSB P5 §4.2.4 / §4.2.5 ne sont pas activées par elle. Rien n'est
  publié pour une réduction de largeur d'aile à l'about ; l'âme seule
  transmet `VEd` (déjà vérifié) et la recoupe est au voisinage de la rotule ;
- **P-4** : aucune épaisseur minimale d'aile de poteau n'est imposée — ni
  l'EN 1993-1-8 ni les guides n'en donnent pour une attache simple ; ce sont
  la pression diamétrale et le tronçon en T qui décident ;
- **P-5** : l'âme du poteau en traction transversale est prise avec `ω = 1`
  et `beff,t,wc = Σleff` de l'aile, nature `EC + INT`. `ω` dépend du
  cisaillement du panneau d'âme, hors périmètre ici ; `ω = 1` est la valeur en
  l'absence de moment d'étage, donc la plus favorable. Un bureau qui veut
  être conservatif impose `ω` ;
- **P-6** : la flexion hors plan de l'âme du poteau en cas (b) n'est pas
  vérifiée, elle est déclarée (le tronçon en T suppose une aile encastrée sur
  une âme centrale ; une âme entre deux ailes ne s'y ramène pas) ;
- **P-7** : le gauge minimal `p3 ≥ twP + 2·rP + dw` est une règle géométrique
  de dégagement du congé, pas une clause normative (nature INT, alerte
  bloquante). Elle explique les gauges usuels mais ne remplace pas une table
  tabulée par série (SCI P358, non consultée) ;
- **P-8** : `et = 5·d` et `eb = min(p3/2 ; 5d)` pour le cisaillement local de
  l'âme du poteau. Supprimer les bornes `zt` / `zb` du grand frère est correct
  (le poteau continue) mais rend la règle entièrement dépendante du plafond
  `5·d`, cité de mémoire dans le dépôt depuis le 21/09/2026 ;
- **P-9** : `γMu = 1,1` est la valeur du guide, modifiable en saisie.

**Écart assumé, propre à ce module** : la liste du §8.2 de l'étude ne portait
pas de ligne « boulons du groupe S en cisaillement sous la force de
chaînage », que son §7.2 demande pourtant (« OUI pour le groupe S »).
L'outil l'ÉCRIT (`bvT_S`) : laisser le groupe S sans vérification en tying
serait un défaut non conservatif.

#### Sources

*Steel Buildings in Europe – Multi-Storey Steel Buildings, Part 5: Joint
Design* (ArcelorMittal / Peiner Träger / Corus, contenu CTICM et SCI,
06/2009) — §4 doubles cornières et §4.4 Worked Example (benchmark du module),
§2.3 tying ; *SCI P358, Joints in Steel Construction: Simple Joints to
Eurocode 3* ; *ECCS n°126* ; EN 1993-1-8:2005 ; EN 1993-1-1:2005.
"""


def _notations(texte):
    """Le paragraphe « Notations » de l'onglet : la ligne « Notations » du pied
    de la note, mot pour mot (une source : notation.py) — les définitions que
    les cotes, réduites à leur symbole, ne portent plus."""
    return "\n#### Notations\n\n" + texte.split(" : ", 1)[1] + "\n"


METHODE += _notations(NOTATIONS)
