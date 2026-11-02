# -*- coding: utf-8 -*-
"""Onglet Méthode : le modèle mécanique du plat d'âme sur poteau, ses
sources et ses limites — en toutes lettres.

Les LIMITES de validité ont une seule source, ``..textes`` pour les huit
communes à la famille et ce fichier pour la septième : l'onglet Méthode
les écrit en liste, et l'onglet Vérifications les reprend mot pour mot dans
le repli « Périmètre de la vérification » (``ecran_resultats.NON_REVERIFIES``
les lit ici). Elles sont celles de l'étude, `POUTRE_POTEAU.md` §8.4."""
from .. import textes
from .notation import NOTATIONS

# Les neuf limites du §8.4 ont UNE SOURCE, ``..textes`` (consolidation du
# 02/11/2026) : les six premières et les deux dernières sont celles de la
# FAMILLE, mot pour mot. Seule la septième est rédigée ici — ``LIMITE_AME``
# ne vaut qu'en cas (b), et chaque attache la dit à sa façon : ce qui
# sollicite l'âme hors de son plan n'est pas le même (docstring de
# ``..textes``).
LIMITE_AME = ("flexion HORS PLAN de l'âme du poteau sous qx = 3·MS/hp² et sous la force de chaînage — "
              "non couverte par l'EN 1993-1-8 pour une âme non raidie")


def limites(ame=True):
    """Les limites de validité pour le cas d'appui courant (``ame`` : cas b) —
    les neuf du §8.4, assemblées par la source unique de la famille."""
    return textes.limites(LIMITE_AME, ame)


def _liste(items):
    return "\n".join("- " + t + " ;" for t in items)


METHODE = """
#### Ce que l'outil vérifie

Vérification ELU selon EN 1993-1-8 et EN 1993-1-1 (γM1 : ANB belge) ; modèles
MSB Part 5 §3 (fin plate), SCI P358, ECCS n°126 ; **tying** (force de
chaînage, situation accidentelle) selon MSB Part 5 §2.3. Benchmark : le
Worked Example publié du guide, rejoué à chaque ouverture, et quatre cas
calculés à la main (étude du 01/11/2026, `docs/assemblages/POUTRE_POTEAU.md`).

#### Modèle mécanique

Assemblage **articulé** : la rotule est à la **face d'appui du poteau**. Le
plat vertical, soudé par deux cordons pleine hauteur, est boulonné à l'âme de
la poutre portée par **un seul plan de cisaillement**. Le groupe de boulons
reprend l'effort tranchant **et** le moment d'excentricité `MS = VEd·z + |MEd|`
(z = face d'appui → centre du groupe), en répartition élastique (inertie
polaire). Modèle du guide *Steel Buildings in Europe – Part 5: Joint Design*
§3 (fin plate), identique à SCI P358 et ECCS n°126.

#### Les deux cas d'appui — c'est le choix qui commande tout

- **Cas (a), sur l'AILE** : l'attache est centrée sur l'âme du poteau et vit
  **dans son plan**. `VEd` et le moment `VEd·z` y cheminent EN PLAN : il n'y a
  pas de flexion hors plan de l'aile sous l'effort tranchant seul, et aucune
  vérification locale de l'aile n'est demandée par MSB P5 §3.2 ni par
  SCI P358 au-delà du **cordon**. Le groupe « Poteau porteur » est donc
  **vide** : la section du poteau se vérifie dans son propre calcul
  (EC3-1-1 §6.2). L'attache est supposée **centrée** sur l'âme ; aucune clé
  d'excentricité n'est offerte (une attache décentrée sortirait du domaine).
- **Cas (b), dans l'ÂME** : l'attache est **perpendiculaire** à l'âme du
  poteau. `VEd` reste en plan dans l'âme — d'où le **cisaillement local**
  d'un élément support chargé d'un seul côté, `VEd/2 ≤ twP·hp·fy,P/(√3·γM0)`
  (SCI P358, le modèle même de l'âme porteuse des modules poutre–poutre).
  Mais la composante `qx = 3·MS/hp²` et toute traction sollicitent l'âme
  **hors de son plan** : l'EN 1993-1-8 ne couvre pas ce composant pour une
  âme NON raidie, et l'outil le **déclare** au lieu de transposer un modèle
  qui ne tient pas. Géométrie du cas (b) : largeur droite entre congés
  `hP − 2·tfP − 2·rP`, passage de la semelle `hP − 2·tfP − 2·j`, et — si la
  poutre ne passe pas — **recoupe** des semelles en largeur depuis l'about,
  `rc ≥ (bP − twP)/2 + j − gh`. La recoupe en LARGEUR n'est **pas** un
  grugeage : les règles de stabilité locale de MSB P5 §4.2.5 portent sur une
  entaille en hauteur et ne sont pas activées (point tranché, voir plus bas).

#### Vérifications et sources

- **Boulons** : cisaillement Tab. 3.4 + groupe excentré (MSB P5 §3.2) ;
  glissement §3.9 (catégories B et C) ; en **tying**, la force de chaînage est
  dans le plan du plat : elle **cisaille** les boulons, `NEd/n ≤ αv·fub·A/γMu`.
- **Plat** : pression diamétrale (Tab. 3.4, directions ver/hor et interaction
  quadratique — MSB P5 §3.2) ; cisaillement brut `hp·tp/1,27 · fy/√3/γM0`
  (le 1,27 couvre l'interaction avec la flexion — MSB P5 §3.2 / ECCS) ; net ;
  rupture de bloc §3.10.2(3) ; **flexion** sans objet si `hp ≥ 2,73·z`, sinon
  `Wel·fy/γM0` (MSB P5 §3.2.3) ; **déversement du plat LONG** (`z > tp/0,15`) :
  `Mb,Rd = Wel·pb(λLT)/γM1` avec `λLT = 2,8·√(z·hp/(1,5·tp²))` et la courbe de
  BS 5950-1 **Annexe B.2.1** (celle de sa Table 17, E = 205 000 MPa) — méthode
  SCI P358 ; traction section nette EC3-1-1 §6.2.3(2) et rupture de bloc en
  traction §3.10.2(2).
- **Âme portée** : pression diamétrale ; cisaillement brut / net ; rupture de
  bloc avec `(n1 − 1)·d0` si la poutre n'est **pas** grugée, `(n1 − 0,5)·d0`
  sinon (MSB P5 §3.2.5.1) ; traction nette et bloc en traction ; grugeage
  (facultatif sur un poteau, 0 par défaut) : flexion + cisaillement et
  stabilité locale — mêmes règles que le module doubles cornières
  (MSB P5 §4.2.4 / §4.2.5).
- **Poteau porteur** : cas (b) seulement — cisaillement local de l'âme
  `VEd/2 ≤ twP·hp·fy,P/(√3·γM0)` (SCI P358, un seul côté chargé). Cas (a) :
  rien, et c'est écrit.
- **Soudure** : deux cordons verticaux pleine hauteur, méthode simplifiée
  §4.5.3.3 sous `qz = VEd/2hp` et `qx = 3·MS/hp² + |NEd|/2hp` ; en tying
  `qx = NEd/2hp` avec `fu/(√3·βw·γMu)` ; gorge minimale §4.5.2 ;
  recommandation de **pleine résistance** `a ≥ tp·fy·βw·γM2/(2·fu·γM0)`
  (dérivation §4.5.3.3), signalée en alerte.

Le moment que l'attache apporte au poteau, `VEd·e_col` (`e_col = hP/2 + z` en
cas (a), `z` en cas (b)), est **affiché** en information dans « Paramètres
retenus » et au bloc HYPOTHÈSES de la note : il appartient au calcul
d'ÉLÉMENT du poteau (EC3-1-1 §6.2 et §6.3), l'outil ne le vérifie pas.

#### Annexe nationale belge

Les coefficients par défaut suivent la **NBN EN 1993-1-1 ANB** : γM0 = 1,00,
**γM1 = 1,10** (au lieu de 1,00 recommandé — il ne sert ici qu'au déversement
du plat long), γM2 = 1,25. Sources croisées : documentation des annexes
nationales de deux logiciels (SCIA « Theory NA EN 1993 », Bentley STAAD
« Belgian NA to EC3 ») — le texte NBN, payant, n'a pas été consulté ici : à
confirmer, valeurs modifiables dans les paramètres avancés. Pour
l'EN 1993-1-8, aucune divergence belge relevée : valeurs recommandées
(γM2 = 1,25 ; γM3 = 1,25 ; γM3,ser = 1,1). **γMu = 1,1** (tying) est la valeur
des guides (MSB P5 §2.3, SCI P358) : l'EN 1993-1-8 ne la fixe pas et l'ANB
n'a pas été consultée — à confirmer.

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

- **P-2** : aucune vérification locale de l'aile sous `VEd` seul (groupe
  « Poteau porteur » vide en cas (a)). Aucun modèle publié n'a été trouvé pour
  une aile de poteau chargée par un plat soudé sur chant au droit de l'âme ;
  le raisonnement est mécaniquement net, mais c'est un raisonnement, pas une
  clause. L'alternative — inventer une aire de cisaillement `Av = f(tfP, hp)` —
  a été refusée ;
- **P-3** : la recoupe en LARGEUR des semelles n'est **pas** un grugeage ; les
  règles MSB P5 §4.2.4 / §4.2.5 ne sont pas activées par elle. Rien n'est
  publié pour une réduction de largeur d'aile à l'about ; l'âme seule
  transmet `VEd` (déjà vérifié) et la recoupe est au voisinage de la rotule ;
- **P-4** : aucune épaisseur minimale d'aile de poteau n'est imposée — ni
  l'EN 1993-1-8 ni les guides n'en donnent pour une attache simple ;
- **P-6** : la flexion hors plan de l'âme du poteau en cas (b) n'est pas
  vérifiée, elle est déclarée (le tronçon en T suppose une aile encastrée sur
  une âme centrale ; une âme entre deux ailes ne s'y ramène pas) ;
- **P-9** : `γMu = 1,1` est la valeur du guide, modifiable en saisie.

#### Sources

*Steel Buildings in Europe – Multi-Storey Steel Buildings, Part 5: Joint
Design* (ArcelorMittal / Peiner Träger / Corus, contenu CTICM et SCI,
06/2009) — §3 fin plate et §3.4 Worked Example (benchmark du module), §2.3
tying ; *SCI P358, Joints in Steel Construction: Simple Joints to Eurocode 3* ;
*ECCS n°126* ; EN 1993-1-8:2005 ; EN 1993-1-1:2005 ; BS 5950-1:2000
Annexe B.2.1 (courbe de déversement des plats longs).
"""


def _notations(texte):
    """Le paragraphe « Notations » de l'onglet : la ligne « Notations » du pied
    de la note, mot pour mot (une source : notation.py) — les définitions que
    les cotes, réduites à leur symbole, ne portent plus."""
    return "\n#### Notations\n\n" + texte.split(" : ", 1)[1] + "\n"


METHODE += _notations(NOTATIONS)
