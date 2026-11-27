# -*- coding: utf-8 -*-
"""Onglet Méthode : le modèle mécanique du plat d'âme, ses sources et ses
limites — en toutes lettres."""
from .notation import NOTATIONS
from acier.assemblages import materiau as MT

METHODE = """
#### Ce que l'outil vérifie

Vérification ELU selon EN 1993-1-8 et EN 1993-1-1 (γM1 : ANB belge) ; modèles
MSB Part 5 §3 (fin plate), SCI P358, ECCS n°126. Benchmark : Worked Example
publié du guide, rejoué à chaque ouverture.

#### Modèle mécanique

Assemblage **articulé** : la rotule est à la **face de l'âme porteuse**. Le plat
vertical, soudé par deux cordons pleine hauteur, est boulonné à l'âme de la
poutre portée par **un seul plan de cisaillement**. Le groupe de boulons
reprend l'effort tranchant **et** le moment d'excentricité `MS = VEd·z`
(z = face d'appui → centre du groupe), en répartition élastique (inertie
polaire). Le plat, l'âme portée, la soudure et l'âme porteuse sont vérifiés
en conséquence. Modèle du guide *Steel Buildings in Europe – Part 5: Joint
Design* §3 (fin plate), identique à SCI P358 et ECCS n°126.

#### Vérifications et sources

- **Boulons** : cisaillement Tab. 3.4 + groupe excentré (MSB P5 §3.2) ;
  glissement §3.9 (catégories B et C).
- **Plat** : pression diamétrale (Tab. 3.4, directions ver/hor et interaction
  quadratique — MSB P5 §3.2) ; cisaillement brut `hp·tp/1,27 · fy/√3/γM0`
  (le 1,27 couvre l'interaction avec la flexion — MSB P5 §3.2 / ECCS) ; net ;
  rupture de bloc §3.10.2 ; **flexion** sans objet si `hp ≥ 2,73·z`, sinon
  `Wel·fy/γM0` (MSB P5 §3.2.3) ; **déversement du plat LONG** — au sens du
  SCI P358 : `zp > tp/0,15`, `zp` de la face d'appui à la PREMIÈRE file de
  boulons — : `VRd = min(Wel·χLT·fy/(0,6·z·γM1) ; Wel·fy/(z·γM0))`, `χLT` par
  l'EN 1993-1-1 §6.3.2.2 (courbe d, αLT = 0,76) avec
  `λ̄LT = 2,8/(π·√(E/fy))·√(zp·hp/(1,5·tp²))` (SCI P358, Check 3 ; son
  Tableau 5.2 est retrouvé). Le MSB P5 compte le plat long dès que `z > tp/0,15`
  (`z` au CENTRE du groupe) : entre les deux seuils, la section ABCD du MSB
  est calculée et, au-delà de 100 %, le statut passe à « À VÉRIFIER ». Un
  plat long sur une poutre NON maintenue : « À VÉRIFIER » (le guide ne
  l'admet pas sans essais).
- **Âme portée** : pression diamétrale ; cisaillement brut / net ; rupture de
  bloc avec `(n1 − 1)·d0` si la poutre n'est **pas** grugée, `(n1 − 0,5)·d0`
  sinon (MSB P5 §3.2.5.1) ; grugeage : flexion + cisaillement et stabilité
  locale — mêmes règles que le module doubles cornières (MSB P5 §4.2.4 /
  §4.2.5).
- **Âme porteuse** : cisaillement local `VEd/2 ≤ tw·hp·fy/√3/γM0` (un seul
  côté chargé — même modèle que le module doubles cornières, SCI P358) ;
  poinçonnement par le plat (voir plus bas).
- **Soudure** : deux cordons verticaux, méthode simplifiée §4.5.3.3 sous
  `qz = VEd/(2·Lw)` et `qx = 3·MS/Lw² + |NEd|/(2·Lw)`, `Lw = hp − 2·a` ; gorge
  minimale §4.5.2 ; **pleine résistance** `a ≥ tp·fy·βw·γM2/(√2·fu·γM0)`
  (dérivation §4.5.3.3 : le plat plastifie avant le cordon) — sous cette
  gorge, alerte et statut « À VÉRIFIER (articulation à justifier) ».

#### Règles reprises du SCI P358 (lot de clôture du 27/11/2026)

- **Détail éprouvé de l'articulation** (SCI P358 §5.3 et Check 1) : pinces
  du plat et de l'âme `≥ 2·d`, boulons 8.8 non précontraints, cordons de
  pleine résistance. Hors de ce détail la saisie reste admise — le Tableau
  3.3 de l'EN 1993-1-8 décide de la géométrie — mais le statut devient
  « ASSEMBLAGE À VÉRIFIER (articulation à justifier : …) ». Le cas par défaut
  tient le détail (M20, e1 = 40 mm, p1 = 55 mm).
- **Cordons** : longueur efficace `hp − 2·a` par cordon (EN 1993-1-8
  §4.5.1(1)) ; une longueur efficace sous `max(30 mm ; 6·a)` est refusée.
- **Grugeage court** (SCI P358, Check 5) : deux files de boulons et
  `xN = c − (e2,b + p2) < 2·d` → la section au droit de la 2ᵉ file se vérifie
  avec le bras `max(gh + c ; gh + e2,b + p2)`.
- **Section ABCD de l'âme portée** au droit d'un plat long (SCI P358,
  Check 4) : cisaillement de AB et de BC, flexion de la section CD.
- **Poinçonnement de l'âme porteuse** par le plat (SCI P358, Check 10, forme
  rigoureuse) : `VEd·z + |NEd|·hp/6 ≤ t2·fu,2·hp²/(6·γM2)`.
- **Effort tranchant inversé** (`VEd < 0`) : les ruptures de bloc et les
  sections sont celles d'un effort descendant — le sens inverse n'est pas
  modélisé, statut « À VÉRIFIER ».

#### Annexe nationale belge

Les coefficients par défaut suivent la **NBN EN 1993-1-1 ANB** : γM0 = 1,00,
**γM1 = 1,10** (au lieu de 1,00 recommandé — il ne sert ici qu'au
déversement du plat long), γM2 = 1,25. Sources croisées : documentation des
annexes nationales de deux logiciels (SCIA « Theory NA EN 1993 », Bentley
STAAD « Belgian NA to EC3 ») — le texte NBN, payant, n'a pas été consulté
ici : à confirmer, valeurs modifiables dans les paramètres avancés. Pour
l'EN 1993-1-8, aucune divergence belge relevée : valeurs recommandées
(γM2 = 1,25 ; γM3 = 1,25 ; γM3,ser = 1,1).

#### Données de bibliothèque

Nuances : {NUANCES} Profilés : catalogue HEA / HEB /
HEM / IPE ; « Personnalisé » = dimensions h, b, tw, tf, r saisies.

#### Limites du modèle (à justifier séparément le cas échéant)

- flexion **hors plan** de l'âme porteuse sous le moment `VEd·z` (attache
  d'un seul côté) et sous NEd de traction — négligée par les guides pour
  les détails courants, à examiner pour une âme mince ou un z inhabituel ;
- cisaillement global, flexion et déversement de la poutre porteuse ;
- torsion induite par une attache d'un seul côté ;
- deux poutres en vis-à-vis sur la même âme (interaction non traitée) ;
- efforts d'arrachement accidentels (tying) : non couverts par ce module.

#### Sources

*Steel Buildings in Europe – Multi-Storey Steel Buildings, Part 5: Joint
Design* (ArcelorMittal / Peiner Träger / Corus, contenu CTICM et SCI,
06/2009) — §3 fin plate et §3.4 Worked Example (benchmark du module) ;
*SCI P358, Joints in Steel Construction: Simple Joints to Eurocode 3* ;
*ECCS n°126* ; EN 1993-1-8:2005 ; EN 1993-1-1:2005 (§6.3.2.2 : χLT du plat
long).
""".replace("{NUANCES}", MT.texte_nuances())     # la règle belge des nuances, une source (materiau.py)


def _notations(texte):
    """Le paragraphe « Notations » de l'onglet : la ligne « Notations » du pied
    de la note, mot pour mot (une source : notation.py) — les définitions que
    les cotes, réduites à leur symbole, ne portent plus (consolidation du
    01/11/2026, point 4)."""
    return "\n#### Notations\n\n" + texte.split(" : ", 1)[1] + "\n"


METHODE += _notations(NOTATIONS)
