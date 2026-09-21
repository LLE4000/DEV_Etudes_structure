# -*- coding: utf-8 -*-
"""Onglet Méthode : le modèle mécanique du plat d'âme, ses sources et ses
limites — en toutes lettres."""

METHODE = """
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
  `Wel·fy/γM0` (MSB P5 §3.2.3) ; **déversement du plat LONG** (`z > tp/0,15`) :
  `Mb,Rd = Wel·pb(λLT)/γM1` avec `λLT = 2,8·√(z·hp/(1,5·tp²))` et la courbe
  de BS 5950-1 **Annexe B.2.1** (celle de sa Table 17, E = 205 000 MPa) —
  méthode SCI P358 pour les plats longs.
- **Âme portée** : pression diamétrale ; cisaillement brut / net ; rupture de
  bloc avec `(n1 − 1)·d0` si la poutre n'est **pas** grugée, `(n1 − 0,5)·d0`
  sinon (MSB P5 §3.2.5.1) ; grugeage : flexion + cisaillement et stabilité
  locale — mêmes règles que le module doubles cornières (MSB P5 §4.2.4 /
  §4.2.5).
- **Âme porteuse** : cisaillement local `VEd/2 ≤ tw·hp·fy/√3/γM0` (un seul
  côté chargé — même modèle que le module doubles cornières, SCI P358).
- **Soudure** : deux cordons verticaux pleine hauteur, méthode simplifiée
  §4.5.3.3 sous `qz = VEd/2hp` et `qx = 3·MS/hp² + |NEd|/2hp` ; gorge
  minimale §4.5.2 ; recommandation de **pleine résistance**
  `a ≥ tp·fy·βw·γM2/(2·fu·γM0)` (dérivation §4.5.3.3), signalée en alerte.

#### Annexe nationale belge

Les coefficients par défaut suivent la **NBN EN 1993-1-1 ANB** : γM0 = 1,00,
**γM1 = 1,10** (au lieu de 1,00 recommandé — il ne sert ici qu'au
déversement du plat long), γM2 = 1,25. Sources croisées : documentation des
annexes nationales de deux logiciels (SCIA « Theory NA EN 1993 », Bentley
STAAD « Belgian NA to EC3 ») — le texte NBN, payant, n'a pas été consulté
ici : à confirmer, valeurs modifiables dans les paramètres avancés. Pour
l'EN 1993-1-8, aucune divergence belge relevée : valeurs recommandées
(γM2 = 1,25 ; γM3 = 1,25 ; γM3,ser = 1,1).

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
*ECCS n°126* ; EN 1993-1-8:2005 ; EN 1993-1-1:2005 ; BS 5950-1:2000
Annexe B.2.1 (courbe de déversement des plats longs).
"""
