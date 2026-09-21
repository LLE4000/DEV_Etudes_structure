# -*- coding: utf-8 -*-
"""Onglet « Méthode » : le texte de l'outil de référence, repris intégralement."""

METHODE = """
## Modèle mécanique

L'assemblage est nominalement articulé : il transmet l'effort tranchant VEd, et la ligne de transfert (rotule) est placée à la face de l'âme de la poutre principale (MSB Part 5 §4.2.1.1, d'après ECCS n°126). L'outil ne calcule aucune rigidité en rotation et ne transforme jamais l'assemblage en encastrement.

### Cheminement de l'effort

- Âme de la poutre secondaire → groupe S (boulons en double cisaillement, ou cordons) → ailes B des deux cornières, VEd/2 par cornière.
- Talon de chaque cornière → aile A → groupe P (boulons en simple cisaillement, ou cordons) → âme de la poutre principale.

### Excentricités prises en compte

- z, entre la face de l'âme porteuse et le centre du groupe S : MS = VEd·z + |MEd|. Les efforts par boulon viennent d'une répartition élastique avec l'inertie polaire Ip = Σ(x² + y²) ; le boulon d'angle le plus chargé est vérifié. VEd n'est jamais simplement divisé par le nombre de boulons.
- Groupe P : cisaillement centré (VEd/2)/n par boulon, avec le facteur 0,8 sur Fv,Rd qui couvre la traction parasite due à la rotation (MSB Part 5 §4.2.1.2, SCI P358). Option conservatrice : moment (VEd/2)·eP repris par chaque cornière dans son plan.
- Poutre grugée : flexion de la section réduite sous VEd·(gh + ln), avec réduction si VEd dépasse 0,5·Vpl,N,Rd, et règle de stabilité locale du grugeage.
- NEd : cisaillement horizontal des boulons S, traction des boulons P, tronçon en T équivalent des ailes A, sections nettes et ruptures de bloc en traction. HEd : cisaillement horizontal des boulons P et traction par le couple HEd·z/p3. MEd : moment parasite, pris en valeur absolue.

### Trois natures de vérification

- **EC** — formule donnée directement par l'Eurocode (Tableaux 3.3 et 3.4, §3.9, §3.10.2, §4.5.3.3, §6.2.4 de l'EN 1993-1-8 ; §6.2.3, §6.2.5, §6.2.6 de l'EN 1993-1-1).
- **COMP** — modèle complémentaire reconnu : répartition élastique du groupe excentré, facteur 0,8, coefficient 1,27 sur le cisaillement brut des cornières, section nette en cisaillement, poutre grugée, cisaillement local de l'âme porteuse (MSB Part 5, ECCS n°126, SCI P358).
- **INT** — interprétation propre à l'outil, signalée comme telle : bras de levier de la flexion des ailes A, limites S275 appliquées à S235 pour la stabilité du grugeage.

### Ce que l'outil ne vérifie pas

- Cisaillement global, flexion, déversement et torsion de la poutre principale ; cumul du cisaillement local avec le cisaillement global.
- Flexion hors plan de l'âme porteuse sous NEd de traction ; flexion hors plan des cornières sous HEd.
- Deux poutres secondaires en vis-à-vis partageant les mêmes boulons.
- Stabilité d'ensemble d'une poutre grugée non maintenue au déversement ; stabilité du grugeage pour une nuance supérieure à S355.
- Méthode directionnelle des soudures (§4.5.3.2) : seule la méthode simplifiée, plus sévère, est utilisée.

### Données de bibliothèque

Profilés : dimensions nominales h, b, tw, tf, r ; l'aire est recalculée par A = 2b·tf + (h − 2tf)·tw + (4 − π)·r². Cornières : dimensions usuelles EN 10056-1. Boulons : As selon ISO 898, trous normaux d + 1 (M12), d + 2 (M16 à M24), d + 3 (à partir de M27) selon EN 1090-2, dm et dw usuels ISO 4014 et ISO 7089. Aciers : EN 1993-1-1 Tableau 3.1 pour t ≤ 40 mm, βw selon EN 1993-1-8 Tableau 4.1. Coefficients partiels et résistances des aciers sont à caler sur l'ANB applicable ; la nuance « Personnalisé » permet toute autre valeur.

### Hypothèse sur les cordons

Longueur efficace des cordons prise égale à leur longueur totale (cordon vertical sur toute la hauteur Lc + retours) : hypothèse de cordons pleins sur toute leur longueur, EN 1993-1-8 §4.5.1.

### Sources

- EN 1993-1-8:2005, Calcul des assemblages ; EN 1993-1-1:2005, Règles générales.
- Steel Buildings in Europe – Multi-Storey Steel Buildings – Part 5: Joint Design (ArcelorMittal, Peiner Träger, Corus ; CTICM et SCI), noté MSB Part 5 : chapitre 4 « Double angle web cleats » et exemples de calcul utilisés pour le benchmark.
- ECCS n°126, European Recommendations for the Design of Simple Joints in Steel Structures ; SCI P358, Joints in Steel Construction: Simple Joints to Eurocode 3 : cités comme secondes sources des modèles complémentaires.
"""
