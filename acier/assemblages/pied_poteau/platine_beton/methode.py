# -*- coding: utf-8 -*-
"""Onglet Méthode : le modèle mécanique du pied de poteau, ses deux niveaux,
ses sources et ses limites — en toutes lettres."""

METHODE = """
#### Modèle mécanique

Pied de poteau **encastré ou articulé** selon ce que donne le calcul : une
platine rectangulaire centrée sur le poteau, scellée au mortier sur un
massif de béton, tenue par des tiges d'ancrage coulées en place disposées en
grille. Les efforts `N` (compression > 0), `Mx`, `My`, `Vx`, `Vy` et `T`
sont ceux de la section de pied du poteau. La méthode est celle de l'étude
*docs/assemblages/PIED_POTEAU_ETUDE.md* : **sans éléments finis, sans
maillage**, avec les composants de l'EN 1993-1-8 et les ancrages de
l'EN 1992-4.

1. **Aire efficace** (EC3 §6.2.5) : bande `c = tp·√(fy/(3·fjd·γM0))` autour
   de chaque paroi du profilé, découpée par la platine ; `fjd = βj·fcd·√(Ac1/Ac0)`
   avec `βj = 2/3`, `Ac0` = platine, `Ac1` concentrique limitée par le
   massif, par `3·Ac0` et par la hauteur (diffusion 2:1) — EN 1992-1-1 §6.7.
2. **Répartition** : platine rigide sur un lit **sans traction** (raideur
   `kc = Ecm/(1,275·√(beff·leff))`, k13 de l'EC3 §6.3.4) et tiges tendues
   seulement (`kb = E·k16`). L'équilibre de `N`, `Mx`, `My` donne la zone
   comprimée, l'axe neutre, la résultante `C`, la pression maximale et
   l'effort de **chaque** tige — c'est la dérivation des efforts que
   l'EN 1992-4 (section 6) impose pour les fixations.
3. **Résistance, niveau 1 — le verdict** : capacités uniaxiales `Mx,Rd(N)`,
   `My,Rd(N)` au modèle de la **Table 6.7** (EC3 §6.2.8.3 : résultante
   comprimée au centre de la bande de la paroi comprimée, tiges opposées à
   leur résistance ou en équilibre), puis interaction **linéaire**
   `|Mx|/Mx,Rd + |My|/My,Rd ≤ 1` — conservative parce que le domaine
   résistant est convexe (théorème statique).
4. **Résistance, niveau 2 — information** : le domaine **exact** `(Mx, My)`
   à `N` donné, par balayage de l'axe neutre (bloc `fjd` sur l'aire efficace
   comprimée, tiges du côté tendu à leur résistance). C'est une borne
   inférieure (champ statiquement admissible), **modèle d'ingénieur non
   codifié** : elle renseigne, elle ne décide pas.

#### Composants vérifiés et sources

- **Béton comprimé** : `N ≤ fjd·Aeff` (EC3 §6.2.8.2) ; pression maximale de
  la répartition ≤ `fjd` (information).
- **Platine comprimée** : flexion élastique du débord sous la répartition
  réelle, `mEd ≤ tp²·fy/(6·γM0)` — la bande `c` est élastique par
  construction (EC3 §6.2.5).
- **Platine tendue** : tronçon en T de la rangée de la tige la plus
  sollicitée (EC3 §6.2.4, Tab. 6.2 : modes 1-2-3, sans effet de levier si
  `Lb > Lb*`), longueurs efficaces **par tige** dérivées des Tab. 6.4 / 6.6
  (la somme sur deux tiges redonne la rangée extérieure à l'aile). Tiges
  **de coin** (au-delà de l'about d'aile) : longueur plafonnée à `0,25·b`
  par tige ; tiges **latérales** : appui réduit à `(tf + 4m)/2` — modèles
  d'ingénieur signalés en alerte.
- **Tiges, acier** : traction `min(0,9·fub·As/γM2 ; As·fuk/γMs,N)`
  (EC3 Tab. 3.4 ; EC2-4 §7.2.1.3) ; cisaillement EC3 §6.2.2 (`αb = 0,44 −
  0,0003·fyb`, frottement `Cf,d·N` déduit) et EC2-4 §7.2.2.3 (sans
  frottement, bras de levier si `tg > d/2` ou `fg < 30 MPa`) ; interaction.
- **Tiges, béton** (EC2-4, tiges coulées en place) : cône du groupe tendu
  (§7.2.1.4, `k1 = 7,7 / 11`, `ψM,N = 1`), arrachement de la tête (§7.2.1.5)
  ou adhérence de la crosse (EC3 §6.2.6.12 sur EC2 §8.4.2, `η1 = 0,7`),
  rupture du bord (§7.2.2.5, rangée voisine du bord, trous normaux),
  pry-out (§7.2.2.4), interaction `βN^1,5 + βV^1,5 ≤ 1` (§7.2.3.2).
- **Poteau** : aile (ou paroi) comprimée `Fc,fc,Rd = Mc,Rd/(h − tf)` contre
  l'effort de sa bande (EC3 §6.2.6.7 par analogie).
- **Cordons** poteau / platine : méthode directionnelle (EC3 §4.5.3.2),
  groupe « en ligne », point le plus sollicité, `σ⊥ = τ⊥ = σn/√2`.
- **Rigidité** (information) : `Sj,ini = E·z²/Σ(1/ki)` avec k13, k15, k16
  (EC3 §6.3.4) ; seuil « rigide » `30·E·Ic/Lc` (ossature non contreventée) ou
  `48·E·Ic/Lc` (EC3 §5.2.2.5).

#### Annexe nationale belge et coefficients

γM0 = 1,00 (NBN EN 1993-1-1 ANB) ; γM2 = 1,25 ; γc = 1,5 et **αcc = 0,85**
(hypothèse ANB pour fcd — à confirmer) ; γMc = 1,5 (EN 1992-4, Tab. 4.1) ;
γMs,N = max(1,4 ; 1,2·fuk/fyk), γMs,V = max(1,25 ; fuk/fyk). Toutes ces
valeurs sont modifiables dans les paramètres avancés.

#### Ce qui est volontairement conservatif

`Ac0` = platine pour `√(Ac1/Ac0)` ; `ψM,N = 1` ; crosse non comptée ;
rupture de bord pour la résultante entière sur le bord le plus défavorable ;
cordons supposés transmettre aussi la compression ; niveau 1 : seule la
bande de la paroi comprimée résiste (au-delà de `N = 2·FC,Rd`, alerte — le
niveau 2 renseigne).

#### Limites du modèle (à justifier séparément le cas échéant)

- fendage et éclatement latéral (EC2-4 §7.2.1.7 et §7.2.1.8) quand un bord
  du massif est à moins de `1,5·hef` — signalés, non calculés ;
- tiges post-installées (valeurs ETA), raidisseurs, bêche de cisaillement,
  profils creux circulaires — hors V1 ;
- flambement et classification de la section du poteau ; poinçonnement du
  massif au-delà de l'EN 1992-1-1 §6.7 ; frottement platine / mortier sous
  soulèvement (ignoré) ;
- flexion biaxiale : le niveau 2 n'a pas d'exemple publié — validation par
  éléments finis hors outil prévue (lot C) avant toute promotion.

#### Sources

EN 1993-1-8:2005 §6.2.4, §6.2.5, §6.2.6, §6.2.8, §6.3.4, §4.5.3 ;
EN 1992-1-1:2004 §6.7, §8.4.2 ; EN 1992-4:2018 §7.2 ; *Steel Buildings in
Europe – Part 5: Joint Design* (pied articulé, aire efficace) ; SCI P398
(pied encastré) ; étude de méthode du bureau (30/10/2026).
"""
