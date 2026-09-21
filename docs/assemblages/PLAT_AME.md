# Assemblage poutre–poutre par plat d'âme soudé (fin plate) — v1.0

Deuxième assemblage du module ACIER → ASSEMBLAGES, développé le 21/09/2026
avec la même identité graphique, la même ergonomie et la même qualité de
note de calcul que le module doubles cornières.

## Pourquoi celui-là

La consigne : choisir seul, parmi les trois familles d'attaches simples les
plus courantes en charpente métallique (doubles cornières d'âme, plat d'âme
soudé « fin plate », platine d'about flexible), l'assemblage le plus
pertinent à développer ensuite — sans assemblage exotique, sans rigide
complexe. Le plat d'âme s'est imposé :

- c'est, avec les doubles cornières, l'attache articulée la plus posée en
  Europe continentale (une seule pièce, montage rapide, une seule face) ;
- il vit dans la MÊME famille poutre–poutre : porteuse, portée, grugeage,
  jeu gh, décalage Δz — tout le kit de dessin et toute l'ergonomie se
  réutilisent sans forcer le modèle mécanique ;
- sa validation était déjà à portée : le benchmark BM4 du module doubles
  cornières transposait (×2) le Worked Example FIN PLATE du guide « Steel
  Buildings in Europe – Part 5 » §3.4 — ce module le rejoue EN DIRECT ;
- son moteur est réellement différent : un seul plan de cisaillement,
  flexion du plat, DÉVERSEMENT du plat long, soudure d'un seul côté —
  rien n'est recopié du modèle des cornières là où il ne s'applique pas.

## Modèle et vérifications

Rotule à la face de l'âme porteuse ; le groupe de boulons reprend VEd et le
moment d'excentricité MS = VEd·z (répartition élastique, inertie polaire).

| Élément | Vérifications | Sources |
|---|---|---|
| Boulons | cisaillement du groupe excentré ; glissement (cat. B/C) | Tab. 3.4 ; MSB P5 §3.2 ; §3.9 |
| Plat | pression diamétrale (ver/hor, interaction) ; cisaillement brut (·1/1,27) / net / bloc ; flexion (sans objet si hp ≥ 2,73·z) ; DÉVERSEMENT du plat long (z > tp/0,15) | Tab. 3.4 ; MSB P5 §3.2, §3.2.3 ; §3.10.2 ; SCI P358 + BS 5950-1 Annexe B.2.1 (Table 17), E = 205 000 |
| Âme portée | pression diamétrale ; brut / net / bloc ((n1 − 1)·d0 non grugée, (n1 − 0,5)·d0 grugée) ; grugeage (flexion + V, stabilité c et dc) | Tab. 3.4 ; MSB P5 §3.2.5.1 ; §4.2.4 / §4.2.5 |
| Âme porteuse | cisaillement local (un seul côté chargé), Av = tw·hp | SCI P358 — même modèle que le module 1 |
| Soudure | double cordon pleine hauteur sous qz = V/2hp et qx = 3·MS/hp² ; gorge min. ; pleine résistance recommandée (alerte) | EN 1993-1-8 §4.5.3.3 ; §4.5.2 |

Alertes de détail : ductilité tp ≤ 0,5·d ; hp ≥ 0,6·h (maintien) ; plat
long non maintenu ; gorge sous la pleine résistance. Limites écrites (écran
Vérifications + Méthode) : flexion HORS PLAN de l'âme porteuse sous VEd·z
et NEd (attache d'un seul côté), tying, torsion, vis-à-vis.

## Annexe nationale belge

Défauts : NBN EN 1993-1-1 ANB — γM0 = 1,00, **γM1 = 1,10** (≠ 1,00
recommandé ; ne sert qu'au déversement du plat long), γM2 = 1,25. Sources
croisées : SCIA « Theory NA EN 1993 » et Bentley STAAD « Belgian NA to
EC3 » ; le texte NBN (payant) n'a pas pu être consulté depuis cet
environnement — À CONFIRMER sur la norme, valeurs modifiables (paramètres
avancés). EN 1993-1-8 : aucune divergence belge relevée, valeurs
recommandées. Le benchmark rejoue l'exemple publié avec γM1 = 1,0 (comme sa
source) ; l'écran et la note du cas courant utilisent l'ANB.

## Validation

- **Exemple publié** (MSB P5 §3.4, fin plate, VEd = 350 kN, 2 files M20,
  z = 80) : 10 valeurs — Ip 107 000 mm² ; boulons 584 ; plat 605 / 450 /
  497 / 483 ; âme 624 / 953 / 995 / 507 kN — pire écart 0,76 %, tous
  expliqués (arrondis de la source, aire catalogue). Statut : **VALIDÉ**.
- **Calculs manuels indépendants** : déversement du plat long (λLT 38,80 ;
  pb 264,66 MPa ; Mb,Rd 57,17 kNm — Perry-Robertson recalculé hors moteur),
  soudure, flexion du plat, et les ancres PARTAGÉES avec le module 1
  (pression diamétrale de l'âme 0,739914… ; grugeage VAL-A) qui tombent sur
  les mêmes nombres.
- **Régression figée** : `tests/ref_plat_ame.json` — 12 cas, 187
  vérifications (Ed, Rd, η) + scalaires + alertes, à 1e-9 près.
- Suites : `test_plat_ame_moteur.py` (39) et `test_plat_ame_ecran.py` (40).

## Écarts assumés (v1.0)

- pas de rapport détaillé 3 pages : la note 2 pages (synthèse + plan de
  principe) et l'export texte couvrent la livraison ; le rapport détaillé
  du module 1 reste le modèle si le besoin apparaît ;
- le numéro de sous-clause MSB P5 n'est cité que là où il est établi
  (§3.2.3 flexion du plat, §3.2.5.1 âme, §4.2.4/§4.2.5 grugeage) ; ailleurs
  la référence reste au chapitre (« MSB P5 §3.2 ») — aucun numéro inventé ;
- la valeur imprimée du déversement dans le Worked Example n'a pas pu être
  extraite de la source depuis cet environnement (réseau restreint) : la
  vérification est ancrée par calcul manuel indépendant (VAL-A) et listée
  dans « sans exemple publié consulté ».
