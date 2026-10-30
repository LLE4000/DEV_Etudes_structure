# Assemblage poutre–poutre par platine d'about de hauteur partielle — v1.0

Troisième assemblage du module ACIER → ASSEMBLAGES, développé le
30/10/2026 avec la même identité graphique, la même ergonomie et la même
note de calcul que les modules doubles cornières et plat d'âme.

## Pourquoi celui-là

Les trois attaches simples les plus courantes en charpente métallique sont
les doubles cornières d'âme, le plat d'âme soudé et la platine d'about
flexible (« partial depth end plate », « header plate ») — c'est l'ordre
des chapitres du guide *Steel Buildings in Europe – Part 5* et de SCI P358.
Les deux premières étaient faites ; le registre annonçait déjà la carte
« Platine d'about — à venir ». C'est donc elle :

- très posée en Belgique et en Europe : une platine soudée en atelier, deux
  files de boulons, montage rapide ;
- même famille poutre–poutre : porteuse, portée, grugeage, décalage Δz —
  le kit de dessin, l'écran et la note se réutilisent ;
- moteur réellement différent : groupe CENTRÉ (aucun moment
  d'excentricité), deux plans de cisaillement dans la platine, platine
  mince qui donne la rotation (ductilité), tronçon en T en traction,
  tying — rien n'est recopié des deux autres là où ça ne s'applique pas ;
- validable : le même guide publie un Worked Example complet (§2.4).

## Modèle et vérifications

Rotule à la face de l'âme porteuse ; deux files symétriques (entraxe p3),
un plan de cisaillement ; chaque boulon reprend VEd/n.

| Élément | Vérifications | Sources |
|---|---|---|
| Boulons | groupe en trois branches (ΣFb,Rd ; ns·(Fb,Rd)min ; 0,8·ns·Fv,Rd — le 0,8 seulement si le cisaillement des boulons gouverne), pression diamétrale sur la platine (extrémité / intérieur) ; glissement (B, C) ; interaction cisaillement + traction (NEd ELU) | Tab. 3.4 ; MSB P5 §2.2.4 sur EN 1993-1-8 §3.7 ; §3.9 |
| Platine | cisaillement brut 2·(hp·tp/1,27)·fy/√3 ; net ; rupture de bloc k = 1 si hp ≥ 1,36·p3, 0,5 sinon ; tronçon en T (traction) ; poinçonnement | MSB P5 §2.2 ; §3.10.2 ; §6.2.4 (Tab. 6.2, méthode 2) |
| Âme portée | cisaillement sur la platine Av = hp·tw ; grugeage (cisaillement réduit, flexion + V avec le bras tp + c, stabilité c et dc) ; traction tw·hp | EN 1993-1-1 §6.2.6 ; MSB P5 §2.2.1 ; §4.2.4 / §4.2.5 |
| Âme porteuse | pression diamétrale (boulons intérieurs) ; cisaillement local brut / net sous VEd/2 | Tab. 3.4 ; SCI P358 (le guide MSB P5 §2.2 ne vérifie pas l'élément porteur) |
| Soudure | double cordon sur hp, méthode simplifiée ; gorge minimale ; pleine résistance a ≥ 0,46 / 0,48 / 0,55·tw (S235 / S275 / S355) en alerte | §4.5.3.3 ; §4.5.2 ; MSB P5 §2.2 |

Alertes de détail : hp ≥ 0,6·h (maintien en torsion, MSB P5 §2.1) ;
ductilité tp ≤ 0,36·d·√(fub/fy) (EN 1993-1-8 §6.4.2(2), règle d'ECCS
n°126 pour les platines d'about) ; tp < 8 mm déconseillé (§2.1) ; la
rondelle dégage le cordon (p3 ≥ tw + 2·a·√2 + dw) ; géométrie (platine
dans les parties droites des deux âmes, grugeage suffisant).

## Traction NEd : deux natures

- **ELU fondamental** : combinée avec VEd (fy/γM0, γM2) — interaction des
  boulons, tronçon en T, poinçonnement, âme portée, cordons sous la
  résultante.
- **Tying** (force de chaînage, situation accidentelle) : vérifiée SEULE
  avec fu/γMu, γMu = 1,1 (MSB P5 §2.3) — tronçon en T, âme portée,
  cordon.

Tronçon en T : une seule résistance porte les DEUX ailes (pas de ×2) ;
m = (p3 − tw)/2 − 0,8·a·√2, n = min(e2 ; 1,25·m), ew = dw/4 ; Σleff des
rangées en groupe ou isolées, platine à bords libres (formules du Tableau
6.4) — Σleff = hp sur les détails courants, la valeur du guide.

Non couvert, écrit en clair (écran, Méthode, alerte) : flexion
TRANSVERSALE de l'âme porteuse sous la traction des boulons — l'EN
1993-1-8 ne la traite pas pour une âme de poutre.

## Annexe nationale belge

γM0 = 1,00 (NBN EN 1993-1-1 ANB) ; EN 1993-1-8 : valeurs recommandées,
aucune divergence belge relevée (γM2 = 1,25, γM3 = 1,25, γM3,ser = 1,1).
Aucune vérification n'utilise γM1 : il n'est pas demandé. γMu = 1,1 est la
valeur du guide pour le tying ; l'EN 1993-1-8 ne la fixe pas — à confirmer
pour un projet belge. Le texte NBN (payant) n'a pas pu être consulté
depuis cet environnement : valeurs modifiables.

## Validation

- **Exemple publié** (MSB P5 §2.4, IPE A 550 S275, platine 430×200×12
  S275, 12 M20 8.8, p3 = 140, VEd = 550 kN, tying 275 kN) : 9 valeurs —
  âme 614 ; Fv,Rd 94 ; pression diamétrale 107 / 142 ; groupe 902 ;
  platine 1182 (rupture de bloc) ; tying : platine 493, âme 1513 kN ;
  pleine résistance 0,48·tw. Pire écart 0,95 % (αb arrondi à 0,61 par la
  source). Statut : **VALIDÉ**. e1 = 40 et p1 = 70 sont reconstitués de
  hp = 430 et des 6 rangées ; l'élément porteur n'est pas défini par
  l'exemple (âme de 15 mm retenue, sans effet sur les valeurs comparées).
- **Calculs manuels indépendants** : VAL-A (tronçon en T de l'exemple : m,
  Σleff, trois modes) ; VAL-B (cas par défaut : pression diamétrale,
  groupe, âme, bloc, grugeage, âme porteuse, soudure, pleine résistance,
  ductilité) — écarts nuls.
- **Régression figée** : `tests/ref_platine_about.json` — 13 cas, 183
  vérifications (Ed, Rd, η) + scalaires + alertes, à 1e-9 près.
- Suites : `test_platine_about_moteur.py` (57) et
  `test_platine_about_ecran.py` (43) ; fumée : les trois assemblages
  s'ouvrent.

## Changements communs à la famille

- **Composant de dessin** : une pièce peut glisser verticalement SEULE
  (`data-drag2` sans `data-drag`) — ici le décalage Δz, sans jeu gh.
- **Note du plat d'âme** : la page 1 lit ses dépendances (titre,
  synthèse, schémas, en-tête, hypothèses, tableaux, mention normative)
  dans des attributs de classe ; la platine d'about la sous-classe au lieu
  de la recopier. Sorties du plat d'âme inchangées (ses 40 tests).
- **Feuille de style des dessins** : curseur de glissement aussi pour
  `data-drag2`.

## Écarts assumés (v1.0)

- l'âme porteuse est vérifiée avec le modèle de SCI P358 (cisaillement
  local, pression diamétrale), comme dans les deux autres modules — le
  guide MSB P5 §2.2 ne la vérifie pas ;
- la boulonnerie des dessins (tête, écrou, rondelles) est en PROPORTIONS
  du diamètre : dessin de principe, pas des cotes ;
- numéros de sous-clauses MSB P5 cités seulement là où ils ont été lus
  (§2.1, §2.2, §2.2.1, §2.2.4, §2.3, §2.4) — aucun numéro inventé.
