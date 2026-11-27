# Écarts et corrections — doubles cornières d'âme

**Règle appliquée** : parité stricte avec le HTML de référence d'abord, erreurs
éventuelles comprises. Les tests de parité restent dans le dépôt tels quels.
Une erreur **manifeste** se corrige dans une étape séparée, avec un test qui
documente l'ancien et le nouveau résultat ; un **choix de modèle** est proposé,
jamais appliqué sans l'accord du bureau.

**État au 25/11/2026 : treize corrections appliquées aux six attaches** après une relecture indépendante de chaque
formule (§7) — celles des doubles cornières poutre–poutre le sont À CÔTÉ du corrigé, que la parité rejoue toujours tel
quel.

**État au 21/09/2026 : aucune correction appliquée au moteur.** La
transcription est identique au HTML (31 cas + benchmark + 1 900 cas aléatoires,
0 écart). Ce document consigne ce qui a été examiné, ce qui n'a pas pu être
vérifié, et ce qui est proposé.

---

## 1. Erreurs manifestes : aucune trouvée

Chaque formule a été relue en la transcrivant, avec les contrôles de cohérence
suivants (tous concluants) :

| Point contrôlé | Résultat |
|---|---|
| Inertie polaire du groupe `Ip = n·[p1²(n1² − 1) + p2²(n2² − 1)]/12` | égale à Σ(x² + y²) d'une grille n1 × n2 |
| Effort horizontal du prédimensionnement `Fx = |N|/n + 6·Mz/(p1·n·(n + 1))` | égal à M·ymax/Σy² pour une file de n boulons |
| Traction des boulons P sous MEd : `MEd·ymax / (2·n2·Σy²)` | cohérent avec deux cornières |
| Couple HEd·z repris par les deux groupes P : `HEd·z/(p3·n)` | cohérent |
| Aire de cisaillement non grugée `max(A − 2b·tf + (tw + 2r)·tf ; hw·tw)` | EN 1993-1-1 §6.2.6(3), η = 1 |
| Réduction moment–effort tranchant `ρ = 1 − (2V/Vpl − 1)²` si V > 0,5·Vpl | forme classique |
| Limites de stabilité du grugeage : h/tw ≤ 54,3 (S275) / 48 (S355) ; K = 160 000 / 110 000 | valeurs habituellement citées de SCI P358 |
| Tronçon en T : modes 1, 2, 3 (méthode 2, ew = dw/4) | EN 1993-1-8 Tableau 6.2 |
| Cordons : `fvw,d = fu/(√3·βw·γM2)`, `Fw,Rd = fvw,d·a` | §4.5.3.3 (méthode simplifiée) |
| Bibliothèques (profilés, boulons, classes, aciers) | voir §4.3 |
| Benchmark : 37 valeurs publiées du MSB Part 5 (écart ≤ 1 % ou expliqué), 22 valeurs manuelles (≤ 0,1 %) | reproduit, statut VALIDÉ |

Deux particularités du HTML, relevées et **conservées** (ce ne sont pas des
erreurs) :

- pour un profilé secondaire *Personnalisé*, les alertes citent les champs
  `hS_u, tfS_u, rS_u` mais pas `bS_u, twS_u` : seules les dimensions qui
  interviennent dans les relations contrôlées sont désignées ;
- la sentinelle interne 1e9 (résistance nulle) affiche « > 999 % », jamais
  une exception.

## 2. Références citées de mémoire — **non vérifiées**

Les textes n'étaient pas disponibles dans cette session. Les références sont
reprises **telles quelles** dans l'écran, l'export texte et la note ; aucune
n'a été « corrigée ». À contrôler par le bureau :

| Source | Références citées | Où |
|---|---|---|
| SCI P358 | Check 5 (flexion de la poutre grugée), Check 6 (stabilité locale du grugeage), Check 7 (stabilité d'ensemble, non maintenue), Check 8 (pression diamétrale de l'âme porteuse), Check 10 (cisaillement local de l'âme porteuse) | `mN`, `stN`, `val_N`, `pdP`, `vlP`, `vnP` |
| EN 1993-1-8 | Tableau 3.3, Tableau 3.4, Tableau 3.6, Tableau 4.1, §3.4.1, §3.8, §3.9.1, §3.9.2, §3.10.2(2), §3.10.2(3), §4.5.1(2), §4.5.2(2), §4.5.3.3, §5.2, §6.2.4 Tableau 6.2 | vérifications et alertes |
| EN 1993-1-1 | §6.2.3(2), §6.2.5, §6.2.6 (Tableau 3.1 : seulement le matériau des exemples du benchmark) | `tnB`, `tnS`, `flB`, `flA`, `vgS` |
| NBN EN 1993-1-1 ANB:2018 | 3.2.1(1) (option a), tableaux 3.3 ANB et 3.4 ANB | f_y et f_u de chaque pièce (`acier/assemblages/materiau.py`) |
| MSB Part 5 | §3.2.3, §3.2.5.1, §4.1, §4.2.1.1, §4.2.1.2, §4.2.2, §4.2.2.1, §4.2.2.2, §4.2.3.1, §4.2.3.2, §4.2.4, §4.2.5, §4.3.1.1, §4.3.1.4, §4.3.2.2, §4.3.2.3, §4.4 | modèles COMP |
| ECCS n°126 | modèles complémentaires (cisaillement des cornières 1,27, section nette) | `cgB`, `cnB`, `vnS` |

Élément corroborant : les exemples §4.4 et §3.4 du MSB Part 5 sont reproduits
numériquement par le benchmark (37 valeurs) — les paragraphes cités pour ces
vérifications sont donc au moins cohérents avec le document source.

## 3. Points fragiles audités en priorité

### 3.1 `fy` et `fu` de chaque pièce (tableau 3.3 ANB) et coefficients partiels

- Depuis le lot A (16/11/2026), `fy` et `fu` de chaque pièce suivent SON
  épaisseur : NBN EN 1993-1-1 ANB:2018 3.2.1(1) (option a), tableau 3.3 ANB
  (EN 10025-2) ; S355 N/NL et S460 N/NL : tableau 3.4 ANB. La table unique du
  site (`acier/materiaux.py`) les porte ; `acier/assemblages/materiau.py` dit
  quelle épaisseur décide (plat, platine, cornière : la sienne ; profilé :
  max(tf, tw) ; la gorge de pleine résistance : l'âme à tw). S355 :
  `fu = 470` de 3 à 100 mm — le `fu = 510` de l'EN 1993-1-1:2005 Tableau 3.1
  ne sert plus qu'au matériau des exemples du benchmark (`materiau.EXEMPLES`).
  La ligne « S355 (fu 490) » suit le S355 belge ; « S355 N/NL » existe dans la
  bibliothèque.
- γM0 = 1,00 ; γM2 = 1,25 (boulons, pression diamétrale, soudures) ;
  γM2 = 1,25 (sections nettes, blocs) ; γM3 = 1,25 ; γM3,ser = 1,10 : valeurs
  recommandées, modifiables à l'écran (groupe « Coefficients partiels »).
- **Où caler sur l'ANB** : les défauts dans
  `acier/assemblages/poutre_poutre/doubles_cornieres/entrees.py` (`defaults()`),
  `fy` et `fu` dans la table unique `acier/materiaux.py`, `βw` dans
  `acier/donnees/bibliotheques.json` (une ligne par nuance).
  Le test `tests/test_assemblages_bibliotheques.py` verrouille les valeurs
  actuelles : le changer est un acte volontaire, tracé.

### 3.2 Bibliothèques saisies de mémoire — contrôlées

- 31 profilés communs avec le catalogue du dépôt (`profiles_test.json`) :
  **0 divergence** sur h, b, tw, tf, r ; aire recalculée
  `A = 2·b·tf + (h − 2tf)·tw + (4 − π)·r²` à moins de 0,6 % de l'aire
  catalogue (arrondi du catalogue à 0,1 cm²).
- Les 59 autres profilés (HEA/HEB/HEM 320 → 1000, IPE hors catalogue) n'ont
  pas de valeur de comparaison dans le dépôt : **non contrôlés**.
- Boulons : As ISO 898 exactes, A = π·d²/4 à 0,1 mm², d0 selon EN 1090-2
  (d + 1 / + 2 / + 3). `dm` et `dw` (ISO 4014 / ISO 7089) : vraisemblables,
  non contrôlés contre un catalogue.
- Classes 4.6 / 5.6 / 8.8 / 10.9 : fyb, fub, αv conformes aux Tableaux 3.1
  et 3.4.
- Cornières : dimensions usuelles ; rayons de congé vraisemblables (8 → 18 mm),
  **non contrôlés** contre l'EN 10056-1.

### 3.3 Vérifications INT / COMP sans exemple publié

| Clé | Ce que fait l'outil | Analyse | Statut |
|---|---|---|---|
| `flA` (INT) | Flexion de l'aile A dans son plan, `MEd = (VEd/2)·(gA − tc/2)`, `MRd = tc·Lc²/6·fy/γM0` | Bras de levier = de la file P au milieu de l'épaisseur de l'aile B (talon). Vérification supplémentaire, côté sécurité, non demandée par les guides (le talon est maintenu). Taux faible (14,6 % sur le cas par défaut). | conservée, affichée INT |
| `m2` (COMP) | Interaction à la 2ᵉ file (n2 = 2, ln > e2,b + p2) : `MEd = VEd·(gh + e2,b + p2)` contre `Mc,2` réduit par `min(Vpl,N ; Vbloc)` | La réduction par le minimum des deux résistances au cisaillement est un choix prudent. | conservée |
| `stD` (COMP) | Profondeur : `dn ≤ h/2` (une semelle grugée), `dn ≤ h/5` (deux) | Limites de détail rappelées de SCI P358 / MSB §4.2.5 ; à confirmer sur le texte. | conservée, référence non vérifiée |
| `vlP`, `vnP` (COMP) | Cisaillement local de l'âme porteuse : `Av = tw·(et + (n1 − 1)·p1 + eb)`, `et = min(zt ; 5d)`, `eb = min(zb ; p3/2 ; 5d)`, sous VEd/2 | Un seul côté chargé (déclaré). Deux poutres en vis-à-vis partageant les boulons : hors domaine, signalé dans « Méthode ». | conservée |
| HEd | Cisaillement horizontal des boulons P (`H/2/n`), traction par le couple `H·z/p3`, cordons A (`qx`) ; **pas** d'effet sur le groupe S (HEd est perpendiculaire à l'âme secondaire) | cohérent avec le cheminement décrit ; alerte informative au-delà de 10 % de VEd | conservée |
| MEd | Valeur absolue, ajoutée à VEd·z (groupe S), traction élastique des boulons P, moitié par cornière (`flB`), ajoutée au moment au grugeage | cohérent avec « moment parasite » ; alerte informative | conservée |

### 3.4 Soudures : longueur efficace = longueur totale

L'outil prend `Lw = Lc + 2·lh` comme longueur efficace du groupe de cordons
(cordons pleins sur toute leur longueur). C'est **affiché comme hypothèse** :
onglet Méthode, hypothèses de l'export texte et de la note (quand une fixation
est soudée).

**Proposition P1 (non appliquée)** : EN 1993-1-8 §4.5.1(2) admet la longueur
totale seulement si le cordon est de section pleine sur toute sa longueur,
sinon la longueur efficace est réduite de 2a à chaque extrémité. Prendre
`Leff = L − 2a` par cordon réduirait légèrement `Lw` et `Iw`. Effet chiffré sur
le cas VAL-B (a = 5 mm, Lc = 190, retours 40) : `Lw` 270 → 240 mm (trois
cordons), soit environ +12 % sur `qz` et `qn` ; η de `wS` passerait d'environ
49 % à 55 %. **À trancher par le bureau.**

## 4. Corrections proposées (non appliquées)

| N° | Objet | Nature | Décision attendue |
|---|---|---|---|
| P1 | Longueur efficace des cordons `L − 2a` | choix normatif (§4.5.1(2)) | bureau |
| P2 | Aucune autre | — | — |

## 5. Comment appliquer une correction, le jour venu

1. Écrire le test qui documente l'ancien et le nouveau résultat
   (`tests/test_assemblages_corrections.py`, à créer) ;
2. modifier `moteur.py` **et** le HTML de référence (pour garder l'oracle),
   ou marquer dans `tests/test_assemblages_parite.py` la grandeur qui diverge
   désormais, avec le numéro de la correction ;
3. régénérer `acier/reference/reference_double_corniere.json` par l'oracle
   Node si le HTML a été corrigé ;
4. consigner ici : où, quoi, pourquoi, source, effet chiffré sur le cas par
   défaut.

---

## 6. Attaches POUTRE – POTEAU (02/11/2026) — ce qui attend le bureau

Ce document était celui des doubles cornières poutre–poutre. La famille
**Poutre – Poteau** (`plat_ame_poteau`, `doubles_cornieres_poteau`,
`platine_about_poteau`, livrées le 02/11/2026) y ajoute **neuf points tranchés
sans source franche** et des **écarts assumés** par module.

**Où vit quoi — une seule source par chose.** Le RAISONNEMENT de chaque point
est au §9 de `docs/assemblages/POUTRE_POTEAU.md` ; le texte que l'ÉCRAN affiche
est dans `acier/assemblages/poutre_poteau/textes.py` (les huit limites
communes) et dans le `methode.py` de chaque module (la septième, propre à
chacun, et les points de §9) ; les écarts par module sont dans les trois fiches
`PLAT_AME_POTEAU.md`, `DOUBLES_CORNIERES_POTEAU.md`,
`PLATINE_ABOUT_POTEAU.md`, rassemblés au §13.3 de l'étude. Ce qui suit est le
**registre** : ce qu'il faut décider, et ce qui bouge si la décision change.

### 6.1 Les neuf points — registre de décision

| N° | Décision attendue | Nature | Si le bureau tranche autrement |
|---|---|---|---|
| **P-1** | `Σleff` de l'aile de poteau non raidie : Tableau **6.4** (appliqué) ou 6.6 (cité par la consigne) ? **C'est le NUMÉRO, pas la formule** — les formules sont celles de l'aile non raidie, et `platine_about` les applique déjà. | référence | aucun chiffre ne bouge : seule la RÉFÉRENCE imprimée change (`notation.py`, `methode.py`, `benchmark.py` de `pac_` et `dcc_`) |
| **P-2** | Faut-il une vérification locale de l'aile sous `VEd` seul, pour un plat soudé au droit de l'âme ? L'outil n'en pose aucune (groupe « Poteau porteur » VIDE en cas a). | modèle | il faudrait inventer un `Av = f(tfP, hp)` — refusé faute de source. Un modèle imposé s'ajouterait au moteur de `fpc_` |
| **P-3** | La recoupe en LARGEUR des semelles est-elle un grugeage ? L'outil dit **non** : MSB P5 §4.2.4 / §4.2.5 ne sont pas activés. | modèle | les trois modules devraient activer les règles du Té résiduel en cas (b) recoupé |
| **P-4** | Faut-il une épaisseur minimale d'aile de poteau (`tfP ≥ tp` pour un cordon de pleine résistance) ? L'outil n'en pose aucune. | détail | une alerte de plus dans les trois modules |
| **P-5** | Âme du poteau en traction transversale avec **`ω = 1`** (valeur la plus favorable, en l'absence de moment d'étage). | modèle | `ω` deviendrait une saisie ; `pac_` et `dcc_` perdent de la résistance en tying |
| **P-6** | Flexion hors plan de l'âme du poteau en cas (b) : **DÉCLARÉE non vérifiée**, pas transposée. | périmètre | un modèle imposé s'ajouterait aux trois modules en cas (b) |
| **P-7** | Gauge minimal `p3 ≥ twP + 2·rP + dw` (dégagement du congé, nature **INT**) — il explique les gauges usuels 90 / 100 / 120 / 140 mais ne les remplace pas ; SCI P358 tabule des gauges minimaux par série, **non consultés**. | détail | une table remplacerait la formule dans `mecanique.geometrie_poteau` |
| **P-8** | `et = 5·d` pour le cisaillement local de l'âme du poteau, bornes `zt` / `zb` du grand frère supprimées (le poteau continue). | modèle | la règle dépend entièrement du `5·d`, cité de mémoire de SCI P358 depuis le 21/09/2026 |
| **P-9** | `γMu = 1,1` (valeur du guide ; l'ANB belge n'a pas été consultée). | coefficient | modifiable en saisie ; toutes les résistances de tying bougent |

### 6.2 Écarts assumés — ce qui a été écrit autrement que l'étude

Un seul est **non conservatif s'il n'est pas appliqué**, et il l'a été :

- **`dcc_` · `bvT_S` AJOUTÉE** — le §8.2 de l'étude ne demandait pas le
  cisaillement des boulons du groupe S sous la force de chaînage, alors que son
  §7.2 attribue le tying à ce groupe. L'omission était **non conservative** ;
  la ligne existe, elle est nommée comme ajoutée dans `methode.py` et un
  contrôle la fige.

Les autres ne changent aucune résistance, ou les changent dans le sens de la
formule publiée plutôt que du nombre imprimé :

- **la recoupe minimale** : l'étude imprime 154,5 / 144,5 mm (valeurs à
  `j = 10`) alors qu'elle déclare `j = 5` ; les trois modules appliquent la
  FORMULE `(bP − twP)/2 + j` → 149,5 / 139,5 mm. Pour `pac_`, l'about est à
  `tp` de la face d'appui, pas à `gh` : `(bP − twP)/2 + j − tp` ;
- **`dcc_`** : poinçonnement coupé en deux lignes nommées (cornière, poteau) ;
  gauge `p3` aussi plafonné par le Tableau 3.3 ; tronçon en T de l'aile écrit
  pour UNE file par cornière (alerte d'information `n2P_T` sinon) ;
- **`pac_`** : la limite du cas (b) ne cite pas `qx = 3·MS/hp²` (groupe
  CENTRÉ) ; `bp ≤ bP` en cas (a) est une INFORMATION, `bp` entre congés en cas
  (b) est bloquante ;
- **`fpc_`** : BM3' de l'étude n'est pas complet — l'outil trouve en plus
  `p2 = 60 < 2,4·d0 = 62,4` et une pression diamétrale d'âme portée à
  `η = 1,148` ;
- **les trois** : `rec_S`, `j_lat`, `d_nt`, `d_nb`, `l_n` sont des cotes du
  DESSIN et non des paramètres avancés (règle P6 de la charte).

---

## 7. Corrections du 25/11/2026 — relecture indépendante des six attaches

Six relectures indépendantes (une par attache : chaque vérification recalculée à la main sur le cas par défaut, sept
variantes et des tirages aléatoires ; bancs rejoués, tous VALIDÉS) ont confirmé les formules des cas courants et trouvé
les défauts ci-dessous. **Règle suivie** : une erreur manifeste, ou une omission du côté non sûr, est corrigée, mesurée
et figée par un test qui rougit si on la retire ; un choix de modèle qui n'est pas du côté non sûr reste proposé (§7.3).

**Comment les doubles cornières poutre–poutre restent comparées au corrigé** : le moteur applique les corrections
partout, SAUF quand il rejoue le corrigé — `parite.materiau_du_corrige` porte « formules du corrigé », que
`moteur.compute` lit (argument `materiau`, ou table du site remplacée par les suites de parité). La parité (31 cas,
oracle, 1 900 tirages, textes) reste verte sans retouche ; l'ancien et le nouveau résultat de chaque correction sont
dans `tests/test_assemblages_corrections.py`.

### 7.1 Corrections appliquées

| N° | Où | Avant | Après | Effet mesuré | Test |
|---|---|---|---|---|---|
| **K1** | platine d'about (poutre et poteau) : rupture de bloc, `moteur.py` et prédimensionnement | `Anv = tp·(hp − e1 − (n1 − ½)·d0)` avec la pince HAUTE | `Anv = tp·(e1,min + (n1 − 1)·p1 − (n1 − ½)·d0)` : sous `VEd`, les boulons poussent la platine vers le haut, le plan cisaillé va du boulon bas au bord haut ; `VEd` étant pris en valeur absolue, la plus petite des deux pinces | sans effet si pinces égales ; pinces 35 / 95 : 655 → 409 kN ; le prédimensionnement allonge la pince basse dès que `hp ≥ 0,6·h` commande (cas figé : 692 → 364 kN) | `test_platine_about_moteur`, `test_platine_about_poteau_moteur` |
| **K2** | ailes A des cornières (poutre : C1, et poteau) : rupture de bloc | idem, `hc − e1` | idem, par `e1,min` (l'aile B, poussée vers le bas, garde `hc − e1`) | pinces 35 / 95 : 639 → 393 kN | `test_assemblages_corrections`, `test_doubles_cornieres_poteau_moteur` |
| **K3** | platine à UNE rangée : `Σleff` du tronçon en T (`platine_about`, `mecanique.leff_platine_bords_libres`) | `π·m + e1h + e1b` et `2m + 0,625e + (e1h + e1b)/2` — pas des mécanismes | `min(π·m ; 2·e1h) + min(π·m ; 2·e1b)` et `min(2m + 0,625e ; e1h) + min(… ; e1b)` : `Σleff ≤ hp` | hp = 100 : 139 → 100 mm (poutre), 195 → 100 mm (poteau) | `test_assemblages_mecanique`, les deux suites des platines |
| **K4** | groupe de boulons des platines (`groupe_boulons`, deux copies) | le 0,8 seulement en 3e branche (lettre du MSB P5 §2.2.4) | plafond `FRd ≤ 0,8·ns·Fv,Rd` dans les trois branches | cas par défaut : 506,8 → 451,6 kN (η 0,247 → 0,277) ; la résistance ne descend plus quand `tp` augmente (563 kN à 11,1 mm, 452 kN à 11,2 mm) | `test_assemblages_mecanique`, `test_platine_about_moteur` |
| **K5** | plats d'âme (poutre et poteau) : gorge de PLEINE RÉSISTANCE (avis `a_rec`, prédimensionnement) | `tp·fy·βw·γM2/(2·fu·γM0)` (plat cisaillé : 0,39·tp en S355) | `…/(√2·fu·γM0)` (plat tendu : la règle des guides, 0,46 / 0,48 / 0,55·tp ; 0,50 / 0,60 avec les `fu` de l'ANB), tolérance de lecture 0,05 mm ; gorge par défaut 5 → 6 mm | tp = 10 : 4,2 → 6,0 mm ; prédimensionnement tp = 9 : a = 4 → 5 ou 6 mm. Aucun taux ne change à gorge égale | `test_plat_ame_moteur`, `test_plat_ame_poteau_moteur` |
| **K6** | plats d'âme : flexion du plat `pfl` | sans objet dès `hp ≥ 2,73·z`, même avec `MEd` | active aussi si `MEd ≠ 0` | HEB 500, plat 315 × 15, VEd 60, MEd 88 : η 1,06 au lieu de « vérifié » | idem |
| **K7** | plats d'âme : 2e file de boulons `m2` | `VEd·(gh + e2,b + p2)` | `+ |MEd|`, comme `mN` | MEd = 4 : 13,75 → 17,75 kNm | idem |
| **K8** | cornières, ailes A SOUDÉES (poutre : C2, et poteau) | `MP = 0` avec l'option « Non » : cordon « centré » | `MP = (VEd/2)·(bA − xg)` toujours (l'option ne vaut que pour des boulons P) | cordon 345 → 792 N/mm (η 0,29 → 0,66) | `test_assemblages_corrections`, `test_doubles_cornieres_poteau_moteur` |
| **K9** | cornières, cordons P (C3, et poteau) | `qn = NEd/(2·ΣL)` | `+ (MEd/2)·(hc/2)/Ixx + HEd·z/((tw + 2·(bA − xg))·ΣL)` | N 60, H 20, M 5 : 369 → 685 N/mm | idem |
| **K10** | cornières : tronçon en T des ailes A, et aile et âme du poteau (C4, et poteau) | effort `NEd` seul : lignes inactives sous `MEd` ou `HEd` | effort `2·nP·Ft,Ed`, la traction réelle des boulons P | MEd = 5 : ligne active (η 0,20 à 0,23) | idem |
| **K11** | cornières, ailes A SOUDÉES en traction (C5, et poteau) | aucune vérification de l'aile | `NT ≤ 2·Mpl/m`, `Mpl = 0,25·hc·tc²·f/γ`, `m = bA − tc − 0,8·rc` (cordon de bout d'aile = rotule ; `fu/γMu` en chaînage) — sous la clé `tsA` | L100×100×10, hc 190 : 42 kN ; NEd = 300 kN passait « VÉRIFIÉ », η réel 4,7 | idem |
| **K12** | chaînage, côté poteau : `bvT` (plat d'âme) et `bvT_S` (cornières) | cisaillement des boulons seul | `min(cisaillement ; Fb,hor du plat ou des ailes B ; Fb,hor de l'âme portée)/γMu` | IPE 400, e2,b = 30, NEd 400 : η 0,94 → 1,10 | `test_plat_ame_poteau_moteur`, `test_doubles_cornieres_poteau_moteur` |
| **K13** | poteau : pince au bout d'aile à deux files par cornière ; `n` du tronçon en T de l'aile ; `e_col` en cas (b) | `(bP − p3)/2` ; `min(e ; 1,25·m)` ; `z` | `− (n2 − 1)·p2` ; `min(e ; e2 de la pièce serrée ; 1,25·m)` (Figure 6.8) ; `twP/2 + z` (de l'axe du poteau) | HEB 220 + L150 à deux files : pince négative au lieu de 55,7 mm ; `FT,2` 955 → 902 kN (et `FT,1` +2 %) ; 50 → 55,5 mm | les trois suites du poteau |

### 7.2 Ce que le prédimensionnement fait (vérifié sur les six attaches)

`apply_solution` n'écrit que le mode, le boulon, les rangées, les pinces et entraxes, la pièce d'attache et sa gorge —
jamais un effort, un profilé, un acier, un niveau (`Δz`, `zc`), le grugeage, le jeu, l'appui, la classe ou la catégorie.
`tests/test_assemblages_predim.py` le fige (238 solutions recopiées, par le moteur et par l'écran du site).
Le prédimensionnement n'estime PAS le glissement (catégories B et C), le chaînage, un `d0` imposé ni les cordons des
cornières : il peut afficher « Proposition complète » pour une solution que la vérification refuse ensuite — le verdict
affiché reste celui de la vérification complète.

### 7.3 Examiné, non changé — à décider par le bureau

| Objet | Constat | Sens |
|---|---|---|
| `bvt` des platines | `Ft,Ed = NEd/n`, sans effet de levier (0,73 → 0,86 sur un cas) | déclaré, non sûr |
| `Σleff` de la platine en traction | Tableau 6.4 ; dépasse la valeur du guide quand la pince basse est longue (chaînage : 0,91 contre 1,25) | déclaré, moins sûr que le guide |
| `ks` en trou surdimensionné | reste à la saisie (Tableau 3.6 : 0,85) — catégories B et C seulement | à saisir à la main |
| section nette en catégorie C | `Anet·fy/γM0` absent (+7 à +10 %) | non sûr, rare |
| `mN` (cornières) | `NEd` n'entre pas dans la section grugée | non sûr en compression |
| cisaillement local de l'âme du poteau (platine) | `twP·hp`, sans le plafond `5d` des cornières | +17 à +40 % si platine dissymétrique |
| compression `NEd < 0` | transmise « par contact », non vérifiée (cordons, âme, flambement du plat) | déclaré |
| manques | poinçonnement de l'âme porteuse par un plat épais ; interaction V–N de l'âme portée ; plat long non grugé (bloc en flexion–cisaillement) ; âme du poteau hors plan en chaînage (cas b) | périmètre |
| conservatifs | déversement du plat long (sans le 1/0,6 du guide) ; `Σleff` de l'aile du poteau = `n1·p1` ; cordons en méthode simplifiée ; pression diamétrale en enveloppe | sûr |
| de mémoire | MSB Part 5, SCI P358 et ECCS n°126 n'ont pas pu être relus en ligne ; `γM1` et `γMu` belges | à confirmer |


---

## 8. Lot de clôture du 27/11/2026 — contrôle indépendant des sept attaches

Un second contrôleur a relu le dossier de contre-expertise et proposé onze arbitrages (A1 à A11) et neuf constats
nouveaux (N1 à N9). Chaque proposition a été revérifiée ici, sur le texte des guides et par le calcul, avant d'être
portée au moteur. Les guides ont été lus à la source (steelconstruction.info : SCI P358, SCI P398, MSB Part 5) ; aucun
n'est dans le dépôt — le code ne garde que les valeurs utiles, avec leur page.

**Principe retenu.** Pour les platines d'about et les plats d'âme, la procédure du **SCI P358** est appliquée comme un
tout dans son domaine (A1, A7) ; le MSB P5 reste la référence des doubles cornières et sert d'information là où il est
plus sévère. Les règles communes vivent dans `acier/assemblages/sci.py`, une fonction par règle, avec sa page et la
valeur publiée qui l'ancre. **Les doubles cornières poutre–poutre gardent leur parité** avec le corrigé de référence :
leurs corrections restent derrière le garde « corrigé » et `tests/test_assemblages_parite.py` est inchangé.

Tests : `tests/test_assemblages_cloture.py` (l'ancien et le nouveau de chaque règle, 62 contrôles),
`tests/test_pied_poteau_moteur.py` §8 (24 contrôles), les suites de chaque moteur et leurs bancs. Une passe de
mutations retire chaque correction une à une : 41 mutations, 41 vues (au premier passage, le rang des propositions n'était plus vu — contrôle renforcé — et une mutation mal écrite a été refaite).

### 8.1 Six attaches de poutre — corrections appliquées

| N° | Où | Avant | Après | Effet mesuré |
|---|---|---|---|---|
| **L1** (A1) | platines d'about : groupe de boulons | trois branches, dont `ΣFb,Rd` (somme des pressions diamétrales), plafonnées à `0,8·n·Fv,Rd` | SCI P358 Check 8 : `n·Fb,min` si `Fb,min ≤ 0,8·Fv,Rd`, sinon `0,8·n·Fv,Rd` ; `Fb,min` sur la platine ET sur le porteur | −3,9 à −13,9 % sur quatre cas figés ; exemple 1 du SCI : 602 kN retrouvés |
| **L2** (A1) | platines : âme portée au droit de la platine | `Av = hp·tw` | `Av = 0,9·hp·tw` (Check 4) ; double grugeage `0,9·tw·hT`, platines ET plats (Checks 4 et 5) | +11 % de taux : exemple du MSB 89,5 → 99,5 % |
| **L3** (A1) | platines courtes (`hp < 1,36·p3`) | rien | flexion dans le plan, `VRd = 2·tp·hp²·fy/(3·(p3 − tw)·γM0)` (Check 9), ligne `pip` | platine 80 × 220 × 10, 120 kN : « vérifié » 97,6 % → 105,0 % |
| **L4** (A1) | platines : cisaillement local du porteur | `Av = t2·hp` | `Av = t2·(et + (n1 − 1)·p1 + eb)`, `et = min(e1,t ; 5d)`, `eb = min(e1,b ; p3/2 ; 5d)` (Check 10) | +8 à +60 % de résistance selon le cas ; exemple 1 du SCI : 742 et 795 kN retrouvés |
| **L5** (A2) | platines en CHAÎNAGE : `Σleff` du tronçon en T | Tableau 6.4 seul | de plus plafonnée par `e1A`, `p1A` (Check 11 ; MSB §2.3.1) ; la traction ELU garde le Tableau 6.4 | exemple 2 du SCI : 290 mm ; 381 kN avec sa tête de 33 mm, 393 kN avec la rondelle de 37 mm du site |
| **L6** (A4) | cordons des platines et des plats | longueur `hp` | longueur efficace `hp − 2·a` par cordon ; refus sous `max(30 mm ; 6·a)` | +2,7 à +7,4 % (platines), +5 à +24 % (plats) |
| **L7** (A3) | détail éprouvé de l'articulation | avis | statut « À VÉRIFIER (articulation à justifier : …) » : platine au-delà de `0,36·d·√(fub/fy)`, gorge sous 0,8 × pleine résistance (platine) ou sous la pleine résistance (plat), pinces du plat `< 2·d`, boulons hors 8.8 non précontraints | PA-4, PA-9, PAP-7, PL-2, PL-10, PLP-7, PLP-8 passent de « vérifié » à « à vérifier » |
| **L8** (A3) | plat d'âme, cas par défaut | M20, `e1 = 35`, `p1 = 60` (pince `< 2·d`) | `e1 = 40`, `p1 = 55` : le cas par défaut tient le détail ; le prédimensionnement propose des pinces `≥ 2·d` | cas par défaut 80,3 → 92,9 % |
| **L9** (A7) | plat d'âme LONG | seuil sur `z` (centre du groupe), courbe BS 5950-1, sans le 0,6 | seuil du SCI sur `zp` (première file) ; `VRd = min(Wel·χLT·fy/(0,6·z·γM1) ; Wel·fy/(z·γM0))`, `χLT` de l'EN 1993-1-1 (Check 3) ; section ABCD (Check 4), ligne `abS` | +16 à +36 % ou −16 % de résistance selon le cas ; Tableau 5.2 et exemple 1 (743 kN) retrouvés |
| **L10** (A7, N1) | plat long pour le MSB, court pour le SCI | rien | section ABCD du MSB calculée ; au-delà de 100 % : statut « À VÉRIFIER (section ABCD … selon le MSB P5) » | cas N1 : « vérifié » 99,6 % → à vérifier, 27,50 / 22,69 kNm = 121,2 % |
| **L11** (A6) | plat d'âme : âme porteuse | rien | poinçonnement, forme rigoureuse `VEd·z + |NEd|·hp/6 ≤ t2·fu,2·hp²/(6·γM2)` (Check 10), ligne `poi` | PL-3 : ligne à plus de 400 % ; PL-10 : 32,1 % |
| **L12** (A5) | âme du poteau en CHAÎNAGE (appui dans l'âme) | avis « non couverte » | SCI P358 Check 14, ligne `ahP` dans le verdict (deux formules : platine, plat) | plat par défaut, 400 kN : « vérifié » 93,5 % → 120,4 % (332,13 kN) ; platine, 370 kN : 97,6 % → 102,1 % (362,47 kN) ; exemples 2 du SCI retrouvés : 405,4 kN pour 405 (platine, p. 49), 411,7 kN pour 412 (plat, p. 159) |
| **L13** (R07) | grugeage COURT à deux files (`xN < 2·d`) | bras `gh + c`, résistance au bout du grugeage | bras `max(gh + c ; gh + e2,b + p2)` et résistance au droit de la 2ᵉ file — plats : SCI Check 5 ; cornières : MSB §4.2.4.2 (C6) | cornières à deux files : résistance réduite (cisaillement fort au droit de la file) |
| **L14** (A9) | effort tranchant inversé, traction sur âme porteuse, poutre grugée non maintenue, cornières soudées des deux côtés, `HEd` | calculés comme le cas courant, sans réserve | statut « À VÉRIFIER (raison) » ; avis `V_inv` | DC-5, DC-10, PA-2 passent à « à vérifier » |
| **L15** (A10) | βLf des platines longues | « Non » par défaut, sans avis | « Non » par défaut (les exemples publiés à six rangées n'appliquent pas βLf) ; avis `blf_off` dès que `Lj > 15·d` | option « Oui » : 903,2 → 891,9 kN sur l'exemple du MSB |
| **L16** (D1) | note PDF : bloc HYPOTHÈSES | trois lignes, sinon la note est refusée | une ligne de plus quand il le faut (`note_kit.empaqueter_au_mieux`) | 234 combinaisons d'options (attaches sur poteau et plat poutre–poutre) rejouées sur la révision en ligne : 44 notes refusées (28 plats, 10 platines, 6 cornières, sur poteau) → 0 |
| **L17** (D4) | aide de la gorge du plat (six textes) | `…/(2·fu·γM0)` | `…/(√2·fu·γM0)`, la formule du moteur depuis K5 | texte seulement |
| **L18** (N9) | sélecteurs et propositions | une ligne « à vérifier » comptait comme vérifiée (étoile « le plus petit qui tient », « conseillée ») | elle reste orange, dit sa raison, et passe après les lignes vérifiées ; une proposition n'est retenue que vérifiée et sans raison de détail (`propositions._valide`) — une raison de DONNÉE (`sci.donnee` : traction, V < 0, maintien…) ne l'écarte pas | — |
| **L19** (A1, A4, A5) | note PDF : hypothèses | « cordons sur hp » ; « flexion hors plan NON vérifiée » pour tout appui dans l'âme | « cordons sur hp − 2a » ; au chaînage, « flexion hors plan (SCI P358 Check 14) », sous une traction ELU « NON vérifiée », sans traction rien ; la poutre grugée « maintenue au déversement (donnée de la saisie) », en DERNIER segment (`note_kit.segment_maintien` : c'est lui que le kit raccourcit d'abord, la note garde les trois lignes de la charte) ; la seconde poutre éventuelle n'y tient pas : elle reste dans la méthode | texte seulement — un premier essai avec deux segments faisait passer presque toutes les notes à quatre lignes : refusé par les suites de la charte, corrigé dans le code |
| **L20** (A7) | plat d'âme : notation | « zp » désignait à la fois la position verticale du plat (saisie, dessin, alertes) et, dans la méthode, la note et les paramètres, la distance du SCI à la PREMIÈRE file | la position verticale s'écrit « zc » (comme aux cornières) ; « zp » est celui du SCI P358 ; le dessin cote zp à côté de z quand il y a deux files | texte et dessin seulement |
| **L21** (A10) | βLf : la condition de l'exception | « comme les exemples des guides » | écrite dans l'avis, l'aide et la méthode : l'âme est soudée sur toute la hauteur de la platine, transfert uniforme — l'exception du §3.8(2) | texte seulement |
| **L22** (A1) | note PDF : verdict | le détail éprouvé TENU n'était pas écrit (seul son défaut l'était, par le statut) | « Articulation : détail éprouvé tenu (tp = 10 ≤ 15,9 mm ; a = 9 ≥ 6,7 mm) » à la conclusion des platines et des plats, valeurs nommées par le moteur (`note_kit.phrase_articulation`) | 192 notes rejouées : 43 la portent, aucune ne change de disposition ni ne déborde ; sur cinq notes du plat sur poteau aux options cumulées, elle prend la place du dernier bloc de formules facultatif |

### 8.2 Pied de poteau — corrections appliquées

| N° | Où | Avant | Après | Effet mesuré |
|---|---|---|---|---|
| **P1** (R05) | `fjd` : aire de diffusion | chaque direction diffusée à part | aire SEMBLABLE à la platine, un seul rapport `k = min(3 ; b2/bp ; l2/lp)` (EN 1992-1-1 §6.7) | massif 600 × 1 500 sous platine 500 × 500 : 18,41 → 13,60 MPa ; 17 fiches changent de taux |
| **P2** (R05) | mortier | deux conditions de `βj = 2/3` | de plus : au-delà de 50 mm, `fg ≥ fck` (EN 1993-1-8 §6.2.5(7)) | 60 mm / 20 MPa : refusé |
| **P3** (R12) | verdict | une alerte bloquante hors géométrie laissait « VÉRIFIÉ » (mortier ; massif en calcul manuel) | toute alerte bloquante refuse, dans les deux modes | mortier de 3 MPa : 79,2 % « vérifiée » → refus |
| **P4** (R13) | tiges près du profilé | bras `m` forcé à 1 mm : tronçon en T énorme | refus si le trou mord sur l'acier ou sur son cordon (`a·√2 + d0/2`) ; avis si la rondelle chevauche le cordon — deux modes | PP-8, PP-10, PP-20, PP-22, PP-29 refusées |
| **P5** (N4) | tige à crosse | admise avec tout acier | refus au-delà de `fyb = 300 MPa` (EN 1993-1-8 §6.2.6.12(5)) | PP-5, PP-19 |
| **P6** (N2) | `k1` du cône | libellé « tiges coulées en place » sur 7,7 / 11,0 | valeurs gardées, libellé corrigé (ce sont celles des fixations post-installées) | texte seulement |
| **P7** (N3, N5, A8) | statut des ancrages | orange seulement pour un béton d'ancrage hors périmètre au-delà de 100 % | de plus : torsion (bord et pry-out ne la comptent pas), deux périmètres ; fendage et éclatement latéral au périmètre complet | T seul 20 kNm : « vérifié » → à vérifier |
| **P8** (N9) | raison « à vérifier » | écrite aussi sur un pied refusé | seulement sur un pied vérifié, dans les deux modes (le calcul manuel la gardait sur une platine refusée, sans l'afficher) | — |
| **P9** (A11) | cordons poteau / platine | reprennent toujours la compression | option « Appui direct » (« Non » par défaut) : la compression passe par le contact | cas par défaut : 40,5 → 25,7 % ; les exemples publiés (appui direct) ne sont plus refusés par leurs cordons |
| **P10** | tige entre les ailes : bras `m` | `|x| − tw/2 − 0,8·r` (cas laminé) | `|x| − tw/2 − 0,8·a·√2` (platine soudée — EN 1993-1-8 Figure 6.2) | HEA 300, x = 100 : 74,2 → 86,7 mm |
| **P11** | rangée devant l'aile : `Σleff` | motifs par tige seulement | de plus la charnière droite `0,5·bp` (EN 1993-1-8 Tableau 6.6) | platine 500, tiges à 250 : 268,8 → 250 mm ; SCI P398 E.1 : 300 mm et 47,8 kNm retrouvés |
| **P12** | banc | aucun exemple publié | cinq rejoués : MSB §6.5, SCI P358 (pieds, ex. 1 à 3), SCI P398 E.1 (`docs/validation/exemples_publies.json`) | écarts ≤ 1 % ou expliqués |
| **P13** | γMs,V de l'EN 1992-4 (tiges au cisaillement) | `max(1,25 ; fuk/fyk)` seul | 1,5 dès que fuk > 800 MPa ou fyk/fuk > 0,8 (Tableau 4.1 — seconde branche citée de mémoire, concordante avec le recalcul indépendant ; appliquée parce qu'elle ne peut que rendre le calcul plus prudent) | tiges 10.9 : VRd,s − 17 % ; le reste inchangé |

P10 et P11 ne figuraient pas dans les arbitrages : ils ont été trouvés en rejouant l'exemple E.1 du SCI P398, et
corrigés parce qu'ils vont du côté non sûr.

### 8.3 Les 92 exemples du dossier, avant et après

Rejoués avec les mêmes saisies sur la révision en production et sur le lot : 60 restent non vérifiés, 16 restent
vérifiés, 10 passent de « vérifié » à « à vérifier » (PA-2, PA-4, PA-9, PL-2, PL-10, DC-5, DC-10, PAP-7, PLP-7,
PLP-8), 4 restent « à vérifier », 2 passent de « à vérifier » à « non vérifié » (PP-10 et PP-29 : trou d'une tige dans
le cordon). **Aucun exemple ne devient plus favorable.**

### 8.4 Non appliqué — à décider par le bureau

| Objet | Constat | Pourquoi ce n'est pas dans le lot |
|---|---|---|
| cordons des cornières : longueur efficace | `L − 2·a` changerait le banc de parité (ΣL 270 → 260 mm) | règle de la maison : les tests de parité restent tels quels ; choix de modèle |
| βLf (A10) | le second contrôleur propose de l'appliquer dès que `Lj > 15·d` | les deux exemples publiés à six rangées ne l'appliquent pas (transfert uniforme) ; « Non » + avis, option en saisie |
| épaisseur du plat d'âme, règle du SCI §5.3 (3) | proposée en statut | décision du bureau du 16/11/2026 : un plat de 10 mm en S355 avec des M20 reste admis — elle reste un avis |
| `k1` d'une tige à tête (8,9 / 12,7) | cas par défaut : interaction du béton 107,7 → 99,4 % | valeurs indicatives de l'EN 1992-4, texte sous licence non consulté : 7,7 / 11,0 gardés, du côté sûr |
| 0,9 sur le double grugeage des cornières | le SCI l'applique, le MSB non | les cornières suivent le MSB |
| `m2` (2ᵉ file, plats) | le moteur prend `min(VRd,g ; VRd,b)` (MSB), plus sévère que `min(VRd,g ; VRd,n)` (SCI) ; les cornières à grugeage court (L13) prennent ce même `Mc,Rd` (MSB §4.2.4.2) : 130 kNm au lieu des 164 du SCI sur la géométrie de son exemple 1 | du côté sûr, gardé |
| γM2 du poinçonnement | imprimé dans la procédure du SCI, omis dans ses exemples | le moteur le garde (côté sûr) |
| poutres de plus de 610 mm | le SCI borne `gh`, `(n1 − 1)·p1` et l'élancement | non porté : à ajouter comme avis |
| deux poutres en vis-à-vis | hors périmètre, déclaré dans la méthode (les hypothèses de la note n'ont pas la place, charte de trois lignes) | il faudrait une saisie « vis-à-vis » pour en faire un statut |
| taux du pied en soulèvement (fc1) | mesuré à N fixé (0,640 sur PP-21) ; à excentricité constante (Tableau 6.7) : 0,705 | même frontière, même verdict ; la marge affichée en traction est plus faible de 9 % à e constant — définition du taux à choisir |
| βw d'un cordon entre aciers différents (§4.5.3.2(6)) | les attaches de poutre prennent le βw de la pièce de plus faible fu, le pied le plus grand des deux | sur un acier personnalisé (fu 430, βw 1,0) : résistance − 11 à − 15 % selon la lecture, aucun verdict ne change — une convention à fixer |
| cordons du pied : Vx | porté par les seuls cordons parallèles (§4.9(2) l'admet) | la répartition élastique classique donne σ⊥ + 7,6 % (PP-12, sous la résistance) |
| bras de levier de l'EN 1992-4 (tiges au cisaillement) | αM = 2 (platine encastrant la tige) et e1 = tg | αM = 1 : PP-8 de 0,647 à 0,786 ; e1 au plan moyen de la platine : PP-19 à 1,87 — texte de l'EN 1992-4 à lire |
| tronçons en T non codifiés du pied (tige latérale, boulon de coin) | modèles du moteur prudents d'après des mécanismes de lignes de rupture (411 à 423 kN contre 234 ; 243 contre 131) | bornes supérieures : une indication, pas une preuve ; aucune règle publiée |
| Lb* d'une rangée d'une seule tige (Tableau 6.2, nb) | As/leff par tige : le double de la lecture « rangées de deux boulons » | sens non sûr possible si Lb est entre Lb*/2 et Lb* ; aucun des 92 exemples n'est dans ce cas — définition à lire |
| cornières sur poteau : bras de flA et wP ; effet de levier dans bi_P | 66,5 mm contre 70 mm ; Ft,Ed sans effet de levier (§3.11) | sans conséquence sur les exemples (flA à 0,22 ; bi_P couvert par la répartition prudente, non démontré) |
| A1 : maintien réel au déversement, seconde poutre | la note révisée écrit le maintien dans ses hypothèses et l'articulation justifiée à son verdict (L22) ; le verdict (vérifié, 87,8 %) vaut sous ces hypothèses | données du projet à confirmer par le bureau : si la poutre n'est pas maintenue, la saisie « Non » rend « À VÉRIFIER » ; une seconde poutre sur la même âme se vérifie avec les deux réactions |
| seconde poutre dans les hypothèses de la note | « âme porteuse chargée d'un seul côté » demanderait une quatrième ligne à presque toutes les notes grugées | la charte des notes (31/10/2026 : trois lignes, une quatrième seulement quand les options se cumulent) ; l'admettre est votre décision — le point est écrit dans la méthode |
| torsion du pied : béton en cisaillement | statut « à vérifier » | la répartition rive par rive (EN 1992-4 §7.2.2.5) est un modèle à écrire et à valider |
| fendage, éclatement latéral | statut au périmètre complet seulement ; avis au périmètre EC3 | le périmètre EC3 dit déjà que le béton d'ancrage est hors verdict (décision du lot B) |
| calcul manuel : couleur du statut | le second contrôleur propose « ancrages à justifier » en orange, toujours | presque toutes les platines ont des tiges chargées : l'orange ne distinguerait plus rien ; la portée est écrite (note, écran), la couleur reste celle du lot B |
| tiges du pied : acier de l'EN 1992-4 | `NRd,s = As·fuk/1,5` borne la tige (−7,3 % sur le SCI P398 E.1 : 105,2 % au lieu de 98,4 %) | plus sévère que l'exemple publié, du côté sûr |
| platine du pied : épaisseur retenue | `tc` (pression de la platine réelle) dépasse le critère d'aire des guides de 1 à 4 % | du côté sûr |
| tiges du pied au cisaillement | avec un mortier de plus de `d/2`, bras de levier de l'EN 1992-4 sans frottement | les exemples publiés comptent sur le frottement ; le moteur est plus sévère |
