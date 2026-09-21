# Écarts et corrections — doubles cornières d'âme

**Règle appliquée** : parité stricte avec le HTML de référence d'abord, erreurs
éventuelles comprises. Les tests de parité restent dans le dépôt tels quels.
Une erreur **manifeste** se corrige dans une étape séparée, avec un test qui
documente l'ancien et le nouveau résultat ; un **choix de modèle** est proposé,
jamais appliqué sans l'accord du bureau.

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
| EN 1993-1-1 | Tableau 3.1, §6.2.3(2), §6.2.5, §6.2.6 | `tnB`, `tnS`, `flB`, `flA`, `vgS` |
| MSB Part 5 | §3.2.3, §3.2.5.1, §4.1, §4.2.1.1, §4.2.1.2, §4.2.2, §4.2.2.1, §4.2.2.2, §4.2.3.1, §4.2.3.2, §4.2.4, §4.2.5, §4.3.1.1, §4.3.1.4, §4.3.2.2, §4.3.2.3, §4.4 | modèles COMP |
| ECCS n°126 | modèles complémentaires (cisaillement des cornières 1,27, section nette) | `cgB`, `cnB`, `vnS` |

Élément corroborant : les exemples §4.4 et §3.4 du MSB Part 5 sont reproduits
numériquement par le benchmark (37 valeurs) — les paragraphes cités pour ces
vérifications sont donc au moins cohérents avec le document source.

## 3. Points fragiles audités en priorité

### 3.1 `fu` du S355 (510 MPa) et coefficients partiels — inchangés

- `S355 : fu = 510` (EN 1993-1-1:2005 Tableau 3.1, t ≤ 40 mm) reste la valeur
  par défaut ; la ligne « S355 (fu 490) » et « S355 N/NL » existent dans la
  bibliothèque.
- γM0 = 1,00 ; γM2 = 1,25 (boulons, pression diamétrale, soudures) ;
  γM2 = 1,25 (sections nettes, blocs) ; γM3 = 1,25 ; γM3,ser = 1,10 : valeurs
  recommandées, modifiables à l'écran (groupe « Coefficients partiels »).
- **Où caler sur l'ANB** : les défauts dans
  `acier/assemblages/poutre_poutre/doubles_cornieres/entrees.py` (`defaults()`),
  les aciers dans `acier/donnees/bibliotheques.json` (une ligne par nuance).
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
