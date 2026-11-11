# Fondations de grues — étude avant le code (11/11/2026)

Ce document est l'**étape 1** de la demande : lire tout, redémontrer chaque formule, dire ce qui diffère, et fixer où
chaque calcul des classeurs ira dans le module. Rien n'est inventé : ce qui n'est pas dans les classeurs ou dans RBAR
est écrit comme tel (§ 9). La référence du module (méthode, preuve de fidélité) sera `docs/beton/FONDATIONS_GRUE.md`.

**En une page**

- Les quatre classeurs appliquent **une seule méthode** (contraintes admissibles : σ'b, μ, σ adm) à **trois schémas
  de fondation** : 4 pieux et 4 poutres en cadre ; 8 pieux, poutres primaires sur 2 pieux + poutres secondaires ;
  8 pieux (ou 4 pieux + une paroi de pieux sécants), 2 poutres support de châssis posées sur 4 semelles de 2 pieux.
  Le classeur « Greenwood » est le troisième schéma avec d'autres données (43 cellules de saisie changées, aucune
  formule changée). Les deux fichiers « BASE » sont identiques à l'octet près.
- Chaque formule a été recalculée à part : **toutes les valeurs des classeurs sont retrouvées** (541 cellules
  calculées des quatre fiches et 7 800 valeurs des grands tableaux, à 10⁻⁹ près ; `tests/test_fondation_grue_moteur.py`). La statique est
  partout celle d'une **poutre sur deux appuis** dont on déplace un appui ou la charge de la tolérance d'exécution
  « e ». Les grands tableaux (4 feuilles de 48 colonnes × 211 lignes) ne font que cela, 192 fois.
- **Deux erreurs de formule non conservatives** dans la variante 4 pieux — l'effort horizontal par pied (§ 5 E1,
  aussi dans la variante 8 pieux) et le moment maximal pris sous la charge (E16) — et une douzaine d'écarts ou
  d'incohérences entre classeurs, chiffrés au § 5. Aucun ne change un verdict des quatre exemples, mais E1
  sous-estime l'effort de 15 à 31 % et E16 le moment de 5 % dans la variante 4 pieux.
- Le plan d'armatures Greenwood **ne porte pas les armatures du classeur** sur quatre postes (§ 6) : c'est exactement
  l'écart que l'export doit supprimer.
- Le module **Poutres** du site n'est **pas** strictement identique aux classeurs (τ sur h au lieu de d, seuils de τ,
  bras de levier des étriers, μ au 4ᵉ chiffre, minimums d'armature) : il n'est donc **pas réutilisé** ; le module
  reprend les formules des classeurs, cellule par cellule (§ 7).
- **RBAR est un plugin AutoCAD** (pas Revit). Il sait déjà **importer des « modèles de poutre » en JSON**
  (nombre et Ø des barres basses, hautes, latérales, étriers, brins, pas, enrobage) et des types de pieux : c'est le
  format d'échange retenu (§ 8). Il n'a pas, aujourd'hui, de commande qui dessine tout un élément depuis un fichier :
  le plan « tout fait » demande une commande de plus dans RBAR, hors de ce dépôt.

---

## 1. Ce qui a été lu

| Source | Contenu | Nom court |
|---|---|---|
| `BASE - Fondation grue sur 8 pieux (4 semelles) ou 4 pieux + pieux sécants.xlsx` (+ sa copie, identique) | 9 feuilles | **BASE** |
| `GWW - …xlsx` (Greenwood, grue WOLF 7534) | 9 feuilles, mêmes formules que BASE | **GWW** |
| `VARIANTE - … 4 pieux….xlsx` | 2 feuilles | **VAR4** |
| `VARIANTE - … 8 pieux….xlsx` | 2 feuilles | **VAR8** |
| Plan de coffrage et plan d'armatures Greenwood (PDF) | cotes, niveaux, 12 repères, bordereau | **COF**, **ARM** |
| Dépôt `rbar` (lecture seule) | plugin AutoCAD, formats, commandes | **RBAR** |
| `beton/poutre/moteur.py`, `beton_classes.json` | module Poutres du site | **Poutres** |

Chaque classeur : aucune feuille, ligne ni colonne masquée ; une feuille `Base de données` identique dans les quatre ;
noms définis `Béton`, `Acier` (listes de validation) et `TypeBeton` (lien vers le classeur du bureau
« Dimensionnement poutre.xlsx ») ; mises en forme conditionnelles sur OK / LIMITE / PAS OK / A VERIFIER / Zone.
**Les seuls arrondis sont dans `Base de données`** ; tout le reste est calculé sans arrondi (seul l'affichage arrondit).

Feuilles de BASE et GWW : `Fiche de calcul` ; `R+V+M support chassis_CAS A.a`, `_CAS A.b`, `_CAS B.a`, `_CAS B.b`
(les grands tableaux) ; `R+V+M support chassis_CAS A`, `_CAS B` (maximum de deux tableaux) ;
`M semelle sur pieux_CAS A` (101 lignes) ; `Base de données`.

## 2. La méthode commune (redémontrée)

**Matériaux** (`Base de données`, identique partout) :

| Grandeur | Formule du classeur | C20/25 | C25/30 | C30/37 | C35/45 | « C45/50 » |
|---|---|---|---|---|---|---|
| R'wk (N/mm²) | donnée | 23,5 | 28 | 35 | 42 | 47,5 |
| σ'b | ARRONDI(R'wk / 2,7 ; 2) | 8,70 | 10,37 | 12,96 | 15,56 | 17,59 |
| μ (BE500) | ARRONDI(k·(1 − k/3)/2 ; 4), k = 15·σ'b / (300 + 15·σ'b) | 0,1363 | 0,1513 | 0,1708 | 0,1869 | 0,1975 |
| σ adm acier | ARRONDI(fy / 1,5 ; 2) | BE220 146,67 | BE400 266,67 | BE500 333,33 | | |

Démonstration de μ : section rectangulaire fissurée, n = 15, béton à σ'b et acier à σs : axe neutre x = k·d,
M = ½·σ'b·b·x·(d − x/3) = μ·σ'b·b·d². D'où la « hauteur calculée » : d nécessaire = √(M / (μ·b·σ'b)).
Remarque (pas une erreur du classeur, c'est la table du bureau, la même que dans Poutres) : k est établi avec
σs = 300 (BE500), 240 (BE400), 131 (BE220), soit 0,6·fy, alors que l'acier est dimensionné à fy/1,5.

**Section** (les mêmes dans les quatre classeurs, aux conventions du § 4 près) — M en kN·m, V en kN, b en cm, d en mm :

| Vérification | Formule |
|---|---|
| Hauteur | h calc = √(M·10⁶ / (μ·b·10·σ'b)) ≤ d (hauteur utile choisie) |
| Armature | Ω = M·10⁶ / (σ adm·0,9·d) ≤ Ω choisie = π·n·(Ø/2)² |
| Cisaillement | τ = V·10³ / (0,75·b·10·d) ; zones à 0,016·R'wk, 0,032·R'wk, 0,048·R'wk (0,56 / 1,12 / 1,68 en C30/37) |
| Étriers | pas calculé = n brins·π(Ø/2)²·σ adm·z / (V·10³) ≥ pas choisi |
| Renfort au pieu | console de largeur D sous la poutre : p1 = P/b, p2a = P/D, p2b = P/(D/2) ; M = p1·(b/2)²/2 − p2·(bras)²/2 ; suspentes Ω = P·10³/σ adm |
| Perfos | Ω = H·10³ / 150 (taux de travail 150 N/mm²) ; 3 par plaque |

## 3. Les quatre configurations

### 3.1 VAR4 — 4 pieux, 4 poutres en cadre carré

- **Géométrie** : un pieu sous chaque pied de la grue ; 4 poutres (90 × 50) de pieu à pieu, longueur = largeur du
  châssis (450 cm). Pieux Ø70.
- **Charges** : UNE charge verticale par pied (474 kN), H hors service 69, H en service 34, M 200 ; poids propre
  25·b·h.
- **Tolérance** e = 15 cm, deux positions du pieu A :
  - CAS A, pieu rentré de e : la charge est en porte-à-faux de e → R max, M min (fibre supérieure), V = P + pp·e ;
  - CAS B, pieu sorti de e : portée L + e, charge à e de l'appui → M max (fibre inférieure), V = RA.
- **Vérifications** : poutre (hauteur, Ω inf, Ω sup, τ avec « % de V repris », étriers à l'appui et en travée à
  80 cm du pieu) ; capacité du pieu (R max + pp·L/2 ≤ 600 kN) ; renfort au droit du pieu ; effort horizontal et perfos.

### 3.2 VAR8 — 8 pieux, 4 poutres primaires + 4 poutres secondaires

- **Géométrie** : à chaque angle, une poutre primaire (90 × 70, longueur 240) sur 2 pieux Ø41 à 160 d'entraxe ;
  4 poutres secondaires (40 × 50) de primaire à primaire, longueur = châssis (600).
- **Poutre secondaire** : mêmes formules que la poutre de VAR4 (sans « V en travée »), e = 15.
- **Poutre primaire** : charge = R max secondaire + pp secondaire·L/2, e = 20 :
  - CAS A, pieux rentrés de e/2 chacun et charge décalée de e/2 : portée s − e → R max, V max ;
  - CAS B, pieux sortis de e/2 chacun : portée s + e, charge centrée → M max.
- **Vérifications** : secondaire et primaire (hauteur, Ω, τ, étriers) ; renfort au pieu ; effort horizontal.
  **Pas de vérification de la capacité des pieux** (§ 5 E4).

### 3.3 BASE — 8 pieux (4 semelles) ou 4 pieux + paroi de pieux sécants

- **Géométrie** : 2 « poutres support châssis » parallèles (120 × 75), chacune sous deux pieds de la grue. Chaque
  bout repose sur une « semelle sur pieux » (250 × 90 × 75, 2 pieux Ø41 à 120) perpendiculaire à la poutre, ou, d'un
  côté, sur une paroi de pieux sécants. Le pied est à la distance a (25) de l'axe de la semelle et b (25) de l'axe de
  l'autre appui : portée = châssis + a + b.
- **Charges** : 6 cas de la fiche de la grue (3 en service, 3 hors service), une valeur par pied A, B, C, D ;
  H en service, H hors service, M. Chaque cas est joué dans les 8 orientations (A/B, B/A, B/C, C/B, C/D, D/C, D/A,
  A/D) : 48 colonnes.
- **Tolérance** e = 20 cm, quatre géométries de la poutre support :

  | Sous-cas | a' | b' | Sens |
  |---|---|---|---|
  | A.a | a + e | b | pieux d'une semelle décalés de e (portée + e) |
  | A.b | a | b + e | idem, de l'autre côté |
  | B.a | a + e | b − e | châssis posé décalé de e |
  | B.b | a − e | b + e | idem, dans l'autre sens |

  Par colonne : RA, RB, V max = max(RA ; RB), V central (entre les deux pieds), M max (maximum de M(x) tous les 5 cm),
  M hor éq (flexion horizontale de la poutre sous H/4 par pied, ramenée à un moment vertical : × h/b / 1,25),
  M max éq = M max + M hor éq ; « RB respectif » = RB de l'AUTRE poutre support sous le même cas (A/B ↔ D/C,
  B/A ↔ C/D, B/C ↔ A/D, C/B ↔ D/A) et « RB symétrique » = le plus grand RB quand les deux poutres donnent le même
  (pour la paroi de pieux sécants).
- **Semelle sur pieux** : charge = RA max de la poutre support, répartie sur c = 50 cm, e = 20 :
  - CAS A, pieux rentrés de e/2 chacun et châssis décalé de e/2 : portée s − e → R max, V max, M (maximum sur la
    zone chargée, 101 points) ;
  - CAS B, pieux sortis de e/2 chacun : portée s + e, charge centrée → M max.
- **Vérifications** : poutre support (hauteur, Ω inf sur M max éq, Ω sup = 25 % Ω inf, τ par zone, étriers aux
  extrémités sur V max et en partie centrale sur V central) ; semelle (idem, un seul pas d'étriers) ; pieu
  (sollicitation en tonnes ≤ capacité, avec marges 1,10 / 1,05) ; renfort au pieu ; effort horizontal et perfos ;
  contrôles de géométrie (h semelle ≥ h poutre + 5 ; largeur semelle ≥ Ø pieu + 40 ; entraxe ≥ 3 Ø ; a et b ≥ 25).
  États : OK / LIMITE (à 5 % près) / PAS OK.

### 3.4 GWW — Greenwood (le chantier des deux plans)

Même classeur que BASE. Données : poutre support 140 × 70 ; semelle 270 × 100 × 75 ; pieux Ø53 à 170, 75 T ;
châssis 600, a = b = 25 ; cas 1 : 369 / 1004 / 369 / 0, cas 2 : 790 / 790 / 80 / 80, cas 4 : 394 / 770 / 394 / 18,
cas 5 : 660 / 660 / 128 / 128 (cas 3 et 6 vides) ; H 51 / 129, M 430.
Résultats : RA max 1 101,45 kN ; M max 468,27 ; M max éq 470,50 ; V central 145,07 ; semelle R 649,47, M 461,31,
V 638,22 ; pieu 63,8 T ≤ 75 T.

## 4. Ce qui diffère entre les classeurs

| Sujet | VAR4 | VAR8 | BASE / GWW |
|---|---|---|---|
| Charges verticales | 1 valeur | 1 valeur | 6 cas × 4 pieds × 8 orientations |
| Hauteur utile d | h − 50 mm (− 75 au pieu) | secondaire h − 50 ; primaire h − 75 | h − enrobage − Ø étrier − Ø inf/2 |
| Ω sup | sur M min, avec d − 50 | secondaire : idem ; primaire : 40 % Ω inf | 25 % Ω inf |
| Moment de calcul de Ω inf | M max | M max | M max éq (avec l'effet de H) |
| Seuil de τ | 0,56 écrit en dur | 0,56 écrit en dur | 0,016 / 0,032 / 0,048 × R'wk |
| « % de V repris par la poutre » | 50 % sur τ seulement | 10 % sur τ et sur les étriers | 100 % (τ et étriers) |
| Bras des étriers z | 0,9·d | 0,9·d | hauteur de l'étrier (V max) ; d − enrobage (V central, semelle) |
| Capacité du pieu | R max + pp·L/2 ≤ capacité (kN) | **absente** | V max semelle / 10 ≤ capacité (T), marges 1,10 / 1,05 |
| Effort horizontal par pied | max(H hs/3 ; H s/3 + M/2/L/100) | idem | max(H hs/3 ; H s/3 + M / (2·L/sin 45°)) |
| Ω brins horizontaux au pieu | π(Ø/2)²·(brins/2)·(D/2 / pas) | idem | π(Ø/2)²·(D / pas) |
| Étriers au pieu choisis | barres complémentaires seules | idem | étriers de base (à compter) + complémentaires |
| États | OK / PAS OK | OK / PAS OK | OK / LIMITE / PAS OK |

## 5. Erreurs, incohérences et points à trancher

Le module reproduit **le classeur tel qu'il est** (règle de fidélité) et **affiche** chaque écart ci-dessous à côté
du résultat concerné, avec la valeur redémontrée. Rien n'est corrigé d'office : c'est au bureau de trancher.

| N° | Classeur · cellule | Constat | Effet chiffré | Sens |
|---|---|---|---|---|
| **E1** | VAR4 `C130`, VAR8 `C155` | `Hs/3 + M/2/L/100` avec L en cm : le terme du moment est divisé par 100·L(cm) au lieu de L(m). Il devient nul. | VAR4 : 23,0 kN au lieu de 27,0 (formule de BASE, bras = diagonale) ou 33,6 (M/2L) ; Ω perfos 153 au lieu de 180 ou 224 mm² (1Ø20 = 314 tient). VAR8 : 44,3 inchangé avec la formule de BASE, 45,9 avec M/2L (Ω 306 ≤ 314). | **non conservatif** |
| E2 | VAR4 `F63` `F112`, VAR8 `F78` `F103` `F137`, BASE/GWW `F231` | seuil de τ « 0,56 » écrit en dur | juste en C30/37 seulement (0,016 × 35) | faux pour un autre béton |
| E3 | BASE/GWW `C222` (= `C121`/10) | le pieu est vérifié avec V max de la semelle (effort tranchant au nu intérieur du pieu), pas avec sa réaction R max (`C119`, calculée et jamais utilisée) | BASE : 60,2 T au lieu de 61,5 T (capacité 65 T, état LIMITE dans les deux cas) ; GWW : 63,8 au lieu de 64,9 T | non conservatif (2 %) |
| E4 | VAR8 | aucune vérification de la capacité des pieux ; R max primaire (`C62` = 516,8 kN) n'est comparé à rien | — | manque |
| E5 | VAR8 `C37` | RB = −6,4 kN en CAS A (soulèvement de l'appui B de la poutre secondaire), sans avertissement | — | à signaler |
| E6 | VAR4 `C63` / `C66` | τ calculé sur 50 % de V, étriers à l'appui calculés sur 100 % de V | les deux lignes ne parlent pas du même effort | incohérent (VAR8 et BASE sont cohérents) |
| E7 | BASE/GWW `C152`, `C189` | bras des étriers = d − enrobage (648) alors que la « hauteur » `G156` / `G193` (658) est calculée à côté et n'est pas utilisée ; `C145` utilise bien `G149` | pas calculé 1,5 % plus petit | conservatif |
| E8 | BASE/GWW tableaux | M max = maximum de M(x) tous les 5 cm (semelle : tous les 1 % de la zone chargée) ; tableau arrêté à x = 10,5 m | ici 0,001 % ; jusqu'à ≈ 1 % si a ± e n'est pas un multiple de 5 cm ; poutre de plus de 10,5 m tronquée | non conservatif (faible) |
| E9 | BASE/GWW `C126` | h calc sur M max, Ω inf sur M max éq | 0,9 % ici | incohérent (faible) |
| E10 | BASE/GWW `C104` | la semelle est calculée avec RA max seulement ; avec 4 semelles et a ≠ b, celle du côté b reçoit RB max | nul quand a = b (les deux exemples) | limite à connaître |
| E11 | BASE/GWW `C254` vs VAR | deux bras de levier pour le moment d'orientation : diagonale du châssis (BASE) ou côté (VAR, par intention) | voir E1 | à unifier |
| E12 | libellés | feuille `…_CAS A` : en-têtes « CAS B.a / CAS B.b » (`B15:C15`) ; « Effort tranchant poutre secondaire » au-dessus de la semelle (`B198`) ; « PP poutre secondaire / primaire » dans les tableaux de BASE ; classe « C45/50 » pour R'wk 47,5 (C40/50) ; σs = 131 pour BE220 | aucun | présentation |
| E13 | BASE/GWW libellés des cas | le cas « gouvernant » affiché dépend du dernier bit quand deux cas miroirs donnent la même valeur (M max « CAS 1.b-A/B » = « CAS 1.a-B/A ») | aucun | présentation |
| E14 | tous, cellules d'état | « OK » exige une inégalité stricte alors que le signe affiché est « ≤ » | à l'égalité exacte seulement | présentation |
| E15 | BASE `G152` | pas choisi en partie centrale 750 mm > hauteur de l'étrier 658 mm : le « Check » `G158` est un texte, pas un état | l'exemple BASE est hors de sa propre règle | à signaler |
| **E16** | VAR4 `G37` (et VAR8 `G39`, même formule) | « M max (inf.) » = RA·e − pp·e²/2 est le moment SOUS LA CHARGE. Quand l'effort tranchant ne change pas de signe sous la charge (RA > P + pp·e), le maximum est plus loin dans la travée. Trouvé par la statique indépendante. | VAR4 : 72,6 kN·m au lieu de 76,3 (+5,2 %), Ω inf 538 au lieu de 566 mm² (5Ø16 = 1 005 tient). VAR8 : aucun effet (l'effort tranchant change de signe sous la charge). | **non conservatif** |

Différences de **convention** (pas des erreurs, à garder par configuration) : d = h − 50 ou d = h − enrobage − Ø − Ø/2 ;
Ω sup à 25 %, 40 % ou sur M min ; une charge par poutre dans les variantes.

## 6. Plan Greenwood ↔ classeur GWW

Le plan nomme « poutre transversale » la poutre support châssis (140 × 70, 750 de long, 650 entre axes de semelles)
et « poutre longitudinale » la semelle sur pieux (270 × 100 × 75, 2 pieux Ø53 à 170). Quatre semelles, pas de paroi.

| Élément | Poste | Classeur GWW | Plan ARM | |
|---|---|---|---|---|
| Poutre support | inférieures | 8Ø20 (2 513 ≥ 2 458 mm²) | 9Ø20 (rep. 10) | ≠ |
| | supérieures | 4Ø16 | 6Ø16 (rep. 9) | ≠ |
| | étriers aux extrémités | 6 brins Ø12 pas 100 | rep. 6 + 7 : 6 brins Ø12 – 10 sur 2 × 200 cm | = |
| | étriers partie centrale | 4 brins Ø12 pas 300 | rep. 8 : 4 brins Ø12 – 30 sur 350 cm | = |
| | latérales | « 2Ø12 tous les 30 à 40 cm » | rep. 11 : Ø12 | = |
| Semelle | inférieures | 8Ø20 (2 513 ≥ 2 235 mm²) | 6Ø25 (rep. 3, 2 945 mm²) | ≠ |
| | supérieures | 4Ø16 | 4Ø16 (rep. 2) | = |
| | étriers | 4 brins Ø12 pas 125 | rep. 1 : Ø12 – 12,5 | = |
| | renfort au pieu | 5 brins de base + 16 brins Ø12 | rep. 5 : 2 × 10 U Ø12 par pieu | ≠ |
| Perfos | par plaque | 3Ø25 | rep. 12 : 3Ø25 | = |

Le plan ajoute ce que le classeur ne dit pas : formes (U 65/60/65, C 50/742/50…), retours, longueurs, longueur des
zones d'étriers (200 / 350 / 200), nombre de barres latérales. Règle de longueur des étriers du bordereau non
retrouvée (2(a + b) + d = 247 / 277 / 327 affiché 2,40 / 2,70 / 3,20).

## 7. Module Poutres du site ↔ classeurs : pas identiques

| Sujet | Poutres (site) | Classeurs grue |
|---|---|---|
| μ C30/37 BE500 | 0,1709 (σ'b non arrondi avant μ) | 0,1708 |
| μ C35/45 BE500 ; C20/25 BE400 | 0,1868 ; 0,1555 | 0,1869 ; 0,1554 |
| σ acier | fyk/γs = 333,333… | 333,33 (arrondi) |
| Hauteur | hu,min + yG ≤ h | h calc ≤ d |
| τ | V / (0,75·b·**h**) | V / (0,75·b·**d**) |
| Seuils de τ | k·fck,cube/1,05, k = 0,016 / 0,032 / **0,064** | k·R'wk, k = 0,016 / 0,032 / **0,048** (R'wk = 35 ≠ 37/1,05) |
| Étriers | s = Ast·fyd·d / V, plafonné à min(0,75·d ; 30 cm) | 0,9·d ou hauteur d'étrier, sans plafond |
| Minimums | 0,26·fctm/fyk·b·h ; 0,0013·b·h ; 25 % opposé ; maximum 4 % | 25 % ou 40 % de Ω inf en haut seulement |

Conséquence : le module Fondations de grues a **ses propres formules, recopiées des classeurs** ; il ne partage avec
Poutres que la présentation (écran, note). Que deux outils « du bureau » ne donnent pas le même τ est un point à
connaître ; il n'est pas traité ici.

## 8. RBAR et le dessin automatique

- RBAR est un plugin **AutoCAD** (C#, .NET 8). Principe écrit dans son dépôt : « RBAR dessine, il ne dimensionne
  pas » ; Excel n'y est jamais une source de vérité ; un modèle = une **intention**, jamais des coordonnées.
- **Ce qui existe et que l'export utilise** :
  1. **Modèles de poutre en JSON** (palette › liste des modèles › « Importer JSON… ») : un modèle seul, un tableau,
     ou un objet `{"modelesPoutre": [...]}`. Champs : `nom`, `bas {n, d}`, `haut {n, d}`, `basSuivants`,
     `lat {nParFace, d}`, `etrier {d, espacement, brins, appuis {espacement, longueur_mm}, zones [...]}`, `enrobage`
     (mm). Les champs inconnus sont gardés tels quels : l'export y joint la traçabilité (note, date, vérification).
  2. **Types de pieux et gabarits de massifs** (`types.json` : `code`, `nom`, `diametre_mm`, `charge_t`, `etat` ;
     `sorte: massif`, `hauteur_mm`, `debord_mm`) pour `RBARPIEU` et `RBARMASSIF`.
  3. `RBARPOUTRE` › Dessiner « 140x70 » › modèle › longueur : RBAR dessine la coupe, les barres, les étriers, les
     étiquettes et le bordereau ; `RBARELEVATION` l'élévation.
- **Ce que le calcul fournit** (et que l'export transmet sans le recalculer) : par élément, dimensions, enrobage,
  barres inférieures et supérieures (nombre, Ø), étriers par zone (brins, Ø, pas), armatures latérales, étriers de
  renfort au pieu (brins, Ø), perfos (nombre, Ø), pieux (Ø, capacité), implantation (entraxes, a, b).
- **Ce que ni les classeurs ni l'export ne fixent** : formes et retours des barres, ancrages, recouvrements,
  longueur des zones d'étriers. RBAR les prend dans le standard du bureau (`CompanyStandard.json`) ; le module ne les
  invente pas.
- **Limites de RBAR aujourd'hui** : un seul nombre de brins par modèle (extrémités à 6 brins et centre à 4 brins =
  deux modèles, donc deux coupes) ; étriers de renfort au pieu et perfos hors du modèle de poutre (ils partent dans
  le tableau des armatures, à poser avec `RBARPOSER`) ; **aucune commande ne dessine un élément complet depuis un
  fichier**. Le plan « tout fait » demande une commande d'import dans RBAR ; le fichier exporté est déjà fait pour
  elle (géométrie + modèles + barres dans un seul JSON).

## 9. Le module

**Emplacement** : `beton/fondation_grue/` (moteur, bibliothèque standard seule), `api_web/fondation_grue.py` (écran du
site, format de l'établi Poutres), tests `tests/test_fondation_grue_*.py`, valeurs des classeurs figées dans
`tests/reference_fondation_grue.json` (extraites cellule par cellule, datées).

**Deux étages, pour que chaque résultat s'explique** :

1. `classeur` — la transcription mot pour mot des formules (mêmes opérations, mêmes unités, mêmes arrondis, mêmes
   conventions) : c'est elle qui donne les résultats du module. Preuve : chaque cellule calculée des quatre classeurs
   retrouvée à 10⁻⁹ près.
2. `statique` — **la poutre sur deux appuis**, résolue exactement (réactions, V(x), M(x), extremums), indépendante
   du premier étage. Elle trace les diagrammes et **redémontre** les formules : un test exige que les deux étages
   concordent, et nomme les endroits où ils diffèrent (§ 5).

**Ce que voit l'ingénieur à la place des grands tableaux** : pour chaque poutre et chaque semelle, UN dessin — la
poutre, ses appuis déplacés de ± e, les charges — avec une courbe de moment et une courbe d'effort tranchant par
position de tolérance (2 à 4 courbes), l'enveloppe en trait fort, les extremums cotés et le cas qui gouverne nommé.
Le tableau des cas reste disponible dans la fenêtre de calcul.

**Écran** (liste / dessin / résultat) : choix de la configuration (les trois schémas ; « paroi de pieux sécants d'un
côté » pour le troisième) ; rubriques courtes : matériaux, géométrie, tolérances, charges de la grue (tableau cas ×
pieds), armatures choisies ; **vue en plan redessinée à chaque saisie** (axes, pieux, semelles, poutres, pieds de la
grue, cotes a, b, entraxes, réactions) ; diagrammes par élément ; résultats groupés par élément avec leur fenêtre
de calcul (symbole = formule = valeurs = résultat, cellule du classeur en regard).

**Note PDF** : même gabarit que Poutres et Dalles ; hypothèses, charges, sollicitations avec diagrammes,
vérifications avec formules, armatures retenues ; dernière page : le plan de principe coté.

**Export** : (1) `…-RBAR.json` (§ 8) ; (2) tableau des armatures (repère, élément, poste, forme si connue, Ø,
nombre, pas, zone) ; (3) plan de principe en DXF pour AutoCAD (pieux en cercles au diamètre réel, contours, axes).
Contrôle avant export : une liste « donnée → valeur → origine » et les manques en clair ; l'export est refusé si une
vérification est PAS OK ou si une armature choisie manque.

**Non couvert par les classeurs** (dit à l'écran, jamais calculé en silence) : pieu décalé de façon voulue (hors
tolérance) sur sa semelle, semelle à plus de 2 pieux, poutre support à plus de 2 pieds, vérification de la paroi de
pieux sécants (le classeur donne seulement les réactions RB), ancrages et recouvrements.

## 10. Correspondance classeur → module

Clés du résultat du moteur (`verifier(fondation)`), puis leur place à l'écran (rubrique de résultats) et dans la note.

### VAR4 (`type = "cadre4"`)

| Cellules | Grandeur | Clé du moteur | Écran · note |
|---|---|---|---|
| `G13:G16` | R'wk, σ'b, μ, σ adm | `materiaux.*` | Hypothèses |
| `C21` | pp poutre | `pp.poutre` | Charges |
| `C34 C35 C37 C38 C39` | CAS A : RA, RB, M min, V max, V travée | `poutre.cas_A.*` | Poutre · diagrammes |
| `G34 G35 G37 G38 G39` | CAS B : RA, RB, M max, V max, V travée | `poutre.cas_B.*` | Poutre · diagrammes |
| `C43:C47` | valeurs dimensionnantes | `poutre.dim.*` | Poutre · verdict |
| `C53 G53` | h calc, d | `section.poutre.hauteur` | Poutre › hauteur |
| `C55 G55`, `C59 G59` | Ω inf, Ω sup (calculées, choisies) | `section.poutre.inf`, `.sup` | Poutre › armatures |
| `C63 F63` | τ, texte | `section.poutre.tau` | Poutre › cisaillement |
| `C66 G66`, `C70 G70` | pas à l'appui, en travée | `section.poutre.etriers_appui`, `.etriers_travee` | Poutre › étriers |
| `C84 G84 G88` | charge du pieu, capacité, état | `pieux.*` | Pieux |
| `C94:C97 C101 C102` | P, p1, p2a, p2b, M A, M B | `renfort.charges.*`, `renfort.moments.*` | Renfort au pieu |
| `C105 C106 G105 C108 C109 G108` | h calc, d, Ω sup, Ω brins horizontaux | `renfort.hauteur`, `renfort.sup` | Renfort au pieu |
| `C112 C114 G114` | τ, Ω étriers, choisie | `renfort.tau`, `renfort.etriers` | Renfort au pieu |
| `C130 C133 G133 D139` | H par pied, Ω perfos, choisie, par plaque | `horizontal.*` | Effort horizontal |
| `G75 G77:G80 G88 G120 G122 G123 G139` | états | `verifications[].etat` | pastilles |

### VAR8 (`type = "diagonales8"`)

| Cellules | Grandeur | Clé du moteur |
|---|---|---|
| `G13:G16`, `C22 C23` | matériaux, poids propres | `materiaux.*`, `pp.secondaire`, `pp.primaire` |
| `C36 C37 C39 C40`, `G36 G37 G39 G40`, `C43:C46` | poutre secondaire : CAS A, CAS B, dimensionnant | `secondaire.cas_A.*`, `.cas_B.*`, `.dim.*` |
| `C50` | charge sur la primaire | `primaire.charge` |
| `C55 C56 C58 C59`, `G55 G56 G58 G59`, `C62:C64` | poutre primaire : CAS A, CAS B, dimensionnant | `primaire.cas_A.*`, `.cas_B.*`, `.dim.*` |
| `C68 G68 C70 G70 C74 G74 C78 C81 G81` | section secondaire | `section.secondaire.*` |
| `C93 G93 C95 G95 C99 G99 C103 C105 G105` | section primaire | `section.primaire.*` |
| `C119:C122 C126 C127 C130 C131 G130 C133 C134 G133 C137 C139 G139` | renfort au pieu | `renfort.*` |
| `C155 C158 G158 D164` | effort horizontal | `horizontal.*` |
| `G86 G88:G90 G110 G112:G114 G145 G147 G148 G164` | états | `verifications[].etat` |

### BASE et GWW (`type = "semelles8"`)

| Cellules | Grandeur | Clé du moteur |
|---|---|---|
| `G12:G15`, `C19`, `C43 C44` | matériaux, longueur de la poutre, poids propres | `materiaux.*`, `geometrie.longueur_support`, `pp.support`, `pp.semelle` |
| tableaux `…CAS A.a` … `B.b` : `B5:B9` | Lc, a', b', L, pp | `support.sous_cas[k].geometrie` |
| idem, lignes 24 à 35, colonnes B à AW | P1, P2, H/4, RA, RB, RB respectif, symétrie, V max, V central, M éq, M hor éq, M max | `support.sous_cas[k].colonnes[j].*` |
| idem, `B12:B19`, `D12:D19` | maximums du sous-cas et cas gouvernant | `support.sous_cas[k].max.*` |
| feuilles `…CAS A`, `…CAS B` : `B5:B12` ; fiche `C79:C87`, `G79:G87` | maximums CAS A, CAS B | `support.cas_A.*`, `support.cas_B.*` |
| fiche `C90:C97` | valeurs dimensionnantes | `support.dim.*` |
| fiche `C104 C112 C113 C115 C116`, `G112 G113 G115 G116`, `C119:C121` ; feuille `M semelle…` `B5:B10 B16` | semelle : charge, CAS A, CAS B, dimensionnant | `semelle.charge`, `semelle.cas_A.*`, `semelle.cas_B.*`, `semelle.dim.*` |
| fiche `C126 G126 C128 G128 C135 G135 C142 G142 C145 G145 G149 C152 G152` | section de la poutre support | `section.support.*` |
| fiche `C172 G172 C174 G174 C180 G180 C187 C189 G189` | section de la semelle | `section.semelle.*` |
| fiche `C209 C212:C214 C218 C219 C222 G222 C224 C225 C227 C228 G227 C231 C233 G233` | pieu et renfort au pieu | `pieux.*`, `renfort.*` |
| fiche `C254 C257 G257 D263` | effort horizontal | `horizontal.*` |
| fiche `G161 G163 G165:G168 G197 G199 G201:G203 G241 G243 G245 G246 G263` | états | `verifications[].etat` |
| fiche `F25 F26 F29 F35` (mises en forme) | contrôles de géométrie | `controles_geometrie[]` |

Chaque ligne de ces tableaux devient une ligne du test de fidélité (classeur, feuille, cellule, clé, valeur figée).

## 11. Questions au bureau

1. **E1** : quelle formule pour l'effort horizontal dû au moment dans les variantes — celle de BASE (bras = diagonale
   du châssis) ou M / (2·L) ? En attendant, le module affiche la valeur du classeur **et** les deux autres.
2. **E3 / E4** : le pieu se vérifie-t-il sur sa réaction R max (proposé) ? Faut-il la même vérification dans la
   variante 8 pieux diagonale ?
3. **E2, E6, E7, E9** : accord pour, plus tard, une option « formules harmonisées » (seuil de τ lié au béton, même
   effort pour τ et pour les étriers, hauteur d'étrier partout) à côté du mode « classeur » ?
4. **Plan** : la règle « le plan porte ce que dit la note » vaut-elle aussi quand le dessinateur veut plus d'acier
   (9Ø20 pour 8Ø20) ? Proposition : la note porte l'armature **choisie** ; si le plan change, on change le choix dans
   la note, qui revérifie.
5. **RBAR** : accord pour l'export « modèles + tableau + DXF » maintenant, et la commande d'import complète dans
   RBAR comme lot suivant de ce plugin ?
6. Les deux « méthodes du bureau » (Poutres du site et classeurs grue) doivent-elles converger (§ 7) ?
