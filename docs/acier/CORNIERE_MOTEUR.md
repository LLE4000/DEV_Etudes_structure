# Cornières d'appui — le moteur refondu sur la méthode du classeur (10/11/2026)

*Moteur `acier/corniere/` (`entree.py`, `statique.py`, `moteur.py`) : bibliothèque standard, le catalogue des profilés
`acier/profiles` (numpy, présent dans le navigateur) et les nuances d'`acier/flambement/normes.py` ; Python 3.12 et
3.13 (Pyodide). Référence : le **classeur de l'ingénieur**, feuille `P8`, analysé dans
`docs/acier/corniere_excel/ANALYSE_CORNIERE_EXCEL.md` (07/11) ; les choix sont ceux de `CORNIERE_DECISIONS.md` (10/11),
qui font foi. Preuve : `tests/test_corniere_moteur.py` (106 contrôles, 15 s). Ce document dit ce qui est LIVRÉ : le
moteur (lot « moteur »), les vues (`acier/corniere/schemas.py`, lot « schémas », § 10), le module du site
(`api_web/corniere.py`, lot « écran », § 11) et la note PDF (`acier/corniere/note.py`, `api_web/note_corniere.py`,
lot « note », § 12).*

## 1 · Ce qui est livré

| Pièce | Rôle |
|---|---|
| `acier/corniere/entree.py` | les champs du § 2 des décisions (mm, kN, N/mm²), leurs défauts et leur domaine ; le catalogue (familles L et LI, tri par masse, désignation tapée) ; les nuances du site (source unique) ; `normaliser` stricte ; la relecture d'une note v1 (modèle lame) |
| `acier/corniere/statique.py` | la statique d'UNE combinaison (une position de la charge, une de l'ancrage) : M, N, V le long des deux ailes pour chaque mode et configuration ; l'épaisseur du congé t(s) ; la recherche du maximum (grille ≤ 0,1 mm puis section dorée) |
| `acier/corniere/moteur.py` | l'enveloppe des neuf combinaisons, les vérifications (S1–S4, maxima, section nette), les ancrages, les diagrammes, les formules du détail (avec la cellule du classeur en historique), les avertissements, la comparaison des modes, la garde, le sélecteur — et le **contrat complet** dans sa docstring |
| `tests/test_corniere_moteur.py` | cas de contrôle à 1e-9 face aux 93 formules de `P8` codées dans le test, 240 saisies au sort, enveloppe, équilibre, B, ailes inégales, maxima, L_eff, formules, domaine, relecture v1, sélecteur, pureté 3.12 et 3.13, 28 mutations |
| `acier/corniere/schemas.py` (lot « schémas », 11/11) | les quatre VUES partagées par l'écran et la note — coupe, moments, contraintes, élévation — construites sur le kit de dessin des assemblages à partir de la seule sortie de `verifier` ; le contrat de sélection `data-sel` ; `ajuster`, `legende`, `secours` (§ 10) |
| `api_web/dessin_corniere.py` | la couche mince de l'écran : `vues(res, ident)` → `[{cle, titre, court, svg, legende, echelle}]`, ajustées à la toile de l'établi (13 px dans 626 × 682) |
| `tests/test_corniere_schemas.py` | 96 contrôles : échelle, orientation A/B, sections exactes, bras du mode, data-sel, diagrammes au prorata de s, peintres de la note, pureté, 9 mutations et un témoin |

## 2 · L'API et le contrat

```python
from acier.corniere import moteur as M

M.nouvelle_corniere()                 # L 120×120×15, A, historique, S355, coef 1,5, R = 35 kN, L = 300, X = 75 ± 20, Y = 85 ± 10, 3 Ø16
M.normaliser(c)                       # complétée et contrôlée ; ValueError hors domaine, jamais une valeur ramenée en silence
M.verifier(c)                         # le calcul complet (= calculer(c, diagrammes=True, annexes=True))
M.calculer(c, diagrammes=False, annexes=False)   # le même sans les tableaux des diagrammes ni la comparaison / la garde (le sélecteur)
M.combinaisons(c)                     # les 1, 3 ou 9 combinaisons de jeux, la nominale d'abord
M.candidats(c, familles=None)         # les 233 cornières triées par masse, chacune calculée avec les charges de c
M.optimal(c)                          # la plus légère qui tient ET dans le domaine
M.section(c) · M.cornieres() · M.section_catalogue(nom) · M.nom_catalogue("80x80x8") · M.designation("L 45x45x4.5")
```

**Entrée** (une clé absente ou `null` prend le défaut ; `a`, `R_ELU`, `uV` vides = auto) : `nom`, `standard`, `profil`
(catalogue) ou `A`, `B`, `t`, `r` (saisie libre), `aile_horizontale` (« A » : la première aile du nom porte la
charge), `configuration` (« A » en bas, « B » en haut), `fy` (235, 275, 355, 460), `coef` (σ_adm = f_y/coef,
τ_adm = σ_adm/√3), `R` (kN ELS), `L`, `X`, `dx`, `a` (vide = L), `R_ELU` (vide = 1,5·R), `Y`, `dy`, `n`, `d`
(d0 = d + 2), `mode` (« historique » ou « equilibre » ; B impose l'équilibre), `jeu_ancrage` (le C76 du classeur),
`yc`, `uV` (vide = e/2), `v1` (relue d'une note du module d'avant). Y se compte depuis la **face extérieure de l'aile
horizontale** dans les deux configurations : la même cote pour le même trou quand on retourne la cornière.

**Domaine** (ValueError avec message) : e + r < X − dx et X + dx ≤ larg (la charge sur la partie droite pour ses
trois positions) ; Y − dy − d0/2 ≥ e + r et Y + dy + d0/2 ≤ haut (l'ancrage, trou compris, dans la partie droite — les
mêmes deux conditions en A et en B) ; t < A, t < B, e + r < A, e + r < B ; r ≥ 0 ; L > 0 ; R ≥ 0 ; coef ≥ 1 ; dx ≥ 0 ;
dy ≥ 0 ; n ≥ 1 entier ; d > 0 ; n·d0 < L ; a > 0 ou vide ; R_ELU ≥ 0 ou vide ; 0 ≤ jeu_ancrage < Y ; 0 ≤ yc < bras le plus
court (A : Y − dy ; B : haut − Y − dy) ; 0 ≤ uV ≤ e ; f_y parmi les nuances du site ; nom ≤ 60 caractères.

**Sortie** (JSON strict, nombres non arrondis ; le contrat clé par clé est dans la docstring de `moteur.py`) :
`corniere`, `section` (dont `larg`, `haut`, `e`, `r`, masse et aire — calculées en saisie libre, +0,5 % sans les
arrondis de bout d'aile), `configuration`, `mode` (effectif) et `mode_demande`, `materiau`, `geometrie` (sections
repères, congé, bloc d'angle, positions de la charge et de l'ancrage, L_eff), `combinaisons`, `efforts` (Mmax, H,
H_sections, C, M_angle), `verifications` (S1, S2, S3, S4, max_h, max_v, nette en B — chacune avec s, t, largeur, W, A,
M signé, N, V, σ, τ, σ_éq, taux, taux_σ, taux_τ, ok, `verdict`, sa **combinaison**, sa **cellule** en historique),
`informations` (pression diamétrale, arrachement, compression du mur ou pression sur le support, taux de l'autre mode,
L_eff), `ancrages` (N et V ELS et ELU par ancrage, positions z, bras, méthode), `diagrammes` (par aile : les points de
l'enveloppe, les trois courbes de moment, les tronçons N/V, les repères), `maxima`, `comparaison`, `garde`,
`taux_max`, `statut`, `gouvernant`, `calcul` (les blocs de formules « symbole = expression = valeurs = résultat »),
`hypotheses`, `avertissements`.

Signe des moments : M > 0 quand la face côté congé (l'intérieur de l'angle) est tendue — tout positif en A (u_V = e/2),
tout négatif en B. Abscisse commune `s` = distance au dos de l'autre aile : la charge est à s = X′, l'ancrage à s = Y′,
dans les deux configurations ; sous s = e le bloc d'angle ; le congé entre e et e + r.

## 3 · La méthode

### 3.1 Géométrie commune
e = t ; r = congé (r1 au catalogue, saisi sinon) ; larg = aile qui porte la charge, haut = l'autre ; X′ = X + Δx,
Y′ = Y + Δy pour une combinaison ; W(t) = L·t²/6, A(t) = L·t ; épaisseur dans le congé t(s) = e + r − √(r² − (e + r − s)²)
pour e ≤ s ≤ e + r, t = e au-delà. Sections repères : S1 (aile horizontale, s = e + r, t = e), S2 (s = e, t = e + r),
S3 (aile verticale, s = e + r, t = e), S4 (s = e, t = e + r). Contraintes sur la fibre la plus sollicitée :
σ = |M|/W + |N|/A, τ = |V|/A, σ_éq = √(σ² + 3τ²), taux = σ_éq/σ_adm.

### 3.2 Les jeux : trois positions et l'enveloppe (décisions § 3.0)
La charge à X − dx, X, X + dx ; l'ancrage à Y − dy, Y, Y + dy : neuf combinaisons (une quand dx = dy = 0, trois quand un
seul jeu est nul), la nominale d'abord. Chaque grandeur vérifiée — chaque section, chaque point des diagrammes, la
traction, la compression du mur, la section nette, la largeur efficace — est calculée pour chaque combinaison et garde
la plus défavorable, avec la combinaison qui la gouverne (« charge à X + 20, ancrage à Y − 10 » ; à égalité la
nominale). L'enveloppe retrouve seule que l'ancrage haut charge les sections en A et l'ancrage bas la traction (en B
aussi, « bas » = vers le bout libre = Y + dy).

| Cas de contrôle (L120×120×15, R = 42,76, X = 75, Y = 85) | taux max | M1 | M3 | traction / ancrage |
|---|---|---|---|---|
| dx = dy = 0 (P8 exactement) | **88,86 %** (S3) | 2,00972 | 2,35870 | 14,253 kN |
| dx = 20 | 112,43 % | 2,86492 | 2,98768 | 18,054 |
| dy = 20 | 95,38 % | 2,00972 | 2,53271 | 14,253 (les ancrages du classeur ne lisent pas Yj) |
| dx = dy = 20 | 120,69 % | 2,86492 | 3,20810 | 18,054 |
| jeu_ancrage = 20 | 88,86 % | — | — | 16,446 |

Les diagrammes rendent, par aile, **une courbe de moment par position de la charge** (chacune enveloppe des trois
positions de l'ancrage, point par point ; l'ancrage qui gouverne la section de référence est nommé) et l'**enveloppe**
(le maximum point par point des courbes, qui est aussi celui des neuf combinaisons), les contraintes σ, τ, σ_éq et le
taux de l'enveloppe, l'épaisseur t(s), les tronçons N/V, les repères (sections, congé, bloc d'angle, positions de la
charge, bande de l'ancrage, contact, maximum). Le pas : ≈ r/26 dans le congé, ≈ 1/40 de la partie droite.

### 3.3 Mode historique (A) — `P8` mot pour mot
Xe = X − r − e (`C19`), Xe+r = X − e (`C20`), Ye = Y − e − r (`C23`), Ye+r = Y − e (`C24`), YH = Y − e/2 (`C25`), tous
NOMINAUX ; le jeu de la charge Xj = Δx s'ajoute dans les moments : M1 = R·(Xe + Xj) (`C35`), M2 = R·(Xe+r + Xj) (`C45`),
Mmax = R·(X + Xj) (`C55`) ; le jeu de l'ancrage Yj = Δy entre dans ρ₃ = (Ye + Yj)/(YH + Yj) (`C56`) et ρ₄ (`C64`) ;
M3 = ρ₃·Mmax (`C57`), M4 = ρ₄·Mmax (`C65`) ; σᵢ = Mᵢ/Wᵢ (`G31`…), τᵢ = R/(L·tᵢ) (`G34`… — le classeur, même dans
l'aile verticale où c'est en réalité un effort normal), σ_éq,ᵢ (`G37`…), taux (`I37`, `I47`, `I57`, `I67`) ; le verdict
est le maximum des quatre (`D3`). Diagrammes : M(s) = R·(X′ − s) sur l'aile horizontale, M(s) = H_s·(Y′ − s) sur la
verticale avec H_s = Mmax/(YH + Yj). Ancrages : N_ELS = Mmax/((Y − jeu_ancrage)·n) (`C82`), V_ELS = R/n (`C83`), ELU ×
R_ELU/R (`C85`, `C86`) ; pression diamétrale V_ELS/(d·e) (`C88`) et arrachement V_ELS/(2·(haut − Y)·e) (`C90`) en
information, Y nominal comme le classeur. Les **maxima du congé** sont calculés et rendus (vue Contraintes) mais restent
en **information** dans ce mode (`verdict: false`) : le classeur ne les a pas, et son cas de contrôle doit rendre
88,8621 %.

Exemple, le cas de contrôle (dx = dy = 0, jeu 10, R_ELU = 68,9) : M1 = 2,009720 ; M2 = 2,565600 ; M3 = 2,358697 ;
M4 = 2,896645 kN·m ; σ₃ = 209,662, τ₃ = 9,502, σ_éq,3 = 210,307 N/mm², σ_adm = 236,667 → **88,8621 %** en S3 ;
N_ELS = V_ELS = 14,2533 kN, N_ELU = V_ELU = 22,9667 kN (rapport 1,61132) ; maximum du congé à 1,66 mm du pied de S3
(+1,45 %, 90,15 %, information) ; comparaison : l'équilibre P0 donne 77,94 %.

### 3.4 Mode équilibre (A) — un seul corps libre
Pour une combinaison : H = R·(X′ − u_V)/(Y′ − y_c) (pivot y_c depuis le dessous, cisaillement d'ancrage à u_V du nu du
mur, frottement nul). Aile horizontale : M = R·(X′ − s), V = R, N = 0. Aile verticale sous l'ancrage :
M = H·(Y′ − s) − R·(e/2 − u_V), N = +R, V = H ; rien au-dessus. Vérifications : S1–S4 et le **maximum le long de t(s)
dans chaque aile** (dans le verdict). Ancrages : N_ELS = H/n, V_ELS = R/n, ELU × R_ELU/R ; compression du mur C = H à
y_c (information) ; arrachement avec l'ancrage le plus près du bord (Y + dy).

Exemple, le cas de contrôle en P0 (y_c = 0, u_V = 7,5) : H = 33,956 kN, M3 = 1,93552 kN·m, S3 76,91 %, maximum de l'aile
verticale 77,94 % (verdict) ; en P1 (y_c = 7,5) : H = 37,243, S3 83,96 % ; en Q1 (y_c = 7,5, u_V = 0) : H = 41,381,
M3 = 2,03800, 80,84 %.

**Garde** (mode historique) : le moteur calcule aussi l'équilibre P1 (y_c = e/2, u_V = e/2) ; si son taux sur l'aile
verticale (S3, S4) dépasse celui de la méthode historique, un avertissement **rouge** le dit avec les deux chiffres
(ancrage bas : Y = 40 mm sur le cas de contrôle, 46,3 % contre 45,0 % ; le seuil est 43,5 mm). Dans chaque mode, le
taux maximal de l'autre mode est rendu en information (`comparaison`).

### 3.5 Configuration B — aile horizontale en haut (toujours l'équilibre)
y_a = haut − Y′ (hauteur de l'ancrage au-dessus du bout libre), contact à c = y_c du bout libre :
H = R·(X′ − u_V)/(y_a − c). Aile horizontale comme en A (face du congé comprimée). Aile verticale entre l'angle et
l'ancrage : **M = R·(X′ − e/2) constant** (statique de la partie au-dessus de la coupe, où seule la charge agit : il
ne dépend pas de l'ancrage), N = −R, V = 0 ; entre l'ancrage et le contact : M = H·(y − c), V = H, N = 0. **Section
nette** au droit du trou (largeur L − n·d0, épaisseur e), la face la plus sollicitée du trou — au-dessus (M = R·(X′ −
e/2), N = −R) ou au-dessous (M = R·(X′ − u_V), V = H). **Pression sur le support** au bout libre C/L (N/mm), en
information et en avertissement. Les formules du classeur donneraient −18,3 % sur M3 : le mode historique est
imposé à « équilibre » en B (avertissement `mode_impose_B` quand il était demandé).

Exemple, le cas de contrôle retourné (Y = 85, dx = 0) : H = 82,47 kN (bras 35 mm), M3 = M4 = 2,886 kN·m, σ = 256,56 +
9,50 = 266,06 N/mm² → S3 112,42 % ; section nette (3 trous Ø18 sur 300) **137,1 %**, qui gouverne ; C/L = 274,9 N/mm ;
Y = 45 → H = 38,48 kN ; c = 10 → 115,45 kN ; ancrage décalé de 10 mm vers le bout libre → 115,45 kN, vers l'angle →
64,14 kN ; dx = 20 → S3 144,54 %.

### 3.6 Ailes inégales, largeur d'appui, ancrages
L'orientation est le paramètre `aile_horizontale` (le catalogue reste la source unique, r1 = congé) : L150×100×12,
150 horizontal, Y = 65 → taux 1 128,3 %, taux 3 131,1 %, traction 58,31 kN ; 100 horizontal, Y = 115 → 128,3 %,
157,4 %, 30,54 kN. Largeur d'appui `a` : L_eff = min(L ; a + 2·Xe) avec Xe = X′ − r − e (diffusion à 45° jusqu'à la
section 1), appliquée à toute l'aile horizontale (W et A) : a = 100 → L_eff = 194, σ₁ × 1,546, taux 1 117,2 % ; a vide →
L. Ancrages : n sur L à l'entraxe L/n, le premier à L/(2n) (z = 50, 150, 250 pour n = 3) ; N et V ELS et ELU par ancrage,
la combinaison qui gouverne N ; R_ELU vide = 1,5·R dit « auto » (avertissement `R_ELU_auto`).

## 4 · D'où vient chaque grandeur (mode historique : la cellule de `P8`)

| Grandeur | Cellule | Dans le moteur |
|---|---|---|
| larg, haut, e, r | `C4`…`C7` (ou `K4`…`K7` en saisie libre) | `section` |
| Xe, Xe+r, Ye, Ye+r, YH | `C19`, `C20`, `C23`, `C24`, `C25` | formules des blocs S1–S4 (nominales) |
| M1, M2, Mmax, ρ₃, M3, ρ₄, M4 | `C35`, `C45`, `C55`, `C56`, `C57`, `C64`, `C65` | `verifications[Sᵢ].M`, formules `M_bras_j`, `Mmax_j`, `rho`, `M_rho` |
| tᵢ, Wᵢ, σᵢ, τᵢ, σ_éq,ᵢ | `C31`/`C32`/`G31`/`G34`/`G37` (S1) … `C61`/`C62`/`G61`/`G64`/`G67` (S4) | `verifications[Sᵢ]` : t, W, sigma, tau, sigma_eq |
| taux σ, taux τ, taux | `I31`, `I34`, `I37` … `I61`, `I64`, `I67` | taux_sigma, taux_tau, taux |
| σ_adm, τ_adm | `H31`, `H34` | `materiau` |
| taux max | `D3` | `taux_max` |
| N_ELS, V_ELS, N_ELU, V_ELU | `C82`, `C83`, `C85`, `C86` | `ancrages` |
| pression diamétrale, arrachement | `C88`, `C90` | `ancrages`, `informations` |
| R, R_ELU, L, X, Xj, Y, Yj, n, d, jeu | `C12`, `C71`, `C8` (cm), `C13`, `C14`, `C16`, `C17`, `C73`, `C74`, `C76` (cm) | l'entrée, en mm |

Le test code les **93 formules de `P8`** (annexe A de l'analyse) dans les unités du classeur et les évalue par un petit
interpréteur indépendant : sur le cas de contrôle et sur 240 saisies au sort, chaque cellule qui a une grandeur dans le
moteur lui est égale à 1e-9 (58 cellules ; les copies `F31` = `C35`… et `L22` = 864,69 sont vérifiées comme telles).

## 5 · Avertissements, informations, hypothèses

Toujours (niveau info) : « Chevilles non vérifiées : efforts N et V par ancrage fournis pour Hilti PROFIS, ELS et
ELU séparés » ; « Pic de contrainte au congé non vérifié (≈ 1,3 × la contrainte de poutre : statique seulement, à
considérer en fatigue) » ; « Effet de levier non couvert si les chevilles sont très raides (> ≈ 250 kN/mm) ». Selon le
cas : `equilibre_plus_severe` (rouge, la garde), `pression_support` (attention, B), `mode_impose_B`, `R_ELU_auto`,
`note_v1` (attention), `fy_epaisseur` (e > 40 mm : le Tableau 3.1 réduit f_y, la valeur saisie est gardée). Les lignes
d'information (pression diamétrale, arrachement, compression du mur, pression sur le support, taux de l'autre mode,
L_eff) n'entrent pas dans le verdict. Les `hypotheses` disent la méthode aux contraintes admissibles (« efforts ELS non
majorés, coefficient sur f_y : ce n'est pas une justification EC3 aux états limites »), la configuration et le mode,
le pivot et u_V, les jeux et le nombre de combinaisons, L_eff, R_ELU.

## 6 · Ce qui n'est pas couvert

- la **cornière longue** à ancrages multiples (modèle longitudinal, analyse § 5) : tranche courte seulement, R/n et
  traction partagée ; les ancrages sont déjà positionnés sur L ;
- les **chevilles** (cône de béton, bords, support) : efforts fournis, rien de vérifié ;
- le **pic de contrainte** au congé (contrainte de pointe ≈ 1,3 × la poutre, fatigue) ; le **support souple** (pivot
  qui remonte) et l'**effet de levier** (chevilles très raides) : avertis, pas calculés ;
- les cornières **PRS** (deux plats soudés) ; le maillage de contrôle ;
- f_y au-delà de 40 mm d'épaisseur (averti, non réduit) ; la nuance S420 (absente des sources du site).

## 7 · La preuve (`tests/test_corniere_moteur.py`, 106 contrôles, ~15 s)

1 cas de contrôle à 1e-9 face aux 93 formules de `P8` codées dans le test, et aux chiffres de l'analyse à leur dernière
décimale (M1…M4, 88,8621 %, σ, τ, σ_éq, ancrages, pression, arrachement, ρ₃, ρ₄, H) ; cellules nommées. 2 P8 sur 110
saisies sans jeu (une combinaison : P8 exactement) et 130 avec jeux (l'enveloppe = P8 à Xj = +dx, Yj = +dy pour les
sections, Xj = +dx pour les ancrages). 3 l'enveloppe : 1/3/9 combinaisons, positions nommées, le tableau des jeux,
dy < 0 refusé, balayage (aucune grandeur ne baisse quand dx ou dy croît : A historique, A équilibre avec y_c = 0 et
y_c = 5 / u_V = 0, B), combinaisons gouvernantes justes, diagrammes face à une statique recalculée dans le test (trois
courbes, enveloppe = max point par point, σ_éq des neuf combinaisons, t(s)). 4 équilibre P0, P1, Q1, comparaison,
garde (Y = 40 rouge, Y = 45 et 85 rien). 5 B : 82,47 kN, 2,886 kN·m, 112,42 %, nette 137,1 % qui gouverne, moment
indépendant de l'ancrage, Y = 45, c = 10, u_V = 0, jeux, mode imposé, pression sur le support, arrachement vers le bout
libre. 6 ailes inégales. 7 maxima (S1 +2,11 % à 1,98 mm ; S3 +1,45 % à 1,66 mm ; information en historique, verdict en
équilibre ; jamais sous une grille fine indépendante). 8 L_eff. 9 les 11 428 formules de 131 cornières recalculées à
partir de leurs seuls termes (1e-9), blocs et combinaisons, taux/ok/état/gouvernant. 10 domaine : 40 refus avec leur message, 18 bornes
admises, -0, idempotence, JSON strict, entrée intacte, défauts du § 2, nuances (source unique), masse en saisie libre.
11 relecture v1 : les six photos du gel d'avant la refonte et une saisie libre. 12 sélecteur. 13 pureté (sources ;
sous-processus sans streamlit, pandas, reportlab, matplotlib, scipy, PIL, fitz, sous 3.12 et 3.13, résultats identiques).
14 **28 mutations** rechargées depuis les sources (ρ₃ sans Yj, pivot, N en B, nette sans trous, L_eff, dx oublié,
M_angle, u_V, M de l'équilibre sans le terme R·(e/2 − u_V), t(s), σ_éq, τ, bras, ELU, enveloppe inversée, maximum
grossier, garde muette, gouvernant, verdict, refus, défaut, tri, optimal, v1, diagrammes, positions, nuance, contact)
toutes rejetées, 1 à 9 reproches chacune ; témoin (un commentaire) non rejeté.

## 8 · Écarts et conventions notés pendant le lot (à lire avec les décisions)

- **B, moment entre l'ancrage et l'angle** : les décisions écrivent R·(X′ − u_V) ; la statique de la partie au-dessus de
  la coupe (« certain, sans hypothèse de pivot ») donne **R·(X′ − e/2)** autour de l'axe de l'aile, identique avec
  u_V = e/2 (le défaut, le seul cas chiffré par l'analyse : 2,886 kN·m). Le moteur rend R·(X′ − e/2) ; le saut
  R·(e/2 − u_V) à l'ancrage est le couple du cisaillement d'ancrage excentré, et la section nette regarde les deux faces.
- **Maxima du congé en historique** : calculés (vue Contraintes, +1,45 % sur S3) mais hors verdict, pour que la méthode
  historique rende le classeur exactement (88,8621 %, § 1 et § 3.2 des décisions) ; en équilibre ils entrent dans le
  verdict (§ 3.3).
- **Arrachement** (`C90`) : Y nominal en historique (P8 mot pour mot), l'ancrage le plus près du bord (Y + dy) en
  équilibre.
- **Nuances** : les décisions citent 235, 275, 355, 420, 460 ; les sources du site (`acier/flambement/normes.py`,
  Tableau 3.1 ; `acier/donnees/bibliotheques.json`) n'ont pas S420 ; le moteur lit la première (f_y pour t ≤ 40 mm) et
  admet 235, 275, 355, 460 — ajouter S420 se fait là, pas ici.
- **Relecture v1** : « charges et ancrage aux défauts » poserait la charge hors des petites cornières (X = 75 sur une
  L 45) et ferait refuser la note ; les défauts sont donc posés **dans les ailes** de la cornière relue, au rapport du
  cas de contrôle (X, dx sur larg/120 ; Y, dy sur haut/120 ; d ramené au trou qui tient), l'avis disant qu'ils sont à
  ressaisir. La L 120 redonne les défauts exacts. Le congé d'une saisie libre v1, inconnu du modèle lame, vaut 0.
- **Masse en saisie libre** : calculée sans les arrondis de bout d'aile (+0,5 % sur la L 120×120×15), dite « calculée ».

## 9 · L'ancien modèle lame (07/11 – 10/11)

Le premier moteur `acier/corniere` (07/11) reproduisait au bit près l'écran Streamlit `modules/corniere.py` : une bande
d'acier de 1 m, M = V·e sur W = b·t²/6, sans congé, sans aile verticale, sans ancrage, et une vérification
« longitudinale » qui répétait la transversale (défaut non conservatif, `docs/site/DEFAUTS_STREAMLIT.md`). L'analyse du
classeur l'a mesuré : 120,65 % sur le cas de contrôle avec σ_lim = f_y/1,5 saisie, 184,22 % avec ses défauts. Il est
**retiré** avec sa preuve de parité, remplacé par la méthode du classeur ; **l'écran Streamlit n'est plus reproduit par
le site** (il reste tel quel dans l'application du bureau). Les notes enregistrées avec lui (version 1 : clés V, e, s,
gamma_ELU, bord, sig_lim_*) se rouvrent : le nom et la cornière sont gardés, le reste est à ressaisir (§ 8).

## 10 · Les vues (lot « schémas », 11/11/2026) — `acier/corniere/schemas.py`

Quatre vues (décisions § 4 ; modèle visuel : figures a, b, c de l'analyse), chacune une fonction
`(res, fs=None, interactif=True) -> Dessin` du kit de dessin des assemblages, qui ne lit que la SORTIE de `verifier`
(rien n'est recalculé : un bras de levier est une cote mesurée sur le dessin, dont la valeur est sa longueur) :

| Vue | Ce qu'elle montre | Échelle |
|---|---|---|
| `coupe` | le schéma statique coté : mur hachuré (rompu), cornière (congé r₁, bouts r₂ du catalogue), charge R à X en plein et X ± dx en tirets, ancrage (trou, tige, rondelle, écrou, axe) à Y et Y ± dy en tirets fins, résultante C au pivot (A : Y′ − bras, soit y_c en équilibre, jeu_ancrage en historique) ou au contact du bout libre (B : haut − c), ligne u_V (équilibre), sections 1 à 4 en orange avec pastilles, cotes larg, haut, e, r, X, Y et les bras DU MODE (historique : Xe, Xe+r, Ye, Ye+r, YH ; équilibre A : X − uV, Y − yc ; B : X − uV, ya − c), valeurs R, H, C ; A (aile horizontale en bas, y ≤ 0) et B (en haut, y ≥ 0 : l'aile verticale descend) | mm |
| `moments` | les deux ailes développées l'une au-dessus de l'autre (abscisse s depuis le dos de l'autre aile) : une courbe par position de la charge (extrêmes fines `cvf`, nominale `cvm`), l'enveloppe en gras (`env`), la bande ± dy (`tol`), N et V par tronçon, M1 à M4 écrits (valeur de l'enveloppe, négative en B), congé et bloc d'angle grisés, la charge et l'ancrage (et le contact en B) repérés | hors échelle (`echelle: False`) |
| `contraintes` | la bande t(s) (`pp`), σéq de l'enveloppe (`sig`), la ligne σadm (`lim`) et l'échelle des taux (50 %, 100 %), le maximum ◉ et sa position (« 1,98 mm avant le pied du congé »), les taux S1 à S4 | hors échelle |
| `elevation` | la tranche L vue de face (les n ancrages à L/(2n) + i·L/n, N et V ELS par ancrage, la chaîne des entraxes, Y) et l'aile horizontale RABATTUE (vue de dessus, le bout libre contre l'élévation) : la charge de largeur a à X, la position X′ qui gouverne L_eff en tirets, le trapèze de diffusion à 45° jusqu'à la section 1 dont la base mesure L_eff (coté), L, larg, X | mm |

**Mise en page en tailles de texte.** `fs` est la taille du texte en unités de la scène ; `schemas.ajuster(construire,
toile, texte_px, fs0)` reconstruit une vue jusqu'à ce que son texte se lise à `texte_px` dans une toile donnée (l'écran :
`dessin_corniere.TOILE` = 626 × 682 px, `TEXTE_PX` = 13 ; la note choisira la sienne). Les diagrammes sont dessinés
tout en tailles de texte (≈ 48 fs de large) : leur texte vaut ≈ 13 px dans la toile quel que soit `fs`.

**Contrat de sélection.** Chaque pièce qu'une donnée pilote est un groupe `<g class="pc" data-sel="<clé>">` sous la
clé de sa ligne dans les données (`schemas.CLES_SEL`) : coupe → X, dx, Y, dy, R, d, profil, jeu_ancrage (historique)
ou yc et uV (équilibre) ; élévation → L, a, n, d, X, Y, profil. `interactif=False` n'en pose aucun (la note).

**Styles ajoutés au kit** (`doubles_cornieres/schemas.py`, additifs) : `sc`/`scb`/`scg` (sections et pastilles orange),
`env`/`cvm`/`cvf` (courbes), `sig`, `lim`, `fzd` (charge déplacée en tirets), `tol`, `cong`, `fc`/`fch`/`fcp`/`fcl`
(résultante de compression, vert `PALETTE["comp"]`) ; et l'attribut `sel` d'un groupe → `data-sel`.

**Pour la note.** `schemas.svg(dessin, identifiant)` donne le SVG ; `ndc_pdf.svg.peindre_svg` le peint, mais il
IGNORE `fill-opacity` : la zone du congé (`cong`) et la bande de tolérance (`tol`) y sortent en aplats opaques
(orange, gris) — le peintre de scène du kit (`doubles_cornieres.rapport.peindre(d, dessin, x, y, w, h)`) honore
l'opacité et rend les mêmes vues : à préférer pour les diagrammes, ou ajouter `fill-opacity` au peintre SVG. Les
légendes (`schemas.legende(res, cle)`) sont la source unique du texte sous chaque vue (écran et note).

**Dessin impossible.** `elevation` lève `ValueError` quand la hauteur de la cornière vaut moins de 1,6 ligne de texte
(une tranche de plusieurs mètres face à une petite aile) ; `dessin_corniere.vues` rend alors le cadre
`schemas.secours` « Dessin impossible à cette échelle » pour cette vue seule.

## 11 · Le module du site (lot « écran », 11/11/2026) — `api_web/corniere.py`

Le module est RÉÉCRIT sur l'établi compact, le modèle de la poutre (`api_web/poutre.py`, `api_web/etabli_compact.py` ;
`construire.ETABLI_COMPACT` et `ARBRE_COMPACT` le portent, décisions § 4). Il ne calcule rien : toute saisie passe par
`moteur.normaliser` (hors domaine, elle est refusée avec le message du moteur et la cornière reste intacte), tout ce qui
s'affiche vient de `moteur.verifier`.

**L'état** : `{"version": 2, "cornieres": [...], "actif": 0}` ; `nouvel_etat()` donne une cornière « C1 » aux défauts du § 2
(les noms neufs sont courts, « C1 », « C2 »…, comme « V1 » de la poutre : « Cornière 1 » ne tenait pas entier dans la ligne de
l'élément à la tablette portrait) ;
`normaliser` relit une note (une liste abîmée est refusée, une note vide est une note neuve) ; `gerer_elements` (ouvrir,
ajouter, copier — la copie s'insère après —, supprimer, jamais la dernière) ; `choisir_section` (le sélecteur) ;
`telecharger(etat, "note_pdf", infos)` appelle `api_web/note_corniere.fichier` par un import paresseux — tant que ce
fichier (lot « note ») n'est pas là, la demande est refusée avec un message clair, jamais une exception.

**La colonne de gauche** : la ligne de la cornière ouverte (nom, liste « nom · désignation · configuration », copier,
supprimer) puis quatre rubriques — **1 Cornière** (catalogue ou saisie libre : passer en saisie libre part des
dimensions de la cornière affichée ; la désignation tapée est relue au catalogue, « 100 x 150 x 12 » comme
« L 150×100×12 » ; la fiche du catalogue : congés r₁/r₂, masse, aire ; l'aile horizontale en segments avec les
longueurs réelles « A = 150 mm · B = 100 mm » ; la configuration A/B en segments avec deux pictogrammes ; la nuance en
menu, le coefficient), **2 Charge** (R, L, X ± dx, a « auto · L », R_ELU « auto · 1,5·R »), **3 Ancrage** (Y ± dy, n,
Ø d), **4 Hypothèses** (secondaire : le mode — verrouillé en B avec la raison écrite —, le jeu C76, y_c, u_V « auto ·
e/2 »). Données condensées à deux champs par ligne ; chaque champ est une PIÈCE (`data-cible`, la clé de sa pièce sur
le dessin) ; les ⓘ disent le sens des jeux (trois positions, l'enveloppe), de Y (depuis la face extérieure de l'aile
horizontale), de R_ELU (à remplacer par 1,35·G + 1,5·Q).

**Le centre** : les quatre vues du § 10 (`dessin_corniere.vues`), chacune sous son en-tête technique (configuration et
mode ; R, L, X ± dx, Y ± dy, n Ø d ; la bande R, M_max, H ; la légende de `schemas.legende`). Le CONTRAT DE
SÉLECTION : chaque `data-sel` des dessins (X, dx, Y, dy, R, L, a, n, d, profil ; jeu_ancrage en historique, yc et uV en
équilibre) a sa ligne dans les données sous la même clé — un appui sur une pièce ouvre sa rubrique et éclaire sa ligne
(js/compact.js), un clic dans un champ éclaire la pièce.

**La colonne de droite** : le VERDICT (η, section gouvernante et son aile, σ_éq face à σ_adm, la combinaison de jeux
qui gouverne, configuration et mode) ; trois blocs THÉMATIQUES (`etabli_compact.bloc(theme=True)`, toujours visibles,
repliables) — **Sections** (la ligne « Cornière », qui ouvre le sélecteur ; les efforts M_max et H avec leur
combinaison ; L_eff si a est donnée ; puis S1 à S4, les maxima du congé, la section nette en B : deux lignes chacune,
colorées par `commun.niveau`, rouges dès que ça ne tient pas, chaque ligne avec la combinaison qui la gouverne ; en
historique, les maxima du congé sont « hors verdict », leur taux dans la phrase), **Ancrages** (N et V par ancrage, ELS
et ELU, la combinaison de la traction, les positions z, le bras ; pression diamétrale, arrachement, compression du mur
en A ou pression sur le support en B : lignes d'information grises, « hors verdict », comme le pied de poteau),
**Comparaison des modes** (A seulement : le taux maximal de l'autre mode en information ; en historique la garde P1,
rouge quand elle dépasse) ; les avertissements du moteur, titrés, le grave en tête ; les hypothèses du moteur ; la masse.
Cinq FENÊTRES DE CALCUL (`s0` sections, `s1` ancrages, `s2` comparaison, `geometrie`, `efforts`) : chaque ligne
« symbole = expression = valeurs = résultat » s'écrit à partir des seuls termes de la formule du moteur (`_fx`, la table
des clés de la docstring du moteur), le résultat est le sien ; en historique la cellule du classeur suit le symbole
(« M3 (C57) », « taux (I37) »). Le test réévalue CHAQUE ligne affichée (1 768 lignes sur seize états) : les valeurs
écrites, recalculées, retombent sur le résultat affiché à sa précision.

**Le sélecteur** (ligne « Cornière ») : `moteur.candidats` — les 233 cornières par masse, chacune calculée avec les
charges de la cornière ouverte ; ✓ la retenue, ★ la plus légère qui tient ET dont la charge et l'ancrage sont dans les
ailes ; une cornière hors domaine porte une pastille grise « exclue » à la place du taux, sa raison au survol, et
sa ligne est inactive (`aria-disabled` : le parcours ne la choisit pas ; cliquée, le refus du moteur se lit) ; une qui
ne tient pas est rouge. Mesuré : 0,6 à 0,8 s en CPython (233 calculs sans diagrammes).

**La note v1** (§ 9) : `entree.est_v1` la reconnaît, le moteur la relit (nom et cornière gardés, défauts dans ses
ailes), l'écran pose l'AVIS d'ouverture (`CLE_AVIS`, le mécanisme des anciennes dalles : dit une fois, en tête du
résultat et dans « avis » pour le téléphone, puis retiré de l'état) ; l'avertissement `note_v1` du moteur n'est pas redit
à cet instant, il reste ensuite tant que la charge ou l'ancrage ne sont pas ressaisis (R, L, X, dx, a, R_ELU, Y, dy, n,
d : la clé `v1` retombe) — renommer ne vaut pas ressaisir.

**Chiffres à l'écran** (la cornière neuve) : η = 95,8 % en section 3 (« 96 % », vert foncé), « charge à X + 20,
ancrage à Y + 10 » ; équilibre 89,2 % (maximum du congé de l'aile horizontale) ; B 144,3 % en section nette, non
conforme, H = 82,5 → le bloc Ancrages dit la pression sur le support ; le cas de contrôle du classeur (R = 42,76, sans
jeu, R_ELU = 68,9) rend 88,86 % (« 88,9 % »), une seule combinaison, N_ELS = 14,25 kN.

**Preuves** : `tests/test_api_web_corniere.py` (133 contrôles : seize états face au moteur, 23 refus, le sélecteur, la
note v1, le contrat de sélection, les 1 768 lignes de formules réévaluées, les éléments, les vues, la note PDF refusée
tant que le lot « note » n'est pas là, onze MUTATIONS vues rouges — le drapeau v1 qui ne retombe plus, 3·τ au lieu de
3·τ², une couleur figée, la combinaison tue, l'avis non posé ou non retiré, le gris du sélecteur, le grave plus en tête,
une pièce sans cible, un bloc non thématique, une fenêtre non annoncée — et un témoin) ; `tests/test_api_web_modules.py
corniere` (le contrat commun, 22 OK) ; `tests/test_reference_refonte.py` (treize cas regelés, § 7 des décisions) ;
`site/web/verifier_site.js`, filtre « corniere » (joué aussi par « module:corniere ») : A puis B à l'écran (la coupe se
retourne : la charge passe au-dessus de l'ancrage, le contact au bas de la cornière, mesuré dans le SVG), dx = 0 qui fait
baisser le taux, l'appui sur l'ancrage du dessin qui ouvre la rubrique, le sélecteur et le choix de l'optimale,
l'équilibre (le pivot dessiné), une inégale tapée dans l'autre ordre, X = 10 refusé (toast, valeur rendue), la note v1
rouverte avec son avis, puis A et B au PC 1440, à la tablette 1024 et à l'iPhone 13 (rien ne déborde, 44 px au doigt,
16 px dans les champs). Les captures `site_corniere_*.png`.

**Ce que l'intégration fait ensuite** : `construire.NOTE_PDF` reçoit « corniere » quand `api_web/note_corniere.py` est
là, avec sa note riche (`NOTES_RELEVE["corniere"]` : deux cornières, A historique et B) ; le gel `corniere.json` est
regelé par ce lot (commit distinct).
## 12 · La note PDF (lot « note », 11/11/2026) — `acier/corniere/note.py`

**Ce qu'elle est.** Une note = la page de garde du site (A4 portrait — projet, partie, date, indice, sommaire avec le
statut et la page de chaque cornière : `ndc_pdf.styles.Style.garde_lignes`, la garde des notes Poutre, Dalle et
Assemblages), puis, PAR CORNIÈRE, deux pages A4 paysage sur le kit `ndc_pdf` (palette Encre, polices et marges des
notes du garde-corps et des soudures) :

| Page | Contenu |
|---|---|
| 1 · CALCUL | bandeau (statut, η max, section gouvernante et sa combinaison de jeux, configuration et mode) ; colonne 1 : HYPOTHÈSES en lignes « libellé · valeur » (cornière, configuration, acier, méthode aux contraintes admissibles avec le coefficient et σ_adm, τ_adm, le mode — pivot, frottement nul, jeu C76 —, les jeux, la tranche, L_eff), CHARGE ET ANCRAGE (R ELS, R_ELU auto ou saisie, X ± dx, a, n × Ø à Y ± dy, entraxes), la COUPE cotée ; colonnes 2 et 3 : **1 · géométrie des sections** (W et A par (b, t), largeur efficace, d0 et b_net en B), **2 · efforts et bras de levier** (la chaîne de chaque section S1 à S4 — et nette en B — avec la combinaison qui la gouverne et, en historique, les cellules du classeur), **3 · vérification des sections** (tableau s/t, M, σ, τ, σ_éq, η de S1 à S4, des maxima du congé — « (info) » en historique — et de la section nette ; puis les formules de la section gouvernante), **4 · ancrages** (N et V par ancrage ELS puis ELU, pression diamétrale, arrachement, pression sur le support ; les informations : compression du mur, taux de l'autre mode, L_eff), RÉCAPITULATIF (barres, combinaisons), CONCLUSION (taux, conformité, efforts à donner au calcul des chevilles) |
| 2 · DÉTAILS ET MÉTHODE | les trois vues des schémas peintes par le peintre de scène du kit (MOMENTS : une courbe par position de la charge, l'enveloppe en gras, la bande ± dy ; CONTRAINTES : t(s), σ_éq, le maximum ◉ ; ÉLÉVATION : la tranche, les ancrages, la diffusion de la charge) ; les textes de MÉTHODE (`methode.py` : modèle et contraintes admissibles ; mode historique, équilibre ou configuration B ; ce qui change quand on retourne la cornière ; jeux et enveloppe ; largeur efficace) ; LIMITES ET AVERTISSEMENTS = les avertissements du moteur tels quels (rouge, attention, information) ; RÉFÉRENCES ; cartouche (cornière, section, configuration · mode, acier, date · indice, échelle de la coupe) |
| 3 · SUITE (si besoin) | la note est composée en taille normale, puis serrée (0,92) ; si une colonne déborde encore, ses derniers blocs passent un à un sur une page de suite (le mécanisme de la note du garde-corps : `Suite`, `mise_en_page`, `BLOCS`), la colonne d'origine dit « Suite p. 3 : … », le sommaire compte la page. Sur les onze états d'essai aucune cornière n'en a besoin (trois notes serrées) ; une note qui ne tiendrait toujours pas est refusée avec un message, jamais tronquée |

Les dessins sont ceux de l'écran (`schemas.coupe`, `moments`, `contraintes`, `elevation`, `interactif=False`), peints
par `doubles_cornieres.rapport.peindre` (qui honore l'opacité du congé et de la bande de tolérance) ; les vues à
l'échelle sont reconstruites par `schemas.ajuster` pour un texte de 5,6 pt dans leur boîte ; une vue impossible
(proportions extrêmes) laisse le cadre « Dessin impossible à cette échelle », le calcul est rendu. Les légendes sont
`schemas.legende` (source unique écran / note).

**Les formules** (`acier/corniere/formules.py`). Le moteur nomme chaque formule (`calcul` : clé, symbole, termes,
résultat, cellule) ; `formules.GABARITS` ajoute le seul savoir qui manque pour l'écrire — l'expression Python des termes,
dans l'ordre du moteur — et en fait des substitutions `Sub` du kit des assemblages : « M3 = ρ₃·Mmax = 0,7657 × 3,325
= 2,546 kN·m ». Chaque ligne est ré-évaluée deux fois par le test : avec les nombres imprimés (`verifier`, l'arrondi
d'affichage) et avec les termes exacts (`exact`, 1e-9) — 1 017 formules sur 12 cornières, aucun écart ; chaque clé du
moteur a son gabarit et chaque gabarit sert. La note renomme « taux » en η et le moment d'une section repère par son
numéro (M3) ; η_σ et η_τ (I31, I34…) restent des lignes du détail, non imprimées. Les cellules du classeur sont
imprimées avec les étiquettes (« S3 — charge à X + 20, ancrage à Y + 10 · C55 C25 C23 C56 C57 »). Les nombres des
messages du moteur passent par `note.francais` (virgule française) : le moteur écrit certains nombres de ses
avertissements et hypothèses avec le point (« 52.50 kN », « 236.7 N/mm² ») — à corriger dans le moteur, la note ne
fait que formater.

**La plomberie du site** (`api_web/note_corniere.py`). `fichier(etat, infos) -> {"nom", "type", "base64"}`, le contrat
de `note_poutre.fichier` : `api_web/corniere.telecharger(etat, "note_pdf", infos)` l'appelle par import paresseux ;
`infos_doc` traduit la note du site (nom, chantier, indice, date) comme la poutre (projet = chantier ou nom ; partie =
nom ou « Cornières d'appui » ; indice ou « A » ; date ou le jour) ; le fichier s'appelle « Note de calcul - <nom>.pdf ».
`FICHIERS` déclare ce que la note charge en plus de l'écran et du paquet de la note de la poutre : `acier/corniere/
note.py`, `formules.py`, `methode.py` et `acier/assemblages/poutre_poutre/doubles_cornieres/rapport.py` (le paquet de
la note, `construire.paquet_note`, les reçoit dès que « corniere » entre dans `NOTE_PDF` avec sa note riche dans
`NOTES_RELEVE`). Dans le navigateur (sonde du test : fontTools, accélérateurs C, streamlit, matplotlib, pandas, scipy,
pymupdf refusés ; Pillow factice ; les seules polices du paquet), la note se fait et son texte est celui de CPython ;
numpy est déjà chargé par le catalogue des cornières. Génération : 0,25 à 0,85 s par note en CPython (une à deux
cornières), ≈ 150 ko par cornière.

**La preuve** (`tests/test_note_corniere.py`, 59 contrôles, ~20 s) : onze états (A historique, A équilibre, B avec
Y = 85 et Y = 45, ailes inégales 150 et 100 horizontal, L 40×40×4, L 250×250×35 en équilibre à quatre ancrages, deux
cornières dans une note, la garde avec R_ELU saisie, une note v1 relue) ; la parité avec `verifier` par pdftotext (nom,
statut, η max, gouvernante et combinaison, chaque taux, M1 à M4, H, N et V ELS et ELU, le résultat de chaque formule
écrite, les sections de méthode, les avertissements, les textes des dessins, le sommaire de la garde) ; l'identification
piégée (accents, guillemets, « & », indice « C ») dans la garde et l'en-tête de chaque page, jamais « indice 0 » ni
« indice A » ; la page de suite forcée (1 + 3 + 3 pages, sommaire juste) ; les refus (hors domaine avec le nom et le
message du moteur, note vide) ; la sonde du navigateur ; cinq mutations qui rougissent (une cornière oubliée, la
tolérance dy non transmise, le mode mal nommé, un gabarit faux, l'indice perdu). `tests/test_note_identification.py`
range déjà « corniere » avec la poutre, la dalle et les assemblages (garde + « date · indice » sur chaque page) :
l'intégration n'a plus qu'à compter sept modules.
