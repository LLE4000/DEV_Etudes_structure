# Cornières d'appui — décisions de conception du module refondu (10/11/2026)

*Écrit par le coordinateur avant de lancer la refonte, à la demande de l'ingénieur du 10/11/2026 : « module des deux
cornières, organisé comme la poutre (données à gauche, schémas au centre, résultats à droite), avec sa note de calcul ;
mon classeur Excel ne traite que la cornière vers le bas, il faut réfléchir à la cornière vers le haut ».
Base : l'analyse critique du classeur, `docs/acier/corniere_excel/ANALYSE_CORNIERE_EXCEL.md` (07/11/2026), dont les
chiffres font foi (chaque nombre y sort d'un script). Ce document FIXE les choix ; `CORNIERE_MOTEUR.md` (réécrit par le
lot) décrit ensuite ce qui est livré. Les questions 8.6 de l'analyse restent ouvertes : là où il a fallu trancher sans
l'ingénieur, le choix est marqué **[à confirmer]** et repris dans `docs/site/SUITE_DU_CHANTIER.md`.*

## 1. Ce que le module devient

- **La référence n'est plus l'écran Streamlit `modules/corniere.py`** (modèle « lame », vérification longitudinale qui
  répète la transversale : défaut non conservatif signalé dans `DEFAUTS_STREAMLIT.md`) mais **le classeur de l'ingénieur**
  (feuille `P8`, cas de contrôle L120×120×15, S355, L = 300 mm, R = 42,76 kN, X = 75, Y = 85, Xj = Yj = 0,
  coefficient 1,50 → M1 = 2,00972 ; M2 = 2,56560 ; M3 = 2,35870 ; M4 = 2,89665 kN·m ; taux 88,8621 % en section 3).
  L'ancien moteur lame (`acier/corniere/entree.py`, `moteur.py` du 07/11) et sa preuve de parité
  (`tests/test_corniere_moteur.py`) sont RETIRÉS : ils ne décrivent plus le module. L'écran Streamlit n'est pas modifié
  (il reste tel quel dans l'application du bureau ; sa ligne de `DEFAUTS_STREAMLIT.md` est complétée : « le site ne le
  reproduit plus »).
- **Deux configurations**, chacune avec son propre équilibre (analyse § 6) :
  - **A — aile horizontale EN BAS** (celle du classeur) : la charge pose sur l'aile horizontale, l'aile verticale monte
    le long du mur, l'ancrage est au-dessus de l'angle. Le talon (arête inférieure) s'appuie sur le mur, l'ancrage est
    tendu, l'aile verticale est TENDUE (N = +R), son moment est nul à l'ancrage et croît vers l'angle.
  - **B — aile horizontale EN HAUT** : la charge pose sur l'aile horizontale, l'aile verticale DESCEND le long du mur,
    l'ancrage est au-dessous de l'angle. La cornière tourne autour du BOUT LIBRE de l'aile verticale, qui s'appuie sur
    le mur ; l'ancrage est tendu ; l'aile verticale est COMPRIMÉE (N = −R) ; entre l'ancrage et l'angle, le moment est
    CONSTANT et MAXIMAL, M = R·(X' − u_V), quelle que soit la position de l'ancrage (statique de la partie au-dessus de
    la coupe : certain) ; le trou d'ancrage est dans ce moment maximal (section nette à vérifier) ; la compression du
    mur au bout libre est une pression locale à signaler. **Les formules du classeur ne s'appliquent pas à B**
    (elles donneraient −18,3 % sur M3) : B se calcule TOUJOURS en mode « équilibre ».
- **Deux modes de calcul** pour la configuration A :
  - **« Méthode historique »** (par défaut en A) : les formules de `P8` mot pour mot, chaque grandeur portant le nom de
    sa cellule dans la fenêtre de calcul (`C19` Xe, `C25` YH, `C56` ρ₃, `C82` N_ELS…). C'est la méthode validée de
    l'ingénieur ; le test la reproduit à l'identique.
  - **« Équilibre »** : un seul corps libre, pivot y_c et ligne d'action du cisaillement u_V nommés et dessinés ;
    efforts N et V exacts dans chaque aile ; maximum cherché le long du congé. Le même mode sert à B.
  - Dans chaque mode, le résultat montre EN INFORMATION le taux maximal de l'autre mode (A seulement), pour que
    l'ingénieur voie l'écart (sur le cas de contrôle : historique 88,86 %, équilibre P0 76,91 %).
  - **Garde** (mode historique) : le moteur calcule aussi l'équilibre P1 (pivot e/2, u_V = e/2) ; si l'équilibre donne
    un taux PLUS GRAND que la méthode historique sur l'aile verticale (ancrage bas : Y′ sous ≈ 33 à 43 mm pour le cas
    de contrôle, analyse § 1.3 et D.3), un avertissement rouge le dit avec les deux chiffres. La méthode historique
    n'est alors plus du côté sûr.
- **Ailes inégales** : l'orientation est un PARAMÈTRE (`aile_horizontale` : laquelle des deux ailes porte la charge),
  jamais une ligne en double du catalogue. Le catalogue du site (`acier/profiles`, familles L et LI, r1 = congé,
  r2 = bout d'aile) reste la source unique ; la saisie libre demande A, B, t et r.
- **Tranche courte, comme le classeur** : longueur reprise L, n ancrages sur L qui se partagent R/n et la traction. La
  cornière LONGUE à ancrages multiples (modèle longitudinal, analyse § 5 ; « deuxième grand développement » de
  l'ingénieur) n'est PAS dans ce lot : l'architecture le prévoit (charges et ancrages déjà nommés et positionnés sur
  L dans l'élévation), rien n'est inventé. Écrit comme suite dans SUITE_DU_CHANTIER.

## 2. Les grandeurs d'entrée (toutes en mm, kN, N/mm² — plus de cm : les pièges d'unités du classeur disparaissent)

| Rubrique | Champ | Défaut (cornière neuve) | Sens |
|---|---|---|---|
| 1 Cornière | `standard` / `profil` ou `A`, `B`, `t`, `r` | catalogue, **L 120x120x15** | la section ; `r` = congé r1 (catalogue) ou saisi |
| | `aile_horizontale` | `"A"` | laquelle des ailes porte la charge (A = la première du nom, h au catalogue) |
| | `configuration` | `"A"` | A = aile horizontale en bas (classeur), B = en haut |
| | `fy` | 355 | nuance (235, 275, 355, 420, 460 : les nuances du site) |
| | `coef` | 1,50 | coefficient global sur f_y : σ_adm = f_y / coef, τ_adm = σ_adm / √3 (méthode aux contraintes admissibles, efforts ELS non majorés — dit tel quel à l'écran et dans la note : ce n'est pas une justification EC3 aux états limites) |
| 2 Charge | `R` (kN, ELS) | **35** | la réaction posée sur l'aile horizontale, sur la longueur reprise L (le cas de contrôle en met 42,76 : le défaut est plus bas pour qu'une cornière neuve, avec son jeu de 20 mm, soit conforme) |
| | `L` (mm) | 300 | longueur reprise : largeur de toutes les sections |
| | `X` (mm) | 75 | distance du nu du mur (dos de l'aile verticale) à la ligne d'action de R |
| | `dx` (mm) | **20** **[à confirmer]** | jeu de pose de la charge, appliqué vers l'extérieur (défavorable partout) : X′ = X + dx. Le classeur l'appelle Xj (`C14`), à 0 dans la note validée ; l'ingénieur demande 20 mm par défaut (sa demande du 07/11, § 3). Le cas de contrôle du test est joué avec dx = 0 ; l'écran dit, dans l'ⓘ et dans la fenêtre, le taux sans jeu |
| | `a` (mm) | vide = L | largeur d'appui réelle de la charge ; si a + 2·Xe < L, la largeur efficace à la section 1 vaut L_eff = min(L ; a + 2·Xe) (diffusion à 45° jusqu'au pied du congé, analyse § 5.3) et σ de l'aile horizontale est majorée de L/L_eff ; information, du côté sûr, sans effet quand a est vide |
| | `R_ELU` (kN) | vide = 1,5·R **[à confirmer]** | la réaction ELU pour les efforts d'ancrage à donner à Hilti ; vide : 1,5·R affiché « auto · 1,5·R » avec l'ⓘ « à remplacer par la combinaison réelle 1,35·G + 1,5·Q » ; jamais un coefficient caché (le classeur tapait 68,9) |
| 3 Ancrage | `Y` (mm) | 85 | distance de la FACE EXTÉRIEURE de l'aile horizontale (le dessous en A, le dessus en B) à l'axe de l'ancrage : la même cote pour le même trou quand on retourne la cornière |
| | `dy` (mm) | 0 | tolérance de position de l'ancrage, appliquée du côté défavorable de chaque vérification : en A, vers le haut pour les sections (Yj du classeur, `C17`), vers le bas pour la traction d'ancrage ; en B, vers le bout libre pour la traction d'ancrage (les sections n'en dépendent pas) |
| | `n` | 3 | nombre d'ancrages sur L, répartis à l'entraxe L/n, le premier à L/(2n) du bord (convention de dessin et de calcul) |
| | `d` (mm) | 16 | diamètre de l'ancrage ; trou d0 = d + 2 (section nette en B, pression diamétrale) |
| 4 Hypothèses (rubrique secondaire, repliée) | `mode` | `"historique"` | « historique » ou « equilibre » (sans effet en B : toujours équilibre, l'écran le dit) |
| | `jeu_ancrage` (mm) | 10 | mode historique seulement : le `C76` du classeur (1 cm) — bras de la traction d'ancrage Y − jeu_ancrage |
| | `yc` (mm) | **0** **[à confirmer]** | mode équilibre : position de la résultante de compression sur le mur, comptée depuis la face extérieure de l'aile horizontale en A (0 = l'arête, ce que trouve le maillage sur un support raide ; e/2 = 7,5 mm redonnerait l'équilibre P1, proche du classeur) ; en B, comptée depuis le bout libre (0 = le bout libre, confirmé par le maillage) |
| | `uV` (mm) | e/2 (vide = auto) | mode équilibre : ligne d'action du cisaillement d'ancrage, depuis le nu du mur (e/2 = l'axe de l'aile verticale) |

Relecture d'une note : tout passe par `normaliser` ; hors domaine → refus avec message (jamais une valeur ramenée en
silence). Refus de domaine (analyse § 2.2, § 7 : « aucun contrôle de domaine » dans le classeur) :
- la charge doit tomber sur la partie droite de l'aile horizontale : e + r < X′ ≤ larg (sinon : « la charge tombe dans
  l'angle » / « au-delà de l'aile ») ;
- l'ancrage doit être dans la partie droite de l'aile verticale, trou compris : en A, Y′ − d0/2 ≥ e + r et Y + d0/2 ≤ haut ;
  en B, l'axe à y_a = haut − Y au-dessus du bout libre avec y_a − d0/2 ≥ 0 et Y − d0/2 ≥ e + r ;
- t < A et t < B (une cornière), r ≥ 0, L > 0, n ≥ 1, R ≥ 0, coef ≥ 1, dx ≥ 0, dy ≥ 0, 0 ≤ yc < bras.

## 3. Le calcul

### 3.1 Géométrie commune
e = t ; r = congé ; larg = aile horizontale ; haut = aile verticale ; X′ = X + dx ; Xe = X′ − r − e ; Xe+r = X′ − e ;
W(t) = L·t²/6 ; A(t) = L·t ; épaisseur dans le congé t(s) = e + r − √(r² − (e + r − s)²) pour e ≤ s ≤ e + r (s =
distance au dos de l'autre aile), t = e au-delà ; sous s = e : le bloc d'angle, hors théorie des poutres (grisé).
Sections repères : S1 aile horizontale au pied du congé (u = e + r, t = e) ; S2 aile horizontale au nu intérieur de
l'aile verticale (u = e, t = e + r) ; S3 aile verticale au pied du congé (t = e) ; S4 aile verticale au niveau de la face
intérieure de l'aile horizontale (t = e + r).

### 3.2 Mode historique (A) — `P8` mot pour mot
M1 = R·(Xe + dx)… **non** : attention, dans `P8` le jeu Xj est AJOUTÉ aux bras : M1 = R·(Xe₀ + Xj) avec Xe₀ = X − r − e,
soit R·(X′ − r − e) ; M2 = R·(X′ − e) ; Mmax = R·X′ ; YH = Y − e/2 ; Ye = Y − e − r ; Ye+r = Y − e ;
ρ₃ = (Ye + dy)/(YH + dy) ; ρ₄ = (Ye+r + dy)/(YH + dy) ; M3 = ρ₃·Mmax ; M4 = ρ₄·Mmax ; H_hist = Mmax/(YH + dy) ;
σᵢ = Mᵢ/Wᵢ ; τᵢ = R/(L·tᵢ) (le classeur, même dans l'aile verticale où c'est en réalité un effort normal) ;
σ_éq,ᵢ = √(σᵢ² + 3τᵢ²) ; tauxᵢ = σ_éq,ᵢ/σ_adm ; taux τᵢ/τ_adm aussi (`I34`…) ; taux max = max des quatre.
Ancrages : N_ELS = Mmax/((Y − jeu_ancrage)·n) ; V_ELS = R/n ; N_ELU, V_ELU = × R_ELU/R ; pression diamétrale
V_ELS/(d·e) et arrachement V_ELS/(2·(haut − Y)·e) : en INFORMATION, sans limite (le classeur n'en a pas).
Le cas de contrôle (dx = 0, dy = 0, jeu_ancrage = 10, R_ELU = 68,9) rend M1…M4 et 88,8621 % à 1e-9 près, et
N_ELS = V_ELS = 14,2533 kN, N_ELU = 22,9667 kN. Diagrammes du mode : M(u) = R·(X′ − u) ; M(y) = H_hist·(Y′ − y).

### 3.3 Mode équilibre, configuration A (analyse § 1.2, `synthese/s_commun.corps_libre_A`)
Pour les sections : Y′ = Y + dy ; H = R·(X′ − u_V)/(Y′ − y_c). Aile horizontale (e ≤ u ≤ X′) : M(u) = R·(X′ − u),
V = R, N = 0. Aile verticale (e ≤ y < Y′) : M(y) = H·(Y′ − y) − R·(e/2 − u_V), N = +R (traction), V = H ; au-dessus de
l'ancrage : 0. Contraintes : σ = M/W + N/A (fibre côté congé, N ajouté quand il tend cette fibre), τ = V/A,
σ_éq = √(σ² + 3τ²). Vérifications : S1, S2, S3, S4 (repères) + **maximum le long de t(s) dans chaque aile** (le pas
d'exploration ≤ 0,1 mm dans le congé ; la position et la valeur du maximum sont rendues). Pour les ancrages :
H_anc = R·(X′ − u_V)/((Y − dy) − y_c) ; N_ELS = H_anc/n ; V_ELS = R/n ; ELU × R_ELU/R ; compression du mur C = H_anc
(information, « résultante à y_c »).

### 3.4 Mode équilibre, configuration B (analyse § 6.1, `synthese/s04_retournee.py`)
y compté depuis le bout libre de l'aile verticale ; y_a = haut − Y (axe de l'ancrage) ; contact à c = yc (0 = bout
libre). H = R·(X′ − u_V)/(y_a − c) pour les sections ; H_anc avec y_a − dy. Aile horizontale : comme en A (face du congé
comprimée). Aile verticale : entre l'ancrage et l'angle (y_a ≤ y ≤ haut − e), M = R·(X′ − u_V) constant, N = −R,
V = 0 ; entre le contact et l'ancrage (c ≤ y < y_a), M = H·(y − c), V = H, N = 0. S3 et S4 sont dans la zone de moment
constant (sinon refus : l'ancrage est dans le congé). Contraintes : σ = |M|/W + |N|/A, τ = V/A, σ_éq. **Section nette**
au droit de l'ancrage : W_net = (L − n·d0)·e²/6, A_net = (L − n·d0)·e, M = R·(X′ − u_V), N = R → vérification « nette ».
**Pression sur le support** au bout libre : C = H_anc, soit C/L en N/mm, en information avec un avertissement (le
support n'est pas connu du module). Ancrages : N_ELS = H_anc/n, V_ELS = R/n, ELU × R_ELU/R. Cas de contrôle retourné
(L120×120×15, R = 42,76, X = 75, Y = 85, dx = 0) : H = 82,47 kN, M3 = M4 = 2,886 kN·m, taux S3 = 112,42 % (σ = M/W + R/A
= 256,56 + 9,50 = 266,06 N/mm²), section nette (3 trous Ø18) 137,1 % ; Y = 45 : H = 38,48 kN.

### 3.5 Verdict, avertissements
- `taux_max` = max des vérifications du mode (sections repères, maxima du congé, section nette en B) ; `statut`
  ok/nok ; `gouvernant` (clé, nom, aile, position). Les taux τ/τ_adm sont des lignes du détail, pas du verdict (comme
  le classeur, dont le taux max est MAX(I37, I47, I57, I67)).
- Avertissements, toujours : « Chevilles non vérifiées : efforts N et V par ancrage fournis pour Hilti PROFIS, ELS et
  ELU séparés » ; « Pic de contrainte au congé non vérifié (contrainte de pointe ≈ 1,3 × la contrainte de poutre :
  statique seulement, à considérer en fatigue) » ; « Effet de levier non couvert si les chevilles sont très raides
  (> ≈ 250 kN/mm) ». Configuration B : « Pression du bout libre sur le support à vérifier ». Mode historique : la garde
  du § 1 (équilibre P1 plus sévère). Charge dans le congé de l'aile horizontale ou ancrage dans celui de l'aile verticale :
  REFUS (pas un avertissement).
- Les lignes « information » (pression diamétrale, arrachement, pression sur le support, taux de l'autre mode)
  n'entrent pas dans le verdict et ne portent pas de taux coloré (comme « hors verdict » du pied de poteau).

## 4. L'écran (établi compact, le modèle de la poutre)
- Gauche : la ligne de l'élément (nom, liste, copier, supprimer — `etabli_compact.arbre(…, compact=True)`), puis les
  rubriques 1 Cornière, 2 Charge, 3 Ancrage, 4 Hypothèses (secondaire). Données condensées (`commun.ligne2`), ⓘ.
  La configuration A/B en segments avec deux pictogrammes (l'aile en bas / en haut).
- Centre, les VUES à l'échelle (`acier/corniere/schemas.py`, partagées avec la note ; `api_web/dessin_corniere.py` ne
  fait que les poser) :
  1. **Coupe** — le schéma statique coté (figure a de l'analyse) : mur hachuré, cornière à l'échelle (congé, bouts),
     charge R (et R déplacée de dx en tirets), ancrage (axe, Ø, écrou), résultante de compression (pivot y_c ou bout
     libre), bras de levier du mode (Xe, Xe+r, Ye, Ye+r, YH en historique ; X′ − u_V, Y′ − y_c en équilibre), sections
     1 à 4 (orange, numérotées), cotes larg, haut, e, r, X, Y. A et B. Chaque pièce cliquable (`data-sel`) sous la clé
     de sa donnée (X, dx, Y, dy, R, profil…).
  2. **Diagrammes** — le long des deux ailes développées (figure b) : M (avec N et V en étiquettes ou bandes), l'épaisseur
     t(s), σ_éq et le taux ; sections 1 à 4 repérées avec leurs valeurs ; congés et bloc d'angle grisés ; maximum marqué
     (mode équilibre) ; en mode historique, les droites du classeur. Hors échelle (`echelle: False`).
  3. **Élévation** — la tranche L vue de face : cornière, n ancrages à l'entraxe L/n, charge de largeur a (ou L),
     N et V ELS par ancrage en étiquettes.
  4. **Section** — la cornière seule cotée (A, B, t, r1, r2, axes, masse), avec l'orientation (aile horizontale marquée).
- Droite : verdict (taux max, section gouvernante, mode et configuration), bloc « Sections » (S1 à S4 + maxima +
  nette, deux lignes par vérification, colorées), bloc « Ancrages » (N, V ELS et ELU par ancrage, positions ; pression
  diamétrale, arrachement, pression sur le support en information), bloc « Comparaison » (l'autre mode, information),
  hypothèses, avertissements, masse. Fenêtres de calcul : « symbole = expression = valeurs = résultat », chaque ligne
  réévaluée par le test ; en mode historique, la cellule du classeur entre parenthèses (« M3 = ρ₃ · Mmax (C57) »).
- Sélecteur de cornières (ligne « Cornière » du résultat) : les 233 cornières du catalogue triées par masse, chacune
  calculée avec les charges de la cornière ouverte, ✓ la retenue, ★ la plus légère qui tient ET où la charge et
  l'ancrage sont dans le domaine ; une cornière hors domaine est grise avec sa raison.
- Téléphone : onglets Données · Dessin · Résultat (D6).

## 5. La note PDF (`acier/corniere/note.py`, kit `ndc_pdf`, A4 paysage, comme le garde-corps)
Une note = la page de garde du site (nom, chantier, indice, date — `infos` du site, comme les six autres modules) puis,
par cornière, DEUX pages : page 1 CALCUL — bandeau (statut, η max, section gouvernante, configuration, mode) ;
colonne 1 : identification et hypothèses (méthode aux contraintes admissibles, coef affiché, σ_adm, τ_adm, pivot,
frottement nul, configuration), charges ELS et ELU, la COUPE cotée peinte depuis `schemas` ; colonnes 2 et 3 :
1 géométrie et bras de levier (formules développées), 2 efforts (H, Mmax, M1…M4 ou M(s)), 3 vérification des sections
(σ, τ, σ_éq, taux à droite, le maximum du congé, la section nette en B), 4 ancrages (N et V par ancrage, ELS et ELU, la
position de chaque ancrage, les informations), récapitulatif des taux (barres), conclusion. Page 2 DÉTAILS — les
DIAGRAMMES, l'ÉLÉVATION, la SECTION, les textes de méthode (historique ou équilibre, A ou B — ce qui change quand on
retourne la cornière), les limites (chevilles non vérifiées, effet de levier, pic au congé, support), références,
cartouche. Le nom du fichier : « Note de calcul - <nom de la note>.pdf ». Chaque chiffre de la note est celui de l'écran
(test de parité par pdftotext), chaque formule réévaluée.

## 6. Les notes d'avant (version 1, modèle lame)
Une note du module d'avant (clés V, e, s, gamma_ELU, sig_lim_*…) se rouvre : le nom, la cornière (catalogue ou A, B, t)
sont gardés, le reste prend les défauts ci-dessus, et un AVIS le dit à l'ouverture (« note d'avant le 10/11/2026, modèle
lame : charges et ancrage à ressaisir, les anciens chiffres ne sont plus calculés »). Rien n'est refusé ; rien n'est
converti en silence.

## 7. Le gel et les tests
- `tests/reference_refonte/corniere.json` est REGELÉ (`--geler --accepter="Refonte du module Cornières sur la méthode du
  classeur de l'ingénieur (décision du 07/11, demande du 10/11) : l'ancien modèle lame disparaît"`), dans un commit
  distinct ; 239 contrôles toujours.
- `tests/test_corniere_moteur.py` est RÉÉCRIT : le cas de contrôle (M1…M4, 88,8621 %, ancrages) à 1e-9 ; les 93 formules
  de `P8` (annexe A de l'analyse) recalculées telles qu'elles sont écrites sur ≥ 200 saisies tirées au sort (dx, dy,
  jeu_ancrage, L, X, Y, R, R_ELU, e, r, f_y, coef) face au mode historique ; la configuration B sur le cas de contrôle
  retourné (82,47 kN ; 2,886 kN·m ; 112,42 % ; nette 137,1 %) et sur Y = 45 (38,48 kN) ; l'équilibre P0 et P1 sur le cas
  de contrôle (76,91 %, 83,96 %) ; les effets des jeux (tableau § 2.2 de l'analyse : dx = 20 → 112,43 %, dy = 20 →
  95,38 %, jeu_ancrage = 20 → 16,446 kN) ; la garde (ancrage bas → avertissement) ; les ailes inégales (tableau § 6.2 :
  L150×100×12 r = 12, 150 horizontal → taux 1 128,3 %, taux 3 131,1 %, traction 58,31 kN ; 100 horizontal → 157,4 %,
  30,54 kN) ; le maximum dans le congé (cas de contrôle : S1 +2,11 % à 1,98 mm du pied, S3 +1,45 % à 1,66 mm) ; les
  refus de domaine ; la pureté (bibliothèque standard + numpy du catalogue, sous 3.12 et 3.13) ; des mutations (une par
  formule clé : ρ₃ sans dy, pivot, N en B, section nette sans trous, L_eff, dx oublié…).
- `tests/test_api_web_corniere.py` réécrit (ce que le site affiche = ce que le moteur calcule, les deux configurations,
  les deux modes, le sélecteur, la relecture d'une note v1, les refus) ; `tests/test_note_corniere.py` (parité écran ↔
  PDF, conditions du navigateur, mutations) ; le contrat commun `test_api_web_modules.py` ; `verifier_site.js` :
  `module:corniere`, `formats:corniere`, `note_pdf` (NOTES_RELEVE["corniere"] : une note riche à deux cornières, A et B).

## 8. Ce qui reste ouvert pour l'ingénieur (repris dans SUITE_DU_CHANTIER)
Les dix questions 8.6 de l'analyse ; les trois **[à confirmer]** ci-dessus (dx = 20 mm par défaut ; R_ELU = 1,5·R par
défaut ; y_c = 0 en mode équilibre) ; la cornière longue (étape 4) ; le maillage de contrôle (option, pas dans ce lot) ;
les cornières PRS (architecture seulement).
