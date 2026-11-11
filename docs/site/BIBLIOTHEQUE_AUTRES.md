# Bibliothèque « Autres » — conception (11/11/2026)

Demande de l'ingénieur du 11/11/2026 : « créer la bibliothèque « Autres » de Stab Méthodes. Elle doit réunir trois
outils : 1. Formulaire de RDM (poutres) ; 2. Calcul de plaques ; 3. Calcul des caractéristiques de sections
quelconques […] La justesse et la vérifiabilité des calculs passent avant le nombre de cas proposés […] Commence par
analyser l'existant […] puis propose un périmètre réaliste pour une première version. »

Ce document est écrit AVANT le code : ce qui existe et se réutilise, le périmètre de la première version, les formules
et leurs références, la méthode de calcul des sections, l'écran, les contrôles. La référence d'usage sera tenue dans
`docs/site/BIBLIOTHEQUES.md` (§ 2.6).

## 1 · Ce qui existe et se réutilise

| Existant | Ce qu'il apporte |
|---|---|
| Le cadre des bibliothèques (`api_web/biblio_beton.py` : `Lecture`, briques de la barre, bande, « Pourquoi ? », fiche ; `js/biblio.js` ; `construire.page_bibliotheque` ; `css/bibliotheque.css`) | une page par bibliothèque, des rubriques, les paramètres dans l'adresse (une adresse se partage), le même HTML dans la page et dans le panneau d'une note ; la carte de l'accueil prend déjà la hauteur de deux rangées quelle que soit la famille |
| `acier/donnees/profiles.json` (1 419 profilés, 21 familles : IPE, IPN, HE A/B/M/AA, HL, HD, HP, UPE, UPN, L, CHS, SHS, RHS, laminés et formés à froid) lu par `acier/profiles/catalogue.py` | A, I_y, I_z, W_el, W_pl, masse, dimensions — repris tels quels |
| `acier/profiles/geometrie.py` et `familles.py` | le CONTOUR exact de chaque profilé (congés et pentes compris), les intégrales de Green d'un polygone (A, moments statiques, inerties, produit d'inertie), le découpage d'un polygone par un demi-plan, les axes principaux, les modules élastiques et plastiques : la méthode même qui a calculé le catalogue |
| `bois/poutre/donnees.py` | classes de résistance EN 338:2016 et EN 14080:2013 (E_0,mean), sections courantes du bois massif et du lamellé-collé |
| `beton/donnees/ec2.py` | E_cm d'une classe de béton (tableau 3.1) |

Les modules Béton et Acier ne sont pas modifiés. Le câblage partagé `js/biblio.js` gagne quatre gestes génériques,
utiles à toutes les bibliothèques : un champ ou un bouton DANS le cadre du tableau règle un paramètre ; un élément du
dessin se GLISSE (le moteur écrit dans le dessin ce que le glisser règle) ; « Imprimer » ; le champ en cours de saisie
garde le focus quand la page se réécrit.

## 2 · Règles communes aux trois outils

- Le calcul vit dans le moteur Python (`rdm/poutres.py`, `rdm/plaques.py`, `rdm/sections.py`), sans écran ; l'écran
  choisit, compose et formate. Chaque formule affichée est ÉVALUÉE par le moteur : forme générale, valeurs, résultat.
- Une donnée n'est demandée que si la formule s'en sert. La rigidité E·I d'une poutre n'est demandée que pour la
  flèche (les efforts d'une poutre de rigidité constante n'en dépendent pas) ; le coefficient de Poisson d'une plaque
  vient du matériau choisi et s'écrit comme hypothèse.
- Un cas qui n'a pas de solution classique dans le périmètre est REFUSÉ avec sa raison, jamais approché en silence.
- Unités : m, kN, kN/m, kN·m ; flèches en mm ; E en MPa, I en cm⁴ ; sections en mm, cm², cm⁴, cm³.

## 3 · Formulaire de poutres (`rdm/poutres.py`)

### 3.1 Périmètre de la première version

| Modèle | Clé |
|---|---|
| poutre sur deux appuis simples | `appuis` |
| console (encastrée à gauche, libre à droite) | `console` |
| poutre encastrée à gauche et appuyée à droite | `enc_app` |
| poutre bi-encastrée | `bi_enc` |
| poutre continue de 2 à 5 travées sur appuis simples, rigidité constante | `continue` |

| Charge | Données |
|---|---|
| répartie uniforme, totale ou partielle | q (kN/m), de a à b |
| triangulaire, totale ou partielle, croissante ou décroissante | q_max (kN/m), de a à b, sens |
| ponctuelle | P (kN), abscisse a |
| couple | C (kN·m), abscisse a |

Plusieurs charges se superposent (jusqu'à huit) : le comportement est linéaire, les effets s'additionnent.

Non traités dans cette version, et dits à l'écran : porte-à-faux d'une poutre sur appuis, encastrements d'extrémité
d'une poutre continue, appuis élastiques ou dénivelés, rigidité variable, déformation d'effort tranchant.

### 3.2 Conventions

x se compte depuis l'appui de gauche A. Une charge vers le bas est positive. V(x) est la somme des forces à gauche de
la section (une réaction vers le haut est positive) : V = dM/dx. M(x) est positif quand il tend la fibre inférieure
(les diagrammes le dessinent du côté tendu). La flèche f(x) est positive vers le bas : E·I·f″ = −M. Un couple C est
positif dans le sens horaire.

### 3.3 Formules et références

Tout repose sur trois familles de formules de formulaire, que l'on additionne :

**a) La travée isostatique sur deux appuis** (formulaires de RDM : *Techniques de l'ingénieur* C 2 060 « Formulaire
de la résistance des matériaux — poutres droites » ; Roark, *Formulas for Stress and Strain*, tableau 8.1, cas 1e, 2e,
3e ; Timoshenko, *Résistance des matériaux*, t. 1). Avec b = L − a pour une charge ponctuelle :

| Charge | R_A | R_B | E·I·ω_A | E·I·ω_B |
|---|---|---|---|---|
| q sur toute la travée | q·L/2 | q·L/2 | q·L³/24 | q·L³/24 |
| P en a | P·b/L | P·a/L | P·a·b·(L + b)/(6·L) | P·a·b·(L + a)/(6·L) |
| C en a | −C/L | C/L | C·(3·b² − L²)/(6·L) | C·(L² − 3·a²)/(6·L) |
| q de a à b | W·(L − x_c)/L | W·x_c/L | q·[F(b) − F(a)]/(6·L) | q·[G(b) − G(a)]/(6·L) |

avec W = q·(b − a), x_c = (a + b)/2, F(ξ) = L²·ξ² − L·ξ³ + ξ⁴/4, G(ξ) = L²·ξ²/2 − ξ⁴/4 (les rotations d'une charge
répartie sont celles de la charge ponctuelle sommées sur sa longueur). ω_A = f′(0) et ω_B = −f′(L) sont les rotations
de la travée isostatique à ses appuis, toutes deux positives pour une charge vers le bas. La charge triangulaire a les
mêmes écritures fermées (polynômes d'un degré de plus).

Les expressions le long de la travée sont celles des **paramètres initiaux** (Clebsch ; Macaulay, 1919 ; Roark,
tableau 8.1), où ⟨x − a⟩ vaut x − a quand x > a et 0 sinon :

- V(x) = R_A − P·⟨x − a⟩⁰ − q·(⟨x − a⟩ − ⟨x − b⟩)
- M(x) = R_A·x − P·⟨x − a⟩ − q·(⟨x − a⟩² − ⟨x − b⟩²)/2 + C·⟨x − a⟩⁰
- E·I·f(x) = E·I·ω_A·x − R_A·x³/6 + P·⟨x − a⟩³/6 + q·(⟨x − a⟩⁴ − ⟨x − b⟩⁴)/24 − C·⟨x − a⟩²/2

Pour les cas d'école, l'écran affiche la formule classique (q·L²/8, 5·q·L⁴/(384·E·I), P·a·b/L, P·L³/(48·E·I)…) et
le moteur vérifie à chaque calcul qu'elle donne le même nombre que l'écriture générale (deux calculs indépendants ;
un écart arrête le calcul).

**b) Les moments aux appuis d'une travée** M_A et M_B (négatifs quand ils tendent la fibre supérieure) :
M(x) = M_A·(1 − x/L) + M_B·x/L ; V = (M_B − M_A)/L ;
E·I·f(x) = M_A·x·(L − x)·(2·L − x)/(6·L) + M_B·x·(L² − x²)/(6·L).

**c) Les liaisons hyperstatiques**, par l'égalité des rotations (méthode des forces ; Clapeyron, 1857) :

- encastrée en A, appuyée en B : M_A = −3·(E·I·ω_A)/L ;
- bi-encastrée : M_A = −2·(2·E·I·ω_A − E·I·ω_B)/L et M_B = −2·(2·E·I·ω_B − E·I·ω_A)/L ;
- poutre continue, appui i entre les travées i et i + 1 (théorème des trois moments, E·I constant) :
  M_{i−1}·L_i + 2·M_i·(L_i + L_{i+1}) + M_{i+1}·L_{i+1} = −6·(E·I·ω_B,i + E·I·ω_A,i+1).

E·I se simplifie : les efforts n'en dépendent pas. Les formules de formulaire usuelles en découlent (q·L²/8 et
q·L²/12, P·a·b²/L², 9·q·L²/128…) et servent de contrôles.

**d) La console** (encastrée en A) : R_A = somme des charges, M_A = −somme de leurs moments en A ;
E·I·f(x) = −M_A·x²/2 − R_A·x³/6 + les mêmes termes de charge qu'en a). Cas d'école : q·L⁴/(8·E·I),
P·a²·(3·L − a)/(6·E·I).

Les extremums sont cherchés sur les expressions fermées : M est extrême où V s'annule ou saute, f où la rotation
s'annule (dichotomie sur l'expression, à 10⁻⁹·L) ; aucune intégration numérique n'intervient.

### 3.4 Rigidité

Sans rigidité : réactions, V, M. Pour la flèche, au choix : un **profilé acier** du catalogue (E = 210 000 MPa,
EN 1993-1-1 §3.2.6 ; I_y ou I_z selon le sens de pose ; σ = M/W_el affiché) ; une **section bois** du module bois
(classe → E_0,mean ; b × h → I = b·h³/12, W = b·h²/6, sur chant ou à plat) ; ou E et I saisis. La flèche bois est la
flèche instantanée élastique (ni fluage k_def ni effort tranchant : le module « Poutres bois » les traite).

## 4 · Plaques rectangulaires (`rdm/plaques.py`)

### 4.1 Périmètre de la première version

Plaque mince rectangulaire a × b, épaisseur t constante, charge UNIFORME q, matériau élastique isotrope (théorie de
Kirchhoff-Love, petites flèches). Chaque bord est simplement appuyé, encastré ou libre.

| Configuration | Méthode | Référence |
|---|---|---|
| quatre bords appuyés | série de Navier et série de Lévy (les deux, comparées) | Timoshenko & Woinowsky-Krieger, *Theory of Plates and Shells*, 2ᵉ éd., §§ 28-30, tableau 8 |
| deux bords opposés appuyés, les deux autres appuyés, encastrés ou libres (toute combinaison) | série de Lévy | ibid., §§ 41-43, 46-48 |
| toute combinaison de bords appuyés et encastrés (dont quatre bords encastrés, deux bords adjacents encastrés, trois bords encastrés) | superposition de Timoshenko : plaque appuyée + moments de bord en séries, rotations nulles sur les bords encastrés | ibid., §§ 42-45, tableau 35 |
| un bord libre dont les deux bords voisins ne sont pas tous deux appuyés (console, trois bords encastrés et un libre, coin libre…) | **non pris en charge** : pas de solution de Lévy, la superposition ci-dessus ne traite pas le bord libre | — |

Résultats : flèche maximale et sa position, M_x et M_y au centre, moments au milieu des bords encastrés et du bord
libre, σ = 6·M/t². Écriture : w = α·q·a⁴/D, M = β·q·a², D = E·t³/(12·(1 − ν²)) ; α et β sont affichés pour être
comparés aux tableaux publiés (qui valent pour ν = 0,3).

### 4.2 Le coefficient de Poisson

D et les moments en dépendent : la formule en a besoin. Il n'est pas demandé pour autant : il vient du MATÉRIAU —
acier E = 210 000 MPa, ν = 0,3 (EN 1993-1-1 §3.2.6) ; béton non fissuré E_cm de la classe, ν = 0,2 (EN 1992-1-1
§3.1.3(4)) ; aluminium E = 70 000 MPa, ν = 0,3 (EN 1999-1-1 §3.2.5) ; « autre » : E et ν saisis. L'hypothèse est
écrite à l'écran.

### 4.3 Limites dites à l'écran

Plaque mince (t ≤ petit côté / 10 environ) ; flèche petite devant l'épaisseur (au-delà de t/2 l'effet de membrane
raidit la plaque : la flèche affichée est alors majorée, l'écran le signale) ; coins maintenus (une plaque appuyée
tend à se soulever aux coins) ; charge uniforme seule ; pas de vérification de résistance au sens d'un Eurocode.

### 4.4 Contrôles prévus

Valeurs publiées (Timoshenko, tableaux 8 et 35 : plaque carrée appuyée α = 0,00406, β = 0,0479 ; encastrée
α = 0,00126, moment d'encastrement −0,0513·q·a², au centre 0,0231·q·a²) ; limites de poutre quand b/a → ∞
(5/384 = 0,01302, 1/8 ; 1/384 = 0,00260, −1/12, 1/24) ; accord des deux méthodes là où elles se recouvrent (Navier,
Lévy, superposition) ; symétries ; convergence (doubler le nombre de termes ne change pas la 4ᵉ décimale).

## 5 · Caractéristiques d'une section quelconque (`rdm/sections.py`)

### 5.1 Méthode

Une section est une liste d'ÉLÉMENTS ; chaque élément est un ou plusieurs contours polygonaux (les arcs — congés,
cercles — discrétisés finement : 36 segments par congé, 180 par cercle, comme le catalogue). Sur chaque contour, les
**intégrales de Green** donnent exactement, pour le polygone : A, les moments statiques S_y, S_z, les moments
d'inertie et le produit d'inertie. Puis, comme à la main :

- centre de gravité : y_G = Σ A_i·y_i / Σ A_i ; z_G = Σ A_i·z_i / Σ A_i ;
- **Huygens-Steiner** : I_y = Σ (I_y,i + A_i·(z_i − z_G)²) ; I_z = Σ (I_z,i + A_i·(y_i − y_G)²) ;
  I_yz = Σ (I_yz,i + A_i·(y_i − y_G)·(z_i − z_G)) ;
- axes principaux : tan 2α = −2·I_yz/(I_y − I_z) ; I_1,2 = (I_y + I_z)/2 ± √(((I_y − I_z)/2)² + I_yz²) ;
- rayons de giration i = √(I/A) ; modules élastiques W = I/v pour chaque fibre extrême ; modules plastiques (axe
  d'égales aires, par dichotomie et découpage — section d'un seul matériau).

Un profilé du catalogue non coupé retrouve ses valeurs du catalogue (c'est la même méthode) : contrôle systématique.
Un élément peut être déplacé, tourné, retourné (miroir), COUPÉ par une droite (un demi-HEA : coupe à mi-hauteur) ou
déclaré ÉVIDEMENT (sa matière est retirée).

### 5.2 Ce qui n'est pas calculé, et pourquoi

- **Centre de cisaillement, inertie de torsion, inertie de gauchissement** : ils demandent la fonction de
  gauchissement de la section entière (ou la théorie des parois minces sur sa ligne moyenne), qui n'est pas définie
  pour un assemblage quelconque d'éléments pleins, de tubes et de pièces coupées. Ils sont affichés « non calculé ».
  Seule exception écrite : une section à deux axes de symétrie a son centre de cisaillement au centre de gravité.
- **Matière comptée deux fois** : deux éléments qui se recouvrent comptent leur partie commune deux fois. Le moteur
  détecte le recouvrement et l'écran le signale.
- Sections de plusieurs matériaux (acier + bois, béton) : pas d'homogénéisation dans cette version.

## 6 · L'écran

Une page `bibliotheque-autres` (carte « Bibliothèque Autres » à l'accueil, de la hauteur de deux rangées), trois
rubriques : **Poutres**, **Plaques**, **Sections**. Même disposition pour les trois : la barre (le modèle et les
données générales), puis un cadre — le DESSIN à gauche, toujours visible pendant la saisie, les DONNÉES et les
RÉSULTATS à droite, la FEUILLE DE CALCUL dessous (formule générale → valeurs → résultat) ; la bande donne le résultat
choisi en grand ; « Imprimer » sort la feuille (données, hypothèses, dessin, calculs, sources). Trois formats : PC et
tablette en deux colonnes, téléphone en une colonne.

- Poutres : le schéma (appuis, charges, cotes, réactions), puis V, M et la déformée alignés sous lui ; une charge se
  glisse le long de la poutre ; un curseur lit V, M et f dans une section au choix (la feuille montre alors chaque
  formule évaluée en cette section).
- Plaques : le plan de la plaque avec le symbole de chaque bord (un clic sur un bord change sa condition), les
  lignes de niveau de la flèche, les points où les moments sont donnés.
- Sections : le dessin à l'échelle (axes de référence, G, axes principaux), la liste des éléments ; un élément se
  glisse, se tourne, se règle au clavier ; « Dupliquer », « Supprimer », « Couper », « Évidement ».

## 7 · Les contrôles

- `tests/test_rdm_poutres.py`, `tests/test_rdm_plaques.py`, `tests/test_rdm_sections.py` : chaque cas d'école contre
  sa formule écrite dans le test (copie nommée), les valeurs publiées, l'équilibre (Σ R = Σ charges, Σ moments = 0),
  la continuité, les symétries, le catalogue ; des mutations rejouées.
- `tests/test_api_web_bibliotheque_autres.py` : le contrat des bibliothèques, ce que l'écran affiche = le moteur,
  les refus, l'échappement, l'adresse.
- `verifier_site.js`, filtre « bibliotheque » : les trois rubriques aux quatre formats, les gestes (glisser,
  ajouter, supprimer), l'impression.
