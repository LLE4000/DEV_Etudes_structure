# Garde-corps — document de référence

*Version 2, livrée le 05/11/2026 (version 1 le 04/11/2026). Page « Garde-corps » (tuile Autres),
moteur `acier/garde_corps/`, étude de conception `docs/acier/GARDE_CORPS_ETUDE.md`.*

**Ce que la version 2 change.** Le type de garde-corps se choisit dans un MENU (barreaudage,
lisses, panneau, main courante seule, main courante murale) ; les QUATRE charges sont nommées
et agissent en cas DISTINCTS (le cumul q_h + q_v est une option) ; la main courante se
dimensionne en poutre sur deux appuis (défaut) ou continue ; la portée des barreaux se déduit
de H, de h_bas et des sections ; les RHS sont proposés par défaut ; sous chaque élément, le
TABLEAU des profilés compatibles est cliquable ; les dessins sont des dessins techniques (six
vues) ; la note met les HYPOTHÈSES en tête et déroule 1 · montant, 2 · main courante,
3 · barreaux, 4 · ancrage avec les formules développées, ELS du bureau par défaut.

## 1 · Ce qui est livré

| Pièce | Rôle |
|---|---|
| `acier/garde_corps/types.py` | les cinq TYPES du menu ; chacun pose le système (montants verticaux, consoles au mur) et le remplissage (barreaux, lisses, panneau, aucun) |
| `acier/garde_corps/charges.py` | les quatre charges nommées : q_h (Tab. 6.12 par catégorie, ou valeur libre, ou P direct), q_v, Q_r (NBN B 03-004 Q_kh,2), q_r ; les cas de charge (distincts, ou cumul 1 + 2) ; le tableau des hypothèses |
| `acier/garde_corps/geometrie.py` | la forme en plan (droit, L, U), les tronçons à entraxes libres (réguliers ou saisis), les extrémités (montant, mur), les MONTANTS avec leur longueur tributaire par direction (angles compris) |
| `acier/garde_corps/bibliotheque.py` | les sections : 879 creux du catalogue, 56 plats et 8 ronds de stock, sections libres (tube, caisson, plat, rond) par `familles.calculer` ; axe fort = sens de h |
| `acier/garde_corps/membres.py` | la mécanique : console (M = P·H, f = PH³/3EI), poutre sur deux appuis (qL²/8 + QL/4, 5qL⁴/384EI) ou CONTINUE (coefficients des tables 0,125 / 0,100 / 0,107 qL²), vérification σ = M/W + N/A, τ = V/A_v, flèche ; limites ELS et Eurocode |
| `acier/garde_corps/fixation.py` | la soudure sur la platine comme cas du module Soudures (C6 tube, C3 plat, C8 section libre) ; la platine boulonnée (bras de levier, rangée tendue, boulons EN 1993-1-8 ou chevilles ETA, flexion de la platine, épaisseur requise) |
| `acier/garde_corps/etat.py` | les cas, les défauts (RHS 50x30x3), le JSON, la vérification complète sous chaque cas de charge (montants un à un, main courante, barreaux ou lisses, consoles, soudure et platine), les CANDIDATS par rôle et le plus léger qui tient |
| `acier/garde_corps/formules.py` | les formules écrites comme à la main, par BLOC (cas gouvernant, autres cas et leur taux, substitutions testées contre le moteur) : montant, console, main courante (deux appuis ou continue), barreaux, platine ; la soudure reprend celles du module Soudures |
| `acier/garde_corps/methode.py` | les textes de méthode, courts (onglet Détail et page 2 de la note), les références |
| `acier/garde_corps/schemas.py` | les six vues en dessin technique : élévation, coupe A-A, vue en plan, platine en plan, coupe sur la fixation, section |
| `acier/garde_corps/note.py` | la note PDF en deux pages, les hypothèses en tête |
| `modules/garde_corps.py` | l'écran : menu des types, cinq onglets, tableaux cliquables ; `modules/liaison.py` : la liaison état → widgets partagée |
| `tests/test_garde_corps_moteur.py`, `_formules.py`, `_schemas.py`, `_ecran.py`, `_note.py` | les tests (87 + 39 + 64 + 55 + 47) ; `tools/captures_garde_corps.js` : captures et mesures aux trois formats, clic sur une ligne de tableau dans un vrai navigateur |

## 2 · Le modèle

**Types.** Barreaudage vertical (défaut), lisses horizontales, remplissage en panneau, main
courante seule sur montants, main courante murale sur consoles. Le type est un préréglage :
il pose le système et le remplissage, tout le reste reste libre.

**Montants.** Le garde-corps est une ligne en plan (droit, L, U) découpée en travées ; les
entraxes sont réguliers (longueur et entraxe maximal) ou saisis un à un. Chaque montant est une
console de hauteur H chargée en tête par P₁ = q_h × s_trib (s_trib = demi-somme des entraxes
voisins) et N = q_v × s_trib ; un montant d'angle reçoit une demi-travée de chaque tronçon,
dans deux directions perpendiculaires, et une section rectangulaire y est vérifiée dans les
deux orientations. Une extrémité ancrée dans un mur n'a pas de montant : le mur reprend la
demi-travée, et la note dit l'effort à justifier avec la cheville.

**Main courante murale.** La main courante est l'élément principal (poutre sur les consoles,
portée l'entraxe le plus grand) ; les consoles, aux mêmes entraxes, sont des consoles de
longueur e sous R_v = q_v·s_trib (flexion, cas 2), R_h = q_h·s_trib (axial, tire la console
hors du mur, cas 1) ; leur platine est tirée (enveloppe des deux modèles).

**Main courante, remplissage.** La main courante est une poutre entre montants fléchie dans
deux plans (q_h horizontal → axe faible, q_v vertical → axe fort), sur deux appuis (défaut :
les lisses sont rarement continues d'un montant à l'autre) ou continue sur n travées
(coefficients des tables). Un barreau est une poutre entre le dessus de la lisse basse (axe à
h_bas) et le dessous de la main courante — sa portée se DÉDUIT de H, de h_bas et des sections ;
une lisse intermédiaire est une poutre entre montants ; chacun sous Q_r à mi-portée (cas 3) OU
q_r × entraxe (cas 4), le pire des deux. Un panneau n'est pas vérifié (à justifier par le
fournisseur), la lisse basse n'est pas calculée.

**Cas de charge.** Les quatre charges agissent SÉPARÉMENT : cas 1 (q_h seule), 2 (q_v seule),
3 (Q_r seule), 4 (q_r seule) ; chaque élément, la soudure et la platine sont vérifiés sous
chaque cas et le pire gouverne — la note nomme le cas gouvernant de chaque élément et les
autres cas avec leur taux. L'option « cumul » fait agir q_h et q_v ensemble (cas 1 + 2, côté
sûr) ; les charges 3 et 4 restent des cas séparés.

**Fixation.** La soudure du montant (ou de la console) sur la platine est un cas du module
Soudures : tube sur platine (C6) pour un tube du catalogue, console fléchie (C3) pour un plat,
cordons libres sur le contour (C8) pour une section libre — avec la jonction choisie (angle,
pénétration partielle, pleine) et la méthode du garde-corps. La platine est rigide, deux
rangées de n chevilles à ±e_x/2 : z = e_x/2 + h_p/2, Z = (M − N·h_p/2)/z, F_t = Z/n,
F_v = V/(2n) ; boulon (Tab. 3.4) ou cheville (N_Rd, V_Rd de l'ETA, interaction (β)^1,5) ;
bande de platine en console du nu du montant à la rangée tendue (m, b_eff, σ_p, t_req).

## 3 · Les deux méthodes et le switch

| | Méthode ELS (bureau) — défaut | Eurocode (ELU) |
|---|---|---|
| Charges | de service, sans coefficient | × γ_Q = 1,5 (flèche : service) |
| σ | M/W_el + N/A ≤ σ_adm (160 N/mm²) | ≤ f_y/γ_M0 (élastique, W_el) |
| τ | V/A_v ≤ τ_adm (90 N/mm²) | ≤ f_y/(√3·γ_M0) |
| Flèche | ≤ L/100 (ou mm) | ≤ L/100 (ou mm) |
| Soudure | méthode ELS du bureau, k = 100 kg/(mm·cm) | directionnelle EN 1993-1-8, γ_M2 = 1,25 |
| Boulons, chevilles | résistances de calcul / γ_F = 1,5 | résistances de calcul |

Le cas par défaut (barreaudage, RHS 50x30x3, H = 1,10 m, 0,45 kN en tête du montant M2)
donne 64 % (flèche 7,0 mm pour 11 admissibles, cas 1) en ELS ; les barreaux Plat 30x8 sont à
62 % (cas 3, Q_r), la soudure à 59 %, les chevilles à 55 %, la platine à 34 %.

## 4 · Les charges

| N° | Charge | Valeur par défaut | Origine |
|---|---|---|---|
| 1 | q_h — charge linéaire horizontale sur la main courante, à la hauteur H | 0,5 kN/m (catégorie A) | EN 1991-1-1 §6.4, Tab. 6.12 (NBN B 03-004 : Q_kh,1) |
| 2 | q_v — charge linéaire verticale sur la main courante | 0,5 kN/m | usage du bureau (NBN B 03-004 : Q_kv, à confirmer) |
| 3 | Q_r — charge ponctuelle horizontale sur le remplissage | 0,5 kN | NBN B 03-004 : Q_kh,2 |
| 4 | q_r — pression horizontale répartie sur le remplissage | 0,5 kN/m² | usage du bureau |

Catégories du Tableau 6.12 (valeurs RECOMMANDÉES de l'EN : A, B, C1 0,5 ; C2–C4, D 1,0 ;
C5 3,0 ; E 2,0 kN/m, ou valeur libre) : l'annexe belge peut en fixer d'autres (point P-1 de
l'étude) ; la note imprime la valeur retenue et son origine. En option : P directe par
montant (la main courante reçoit alors P/s), Q ponctuelle sur la main courante.

## 5 · Les schémas

![schémas](images/garde_corps_schemas.png)

Six vues à l'échelle, en style dessin technique, les mêmes à l'écran et dans la note :
ÉLÉVATION (dalle hachurée, montants à leurs entraxes, main courante, lisse basse, barreaux
centrés par travée à leur entraxe — ou lisses, ou panneaux —, platines et chevilles, murs
d'extrémité, angles d'un L ou d'un U, q_h ⊙, q_v ↓, Q_r ⊙, ◉ le montant gouvernant et son
taux, chaîne des entraxes, L, H, h_bas, e_b, le repère A-A) ; COUPE A-A (le garde-corps vu de
bout dans une travée : dalle, platine et chevilles, montant vu de côté, main courante et lisse
basse coupées avec leur contour du catalogue hachuré, q_h, q_v, Q_r, H, h_bas, b_p, e_x) ; VUE
EN PLAN (droit / L / U, main courante en bande, montants orientés, charges perpendiculaires,
L₁ L₂ L₃, repère A-A) ; PLATINE EN PLAN (section du montant avec sa bande de cordon, trous des
chevilles et leurs axes, e_x, e_y, b_p, l_p, chaînes des bords) ; COUPE SUR LA FIXATION (montant
et platine coupés, cordons à a·√2, chevilles dans le béton ou boulons dans l'acier, bras m et
z, V N M du cas gouvernant) ; SECTION (contour, axes y et z, h × b ou Ø, t). Au mur : les mêmes
vues avec la console et la platine verticale.

## 6 · L'écran

![écran PC](images/garde_corps_pc.png)

Barre d'outils du cas, switch de méthode (ELS du bureau par défaut), les six dessins à gauche,
la carte RÉSULTAT (statut, η, élément gouvernant avec son critère et son cas, une barre par
élément, masse indicative) et cinq onglets à droite :

1. **Garde-corps** — le TYPE en menu déroulant, la forme en plan, H, les tronçons (longueur,
   entraxe maxi, entraxes saisis), les extrémités, le remplissage du type (e_b et h_bas, ou le
   nombre et l'entraxe des lisses) avec la portée déduite, le tableau des montants ;
2. **Charges** — les quatre charges numérotées avec leur nom et leur origine, la catégorie
   d'usage, la case « cumuler q_h et q_v », les options (P direct, Q) ;
3. **Sections** — pour chaque élément du cas (montant, main courante avec son modèle sur deux
   appuis ou continue, barreaux avec la lisse basse, ou lisses, ou console) : la section, son
   taux et son cas, puis le TABLEAU des profilés compatibles (familles cochées — RHS par défaut
   pour le montant et la main courante, plats et ronds pour les barreaux —, η σ, η flèche, η en
   barre, la section retenue marquée ◀) : un CLIC sur une ligne retient la section, et le
   bouton « Prendre le plus léger qui tient » reste ;
4. **Ancrage** — fixation (platine, soudé sur plat, scellé), console au mur, soudure (jonction,
   gorge, t_p, η et dessins du module Soudures), platine (b_p, l_p, e_x, n, e_y), chevilles ou
   boulons, les résultats (Z, F_t, F_v, interaction, m, σ_p, t_req) ;
5. **Détail** — tableaux des éléments (avec leur cas) et des montants, les formules par bloc,
   les paramètres de la méthode, l'identification de la note, la méthode et les sources.

![tablette](images/garde_corps_tablette_lisses.png) ![téléphone](images/garde_corps_telephone_murale.png)

## 7 · La note

![note p1](images/garde_corps_note_p1.png)

Page 1 · CALCUL, les hypothèses d'abord : bandeau (statut, η, élément gouvernant, critère, cas
de charge, méthode) ; HYPOTHÈSES (garde-corps, sections, matériau et limites, fixation) ; le
tableau des QUATRE CHARGES avec leur origine et la règle des cas distincts ; les dessins
(élévation et coupe A-A côte à côte, vue en plan) ; les CRITÈRES en typographie mathématique ;
1 · MONTANT (cas gouvernant, autres cas et leur taux, P₁, M₁, σ, τ, f, η) ; 2 · MAIN COURANTE ;
le RÉCAPITULATIF ; 3 · BARREAUX ; 4 · ANCRAGE (soudure, z, Z, F_t, F_v, résistances,
interaction, m, b_eff, σ_p, t_req) ; la CONCLUSION. Page 2 · DÉTAILS ET MÉTHODE : platine en
plan et coupe sur la fixation, les montants un à un avec leur cas, la section et la
bibliothèque, la soudure (vue et formules du module Soudures), la méthode choisie, les
références, le cartouche avec l'échelle de l'élévation.

![note p2](images/garde_corps_note_p2.png)

## 8 · Les preuves

- `tests/test_garde_corps_moteur.py` (87) : types, charges nommées et cas, géométrie (entraxes
  saisis, angles, mur), mécanique (console, poutre sur deux appuis et continue, limites),
  bibliothèque, vérification complète sous chaque cas, fixation, candidats par rôle, JSON ;
- `tests/test_garde_corps_formules.py` (39) : plus de 300 substitutions ré-évaluées sur treize
  cas, les blocs nomment leur cas gouvernant et les autres ;
- `tests/test_garde_corps_schemas.py` (64) : les six vues et leur contenu à l'échelle, le
  peintre, la pureté ;
- `tests/test_garde_corps_ecran.py` (55) : navigation, écran, le menu des types, source unique,
  switch, tableaux cliquables (colonnes, familles par défaut, section retenue ◀, génération de
  la clé après un changement), cas multiples, note, étanchéité ;
- `tests/test_garde_corps_note.py` (47) : deux pages sans débordement sur huit cas, les
  hypothèses d'abord, le calcul 1 à 4, les dessins peints, la page 2, les variantes ;
- `tools/captures_garde_corps.js` : cinq états × trois largeurs, mesures de ce que l'œil voit,
  dont le CLIC sur une ligne de tableau dans un vrai navigateur (le sélecteur affiche la section
  cliquée).

## 9 · Couverture : ce que l'outil calcule, ce qu'il ne calcule pas

**Calculé** : tout garde-corps droit, en L ou en U, à entraxes libres, des cinq types, sur
montants verticaux (tubes, plats, ronds, sections libres) encastrés sur platine (soudée,
boulonnée par chevilles ou boulons, ou scellée sans vérification) ou sur un plat existant ; la
main courante murale sur consoles ; main courante sur deux appuis ou continue, barreaux (portée
déduite), lisses intermédiaires ; les quatre charges en cas distincts ou q_h + q_v cumulées.

**Non calculé** : béton sous la platine et modes de ruine du béton d'ancrage (module Pied de
poteau), lisse basse, panneaux (verre, tôle), chocs et fatigue, flambement du montant,
aluminium et inoxydable, garde-corps de véhicules (EN 1991-1-1 annexe B), fixation latérale en
nez de dalle (prendre H jusqu'à la fixation).
