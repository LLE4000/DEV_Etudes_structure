# Garde-corps — étude de conception avant développement

*Écrite le 04/11/2026 à la demande du bureau : « la structure est mise, on utilise la vieille
formule, mais c'est mal organisé ; il faudrait tout réorganiser pour qu'on puisse dimensionner
n'importe quel garde-corps et que ça fasse une belle note de calcul, sans aller à cent clics ».*

## 0 · La décision en une page

**Un garde-corps est une ligne en plan portée par des éléments principaux.** Soit des
MONTANTS verticaux encastrés en pied sur une platine (le cas courant), soit une LISSE
horizontale ancrée dans un mur par des consoles (la main courante d'escalier déjà rencontrée).
La ligne est droite, en L ou en U ; ses entraxes sont libres — réguliers à partir d'une
longueur et d'un entraxe maximal, ou saisis un à un (« un poteau central, un à gauche, un à
droite, pas à la même distance »). Chaque montant reçoit la charge de sa **longueur
tributaire**, la demi-somme des deux entraxes voisins ; un montant d'**angle** reçoit une
demi-travée de chaque tronçon, dans deux directions perpendiculaires. Une extrémité ancrée dans
un mur n'a pas de montant : le mur reprend la demi-travée, et la note dit l'effort à justifier.

**Les charges viennent de la norme, et restent modifiables.** La charge linéaire horizontale
q_h sur la lisse haute est celle de l'EN 1991-1-1 §6.4, Tableau 6.12, par catégorie d'usage
(0,5 kN/m en habitation, 1,0 en C2–C4 et D, 3,0 en C5, 2,0 en E — valeurs RECOMMANDÉES de
l'EN ; l'ANB belge n'a pas pu être consultée depuis la session : la valeur est modifiable,
« valeur libre », et la note imprime la valeur retenue et son origine). On peut aussi saisir
directement la charge ponctuelle P qui revient à chaque montant. S'y ajoutent une charge
verticale q_v sur la main courante (usage du bureau), une charge ponctuelle Q en option, et les
charges du remplissage (pression q_r, charge ponctuelle Q_r = 0,5 kN de la NBN B 03-004).

**Les vieilles formules du bureau, et l'Eurocode en face.** Un switch, comme pour les
soudures : « Vérification à l'ELS » (nommée ainsi depuis la version 3 ; l'étude l'appelait
« Méthode ELS (bureau) ») = efforts de service et contraintes admissibles
(σ = M/W_el + N/A ≤ 160 N/mm², τ = V/A_v ≤ 90 N/mm², flèche ≤ L/100) ; « Eurocode (ELU) » =
charges × γ_Q = 1,5, σ ≤ f_y/γ_M0 (élastique), τ ≤ f_y/(√3·γ_M0), flèche de service. La
fixation suit : la soudure du montant sur la platine est UN CAS DU MODULE SOUDURES (tube sur
platine C6, console fléchie C3 pour un plat, cordons libres C8 pour une section non standard —
mêmes dessins, mêmes formules, même switch), les chevilles ou boulons et la platine sont
calculés comme à la main (platine rigide, bras de levier, rangée tendue, bande en console).

**Une bibliothèque qui choisit.** Tous les profilés creux du catalogue (CHS, SHS, RHS finis à
chaud et formés à froid : 879 sections), les plats et ronds de stock (64), et les sections NON
STANDARD définies par leurs dimensions (tube, caisson, plat, rond) calculées par la même
fonction que le catalogue. Chaque section est essayée comme élément principal du cas ; le
tableau donne son taux ; un bouton prend « le plus léger qui tient ».

**Une note de deux pages** sur le kit `ndc_pdf` : dessins (élévation développée, plan, détail
de la fixation), charges avec leur origine, montants et longueurs tributaires, éléments,
vérification du montant écrite comme à la main (substitutions testées contre le moteur),
fixation, conclusion ; méthode, soudure (vue et formules du module Soudures), bibliothèque.

**Zéro clic pour un premier résultat, un clic pour la note.** Le cas par défaut est un
garde-corps complet et conforme (droit 3,60 m, entraxe ≤ 1,00 m, H = 1,10 m, catégorie A,
CHS 48,3 × 3,2 sur platine 150 × 150 × 10 à quatre chevilles M12, lisse CHS 42,4 × 3,2,
barreaux plats 30 × 8 à 110 mm). Tout le reste se règle dans six onglets à libellé fixe.

## 1 · Ce que l'application a déjà, et ce que le module réutilise

| Existant | Réemploi |
|---|---|
| `modules/garde_corps.py` (ancien) : σ = M/W, τ = V/A_v, flèche L/x, console encastrée M = P·H, poutre qL²/8 + QL/4, combinaison max / somme, charge « P directe » ou « q × s », barreau sous q_panneau × entraxe | les formules, reprises telles quelles dans `acier/garde_corps/membres.py` (le module est réécrit, la mécanique est la même) |
| `acier/donnees/profiles.json`, `acier/profiles/familles.calculer` | la bibliothèque : creux du catalogue, plats et ronds de stock, sections libres — mêmes nombres partout |
| `acier/soudures` (moteur, schémas, formules, méthode ELS / Eurocode) | la soudure du montant ou de la console sur la platine : un cas C6 / C3 / C8 construit par `fixation.cas_soudure` |
| `acier/resistances.py` (F_t,Rd, F_v,Rd), `acier/donnees/bibliotheques.json` (boulons, classes, aciers) | les boulons de la platine, les nuances |
| kit de dessin des assemblages (feuille de cotation, primitives, hachures, ruptures, rendu SVG, peintre PDF) et briques des schémas de soudure (flèches, ⊙, étiquettes, étendue) | les quatre vues |
| kit `ndc_pdf`, `Sub` / `rendre` / `verifier` des formules d'assemblage | la note, les formules testables |
| `modules/liaison.py` (liaison état → widgets), charte des écrans (barre d'outils, switch, carte résultat, onglets à libellé fixe) | l'écran |

## 2 · Le modèle mécanique

### 2.1 Géométrie et longueurs tributaires

Une géométrie est : un système (montants | mur), une forme (droit | L | U → un, deux ou trois
tronçons), une hauteur H (la main courante au-dessus de la fixation — c'est le bras de levier ;
fixation en nez de dalle : ajouter la descente), des tronçons (longueur L, entraxe maximal
s_max, et facultativement des entraxes saisis), deux extrémités (montant | mur), et, pour le
système mur, la longueur e des consoles.

`geometrie.analyser` déroule les travées le long du développement et pose les montants :
M1 à l'origine (sauf extrémité « mur »), un montant à la fin de chaque travée, le dernier
seulement si l'extrémité est un montant. Longueur tributaire d'un montant courant :
(s_gauche + s_droite)/2 dans la direction normale au tronçon ; d'un montant d'angle :
s_gauche/2 normale au tronçon précédent et s_droite/2 normale au suivant. Une extrémité
« mur » laisse la demi-travée au mur (effort imprimé dans la note : H = q_h·s/2, V = q_v·s/2).

Exemple de l'énoncé : entraxes saisis « 1200 900 » → trois montants, tributaires 600 / 1050 /
450 mm — le central n'est pas symétrique et c'est lui qui gouverne.

### 2.2 Charges et nature

- q_h (kN/m) à la hauteur H (la norme la limite à 1,20 m : au-delà, H est pris tel quel, du
  côté sûr, et la page le dit) ; ou P (kN) direct par montant, la lisse recevant P/s ;
- q_v (kN/m) sur la main courante, agissant AVEC q_h (côté sûr) ;
- Q (kN) ponctuelle sur la lisse, en « somme » avec q_h ou en « max » (cas séparés) ;
- remplissage : q_r (kN/m²) × entraxe sur un barreau ou une lisse intermédiaire, OU Q_r à
  mi-portée — deux cas, le pire gouverne (la NBN B 03-004 fait agir Q_kh,2 seule).

Méthode ELS : charges de service. Eurocode : × γ_Q pour les résistances, service pour les flèches.

### 2.3 Éléments

| Élément | Modèle | Efforts | Vérification |
|---|---|---|---|
| Montant (système montants) | console de hauteur H | P₁ = q_h·s_trib (axe fort, sens de h), P₂ = q_h·s_trib,2 (angle), N = q_v·s_trib | σ = M₁/W_el,y + M₂/W_el,z + N/A ; τ = V/A_v ; f = √(f₁² + f₂²), f_i = P_i·H³/(3EI) |
| Montant d'angle rectangulaire | les deux orientations | P₁ selon h puis P₂ selon h | la pire des deux gouverne |
| Lisse haute (ou lisse principale, système mur) | poutre sur deux appuis, portée s_max | q_h (+ Q) horizontal → axe faible, q_v vertical → axe fort | σ = M_v/W_el,y + M_h/W_el,z ; f = √(f_h² + f_v²) |
| Barreau | poutre sur deux appuis, portée L_b, entraxe e_b | q_r·e_b ou Q_r | σ, τ, f ; le pire des deux cas |
| Lisse intermédiaire | poutre entre montants, entraxe e_l | q_r·e_l ou Q_r | idem |
| Console (système mur) | console de longueur e | R_v = q_v·s_trib (fléchit), R_h = q_h·s_trib (axial), R_h,2 (angle, latéral) | σ = M/W + R_h/A, τ, f |
| Panneau | non vérifié | — | message : à justifier par le fournisseur |

L'aire de cisaillement est celle du catalogue (EN 1993-1-1 §6.2.6 : 2A/π pour un tube rond,
A·h/(b + h) pour un rectangulaire), l'aire entière pour un plat ou un rond — l'ancien module
prenait toute la paroi ; le cisaillement ne gouverne jamais un garde-corps, et la source unique
prime (décision écrite ici).

### 2.4 Fixation

- **Soudure** : cas du module Soudures avec les efforts de l'élément fixé le plus chargé
  (N, V = P, M = P·H ; système mur : N = R_h en traction, V = R_v, M = R_v·e), la jonction
  choisie (angle a / partielle p + renfort / pleine) et la méthode du garde-corps (ELS → ELS
  bureau k = 100 ; Eurocode → directionnelle, γ_M2 = 1,25).
- **Platine boulonnée** (deux rangées de n boulons à ±e_x/2 dans le sens de l'effort) :
  z = e_x/2 + h_p/2 ; Z = (M − N·h_p/2)/z (N compression soulage, ≥ 0) ; sous traction N_t,
  enveloppe des deux modèles : (M + N_t·h_p/2)/z et M/e_x + N_t/2 ; F_t = Z/n, F_v = V/(2n).
  Boulon : F_t,Rd = 0,9·f_ub·A_s/γ_M2, F_v,Rd = α_v·f_ub·A_s/γ_M2, F_v/F_v,Rd + F_t/(1,4·F_t,Rd)
  ≤ 1 ; cheville : N_Rd, V_Rd de l'ETA, (β_N)^1,5 + (β_V)^1,5 ≤ 1. En ELS, résistances de
  calcul / γ_F = 1,5 (l'équivalent de service, comme le k des cordons).
- **Platine** : bande en console du nu du montant à la rangée tendue, m = e_x/2 − h_p/2 −
  0,8·a·√2, b_eff = min(l_p ; Σ min(2πm ; 4m + 1,25e)), σ_p = 6·Z·m/(b_eff·t_p²) ≤ σ_lim,
  épaisseur requise.
- **Platine latérale en nez de dalle (version 3)** : platine verticale l_p × b_p (largeur ×
  hauteur) en applique, descendue de d_h sous le dessus de dalle ; le montant la longe et y est
  soudé par deux cordons longitudinaux de hauteur b_p (cas C8 libre du module Soudures, repère
  au centre des cordons, torseur transporté). Le montant est une console de H + d_h ; au centre
  de la platine M = P·(H + d_h + b_p/2), N = −P (arrachement), V = N_v = q_v·s_trib ; un
  montant d'angle ajoute M₂ = P₂·(H + d_h + b_p/2) en torsion des ancrages. Deux rangées
  horizontales de n ancrages à ±e_x/2 (la haute tendue, la basse en appui sur le béton) :
  z = e_x, Z = M/z + P/2 (P se partage entre les deux rangées, côté sûr pour la haute),
  F_t = Z/n, F_v = V/(2n) + M_z·r_max/Σr². Platine : bande en console du nu du montant à la
  colonne d'ancrages, m = e_y/2 − b/2 − 0,8·a·√2, b_eff = min(b_p ; Σ min(2πm ; 4m + 1,25e)),
  σ_p = 6·Z·m/(b_eff·t_p²). Refus : n impair (les colonnes vont par paires, une de chaque côté
  du montant), colonne dans le montant (e_y ≤ b + d_0), rangée hors de la platine.
- Pas de béton sous la platine ni de modes de ruine du béton d'ancrage (module Pied de poteau).

### 2.5 Statuts

η ≤ 0,85 conforme ; ≤ 1 conforme avec attention ; > 1 non conforme ; refus sur données
impossibles (aucune travée, section inconnue, rangées de boulons dans le montant).

## 3 · La bibliothèque

`bibliotheque.standards()` : les 879 creux du catalogue + 56 plats (EN 10058 : 30 à 120 ×
5 à 20) + 8 ronds (Ø 10 à 30) = 943 sections nommées ; `section_libre(famille, dims)` : TUBE,
CAISSON, PLAT, ROND par `familles.calculer`. Axe fort = sens de h ; aire de cisaillement entière
pour un plein. `etat.candidats(cas, groupes, masse_max)` vérifie chaque section comme élément
principal (montant gouvernant ou lisse) ; `plus_leger` prend la première qui tient. Sur le cas
par défaut : 398 candidats en 20 ms ; le plus léger tube qui tient est le CHS 48,3 × 2,5 (91 %).

## 4 · Les schémas

Quatre vues sur le kit des assemblages (`schemas.py`) :

1. **Élévation développée** : montants à leurs abscisses, lisse, barreaux ou lisses
   intermédiaires, platines avec chevilles, murs d'extrémité hachurés, angles d'un L ou d'un U
   (trait d'axe + « angle »), ⊙ pour q_h et flèches pour q_v, la chaîne des entraxes sur une
   seule rangée (`_chaine`), L, H, les noms et le gouvernant « ◉ M2 · 75 % » ;
2. **Vue en plan** : la ligne droite / en L / en U, les montants, les flèches de charge
   perpendiculaires à chaque tronçon (de l'intérieur vers l'extérieur), L₁, L₂, L₃, le mur du
   système mur du côté extérieur ;
3. **Détail de la fixation** : montant coupé (rupture), platine, cordons à a·√2, chevilles
   (support béton) ou boulons (support acier), h_p, b_p, e_x, t_p, bras m, a, efforts V, N, M ;
   console au mur dans le système mur ;
4. **Section** : contour du catalogue, hachures de la paroi seulement (règle pair-impair sur
   tous les anneaux), Ø ou h × b, t, sens de P₁.

## 5 · L'écran

Barre d'outils (cas, ＋, copie, suppression, JSON, Note, nom), le switch de méthode avec une
ligne qui dit ce qu'il change, à gauche les dessins, à droite la carte RÉSULTAT (statut, η, élément
gouvernant, une barre par élément, masse indicative) et six onglets : **Garde-corps** (système,
forme, tronçons avec L / entraxe maxi / entraxes saisis, H, extrémités, tableau des montants et
de leurs longueurs tributaires), **Charges** (catégorie, q_h, P direct, q_v, Q, q_r, Q_r),
**Sections** (montant et lisse, section libre avec ses dimensions, BIBLIOTHÈQUE : familles,
masse maxi, « seulement ceux qui tiennent », bouton « le plus léger qui tient »),
**Remplissage** (barreaux, lisses, panneau, aucun), **Fixation** (type, soudure avec ses
dessins du module Soudures, platine, ancrage), **Détail** (tableau des éléments, des montants,
formules comme à la main, paramètres de la méthode, identification, textes de méthode).

Liaison état → widgets (`modules/liaison.py`) : le sélecteur de section AFFICHE la section
retenue par le bouton de la bibliothèque (contrôle du harnais et de la capture « biblio »).

## 6 · La note

Voir le document de référence `GARDE_CORPS.md` §7. Composée en taille normale puis « serrée »
si une colonne déborde ; `derniers_avertissements` vide sur les huit cas du test.

Version 3 (05/11/2026), à la demande du bureau (« trop d'explications », « la nomenclature »,
« charge répartie sur deux lignes », « qu'elle soit belle à voir ») :
- **les symboles sont composés** — `ndc_pdf/riche.py` traduit la notation des substitutions
  (« W_el,y », « P₁ », « √(…) », « t_p² ») vers le mini-LaTeX de `mathx` et compose les segments
  « $…$ » d'un texte courant à la taille et dans la couleur du texte ; `auto()` compose les
  symboles indicés d'une prose ordinaire (les textes de `methode.py`, les libellés du moteur) ;
  la notation « q_h » reste la source unique (écran, JSON, tests), la note en est une vue ;
- **l'ordre de lecture** — `formules.py` rend chaque bloc par GROUPE (charges, sollicitations,
  contraintes avec la limite et le taux sous chaque contrainte, flèche, taux) ; « subs » reste la
  liste à plat, dans le même ordre, pour l'écran et les tests ; un taux reprend le nombre
  IMPRIMÉ de sa contrainte (η_σ = 91,2/160 et non 91,25/160) ;
- **la mise en page** — hypothèses en lignes « libellé · valeur », charges en « symbole et
  valeur, nom dessous, origine en gris », le résultat de chaque formule À LA SUITE du calcul
  (le bureau a refusé la première version, qui calait tous les résultats à droite : « le égal
  doit se mettre à la suite du calcul ; on peut juste mettre les pourcentages sur la droite »),
  seuls les η calés à droite en couleur (`riche.formule_calee`, `cale="suite"` ou
  `"droite"`), pastille η par élément, barres de taux au récapitulatif, une légende d'une
  ligne ; la prose d'explication (sous-titres, notations, « platine rigide ») a disparu ;
  la ligne « Méthode » des hypothèses est devenue « Vérification » (à l'ELS … / à l'ELU selon
  l'Eurocode …) — la note ne dit plus « méthode du bureau » ; t_p est imprimé face à t_req ;
- **ce qui garde sa notation** : les cotes des dessins (« e_x = 100 », « b_p 150 »), écrites par
  le kit des schémas partagé avec l'écran SVG — à composer dans un lot ultérieur si le bureau le
  souhaite ; les parenthèses des formules composées sont des tracés, pas des glyphes (le texte
  extrait du PDF les perd : `tests/test_garde_corps_note.py` en tient compte).

## 7 · Architecture et lots

```
acier/garde_corps/
  charges.py        catégories Tab. 6.12, défauts, normalisation, phrase, origines
  geometrie.py      formes, tronçons, entraxes libres, montants et longueurs tributaires
  bibliotheque.py   sections standard (catalogue + stock) et libres, catalogue filtré
  membres.py        console, poutre, vérification de section, limites ELS / Eurocode, flèche
  fixation.py       cas de soudure (C6 / C3 / C8), platine boulonnée (boulons, chevilles, flexion)
  etat.py           cas, état, JSON, vérification complète, candidats, plus léger
  formules.py       substitutions testables (montant, console, lisse, barre, platine) + soudure
  methode.py        textes de méthode, références
  schemas.py        quatre vues sur le kit des assemblages
  note.py           note PDF deux pages
modules/garde_corps.py   l'écran (réécrit)
tests/test_garde_corps_{moteur,formules,schemas,ecran,note}.py
tools/captures_garde_corps.js
```

Lots livrés le 04/11/2026 : GC0 (cette étude), GC1 (moteur + tests), GC2 (schémas + écran),
GC3 (note), GC4 (captures, docs, batterie, commit).

Version 2, livrée le 05/11/2026 après la relecture du bureau (« présentation à 10 sur 100 ») :
GC5a moteur (types en menu, quatre charges nommées en cas distincts, main courante sur deux
appuis ou continue, portée des barreaux déduite, candidats par rôle, RHS par défaut), GC5b
schémas de dessin technique (six vues), GC5c écran (menu des types, cinq onglets, tableaux
cliquables), GC5d note (hypothèses en tête, 1 à 4 avec formules développées), GC5e docs,
captures, batterie. Le document de référence est `docs/acier/GARDE_CORPS.md`.

Version 3, livrée le 05/11/2026 : GC6 note composée (`ndc_pdf/riche.py`, formules par groupe,
charges symbole / nom / valeur), puis, après la seconde relecture (« le égal à la suite du
calcul », « d'autres cas courants », « est-ce que l'épaisseur de la platine est vérifiée ? »,
« 95 % des garde-corps ») : GC7a note (résultats à la suite, η seuls à droite, « vérification à
l'ELS »), GC7b fixation latérale en nez de dalle (moteur, soudure C8 longitudinale, vue de face
et coupe horizontale, écran, note), GC7c couverture (sans montant de mur à mur, rampant,
plateforme industrielle, câbles), GC7d vérification complète (cinq suites, huit états de
captures, docs, batterie). `fixation.py` et `schemas.py` ont grossi d'autant ; le découpage de
`schemas.py` (élévation / coupes / détails) est au plan.

## 8 · Points à trancher par le bureau (défauts retenus en attendant)

- **P-1 · Valeurs belges du Tableau 6.12.** L'ANB (NBN EN 1991-1-1 ANB) n'a pas pu être lue
  depuis la session (sites Buildwise / CSTC inaccessibles). Les catégories portent les valeurs
  recommandées de l'EN ; dès que le bureau confirme les valeurs belges, `charges.CATEGORIES`
  les reçoit (une ligne par catégorie) et la mention « valeur recommandée » devient « ANB ».
- **P-2 · Charge verticale q_v = 0,5 kN/m et pression q_r = 0,5 kN/m².** Valeurs d'usage du
  bureau, marquées comme telles ; la NBN B 03-004 §6 à confirmer.
- **P-3 · Flèche admissible L/100** (l'ancien module proposait L/200, que le montant courant ne
  tient pas). Modifiable (L/x ou mm), imprimée dans la note.
- **P-4 · Aire de cisaillement du catalogue** (EN 1993-1-1) au lieu de toute la paroi.
- **P-5 · Angle d'un L ou d'un U sur platine** : le moment résultant √(M₁² + M₂²) est appliqué
  aux chevilles dans le sens de b_p avec h_p = h (approximation du côté sûr pour un tube rond,
  légèrement optimiste pour un rectangulaire chargé selon b).
- **P-6 · Lisse continue sur plusieurs travées** : prise appuyée (qL²/8, côté sûr) par
  défaut ; depuis la version 2, l'option « poutre continue » applique les coefficients des
  tables (0,125 / 0,100 / 0,107 qL² pour 2 / 3 / ≥ 4 travées égales, flèche k_f·qL⁴/EI), la
  charge ponctuelle gardant ses valeurs sur deux appuis.
- **P-7 · Cas de charge distincts (version 2).** Les quatre charges (q_h, q_v, Q_r, q_r)
  agissent séparément : c'est la pratique des normes de garde-corps (charges de service qui ne
  se cumulent pas), demandée par le bureau et à confirmer contre la NBN B 03-004 §6 ; l'option
  « cumul » (q_h + q_v ensemble) reste disponible, du côté sûr.
- **P-8 · Sections par défaut (version 2)** : RHS 50x30x3 pour les montants et la main
  courante (RHS demandés par le bureau), Plat 30x8 pour les barreaux, RHS 50x25x2.5 formé à
  froid pour la lisse basse (non calculée), Plat 50x10 pour les consoles.
- **P-9 · Main courante entre deux murs (version 3).** Question du bureau : « une main courante
  scellée dans le mur à gauche et à droite, un garde-corps de 8 m, il faudrait que les 8 m du
  profilé résistent en flèche ? » Réponse : OUI si rien ne la porte entre les deux murs — c'est
  le garde-corps SANS MONTANT, la main courante est une poutre de mur à mur sur toute la
  longueur (un RHS 50x30x3 n'y tient déjà plus à 3,60 m : η = 244 %, flèche) ; NON dès qu'il y a
  des montants intermédiaires : la portée est l'entraxe, le mur compte pour un appui et reprend
  la demi-travée (l'outil le dit depuis la version 1 avec « fin : lisse ancrée dans le mur »),
  et l'option « poutre continue » applique les coefficients des tables. Le module choisit
  seul : deux bouts ancrés et une seule travée = sans montant.
- **P-10 · Fixation latérale (version 3).** Modèle retenu en 2.4 : couple d'ancrages z = e_x
  avec P/2 de part et d'autre, platine en console vers la colonne d'ancrages, soudure par deux
  cordons longitudinaux. À confirmer par le bureau : la part de P donnée à la rangée haute
  (Z = M/z + P/2 ; l'alternative M/z + P est plus sûre de 0,2 kN sur le cas courant) et la
  descente d_h par défaut (50 mm).
- **P-11 · Plateforme industrielle (version 3).** 0,3 kN/m d'après l'EN ISO 14122-3 (machines,
  accès), imprimé « à confirmer sur l'édition en vigueur » : la valeur est modifiable, et
  l'origine suit.

## 9 · Sources

EN 1991-1-1:2002 §6.4 et Tableau 6.12 (valeurs recommandées) ; NBN B 03-004 : garde-corps de
bâtiments (Q_kh,2 = 0,5 kN sur le remplissage, cité par le CCTB wallon) ; EN 1993-1-1 §6.2,
§6.2.6 ; EN 1993-1-8 Tab. 3.4, §4.5.3, §6.2.6.5, Tab. 6.6 ; EN 1992-4 §7.2.3 ; pratique du
bureau (σ_adm 160, τ_adm 90, k = 100 kg/(mm·cm)) ; ancien `modules/garde_corps.py`.
