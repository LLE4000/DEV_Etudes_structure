# Enrobage du béton — le moteur sans Streamlit

*Livré le 07/11/2026, relu le même jour. Moteur `beton/enrobage/` : bibliothèque standard seule, Python 3.12 et 3.13
(donc Pyodide 0.29.5). Extrait de l'écran Streamlit `modules/enrobage.py` (VERSION 2.0), qui reste la **référence** et
n'a pas été modifié ; la preuve de parité est `tests/test_enrobage_moteur.py`, la parité bout à bout (écran Streamlit
rejoué ↔ fragments HTML du site) est `tests/test_api_web_enrobage.py` § 8.*

> **Depuis le 10/11/2026 (Lot Q, docs/site/LOT_Q_QUATRE_MODULES.md § 4) : D1, D2 et D6 sont CORRIGÉS sur le site.**
> `beton/enrobage/donnees.py` lit ses distances au feu dans `beton/bibliotheque/feu.py` (la source unique des
> tableaux de l'EN 1992-1-2 § 5, recoupés sur deux rapports de sources) : poteaux dans la colonne μ_fi = 0,7, exposés
> sur plusieurs côtés (R90 : 350/53, 450/40** ; R240 : « non couvert par le tableau », aucune valeur inventée) ;
> voiles μ_fi = 0,7, une face (REI 180 : 210/50, REI 240 : 270/60) ; pour une dalle ou un voile, aucun étrier n'est
> plus retiré de a. Les avertissements rouges de D1, D2 et D6 sont remplacés par la source ; la parité avec
> `modules/enrobage.py` (`tests/test_enrobage_moteur.py`) exclut ces trois points, nommés « écarts voulus avec l'écran
> Streamlit depuis le 10/11/2026 », et garde le reste. Un bandeau renvoie au calcul complet : **Bibliothèque ›
> Enrobages › Calcul** (`bibliotheque-beton?rubrique=enrobages&onglet=calcul`, `api_web/biblio_enrobages.py`, moteur
> `beton/bibliotheque/enrobage.py`). La photo de gel `tests/reference_refonte/enrobage.json` est regelée avec cette
> raison. Le relevé ci-dessous reste celui de l'écran Streamlit (D3, D7 et les autres ne changent pas).
>
> **⚠ Défauts NON CONSERVATIFS de l'écran Streamlit, reproduits par le site (parité) et signalés à l'écran — à trancher
> par le bureau avant tout usage des valeurs en cause** (détail et sources au § 6 ; les valeurs « de la norme » sont
> celles relevées à l'extraction et à la relecture, **à confirmer dans les textes et l'ANB**) :
>
> - **D1 — poteaux au feu (tab. 5.2a)** : l'écran annonce μ_fi ≈ 0,7 mais ses distances a suivent la colonne μ_fi = 0,2
>   (et R240 porte des variantes absentes du tableau) : **a sous-estimé de 15 à 25 mm** (R90 : 31 mm à b ≥ 200 au lieu
>   de 53 mm à b ≥ 350). Message **rouge** sur le site dès que le feu d'un poteau est vérifié.
> - **D2 — voiles R180 / R240 (tab. 5.4)** : 45 / 55 mm au lieu de 50 / 60 mm (μ_fi = 0,7, une face) : **−5 mm**.
>   Message rouge sur le site.
> - **D3 — classe structurale proposée** : S1 pour X0 et S2 pour XC1, sans les conditions du tableau 4.3N (une poutre
>   ne peut pas descendre de trois classes) : c_min,dur sous-estimé (XC1 : 10 mm au lieu de 15 mm en S4 normatif).
>   Message sur le site dès que la classe est sous S4.
> - **D6 — Ø étrier de 8 mm par défaut retiré aussi pour une dalle ou un voile** : c_feu,min sous-estimé de 8 mm si l'on
>   oublie de mettre 0. Message sur le site.
> - **D7 — affichage arrondi au pair** (« 18 » pour 18,5 mm, « 28 » pour un c_nom de 28,5) : −0,5 mm affiché. Le site
>   affiche la valeur exacte.

## 1 · Ce qui est livré

| Pièce | Rôle |
|---|---|
| `beton/enrobage/donnees.py` | les tableaux de l'écran (classes d'exposition et leurs textes, classe structurale proposée, c_min,dur, distances à l'axe au feu), les listes et défauts des widgets, les textes d'aide, les captions et le guide « Comment choisir la classe d'exposition ? », recopiés mot pour mot |
| `beton/enrobage/entree.py` | l'élément, ses défauts, sa normalisation stricte, la saisie d'un champ avec les effets de bord des widgets de l'écran |
| `beton/enrobage/moteur.py` | `a_feu`, `c_durabilite`, `c_feu` (les fonctions de l'écran, arbres syntaxiques identiques), `calculer` (l'enchaînement de `show()`), les termes nommés avec leur règle, les avertissements, le sélecteur de la largeur au feu — et le format de sortie dans sa docstring |
| `api_web/enrobage.py` | le module du site (clé `enrobage`, établi `enrobage`) |
| `api_web/dessin_enrobage.py` | le dessin à l'échelle : détail d'angle (poutre, poteau), coupe de la face exposée (dalle, voile) |
| `site/web/css/modules/enrobage.css` | la feuille propre à l'établi (verdict en mm, décomposition, vue Tableaux, guide) |
| `tests/test_enrobage_moteur.py` | la parité avec l'écran (87 contrôles, 6 s) |
| `tests/test_api_web_enrobage.py` | ce que l'écran du site affiche = ce que le moteur calcule, et = ce que l'écran Streamlit d'origine affiche (bout à bout, 20 cas) ; saisies absurdes ; dessin (113 contrôles, 3 s) |

## 2 · L'API

```python
from beton.enrobage import moteur as M

M.nouvel_element(nom="Élément 1") -> dict    # les défauts de l'écran
M.normaliser(element) -> dict                # complété et contrôlé ; ValueError hors du domaine de l'écran
M.appliquer(element, cle, valeur) -> dict    # une saisie, avec les effets de bord des widgets ; ValueError si refusée
M.calculer(element) -> dict                  # durabilité, feu, valeur retenue, termes, avertissements
M.candidats_feu(element) -> list             # les largeurs du tableau feu, recalculées une à une
M.a_feu / M.c_durabilite / M.c_feu           # les trois fonctions de l'écran, telles quelles
```

**Entrée** (mm ; une clé absente ou `null` prend le défaut de l'écran) :

```json
{"nom": "E1", "type": "Poutre", "position": "Intérieur", "diam_barre": 20, "diam_etrier": 8, "delta_dev": 10,
 "classe_expo": "XC4", "classe_struct": "S4", "feu": true, "r": "R90", "b_min": 200}
```

`type` ∈ Poutre, Poteau, Dalle, Voile ; `position` ∈ Intérieur, Extérieur ; `diam_barre` entier de 6 à 60 ;
`diam_etrier` entier de 0 à 40 (les maximums sont un garde-fou du site, § 5) ; `delta_dev` ∈ 10, 5, 0 ; `classe_expo` ∈ X0, XC1, XC2/XC3, XC4, XD1/XS1, XD2/XS2, XD3/XS3 ; `classe_struct`
∈ S1…S6 (absente : la classe proposée pour l'exposition) ; `r` ∈ R30…R240 ; `b_min` : une largeur du tableau de la
poutre (5.5) ou du poteau (5.2a) pour la durée, `null` pour une dalle ou un voile.

**Sortie de `calculer`** (JSON strict ; les types sont ceux de l'écran : entiers pour la durabilité, flottants pour le
feu) — l'exemple ci-dessus :

```json
{"c_nom": 40, "gouvernant": "durabilite", "c_nom_dur": 40, "c_nom_feu": 37.0, "proches": false,
 "durabilite": {"classe_structurale": "S4", "classe_exposition": "XC4", "classe_structurale_proposee": "S4",
                "c_min_dur": 30, "c_min_b": 20, "plancher": 10, "delta_dur_gamma": 0, "delta_dur_st": 0,
                "delta_dur_add": 0, "c_min": 30, "terme_c_min": "c_min_dur", "delta_dev": 10, "c_nom": 40},
 "feu": {"r": "R90", "tableau": "5.5", "section": "§5.6", "a": 45, "b_min": 200,
         "note": "Poutre sur appuis simples, b_min=200 mm (tab. 5.5).",
         "variantes": [[150, 55], [200, 45], [300, 40], [400, 35]], "demi_barre": 10.0, "diam_etrier": 8,
         "c_brut": 27.0, "c_min": 27.0, "delta_dev": 10, "c_nom": 37.0, "couvert": false},
 "exposition": {"classe": "XC4", "titre": "Alternativement humide et sec", "…": "…"},
 "termes": [{"cle": "c_min_b", "symbole": "c_min,b", "valeur": 20, "unite": "mm",
             "regle": "EN 1992-1-1 §4.4.1.2(3), tab. 4.2 : adhérence, barre individuelle — c_min,b = Ø …"}, "…"],
 "avertissements": [{"code": "…", "message": "…", "grave": false}]}
```

`candidats_feu` rend, pour chaque largeur de la durée : `b_min`, `a`, `c_min_feu`, `c_nom_feu`, `c_nom` (retenu),
`gouvernant`, `retenue`, `optimale` (la plus petite largeur qui donne le plus petit enrobage retenu : au-delà, élargir
l'élément ne réduit plus l'enrobage).

## 3 · La méthode de l'écran (reproduite, pas réécrite)

1. **Durabilité** (EN 1992-1-1 §4.4.1) : c_min,dur = `C_MIN_DUR[classe structurale][exposition]` ; c_min,b = Ø de la
   barre principale ; c_min = max(c_min,b ; c_min,dur ; 10) — les Δc_dur,γ, Δc_dur,st, Δc_dur,add de l'éq. (4.2) sont
   laissés à 0 ; c_nom,dur = c_min + Δc_dev.
2. **Feu** (EN 1992-1-2, méthode tabulée), seulement si la case est cochée : a lu dans le tableau du type (dalle 5.8,
   voile 5.4, poteau 5.2a, poutre 5.5 — pour la poutre et le poteau, à la largeur choisie) ; c_feu,min = max(a − Ø/2 −
   Ø_étrier ; 0) (l'étrier est décompté : a est mesuré à l'axe de la barre principale, à l'intérieur du cadre) ;
   c_nom,feu = c_feu,min + Δc_dev. Feu non coché : c_nom,feu = 0.
3. **Retenu** : c_nom = max(c_nom,dur ; c_nom,feu) ; gouverné par la durabilité si c_nom,dur ≥ c_nom,feu (à égalité, la
   durabilité).
4. **Encadrés** : « Distance à l'axe couverte par la durabilité · feu non dimensionnant » si c_feu,min ≤ 0 ;
   « Durabilité et feu proches · vérifier le cas réel » si le feu est coché et |c_nom,dur − c_nom,feu| ≤ 2 mm (bornes
   comprises : 2 mm pile l'affiche, 2,5 mm non).
5. **Classe structurale proposée** : X0 → S1, XC1 → S2, XC2/XC3 et XC4 → S4, XD1/XS1 et XD2/XS2 → S5, XD3/XS3 → S6.

## 4 · D'où vient chaque grandeur

| Moteur | Écran `modules/enrobage.py` |
|---|---|
| `donnees.EXPOSITION_INFO`, `CLASSE_STRUCTURALE_DEFAULT`, `C_MIN_DUR`, `R_LEVELS`, `FEU_DALLE`, `FEU_VOILE`, `FEU_POTEAU`, `FEU_POUTRE` | les constantes du même nom (relues par l'arbre syntaxique du module) |
| `TYPES`, `POSITIONS`, `DELTAS_DEV`, `DIAM_*_DEFAUT`, `DIAM_*_MIN`, `R_DEFAUT`, aides, captions, guide | les arguments des widgets et des `st.markdown` / `st.caption` de `show()` (relus par le faux st et l'AppTest) |
| `durabilite.c_min_dur`, `c_min_b`, `c_min`, `c_nom` | `c_durabilite(classe_struct, classe_expo, diam_barre, delta_dev)` |
| `feu.a`, `feu.note` | `a_feu(type_element, r_level, b_choice)` |
| `feu.c_min`, `feu.c_nom` | `c_feu(a_val, diam_barre, diam_etrier, delta_dev)` |
| `c_nom`, `gouvernant`, `proches`, `feu.couvert` | `c_final`, `gouv`, la condition de l'encadré warn, `cmin_f <= 0` dans `show()` |
| `durabilite.terme_c_min`, `termes[].regle`, `candidats_feu`, avertissements autres que les deux encadrés | nommés par le moteur pour le site (aucun calcul nouveau : un choix parmi les termes calculés, des textes) |

## 5 · Effets de bord des widgets, et les écarts voulus du site

Les widgets « Classe structurale » et « Largeur minimale » n'ont pas de clé : Streamlit calcule leur identité avec
leurs paramètres (`index`, `options`). D'où, prouvé par l'AppTest pas à pas :

- changer d'exposition **remet** la classe structurale à la classe proposée quand celle-ci change (X0 → XC1 : S1 → S2 ;
  une S5 choisie en XC1 devient S4 en XC2/XC3), et la **garde** quand elle est la même (S6 choisie en XC2/XC3 reste S6
  en XC4) — `entree.appliquer` fait de même ;
- changer de type ou de durée **remet** la largeur à la première variante quand la liste change, et la garde quand
  elle est la même (poteau R30 → R60, mêmes variantes) — idem.

**Écarts voulus** (le calcul est inchangé, seul le confort de saisie diffère) :

- décocher puis recocher le feu : l'écran oublie la durée et la largeur (widgets effacés, retour à R60 et à la première
  variante) ; le site les garde. Le test le vérifie des deux côtés (il rougit si l'écran change) ;
- la durée est un menu (l'écran : un curseur) ; la largeur est un menu ET un sélecteur (✓ retenue, ★ optimale) ;
- une saisie que l'écran ne laisse pas faire (Ø non entier, sous 6 mm, Ø d'étrier négatif, valeur hors liste) est
  **refusée avec un message**, l'élément reste intact ;
- **garde-fou** (relecture du 07/11/2026) : l'écran n'a pas de maximum — un Ø de 10⁹ mm s'y calcule, et un Ø de 10⁴⁰⁰
  venu d'un fichier faisait déborder les flottants (OverflowError). Le site refuse avec un message un Ø au-delà de
  **60 mm** (la plus grosse barre, Ø40, Ø50 exceptionnel, et le Ø équivalent d'un paquet, Ø_n ≤ 55 mm, EN 1992-1-1
  §8.9.1) et un Ø d'étrier au-delà de **40 mm** — à la saisie (le navigateur, `data-max`, puis le moteur) comme dans un
  fichier `.stab` ; la case du feu n'accepte que vrai ou faux (le texte « false » était lu VRAI) ; un nom d'élément a
  80 caractères au plus ; un rang d'élément illisible est refusé ;
- les valeurs sont affichées **exactes** (18,5 mm), là où l'écran arrondit au mm (voir D7) ;
- des **garde-fous** sont affichés (sans rien changer au calcul) : D1, D2, D3, D6 ci-dessous, élément extérieur classé
  X0 ou XC1, béton coulé contre le sol (en XC2/XC3, au conditionnel : « si le béton est coulé contre le sol… »,
  §4.4.1.3(4) non vérifié).

## 6 · Défauts trouvés dans l'écran Streamlit (reproduits, écrits, non corrigés)

Les valeurs « de la norme » citées ci-dessous sont celles de l'EN 1992-1-1 / EN 1992-1-2 telles que relevées à
l'extraction ; **elles sont à confirmer par le bureau dans les textes (et l'ANB)** avant toute correction de
`donnees.py` — une correction changerait la parité et se fera dans l'écran ET le moteur.

| | Défaut | Effet |
|---|---|---|
| **D1** | **Poteaux (tab. 5.2a)** : la caption annonce μ_fi ≈ 0,7, mais les distances a de `FEU_POTEAU` suivent la colonne μ_fi = 0,2 (R90 : 200/31, 300/25 ; R120 : 250/40, 350/35 ; R180 : 350/45), plus des variantes absentes du tableau (R240 : 400/50 et 450/40, alors que μ_fi = 0,2 donne déjà 350/61). Pour μ_fi = 0,7, le tableau donne par exemple R60 : 250/46, 350/40 ; R90 : 350/53, 450/40 | **non conservatif probable** (a sous-estimé de 15 à 25 mm) ; avertissement **rouge** sur le site dès que le feu d'un poteau est vérifié |
| **D2** | **Voiles (tab. 5.4)** : R180 = 45 et R240 = 55 mm ; la colonne μ_fi = 0,7, une face exposée, semble donner 50 et 60 mm (45 / 55 sont ceux de μ_fi = 0,35, deux faces) | **non conservatif probable** de 5 mm ; avertissement rouge sur le site (R180, R240) |
| **D3** | **Classe structurale proposée** : S1 pour X0 et S2 pour XC1, soit trois et deux classes sous S4 (« référence 50 ans », dit l'aide) sans demander les conditions du tableau 4.3N | **non conservatif** par défaut (XC1 : 10 mm au lieu de 20) ; message sur le site dès que la classe est sous S4 |
| **D4** | **C_MIN_DUR**, colonnes X0 et XC1 : S3/XC1 = 15 (tab. 4.4N : 10) ; S4 = 15 / 20 (10 / 15) ; S5 = 20 / 25 (15 / 20) ; S6 = 25 / 30 (20 / 25) | côté sûr (+5 mm) ; documenté seulement |
| **D5** | **« Position dans l'ouvrage »** n'entre dans aucun calcul | trompeur ; le site le dit sous le champ, et prévient si un élément extérieur est classé X0 ou XC1 |
| **D6** | **Ø étrier** (8 mm par défaut) décompté aussi pour une dalle ou un voile, où il n'y a souvent pas d'étrier | non conservatif si on oublie de mettre 0 (c_feu,min sous-estimé de 8 mm) ; message sur le site (feu vérifié, dalle ou voile, Ø étrier > 0) |
| **D7** | **Affichage** au mm par `:.0f` (arrondi au pair) : 18,5 s'affiche « 18 », c_nom 28,5 « 28 » ; la substitution « a − Ø/2 − Ø_étrier = 40 − 12 − 8 = 20 » écrit Ø/2 = 12,5 « 12 » et ne se vérifie pas | non conservatif de 0,5 mm à l'affichage ; le site affiche les valeurs exactes |
| **D8** | **Δc_dev ajouté au feu** : l'EN 1992-1-2 donne a en valeur nominale (sans tolérance à ajouter, à confirmer) | côté sûr (+Δc_dev) |
| **D9** | **c_min,b = Ø de la barre principale** appliqué à l'enrobage mesuré au nu de l'étrier (celui du feu) | côté sûr (la condition d'adhérence de la barre principale serait c_nom + Ø_étrier ≥ Ø + Δc_dev) |
| **D10** | le **plancher de 10 mm** est sans effet : c_min,dur ≥ 10 dans tout le tableau (mutation équivalente, prouvée) | aucun |
| **D11** | décocher le feu efface la durée et la largeur choisies (§ 5) ; changer d'exposition remet en silence une classe structurale choisie | ergonomie ; le site garde la durée, et explique la reprise de la classe proposée |

## 7 · Ce qui n'est pas couvert (ni par l'écran, ni par le site)

Classes XF (gel) et XA (chimique) ; Δc_dur,γ, Δc_dur,st, Δc_dur,add (fixés à 0) ; barres en paquet (c_min,b = Ø_n) et
granulats de plus de 32 mm (+5 mm, tab. 4.2) ; béton coulé contre le sol (c_nom ≥ 40 / 75 mm, §4.4.1.3(4) — signalé
en XC2/XC3) ; surfaces irrégulières ; précontrainte ; Δc_dev autre que 10, 5 ou 0 mm. Au feu : a_sd des barres d'angle
des poutres à un lit (+10 mm), épaisseurs et largeurs minimales (b_min, h_s de la dalle, épaisseur du voile — non
vérifiées : la largeur n'est qu'un choix de ligne du tableau), poutres continues (tab. 5.6), dalles portant dans deux
directions ou continues, conditions de la méthode A des poteaux (l₀,fi ≤ 3 m, e ≤ 0,15 b, A_s < 4 % A_c), μ_fi autre
que 0,7. Pas de vérification d'un enrobage prévu : le module est une **consultation** (niveau « i » sur le site). La
note PDF est hors périmètre (l'écran n'en a pas).

## 8 · La preuve de parité (`tests/test_enrobage_moteur.py`, 87 contrôles ; bout à bout : `tests/test_api_web_enrobage.py` § 8)

1. Les huit tableaux relus dans le source du module (arbre syntaxique) ; `a_feu`, `c_durabilite`, `c_feu` identiques
   au module, mot pour mot (arbres syntaxiques égaux, docstrings comprises).
2. Les trois fonctions comparées au module, **même valeur et même type** (int ≠ float) : 108 appels de `a_feu`, 7 182 de
   `c_durabilite`, 62 370 de `c_feu` (toutes les distances a des tableaux, Ø 6 à 60, Ø d'étrier 0 à 20).
3. **11 066 éléments** rejoués dans `show()` avec un faux `st` (qui répond aux widgets et enregistre ce qui
   s'affiche) et des espions sur les trois fonctions : widgets (libellés, options formatées, index par défaut, aides,
   bornes), encadrés (titre, valeur, état), chaque ligne de markdown, chaque caption, et les flottants exacts passés et
   rendus — tous identiques au moteur ; plus 8 bornes exactes (égalité, écarts de ±2 et 2,5 mm, c_feu,min nul pile et
   négatif, deux demi-millimètres arrondis au pair par l'écran).
4. Couverture : 78 situations atteintes (chaque type × durée, chaque variante, chaque terme gouvernant c_min,
   gouvernant feu et durabilité, bornes…).
5. **AppTest** (le vrai écran Streamlit, 37 pas en trois séquences) : à chaque pas, les valeurs des widgets et les
   encadrés de l'écran = l'élément et le calcul du moteur après les mêmes saisies ; plus l'écart voulu du § 5.
6. Cinq éléments recalculés à la main ; le sélecteur sur deux cas.
7. **14 mutations** du moteur (une valeur de tableau 5.5 ou 4.4N, la borne à 0, Ø/2 entier, Δc_dev oublié, ≥ devenu >,
   ≤ 2 devenu < 2, feu calculé non coché, largeur ignorée, effets de bord retirés ou forcés) : **toutes rougissent**.
8. Format : défauts, idempotence, 21 refus (ValueError, dont les garde-fous Ø > 60, Ø de 10⁴⁰⁰, étrier > 40), JSON
   strict, règles nommées.
9. Pureté : le moteur calcule sous un `sys.meta_path` qui refuse streamlit, numpy, pandas…, en Python 3.12 et 3.13,
   avec le même résultat.
10. **Bout à bout** (`tests/test_api_web_enrobage.py` § 8, ajouté à la relecture) : 20 cas (chaque type, défaut, feu
    gouvernant, égalité, écart 2 et 2,5 mm, c_feu,min nul et négatif, demi-millimètres, Δc_dev 0 / 5, Ø6 et les bornes
    Ø60 / étrier 40, extérieur) : l'écran Streamlit d'origine est rejoué avec un faux `st` (encadrés, markdown,
    captions, menus), le site par `routeur.traiter` ; verdict, gouvernant, décomposition, lignes, fenêtres de calcul,
    substitution du feu, caption du tableau, encadrés « couverte » et « proches », options d'exposition sont égaux à
    l'arrondi de l'écran près (`:.0f`). Le contrôle rougit si le site affiche c_min,dur faux d'1 mm.

## 9 · Le module du site : ce que l'écran Streamlit offre, et où

| Écran Streamlit | Site |
|---|---|
| Type d'élément, position | section 1 (segments) ; la position porte la mention « n'entre dans aucun calcul » |
| Ø max armatures principales, Ø étrier / cadre (aide), Δc_dev (aide) — les « Paramètres généraux » | section 2 « Armatures et tolérance » (Δc_dev s'ajoute aux deux enrobages : il n'est plus rangé sous « Durabilité ») |
| Classe d'exposition (« X — titre »), encadré de la classe, Mécanisme, Quand la choisir, Exemples | section 3 |
| Classe structurale (classe proposée, aide) | section 3 ; chaque classe affiche son c_min,dur, la proposée est marquée |
| Résistance au feu (case), durée, largeur minimale (aide), caption μ_fi | section 4 ; la largeur aussi par le sélecteur de la ligne Feu |
| caption sous « B — Feu » (l'hypothèse du tableau : « Poutre sur appuis simples, b_min = … (tab. 5.5) »…) | section 4, en gras sous la largeur, et dans la fenêtre « feu » (μ_fi n'apparaît dans les hypothèses que pour un poteau ou un voile) |
| A — Durabilité et ses cinq lignes | ligne « Durabilité », décomposition sous le verdict, fenêtre de calcul « durabilité » (formules substituées, règles) |
| B — Feu, sa caption, ses quatre lignes ; « non vérifié » et l'invite | ligne « Feu · R… », fenêtre « feu » (avec la caption de l'écran) ; message d'invite |
| « Distance à l'axe couverte par la durabilité » | message |
| C — Enrobage nominal retenu, « Valeur gouvernée par … » | verdict (c_nom en mm, gouvernant), fenêtre « retenu » |
| « Durabilité et feu proches » | message |
| Guide « Comment choisir la classe d'exposition ? », règle pratique, plusieurs classes | section 5 (repliée) |
| — | dessin à l'échelle (détail d'angle ou coupe de la face exposée : c_nom, c_min, bande Δc_dev, a et a_feu) ; vue « Tableaux » (4.4N et tableau feu du type, case retenue marquée) ; plusieurs éléments par note |

## 10 · Relecture du 07/11/2026 (relecteur indépendant)

- **Chiffres** : 20 cas bout à bout, écran Streamlit d'origine ↔ site : aucun écart (hors D7, voulu).
- **Saisies absurdes** (corrigé) : Ø 10⁹ et 99 999 mm, étrier 10⁶ mm étaient acceptés et calculés ; un `.stab` portant
  un Ø de 10⁴⁰⁰ rendait « OverflowError » ; la case du feu lisait le texte « false » comme VRAI (`bool("false")`) ; un
  nom de 5 000 caractères passait ; un rang d'élément illisible rendait l'exception brute de `int()`. Tous refusés
  désormais avec un message (§ 5), et testés.
- **Dessin** (corrigé) : dans le détail d'angle, quand « c_nom = … » ne tenait pas entre ses traits, il se posait à
  droite… sur la ligne d'attache de l'axe de la barre, qui le barrait (cas par défaut). Il se pose désormais à gauche ;
  un test rougit s'il est de nouveau barré. La coupe d'une dalle ou d'un voile, large et basse, s'affichait à ~10 px de
  texte sur PC (illisible sur tablette) : refaite (texte rapporté au dessin, cotes nommées courtes « a = 79 »,
  décomposition « a = c_nom + Ø_e + Ø/2 = … » sous le parement).
- **Écran** (corrigé) : la fenêtre « durabilité » listait deux règles « c_min » (le plancher portait le symbole c_min) ;
  les colonnes Formule et Limite y répétaient la même référence (« tab. 4.4N » deux fois) — Limite porte maintenant la
  référence de la norme (EC2 tab. 4.2, éq. (4.2)…), Formule le calcul ; μ_fi n'apparaît plus dans les hypothèses d'une
  poutre ou d'une dalle (il ne concerne que les poteaux et les voiles) ; la caption de l'écran qui dit l'hypothèse du
  tableau feu (« poutre sur appuis simples », « dalle portant sur 1 direction », « voile exposé sur 1 face ») n'était
  que dans la fenêtre de calcul : elle est aussi dans la section 4 ; Δc_dev rejoint la section 2 ; sur tablette, la
  liste des éléments n'écrit plus le type quand le nom le dit déjà, et les quatre types tiennent dans leur ligne.
