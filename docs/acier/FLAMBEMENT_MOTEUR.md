# Flambement par flexion — moteur `acier/flambement` et module du site

> **NOUVEAU CALCUL, À VALIDER PAR LE BUREAU.** L'écran Streamlit `modules/flambement.py` n'a aucun calcul : il
> affiche « Page en construction ». Ce moteur n'est donc pas extrait d'un écran existant, il est **écrit** ici,
> d'après l'EN 1993-1-1 §6.3.1 et l'ANB belge. Le site le dit en tête de chaque résultat et dans chaque fenêtre de
> calcul.

> **DÉFAUT NON CONSERVATIF trouvé à la relecture (07/11/2026) et CORRIGÉ** : la première version prenait la largeur
> des parois des tubes rectangulaires **formés à froid** dans le catalogue, qui y écrit la largeur plate nominale
> b − 2·r_o (r_o = 2t à 3t selon l'EN 10219-2, soit b − 4t à b − 6t). L'EN 1993-1-5 §4.4(2) prescrit **b − 3t** pour
> les parois des RHS, sans distinguer chaud et froid (et la tolérance de l'EN 10219-2 sur r_o, 1,6t à 2,4t pour
> t ≤ 6 mm, rend la largeur plate réelle incertaine). Effet : **63 tubes** (toutes nuances) classés une classe trop bas
> et une aire efficace **jusqu'à 9 % trop forte** (SHS 350×350×10 formé à froid en S460 : classe 3 au lieu de 4).
> Le moteur prend désormais b − 3t pour tous les tubes rectangulaires (identique au catalogue pour les finis à
> chaud) ; un contrôle à la main et une mutation le gardent. **L'écran Streamlit n'est pas en cause** (il n'a pas
> de calcul de flambement) ; le catalogue partagé reste inchangé — signalé comme besoin commun.

Fichiers : `acier/flambement/` (`normes.py`, `section.py`, `entree.py`, `moteur.py`), le module du site
`api_web/flambement.py`, ses dessins `api_web/dessin_flambement.py`, sa feuille `site/web/css/modules/flambement.css`,
les tests `tests/test_flambement_moteur.py` et `tests/test_api_web_flambement.py`.

## 1. Ce qui est calculé

Une note contient **une ou plusieurs barres**. Chaque barre : un profilé du catalogue (`acier/donnees/profiles.json`,
source unique), une nuance (S235, S275, S355, S460), un effort de compression N_Ed (kN), une longueur L (m) et, **pour
chacun des deux axes**, ses appuis. Trois vérifications, la plus chargée gouverne :

| Vérification | Formule | Référence |
|---|---|---|
| Flambement autour de y-y (u-u) | N_Ed ≤ N_b,y,Rd = χ_y·A·f_y / γ_M1 | §6.3.1.1 éq. 6.46–6.47 |
| Flambement autour de z-z (v-v) | N_Ed ≤ N_b,z,Rd = χ_z·A·f_y / γ_M1 | idem |
| Résistance de la section | N_Ed ≤ N_c,Rd = A·f_y / γ_M0 | §6.2.4 éq. 6.10 |

En classe 4, A_eff remplace A partout (éq. 6.11 et 6.48, λ̄ par éq. 6.51).

### D'où vient chaque grandeur

| Grandeur | Source |
|---|---|
| A, I_y, I_z, i_y, i_z, I_u, I_v, i_u, i_v, masse | catalogue `acier/donnees/profiles.json` (propriétés recalculées sur le contour exact, voir `docs/acier/PROFILES.md`) |
| c/t des parois | catalogue (`cf_tf` : semelle, grande aile ; `cw_tw` : âme, petite aile), définitions du Tableau 5.2 ; **tubes rectangulaires : c = b − 3t et h − 3t** (EN 1993-1-5 §4.4(2)), finis à chaud comme formés à froid — voir l'encadré en tête |
| f_y | **Tableau 3.1** de l'EN 1993-1-1 selon l'épaisseur nominale la plus forte (t_f, t_w ou t) : t ≤ 40 mm → 235 / 275 / 355 / 460 ; 40 < t ≤ 80 mm → 215 / 255 / 335 / 430 N/mm². Creux formés à froid (EN 10219) : t ≤ 40 mm seulement. Au-delà : **refusé** (hors du tableau). |
| E | 210 000 N/mm² (§3.2.6) |
| ε | √(235 / f_y) |
| Classe en compression pure | Tableau 5.2 : parois internes 33 / 38 / 42·ε, en console 9 / 10 / 14·ε, tubes ronds 50 / 70 / 90·ε², cornières h/t ≤ 15·ε **et** (b + h)/2t ≤ 11,5·ε (classe 3, sinon 4) |
| A_eff (classe 4) | EN 1993-1-5 §4.4, compression uniforme ψ = 1 : λ̄_p = (b̄/t)/(28,4·ε·√k_σ), k_σ = 4 (interne), 0,43 (console) ; ρ = (λ̄_p − 0,22)/λ̄_p² au-delà de 0,673 (interne), (λ̄_p − 0,188)/λ̄_p² au-delà de 0,748 (console) ; A_eff = A − Σ n·(1 − ρ)·c·t. **Cornières** : b̄ = h (la grande aile) pour les deux ailes, comme le prescrit §4.4(2) pour les cornières à ailes égales et inégales. |
| L_cr | par axe : cas types (k = 1 ; 0,7 ; 0,5 ; 1 ; 2 — valeurs **théoriques** d'Euler), coefficient k saisi, longueur saisie ; entre deux articulations, n maintiens équidistants donnent L_cr = L / (n + 1) |
| N_cr | π²·E·I / L_cr² |
| λ̄ | √(A·f_y / N_cr) (et, en contrôle dans la fenêtre de calcul, L_cr / (i·λ₁) avec λ₁ = 93,9·ε) |
| Courbe, α | Tableau 6.2 (choix), Tableau 6.1 (α = 0,13 ; 0,21 ; 0,34 ; 0,49 ; 0,76) — voir §2 |
| φ, χ | φ = 0,5·[1 + α·(λ̄ − 0,2) + λ̄²] ; χ = 1 / (φ + √(φ² − λ̄²)) ≤ 1 (§6.3.1.2) |
| γ_M0, γ_M1 | 1,0 et 1,0 (ANB belge) ; modifiables entre 1,0 et 1,5 |
| Domaine des saisies | 0 ≤ N_Ed ≤ 10⁶ kN ; 0,01 ≤ L ≤ 100 m ; 0,01 ≤ k ≤ 10 ; 0,001 ≤ L_cr ≤ 1 000 m ; 0 à 9 maintiens. Au-delà : **refusé avec un message** (avant la relecture, L = 10²⁰⁰ m ou k = 10²⁰⁰ faisaient déborder le calcul, L = 10⁻³⁰⁰ m divisait par zéro, et une note .stab avec N = 10³⁰⁸ plantait l'affichage) |

### 2. Le choix de la courbe (Tableau 6.2)

| Profilé | Condition | y-y | z-z | S460 |
|---|---|---|---|---|
| Laminés en I et H (IPE, HE, HD, HL, HP, IPN) | h/b > 1,2 ; t_f ≤ 40 mm | a | b | a0 / a0 |
| | h/b > 1,2 ; 40 < t_f ≤ 100 mm | b | c | a / a |
| | h/b ≤ 1,2 ; t_f ≤ 100 mm | b | c | a / a |
| | h/b ≤ 1,2 ; t_f > 100 mm | d | d | c / c |
| Creux finis à chaud (CHS, SHS, RHS — EN 10210) | | a | a | a0 / a0 |
| Creux formés à froid (EN 10219) | | c | c | c / c |
| U (UPE, UPN) | | c | c | c / c |
| Cornières (L, LI) | autour de u-u et v-v | b | b | b / b |

h/b = 1,2 exactement relève de « h/b ≤ 1,2 ». La courbe d n'est jamais atteinte avec le catalogue : les laminés de
t_f > 100 mm sont hors du Tableau 3.1 (t > 80 mm), donc refusés avant.

**Cornières** : elles flambent autour de leurs axes **principaux**. L'axe « y » du module est l'axe fort u-u, l'axe
« z » l'axe faible v-v (I_u, I_v, i_u, i_v du catalogue, angle α du catalogue sur le dessin).

## 3. La preuve

`/usr/bin/python3.12 tests/test_flambement_moteur.py` — **114 contrôles, 0 échec** (07/11/2026, après relecture) :

1. **La norme** : Tableau 3.1, ε, λ₁ = 93,9·ε, Tableau 6.1, chaque ligne du Tableau 6.2, les limites du Tableau 5.2
   (bornes incluses), les deux formules de ρ.
2. **Valeurs publiées** : les coefficients de réduction χ(λ̄) du **tableau 5.5.2 de l'ENV 1993-1-1:1992**
   (« Coefficients de réduction χ », courbes a, b, c, d), repris tels quels par les manuels (ex. b : 0,9641 ; 0,9261 ;
   0,8842 ; 0,8371 ; 0,7837 ; 0,7245 ; 0,6612 ; 0,5970 pour λ̄ = 0,3 à 1,0), et la courbe a0 (α = 0,13 : 0,7253 à
   λ̄ = 1,0) — 49 valeurs à 10⁻⁴. *Les valeurs de a0 viennent des tableaux usuels qui l'ajoutent ; leur source n'a pas
   été relue hors ligne : à recontrôler par le bureau.*
3. **À la main**, sans le moteur : HEA 200 S235 bi-articulé de 4 m (N_cr, λ̄, φ, χ, N_b,Rd sur les deux axes ;
   χ_z = 0,628, N_b,z,Rd = 794,0 kN, η = 63 %) ; λ̄ par L_cr/(i·λ₁) ; IPE 400 S355 de classe 4 (âme c/t = 38,5 >
   42·ε = 34,2, ρ = 0,884, A_eff recalculée) ; cornière L 100×100×10 (axes u, v) ; L 100×65×7 S355 de classe 4
   (b̄ = h) ; SHS 350×350×10 formé à froid S460 (paroi b − 3t = 320 mm, classe 4, A_eff recalculée ; le catalogue
   l'aurait dit de classe 3) ; RHS fini à chaud (b − 3t = la largeur du catalogue) ; longueurs de flambement ; γ_M0
   et γ_M1 ; f_y entre 40 et 80 mm (HL 920×588).
4. **Parité** avec la seule expression de flambement que l'application Streamlit porte déjà :
   `ec3_buckling_chi` de `modules/choix_profile.py` (option « Vérifier flambement »), chargée telle quelle avec un faux
   `streamlit`. Sur **632 axes** tirés au hasard dans le catalogue (nuances, longueurs, appuis, maintiens), χ, λ̄,
   N_cr et N_b,Rd sont **égaux au bit près** pour les 492 axes de classe 1 à 3 (le moteur écrit les deux lignes de
   φ et χ mot pour mot, `normes.phi_chi`), et à 10⁻¹² en classe 4 (l'écran ne connaît pas A_eff : on la lui passe).
   Le menu des courbes de l'écran (« a0 (0.13) » … « d (0.76) ») est le Tableau 6.1 du moteur.
5. **Domaine** : saisies refusées avec un message (y compris les valeurs démesurées), le calcul aboutit aux bornes
   du domaine sans débordement, hors du Tableau 3.1, tubes ronds de classe 4, couverture (courbes
   a0, a, b, c ; classes 1 à 4 ; f_y réduit), tri et « plus léger qui tient » des candidats, avertissements.
6. **Mutations** : 18 altérations plausibles (α, φ, χ ≤ 1, seuil h/b, S460, axes permutés, f_y épais, ρ, N_cr, k, L/n,
   b̄ des cornières, largeur plate des tubes formés à froid, γ_M1, limite de classe, critère des cornières) font chacune
   rougir au moins un contrôle.
7. **Pureté** : le moteur se charge et calcule sous Python 3.12 et 3.13 (celui de Pyodide) dans un processus où
   streamlit, pandas, matplotlib, scipy, reportlab, PIL et fitz sont refusés.

`python3.12 tests/test_api_web_flambement.py` — **183 contrôles, 0 échec** : l'écran du site affiche ce que le moteur
calcule, et surtout **les chiffres AFFICHÉS** (N_b,Rd et N_c,Rd, λ̄, χ, courbe, f_y, classe, taux du résumé) de
**23 barres variées** — chaque famille (I/H laminés, U, cornières égales et inégales, tubes ronds, carrés et
rectangulaires finis à chaud et formés à froid), chaque nuance, classes 1 à 4, courbes a0 à c, k = 0,5 / 0,7 / 1 / 2,
une barre très courte (η ≈ 0), une très élancée (η = 6 639 %), une pile à la limite (η = 99,9997 % : vert foncé, 100 %,
« Conforme, attention ») — sont égaux à un **calcul refait à la main dans le test, sans le moteur** (catalogue brut et
tableaux de la norme réécrits), et χ, λ̄ à ceux d'`ec3_buckling_chi` de l'écran Streamlit avec la même aire, la même
courbe et le même f_y. Saisies absurdes refusées proprement ; dessins lisibles d'une barre de 1 cm à 100 m.

## 4. Ce qui n'est PAS couvert

- le flambement par **torsion** et par **flexion-torsion** (§6.3.1.4) : il peut gouverner pour les U, les cornières et
  les sections ouvertes courtes ; le module le **dit** sur les U et les cornières ;
- l'**excentricité des attaches** des cornières (annexe BB.1.2, λ̄_eff) : dit sur les cornières ;
- le **déversement** (§6.3.2) et la **flexion composée** (§6.3.3) : effort normal seul ;
- les **tubes ronds de classe 4** (EN 1993-1-6) : refusés ;
- le moment dû au **décalage e_N** du centre de gravité efficace d'une section de classe 4 non bi-symétrique (U,
  cornière, §6.2.2.5(4)) : dit quand une paroi est réellement réduite ;
- les **sections reconstituées soudées** (PRS, caissons : autres lignes du Tableau 6.2), les barres composées (§6.4),
  les barres à inertie variable ;
- l'option du §6.3.1.2(4) qui permet de **négliger** le flambement si N_Ed/N_cr ≤ 0,04 : non employée (côté sûr ; χ
  vaut de toute façon 1 pour λ̄ ≤ 0,2, ce que le résultat signale) ;
- la note PDF (hors périmètre du site, bouton grisé).

## 5. Points à trancher par le bureau

- **P-1 · f_y** : le module prend le **Tableau 3.1** (deux paliers : 40 et 80 mm). Si l'ANB belge impose la **norme
  de produit** (EN 10025-2 : paliers 16, 40, 63, 80 mm), f_y serait plus faible entre 16 et 40 mm (S355 : 345 au lieu
  de 355 N/mm²) — le Tableau 3.1 serait alors **non conservatif** de 3 % sur ces épaisseurs. Le texte de l'ANB n'a pas
  été consulté.
- **P-2 · coefficients d'appui** : les cas types emploient les valeurs **théoriques** d'Euler (encastrée–encastrée
  k = 0,5 ; encastrée–articulée 0,7). Un encastrement parfait n'existe pas : la pratique retient souvent 0,65 à 0,7 et
  0,8 à 0,85. Le libellé affiche k, l'aide le rappelle, le coefficient de pratique se saisit (« Coefficient k saisi ») ;
  le bureau peut préférer que les cas types portent les valeurs de pratique.
- **P-3 · cornières** : u-u et v-v, b̄ = h pour les deux ailes en classe 4. Une cornière attachée par une aile
  relève de l'annexe BB.1.2, non couverte.
- **P-4 · paroi des tubes rectangulaires** (relecture du 07/11/2026) : b − 3t pour tous, comme l'écrit l'EN 1993-1-5
  §4.4(2). C'est le côté sûr ; si le bureau admet la largeur plate réelle des formés à froid (b − 2·r_o), il faudra le
  décider en connaissant r_o (tolérance de l'EN 10219-2), et le catalogue partagé devra dire laquelle il écrit.

## 6. Défauts trouvés dans l'écran Streamlit

`modules/flambement.py` n'a pas de calcul. Dans la seule expression de flambement existante,
`modules/choix_profile.py` (option « Vérifier flambement » du choix de profilé, qui n'est pas ce module mais que le
moteur reproduit au bit près pour la prouver), quatre défauts, **non corrigés** (l'écran Streamlit reste inchangé) :

1. **une seule courbe pour les deux axes**, choisie à la main (b par défaut), au lieu du Tableau 6.2 par axe : un
   HEA en flambement autour de z-z (courbe c) est calculé en courbe b — **non conservatif** ;
2. **pas de classe 4** : A au lieu de A_eff (un IPE 400 en S355 est surestimé de 4 %) — **non conservatif** ;
3. **f_y = la nuance**, sans réduction au-delà de 40 mm d'épaisseur — **non conservatif** pour les profilés épais ;
4. **γ_M1 admis dès 0,8** (min_value = 0.8) sous la valeur de l'ANB (1,0) ; et un axe de L_cr nulle est
   **silencieusement ignoré** (`ec3_buckling_chi` rend None) au lieu d'être refusé.

Le module du site refuse ces cas avec un message (γ < 1, L ≤ 0, L_cr ≤ 0) et applique le Tableau 6.2, la classe 4
et le Tableau 3.1.

## 7. L'écran du site

- **Données** (sections numérotées, repliables) : 1 Barre (nom, famille, profilé, nuance, f_y et classe rappelés),
  2 Effort et longueur, 3 et 4 Flambement autour de chaque axe (appuis ; maintiens, coefficient ou longueur selon le
  cas ; L_cr et courbe rappelées), 5 Méthode (γ_M0, γ_M1). Plusieurs barres par note (ajouter, copier, supprimer).
- **Résultat** : l'avertissement « Nouveau calcul EN 1993-1-1 §6.3.1, à valider par le bureau » en tête, le verdict
  (taux, vérification gouvernante, N_Ed ≤ N_Rd), une ligne par vérification (code couleur du bureau), le profilé sur
  la ligne « Section » ouvre le **sélecteur** (trié par masse, ✓ retenu, ★ le plus léger qui tient, familles en
  puces, creux formés à froid), les avertissements du moteur, la masse de la barre.
- **Fenêtre de calcul** : hypothèses, tableau des trois vérifications, puis chaque formule **substituée** (f_y, ε,
  c/t et classe de chaque paroi, λ̄_p, ρ, A_eff, N_c,Rd, puis par axe L_cr, N_cr, λ̄ (deux façons), α et sa règle,
  φ, χ, N_b,Rd, η).
- **Dessins** : la **barre** dans ses deux plans de flambement, à l'échelle (largeur réelle du profilé dans le plan ;
  pour une cornière, l'étendue du contour exact selon v autour de u-u et selon u autour de v-v ; textes à l'échelle de
  la barre ou de six largeurs de section pour une barre courte),
  appuis, maintiens, N_Ed, déformée du mode (sinus, cosinus, mode encastré–articulé tan βL = βL ; amplitude
  amplifiée), cote de L_cr entre points d'inflexion ; la **section** à l'échelle (contour exact, hachures, axes de
  flambement et leur courbe, u-u/v-v pour les cornières, cotes, fiche du calcul) ; les **courbes χ(λ̄)** (a0 à d,
  Euler, le point de chaque axe ; un graphique, dit hors échelle par sa vue, « echelle »: false) ; et, à partir de
  deux barres, le tableau **« Toutes les barres »**.
