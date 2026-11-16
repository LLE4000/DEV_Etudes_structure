# Barres acier — cas à recalculer au bureau dans LTBeam et dans SCIA

*Liste demandée le 16/11/2026 (question 9 du lot 0). Ces calculs qualifient le moment critique de déversement du site
(son solveur aux éléments finis) et ses vérifications Eurocode 3 par deux logiciels extérieurs. À faire avant la fin du
lot 3. Merci de renvoyer, pour chaque cas, la valeur obtenue ET les propriétés de section que le logiciel a employées
(I_z, I_t, I_w, position du centre de cisaillement) : si elles diffèrent des nôtres, je recalcule l'attendu avec les
siennes, au lieu de conclure à un écart.*

**Critère** (brief, règle 7) : écart ≤ 1 % : conforme ; de 1 à 3 % : à expliquer ; plus de 3 % : à corriger ou à
justifier. Une formule n'est jamais ajustée pour coller à un résultat.

**Conventions communes**
- Acier : E = 210 000 N/mm². Le site prend G = 81 000 N/mm² (EN 1993-1-1 § 3.2.6). LTBeam calcule G = E/(2·(1 + ν)) =
  80 769 N/mm² avec ν = 0,3 : les deux colonnes sont données.
- Appuis « à fourche » aux deux bouts : déplacement latéral et rotation de torsion empêchés ; rotation latérale et
  gauchissement libres (k_z = k_w = 1). Dans LTBeam : v = 0 et θ = 0 bloqués ; v′ et θ′ libres.
- Aucun maintien intermédiaire.
- Charges verticales, vers le bas, qui restent verticales pendant le déversement.
- Niveau d'application : la cote du point d'application au-dessus du CENTRE DE CISAILLEMENT (z_g > 0 : au-dessus,
  déstabilisant). Pour une section en I symétrique, centre de cisaillement = centre de gravité.
- La valeur de la charge n'a pas d'importance (le moment critique est M_cr = μ_cr × M_max) : prendre par exemple
  q = 10 kN/m ou P = 100 kN, et relever μ_cr et M_cr.

## 1. LTBeam — IPE 300, portée 6,00 m (cas R1 du brief)

Section : IPE 300 (bibliothèque de LTBeam, ou propriétés saisies) — h = 300, b = 150, t_w = 7,1, t_f = 10,7, r = 15 mm ;
I_z = 603,8 cm⁴ ; I_t = 20,12 cm⁴ ; I_w = 125,9·10³ cm⁶.

| Cas | Chargement | Niveau d'application (z_g) | M_cr attendu, G = 81 000 | M_cr attendu, G = 80 769 |
|---|---|---|---|---|
| L1 | moments égaux aux deux bouts (moment constant) | — | 90,47 kN·m | 90,38 kN·m |
| L2 | charge répartie sur toute la portée | centre de cisaillement (0) | 102,31 kN·m | 102,21 kN·m |
| L3 | charge répartie | dessus de la semelle haute (+150 mm) | 78,83 kN·m | 78,73 kN·m |
| L4 | charge répartie | dessous de la semelle basse (−150 mm) | 132,69 kN·m | 132,59 kN·m |
| L5 | charge concentrée à mi-portée | centre de cisaillement (0) | 123,12 kN·m | 123,00 kN·m |
| L6 | charge concentrée à mi-portée | dessus de la semelle haute (+150 mm) | 89,19 kN·m | 89,08 kN·m |
| L7 | charge concentrée à mi-portée | dessous de la semelle basse (−150 mm) | 168,95 kN·m | 168,84 kN·m |

M_cr = μ_cr × M_max, avec M_max = q·L²/8 (charge répartie) ou P·L/4 (charge concentrée). L1 est la solution exacte
(formule fermée) : un écart y signale une différence de propriétés ou de G, pas de méthode.

## 2. LTBeam — profilé reconstitué soudé bisymétrique « PRS-A », portée 10,00 m

Section : semelles 300 × 30 mm, âme 540 × 15 mm (hauteur totale 600 mm), cordons non comptés. Propriétés du site
(formules du brief, § 18.4) : I_z = 135,152·10⁶ mm⁴ ; I_t = 5,657·10⁶ mm⁴ ; I_w = 1,0965·10¹³ mm⁶. Si LTBeam calcule
lui-même I_t (par exemple Σ b·t³/3 = 6,008·10⁶ mm⁴), noter la valeur.

| Cas | Chargement | z_g | M_cr attendu, G = 81 000 | G = 80 769 |
|---|---|---|---|---|
| L8 | moment constant | — | 1 385,7 kN·m | 1 384,4 kN·m |
| L9 | charge répartie | centre de cisaillement (0) | 1 567,1 kN·m | 1 565,7 kN·m |
| L10 | charge répartie | dessus de la semelle haute (+300 mm) | 1 191,4 kN·m | 1 190,0 kN·m |
| L11 | charge répartie | dessous de la semelle basse (−300 mm) | 2 059,6 kN·m | 2 058,2 kN·m |

## 3. LTBeam — profilé reconstitué soudé MONOSYMÉTRIQUE « PRS-B », portée 10,00 m (le cas qui manque)

C'est le cas le plus important : le solveur du site n'a pas encore le terme de Wagner (lot 3), et aucune référence
extérieure ne qualifie aujourd'hui un M_cr de section monosymétrique.

Section : semelle haute 300 × 30 mm, âme 540 × 15 mm, semelle basse 200 × 30 mm (hauteur 600 mm). Valeurs du brief
(§ 11.8) : centre de gravité à 337,013 mm de la face inférieure ; centre de cisaillement à 454,714 mm de la face
inférieure (117,701 mm au-dessus du centre de gravité) ; I_z = 87,652·10⁶ mm⁴ ; I_t = 4,757·10⁶ mm⁴ (formule du
brief ; valeur exacte 4,864·10⁶) ; I_w = 5,0127·10¹² mm⁶ ; z_j = +144,78 mm (grande semelle comprimée).

| Cas | Chargement | z_g (au-dessus du centre de cisaillement) | M_cr attendu (I_t = 4,757·10⁶, G = 81 000) | avec I_t exact (4,864·10⁶) |
|---|---|---|---|---|
| L12 | moment constant, GRANDE semelle (haute) comprimée | — | 1 241,73 kN·m | 1 249,77 kN·m |
| L13 | moment constant, PETITE semelle (basse) comprimée | — | 715,69 kN·m | 723,73 kN·m |
| L14 | charge répartie | centre de cisaillement (0) | 1 234,80 kN·m | — |
| L15 | charge répartie | dessus de la semelle haute (+145,286 mm) | 1 086,25 kN·m | 1 095,72 kN·m |
| L16 | charge répartie | dessous de la semelle basse (−454,714 mm) | 1 803,30 kN·m | — |

Sans le terme de Wagner, L12 et L13 donneraient tous deux 942,71 kN·m : c'est ce que ce cas doit départager. LTBeam
définit la section monosymétrique par ses dimensions ; il calculera ses propres I_t, I_w et z_j : les noter.

## 4. SCIA Engineer — trois cas avec l'annexe nationale belge

*Valeurs attendues : celles du site après le lot A (couple (λ̄_LT,0 ; β) = (0,2 ; 1,0), f_y selon l'épaisseur, η = 1,00
pour un laminé). Merci de renvoyer, pour chaque ligne, la valeur de SCIA et, s'il y a un écart, le réglage employé.*

**Réglages SCIA à vérifier avant de lancer** (sinon l'écart vient du réglage, pas du calcul) :
- Annexe nationale : Belgique (NBN EN 1993-1-1 ANB:2018). γ_M0 = γ_M1 = 1,00.
- **Limite élastique selon l'épaisseur** (option « Thickness range » / épaisseur du matériau, EN 10025-2) : c'est
  le point que le lot A a corrigé. Le HEB 300 en S235 doit être calculé à **f_y = 225 N/mm²** (t_f = 19 mm,
  tableau 3.3 ANB, 16 < t ≤ 40 mm). Si SCIA prend 235, le dire : on compare alors à la ligne « à 235 » donnée plus bas.
- Déversement : méthode générale des profilés laminés (§ 6.3.2.3), courbes du tableau 6.5 (laminé : b si h/b ≤ 2,
  c au-delà), **λ̄_LT,0 = 0,2 et β = 1,0** (ANB:2018, 6.3.2.3(1)), f = 1 (k_c = 1). Appuis à fourche, k = k_w = 1.
  Pour M_cr, le site résout le problème aux valeurs propres sur le diagramme réel : dans SCIA, prendre le calcul de
  M_cr par valeurs propres (« LTBII ») si vous l'avez ; sinon noter la méthode et les coefficients C1, C2, C3 employés.
- Interaction N + M : méthode 1 (annexe A), comme le site.
- Combinaison : expression (6.10), 1,35·G + 1,5·Q (une seule combinaison par cas, sauf le cas S3).
- Propriétés de section du site (catalogue du site, à comparer à celles de SCIA) :

| Profil | A (mm²) | I_y (cm⁴) | I_z (cm⁴) | I_t (cm⁴) | I_w (10³ cm⁶) | W_pl,y (cm³) |
|---|---|---|---|---|---|---|
| IPE 300 | 5 381 | 8 357 | 603,8 | 19,77 | 126,3 | 628,4 |
| HEB 300 | 14 909 | 25 167 | 8 563 | 187,7 | 1 690 | 1 869 |
| IPE 360 | 7 273 | 16 267 | 1 043 | 37,90 | 314,6 | 1 019 |

(Les I_t du site sont un peu plus bas que ceux des catalogues des fabricants — IPE 300 : 19,77 contre 20,12 cm⁴ :
c'est pourquoi M_cr de S1 est 0,7 % sous la référence R3 du brief, 86,98 kN·m.)

### S1 — poutre IPE 300 en S235, 6,00 m sur deux appuis (cas R2 / R3 du brief)

Charges : G = 2,42 kN/m sur toute la portée + 10 kN à mi-portée ; Q = 15 kN à mi-portée (catégorie A) ; poids
propre NON ajouté (il est compris dans les 2,42 kN/m). Aucun maintien latéral entre les appuis. Deux calculs : charges
sur la semelle HAUTE (z_g = +150 mm), puis au centre de cisaillement.

| Grandeur | Attendu (site) |
|---|---|
| q_Ed ; P_Ed | 3,267 kN/m ; 36,00 kN |
| M_y,Ed (mi-portée) ; V_z,Ed (appui) | 68,70 kN·m ; 27,80 kN |
| Classe ; M_c,Rd (W_pl·f_y) ; taux | 1 ; 147,67 kN·m ; 0,465 |
| V_pl,z,Rd ; taux | 348,47 kN ; 0,080 |
| Déversement, charges sur la semelle haute : M_cr ; λ̄_LT ; χ_LT ; M_b,Rd ; **taux** | 86,36 kN·m ; 1,308 ; 0,4232 ; 62,49 kN·m ; **1,099 (non conforme)** |
| Déversement, charges au centre de cisaillement : M_cr ; λ̄_LT ; χ_LT ; M_b,Rd ; **taux** | 117,52 kN·m ; 1,121 ; 0,5228 ; 77,20 kN·m ; **0,890** |

Courbe de déversement b (h/b = 2,0), α_LT = 0,34. Avec l'ancien couple (0,4 ; 0,75), le site donnait 0,896 en semelle
haute : c'est l'écart le plus fort du lot A, et la raison de ce cas.

### S2 — poteau HEB 300 en S235, 5,00 m, articulé aux deux bouts, N + M

Efforts de calcul imposés (pas de combinaison) : N_Ed = 1 400 kN (compression) ; M_y,Ed = 70 kN·m en tête, 0 au pied
(diagramme linéaire, ψ = 0) ; V_z,Ed = 14 kN. Longueurs de flambement 5,00 m autour des deux axes ; déversement sur
5,00 m, appuis à fourche, aucun maintien.

| Grandeur | Attendu (site, f_y = 225) | Pour mémoire, à f_y = 235 |
|---|---|---|
| f_y (t_f = 19 mm) | 225 N/mm² (tableau 3.3 ANB) | — |
| Classe ; N_pl,Rd ; M_pl,y,Rd | 1 ; 3 354,4 kN ; 420,47 kN·m | 1 ; 3 503,5 kN ; 439,16 kN·m |
| Flambement y-y (courbe b) : N_cr ; λ̄ ; χ ; N_b,y,Rd | 20 864 kN ; 0,401 ; 0,9257 ; 3 105,1 kN | à recalculer par SCIA |
| Flambement z-z (courbe c) : N_cr ; λ̄ ; χ ; N_b,z,Rd | 7 099 kN ; 0,687 ; 0,7325 ; 2 457,0 kN | à recalculer par SCIA |
| Déversement : M_cr ; λ̄_LT ; M_Ed/M_cr | 2 649 kN·m ; 0,398 ; 0,026 ≤ λ̄_LT,0² = 0,04 : **négligé**, χ_LT = 1 | — |
| Annexe A : C_my,0 ; μ_y ; μ_z ; k_yy ; k_zy | 0,933 ; 0,995 ; 0,938 ; 1,081 ; 0,582 | — |
| **(6.61)** = N/(χ_y·N_Rk) + k_yy·M/(χ_LT·M_y,Rk) | **0,631** | — |
| **(6.62)** = N/(χ_z·N_Rk) + k_zy·M/(χ_LT·M_y,Rk) | **0,667** | — |

Vérification à la main : (6.61) = 1 400/3 105,1 + 1,0814 × 70/420,47 = 0,451 + 0,180 = 0,631 ; (6.62) = 1 400/2 457,0 +
0,5821 × 70/420,47 = 0,570 + 0,097 = 0,667. Ce cas contrôle d'abord la limite élastique selon l'épaisseur : à 235,
tous les taux baissent d'environ 4 %.

### S3 — poutre continue IPE 360 en S355, deux travées de 6,00 m

Charges, sur la semelle HAUTE : G = 6 kN/m sur les deux travées ; Q1 = 10 kN/m sur la travée 1, Q2 = 10 kN/m sur la
travée 2 (catégorie B, une même action disposée librement : chaque travée chargée seule, puis les deux). Poids propre
NON ajouté. Appuis à fourche sur les trois appuis, aucun maintien entre eux : un tronçon de déversement par travée.

| Grandeur | Combinaison | Attendu (site) |
|---|---|---|
| f_y (t_f = 12,7 mm) | — | 355 N/mm² |
| M_y,Ed sur l'appui central ; M_c,Rd ; taux | 1,35·G + 1,5·Q1 + 1,5·Q2 | −103,95 kN·m ; 361,82 kN·m ; 0,287 |
| V_z,Ed sur l'appui central ; V_pl,z,Rd ; taux | 1,35·G + 1,5·Q1 + 1,5·Q2 | 86,62 kN ; 720,24 kN ; 0,120 |
| Déversement de la travée 1 : M_Ed (en travée) ; M_cr ; λ̄_LT ; χ_LT ; M_b,Rd ; **taux** | 1,35·G + 1,5·Q1 | 71,81 kN·m ; 134,51 kN·m ; 1,640 ; 0,2732 ; 98,84 kN·m ; **0,727** |

Courbe de déversement c (h/b = 2,12 > 2), α_LT = 0,49. Le diagramme de la travée 1 va de 0 (appui de rive) à +71,81
(en travée) puis −70,20 kN·m (appui central) : c'est le cas où la méthode de M_cr pèse le plus (valeurs propres ou
coefficients C1 à C3). Si SCIA donne un autre M_cr, renvoyer M_cr : je recalcule λ̄_LT et le taux avec sa valeur
avant de conclure.

**Critère** : celui du début de ce document (≤ 1 % conforme ; 1 à 3 % à expliquer ; plus de 3 % à corriger ou à
justifier), comparé grandeur par grandeur, dans l'ordre des lignes : un écart sur M_cr ou f_y explique les suivants.
