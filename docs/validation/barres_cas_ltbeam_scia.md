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

*(complété après le lot A : les valeurs attendues sont celles du site corrigé)*
