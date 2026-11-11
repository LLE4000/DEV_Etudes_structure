# Transcription du catalogue des hourdis Ergon — 2026-11-11

*MESURE datée (ne se réécrit pas). Produite par `tools/transcrire_hourdis_ergon.py` depuis le guide technique Ergon 2018 (92 pages) : trois lectures indépendantes — transcription à la main, relecture du texte du PDF, relevé vectoriel des coupes — qui s'accordent. Le fichier livré est `beton/donnees/hourdis/ergon_2018.json`.*

**7 profils, 53 variantes (M_Rd, V_Rd,c, M_Rd avec 50 mm de couche collaborante) ; relecture du PDF : 0 écart.**

## 1 · Relecture du texte du PDF

Relues mot à mot et comparées à la transcription : les caractéristiques (p. 20 : h, b, poids avec joints en kN/m et kN/m², A, I_x, W_sup, W_inf, joints), les sept tableaux M_Rd / V_Rd,c / M_Rd avec couche collaborante (p. 29-35, ordre des lignes compris), le poids à transporter et le remplissage des joints de chaque page, la chaîne de cotes horizontale et les autres cotes de chaque coupe, les réservations maximales (p. 24), les longueurs d'appui (p. 25), les largeurs des hourdis rétrécis (p. 26) et les charges admissibles des chevilles dans l'alvéole (p. 28). **Aucun écart.**

## 2 · Relevé vectoriel des coupes, confronté aux valeurs publiées

Contour gris et alvéoles blanches de chaque coupe, à l'échelle de la cote 1196 mm ; aire, inertie et modules intégrés sur le relevé brut, puis sur le dessin livré (simplifié à 0,15 mm). Les valeurs de calcul livrées restent celles du fabricant : le relevé sert au dessin et au contrôle.

| Profil | p. | h relevé | alvéoles | A (10³ mm²) | écart | I_x (10⁶ mm⁴) | écart | W_sup | écart | W_inf | écart |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SP 150 | 29 | 149,8 | 8 | 113,8 (114) | -0,2 % | 291 (291) | -0,1 % | 3,84 (3,84) | -0,1 % | 3,92 (3,93) | -0,2 % |
| SP 200 | 30 | 199,8 | 6 | 127,4 (127) | 0,3 % | 629 (630) | -0,1 % | 6,22 (6,22) | 0,0 % | 6,38 (6,38) | 0,0 % |
| SPN 200 | 31 | 200,0 | 9 | 131,9 (134) | -1,6 % | 639 (648) | -1,5 % | 6,13 (6,24) | -1,8 % | 6,67 (6,74) | -1,1 % |
| SP 265 | 32 | 265,0 | 5 | 174,6 (174) | 0,3 % | 1469 (1467) | 0,2 % | 10,66 (10,64) | 0,2 % | 11,56 (11,54) | 0,2 % |
| SP 320 | 33 | 319,6 | 4 | 186,5 (187) | -0,3 % | 2412 (2421) | -0,4 % | 15,00 (15,03) | -0,2 % | 15,20 (15,23) | -0,2 % |
| SP 400 | 34 | 400,1 | 4 | 222,8 (222) | 0,3 % | 4549 (4541) | 0,2 % | 22,49 (22,44) | 0,2 % | 23,00 (22,97) | 0,1 % |
| SPZ 500 | 35 | 500,0 | 4 | 276,1 (276) | 0,0 % | 8557 (8550) | 0,1 % | 33,82 (33,78) | 0,1 % | 34,65 (34,63) | 0,1 % |

Plus grand écart : 1,8 % (SPN 200 : la section dessinée est un peu plus petite que la section de calcul du fabricant) ; les autres profils à 0,4 % au plus.

## 3 · Écarts du guide lui-même

- **SP 265** (p. 32) : la dernière cote de la chaîne horizontale est imprimée 222 ; le bord vaut 154 (symétrie, somme 1196, relevé du tracé 153,9) : 154 est livré.
- **SP 400** (p. 20) : la coupe de la page 20 cote les alvéoles 218 et répète 270 à la place du dernier 193 ; la page 34 cote 216 et 193 (relevé du tracé 215,9 et 193,0) : la page 34 est livrée.
- **SPZ 500** (p. 35) : remplissage des joints 18,2 l/m² à la page 35, 16,6 l/m² au tableau de la page 20 : les deux sont livrés ; le poids propre publié (avec joints) n'en dépend pas.

## 4 · Ce que le guide ne publie pas

- V_Rd,c avec couche collaborante
- moment de fissuration M_cr
- pression locale admissible sur la membrure supérieure
- courbes de performance (charge utile admissible selon la portée, p. 21-22 et 29-35) : graphiques, non transcrits
