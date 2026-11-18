# Barres acier — la revue finale du lot C : les deux méthodes sur tous les cas

*Mesure datée du 18/11/2026. Elle ne se réécrit pas : une correction s'y porte en révision datée, dans un commit à part. Produite par `tools/comparaison_barres.py` (rejoué par `tests/test_barres_revue.py`, qui compare au brief, jamais à ce fichier).*

**Donnée du rapport (pas une règle du module, ni une ligne de la note) : un cas des classeurs ne donne que des efforts de service ; l'Eurocode 3 le reçoit à N_Ed = 1,5 × P et M_Ed = 1,5 × M, la même sécurité que s = 1,5.**

## 0. En bref

- **Défaut non conservatif trouvé, hors de ce bloc** (moteur, non corrigé ici) : une barre à débord gauche ne peut pas être chargée jusqu'au bout de sa travée. `acier/barre/entree.py`, `normaliser` : `Lt = sum(b["travees"])` borne les charges, les maintiens et les points à Σ portées, sans les débords. Mesuré : 6 m + débord gauche 0,5 m, q = 10 kN/m de 0 à 6,5 m refusé ; de 0 à 6 m accepté, et les 0,5 derniers mètres de la travée restent sans charge (M_max 59,06 kN·m). Remède proposé : `Lt = statique.longueur(b)`, débords lus d'abord. À corriger avant la mise en ligne.
- 35 cas, 90 valeurs du brief : **toutes à 1 % au plus** (règle 7).
- Chaque cas a ses deux méthodes, ou un statut nommé (non couvert, sans objet, fonction seule) ; jamais une case vide.
- Écarts de plus de 10 points entre les méthodes : 18, chacun dit par une phrase (§ 1).
- Qualification externe : L17a reçu, conforme avec les propriétés des fabricants (+0,2 %) ; S1 à S3 reçus ; L1 à L16, L17b, L17c, ECCS 17 à 20 et 22, JRC attendus (§ 3).
- À confirmer : `npl_gamma` seul ; sept décisions du bureau dans la table, une dans sa source du site (§ 4).

Révision du 18/11/2026, intégration du lot C : le défaut non conservatif ci-dessus est CORRIGÉ (commit 1a34173). La barre
va de 0 à g + Σ portées + d (`statique.longueur`) au moteur (`entree.normaliser`, les charges lues avec leurs débords) et à
l'écran (`api_web/profil.py`) : 6 m + débord gauche 0,5 m, q de 0 à 6,5 m accepté, réactions et M_max = `rdm/poutres`
(`tests/test_barre_moteur.py`, `tests/test_api_web_profil.py`, cinq mutations). Le rapport de l'intégration est
`docs/validation/lot_c_2026-11-18.md`. La mesure ci-dessus n'est pas réécrite.

## 1. Les deux méthodes, cas par cas

Statuts du brief § 9 : vérifié, à la limite (95 % à 100 %, le seuil des notes du site), non vérifié, hors domaine, non couvert, bloqué. Écart = taux EC3 − taux ELS, en points de %.

### 1. Classeurs « Flambement » (11.1, efforts imposés, u = 1)

| Cas | Profilé | Nuance | f_y ELS / EC3 | L (m) | Mode | Efforts |
|---|---|---|---|---|---|---|
| HEB 200 | HEB 200 (catalogue) | S235 | 235 / 235 | 7,20 | efforts imposés | P = 432,72 kN ; N_Ed = 649,08 kN |
| HEA 120 Chancel | HEA 120 (catalogue) | S235 | 235 / 235 | 1,00 | efforts imposés | P = 112 kN |
| CHS 168,3 × 6 (A du classeur) | CHS 168,3 × 6, données du classeur (saisies) | S235 | 235 / — | 7,00 | efforts imposés | P = 35 kN ; M_z = 2,963 kN·m |
| CHS 168,3 × 6 (A = 3 059 mm²) | CHS 168.3x6 (froid) (catalogue) | S235 | 235 / 235 | 7,00 | efforts imposés | P = 35 kN ; M_y = 3,223 kN·m |
| REALEX Ø610 × 12,5 | Tube Ø610 × 12,5 (absent du catalogue, saisi) | f_y 450 | 450 / — | 10,00 | efforts imposés | P = 5 000 kN |
| FLAMBECBD (HEA 160) | HEA 160, données du classeur (saisies) | S235 | 235 / — | 1,75 | efforts imposés | P = 300 kN ; M = 3 kN·m |
| FLAMBECBD, lecture axe faible | HEA 160 (catalogue) | S235 | 235 / 235 | 1,75 | efforts imposés | P = 300 kN ; M_z = 3 kN·m |
| FLAMBECBD, lecture axe fort | HEA 160 (catalogue) | S235 | 235 / 235 | 1,75 | efforts imposés | P = 300 kN ; M_y = 3 kN·m ; L_d 1,75 m |
| FLAMBEBAD (IPE 270) | IPE 270, données du classeur (saisies) | S355 | 355 / — | 3,40 | efforts imposés | P = 185 kN ; M = 0,522 kN·m |
| FLAMBEBAD, lecture cohérente | IPE 270 (catalogue) | S355 | 355 / 355 | 3,40 | efforts imposés | P = 185 kN ; M_y = 0,522 kN·m ; L_d 3,40 m |

| Cas | Sujet | ELS | Eurocode 3 | Écart (points) | Gouverne (ELS / EC3) |
|---|---|---|---|---|---|
| HEB 200 | Flambement, axe faible | 111,9 % · non vérifié | 114,0 % · non vérifié | 2,1 | H-N-z / E-Nb-z |
| HEB 200 | Flambement, axe fort | 52,9 % · vérifié | 53,4 % · vérifié | 0,5 | H-N-y / E-Nb-y |
| HEB 200 | Flexion composée | — | 35,4 % · vérifié | — | — / E-N |
| HEA 120 Chancel | Flambement, axe faible | 30,7 % · vérifié | 30,6 % · vérifié | -0,1 | H-N-z / E-Nb-z |
| HEA 120 Chancel | Flambement, axe fort | 28,7 % · vérifié | 28,4 % · vérifié | -0,3 | H-N-y / E-Nb-y |
| HEA 120 Chancel | Flexion composée | — | 28,2 % · vérifié | — | — / E-N |
| CHS 168,3 × 6 (A du classeur) | Flambement, axe faible | 22,6 % · vérifié | _non calculé : propriété(s) manquante(s) pour ces efforts : W_pl,y_ | — | H-N-z / — |
| CHS 168,3 × 6 (A du classeur) | Flambement, axe fort | 22,6 % · vérifié | _non calculé : propriété(s) manquante(s) pour ces efforts : W_pl,y_ | — | H-N-y / — |
| CHS 168,3 × 6 (A du classeur) | Flexion composée | 40,5 % · vérifié | _non calculé : propriété(s) manquante(s) pour ces efforts : W_pl,y_ | — | H-SIGMA / — |
| CHS 168,3 × 6 (A = 3 059 mm²) | Flambement, axe faible | 18,6 % · vérifié | 18,7 % · vérifié | 0,2 | H-N-z / E-Nb-z |
| CHS 168,3 × 6 (A = 3 059 mm²) | Flambement, axe fort | 18,6 % · vérifié | 18,7 % · vérifié | 0,2 | H-N-y / E-Nb-y |
| CHS 168,3 × 6 (A = 3 059 mm²) | Flexion composée | 38,1 % · vérifié | 33,4 % · vérifié | -4,7 | H-SIGMA / E-661 |
| REALEX Ø610 × 12,5 | Flambement, axe faible | 83,1 % · hors domaine | _non couvert : f_y = 450 N/mm² n'est pas une nuance du module (S235 à S460)_ | — | H-N-z / — |
| REALEX Ø610 × 12,5 | Flambement, axe fort | 83,1 % · hors domaine | _non couvert : f_y = 450 N/mm² n'est pas une nuance du module (S235 à S460)_ | — | H-N-y / — |
| FLAMBECBD (HEA 160) | Flambement, axe faible | bloqué | _non calculé : propriété(s) manquante(s) pour ces efforts : W_pl,y_ | — | H-N-z / — |
| FLAMBECBD (HEA 160) | Flambement, axe fort | bloqué | _non calculé : propriété(s) manquante(s) pour ces efforts : W_pl,y_ | — | H-N-y / — |
| FLAMBECBD (HEA 160) | Flexion composée | bloqué | _non calculé : propriété(s) manquante(s) pour ces efforts : W_pl,y_ | — | H-SIGMA / — |
| FLAMBECBD, lecture axe faible | Flambement, axe faible | 57,4 % · vérifié | 57,4 % · vérifié | 0,0 | H-N-z / E-Nb-z |
| FLAMBECBD, lecture axe faible | Flambement, axe fort | 50,9 % · vérifié | 50,9 % · vérifié | 0,0 | H-N-y / E-Nb-y |
| FLAMBECBD, lecture axe faible | Flexion composée | 85,3 % · vérifié | 71,5 % · vérifié | -13,8 | H-SIGMA / E-662 |
| FLAMBECBD, lecture axe fort | Flambement, axe faible | 57,4 % · vérifié | 57,4 % · vérifié | 0,0 | H-N-z / E-Nb-z |
| FLAMBECBD, lecture axe fort | Flambement, axe fort | 50,9 % · vérifié | 50,9 % · vérifié | 0,0 | H-N-y / E-Nb-y |
| FLAMBECBD, lecture axe fort | Flexion composée | 68,6 % · vérifié | 62,0 % · vérifié | -6,6 | H-SIGMA / E-662 |
| FLAMBECBD, lecture axe fort | Déversement | 10,8 % · vérifié | 7,8 % · vérifié | -3,0 | H-DEV / E-Mb |
| FLAMBEBAD (IPE 270) | Flambement, axe faible | 48,4 % · vérifié | _non calculé : propriété(s) manquante(s) pour ces efforts : W_pl,y_ | — | H-N-z / — |
| FLAMBEBAD (IPE 270) | Flambement, axe fort | 17,7 % · vérifié | _non calculé : propriété(s) manquante(s) pour ces efforts : W_pl,y_ | — | H-N-y / — |
| FLAMBEBAD (IPE 270) | Flexion composée | 48,9 % · vérifié | _non calculé : propriété(s) manquante(s) pour ces efforts : W_pl,y_ | — | H-SIGMA / — |
| FLAMBEBAD, lecture cohérente | Flambement, axe faible | 48,3 % · vérifié | 48,2 % · vérifié | 0,0 | H-N-z / E-Nb-z |
| FLAMBEBAD, lecture cohérente | Flambement, axe fort | 17,7 % · vérifié | 17,8 % · vérifié | 0,1 | H-N-y / E-Nb-y |
| FLAMBEBAD, lecture cohérente | Flexion composée | 49,5 % · vérifié | 48,6 % · vérifié | -0,9 | H-SIGMA / E-662 |
| FLAMBEBAD, lecture cohérente | Déversement | 1,2 % · vérifié | 0,5 % · vérifié | -0,8 | H-DEV / E-Mb |

- FLAMBECBD, lecture axe faible, flexion composée : la méthode ELS est plus sévère de 13,8 points (W_el et amplification 1/PE (ELS) contre W_pl et k_ij de l'annexe D ANB (EC3)).
- REALEX Ø610 × 12,5 : Décision n° 6 ouverte : le taux ELS va de 0,831 à 1,018 selon la fabrication et la position (test 6 du § 11.5).
- CHS 168,3 × 6 (A du classeur) : Eurocode 3 — non calculé : propriété(s) manquante(s) pour ces efforts : W_pl,y.
- REALEX Ø610 × 12,5 : Eurocode 3 — non couvert : f_y = 450 N/mm² n'est pas une nuance du module (S235 à S460).
- FLAMBECBD (HEA 160) : Eurocode 3 — non calculé : propriété(s) manquante(s) pour ces efforts : W_pl,y.
- FLAMBEBAD (IPE 270) : Eurocode 3 — non calculé : propriété(s) manquante(s) pour ces efforts : W_pl,y.
- FLAMBECBD (HEA 160) : ELS bloqué — I_y ≤ I_z : des valeurs de l'axe faible dans les cases de l'axe fort ? Verdict bloqué.

Valeurs du brief :

| Cas | Valeur | Brief | Site | Écart |
|---|---|---|---|---|
| HEB 200 | ELS, taux de flambement z (σ 175,09 / 156,67) | 1,1180 | 1,1194 | 0,13 % |
| HEB 200 | EC3, taux de flambement z (649,08 / 570,1) | 1,1390 | 1,1404 | 0,12 % |
| HEB 200 | EC3, N_b,z,Rd (kN, 11.2) | 570,10 | 569,16 | -0,17 % |
| HEB 200 | EC3, λ̄_z (11.2) | 1,5122 | 1,5136 | 0,09 % |
| HEB 200 | EC3, χ_z (11.2) | 0,3106 | 0,3102 | -0,14 % |
| HEA 120 Chancel | ELS, φ_z | 0,9181 | 0,9180 | -0,01 % |
| HEA 120 Chancel | ELS, φ_y | 0,9822 | 0,9822 | 0,00 % |
| HEA 120 Chancel | ELS, σ (N/mm²) | 48,14 | 48,15 | 0,02 % |
| CHS 168,3 × 6 (A du classeur) | ELS, φ | 0,3930 | 0,3930 | 0,00 % |
| CHS 168,3 × 6 (A du classeur) | ELS, PE | 0,8770 | 0,8770 | 0,00 % |
| CHS 168,3 × 6 (A du classeur) | ELS, σ (N/mm²) | 63,52 | 63,52 | 0,00 % |
| CHS 168,3 × 6 (A = 3 059 mm²) | ELS, σ (N/mm²) | 59,75 | 59,76 | 0,01 % |
| CHS 168,3 × 6 (A = 3 059 mm²) | ELS, taux | 0,3814 | 0,3814 | 0,01 % |
| REALEX Ø610 × 12,5 | ELS, φ | 0,8552 | 0,8552 | 0,00 % |
| REALEX Ø610 × 12,5 | ELS, P_adm (N) | 6 020 154 | 6 020 154 | 0,00 % |
| REALEX Ø610 × 12,5 | ELS, PE | 0,6546 | 0,6546 | 0,00 % |
| REALEX Ø610 × 12,5 | ELS, σ (N/mm²) | 249,16 | 249,16 | 0,00 % |
| REALEX Ø610 × 12,5 | ELS, lecture a (fini à chaud), taux | 0,8310 | 0,8305 | -0,05 % |
| REALEX Ø610 × 12,5 | ELS, lecture c, taux | 0,9840 | 0,9838 | -0,02 % |
| REALEX Ø610 × 12,5 | ELS, lecture c + poids propre, taux | 1,0180 | 1,0180 | 0,00 % |
| REALEX Ø610 × 12,5 | ELS, lecture a + poids propre, taux | 0,8650 | 0,8647 | -0,03 % |
| FLAMBECBD (HEA 160) | ELS, φ_z | 0,8605 | 0,8605 | 0,00 % |
| FLAMBECBD (HEA 160) | ELS, φ_y | 0,9131 | 0,9131 | 0,00 % |
| FLAMBECBD (HEA 160) | ELS, P_adm (N) | 522 660 | 522 660 | 0,00 % |
| FLAMBECBD (HEA 160) | ELS, σ avec φ_min (N/mm²) | 133,60 | 133,63 | 0,02 % |
| FLAMBECBD, lecture axe faible | ELS, σ (N/mm²) | 133,63 | 133,60 | -0,03 % |
| FLAMBECBD, lecture axe fort | ELS, σ, classeurs (N/mm²) | 107,52 | 107,48 | -0,04 % |
| FLAMBECBD, lecture axe fort | ELS, σ, variante fascicule (N/mm²) | 105,96 | 105,92 | -0,04 % |
| FLAMBEBAD (IPE 270) | ELS, φ_z | 0,3519 | 0,3519 | 0,00 % |
| FLAMBEBAD (IPE 270) | ELS, φ_y | 0,9601 | 0,9601 | 0,00 % |
| FLAMBEBAD (IPE 270) | ELS, P_adm,z (N) | 382 617 | 382 617 | 0,00 % |
| FLAMBEBAD (IPE 270) | ELS, σ avec φ_min (N/mm²) | 115,70 | 115,68 | -0,01 % |
| FLAMBEBAD, lecture cohérente | ELS, σ, classeurs (N/mm²) | 117,42 | 117,22 | -0,17 % |
| FLAMBEBAD, lecture cohérente | ELS, σ, variante fascicule (N/mm²) | 117,20 | 117,00 | -0,17 % |

### 2. Exemples de la doc théorique (11.1)

| Cas | Profilé | Nuance | f_y ELS / EC3 | L (m) | Mode | Efforts |
|---|---|---|---|---|---|---|
| p. 6, caisson soudé | caisson 100 × 100 × 1 cm (saisi) | S235 | 235 / 235 | 20,00 | efforts imposés | P = 4 000 kN |
| p. 8, cornière L 60 × 60 × 8 | L 60 × 60 × 8 (doc, saisie) | S235 | 235 / 235 | 2,00 | efforts imposés | P = 30 kN |
| p. 14, HEA 220 | HEA 220 (catalogue) | S235 | 235 / 235 | 5,00 | efforts imposés, plan vertical seul (L_z = 0) | P = 500 kN ; M = 25 kN·m |
| p. 23, poutre BC HEA 200 | HEA 200 (catalogue) | S235 | 235 / 235 | 8,00 | efforts imposés, déversement non compté (la doc) | P = 38 kN ; M = 48 kN·m |
| p. 25, colonne AB HEA 300 | HEA 300 (catalogue) | S235 | 235 / 235 | 6,00 | efforts imposés, u = 1,5, C = 0,85, fascicule p. 19 | P = 168 kN ; M = 110 kN·m |

| Cas | Sujet | ELS | Eurocode 3 | Écart (points) | Gouverne (ELS / EC3) |
|---|---|---|---|---|---|
| p. 6, caisson soudé | Flambement, axe faible | 71,7 % · vérifié | non couvert | — | H-N-z / E-SECTION |
| p. 6, caisson soudé | Flambement, axe fort | 71,7 % · vérifié | non couvert | — | H-N-y / E-SECTION |
| p. 6, caisson soudé | Flexion composée | — | non couvert | — | — / E-SECTION |
| p. 6, caisson soudé | Déversement | — | non couvert | — | — / E-SECTION |
| p. 6, caisson soudé | Cisaillement | — | non couvert | — | — / E-SECTION |
| p. 8, cornière L 60 × 60 × 8 | Flambement, axe faible | 89,7 % · vérifié | non couvert | — | H-N-z / E-SECTION |
| p. 8, cornière L 60 × 60 × 8 | Flambement, axe fort | 36,7 % · vérifié | non couvert | — | H-N-y / E-SECTION |
| p. 8, cornière L 60 × 60 × 8 | Flexion composée | — | non couvert | — | — / E-SECTION |
| p. 8, cornière L 60 × 60 × 8 | Déversement | — | non couvert | — | — / E-SECTION |
| p. 8, cornière L 60 × 60 × 8 | Cisaillement | — | non couvert | — | — / E-SECTION |
| p. 14, HEA 220 | Flambement, axe faible | 49,6 % · vérifié | — | — | H-N-z / — |
| p. 14, HEA 220 | Flambement, axe fort | 57,5 % · vérifié | 58,6 % · vérifié | 1,1 | H-N-y / E-Nb-y |
| p. 14, HEA 220 | Flexion composée | 94,7 % · vérifié | 91,7 % · vérifié | -3,0 | H-SIGMA / E-661 |
| p. 23, poutre BC HEA 200 | Flambement, axe faible | 17,2 % · vérifié | 17,7 % · vérifié | 0,4 | H-N-z / E-Nb-z |
| p. 23, poutre BC HEA 200 | Flambement, axe fort | 7,7 % · vérifié | 7,8 % · vérifié | 0,0 | H-N-y / E-Nb-y |
| p. 23, poutre BC HEA 200 | Flexion composée | 100,0 % · non vérifié | 150,8 % · non vérifié | 50,8 | H-SIGMA / E-661 |
| p. 23, poutre BC HEA 200 | Déversement | — | 124,9 % · non vérifié | — | — / E-Mb |
| p. 25, colonne AB HEA 300 | Flambement, axe faible | 23,8 % · vérifié | 24,0 % · vérifié | 0,2 | H-N-z / E-Nb-z |
| p. 25, colonne AB HEA 300 | Flambement, axe fort | 12,5 % · vérifié | 12,6 % · vérifié | 0,2 | H-N-y / E-Nb-y |
| p. 25, colonne AB HEA 300 | Flexion composée | 97,9 % · à la limite | 89,3 % · vérifié | -8,6 | H-SIGMA / E-661 |
| p. 25, colonne AB HEA 300 | Déversement | — | 63,8 % · vérifié | — | — / E-Mb |

- p. 23, poutre BC HEA 200, flexion composée : l'Eurocode 3 est plus sévère de 50,8 points (déversement écarté par la doc, compté par l'Eurocode 3 : χ_LT dans (6.61)).
- p. 23, poutre BC HEA 200 : La doc écarte le déversement (dispense de la p. 20, non reprise) ; l'Eurocode 3 le compte sur 8 m.
- p. 6, caisson soudé : Eurocode 3 non couvert — non couvert pour cette section (v1).
- p. 8, cornière L 60 × 60 × 8 : Eurocode 3 non couvert — non couvert pour une cornière (v1).

Valeurs du brief :

| Cas | Valeur | Brief | Site | Écart |
|---|---|---|---|---|
| p. 6, caisson soudé | ELS, λ̄ | 0,5220 | 0,5216 | -0,07 % |
| p. 6, caisson soudé | ELS, φ (courbe b) | 0,8910 | 0,8908 | -0,02 % |
| p. 6, caisson soudé | ELS, P_adm (kN) | 5 582,00 | 5 582,25 | 0,00 % |
| p. 8, cornière L 60 × 60 × 8 | ELS, λ̄ | 1,8200 | 1,8202 | 0,01 % |
| p. 8, cornière L 60 × 60 × 8 | ELS, φ (courbe c) | 0,2360 | 0,2364 | 0,18 % |
| p. 8, cornière L 60 × 60 × 8 | ELS, P_adm (kN) | 33,40 | 33,45 | 0,14 % |
| p. 14, HEA 220 | ELS, σ (N/mm²) | 148,30 | 148,31 | 0,01 % |
| p. 14, HEA 220 | ELS, P_cr (kN) | 4 485,00 | 4 485,10 | 0,00 % |
| p. 23, poutre BC HEA 200 | ELS, σ_P (N/mm²) | 27,00 | 26,99 | -0,03 % |
| p. 23, poutre BC HEA 200 | ELS, σ_M (N/mm²) | 129,70 | 129,68 | -0,02 % |
| p. 23, poutre BC HEA 200 | ELS, σ (N/mm²) | 156,70 | 156,67 | -0,02 % |
| p. 25, colonne AB HEA 300 | ELS, φ_ed | 0,6760 | 0,6760 | 0,00 % |
| p. 25, colonne AB HEA 300 | ELS, σ_P (N/mm²) | 37,30 | 37,25 | -0,13 % |
| p. 25, colonne AB HEA 300 | ELS, σ_M (N/mm²) | 116,10 | 116,06 | -0,03 % |
| p. 25, colonne AB HEA 300 | ELS, σ (N/mm²) | 153,30 | 153,32 | 0,01 % |

### 3. Déversement des classeurs (11.1 bis) : capacité, sans charge

| Cas | Profilé | Nuance | f_y ELS / EC3 | L (m) | Mode | Efforts |
|---|---|---|---|---|---|---|
| HE 200 B, feuille automatique | HEB 200 (catalogue) | S235 | 235 / 235 | 4,00 | efforts imposés, moment constant | sans charge |
| DEVERSEMENT_CAD, feuille S355 (180 × 180 × 14) | HEB 180 (catalogue) | S355 | 355 / 355 | 4,00 | efforts imposés, moment constant | sans charge |
| φ_d = 1 (l = 0,30 m) | HEB 200 (catalogue) | S235 | 235 / 235 | 0,30 | efforts imposés, moment constant | sans charge |
| IPE 600, C_d = 1 | IPE 600 (catalogue) | S235 | 235 / 225 | 12,00 | efforts imposés, moment constant | sans charge |
| IPE 600, C_d = 1,12, charge sur la semelle haute | IPE 600 (catalogue) | S235 | 235 / 225 | 12,00 | efforts imposés, moment constant | sans charge |
| U 140 × 60 × 10 (feuille S235) | U 140 × 60 × 10 (feuille S235) (saisi : h, b, t_f) | S235 | 235 / — | 3,80 | fonction des classeurs | sans charge |
| I 600 × 300 × 50 (t_f > 40, h/b > 1,2) | I 600 × 300 × 50 (t_f > 40, h/b > 1,2) (saisi : h, b, t_f) | S235 | 235 / — | 6,00 | fonction des classeurs | sans charge |
| I 300 × 300 × 80 (t_f = 80) | I 300 × 300 × 80 (t_f = 80) (saisi : h, b, t_f) | S235 | 235 / — | 6,00 | fonction des classeurs | sans charge |

| Cas | σ_adm,d (N/mm²) | Capacité ELS (kN·m) | × 1,5 (kN·m) | M_b,Rd EC3 (kN·m) | ELS × 1,5 / EC3 |
|---|---|---|---|---|---|
| HE 200 B, feuille automatique | 115,70 | 65,9 | 98,9 | 151,00 | 0,655 |
| DEVERSEMENT_CAD, feuille S355 (180 × 180 × 14) | 147,50 | 62,8 | 94,2 | 126,70 | 0,743 |
| φ_d = 1 (l = 0,30 m) | 156,67 | 89,2 | 133,9 | 151,00 | 0,886 |
| IPE 600, C_d = 1 | 42,58 | 130,7 | 196,0 | 214,63 | 0,913 |
| IPE 600, C_d = 1,12, charge sur la semelle haute | 46,86 | 143,8 | 215,8 | 214,63 | 1,005 |
| U 140 × 60 × 10 (feuille S235) | 65,32 | — | — | — | — |
| I 600 × 300 × 50 (t_f > 40, h/b > 1,2) | 119,03 | — | — | — | — |
| I 300 × 300 × 80 (t_f = 80) | 143,92 | — | — | — | — |

- U 140 × 60 × 10 (feuille S235) : Eurocode 3 — non calculé : la section n'est donnée que par h, b et t_f (ni I_t, ni I_w, ni W_pl).
- U 140 × 60 × 10 (feuille S235) : ELS — fonction seule : classe c, φ_d 0,4169, σ_adm,d 65,32 N/mm².
- I 600 × 300 × 50 (t_f > 40, h/b > 1,2) : Eurocode 3 — non calculé : la section n'est donnée que par h, b et t_f (ni I_t, ni I_w, ni W_pl).
- I 600 × 300 × 50 (t_f > 40, h/b > 1,2) : ELS — fonction seule : classe c, φ_d 0,7597, σ_adm,d 119,03 N/mm².
- I 300 × 300 × 80 (t_f = 80) : Eurocode 3 — non calculé : la section n'est donnée que par h, b et t_f (ni I_t, ni I_w, ni W_pl).
- I 300 × 300 × 80 (t_f = 80) : ELS — fonction seule : classe c, φ_d 0,9186, σ_adm,d 143,92 N/mm².

Valeurs du brief :

| Cas | Valeur | Brief | Site | Écart |
|---|---|---|---|---|
| HE 200 B, feuille automatique | ELS, σ_adm,d (N/mm²) | 115,70 | 115,70 | 0,00 % |
| DEVERSEMENT_CAD, feuille S355 (180 × 180 × 14) | ELS, σ_adm,d (N/mm²) | 147,50 | 147,50 | 0,00 % |
| φ_d = 1 (l = 0,30 m) | ELS, σ_adm,d (N/mm²) | 156,67 | 156,67 | 0,00 % |
| IPE 600, C_d = 1 | ELS, σ_adm,d (N/mm²) | 42,58 | 42,58 | -0,01 % |
| IPE 600, C_d = 1,12, charge sur la semelle haute | ELS, σ_adm,d (N/mm²) | 46,86 | 46,86 | 0,00 % |
| U 140 × 60 × 10 (feuille S235) | ELS, σ_adm,d (N/mm²) | 65,32 | 65,32 | 0,00 % |
| I 600 × 300 × 50 (t_f > 40, h/b > 1,2) | ELS, σ_adm,d (N/mm²) | 119,03 | 119,03 | 0,00 % |
| I 300 × 300 × 80 (t_f = 80) | ELS, σ_adm,d (N/mm²) | 143,92 | 143,92 | 0,00 % |

### 4. Eurocode 3 : 11.2, R2 et R3 (IPE 300, 6 m)

| Cas | Profilé | Nuance | f_y ELS / EC3 | L (m) | Mode | Efforts |
|---|---|---|---|---|---|---|
| R2 et R3, charges au centre de cisaillement | IPE 300 (catalogue) | S235 | 235 / 235 | 6,00 | barre chargée (6.10) | G 2,42 kN/m + 10 kN ; Q 15 kN à 3 m |
| R2 et R3, charges sur la semelle haute | IPE 300 (catalogue) | S235 | 235 / 235 | 6,00 | barre chargée (6.10) | G 2,42 kN/m + 10 kN ; Q 15 kN à 3 m |

| Cas | Sujet | ELS | Eurocode 3 | Écart (points) | Gouverne (ELS / EC3) |
|---|---|---|---|---|---|
| R2 et R3, charges au centre de cisaillement | Flexion composée | — | 46,5 % · vérifié | — | — / E-My |
| R2 et R3, charges au centre de cisaillement | Déversement | 143,7 % · non vérifié | 89,0 % · vérifié | -54,7 | H-DEV / E-Mb |
| R2 et R3, charges au centre de cisaillement | Cisaillement | 143,8 % · non vérifié | 8,0 % · vérifié | -135,8 | H-EQ / E-Vz |
| R2 et R3, charges au centre de cisaillement | Âme sous charges (EN 1993-1-5, commun) | 13,9 % · vérifié | 13,9 % · vérifié | — | A-L00-F |
| R2 et R3, charges sur la semelle haute | Flexion composée | — | 46,5 % · vérifié | — | — / E-My |
| R2 et R3, charges sur la semelle haute | Déversement | 143,7 % · non vérifié | 109,9 % · non vérifié | -33,7 | H-DEV / E-Mb |
| R2 et R3, charges sur la semelle haute | Cisaillement | 143,8 % · non vérifié | 8,0 % · vérifié | -135,8 | H-EQ / E-Vz |
| R2 et R3, charges sur la semelle haute | Âme sous charges (EN 1993-1-5, commun) | 13,9 % · non couvert | 13,9 % · non couvert | — | A-L00-F |

- R2 et R3, charges au centre de cisaillement, déversement : la méthode ELS est plus sévère de 54,7 points (formule des classeurs (λ_e,d, sans gauchissement) contre M_cr du solveur).
- R2 et R3, charges au centre de cisaillement, cisaillement : la méthode ELS est plus sévère de 135,8 points (contrôle hérité H-EQ (σ de flambement et τ moyens, brief § 6.1) contre V_pl,Rd).
- R2 et R3, charges sur la semelle haute, déversement : la méthode ELS est plus sévère de 33,7 points (formule des classeurs (λ_e,d, sans gauchissement) contre M_cr du solveur).
- R2 et R3, charges sur la semelle haute, cisaillement : la méthode ELS est plus sévère de 135,8 points (contrôle hérité H-EQ (σ de flambement et τ moyens, brief § 6.1) contre V_pl,Rd).
- R2 et R3, charges au centre de cisaillement : Catalogue du site : I_t 19,77 cm⁴ au lieu de 20,12 (brief) ; M_cr 0,6 % plus bas, accepté à 1 %.
- R2 et R3, charges sur la semelle haute : Catalogue du site : I_t 19,77 cm⁴ au lieu de 20,12 (brief) ; M_cr 0,6 % plus bas, accepté à 1 %.
- R2 et R3, charges sur la semelle haute : âme non couvert — Charge à 3,00 m : semelle haute sans maintien (EN 1993-1-5, 6.1(1)) — 37 % si elle est tenue ; ajouter un maintien (À VÉRIFIER).

Valeurs du brief :

| Cas | Valeur | Brief | Site | Écart |
|---|---|---|---|---|
| R2 et R3, charges au centre de cisaillement | EC3, M_Ed (kN·m) | 68,70 | 68,70 | 0,00 % |
| R2 et R3, charges au centre de cisaillement | EC3, V_Ed (kN) | 27,80 | 27,80 | 0,00 % |
| R2 et R3, charges au centre de cisaillement | EC3, M_cr (kN·m) | 118,18 | 117,52 | -0,56 % |
| R2 et R3, charges au centre de cisaillement | EC3, taux de déversement | 0,8870 | 0,8899 | 0,32 % |
| R2 et R3, charges sur la semelle haute | EC3, M_Ed (kN·m) | 68,70 | 68,70 | 0,00 % |
| R2 et R3, charges sur la semelle haute | EC3, V_Ed (kN) | 27,80 | 27,80 | 0,00 % |
| R2 et R3, charges sur la semelle haute | EC3, M_cr (kN·m) | 86,98 | 86,36 | -0,72 % |
| R2 et R3, charges sur la semelle haute | EC3, taux de déversement | 1,0930 | 1,0993 | 0,58 % |

R4, l'âme de l'IPE 300 S235 (EN 1993-1-5, commune aux deux méthodes ; la méthode ELS ne traite pas l'âme, brief § 6.6) :

| Valeur | Brief | Site | Écart |
|---|---|---|---|
| F_Rd (kN), (a) s_s = 50 | 283,20 | 283,25 | 0,02 % |
| F_Rd (kN), (a) s_s = 100 | 387,80 | 387,76 | -0,01 % |
| F_Rd (kN), (a) s_s = 150 | 425,10 | 425,14 | 0,01 % |
| F_Rd (kN), (c) s_s = 100, c = 0, contact « oui » | 248,90 | 248,91 | 0,00 % |
| F_Rd (kN), (c) s_s = 100, c = 0, contact « non » (défaut, n° 2) | 199,82 | 199,82 | 0,00 % |
| F_Rd (kN), (c) s_s = 100, c = 50, contact « oui » | 335,90 | 335,92 | 0,01 % |

### 5. PRS (11.8, 11.9)

| Cas | Profilé | Nuance | f_y ELS / EC3 | L (m) | Mode | Efforts |
|---|---|---|---|---|---|---|
| PRS-A, oxycoupé, 10 m | PRS 600x300x15x30 (oxycoupé) | S355 | 355 / 345 | 10,00 | efforts imposés, charge répartie sur la semelle haute | sans charge |
| PRS-A, larges plats, 10 m | PRS 600x300x15x30 (larges plats) | S355 | 355 / 345 | 10,00 | efforts imposés, charge répartie sur la semelle haute | sans charge |
| PRS-B, monosymétrique, 10 m | PRS 600x300x15x30x200x30 (oxycoupé) | S355 | 355 / 345 | 10,00 | efforts imposés, charge répartie sur la semelle haute | sans charge |
| PRS-C, 8 maintiens | PRS 1500 (300 × 30 / 1 420 × 25 / 500 × 50, oxycoupé) | S235 | 235 / 215 | 28,00 | barre chargée, exécution (ψ₀ = 1,0) | béton + travail 23,20 kN/m ; zone 2,25 kN/m |
| PRS-C, maintiens aux tiers | PRS 1500 (300 × 30 / 1 420 × 25 / 500 × 50, oxycoupé) | S235 | 235 / 215 | 28,00 | barre chargée, exécution (ψ₀ = 1,0) | béton + travail 23,20 kN/m ; zone 2,25 kN/m |
| PRS-C, 18 maintiens | PRS 1500 (300 × 30 / 1 420 × 25 / 500 × 50, oxycoupé) | S235 | 235 / 215 | 28,00 | barre chargée, exécution (ψ₀ = 1,0) | béton + travail 23,20 kN/m ; zone 2,25 kN/m |

| Cas | Sujet | ELS | Eurocode 3 | Écart (points) | Gouverne (ELS / EC3) |
|---|---|---|---|---|---|
| PRS-C, 8 maintiens | Flexion composée | — | 80,2 % · vérifié | — | — / E-My |
| PRS-C, 8 maintiens | Déversement | 130,6 % · non vérifié | 96,7 % · à la limite | -33,8 | H-DEV / E-Mb |
| PRS-C, 8 maintiens | Cisaillement | 129,0 % · non vérifié | 11,3 % · vérifié | -117,7 | H-EQ / E-Vz |
| PRS-C, 8 maintiens | Âme sous charges (EN 1993-1-5, commun) | bloqué | bloqué | — | A-L00-EL |
| PRS-C, maintiens aux tiers | Flexion composée | — | 80,2 % · vérifié | — | — / E-My |
| PRS-C, maintiens aux tiers | Déversement | 272,9 % · non vérifié | 175,9 % · non vérifié | -97,0 | H-DEV / E-Mb |
| PRS-C, maintiens aux tiers | Cisaillement | 269,6 % · non vérifié | 11,3 % · vérifié | -258,3 | H-EQ / E-Vz |
| PRS-C, maintiens aux tiers | Âme sous charges (EN 1993-1-5, commun) | bloqué | bloqué | — | A-L00-EL |
| PRS-C, 18 maintiens | Flexion composée | — | 80,2 % · vérifié | — | — / E-My |
| PRS-C, 18 maintiens | Déversement | 99,0 % · hors domaine | 80,2 % · vérifié | -18,8 | H-DEV / E-Mb |
| PRS-C, 18 maintiens | Cisaillement | 97,8 % · hors domaine | 11,3 % · vérifié | -86,5 | H-EQ / E-Vz |
| PRS-C, 18 maintiens | Âme sous charges (EN 1993-1-5, commun) | bloqué | bloqué | — | A-L00-EL |

| Cas | σ_adm,d (N/mm²) | Capacité ELS (kN·m) | × 1,5 (kN·m) | M_b,Rd EC3 (kN·m) | ELS × 1,5 / EC3 |
|---|---|---|---|---|---|
| PRS-A, oxycoupé, 10 m | 90,31 | 499,8 | 749,6 | 797,39 | 0,940 |
| PRS-A, larges plats, 10 m | 78,79 | 436,0 | 654,1 | 797,39 | 0,820 |
| PRS-B, monosymétrique, 10 m | 90,31 | 371,0 | 556,6 | 617,55 | 0,901 |

- PRS-C, 8 maintiens, déversement : la méthode ELS est plus sévère de 33,8 points (formule des classeurs (λ_e,d, sans gauchissement) contre M_cr du solveur ; f_y nominale 235 contre 215 N/mm² selon l'épaisseur).
- PRS-C, 8 maintiens, cisaillement : la méthode ELS est plus sévère de 117,7 points (contrôle hérité H-EQ (σ de flambement et τ moyens, brief § 6.1) contre V_pl,Rd ; f_y nominale 235 contre 215 N/mm² selon l'épaisseur).
- PRS-C, maintiens aux tiers, déversement : la méthode ELS est plus sévère de 97,0 points (formule des classeurs (λ_e,d, sans gauchissement) contre M_cr du solveur ; f_y nominale 235 contre 215 N/mm² selon l'épaisseur).
- PRS-C, maintiens aux tiers, cisaillement : la méthode ELS est plus sévère de 258,3 points (contrôle hérité H-EQ (σ de flambement et τ moyens, brief § 6.1) contre V_pl,Rd ; f_y nominale 235 contre 215 N/mm² selon l'épaisseur).
- PRS-C, 18 maintiens, déversement : la méthode ELS est plus sévère de 18,8 points (formule des classeurs (λ_e,d, sans gauchissement) contre M_cr du solveur ; f_y nominale 235 contre 215 N/mm² selon l'épaisseur).
- PRS-C, 18 maintiens, cisaillement : la méthode ELS est plus sévère de 86,5 points (contrôle hérité H-EQ (σ de flambement et τ moyens, brief § 6.1) contre V_pl,Rd ; f_y nominale 235 contre 215 N/mm² selon l'épaisseur).
- PRS-C, 8 maintiens : âme bloqué — Appui A, appui B : décision impossible : s_s à saisir.
- PRS-C, maintiens aux tiers : âme bloqué — Appui A, appui B : décision impossible : s_s à saisir.
- PRS-C, 18 maintiens : âme bloqué — Appui A, appui B : décision impossible : s_s à saisir.

Valeurs du brief :

| Cas | Valeur | Brief | Site | Écart |
|---|---|---|---|---|
| PRS-A, oxycoupé, 10 m | ELS, σ_adm,d (N/mm²) | 90,31 | 90,31 | 0,00 % |
| PRS-A, oxycoupé, 10 m | EC3, M_b,Rd (kN·m) | 797,39 | 797,39 | 0,00 % |
| PRS-A, oxycoupé, 10 m | ELS, capacité de service σ_adm,d·W_el,y (kN·m) | 499,80 | 499,76 | -0,01 % |
| PRS-A, oxycoupé, 10 m | ELS × 1,5 (kN·m) | 749,70 | 749,65 | -0,01 % |
| PRS-A, larges plats, 10 m | ELS, σ_adm,d (N/mm²) | 78,79 | 78,79 | 0,00 % |
| PRS-B, monosymétrique, 10 m | EC3, M_b,Rd (kN·m) | 617,55 | 617,55 | 0,00 % |
| PRS-C, 8 maintiens | ELS, σ_adm,d (N/mm²) | 89,68 | 89,68 | 0,00 % |
| PRS-C, 8 maintiens | ELS, σ = M/W (N/mm²) | 117,08 | 117,11 | 0,02 % |
| PRS-C, 8 maintiens | EC3, M_Ed (kN·m) | 4 199,30 | 4 200,19 | 0,02 % |
| PRS-C, 8 maintiens | EC3, taux de déversement | 0,9670 | 0,9675 | 0,05 % |
| PRS-C, maintiens aux tiers | ELS, σ_adm,d (N/mm²) | 42,91 | 42,91 | 0,00 % |
| PRS-C, maintiens aux tiers | ELS, σ = M/W (N/mm²) | 117,08 | 117,11 | 0,02 % |
| PRS-C, 18 maintiens | ELS, σ_adm,d (N/mm²) | 118,30 | 118,32 | 0,01 % |
| PRS-C, 18 maintiens | ELS, σ = M/W (N/mm²) | 117,08 | 117,11 | 0,02 % |

### 6. Logiciels du bureau (SCIA S1 à S3)

| Cas | Profilé | Nuance | f_y ELS / EC3 | L (m) | Mode | Efforts |
|---|---|---|---|---|---|---|
| S1, charges sur la semelle haute | IPE 300 (catalogue) | S235 | 235 / 235 | 6,00 | barre chargée (6.10) | G 2,42 kN/m + 10 kN ; Q 15 kN à 3 m |
| S1, charges au centre de cisaillement | IPE 300 (catalogue) | S235 | 235 / 235 | 6,00 | barre chargée (6.10) | G 2,42 kN/m + 10 kN ; Q 15 kN à 3 m |
| S2, poteau HEB 300 | HEB 300 (catalogue) | S235 | — / 225 | 5,00 | efforts de calcul (ELU) donnés | N_Ed 1 400 kN ; M_y,Ed 70 kN·m (ψ = 0) |
| S3, continue IPE 360, 2 × 6 m | IPE 360 (catalogue) | S355 | 355 / 355 | 12,00 | barre chargée (6.10) | G 6 kN/m ; Q1, Q2 10 kN/m |

| Cas | Sujet | ELS | Eurocode 3 | Écart (points) | Gouverne (ELS / EC3) |
|---|---|---|---|---|---|
| S1, charges sur la semelle haute | Flexion composée | — | 46,5 % · vérifié | — | — / E-My |
| S1, charges sur la semelle haute | Déversement | 143,7 % · non vérifié | 109,9 % · non vérifié | -33,7 | H-DEV / E-Mb |
| S1, charges sur la semelle haute | Cisaillement | 143,8 % · non vérifié | 8,0 % · vérifié | -135,8 | H-EQ / E-Vz |
| S1, charges sur la semelle haute | Âme sous charges (EN 1993-1-5, commun) | 61,3 % · non couvert | 61,3 % · non couvert | — | A-L00-EL |
| S1, charges au centre de cisaillement | Flexion composée | — | 46,5 % · vérifié | — | — / E-My |
| S1, charges au centre de cisaillement | Déversement | 143,7 % · non vérifié | 89,0 % · vérifié | -54,7 | H-DEV / E-Mb |
| S1, charges au centre de cisaillement | Cisaillement | 143,8 % · non vérifié | 8,0 % · vérifié | -135,8 | H-EQ / E-Vz |
| S1, charges au centre de cisaillement | Âme sous charges (EN 1993-1-5, commun) | 61,3 % · vérifié | 61,3 % · vérifié | — | A-L00-EL |
| S2, poteau HEB 300 | Flambement, axe faible | _sans objet : SCIA ne donne que les efforts de calcul (aucun effort de service)_ | 57,0 % · vérifié | — | — / E-Nb-z |
| S2, poteau HEB 300 | Flambement, axe fort | _sans objet : SCIA ne donne que les efforts de calcul (aucun effort de service)_ | 45,1 % · vérifié | — | — / E-Nb-y |
| S2, poteau HEB 300 | Flexion composée | _sans objet : SCIA ne donne que les efforts de calcul (aucun effort de service)_ | 64,8 % · vérifié | — | — / E-662 |
| S2, poteau HEB 300 | Déversement | _sans objet : SCIA ne donne que les efforts de calcul (aucun effort de service)_ | 16,6 % · vérifié | — | — / E-Mb |
| S2, poteau HEB 300 | Cisaillement | _sans objet : SCIA ne donne que les efforts de calcul (aucun effort de service)_ | 2,3 % · vérifié | — | — / E-Vz |
| S3, continue IPE 360, 2 × 6 m | Flexion composée | — | 28,7 % · vérifié | — | — / E-My |
| S3, continue IPE 360, 2 × 6 m | Déversement | 110,9 % · non vérifié | 72,7 % · vérifié | -38,3 | H-DEV / E-Mb |
| S3, continue IPE 360, 2 × 6 m | Cisaillement | 111,6 % · non vérifié | 12,0 % · vérifié | -99,6 | H-EQ / E-Vz |
| S3, continue IPE 360, 2 × 6 m | Âme sous charges (EN 1993-1-5, commun) | 64,4 % · vérifié | 64,4 % · vérifié | — | A-L00-EL |

- S1, charges sur la semelle haute, déversement : la méthode ELS est plus sévère de 33,7 points (formule des classeurs (λ_e,d, sans gauchissement) contre M_cr du solveur).
- S1, charges sur la semelle haute, cisaillement : la méthode ELS est plus sévère de 135,8 points (contrôle hérité H-EQ (σ de flambement et τ moyens, brief § 6.1) contre V_pl,Rd).
- S1, charges au centre de cisaillement, déversement : la méthode ELS est plus sévère de 54,7 points (formule des classeurs (λ_e,d, sans gauchissement) contre M_cr du solveur).
- S1, charges au centre de cisaillement, cisaillement : la méthode ELS est plus sévère de 135,8 points (contrôle hérité H-EQ (σ de flambement et τ moyens, brief § 6.1) contre V_pl,Rd).
- S3, continue IPE 360, 2 × 6 m, déversement : la méthode ELS est plus sévère de 38,3 points (formule des classeurs (λ_e,d, sans gauchissement) contre M_cr du solveur).
- S3, continue IPE 360, 2 × 6 m, cisaillement : la méthode ELS est plus sévère de 99,6 points (contrôle hérité H-EQ (σ de flambement et τ moyens, brief § 6.1) contre V_pl,Rd).
- S2, poteau HEB 300 : ELS — sans objet : SCIA ne donne que les efforts de calcul (aucun effort de service).
- S1, charges sur la semelle haute : âme non couvert — Charge à 3,00 m : semelle haute sans maintien (EN 1993-1-5, 6.1(1)) — 43 % si elle est tenue ; ajouter un maintien (À VÉRIFIER).

Valeurs du brief :

| Cas | Valeur | Brief | Site | Écart |
|---|---|---|---|---|
| S1, charges sur la semelle haute | EC3, taux de déversement (non-régression) | 1,0993 | 1,0993 | 0,00 % |
| S1, charges au centre de cisaillement | EC3, taux de déversement (non-régression) | 0,8899 | 0,8899 | 0,00 % |
| S2, poteau HEB 300 | EC3, (6.61) (non-régression, sans borne de C_mi,0) | 0,6000 | 0,5998 | -0,03 % |
| S2, poteau HEB 300 | EC3, (6.62) (non-régression) | 0,6480 | 0,6484 | 0,07 % |
| S3, continue IPE 360, 2 × 6 m | EC3, taux de déversement (non-régression) | 0,7266 | 0,7266 | 0,00 % |

## 2. Écarts avec les classeurs du bureau (brief § 6.7)

Chaque correction est nommée par le moteur (`historique_deversement.CORRECTIONS`) et dite dans la note quand elle joue.

| Cas | Classeur | Module | Correction |
|---|---|---|---|
| FLAMBECBD (HEA 160) | σ = 128,45 N/mm² (φ de l'axe fort) | 133,6 N/mm² | φ_min dans la flexion composée (ici l'axe fort) ; le classeur prenait toujours l'axe faible. |
| FLAMBEBAD (IPE 270) | σ = 43,20 N/mm² (φ de l'axe fort) | 115,7 N/mm² | φ_min dans la flexion composée (ici l'axe fort) ; le classeur prenait toujours l'axe faible. |
| HEB 200, HEA 120, CHS 168,3, REALEX (u = 1) | P_cr sans u | identique à u = 1 ; × u⁻² sinon | P_cr avec (u·L)² ; le classeur Flambement oubliait u. |
| DEVERSEMENT_CAD, feuille S355 | σ_adm,d = 97,8 N/mm² (157 × φ_d) | 147,50 N/mm² | σ_adm,d avec f_y = 355 N/mm² ; la feuille S355 de DEVERSEMENT_CAD prenait 157 N/mm² (S235). |
| DEVERSEMENT_CAD, feuille S235 (U 140) | classe b, 72,06 N/mm² | classe c, 65,32 N/mm² | classe du type de profilé (c) ; DEVERSEMENT_CAD prenait celle d'un I laminé (b). |
| HE 200 B, l = 0,30 m | φ_d = 1,0073 (formule prolongée) | φ_d = 1 ; 156,67 N/mm² | φ_d = 1 pour λ̄ ≤ 0,2 (NBN B 51-002) ; prolongée, la formule des classeurs donne 1,0073. |
| I 600 × 300 × 50 | aucune classe (« 4 < t_f < 8 » toujours faux) | classe c ; 119,03 N/mm² | classe pour 40 < t_f ≤ 80 mm (tableau 12), que les tests des classeurs ne donnaient pas. |
| I 300 × 300 × 80 | aucune classe (« t_f < 8 ») | classe c ; 143,92 N/mm² | classe pour 40 < t_f ≤ 80 mm (tableau 12), que les tests des classeurs ne donnaient pas. |
| feuille « Calcul automatique », t_f > 80 mm | seuils de l'Eurocode (100 mm) | 40 et 80 mm (tableau 12) | seuils de 40 et 80 mm du tableau 12 ; la feuille « Calcul automatique » prenait 100 mm. |
| toute poutre des classeurs Déversement | lignes « axe fort » et « axe faible » | « axe faible » seule (σ_adm 6 à 11 % plus bas) | classe de la ligne « axe faible » seule (doc p. 19), sans la ligne « axe fort » des classeurs. |
| contrôle hérité H11 | V_y + V_z signés | \|V_y\| + \|V_z\| | \|V_y\| + \|V_z\| ; le classeur additionnait les valeurs signées, qui se compensaient. |
| saisie des courbes | aucune case : φ = 1 ; deux : α + β | une courbe par axe (refus) | une seule courbe par axe ; le classeur rendait φ = 1 sans choix, α + β avec deux. |

Données fausses des exemples, dites et NON corrigées (ce ne sont pas des erreurs de formule ; les tests de 11.1 les gardent) :

| Cas | Donnée du classeur | Valeur juste | Effet |
|---|---|---|---|
| CHS 168,3 × 6 | A = 2 520 mm² | 3 059 mm² pour 6 mm | le cas du classeur reste à 2 520 (σ 63,52) ; la lecture du catalogue donne 59,75 |
| HEA 120 Chancel | section des ailes 960 mm² (une aile) | 1 920 mm² | sans effet ici (V = 0) |
| REALEX Ø610 × 12,5 | W = 3 506 cm³ (rayon moyen) | I/(D/2) = 3 435 cm³ | les lectures avec poids propre prennent 3 434,6 |
| FLAMBECBD (HEA 160) | valeurs d'axe faible dans les cases « fort » | I_y 1 673 cm⁴, W_y 220,1 cm³ | le mélange bloque le verdict (test 5) |

Les choix du bureau ne sont pas des corrections : f_y nominale, courbes et classes de chaque classeur, s = 1,5, C = 1, C_d = 1, charges de service.

## 3. Qualification externe (brief § 11.4, règle 7)

| Cas | Extérieur | Site | Attendu | Écart | Lecture |
|---|---|---|---|---|---|
| L1 IPE 300, moment constant | attendu — LTBeam au bureau | 90,47 | 90,47 | — | attendu |
| L2 IPE 300, répartie, centre | attendu — LTBeam au bureau | 102,31 | 102,31 | — | attendu |
| L3 IPE 300, répartie, semelle haute | attendu — LTBeam au bureau | 78,83 | 78,83 | — | attendu |
| L4 IPE 300, répartie, semelle basse | attendu — LTBeam au bureau | 132,69 | 132,69 | — | attendu |
| L5 IPE 300, centrale, centre | attendu — LTBeam au bureau | 123,12 | 123,12 | — | attendu |
| L6 IPE 300, centrale, semelle haute | attendu — LTBeam au bureau | 89,19 | 89,19 | — | attendu |
| L7 IPE 300, centrale, semelle basse | attendu — LTBeam au bureau | 168,95 | 168,95 | — | attendu |
| L8 PRS-A, moment constant | attendu — LTBeam au bureau | 1 385,68 | 1 385,68 | — | attendu |
| L9 PRS-A, répartie, centre | attendu — LTBeam au bureau | 1 567,12 | 1 567,12 | — | attendu |
| L10 PRS-A, répartie, semelle haute | attendu — LTBeam au bureau | 1 191,42 | 1 191,42 | — | attendu |
| L11 PRS-A, répartie, semelle basse | attendu — LTBeam au bureau | 2 059,63 | 2 059,63 | — | attendu |
| L12 PRS-B, moment constant, grande semelle comprimée | attendu — LTBeam au bureau | 1 241,73 | 1 241,73 | — | attendu |
| L13 PRS-B, moment constant, petite semelle comprimée | attendu — LTBeam au bureau | 715,69 | 715,69 | — | attendu |
| L14 PRS-B, répartie, centre de cisaillement | attendu — LTBeam au bureau | 1 234,80 | 1 234,80 | — | attendu |
| L15 PRS-B, répartie, semelle haute | attendu — LTBeam au bureau | 1 086,25 | 1 086,25 | — | attendu |
| L16 PRS-B, répartie, semelle basse | attendu — LTBeam au bureau | 1 803,30 | 1 803,30 | — | attendu |
| L17a IPE 360, répartie + moment d'appui (z_g 180 mm, G 80 769) | 221,10 — LTBeamN, reçu le 16/11/2026 | 223,33 | 223,30 | 1,01 % | à expliquer, reçu : propriétés du catalogue du site (I_t 37,90 au lieu de 37,32 cm⁴) ; avec celles des fabricants, ligne suivante |
| L17b IPE 360, répartie + moment d'appui (z_g 0 mm, G 80 769) | attendu — LTBeam au bureau | 385,46 | 385,37 | — | attendu |
| L17c IPE 360, répartie + moment d'appui (z_g -180 mm, G 80 769) | attendu — LTBeam au bureau | 651,17 | 650,95 | — | attendu |
| L17a, propriétés des fabricants (I_t 37,32 ; I_w 313,6·10³) | 221,10 — LTBeamN, reçu le 16/11/2026 | 221,49 | 221,49 | 0,17 % | conforme, reçu |
| SCIA S1, semelle haute : taux de déversement | 1,08 — SCIA 26, reçu le 16/11/2026 | 1,0993 | — | 1,8 % | à expliquer : −1,7 % : M_cr approché de SCIA et son f (k_c tiré de C₁) |
| SCIA S2 : (6.61) | 0,60 — SCIA 26, reçu le 16/11/2026 | 0,5998 | — | 0,0 % | conforme : borne de C_mi,0 retirée au lot B |
| SCIA S2 : (6.62) | 0,65 — SCIA 26, reçu le 16/11/2026 | 0,6484 | — | -0,2 % | conforme |
| SCIA S3, travée Q1 : taux de déversement | 0,72 — SCIA 26, reçu le 16/11/2026 | 0,7266 | — | 0,9 % | conforme : −1 % : SCIA lit le moment aux nœuds de son maillage |
| ECCS 17 (exemple 1) | bloqué — données à fournir | — | — | — | attendu (Q8) |
| ECCS 18 (exemple 2) | bloqué — données à fournir | — | — | — | attendu (Q8) |
| ECCS 19 (exemple 3) | bloqué — données à fournir | — | — | — | attendu (Q8) |
| ECCS 20 (exemple 4) | bloqué — données à fournir | — | — | — | attendu (Q8) |
| ECCS 22 (ossature de bâtiment) | bloqué — données à fournir | — | — | — | attendu (Q8) |
| JRC EUR 22898 (cisaillement, charges transversales, raidisseurs) | bloqué — données à fournir | — | — | — | attendu (Q14) |

ECCS 21 (exemple 5 ; diagramme de M_y déduit de C_my,0, À VÉRIFIER sur la page) :

| Coefficient | Référence | Site | Écart | Règle 7 |
|---|---|---|---|---|
| μ_y | 1,000 | 1,0000 | 0,00 % | conforme |
| w_y | 1,138 | 1,1381 | 0,01 % | conforme |
| μ_z | 0,937 | 0,9371 | 0,01 % | conforme |
| w_z | 1,500 | 1,5000 | 0,00 % | conforme |
| C_my,0 | 0,999 | 0,9987 | -0,03 % | conforme |
| C_mz,0 | 0,771 | 0,7712 | 0,02 % | conforme |
| a_LT | 0,998 | 0,9982 | 0,02 % | conforme |
| b_LT | 0,043 | 0,0433 | 0,76 % | conforme |
| c_LT | 0,468 | 0,4769 | 1,91 % | à expliquer |
| d_LT | 0,347 | 0,3538 | 1,96 % | à expliquer |
| e_LT | 0,719 | 0,7322 | 1,84 % | à expliquer |
| C_yy | 0,981 | 0,9810 | 0,00 % | conforme |
| M_cr,0 (kN·m) | 899,000 | 900,1414 | 0,13 % | conforme |
| λ̄_0 | 0,757 | 0,7569 | -0,02 % | conforme |

c_LT, d_LT, e_LT ∝ 1/χ_LT : ils impliquent χ_LT ≈ 0,826 dans la référence, le module prend 0,811 ; aucune lecture essayée ne retrouve 0,826 (`tests/test_barre_benchmarks.py`). Les taux de la référence ne sont pas lus.

## 4. Ce qui reste ouvert

### 4.1 À CONFIRMER (`parametres.a_confirmer()`)

- `npl_gamma` — γ_M de n_pl quand γ_M0 ≠ γ_M1 : gamma_M1 (annexe D ANB : γ_M1 (appliqué) ; annexe A corrigée par l'AC:2009 : γ_M0 (affiché) — équivalence non établie).
- Gardé vivant : IPE 500, N + M, γ_M0 = 1,00 et γ_M1 = 1,10 — le moteur le rend dans ses hypothèses (npl_gamma).

### 4.2 DÉCISION BUREAU (brief § 13.2)

| n° | Décision | Défaut | Variante | Où |
|---|---|---|---|---|
| 1 | η₁ de l'interaction (7.2), EN 1993-1-5 | « elastique » | « plastique » | `parametres.eta1_ame` |
| 2 | contact d'extrémité inconnu (type (c)) pris pour | « non » | « oui » | `parametres.contact_inconnu` |
| 3 | A_v,y d'un I ou H laminé | « 2btf » | « A_moins_hwtw » | `parametres.avy_lamine` |
| 4 | méthode de déversement des I et H laminés | « 6.3.2.3 » | « 6.3.2.2 » | `parametres.methode_deversement_lamines` |
| 9 | console en flexion composée (N + M), § 6.3.3 | « non_couvert » | aucune | `parametres.console_NM` |
| 10 | I soudé (PRS) « section soudée équivalente » | non | oui | `parametres.prs_soudee_equivalente` |
| 5 | Méthode ELS : les classeurs du bureau, tels quels | tranché le 16/11/2026 (une seule valeur) | — | brief § 13.2 point 5 |
| 6 | Tube REALEX : fabrication et position | ouvert (ELS 0,831 à 1,018 ; EC3 non couvert) | — | hors code (une donnée) |
| 7 | Seuil « à la limite » | 95 % | — | `acier/garde_corps/membres.SEUIL_ATTENTION` (le seuil du site) |
| 8 | Mise en ligne après le lot C ou après le lot 8 ; lot 9 | après le lot C (défaut) | — | hors code |

Aucun défaut de décision n'est écrit en dur hors de la table (`tests/test_barres_revue.py`).

### 4.3 À VÉRIFIER

Les hypothèses prudentes des rapports des lots B et C, une par ligne : hypothèse, endroit, effet sur un résultat, réponse proposée.

| Hypothèse | Endroit | Effet | Réponse proposée |
|---|---|---|---|
| R2 sous la charge sans maintien : « décision impossible » (EN 1993-1-5 § 6.1(1)), le brief disait « pas de raidisseur » | `acier/barre/ame.py` | âme « bloqué » au lieu d'OK (43 % si la semelle est tenue) | garder la règle (côté sûr) et corriger le brief |
| s_s et c non saisis : calcul à la borne 0 ; « décision impossible » si la borne ne passe pas | `acier/barre/ame.py` | barre neuve « bloqué » (157 % à la borne, 22 % à s_s = 100 mm) | garder ; ou pré-remplir s_s (Q11) |
| Console : L_cr = 2,1·L sur l'axe à bout libre, maintiens intermédiaires non comptés | `acier/barre/moteur.py` | 6,30 m au lieu de 3,77 m (console de 3 m tenue à mi-longueur) | garder ; l'ingénieur saisit sa longueur |
| Débord à bout libre : L_cr par Euler sur la barre entière, au moins 2,1 × le débord ; L_d du débord = L_cr,z, ELS hors domaine | `acier/barre/statique.py` | 7,895 m pour 4 + 2,5 m | garder |
| Formule des classeurs pour les poutres hautes peu maintenues | `acier/barre/historique.py` | IPE 750 de 8 m : 171 % ELS contre 83 % EC3 ; PRS-C : 18 maintiens contre 8 | étendre le message « formule prudente » aux laminés hauts ? |
| Appui « par l'âme », âme élancée : raidisseur d'appui sous la réaction entière | `acier/barre/ame.py` | un raidisseur demandé (§ 5.1(2)) | garder |
| γ_G,sup ou γ_G,inf par cas permanent (chaque cas une source, sauf deux de même répartition) | `acier/barre/combinaisons.py` | plus de combinaisons, bilans plus longs ; jamais moins sévère | garder |
| Action de courte durée : seulement si TOUS ses cas sont déclarés | `acier/barre/combinaisons.py` | ψ₀ = 0,3 plus rare | garder |
| Poids propre d'une barre verticale : q·L entier dans N, en compression | `acier/barre/statique.py` | N plus fort au pied | garder |
| N par tronçon : flambement et (6.61)/(6.62) au plus grand N de la barre | `acier/barre/statique.py` | côté sûr | garder |
| PRS-C : béton frais en nature « exécution » (γ_Q 1,5, ψ₀ 1,0) | `tests/test_note_profil.py` | M_Ed 4 199,3 (3 893,5 en permanent) | confirmer avec la note du bureau |
| Verdict ELS quand le contrôle hérité H-EQ dépasse la principale : la principale en tête, « + contrainte équivalente » dessous | `api_web/profil_resultat.py` | pastille 111 %, H-EQ 113 % écrit dessous | garder ; H-EQ ne décide jamais seul (brief § 6.1) |
| Norme produit des plats d'un PRS en S420/S460 : ligne « laminé » de la nuance | `acier/barre/nuances.py` | N/NL par défaut | garder |
| Cornière : u-u et v-v prennent la plus grande longueur saisie | `acier/barre/historique.py` | v-v gouverne | garder |
| Bibliothèque des PRS dans le navigateur ; la page Catalogue ne la lit pas | `api_web/profil_prs.py` | deux bibliothèques | trancher (question de C6) |
| z_j et ψ_f du tableau du PRS pour la semelle haute comprimée | `api_web/profil_prs.py` | le moteur, lui, les recalcule par tronçon | ajouter le cas inverse ? |
| Inclinée sans angle saisi : α = 45° écrit à l'écran | `api_web/profil_mots.py` | poids propre réparti à 45° | garder (champ visible) |
| Charge dont la valeur de tableau manque (classe à fournir, ψ₀ d'exécution) : calcul « bloqué » | `acier/barre/combinaisons.py` | aucun verdict deviné | garder |
| Décision n° 9 : console en N + M « non couvert », aucune variante calculable | `acier/barre/parametres.py` | E-661 et E-662 non couverts | le bureau fixe une méthode |
| NBN B 51-002, tableau 12 : S460 lu dans la colonne f_y ≤ 430 | `acier/barre/historique_deversement.py` | classe du S460 | lire la colonne |
| Charges à hauteurs mêlées : M_cr par tronçon, k_c = 1 hors tableau 6.6 | `acier/barre/ec3_deversement.py` | S1 1,099 (SCIA 1,08) | garder |
| M < 0 en efforts imposés : le sens de la charge est inconnu, le plus petit M_cr des deux sens | `acier/barre/ec3_deversement.py` | IPE 300, 6 m, −60 kN·m : 0,705 → 1,023 | garder ; une ligne « À VÉRIFIER » dans la note |
| Type (c) à la borne : branche m₂ = 0 | `acier/barre/ec3_ame.py` | jusqu'à −24 % sur des profilés hauts en S355 | garder |
| Semelle en console : ψ et α à leur borne prudente (classe 3 à 13,77·ε sous M_z) | `acier/barre/ec3_classe.py` | six laminés en classe 4 sous M_z, taux inchangés | garder |
| Correction 4 (σ_adm avec la f_y de la nuance) dite pour S355 seulement ; C_d borné à [1 ; 3] | `acier/barre/historique_deversement.py` | texte de la note | garder |
| Tableau E.1 ANB : colonne C₃ (k_z = 1) relue sur un texte extrait abîmé | `acier/barre/mcr.py` | C₃ des moments d'extrémité | relire sur l'image du tableau |
| Gorge d'un PRS non saisie : congé du raidisseur pris à t_w | `acier/barre/raidisseurs.py` | chanfrein du raidisseur | garder |
| I soudé non PRS : la clé prs_soudee_equivalente vaut pour tout I soudé (§ 6.3.2.2 par défaut) | `acier/barre/parametres.py` | I soudé h/b = 3, 6 m : M_b,Rd −1,8 à −5,4 % | garder |

Recherche de « à vérifier » (sans accents ni casse) dans le moteur et l'écran du profil : 87 mentions dans 18 fichiers, chacun classé dans la liste de `tools/comparaison_barres.py` (un fichier neuf fait rougir le test) :

- `acier/barre/ame.py` (26) : âme : borne s_s = 0, maintien de la semelle chargée, appui par l'âme, raidisseurs (B4, C1)
- `acier/barre/combinaisons.py` (3) : γ_G,sup ou γ_G,inf par source permanente (C2)
- `acier/barre/ec3_ame.py` (2) : règle des branches de m₂ ; borne haute des forces rapprochées (B4)
- `acier/barre/ec3_classe.py` (1) : ψ et α d'une semelle en console pris à leur borne prudente (B2)
- `acier/barre/ec3_deversement.py` (2) : M < 0 en efforts imposés : le plus petit M_cr des deux sens (intégration B)
- `acier/barre/ec3_interaction.py` (1) : décision n° 9 : console en N + M, aucune méthode fixée
- `acier/barre/historique.py` (8) : cornière u-u et v-v ; console L_d = 2,1·L ; colonne f_y ≤ 430 du tableau 12 (B1, C1b)
- `acier/barre/historique_deversement.py` (4) : correction 4 dite en S355 seule ; C_d borné à [1 ; 3] ; colonne f_y > 430 ; ligne « axe fort » (B1)
- `acier/barre/mcr.py` (1) : colonne C₃ du tableau E.1 ANB, relue sur un texte extrait abîmé
- `acier/barre/moteur.py` (17) : console 2,1·L, débords, sens de M, semelle comprimée, « le cisaillement gouverne » (intégration B, C1)
- `acier/barre/nuances.py` (2) : valeurs M/ML et norme produit des plats d'un PRS en S420 et S460 (C1b)
- `acier/barre/parametres.py` (2) : décision n° 9 sans variante calculable
- `acier/barre/raidisseurs.py` (1) : gorge d'un PRS non saisie : congé pris à t_w (B4)
- `acier/barre/statique.py` (4) : « sections à vérifier » (le mot, pas une hypothèse) ; poids propre vertical, N par tronçon, débord (C1)
- `api_web/profil.py` (1) : « à vérifier (v) », le niveau de couleur (le mot, pas une hypothèse)
- `api_web/profil_mots.py` (4) : angle de 45° d'une barre inclinée (C3) ; le niveau « v » (le mot)
- `api_web/profil_prs.py` (1) : la page Catalogue ne lit pas la bibliothèque des PRS (C6)
- `api_web/profil_resultat.py` (7) : le mot « À vérifier » de l'écran (pas une hypothèse)

### 4.4 Questions au bureau (`docs/acier/BARRES_ACIER.md` § 10)

**Tranchées par le brief v12** : Q1 méthode 1 et borne de C_mi,0 ; Q2 ψ belges ; Q3 f_u et A_v,y ; Q4 méthode ELS des classeurs ; Q7 fichiers hors du dépôt ; Q13 raidisseurs proposés ; Q9 et Q11 caduques (impression au choix ; s_s à la borne).

**Ouvertes** : Q5 = décision n° 6 (REALEX) ; Q6 = décision n° 7 (seuil 95 %) ; Q8 pages SCIA 17 à 22 ; Q10 η₁ (décision n° 1) ; Q12 appuis posés ou par l'âme ; Q14 exemples JRC.

**Nouvelles des lots B et C** : R2 sans maintien sous la charge (brief à corriger ?) ; barre neuve : pré-remplir s_s ? ; message « formule prudente » pour les laminés hauts ? ; Catalogue et bibliothèque des PRS : une seule ? ; béton frais du PRS-C en exécution ? ; écran Streamlit et I_t des PRS (+6,2 %) : le retirer ?.

### 4.5 Hors du module (nommés, non traités)

- γ_Mu = 1,1 des attaches (assemblages) : à pointer sur l'ANB de l'EN 1993-1-8.
- α_cc = 0,85 (béton) : la valeur belge, à pointer.
- Garde-corps, cornière, enrobage : leurs suites sont aux chantiers de leurs modules.
- RBAR : le moteur de plans, hors des barres.
- Ancien écran Streamlit (`modules/tableau_profiles.py`) : I_t d'un PRS = Σ b·t³/3 (+6,2 %, non conservatif) ; à retirer ou à porter sur `prs.py`.

## 5. Avant / après : ce qui change à la réouverture d'une note enregistrée

Une note se relit sans être réécrite (format VERSION 2, clés neuves avec leur défaut, aucune migration). À sa réouverture, le site recalcule avec le moteur du jour :

| Changement | Lot | Effet mesuré | Preuve |
|---|---|---|---|
| f_y nominale en méthode ELS | B1 | HEB 300 S235 : 225 → 235 N/mm² | `tests/test_barre_moteur.py` |
| Déversement ELS des classeurs (au lieu du fascicule) | B1 | barre neuve 74 → 88 % ; IPE 750x134 8 m 93 → 171 % | `tests/test_barre_historique.py` |
| C_mi,0 sans borne inférieure | B3 | S2 : 0,631 / 0,667 → 0,600 / 0,648 | `tests/test_barre_benchmarks.py` |
| Console : 2,1·L, bout libre non compté en ELS | intégration B | console IPE 300 3 m : 94,9 % ok → 185 % hors domaine (ELS) | `tests/test_barre_moteur.py` |
| Statut « bloqué » (PRS sans bords, classe à fournir, s_s à saisir, ψ₀ à fournir) | C0, C2 | le verdict attend la donnée, le taux reste montré | `tests/test_barre_moteur.py` |
| Les deux méthodes toujours calculées ; l'impression au choix | C0, C5 | une note « ec3 » seule montre aussi l'ELS à l'écran | `tests/test_note_profil.py` |
| γ_G,sup ou γ_G,inf par cas permanent | C2 | bilans plus longs, aucun verdict changé sur les photos | `tests/test_barre_combinaisons.py` |
| Poids propre en efforts imposés (deux interrupteurs, éteints par défaut) | C1 | aucun pour une note ancienne | `tests/test_barre_moteur.py` |
| s_s et c non saisis pris à leur borne 0 | B4 | barre neuve : « s_s à saisir », 157 % à la borne | `tests/test_barre_ame.py` |
| Longueurs copiées au passage en efforts imposés remises en automatique | C0 | une note ancienne (sans « longueurs_copiees ») garde ses longueurs, aucun changement ; console neuve : L_z 3 → 6,3 m | `tests/test_api_web_profil.py` |
| Tolérance de 5 % retirée | C0 | une note qui la cochait le dit, le taux n'est plus toléré | `tests/test_note_profil.py` |
| Format de la note | B4, intégration B | VERSION 2 ; clé zero_saisi (0 saisi ≠ non saisi) ; aucune migration | `tests/test_api_web_profil.py` |
