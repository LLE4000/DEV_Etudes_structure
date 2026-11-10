# Lot Q — quatre modules refaits : poutres bois, maturité du béton, taux d'armature, feu (10/11/2026)

- **Statut** : conception arrêtée le 10/11/2026 au soir, AVANT le code ; la réalisation et ses écarts s'écrivent à la
  fin de ce fichier (« Le lot réalisé »).
- **Demande de l'ingénieur (10/11/2026)** : « Fais bois, feu, taux, maturité. Vérifie les formules, car tout n'était pas
  juste. Pas grave si on n'a pas exactement la même chose que sur Streamlit : tu as le principe de ce que je veux, tu
  peux tout réorganiser, supprimer ce qui n'est pas utile. Comprends pourquoi les modules existent et fais l'interface
  correcte sur le site. Sors-moi des modules professionnels et justes, ergonomiques. »
- **Ce que cela change à la règle du site.** Jusqu'ici (`MODULES_SITE.md` § 5, règles 1 et 2) : « on extrait, on ne
  réécrit pas », les défauts de l'écran Streamlit sont REPRODUITS et signalés. Pour ces quatre modules, la règle est
  levée — comme pour les cornières le 10/11 : **le site calcule juste, il ne reproduit plus l'écran Streamlit**. Les
  moteurs sont réécrits sur la norme ; les tests de PARITÉ avec `modules/*.py` sont remplacés par des contrôles à la
  main (chaque formule recalculée à part), des invariants et des mutations. Les écrans Streamlit (`modules/poutre_bois.py`,
  `age_beton.py`, `taux_armature.py`, `enrobage.py`) ne sont PAS modifiés : ils gardent leurs défauts
  (`DEFAUTS_STREAMLIT.md`), le site ne les suit plus.
- **Les notes déjà enregistrées** de ces modules sont des essais (l'ingénieur, 10/11 : « je les efface ») : aucune
  conversion. Une note d'un ancien format s'ouvre sur un élément neuf qui garde les NOMS, et un avis le dit une fois
  (« note d'avant le 10/11/2026 : le module a été refait, les données sont à ressaisir ») — jamais une erreur.
- **Le format** : l'établi COMPACT de la poutre BA et des cornières (`api_web/etabli_compact.py`, `js/compact.js`) — à
  gauche la ligne de l'élément (nom, liste, copier, supprimer) et ses RUBRIQUES numérotées (la première ouverte, la
  dernière « Hypothèses » secondaire), au centre les vues avec leur en-tête technique, à droite la tête, le verdict et des
  BLOCS de vérifications sur deux lignes, chacune ouvrant sa fenêtre de calcul « comme à la main ». Modèle à lire :
  `api_web/corniere.py`. Trois formats (PC, tablette, téléphone), mêmes fonctions (R1).

Pour chaque module : **pourquoi il existe**, **ce qui était faux**, **ce qui disparaît**, **la méthode** (formules et
sources), **l'écran**, **les preuves**.

---

## 1. Poutres bois (`bois/poutre`, `api_web/bois.py`, page `poutre-bois`)

### 1.1 Pourquoi il existe

Vérifier vite une pièce de bois fléchie sur deux appuis : une gîte de plancher, un chevron, un madrier de plateforme, une
poutre de coffrage ou d'étaiement en bois massif ou lamellé-collé. La question posée est toujours la même : « cette
section tient-elle sur cette portée sous ces charges, et sinon laquelle prendre ? ». La réponse attendue : un verdict, la
vérification qui gouverne, et la section la plus légère qui tient.

### 1.2 Ce qui était faux (écran Streamlit, reproduit par l'ancien moteur)

| Défaut | Effet mesuré |
|---|---|
| flèches calculées avec q × 1 000 (1 kN/m vaut déjà 1 N/mm) | flèches 1 000 fois trop grandes : **toute poutre sort à 999 %**, le module était inutilisable à l'ELS |
| seule la combinaison γ_G·G + γ_Q·Q, avec le k_mod de la durée choisie | la combinaison G seule (k_mod permanent 0,6) gouverne dès que G domine : 92 % affichés pour 104 % réels |
| cisaillement sans k_cr = 0,67 | τ sous-estimé d'un facteur 1,49 |
| flèche finale (1 + k_def)·w(G + ψ₂Q) | oublie (1 − ψ₂)·w_inst,Q (EC5 §2.2.3) |
| compression d'appui sous G + Q non pondérées, comparée à une résistance de calcul | non conservatif à l'ELU |
| poids propre jamais compté ; ni déversement, ni flèche d'effort tranchant | — |
| base des classes : mélange d'éditions de l'EN 338 (f_v,k de C14 à C22 de 2003, f_t,0,k de 2009, f_c,90,k trop forts) | résistances fausses de −50 % à +10 % selon la classe |
| k_h < 1 appliqué au-dessus de 150 mm | côté sûr, mais faux |
| section proposée 38×150 sur 4 m | 782 % à l'ouverture |

### 1.3 Ce qui disparaît, ce qui arrive

Disparaissent : le choix à la main de la « durée dominante » (elle se déduit de la NATURE de la charge variable), les
quatre limites de flèche sans rapport avec la norme, la case « appui » décochée par défaut (un appui se vérifie toujours).
Arrivent : le lamellé-collé, le poids propre, les deux combinaisons ELU, le déversement (maintien latéral), la
compression d'appui selon l'EC5, la flèche d'effort tranchant, la saisie en kN/m² avec entraxe (gîtes), l'effet système.

### 1.4 La méthode — EN 1995-1-1:2004 + A1:2008 + A2:2014, annexe belge NBN EN 1995-1-1 ANB ; EN 338:2016 ; EN 14080:2013

Unités du moteur : N, mm (saisie : mm, m, kN/m ou kN/m²).

**Matériaux** (`bois/poutre/donnees.py`, une table par norme, la source et l'édition écrites à côté) : résineux C14 à
C50 (EN 338:2016, tableau 1) ; lamellé-collé GL24h, GL28h, GL32h, GL24c, GL28c, GL32c (EN 14080:2013, tableaux 4 et 5).
Par classe : f_m,k, f_v,k, f_c,90,k, E_0,mean, E_0,05, G_mean, ρ_k, ρ_mean (f_t,0,k, f_c,0,k gardés pour la fiche).
Valeurs au § 1.7 (croisées sur deux sources le 10/11/2026).

**Coefficients** : γ_M = 1,30 (bois massif), 1,25 (lamellé-collé) — ANB ; k_mod (tableau 3.1) : classes de service 1 et
2 : 0,60 / 0,70 / 0,80 / 0,90 / 1,10 (permanent, long, moyen, court terme, instantané) ; classe 3 : 0,50 / 0,55 / 0,65 /
0,70 / 0,90 ; k_def (tableau 3.2) : 0,60 / 0,80 / 2,00 ; k_cr = 0,67 (§6.1.7(2), massif et lamellé) ; k_sys = 1,1 si
« effet système » (§6.6 : pièces semblables, entraxe ≤ 60 cm environ, liées par un platelage continu), appliqué à f_m,d
et f_v,d.

**Nature de la charge variable** → classe de durée (tableau 2.2 et ANB) et ψ₂ (NBN EN 1990 ANB) :

| Nature | Durée | ψ₂ |
|---|---|---|
| A · habitation ; B · bureaux | moyen terme | 0,3 |
| C · réunion ; D · commerces | moyen terme | 0,6 |
| E · stockage | long terme | 0,8 |
| H · toiture (entretien) | court terme | 0 |
| neige (Belgique, altitude ≤ 1 000 m) | court terme | 0 |
| charge de chantier, coffrage (moins d'une semaine) | court terme | 0 |
| autre | au choix | au choix |

**Charges** : g = G + g_pp (poids propre ρ_mean·b·h·9,81, compté par défaut), q = Q ; en mode surfacique G = g_s·e,
Q = q_s·e (e : entraxe).

**ELU** (NBN EN 1990, 6.10), DEUX combinaisons, chacune avec SON k_mod (§3.1.3(2) : celui de l'action de plus courte
durée) : (1) p_d = γ_G·g, k_mod permanent ; (2) p_d = γ_G·g + γ_Q·q, k_mod de la nature de Q. Pour chacune M_d = p_d·L²/8,
V_d = p_d·L/2 (à l'axe de l'appui, sans la réduction du §6.1.7(3) : côté sûr). Chaque vérification retient la
combinaison au taux le plus fort et la nomme.

1. **Flexion** (§6.1.6, §6.3.3) : σ_m,d = M_d/W ≤ k_crit·f_m,d ; f_m,d = k_mod·k_h·k_sys·f_m,k/γ_M ;
   k_h = min((150/h)^0,2 ; 1,3) si h < 150 mm, 1 sinon (massif, §3.2(3)) ; min((600/h)^0,1 ; 1,1) si h < 600 mm, 1 sinon
   (lamellé, §3.3(3)).
   **Déversement** : maintien « continu » de la fibre comprimée (plancher ou platelage fixé) : k_crit = 1 ; « aux appuis
   seulement » : l_ef = 0,9·L + 2·h (tableau 6.1 : charge répartie, appliquée sur la fibre comprimée) ; « entretoises »
   à l'entraxe e_m : l_ef = e_m + 2·h (moment pris constant entre entretoises : côté sûr) ;
   σ_m,crit = 0,78·b²·E_0,05/(h·l_ef) ; λ_rel,m = √(f_m,k/σ_m,crit) ; k_crit = 1 (λ ≤ 0,75), 1,56 − 0,75·λ
   (0,75 < λ ≤ 1,4), 1/λ² (λ > 1,4). Les appuis sont supposés maintenus en torsion (condition du tableau 6.1 : dit à l'écran).
2. **Cisaillement** (§6.1.7) : τ_d = 1,5·V_d/(k_cr·b·h) ≤ f_v,d = k_mod·k_sys·f_v,k/γ_M.
3. **Compression à l'appui** (§6.1.5) : σ_c,90,d = V_d/(b·l_ef,a) ≤ k_c,90·f_c,90,d ; f_c,90,d = k_mod·f_c,90,k/γ_M ;
   l_ef,a = a + min(30 ; a) côté travée + min(30 ; a ; débord) côté about ; k_c,90 = 1,5 (massif résineux), 1,75
   (lamellé résineux, si a ≤ 400 mm), ramené à 1,0 si la distance libre entre appuis L − a < 2·h.
   Toujours vérifiée (a = 100 mm par défaut) ; on peut la retirer de la note, le résultat le dit alors.

**ELS** (§2.2.3, §7.2), combinaison caractéristique, E_0,mean et G_mean :
w(p) = 5·p·L⁴/(384·E·I) + 1,2·p·L²/(8·G·A) (flexion + effort tranchant, écrits séparément) ;
w_inst = w(g) + w(q) ≤ L/300 (au choix : 200, 250, 300, 350, 400, 500) ;
w_fin = w(g)·(1 + k_def) + w(q)·(1 + ψ₂·k_def) ; w_net,fin = w_fin − w_c (contre-flèche, 0 par défaut) ≤ L/250 (au
choix : 150, 200, 250, 300, 350).

**Verdict** : le taux le plus fort ; couleurs du bureau (GC10). **Sélecteur de section** : les sections standard du
matériau (massif : liste du commerce belge ; lamellé : largeurs et hauteurs courantes), chacune recalculée, triées par
aire ; ✓ la retenue, ★ la plus légère qui tient.

**Non couvert, dit à l'écran** : charges concentrées, porte-à-faux, poutres continues, entailles d'appui (k_v),
vibrations des planchers (§7.3), flexion déviée ou composée, feu, perçages ; bois mis en œuvre humide (k_def + 1,0).

### 1.5 L'écran

- **Rubriques** : 1 Poutre (bois : menu des classes, massif puis lamellé ; section : Standard / Sur mesure, b × h ;
  portée L) · 2 Charges (kN/m ou kN/m² × entraxe ; G hors poids propre, Q, nature de Q ; case « poids propre compté ») ·
  3 Appuis et maintien (longueur d'appui a, débord ; maintien latéral : continu / aux appuis / entretoises + entraxe) ·
  4 Hypothèses (classe de service 1, 2, 3 ; γ_G, γ_Q ; ψ₂ et durée si « autre » ; effet système ; limites de flèche ;
  contre-flèche).
- **Résultats** : verdict (taux, vérification gouvernante, combinaison) ; bloc « ELU · résistance » : flexion (pièce : la
  section, qui ouvre le sélecteur), cisaillement, appui ; bloc « ELS · flèches » : instantanée, finale nette ; pied :
  masse, poids propre.
- **Vues** : « Schéma et efforts » (poutre à l'échelle des longueurs, appuis à leur largeur, charges G et Q, V et M de
  la combinaison gouvernante), « Déformée » (w_inst, w_net,fin et leurs limites), « Section » (à l'échelle, fil du bois,
  maintien latéral).
- **Défaut à l'ouverture** : C24, 75 × 225, L = 4,00 m, G = 0,6 et Q = 0,8 kN/m (habitation), classe de service 1, appui
  100 mm, maintien continu : conforme, et le sélecteur propose plus léger.

### 1.6 Les preuves

`tests/test_poutre_bois_moteur.py` réécrit : un exemple déroulé à la main (chaque grandeur recalculée avec `math` seul),
les deux combinaisons (G seule gouverne quand Q est petit), k_h, k_crit aux trois branches, k_c,90, les limites, le
sélecteur, les refus, les invariants (b doublé : contraintes divisées par deux…), 12 mutations au moins (k_cr retiré,
k_mod de la mauvaise durée, poids propre oublié, 5/384 changé, ψ₂ oublié…) qui rougissent toutes.
`tests/test_api_web_bois.py` : l'écran = le moteur. Gel : cas réécrits, regel accepté avec la raison.

### 1.7 Valeurs caractéristiques retenues (N/mm², kg/m³)

EN 338:2016, tableau 1 (résineux, essais de flexion sur chant) :

| | C14 | C16 | C18 | C20 | C22 | C24 | C27 | C30 | C35 | C40 | C45 | C50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f_m,k | 14 | 16 | 18 | 20 | 22 | 24 | 27 | 30 | 35 | 40 | 45 | 50 |
| f_t,0,k | 7,2 | 8,5 | 10 | 11,5 | 13 | 14,5 | 16,5 | 19 | 22,5 | 26 | 30 | 33,5 |
| f_c,0,k | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 24 | 25 | 27 | 29 | 30 |
| f_c,90,k | 2,0 | 2,2 | 2,2 | 2,3 | 2,4 | 2,5 | 2,5 | 2,7 | 2,7 | 2,8 | 2,9 | 3,0 |
| f_v,k | 3,0 | 3,2 | 3,4 | 3,6 | 3,8 | 4,0 | 4,0 | 4,0 | 4,0 | 4,0 | 4,0 | 4,0 |
| E_0,mean | 7 000 | 8 000 | 9 000 | 9 500 | 10 000 | 11 000 | 11 500 | 12 000 | 13 000 | 14 000 | 15 000 | 16 000 |
| E_0,05 | 4 700 | 5 400 | 6 000 | 6 400 | 6 700 | 7 400 | 7 700 | 8 000 | 8 700 | 9 400 | 10 100 | 10 700 |
| G_mean | 440 | 500 | 560 | 590 | 630 | 690 | 720 | 750 | 810 | 880 | 940 | 1 000 |
| ρ_k | 290 | 310 | 320 | 330 | 340 | 350 | 360 | 380 | 390 | 400 | 410 | 430 |
| ρ_mean | 350 | 370 | 380 | 400 | 410 | 420 | 430 | 460 | 470 | 480 | 490 | 520 |

EN 14080:2013 (lamellé-collé) :

| | GL24h | GL28h | GL32h | GL24c | GL28c | GL32c |
|---|---|---|---|---|---|---|
| f_m,k | 24 | 28 | 32 | 24 | 28 | 32 |
| f_t,0,k | 19,2 | 22,3 | 25,6 | 17 | 19,5 | 19,5 |
| f_c,0,k | 24 | 28 | 32 | 21,5 | 24 | 24,5 |
| f_c,90,k | 2,5 | 2,5 | 2,5 | 2,5 | 2,5 | 2,5 |
| f_v,k | 3,5 | 3,5 | 3,5 | 3,5 | 3,5 | 3,5 |
| E_0,mean | 11 500 | 12 600 | 14 200 | 11 000 | 12 500 | 13 500 |
| E_0,05 | 9 600 | 10 500 | 11 800 | 9 100 | 10 400 | 11 200 |
| G_mean | 650 | 650 | 650 | 650 | 650 | 650 |
| ρ_k | 385 | 425 | 440 | 365 | 390 | 400 |
| ρ_mean | 420 | 460 | 490 | 400 | 420 | 440 |

---

## 2. Maturité du béton (`beton/maturite`, `api_web/age.py`, page `maturite-beton`)

### 2.1 Pourquoi il existe

Décider d'une opération de chantier qui dépend de la résistance du béton jeune : décoffrer un voile, désétayer une
dalle, mettre en charge, protéger du gel. La question : « mon béton (classe, ciment) coulé tel jour, par ce temps,
aura-t-il X MPa tel jour ? Sinon, quand ? ». La réponse attendue : une DATE par objectif, avec une variante prudente
(prévisions trop optimistes), et un verdict quand une date d'opération est fixée.

### 2.2 Ce qui était faux

| Défaut | Effet mesuré |
|---|---|
| le seuil et la classe minimale calculés à la température MOYENNE du relevé | 7 j à 2 °C puis 7 j à 20 °C, seuil 15 MPa : annoncé « après 4,8 j », atteint en réalité au jour 8 — **trois jours trop tôt pour un décoffrage** |
| le premier jour compté en entier, quelle que soit l'heure du coulage | jusqu'à une demi-journée de maturité en trop |
| la date du seuil tronquée au jour | 7,93 j → « le 09/11 » au lieu du 09/11 au soir |
| l'essai sur éprouvette lu comme une résistance caractéristique | f_ck,28 surestimé de 8/β_cc : 30 MPa annoncés pour 19,7 |
| f_ck(t) négatif affiché et tracé au très jeune âge | — |
| des jours de gel comptés comme du durcissement (formule appliquée sous 0 °C, hors de son domaine) | maturité surestimée par temps froid |

### 2.3 Ce qui disparaît, ce qui arrive

Disparaissent : les cinq « analyses » à cocher (① à ⑤), le comparateur de classes (④, remplacé par le sélecteur de
classe), le « seuil » unique, numpy (le module démarre sans lui). Arrivent : des OBJECTIFS nommés (autant qu'on veut),
chacun daté ; l'heure du coulage ; l'intégration jour par jour partout ; la variante prudente sur chaque date ;
l'alerte de gel avant 5 MPa ; l'essai sur éprouvettes comparé à la prévision, avec calage de la courbe.

### 2.4 La méthode — EN 1992-1-1:2004 §3.1.2 et annexe B ; EN 13670:2009 §8.5

    f_cm(t) = β_cc(t_T)·f_cm                      f_cm = f_ck + 8                                   (3.1)
    β_cc(t_T) = exp{ s·[1 − √(28/t_T)] }          s = 0,20 (R) · 0,25 (N) · 0,38 (S)                (3.2)
    f_ck(t) = f_cm(t) − 8  pour t_T < 28 j ;  f_ck pour t_T ≥ 28 j                                  §3.1.2(5)
    t_T = Σ Δt_i · k_T(T_i)        k_T(T) = exp{ −(E_a/R)·[1/(273 + T) − 1/293] }    E_a/R = 4 000 K   (B.10)

- Ciments : R = CEM 42,5 R, 52,5 N, 52,5 R ; N = CEM 32,5 R, 42,5 N ; S = CEM 32,5 N (§3.1.2(6)). Les ciments au
  laitier (CEM III), courants en Belgique, durcissent plus lentement à froid : l'énergie d'activation se règle dans les
  hypothèses (E_a de 20 à 60 kJ/mol ; EC2 : 33,26), et l'écran le dit.
- (B.10) écrit « 13,65 » pour 4000/293 = 13,652 : le moteur prend la référence exacte (k_T = 1 à 20 °C), 0,2 % d'écart.
- **Domaine de (B.10) : 0 à 80 °C.** Un jour dont la température moyenne est sous 0 °C ne compte AUCUNE maturité
  (k_T = 0, côté sûr), et il est signalé.
- **Le temps** : t compté depuis la FIN DU BÉTONNAGE (date et heure ; midi si l'heure n'est pas donnée) ; la température
  est constante par jour civil ; t_T(t) est donc linéaire par morceaux, intégré exactement (le premier jour ne compte
  que pour sa fraction restante). Au-delà du dernier jour connu, le relevé est PROLONGÉ à la moyenne de ses cinq
  derniers jours (comme la météo), et c'est dit.
- **Un objectif** : une résistance cible (MPa, ou % de f_ck) comparée à f_ck(t) — la résistance CARACTÉRISTIQUE, côté
  sûr, par défaut — ou à f_cm(t) (la moyenne) au choix dans les hypothèses. L'âge où il est atteint a une forme fermée,
  t_T,req = 28/(1 − ln β_req/s)² avec β_req = (cible + 8)/f_cm (ou cible/f_cm), puis t se lit en inversant t_T(t) :
  plus de dichotomie. Une cible au-dessus de la classe est « non atteignable ». La date et l'heure sont données à
  l'heure près (« le 15/11 vers 14 h »), jamais tronquées.
- **Variante prudente** : les jours PRÉVUS et PROLONGÉS (tous, en température constante) abaissés de la marge (3 °C par
  défaut) ; chaque objectif donne ses deux dates.
- **Opération datée** : si l'objectif porte un âge prévu, η = cible/f(t_prévu) ; conforme si η ≤ 1 ; « conforme,
  attention » si la variante prudente ne passe pas.
- **Domaine de l'EC2** : pour t_T ≤ 3 j, « des valeurs plus précises doivent se fonder sur des essais » (§3.1.2(5)) :
  tout résultat dans ce domaine est marqué « indicatif — à confirmer par essais ».
- **Gel** (EN 13670 §8.5) : le béton ne doit pas geler avant d'avoir 5 MPa. Le moteur cherche le premier jour de gel
  (température moyenne ≤ 0 °C, ou minimale horaire < 0 °C si le détail horaire est chargé) et le compare à l'âge où
  f_cm(t) atteint 5 MPa : alerte « protéger du gel » si le gel vient avant.
- **Essai** : la moyenne des éprouvettes à un âge donné (cube 150 mm ramené au cylindre par f_ck/f_ck,cube de la
  classe), comparée à f_cm(t) prévu : écart en %. « Caler la courbe sur l'essai » recalcule s pour que la courbe passe
  par la mesure (s_cal = ln(f_mes/f_cm)/(1 − √(28/t_T)), borné à [0,10 ; 0,80]) : toutes les dates suivent.
- **Sélecteur de classe** : chaque classe (C16/20 à C50/60) avec le même ciment et les mêmes températures : l'âge où le
  premier objectif est atteint ; ✓ la classe du coulage, ★ la plus petite qui tient l'âge prévu.

Ce que l'outil N'EST PAS, écrit sous le résultat : une estimation par la loi de l'EC2 avec la température de l'AIR ; pour
une opération critique (désétaiement, précontrainte), la résistance se confirme par des éprouvettes conservées sur
chantier ou une maturométrie étalonnée.

### 2.5 L'écran

- **Rubriques** : 1 Béton (classe, ciment R/N/S, date et heure de fin de bétonnage) · 2 Températures (Constante /
  Relevé / Météo du lieu : inchangé, Open-Meteo à la demande) · 3 Objectifs (lignes : nom, cible en MPa ou % f_ck, âge
  prévu facultatif ; « + Objectif » avec des modèles : hors gel 5 MPa, 50 %, 70 %, 100 % de f_ck) · 4 Hypothèses
  (résistance comparée : caractéristique / moyenne ; marge sur les prévisions ; loi de température : EC2 ou E_a saisi ;
  essai sur éprouvettes et calage).
- **Résultats** : verdict = le premier objectif (sa date, sa variante prudente ; son taux s'il a un âge prévu) ; bloc
  « Objectifs » (une ligne par objectif : cible, âge atteint, date ; prudent) ; bloc « Températures » (moyenne, extrêmes,
  jours mesurés / prévus / prolongés, gel) ; bloc « Essai » s'il y en a un ; avertissements.
- **Vues** : « Courbe » (f(t) en âge réel, dates sous l'axe, objectifs en lignes avec leur point, variante prudente en
  tirets, domaine t_T ≤ 3 j grisé, essai), « Températures » (barres par source, gel, t_T cumulé), « Tableau » (jour par
  jour : T, k_T, t_T, β_cc, f_cm, f_ck).
- **Défaut à l'ouverture** : C30/37, ciment N, 20 °C constante, objectif « Décoffrage : 70 % de f_ck ».

### 2.6 Les preuves

`tests/test_maturite_moteur.py` réécrit : β_cc, t_T, f_ck(t) à la main ; forme fermée = balayage numérique fin ; le cas
du défaut n° 1 (7 j à 2 °C puis 7 j à 20 °C : jour 8, plus 4,8 j) ; l'heure du coulage ; le gel ; la variante prudente
toujours plus tardive ; l'essai et son calage (la courbe calée passe par la mesure) ; refus ; 12 mutations. La lecture
de la météo (`meteo.py`, `releve.py`) garde ses contrôles contre le faux Open-Meteo. `tests/test_api_web_age.py` : l'écran
= le moteur.

---

## 3. Taux d'armature (`beton/taux`, `api_web/taux.py`, page `taux-armature`)

### 3.1 Pourquoi il existe

Chiffrer l'acier d'une poutre en béton armé à partir de son ferraillage : la masse à commander, le ratio en kg/m³ pour
un prix ou une comparaison, le bordereau par diamètre. C'est un QUANTITATIF (aucune vérification) : un chiffre faux par
défaut, c'est de l'acier qui manque sur chantier.

### 3.2 Ce qui était faux

| Défaut | Effet mesuré |
|---|---|
| nombre d'étriers tronqué sur des flottants, int(L/s) + 1 | un étrier de moins dans 5,1 % des zones courantes |
| « deux talons » calculée sans ses talons | volume de béton sous-estimé, taux surestimé |
| cadres de talon jamais comptés ; longueur d'étrier = contour de l'âme seule | acier manquant |
| retours ajoutés à 0 cm ; zones « À ≤ De » ignorées sans un mot ; zones qui se recouvrent comptées deux fois | — |
| barres filantes comptées sur L (sans l'enrobage d'about), aucun recouvrement au-delà de la longueur commerciale | — |
| note neuve sans aucune armature | 0,0 kg/m³ à l'ouverture |

### 3.3 Ce qui disparaît, ce qui arrive

Disparaissent : la « qualité d'acier » B500A/B/C (elle n'entrait dans aucun calcul), les codes couleur par dégradé, la
double saisie % / mètres héritée (on garde UNE saisie en mètres, avec « toute la longueur » par défaut). Arrivent : le
BORDEREAU (repère, Ø, nombre, longueur unitaire, longueur totale, masse) et sa synthèse par diamètre, téléchargeable en
CSV ; les épingles et les cadres de talon ; les recouvrements des barres plus longues que la longueur commerciale ; le
ratio en kg par mètre de poutre ; une poutre type à l'ouverture.

### 3.4 La méthode

- **Béton** : A_c = b·h + Σ talons ; V_c = A_c·L ; masse = 2 500 kg/m³ · V_c.
- **Masse linéique** : π·Ø²/4 · 7 850 kg/m³ (EN 10080 : Ø 8 = 0,395 ; Ø 12 = 0,888 ; Ø 16 = 1,578 ; Ø 20 = 2,466 kg/m).
- **Barres longitudinales** (lits filants, renforts, barres de peau) : n barres de x₁ à x₂ (m), bornés à [c ; L − c]
  (une barre filante mesure L − 2c) ; longueur unitaire = (x₂ − x₁) + retours·l_retour + recouvrements ;
  recouvrements (si l'option est active) : ⌈l/l_com⌉ − 1 par barre, de k_rec·Ø chacun (l_com = 12 m, k_rec = 50 par
  défaut). Barres de peau : n par face × 2 faces.
- **Un cadre** : 2·[(b − 2c) + (h − 2c)] + 2·max(10·Ø ; 70 mm) — le contour hors tout à l'enrobage et deux retours
  (EN 1992-1-1 §8.5), sans déduction des cintrages : côté sûr de quelques centimètres. **Une épingle** :
  (h − 2c) + 2·max(10·Ø ; 70 mm). **Un cadre de talon** (forme à talons) : 2·[(b_tot − 2c) + (h_t − 2c)] + retours,
  b_tot la largeur hors tout au niveau des talons.
- **Nombre de cours d'étriers** d'une zone de longueur l_z au pas s : ⌈l_z/s − 10⁻⁹⌉, plus UN cours de fermeture par
  suite de zones jointives (les zones qui se touchent ne comptent pas deux fois l'étrier commun). Zones qui se
  recouvrent : refusées. Longueur de poutre sans zone : avertissement « sans étriers ».
- **Quantités** : masse par repère, par famille (inférieures, supérieures, renforts, peau, étriers), par diamètre ;
  total ; majoration « chutes et ligatures » (5 % par défaut) ; **taux = masse majorée / V_c (kg/m³)** ; kg par mètre
  de poutre.

### 3.5 L'écran

- **Rubriques** : 1 Poutre (forme : rectangulaire, talon gauche, talon droit, deux talons ; b, h, L, enrobage ; talons)
  · 2 Armatures longitudinales (cartes : lit inférieur, lit supérieur, renfort, barres de peau — n, Ø, de, à, retours)
  · 3 Étriers (type : Ø, épingles, cadre de talon ; zones : de, à, pas) · 4 Hypothèses (majoration ; recouvrements :
  longueur commerciale, k_rec ; masses volumiques).
- **Résultats** : verdict de valeur « kg/m³ » (masse d'acier, volume de béton) ; bloc « Acier par famille » (masse,
  part) ; bloc « Par diamètre » (longueur, masse) ; bloc « Béton » ; kg/m ; avertissements ; « Bordereau CSV ».
- **Vues** : « Élévation » (barres à leur niveau et à leur longueur, zones d'étriers cotées avec leur nombre et leur
  pas), « Coupe » (section, cadre, épingles, barres), « Bordereau » (tableau des repères et synthèse par Ø).
- **Défaut à l'ouverture** : poutre 30 × 60, L = 6,00 m, c = 3 cm ; 3 Ø 20 inférieures filantes, 2 Ø 12 supérieures
  filantes, renfort 2 Ø 16 de 1,20 à 4,80 m ; cadres Ø 8 : pas de 15 cm sur 1,50 m à chaque about, 25 cm en travée.

### 3.6 Les preuves

`tests/test_taux_moteur.py` réécrit : la poutre type recalculée à la main ; le compte d'étriers sur une grille (jamais
un de moins que la pose réelle, zones jointives) ; talons ; recouvrements ; bordereau = somme des repères = total ;
refus ; mutations. `tests/test_api_web_taux.py` : l'écran = le moteur ; le CSV se relit.

---

## 4. Feu — la méthode tabulée dans la Bibliothèque Béton armé (`beton/bibliotheque/feu.py`, rubrique « Enrobages »)

### 4.1 Pourquoi il existe

Prescrire l'enrobage et les dimensions minimales d'un élément pour une durée de résistance au feu (R 30 à R 240), en
même temps que l'enrobage de durabilité. La réponse attendue : la distance à l'axe a, la dimension minimale, et
l'enrobage nominal qui en résulte — le plus grand du feu et de la durabilité.

### 4.2 Ce qui était faux

Ancien module « Enrobage » (copie de l'écran Streamlit) : poteaux lus dans la colonne μ_fi = 0,2 sous un libellé
« μ_fi ≈ 0,7 » (R 90 : 200/31 au lieu de 350/53 : **a trop faible de 15 à 25 mm**) ; voiles R 180 et R 240 à 45 et
55 mm au lieu de 50 et 60 ; le Ø d'étrier (8 mm par défaut) retiré de a pour une dalle ou un voile. La bibliothèque,
elle, ne donnait rien pour les poteaux et les voiles (« non transcrits »).

### 4.3 Ce qui est fait

- **Un moteur du feu**, `beton/bibliotheque/feu.py` : les tableaux de l'EN 1992-1-2:2004 section 5, transcrits et
  croisés sur deux sources (§ 4.5), avec leurs notes : poteaux (5.2a, méthode A : μ_fi = 0,2 / 0,5 / 0,7, exposé sur
  plusieurs côtés ou un seul), voiles porteurs (5.4 : μ_fi = 0,35 / 0,7, une ou deux faces) et non porteurs (5.3),
  poutres sur appuis simples (5.5) et continues (5.6), dalles (5.8 : une direction, deux directions, continues) et
  planchers-dalles (5.9).
- **La rubrique** « Enrobages » de la bibliothèque : l'élément, la durée R et ses paramètres (appuis, μ_fi,
  exposition, dimension réelle) ; le tableau entier de l'élément avec la case retenue ; la chaîne
  c_nom = max(durabilité ; a − Ø/2 − Ø_étrier) ; les conditions d'emploi du tableau dans « Pourquoi ? ». Pour une dalle
  ou un voile, aucun étrier n'est retiré de a (le champ devient « Ø de la nappe extérieure », 0 par défaut).
- **Les anciens modules masqués** « Enrobage » et « Données béton » ne portent plus les valeurs fausses : leurs tableaux
  de feu sont lus dans le moteur vérifié (une seule source). Leur retrait définitif (l'accueil doit d'abord tolérer les
  notes d'un module retiré) est un petit lot à part, à décider.
- Rien n'est interpolé (la norme le permet, §5.2 ; ne pas interpoler est du côté sûr) ; la dimension réelle choisit la
  combinaison (b_min, a) la plus favorable qu'elle autorise ; plus petite que le minimum du tableau : « durée non
  atteinte par la méthode tabulée ».

### 4.4 Non couvert, dit à l'écran

Méthode B des poteaux, formule (5.7), poteaux de l_0,fi > 3 m ou d'excentricité > 0,15·h ; sections en I (âme b_w) ;
précontrainte (a + 10 ou 15 mm) ; bétons à haute résistance ; méthodes de calcul simplifiées et avancées.

### 4.5 Tableaux retenus (mm) — EN 1992-1-2:2004

Poteaux, tableau 5.2a (b_min/a ; ** : au moins 8 barres) :

| | μ_fi = 0,2 | μ_fi = 0,5 | μ_fi = 0,7 | un seul côté, μ_fi = 0,7 |
|---|---|---|---|---|
| R 30 | 200/25 | 200/25 | 200/32 · 300/27 | 155/25 |
| R 60 | 200/25 | 200/36 · 300/31 | 250/46 · 350/40 | 155/25 |
| R 90 | 200/31 · 300/25 | 300/45 · 400/38 | 350/53 · 450/40** | 155/25 |
| R 120 | 250/40 · 350/35 | 350/45** · 450/40** | 350/57** · 450/51** | 175/35 |
| R 180 | 350/45** | 350/63** | 450/70** | 230/55 |
| R 240 | 350/61** | 450/75** | — | 295/70 |

Voiles porteurs, tableau 5.4 (épaisseur/a) :

| | μ_fi = 0,35 · une face | μ_fi = 0,35 · deux faces | μ_fi = 0,7 · une face | μ_fi = 0,7 · deux faces |
|---|---|---|---|---|
| REI 30 | 100/10 | 120/10 | 120/10 | 120/10 |
| REI 60 | 110/10 | 120/10 | 130/10 | 140/10 |
| REI 90 | 120/20 | 140/10 | 140/25 | 170/25 |
| REI 120 | 150/25 | 160/25 | 160/35 | 220/35 |
| REI 180 | 180/40 | 200/45 | 210/50 | 270/55 |
| REI 240 | 230/55 | 250/55 | 270/60 | 350/60 |

Voiles non porteurs, tableau 5.3 (épaisseur) : EI 30 : 60 · EI 60 : 80 · EI 90 : 100 · EI 120 : 120 · EI 180 : 150 ·
EI 240 : 175.

Poutres sur appuis simples, tableau 5.5 (b_min/a ; a_sd = a + 10 mm pour les barres d'angle d'un lit unique, sauf
au-delà de la largeur de la dernière colonne) :

| R 30 | 80/25 | 120/20 | 160/15 | 200/15 |
|---|---|---|---|---|
| R 60 | 120/40 | 160/35 | 200/30 | 300/25 |
| R 90 | 150/55 | 200/45 | 300/40 | 400/35 |
| R 120 | 200/65 | 240/60 | 300/55 | 500/50 |
| R 180 | 240/80 | 300/70 | 400/65 | 600/60 |
| R 240 | 280/90 | 350/80 | 500/75 | 700/70 |

Poutres continues, tableau 5.6 :

| R 30 | 80/15 | 160/12 | | |
|---|---|---|---|---|
| R 60 | 120/25 | 200/12 | | |
| R 90 | 150/35 | 250/25 | | |
| R 120 | 200/45 | 300/35 | 450/35 | 500/30 |
| R 180 | 240/60 | 400/50 | 550/50 | 600/40 |
| R 240 | 280/75 | 500/60 | 650/60 | 700/50 |

Dalles, tableau 5.8 (h_s ; a une direction ; a deux directions l_y/l_x ≤ 1,5 ; a deux directions 1,5 < l_y/l_x ≤ 2) ;
dalles continues : colonnes « une direction »… voir §5.7.3 (h_s et a de la colonne l_y/l_x ≤ 1,5) :

| REI 30 | 60 | 10 | 10 | 10 |
|---|---|---|---|---|
| REI 60 | 80 | 20 | 10 | 15 |
| REI 90 | 100 | 30 | 15 | 20 |
| REI 120 | 120 | 40 | 20 | 25 |
| REI 180 | 150 | 55 | 30 | 40 |
| REI 240 | 175 | 65 | 40 | 50 |

Planchers-dalles, tableau 5.9 (h_s/a) : REI 30 : 150/10 · REI 60 : 180/15 · REI 90 : 200/25 · REI 120 : 200/35 ·
REI 180 : 200/45 · REI 240 : 200/50.

### 4.6 Les preuves

`tests/test_bibliotheque_moteur.py` : chaque case de chaque tableau relue contre une transcription INDÉPENDANTE écrite
dans le test (deux copies nommées, liées par le test, R2), la monotonie (a croît avec R, décroît avec b), les cas D1 et
D2 d'hier (R 90 poteau μ 0,7 : 350/53 ; voile REI 180 : 50), la dalle et le voile sans étrier, les refus ; mutations.
`tests/test_api_web_bibliotheque.py` : la rubrique allume la bonne case.

---

## 5. Ce qui est commun aux quatre

- **Rien de silencieux** : une saisie hors domaine est refusée avec un message qui nomme le champ ; ce que le module ne
  couvre pas est écrit sous le résultat ; une valeur indicative est dite indicative.
- **Chaque nombre montré vient du moteur** (R3) ; chaque fenêtre de calcul écrit « symbole = expression = valeurs =
  résultat », recalculable à partir de ses seuls termes (un test le prouve).
- **Les textes** : le français du bureau, des libellés courts, les aides dans ⓘ.
- **Contrôles** : contrat commun (`tests/test_api_web_modules.py`), gel (`tests/reference_refonte/cas.py` : cas réécrits,
  regel accepté avec la raison écrite dans la photo), harnais aux trois formats (`verifier_site.js` : module, formats),
  captures REGARDÉES, suite complète avant la mise en ligne.
- **Hors du lot** : les notes PDF de ces modules (boutons grisés « bientôt » : à concevoir sur le modèle des cornières
  une fois les écrans validés par l'ingénieur) ; le retrait des modules masqués ; les écrans Streamlit.
