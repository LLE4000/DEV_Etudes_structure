# Lot Q — quatre modules refaits : poutres bois, maturité du béton, taux d'armature, feu (10/11/2026)

- **Statut** : conception arrêtée le 10/11/2026 au soir, AVANT le code ; la réalisation et ses écarts s'écrivent à la
  fin de ce fichier (« Le lot réalisé »).
- **Demande de l'ingénieur (10/11/2026)** : « Fais bois, feu, taux, maturité. Vérifie les formules, car tout n'était pas
  juste. Pas grave si on n'a pas exactement la même chose que sur Streamlit : tu as le principe de ce que je veux, tu
  peux tout réorganiser, supprimer ce qui n'est pas utile. Comprends pourquoi les modules existent et fais l'interface
  correcte sur le site. Sors-moi des modules professionnels et justes, ergonomiques. »
- **Précisions de l'ingénieur (10/11/2026, plus tard dans la soirée)** — elles sont intégrées ci-dessous :
  1. « Feu » n'est pas un module : c'est la résistance au feu de l'ancien module Enrobage. « Le module Enrobage, on
     pourrait le mettre dans la bibliothèque : on a le tableau, on pourrait avoir un deuxième onglet avec le calcul…
     comme ça on range l'interface. » → § 4 : la rubrique « Enrobages » de la bibliothèque a DEUX onglets, « Tableaux »
     et « Calcul » ; l'ancien module n'est plus le lieu du calcul.
  2. « Dans Streamlit, dans les poutres, si on mettait des longueurs par section, on pouvait activer dans les options
     le taux d'armature. Ce serait bien de le réactiver… Dans Poutres béton armé on dimensionne, on pourrait envoyer
     directement dans Taux d'armature, tout serait déjà préprogrammé ; après on change l'espacement des étriers, à
     quel endroit, les retours verticaux. » → § 3 (le modèle du taux d'armature est fait pour recevoir une poutre
     dimensionnée) et § 6 (l'option dans les poutres et le bouton d'envoi : seconde vague du lot).
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
l'alerte de gel avant 5 MPa ; le DURCISSEMENT décrit par le bon de livraison (rapide, moyen, lent) et pas seulement par
la classe du ciment ; la loi de température du complément BELGE (NBN B 15-400) ; l'essai sur éprouvettes comparé à la
prévision, avec calage de la courbe.

### 2.4 La méthode — EN 1992-1-1:2004 §3.1.2 ; NBN B 15-400 (complément belge à l'EN 13670) ; EN 13670:2009 §8.5

    f_cm(t) = β_cc(t_T)·f_cm                      f_cm = f_ck + 8                              §3.1.2(6), (3.1)
    β_cc(t_T) = exp{ s·[1 − √(28/t_T)] }                                                       (3.2)
    f_ck(t) = f_cm(t) − 8  pour 3 < t_T < 28 j ;  f_ck pour t_T ≥ 28 j                         §3.1.2(5)
    t_T = Σ Δt_i · k(T_i)                         l'âge équivalent à 20 °C

Les sources ont été croisées le 10/11/2026 (rapport `SOURCES_MATURITE` du lot) ; ce que le croisement a changé est
marqué **[sources]**.

**Le durcissement — le coefficient s.** Deux lectures, au choix dans la rubrique « Béton » :

| Lecture | Choix | s | r = f_cm,2j / f_cm,28j qu'elle suppose |
|---|---|---|---|
| le CIMENT (EN 1992-1-1 §3.1.2(6)) | R : CEM 42,5 R, 52,5 N, 52,5 R | 0,20 | 0,58 |
| | N : CEM 32,5 R, 42,5 N | 0,25 | 0,50 |
| | S : CEM 32,5 N | 0,38 | 0,35 |
| le BÉTON (développement de résistance de la NBN EN 206, écrit sur le bon de livraison) | rapide : r ≥ 0,50 | 0,253 | 0,50 (la borne basse de la classe) |
| | moyen : 0,30 ≤ r < 0,50 | 0,439 | 0,30 |
| | lent : 0,15 ≤ r < 0,30 | 0,692 | 0,15 |
| | r connu (centrale, essais) | −ln r / (√14 − 1) | r |

s se déduit de r par (3.2) écrite à 2 jours : r = β_cc(2) = exp{s·(1 − √14)}. **[sources]** La classe S de l'EC2 n'est
PAS « lente » : elle suppose r = 0,35, un béton « moyen » au sens de l'EN 206 ; et la classe N suppose r = 0,50, la
frontière du « rapide ». Or le ciment le plus courant en Belgique, le CEM III/A 42,5 N LA, est N par la lettre de l'EC2
(et par la règle FEBELCEM : laitier > 35 %) mais « moyen » sur les bons de livraison : la lecture par le ciment est
alors optimiste (à 3 jours, β_cc = 0,60 en N contre 0,41 en « moyen » pris à sa borne). L'écran montre toujours le r que
le choix suppose, et conseille la lecture du bon de livraison pour les ciments au laitier (CEM III/B 42,5 N et
CEM III/A 32,5 N : « lent »). **Le défaut d'ouverture est « béton à durcissement moyen »** (s = 0,439) : c'est le
béton le plus courant ici, et une valeur par défaut doit être du côté sûr.

**La température — le coefficient k(T)**, au choix dans les hypothèses :

| T (°C) | −10 | −5 | 0 | 5 | 10 | 15 | 20 | 30 |
|---|---|---|---|---|---|---|---|---|
| NBN B 15-400 (par défaut) | 0 | 0,15 | 0,30 | 0,45 | 0,60 | 0,80 | 1,00 | 1,00 |
| EC2 annexe B, (B.10) | 0 | 0 | 0,37 | 0,48 | 0,62 | 0,79 | 1,00 | 1,57 |
| E_a = 45 kJ/mol (exemple) | 0 | 0 | 0,26 | 0,37 | 0,52 | 0,73 | 1,00 | 1,84 |

- **NBN B 15-400** **[sources]** : l'« âge fictif » du complément belge à l'EN 13670 — k = 1 à 20 °C et au-dessus,
  0,8 à 15 °C, 0,6 à 10 °C, 0,45 à 5 °C, 0,3 à 0 °C, 0,15 à −5 °C, interpolé linéairement (l'exemple de la norme, 5, 7
  et 5 °C, donne 1,41 j : reproduit par un test). Le moteur le prolonge en ligne droite jusqu'à 0 à −10 °C, et ne
  compte aucun bonus au-dessus de 20 °C. C'est la loi PAR DÉFAUT : belge, et la plus prudente par temps froid. Valeurs
  relevées sur deux sources concordantes (projet d'annexe devenu NBN B 15-400), à contrôler sur l'édition 2024.
- **EC2 (B.10)** : k = exp{−4 000·[1/(273 + T) − 1/293]} (E_a/R = 4 000 K), domaine 0 à 80 °C ; écrite pour la cure
  thermique (§10.3.1.1(3)), son emploi pour un béton coulé en place est une extension (le fib MC2010 fait de même).
  **[sources]** C'est la loi la PLUS OPTIMISTE par temps froid (0,48 à 5 °C contre 0,45 pour la norme belge, 0,29 à
  0,40 pour les méthodes de maturité étalonnées). Hors de son domaine (sous 0 °C) : k = 0. (B.10) écrit « 13,65 » pour
  4000/293 = 13,652 : le moteur prend la référence exacte, k = 1 à 20 °C.
- **E_a saisi** (20 à 60 kJ/mol, Arrhenius) : pour un béton dont la centrale donne l'énergie d'activation ; les ciments
  au laitier montent à 45–60 kJ/mol.
- La température est celle du BÉTON (toutes les méthodes sont définies ainsi) ; à défaut de mesure, celle de l'air —
  une hypothèse, dite à l'écran (un élément mince exposé peut être plus froid que l'air la nuit).

**Le temps** : t compté depuis la FIN DU BÉTONNAGE (date et heure ; midi si l'heure n'est pas donnée) ; la température
est constante par jour civil ; t_T(t) est donc linéaire par morceaux, intégré exactement (le premier jour ne compte que
pour sa fraction restante). Au-delà du dernier jour connu, le relevé est PROLONGÉ à la moyenne de ses cinq derniers
jours (comme la météo), et c'est dit.

**Un objectif** : une résistance cible et la résistance à laquelle on la compare.
- Cible en **% de f_ck** (une exigence de calcul : 70 % de f_ck) → comparée à la résistance CARACTÉRISTIQUE f_ck(t).
- Cible en **MPa** (un seuil de chantier : 5 MPa) → comparée, au choix, à la résistance MOYENNE f_cm(t) — ce que
  mesurent des cubes de chantier, la base des seuils d'usage — ou à la caractéristique (côté sûr, par défaut pour une
  cible saisie à la main). **[sources]** Les cahiers des charges comparent bien la MOYENNE des cubes de chantier au
  seuil (Qualiroutes K : ces essais « n'ont pas pour objet de déterminer f_ck »).
- L'âge où il est atteint a une forme fermée : t_T,req = 28/(1 − ln β_req/s)², β_req = (cible + 8)/f_cm
  (caractéristique) ou cible/f_cm (moyenne) ; puis t se lit en inversant t_T(t) : plus de dichotomie. Une cible
  au-dessus de la classe est « non atteignable ». La date et l'heure sont données à l'heure près (« sam. 15/11 vers
  14 h »), jamais tronquées.
- **Modèles d'objectif**, chacun avec sa source au survol : « Hors gel · 5 MPa » (moyenne ; EN 13670 §8.5(12)) ;
  « Décoffrage vertical · 5 MPa » (moyenne ; annexe informative de l'EN 13670, citée par le CSTC et Infociments) ;
  « Reprise de bétonnage · 10 MPa » (moyenne ; Infociments, bétonnage par temps froid) ; « 50 % », « 70 % », « 100 % de
  f_ck » (caractéristique ; à fixer par le bureau d'études selon la charge au décoffrage).

**Variante prudente** : les jours PRÉVUS et PROLONGÉS (tous, en température constante) abaissés de la marge (3 °C par
défaut) ; chaque objectif donne ses deux dates. **Opération datée** : si l'objectif porte un âge prévu,
η = cible/f(t_prévu) ; conforme si η ≤ 1 ; « conforme, attention » si la variante prudente ne passe pas.

**Domaine de l'EC2** : f_ck(t) = f_cm(t) − 8 vaut pour 3 < t < 28 j ; avant 3 jours équivalents (le domaine de la
plupart des décoffrages), la valeur est marquée « indicative — à confirmer par essais ». f_ck(t) n'est jamais négatif.

**Gel** (EN 13670 §8.5(12)) : la surface du béton ne doit pas descendre sous 0 °C avant d'avoir 5 MPa. Le moteur
cherche le premier jour de gel (minimale horaire < 0 °C si le détail horaire est chargé, sinon moyenne ≤ 0 °C — et il
dit laquelle) et le compare à l'âge où f_cm(t) atteint 5 MPa : alerte « protéger du gel » si le gel vient avant. Et un
rappel (Buildwise, NIT 285 §9.4) quand une température sous 5 °C tombe dans les 72 premières heures : « mesures de
bétonnage par temps froid ».

**Essai** : la moyenne des éprouvettes de chantier à un âge donné (cube de 150 mm ramené au cylindre par
f_ck/f_ck,cube de la classe : 0,78 à 0,83), comparée à f_cm(t) prévu : écart en %. « Caler la courbe sur l'essai »
recalcule s pour que la courbe passe par la mesure (s_cal = ln(f_mes/f_cm)/(1 − √(28/t_T)), borné à [0,10 ; 0,80]) :
toutes les dates suivent, et le r calé s'affiche.

**Sélecteur de classe** : chaque classe (C16/20 à C50/60) avec le même durcissement et les mêmes températures : l'âge
où le premier objectif est atteint ; ✓ la classe du coulage, ★ la plus petite qui tient l'âge prévu.

**Ce que l'outil N'EST PAS**, écrit sous le résultat : une ESTIMATION (loi de l'EC2, température de l'air à défaut de
celle du béton) ; pour une opération critique (désétaiement, mise en charge, précontrainte), la résistance se confirme
par des éprouvettes conservées sur chantier ou une maturométrie étalonnée, et le délai se valide avec le bureau
d'études. **Non couvert**, dit à l'écran : les délais minimaux de décoffrage par type de coffrage de la NBN B 15-400
et les durées de cure (tableau F.8-ANB) — leurs valeurs n'ont pas pu être relues sur le texte ; les critères en
proportion des charges ; les flèches au jeune âge ; l'EN 1992-1-1:2023 (annexe B.4, classes CS/CN/CR : pas encore
d'annexe belge).

### 2.5 L'écran

- **Rubriques** : 1 Béton (classe ; « Durcissement » : un menu en deux groupes, le ciment R / N / S ou le béton rapide
  / moyen / lent / r connu, avec dessous le r supposé ; date et heure de fin de bétonnage) · 2 Températures (Constante
  / Relevé / Météo du lieu : inchangé, Open-Meteo à la demande ; le relevé peut être celui du béton) · 3 Objectifs
  (une carte par objectif : nom, cible en MPa ou % f_ck, comparée à la moyenne ou à la caractéristique, âge prévu
  facultatif ; « + Objectif » et les modèles) · 4 Hypothèses (loi de température : NBN B 15-400 / EC2 (B.10) / E_a
  saisi ; marge sur les prévisions ; essai sur éprouvettes et calage).
- **Résultats** : verdict = le premier objectif (sa date, sa variante prudente ; son taux s'il a un âge prévu) ; bloc
  « Objectifs » (une ligne par objectif : cible, âge atteint, date ; prudent) ; bloc « Températures » (moyenne, extrêmes,
  jours mesurés / prévus / prolongés, gel, âge équivalent) ; bloc « Essai » s'il y en a un ; avertissements.
- **Vues** : « Courbe » (f_cm(t) et f_ck(t) en âge réel, dates sous l'axe, objectifs en lignes avec leur point, variante
  prudente en tirets, domaine t_T ≤ 3 j grisé, essai), « Températures » (barres par source, gel, t_T cumulé),
  « Tableau » (jour par jour : T, k, t_T, β_cc, f_cm, f_ck).
- **Défaut à l'ouverture** : C30/37, béton à durcissement MOYEN, 20 °C constante, objectif « Décoffrage : 70 % de
  f_ck » (J + 10,7 ; J + 6,5 si l'on choisit le ciment N). Le défaut est le béton courant en Belgique (CEM III/A
  42,5 N LA : « moyen » sur le bon de livraison) pris à la borne basse de sa classe : réaliste et du côté sûr, là où
  le ciment N promettrait un décoffrage quatre jours plus tôt.

### 2.6 Les preuves

`tests/test_maturite_moteur.py` réécrit : β_cc, s(r), k(T) des trois lois (le tableau ci-dessus, l'exemple belge
5-7-5 °C = 1,41 j), t_T, f_ck(t) à la main ; forme fermée = balayage numérique fin ; le cas du défaut n° 1 (7 j à 2 °C
puis 7 j à 20 °C, EC2 (B.10) : jour 8, plus 4,8 j) ; l'heure du coulage ; le gel ; la variante prudente toujours plus
tardive ; la loi belge jamais plus rapide que (B.10) sous 20 °C ; l'essai et son calage (la courbe calée passe par la
mesure) ; refus ; 12 mutations. La lecture de la météo (`meteo.py`, `releve.py`) garde ses contrôles contre le faux
Open-Meteo. `tests/test_api_web_age.py` : l'écran = le moteur.

**Réalisé les 10 et 11/11/2026** (`docs/beton/MATURITE_MOTEUR.md`) : 208 contrôles et 46 mutations rejetées côté moteur, 78
contrôles et 7 mutations côté écran. **Un invariant annoncé ci-dessus s'est révélé FAUX et a été mesuré** : « la loi
belge jamais plus rapide que (B.10) sous 20 °C » ne tient que de 0 à 12,2 °C ; de 12,3 à 19,9 °C le tableau belge
dépasse (B.10), de 1,5 % au plus (0,80 contre 0,789 à 15 °C). Le test borne l'écart au lieu d'affirmer l'invariant ; la
loi belge reste le défaut (c'est la norme d'ici, et l'écart est sans effet devant la marge de la variante prudente).
« La plus prudente par temps froid » s'entend FACE À (B.10), de 0 à 12 °C : sous 0 °C la loi belge compte encore un peu
de durcissement là où (B.10) compte 0, et une énergie d'activation de plus de 36 kJ/mol (ciments au laitier : 45 à
60) donne moins que le tableau belge à 5 °C (0,37 contre 0,45 pour 45 kJ/mol). L'aide de l'écran le dit.
Deux choix du moteur qui ne sont pas dans la norme sont écrits à l'écran : la droite belge prolongée jusqu'à k = 0 à
−10 °C, et aucun bonus au-dessus de 20 °C ; aucun gain n'est compté au-delà de 28 jours équivalents.

---

## 3. Taux d'armature (`beton/taux`, `api_web/taux.py`, page `taux-armature`)

### 3.1 Pourquoi il existe

Chiffrer l'acier d'une poutre en béton armé à partir de son ferraillage : la masse à commander, le ratio en kg/m³ pour
un prix ou une comparaison, le bordereau par diamètre. C'est un QUANTITATIF (aucune vérification) : un chiffre faux par
défaut, c'est de l'acier qui manque sur chantier.

Deux niveaux, voulus par l'ingénieur : (1) dans « Poutres béton armé », un taux SIMPLIFIÉ au mètre courant, par section
et pour la poutre quand les sections ont une longueur (§ 6) ; (2) ce module-ci, DÉTAILLÉ, qui reçoit la poutre
dimensionnée (ses sections deviennent des tronçons : lits, cadres, pas) et où l'on règle ensuite où s'arrêtent les
barres, les zones d'étriers, les retours.

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
- **Les étriers se décrivent par TYPES et par ZONES** (comme les sections d'une poutre : un type = ce qu'on pose à
  chaque cours ; une zone = où, et à quel pas). Un type est une liste de lignes : « cadre » (le contour de l'âme),
  « cadre intérieur » (de largeur donnée), « épingle » (verticale), chacune avec son Ø ; plus, pour une forme à
  talons, la case « cadre de talon ».
- **Un cadre** : 2·[(b − 2c) + (h − 2c)] + 2·l_r — le contour hors tout à l'enrobage et deux retours l_r =
  max(10·Ø ; 70 mm) (EN 1992-1-1 §8.5), sans déduction des cintrages : côté sûr de quelques centimètres ; **un cadre
  intérieur** de largeur w : 2·[w + (h − 2c)] + 2·l_r ; **une épingle** : (h − 2c) + 2·l_r ; **un cadre de talon** :
  2·[(b_tot − 2c) + (h_t − 2c)] + 2·l_r, b_tot la largeur hors tout au niveau des talons. (Le taux simplifié des
  poutres, § 6, garde le retour de 10 cm de l'écran Streamlit : c'est un réglage de ses options.)
- **Recevoir une poutre dimensionnée** (`beton/taux/depuis_poutre.py`, fonction pure, testée) : b, h, enrobage ; chaque
  section de longueur l_i devient un tronçon [x_i ; x_i + l_i] ; pour chaque lit (face, rang) : les barres COMMUNES à
  toutes les sections (même Ø, le plus petit nombre) deviennent des barres filantes, le surplus de chaque section un
  renfort sur son tronçon (les tronçons voisins de même surplus sont réunis) ; un Ø différent d'une section à l'autre
  donne des barres par tronçon ; les barres latérales de même ; chaque composition d'étriers distincte devient un
  type (cadre de toute la largeur → « cadre » ; cadre entre deux barres → « cadre intérieur » de la largeur que la
  poutre calcule ; épingle → « épingle »), chaque section une zone à son pas. Sans longueur de section : refus, avec
  le message qui dit où les saisir.
- **Nombre de cours d'étriers** d'une zone de longueur l_z au pas s : ⌈l_z/s − 10⁻⁹⌉, plus UN cours de fermeture par
  suite de zones jointives (les zones qui se touchent ne comptent pas deux fois l'étrier commun). Zones qui se
  recouvrent : refusées. Longueur de poutre sans zone : avertissement « sans étriers ».
- **Quantités** : masse par repère, par famille (inférieures, supérieures, renforts, peau, étriers), par diamètre ;
  total ; majoration « chutes et ligatures » (5 % par défaut) ; **taux = masse majorée / V_c (kg/m³)** ; kg par mètre
  de poutre.

### 3.5 L'écran

- **Rubriques** : 1 Poutre (forme : rectangulaire, talon gauche, talon droit, deux talons ; b, h, L, enrobage ; talons)
  · 2 Armatures longitudinales (cartes : lit inférieur, lit supérieur, renfort, barres de peau — n, Ø, de, à, retours)
  · 3 Étriers (types : nom, lignes cadre / cadre intérieur / épingle avec leur Ø, cadre de talon ; zones : de, à, pas,
  type) · 4 Hypothèses (majoration ; recouvrements : longueur commerciale, k_rec ; masses volumiques).
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

## 4. Enrobage et feu — dans la Bibliothèque Béton armé, deux onglets : « Tableaux » et « Calcul » (`beton/bibliotheque`, rubrique « Enrobages »)

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
  croisés sur des sources indépendantes (§ 4.5, l'état de chaque tableau y est écrit), avec leurs notes : poteaux
  (5.2a, méthode A : μ_fi = 0,2 / 0,5 / 0,7, exposé sur plusieurs côtés ou un seul ; et la FORMULE (5.7) de la même
  méthode, voir ci-dessous), voiles porteurs (5.4 : μ_fi = 0,35 / 0,7, une ou deux faces) et non porteurs (5.3),
  poutres sur appuis simples (5.5) et continues (5.6), dalles (5.8 : une direction, deux directions, continues) et
  planchers-dalles (5.9).
- **Les poteaux ont deux lectures** (croisement des sources du 10/11/2026). L'annexe belge (NBN EN 1992-1-2 ANB)
  REMPLACE le tableau 5.2a par deux tableaux (5.2.a.1-ANB pour l_0,fi ≤ 2 m, 5.2.a.2-ANB pour l_0,fi ≤ 3 m ; largeurs
  de 200 à 450 mm ; 4 barres ou 8), que la formule (5.7) de l'EN reproduit à ± 0,5 mm quand on y met α_cc = 0,85 (la
  valeur belge de l'EN 1992-1-1) : 99 cases sur 102 de l'extrait relevé dans la fiche FEBE DT02. Le texte de l'ANB n'a
  pas pu être LU (sites fermés à la session) ; l'extrait montre aussi des cases « – » que la formule seule n'explique
  pas (colonne de 200 mm à R 90 pour μ_fi = 0,7 et à R 120 dès μ_fi = 0,5). Donc :
  - **par défaut, le tableau 5.2a de l'EN** (α_cc = 1,0 ; l_0,fi ≤ 3 m) : vérifié case par case sur plusieurs sources,
    et le plus SÉVÈRE des deux (R 90, μ_fi = 0,7, colonne de 350 : a = 53 mm contre 47) — la règle du dépôt : le
    durable garde le défavorable ;
  - **au choix, « Formule (5.7) »** : pour une longueur de flambement au feu jusqu'à 6 m, un μ_fi quelconque, la
    largeur réelle — a exigé pour la durée demandée ET la durée R obtenue avec l'enrobage retenu. α_cc = 0,85 (ANB),
    donc le terme en ω disparaît. Son domaine géométrique (a de 25 à 80 mm, b′ de 200 à 450 mm, h ≤ 1,5·b, l_0,fi de
    2 à 6 m) est contrôlé — hors domaine : refus, jamais d'extrapolation ; A_s < 0,04·A_c et e ≤ e_max sont des
    conditions d'emploi écrites sous le résultat. L'écran dit : « ce que
    donnent les tableaux 5.2.a-ANB de l'annexe belge — à contrôler sur le texte de l'ANB:2021 ; le tableau belge exclut
    certaines colonnes de 200 mm ».
- **La rubrique « Enrobages » a deux onglets** (demande de l'ingénieur, 10/11/2026) :
  - **« Calcul »** — le calcul complet de l'enrobage d'un élément, ce que faisait l'ancien module, en plus poussé :
    l'élément (poutre, poteau, dalle, voile) et ses dimensions ; l'environnement (exposition avec son libellé belge,
    béton, durée, qualité, classe S calculée ou imposée, coulé contre le sol) ; les armatures (Ø de la barre, Ø de
    l'étrier ou de la nappe extérieure, granulat, Δc_dev) ; le feu (durée R et les paramètres de l'élément : appuis,
    portée, μ_fi, faces exposées). Le résultat : c_nom en grand, ce qui le gouverne, la chaîne terme par terme avec sa
    règle (c_min,b, classe S, c_min,dur, c_min, Δc_dev, c_nom,dur ; a exigé, a_sd, c_nom,feu ; dimension minimale et
    verdict « atteinte / non atteinte »), la distance à l'axe OBTENUE avec l'enrobage retenu, et le DESSIN du détail à
    l'échelle (barre, étrier, c_nom, a, a_sd). « Inclure dans la note » quand la bibliothèque est ouverte depuis un calcul.
  - **« Tableaux »** — les tableaux de référence, la case de la situation allumée : durabilité (4.4N) et feu (le
    tableau entier de l'élément et de sa variante) ; un clic sur une case règle les paramètres.
  Les deux onglets partagent les mêmes paramètres (l'adresse les garde). Pour une dalle ou un voile, aucun étrier n'est
  retiré de a (le champ devient « Ø de la nappe extérieure », 0 par défaut).
- **L'ancien module « Enrobage »** (masqué de l'accueil depuis le 09/11) n'est plus le lieu du calcul : sa page reste
  servie pour les notes existantes, ne porte plus de valeur fausse (ses tableaux de feu sont lus dans le moteur vérifié,
  une seule source), et un bandeau y dit « Le calcul d'enrobage est maintenant dans la Bibliothèque › Enrobages ›
  Calcul » avec le lien. De même la rubrique « Enrobages » de l'ancien « Données béton ». Leur retrait définitif
  (l'accueil doit d'abord tolérer les notes d'un module retiré) est un petit lot à part.
- Rien n'est interpolé (la norme le permet, §5.2 ; ne pas interpoler est du côté sûr) ; la dimension réelle choisit la
  combinaison (b_min, a) la plus favorable qu'elle autorise ; plus petite que le minimum du tableau : « durée non
  atteinte par la méthode tabulée ».

### 4.4 Non couvert, dit à l'écran

Méthode B des poteaux (tableau 5.2b) et annexe C ; poteaux hors du domaine de la méthode A (l_0,fi > 3 m pour le
tableau, > 6 m pour la formule ; excentricité au-delà de e_max = 0,15·h — la valeur belge n'a pas pu être relue) ;
sections en I (largeur d'âme b_w : la Belgique retient la classe WC) ; précontrainte (a + 10 mm pour les barres,
+ 15 mm pour les fils et torons) ; plusieurs lits (distance moyenne a_m, éq. 5.5 : à la charge de l'ingénieur) ; bétons
à haute résistance ; granulats calcaires (réduction de 10 % non appliquée : côté sûr) ; méthodes de calcul simplifiées
et avancées. Rappel écrit quand a ≥ 70 mm : armature de peau à prévoir (EN 1992-1-2 §4.5.2).

### 4.5 Tableaux retenus (mm) — EN 1992-1-2:2004

État du croisement des sources (rapports `SOURCES_FEU` du lot, 10/11/2026) : **5.2a** confirmé en entier (plusieurs
sources) ; **5.3** confirmé (une source) ; **5.4** confirmé pour μ_fi = 0,35 et, pour μ_fi = 0,7, de REI 120 à REI 240 ;
**5.5** : trois points confirmés, le reste identique aux valeurs déjà en ligne dans la bibliothèque ; **5.6, 5.8 (deux
directions), 5.9** : transcrits, seconde vérification en cours au moment d'écrire — l'état final de chaque tableau est
porté par le moteur (`feu.py`, champ `verification` de chaque table) et montré sous le tableau à l'écran. Une valeur non
recoupée est DITE non recoupée.


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

Poutres sur appuis simples, tableau 5.5 (b_min/a). a_sd = a + 10 mm pour les barres d'angle d'une poutre à un seul
lit, pour les largeurs jusqu'à celle de la COLONNE 4 du tableau 5.5 — la troisième combinaison, les colonnes étant
numérotées avec celle des durées — et de la COLONNE 3 du tableau 5.6 (la deuxième combinaison) ; au-delà, aucune
majoration (§5.6.1(8) ; corrigé le 10/11/2026 : la première rédaction disait « la dernière colonne ») :

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

Poteaux, formule (5.7) (EN 1992-1-2 §5.3.2(4), méthode A) :

    R = 120·[(R_ηfi + R_a + R_l + R_b + R_n)/120]^1,8                                        (5.7)
    R_ηfi = 83·[1,00 − μ_fi·(1 + ω)/((0,85/α_cc) + ω)]      = 83·(1 − μ_fi) avec α_cc = 0,85 (ANB)
    R_a = 1,60·(a − 30)        R_l = 9,60·(5 − l_0,fi)        R_b = 0,09·b′
    R_n = 0 (4 barres, aux angles) ; 12 (plus de 4 barres)      b′ = 2·A_c/(b + h) = 2·b·h/(b + h) ; le diamètre d'un poteau rond
    a exigé pour une durée R :  a = 30 + [120·(R/120)^(1/1,8) − R_ηfi − R_l − R_b − R_n]/1,60, au moins 25 mm

Domaine : 25 ≤ a ≤ 80 mm ; 200 ≤ b′ ≤ 450 mm ; h ≤ 1,5·b ; 2 ≤ l_0,fi ≤ 6 m (sous 2 m : la valeur à 2 m, côté sûr) ;
0,2 ≤ μ_fi ≤ 0,7 (sous 0,2 : la valeur à 0,2). a est arrondi au millimètre SUPÉRIEUR (le tableau belge arrondit au plus
proche : 1 mm d'écart au plus, côté sûr). Contrôles à la main (l_0,fi = 3 m, 4 barres) : R 90, 350 mm, μ_fi = 0,7 →
a = 46,7 → 47 mm (extrait ANB : 47) ; R 90, 300 mm, μ_fi = 0,5 → 39,1 → 40 mm (ANB : 39) ; 8 barres : 31,6 → 32 mm
(ANB : 32) ; R 60, 200 mm, μ_fi = 0,7 → 42,2 → 43 mm (ANB : 42).

### 4.6 Les preuves

`tests/test_bibliotheque_moteur.py` : chaque case de chaque tableau relue contre une transcription INDÉPENDANTE écrite
dans le test (deux copies nommées, liées par le test, R2), la monotonie (a croît avec R, décroît avec b), les cas D1 et
D2 d'hier (R 90 poteau μ 0,7 : 350/53 ; voile REI 180 : 50), la dalle et le voile sans étrier, les refus ; la formule
(5.7) recalculée à la main (les quatre contrôles ci-dessus, l'aller-retour a → R → a, R croissant avec a, b′ et le
nombre de barres, décroissant avec μ_fi et l_0,fi, le domaine refusé de chaque côté), et « la formule n'est jamais
choisie par défaut » ; mutations. `tests/test_api_web_bibliotheque.py` : la rubrique allume la bonne case.

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

---

## 6. Poutres béton armé : le taux d'armature simplifié, et l'envoi vers le module détaillé (seconde vague)

Demande de l'ingénieur (10/11/2026). Elle se fait APRÈS l'intégration des quatre lots (le module « Taux d'armature »
doit exister dans sa nouvelle forme).

### 6.1 L'option « Taux d'armature » des poutres (ce que fait l'écran Streamlit `modules/poutre.py`, v2.39+)

- **Rubrique « Options »** : case « Taux d'armature » (éteinte par défaut), majoration (5 %), retour d'étrier (10 cm,
  compté deux fois par étrier ou épingle), arrondi supérieur (5 kg/m³). Allumée, chaque SECTION montre un champ
  « Longueur (m) » (rubrique « Sections et sollicitations »), gardé quand l'option s'éteint.
- **Le calcul, au mètre courant de poutre, par section** (extrait de `_taux_armature_section`, parité prouvée contre
  l'écran rejoué) : chaque barre des lits compte 1 m ; étriers et épingles : 100/pas par mètre, longueur d'un étrier
  2·(w + h − 2c) + 2·retour (w = b − 2c pour un cadre de toute la largeur, sinon l'entraxe des barres entourées + Ø
  barre + 2 Ø étrier), d'une épingle (h − 2c ou l'entraxe) + 2·retour ; armatures de peau : 2·(⌈d_vert/s_max⌉ − 1)
  barres quand d_vert > s_max ; masse × (1 + majoration) / (b·h·1 m) ; arrondi au multiple supérieur.
- **Le taux de la poutre** : Σ(masse_i·l_i)/Σ(volume_i·l_i) sur les sections qui ont une longueur.
- **À l'écran** : le taux de la section dans l'en-tête de son bloc de résultats, sa fenêtre de calcul (le tableau
  Position · Nb · Ø · Section · kg/m · L/barre · L totale · Poids · %, puis la synthèse mètre courant / poutre) ; le taux
  de la poutre dans la tête des résultats. Les clés nouvelles de l'état sont ABSENTES tant que l'option est éteinte :
  une note existante reste identique à l'octet (le gel ne bouge pas).

**Réalisé le 10/11/2026** (`beton/poutre/taux.py`, `api_web/poutre.py`, `site/web/css/modules/poutre.css`) — ce qui
s'écarte du texte ci-dessus, et pourquoi :

- l'option est celle de CHAQUE poutre (clé `taux` de la poutre), comme ses autres options — l'écran Streamlit la
  réglait pour toute la séance ; une note où elle n'a jamais servi n'a pas la clé ;
- les armatures latérales comptées sont celles de la section (automatiques par la règle des armatures de peau, ou
  imposées), et une épingle est verticale : le calcul suit le ferraillage que le site dessine ;
- le taux de la section est une LIGNE de son bloc de résultats (« Taux d'armature · 90 kg/m³ », avec la masse au mètre),
  qui ouvre sa fenêtre de calcul ; le taux de la poutre est sous son nom, en tête des résultats, et ouvre la sienne ;
- le taux de la poutre ne va dans la NOTE PDF (le bloc « TAUX D'ARMATURE » du générateur) que si une SECONDE case le
  demande, « Écrire le taux dans la note de calcul », éteinte par défaut — précision de l'ingénieur du 10/11/2026 au
  soir : « des fois, on n'a pas envie de donner le taux d'armature aux gens ». Ce sont les deux cases de l'écran
  Streamlit (« Afficher », « Envoyer dans la note de calcul ») ; éteindre l'option retire aussi la seconde ;
- le gel des poutres a été refait (un champ de plus : la case) et porte deux cas de l'option.

Preuves : `tests/test_poutre_taux.py`, `tests/test_poutre_moteur.py` § 9 (parité : 99 sections, 60 poutres).

### 6.2 « Détailler dans Taux d'armature »

Un bouton de la poutre (à côté de l'option, et dans le menu « ⋯ » au téléphone) : il demande au moteur la conversion
de la poutre ouverte (`beton/taux/depuis_poutre.py`, § 3.4), crée une note du module « Taux d'armature » — même chantier,
même dossier, nommée d'après la poutre — et l'ouvre. Rien n'est écrit dans la note de la poutre. Sans longueur de
section, le bouton est éteint et dit pourquoi. Dans le module détaillé, l'en-tête de la vue rappelle la provenance
(« reçue de la poutre V1, le 10/11/2026 »).

**Réalisé le 10/11/2026** (`api_web/poutre.py::transmettre`, `api_web/routeur.py::transmettre`, `site/web/js/etabli.js`,
`beton/taux/depuis_poutre.py`) — ce qui s'écarte du texte ci-dessus, et pourquoi :

- le bouton est sous le nom de la poutre, en tête des résultats, à côté de son taux d'armature : il n'existe que si
  l'option est allumée, et au téléphone il est au même endroit, dans l'onglet « Résultat » (il n'est pas dans le menu
  « ⋯ » : un seul endroit, le même aux trois formats) ;
- la mécanique est GÉNÉRALE et réutilisable (`data-transmettre="<clé>"`, action « transmettre » du routeur :
  `docs/site/MODULES_SITE.md` § 4 sexies) : la destination RELIT l'état qu'on lui envoie avant que la note soit créée ;
- la date de l'envoi vient de l'écran (le moteur n'a pas d'horloge) ; le routeur ne garde qu'une date « JJ/MM/AAAA » ;
- les deux modules ne donnent pas le même nombre, et c'est voulu : le taux simplifié compte au mètre courant et
  s'arrondit au multiple supérieur ; le module détaillé arrête les barres à l'enrobage des abouts, compte l'étrier de
  fermeture et ses crochets. Sur la poutre de l'exemple (30 × 60, 4 m + 2 m) : 100 kg/m³ (97,4 avant arrondi) contre
  96,5 kg/m³. Le lien est PROUVÉ sur les barres : masse détaillée = Σ (masse au mètre de la section × longueur) −
  c × (masse au mètre de la première section + de la dernière).

Preuves : `tests/test_api_web_commun.py` § 9 ter (la mécanique, six mutations du routeur), `tests/test_poutre_taux.py`
§ 4 bis (le bouton, la note envoyée, le lien entre les deux calculs, sept mutations), `tests/test_api_web_taux.py` § 9
(la poutre reçue, de bout en bout), le gel de la poutre (« transmettre:taux » est une action relevée), le harnais
(filtre « envoi », joué aussi par « pilote » : PC et téléphone, dans le vrai navigateur).
