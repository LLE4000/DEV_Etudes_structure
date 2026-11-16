# Barres en acier — brief version 11 : lot 0 (inventaire, écarts, plan)

*16/11/2026. Rapport du lot 0 du brief « Barres acier » version 11 (reçu ce jour). Aucun code n'a changé. La
spécification en service reste `BARRES_ACIER.md` ; ce document dit ce que la version 11 y change, et dans quel ordre.*

## 0. En bref

1. **Le module existe déjà.** Les lots 0 à 6 du premier brief (14/11/2026) sont en ligne : l'écran « Choix de
   profilé », les deux méthodes, l'âme et les raidisseurs, la barre chargée, la note PDF (`BARRES_ACIER.md` § 9). La
   version 11 ne part donc pas de zéro : elle CORRIGE et COMPLÈTE un module en service.
2. **Quatre écarts rendent le module en service NON CONSERVATIF** au regard de la version 11, qui dit les avoir lus sur
   les annexes belges (§ 3.1). Le plus lourd : le déversement d'une poutre sans maintien. Le site prend le couple
   (0,4 ; 0,75) là où l'annexe belge demande (0,2 ; 1,0). Sur le cas R2 du brief, charges sur la semelle haute, le site
   écrit **89,5 %** (vérifié) ; la version 11 donne **109,3 %** (non vérifié).
3. **Les textes des normes ne sont pas ici.** Le brief demande de les déposer dans `docs/normes/` ; ce dépôt est PUBLIC,
   ils ne peuvent pas y entrer. Il faut me les envoyer en pièce jointe : ils seront lus, jamais versés (§ 2.8).
4. Proposition : un **lot A** d'abord, qui corrige les quatre défauts non conservatifs (avec regel des références et
   tableau avant/après), puis les lots 1 à 7 de la version 11, réduits à ce qui manque (§ 6).

## 1. Ce qui existe (rappel)

| Lot du premier brief | Fichiers | État |
|---|---|---|
| 1 — méthode historique | `acier/barre/section.py`, `historique.py`, `formules.py` | en ligne |
| 2 — matériau, classes, classe 4, résistances | `parametres.py`, `ec3_classe.py`, `ec3_section.py` | en ligne |
| 3 — flambement, torsion, déversement (M_cr numérique), annexe A | `mcr.py`, `ec3_flambement.py`, `ec3_deversement.py`, `ec3_interaction.py`, `ec3.py` | en ligne |
| 4 — âme, charges concentrées, raidisseurs | `ec3_ame.py`, `raidisseurs.py`, `ame.py`, `enveloppe.py` | en ligne |
| 5 — barre chargée, combinaisons, sections critiques, tronçons | `combinaisons.py`, `statique.py`, `entree.py`, `moteur.py` | en ligne |
| 6 — écran (trois formats) et note PDF | `api_web/profil*.py`, `dessin_barre.py`, `dessin_profil.py`, `note_profil.py` | en ligne |
| 7 — revue finale | — | à faire |

Preuves : `tests/test_barre_*.py` (historique 83 contrôles, section 128, barre 127, âme 160, moteur 82, écriture),
`tests/test_api_web_profil.py`, `tests/test_note_profil.py`, le harnais du navigateur (filtre « barres »).

## 2. Les huit points du lot 0

### 2.1 Pile technique

- **Moteur en Python, bibliothèque standard**, exécuté dans le navigateur par Pyodide (le même code tourne en CPython
  pour les tests). numpy est disponible dans Pyodide et déjà employé par d'autres modules (soudures, profilés,
  fondations de grues) ; le moteur des barres n'en dépend pas.
- **Contrat d'un module** : `api_web/<module>.py` (nouvel_etat, normaliser, modifier, resultat, telecharger, arbre),
  appelé par `api_web/routeur.py` ; le moteur calcule et nomme, l'écran compose et formate. Unités internes N, mm.
- **Écran** : l'« établi compact » (rubriques à gauche, dessin au centre, résultats à droite) ; trois formats (PC,
  tablette, téléphone) ; JS et CSS dans `site/web/`, construit par `site/web/construire.py`.
- **Tests** : des scripts autonomes `tests/test_*.py`, chaque correction figée par une mutation qui doit rougir ; le
  lanceur `lancer_tests.py` ; le harnais du navigateur `site/web/verifier_site.js` (Chromium, trois formats).
- **Données** : tout est gardé dans le navigateur (`localStorage`) ; pas de serveur ni de comptes (Supabase prévu).

### 2.2 Bibliothèque de profils (`acier/donnees/profiles.json`)

1 419 profilés, 21 familles : IPE, IPE A, IPE O, IPN, HEA, HEAA, HEB, HEM, HD, HL, HP, UPE, UPN, L, LI, CHS, SHS, RHS
(finis à chaud) et CHS-F, SHS-F, RHS-F (formés à froid).

**Champs présents** pour toutes les familles : A, Iy, Iz, Wel,y, Wel,z, Wpl,y, Wpl,z, iy, iz, It, masse, périmètre,
c/t des parois. En plus : Iw (I, H, U, tubes), Avz et Avy (tous sauf les cornières), centre de cisaillement (U), axes
principaux u, v (cornières). Dimensions : h, b, tw, tf, r (I, H, UPE) ; r1, r2 (IPN, UPN, cornières) ; d, t (CHS) ;
h, b, t (SHS, RHS).

**Mode de fabrication des tubes** : porté par la famille (CHS / CHS-F…) ; un tube du catalogue n'est donc jamais de
fabrication inconnue. Le cas « inconnu » ne se pose que pour une section saisie.

**Champs manquants** :
- aucun pour la version 1 ;
- pour le lot 9 : la pente des semelles (IPN 14 %, UPN 8 % puis 5 %) et les rayons des coins des tubes ne sont pas
  STOCKÉS, mais ils sont écrits une fois dans `acier/profiles/geometrie.py` (`PENTE_IPN`, `PENTE_UPN`, `rayons_creux`)
  et servent déjà au calcul des propriétés : rien à ajouter.

**Cohérence, mesurée le 16/11/2026 sur les 1 419 profilés** :
- i² = I/A : écart maximal 0,001 % ;
- W = I/v (I/(h/2), I/(D/2)) : écart maximal 0,011 % (CHS 48,3 × 3,2) ;
- Iy > Iz pour tous les I et H.

Le catalogue est cohérent. It et Iw des laminés sont calculés sur le contour exact (Saint-Venant par différences
finies) ; ils diffèrent un peu des catalogues papier (IPE 300 : It 19,77 cm⁴ contre 20,12 dans le brief, effet
inférieur à 0,7 % sur M_cr).

**Profilés personnalisés** : la page « Catalogue des profilés » sait déjà saisir un PRS, un caisson, un tube, un plat ou
un rond (`api_web/catalogue.py`, `acier/profiles/familles.py`). Ils sont gardés DANS LA NOTE (« persos »), donc dans le
navigateur, par note. Le PRS de cette page est bisymétrique, sans gorge, avec It = Σ b·t³/3 (défaut non conservatif déjà
écrit : `consultation.reserve_torsion`). Le module des barres ne lit pas ces profilés aujourd'hui : il n'accepte que le
catalogue (`acier/barre/section.FAMILLES`) ou des propriétés saisies (mode « efforts imposés »).

### 2.3 Solveur RDM (`rdm/poutres.py`)

Formulaire en écritures fermées (Clebsch, trois moments de Clapeyron) : deux appuis, console, encastrée–appuyée,
bi-encastrée, poutre continue de 2 à 5 travées, E·I constant. Charges réparties (totales ou partielles), triangulaires,
ponctuelles, couples. Il rend M, V, la flèche et les réactions en tout point, et contrôle ses propres résultats.

Ce que le module des barres y ajoute (`acier/barre/statique.py`, `combinaisons.py`, `enveloppe.py`) : les cas de charge
par nature, l'effort normal par cas, les combinaisons ELU et de service, les enveloppes, les sections critiques (appuis,
ruptures de chargement, zéros exacts de V), les tronçons de déversement.

Limites : pas d'appui élastique, pas de portique, au plus quatre cas variables (2⁴ arrangements), huit charges par cas.

### 2.4 Combinaisons

Il n'y a pas de moteur EN 1990 commun au site. Celui des barres (`acier/barre/combinaisons.py`) écrit (6.10) avec les
ψ belges des catégories A à H ; G favorable à 1,00 ; la disposition défavorable par arrangement. Il ne fait pas
l'enveloppe (6.10a)/(6.10b), ni la règle belge de courte durée. ψ₁ de la neige y vaut 0,2 (la version 11 : 0 ; sans
effet aujourd'hui, ψ₁ ne sert à aucune vérification).

### 2.5 Notes de calcul (`ndc_pdf/`)

- Planche commune aux notes Poutre, Dalle, Barres : panneau de gauche avec le dessin, deux colonnes de calcul,
  vérifications numérotées, bande de verdict, page de suite quand une planche déborde (`ndc_pdf/styles.py`, `kit.py`,
  `riche.py`).
- **Figures vectorielles** : `ndc_pdf/svg.py` peint dans le PDF le SVG même de l'écran (rect, line, path, polygon,
  text, g transform, hachures ; mode « riche » pour les indices et les aires translucides). On ajoute une figure en
  écrivant son SVG une fois, pour l'écran et la note.
- Statuts imprimés : vérifié, non vérifié, « attention » au-delà de 95 %, non couvert.

### 2.6 Modules acier existants

Flambement, Choix de profilé (les barres), Cornière, Assemblages (pied de poteau, poutre-poteau, poutre-poutre, double
cornière), Soudures, Garde-corps, Catalogue des profilés, Bibliothèque Acier. Charte reprise par les barres : l'établi
compact, les formules « symbole = formule = valeurs = résultat » (`acier/barre/ecriture.py`), un seul arrondi
d'affichage (`resultat.fixe`), la planche de note commune.

### 2.7 Calcul de section (pour le lot 9)

- Pas de mailleur ni de solveur aux éléments finis 2D dans le site.
- Il y a un solveur de Saint-Venant (fonction de Prandtl, différences finies, gradient conjugué, numpy) sur le contour
  exact : `acier/profiles/geometrie.torsion_saint_venant`. C'est lui qui a calculé It du catalogue.
- Il y a les propriétés d'une section composée (Huygens-Steiner) : `rdm/sections.py`.

Le lot 9 partirait de là.

### 2.8 Ce qui me manque

**Les textes.** Aucun n'est dans la conversation ni dans le dépôt :
- NBN EN 1993-1-1 ANB:2018 (tableaux 3.3 à 3.12 ANB, annexes D, E, F, G, H) ;
- NBN EN 1993-1-5 ANB:2011 ;
- NBN EN 1990 ANB:2021 ;
- NBN EN 1993-1-1:2005 avec l'AC:2009 ;
- NBN EN 1993-1-5:2007 avec l'AC:2009 et les amendements A1:2017 et A2:2019 ;
- la NBN EN 1990, si le bureau l'a.

La règle 2 du brief (« compare chaque formule au texte avant de coder un lot ») ne peut pas être tenue sans eux. Le
brief demande de les déposer dans `docs/normes/` : **refusé**, le dépôt est public et ces textes sont sous licence. Les
envoyer en pièce jointe dans la conversation : je les lis, je n'en verse que les valeurs nécessaires (comme aujourd'hui
dans les docs).

Les plus urgents, parce qu'ils décident un résultat et que le brief ne les recopie pas :
1. les tableaux 3.4 à 3.12 ANB — f_y des tubes EN 10210-1 et EN 10219-1, et des S420 / S460 ;
2. l'annexe D ANB (p. 21 à 28), à coder « depuis le PDF » ;
3. le tableau E.1 ANB complet (C3 selon ψ et le signe de ψf) et les figures E.3 à E.10 ANB ;
4. les annexes F et G ANB.

**Les données de qualification externe** : deux valeurs de M_cr dans LTBeam, dont un PRS monosymétrique ; les pages
« Benchmarks 17 à 22 » de l'aide de SCIA Engineer (domaine non joignable d'ici, déjà demandé : Q8) ; les exemples du
commentaire JRC de l'EN 1993-1-5 (Q14) ; trois cas recalculés au bureau dans SCIA. Je ne peux pas faire tourner LTBeam
ni SCIA.

**Les décisions** : la liste du § 7.

## 3. Les écarts entre le module en service et la version 11

### 3.1 Non conservatifs — à corriger avant le reste

| # | Sujet | Le site aujourd'hui | Version 11 (annexes belges lues, selon le brief) | Effet mesuré |
|---|---|---|---|---|
| N1 | Couple (λ̄_LT,0 ; β), § 6.3.2.3 | (0,4 ; 0,75) pour tout tronçon sans maintien d'une seule semelle — la plupart des poutres ; (0,2 ; 1,0) sinon | (0,2 ; 1,0) par défaut, dès que M_cr est calculé avec les vrais moments et les maintiens (le solveur du module) ; (0,4 ; 0,75) seulement pour une poutre de bâtiment AVEC maintiens, tous ignorés dans M_cr | R2, charges sur la semelle haute : **89,5 % → 109,3 %** (passe → ne passe pas) ; au centre de cisaillement : 74 % → 89 % |
| N2 | f_y selon l'épaisseur | tableau 3.1 de l'EN (option b) : 235 N/mm² jusqu'à 40 mm en S235, 355 en S355 | option a seule admise : tableau 3.3 ANB (EN 10025-2), palier à 16 mm : 225 en S235, 265 en S275, 345 en S355 entre 16 et 40 mm | 162 des 275 I et H laminés ont une paroi de plus de 16 mm : toutes leurs résistances Eurocode baissent de 4,3 % (S235), 3,6 % (S275), 2,8 % (S355). Tubes : 61 de plus de 16 mm, tableaux ANB à recevoir |
| N3 | Âme près d'une extrémité non raidie, type (c) — le cas de presque tous les appuis de rive | garde la branche m₂ > 0 dès qu'elle est cohérente ; pas de question de « contact » (revient à « oui ») | les deux branches, la plus petite retenue ; contact oui / non / inconnu, défaut « inconnu » → verdict sur s_s = 0 dans (6.10) | R4, c = 0 : **261,9 kN → 199,8 kN** (−24 %) ; 248,9 kN si le contact est établi |
| N4 | η de l'EN 1993-1-5 pour les laminés et les tubes | 1,20 | 1,00 (note complémentaire de 5.1(2) de l'ANB:2011 : η > 1 réservé aux sections soudées) | plancher η·h_w·t_w de A_v,z : V_pl,z,Rd jusqu'à +8 % trop haut (IPE 750x147) sur une soixantaine de profilés ; χ_w plafonné à 1,2 au lieu de 1,0 pour les âmes élancées |

Trois écarts plus petits, dans le même sens :

| # | Sujet | Le site | Version 11 | Effet |
|---|---|---|---|---|
| N5 | A_v,y d'un I ou H laminé | A − h_w·t_w | 2·b·t_f (`DÉCISION BUREAU`, défaut) | HEB 200 : 6 280 → 6 000 mm², V_pl,y,Rd −4,5 % |
| N6 | Méthode historique hors du domaine de la doc (f_y hors AE235 à AE355 ; tube formé à froid de plus de 6 mm) | alerte « extrapolation » ou courbe c gardée, le verdict peut être « vérifié » | statut « hors domaine » : calcul affiché, jamais « vérifié » | tube REALEX (f_y = 450) |
| N7 | Poids propre | masse du catalogue × 9,81 (≈ 77,0 kN/m³) | 78,6 kN/m³ × A (classeurs, H13) | +2 % sur le seul poids propre |

### 3.2 Écarts dans le sens sûr (le site est plus prudent)

- Semelles toujours classées en compression uniforme, même en flexion d'axe faible : la version 11 donne les limites
  sous gradient (9ε/α…), plus favorables.
- Tube rond sous N + M : critère linéaire ; la version 11 donne M_pl,Rd·(1 − n^1,7) (AC:2009), plus favorable.
- Borne inférieure de C_mi,0 (1 − N_Ed/N_cr) appliquée ; l'annexe D ANB n'en a pas.
- Contribution des semelles V_bf,Rd négligée ; montant d'extrémité non rigide.
- Seuil du voilement par cisaillement 60·ε (η = 1,20) : le site vérifie plus d'âmes que la version 11 (72·ε).
- Deux charges rapprochées : « non couvert » ; la version 11 les vérifie ensemble (6.3(2)).
- s_s non saisi = 0, le cas le plus défavorable, écrit dans la note ; la version 11 veut « décision impossible »
  (question Q6).

Aligner ces points sur le texte rend des résultats plus favorables. Ils viennent APRÈS les corrections du § 3.1.

### 3.3 Ce que la version 11 ajoute (absent du site)

- **PRS en I** bi- et monosymétriques (section 18) : saisie, gorge, propriétés de 18.4 (It sur les trois plats avec la
  correction 1 − 0,63·t/b, Iw, centre de cisaillement, ψf, zj), f_y et ε plat par plat, classification depuis le pied
  des cordons, η = 1,20, terme de Wagner dans M_cr, courbes des I soudés, « section soudée équivalente », croquis coté
  dans la note, enregistrement réutilisable.
- **Annexe D ANB** à la place de l'annexe A (gardée en comparaison), avec l'étape 1 (barre susceptible de se déformer
  par torsion ou non) et les étapes 2.a et 2.b.
- **Annexe E ANB** comme contrôle analytique de M_cr (C1, C2, C3 belges, note b, Kωt), et C1 de l'annexe E au lieu de
  1/k_c².
- k_c : les quatre schémas du tableau 6.6 qui manquent (0,90 ; 0,91 ; 0,77 ; 0,82).
- Console en flexion composée : « non couvert » en Eurocode (aujourd'hui calculée par l'annexe A).
- Âme : semelle chargée comprimée ou TENDUE d'après σ_x (charge sur la semelle tendue : critère élastique 6.2.1(5) avec
  σ_z,Ed de 3.2.3) ; f_y des plats de raidisseur selon leur épaisseur (test 18 : 225 N/mm² pour 20 mm) ; N_st du
  raidisseur qui sert au cisaillement ; seuil de résistance locale à moment imposé ; vue avant / après raidisseur.
- Combinaisons : enveloppe (6.10a)/(6.10b) avec ξ = 0,85 (1,00 en géotechnique), règle belge de courte durée (ψ₀ = 0,3
  entre neige et vent), type d'ouvrage, case « de courte durée » par action, machines sans valeur par défaut.
- Saisie : position de la barre (horizontale, verticale, inclinée), maintien « négligé dans M_cr », L_cr = u·L par axe
  avec l'aide du tableau 13.3 (doc p. 16) et de l'annexe G ANB.
- Statuts : « hors domaine » et « bloqué » à côté de « non couvert » ; contrôles d'import (Iy < Iz, champs d'axe
  mélangés → verdict bloqué).
- Note : choix à l'impression (historique, Eurocode 3, les deux par défaut, tableau de comparaison en tête), la longue
  liste d'hypothèses imprimées de la section 9 du brief.

### 3.4 Points du brief que je ne suis pas tels quels

- Le dépôt des normes dans `docs/normes/` (§ 2.8).
- Les noms de personnes du brief et les documents d'origine (classeurs, documentation numérisée) restent hors du dépôt.
- Les fiches de la version 11 que le site a déjà corrigées autrement le restent si le texte ne tranche pas (par
  exemple l'arrondi d'affichage, la relecture des notes d'avant).

## 4. Architecture proposée

On garde le paquet `acier/barre/` (bibliothèque standard, N et mm) et ses noms. Correspondance avec la section 10 du
brief, et ce qui s'ajoute :

| Brief (§ 10) | Site | Changement |
|---|---|---|
| `parametres_nationaux` | `acier/barre/parametres.py` | statuts « lu sur le texte » une fois les PDF reçus ; `DÉCISION BUREAU` (défaut + variante affichée) ; type d'ouvrage |
| matériau (fy, fu) | **nouveau `acier/materiaux.py`** | tableaux 3.3 à 3.12 ANB, une source pour tout le site : le module Flambement et la Bibliothèque Acier y lisent aussi (`acier/flambement/normes.limite_elastique` y délègue) |
| `profils` | `acier/barre/section.py` | contrôles d'import, statut « bloqué » |
| PRS (§ 18) | **nouveau `acier/barre/prs.py`** | propriétés de 18.4, contrôles, f_y par plat ; enregistrement dans la note |
| `historique` | `acier/barre/historique.py` | statuts, PRS (18.6) |
| `ec3_section` | `ec3_classe.py`, `ec3_section.py` | ε par plat, η par fabrication, A_v,y, contrôle élastique pour information |
| `ec3_flambement` | `ec3_flambement.py` | PRS, (F.3) |
| `ec3_deversement` | `mcr.py`, `ec3_deversement.py` + **nouveau `annexe_e.py`** | couple belge, terme de Wagner, centre de cisaillement, contrôle par l'annexe E |
| `ec3_interaction` | **nouveau `ec3_annexe_d.py`** ; `ec3_interaction.py` gardé | l'annexe D décide, l'annexe A s'affiche en comparaison |
| `ec3_ame`, `raidisseurs` | `ec3_ame.py`, `raidisseurs.py`, `ame.py` | deux branches de m₂, contact, semelle tendue, σ_z, charges rapprochées, N_st, f_y des plats |
| combinaisons | `combinaisons.py` | (6.10a)/(6.10b), ξ, courte durée |
| résultat | `resultat.py` | statuts « hors domaine », « bloqué » |
| écran, note | `api_web/profil*.py`, `note_profil.py`, `dessin_profil.py` | saisies nouvelles, éditeur de PRS, option d'impression, hypothèses |
| `torsion` (lot 8), `section_ef` et `cartes` (lot 9) | plus tard | numpy est disponible dans le navigateur pour le lot 9 |

## 5. Ce qui change pour les notes déjà faites

Chaque correction du § 3.1 change des taux imprimés. Les notes gardées dans les navigateurs se recalculent à
l'ouverture (le site ne fige pas les résultats) : une note rouverte après le lot A montrera les nouveaux taux. Les
références des tests (`tests/reference_refonte/`, cas R1 à R4, sélecteur) seront regelées avec un tableau avant / après
dans le commit. L'empreinte du calcul change (`PROJETS_EQUIPE.md` § 8.3) : la marque « vérifié » du module Choix de
profilé tombe, et celle des modules qui lisent la même table de f_y (Flambement, assemblages) si Q3 est acceptée.

## 6. Les lots, ajustés

| Lot | Contenu | Il faut |
|---|---|---|
| **A** | Les défauts non conservatifs N1 à N7 (§ 3.1) : couple belge et case « négligé dans M_cr » ; f_y option a (EN 10025-2 tout de suite, tubes dès réception des tableaux) ; m₂ et contact ; η = 1,00 hors PRS ; A_v,y ; statut « hors domaine » ; poids propre. Regel des références, avant / après, mutations | votre accord (Q2) ; tableaux 3.8 à 3.12 ANB pour les tubes |
| 1 | Méthode historique : statuts « hors domaine » / « bloqué », contrôles d'import, variante du facteur 5,1, PRS (18.6) | — |
| 2 | Section : ε par plat, semelles sous gradient, CHS (1 − n^1,7), contrôle élastique pour information ; PRS : saisie, propriétés, classes, résistances, enregistrement | — |
| 3 | Annexe D ANB (l'annexe A en comparaison), C_mi,0 sans borne, annexe E ANB (C1, C2, C3), terme de Wagner, k_c complets, console N + M « non couvert » ; qualification externe de M_cr | annexes D, E, F, G de l'ANB ; LTBeam |
| 4 | EN 1993-1-5 : semelle tendue et σ_z, charges rapprochées, V_bf,Rd, N_st, f_y des plats, seuil local, vue avant / après ; PRS (η = 1,20) | ANB:2011 ; exemples JRC |
| 5 | Combinaisons (6.10a)/(6.10b), ξ, courte durée, type d'ouvrage ; position de la barre ; L_cr = u·L, aide du tableau 13.3 et de l'annexe G | ANB:2021 de l'EN 1990 |
| 6 | Écran et note : nouvelles saisies, éditeur de PRS et son croquis, option d'impression, hypothèses de la section 9 | Q3, Q7 |
| 7 | Revue finale, comparaison des méthodes sur tous les cas, `À CONFIRMER` restants, documentation | — |
| 8, 9 | Torsion ; calcul de section et cartes de contraintes — après la mise en ligne, si vous les demandez | votre décision |

Chaque lot se clôt par : tests ciblés au vert, mutations, harnais du navigateur sur les trois formats si l'écran
change, puis la batterie complète en fin de lot ; un court rapport ; votre feu vert avant le suivant.

## 7. Questions (une seule liste)

| # | Question | Ma proposition |
|---|---|---|
| Q1 | M'envoyer les textes en pièce jointe (§ 2.8), en commençant par les tableaux 3.4 à 3.12 ANB, l'annexe D ANB et les tableaux E.1 / E.2 ANB | — |
| Q2 | Corriger les défauts non conservatifs N1 à N7 tout de suite (lot A), avant le reste ? Ils changent des taux déjà affichés | oui |
| Q3 | f_y (N2) : changer la table pour TOUT le site (modules Flambement et Choix de profilé, Bibliothèque Acier, assemblages) ? Une seule source | oui |
| Q4 | Les dix `DÉCISION BUREAU` du § 13.2 du brief, avec leurs défauts : η₁ élastique ; contact inconnu → s_s = 0 dans (6.10) ; A_v,y = 2·b·t_f ; § 6.3.2.3 pour les laminés ; méthode historique (charges à 100 %, pas de tolérance de 5 %, déversement de la doc activé, C = 1, f_y des plats épais = celui de l'EC3) ; tube REALEX ; seuil « à la limite » ; mise en ligne après le lot 7 ; console N + M « non couvert » ; PRS en § 6.3.2.2 | les défauts du brief ; seuil « à la limite » : 95 %, celui des notes du site |
| Q5 | L'impression : choisir la méthode AU MOMENT d'imprimer (trois choix, « les deux » par défaut, comme la version 11), ou garder la règle actuelle « la note imprime les méthodes cochées à l'écran » (votre choix du 14/11) ? | la version 11 : un choix à l'impression, prérempli avec les méthodes cochées |
| Q6 | Une longueur d'appui s_s non saisie : garder 0 (le pire cas, écrit dans la note — aujourd'hui) ou « décision impossible » (version 11) ? | garder 0 et l'écrire : le verdict reste du côté sûr, et une note sans s_s se lit quand même |
| Q7 | Les PRS enregistrés : dans la note et dans une liste propre au navigateur (comme les projets), en attendant le partage par Supabase ? | oui |
| Q8 | Le poids propre (N7) : 78,6 kN/m³ × A comme les classeurs, ou la masse du catalogue × 9,81 ? | 78,6, la valeur la plus défavorable |
| Q9 | Qualification externe : pouvez-vous faire calculer au bureau deux M_cr dans LTBeam (le cas R1 et PRS-B) et trois cas dans SCIA ? | — |
