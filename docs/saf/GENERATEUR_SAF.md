# Générateur SAF — premier outil : la fondation de grue

*Module du site « Générateur SAF » (page `generateur-saf`, clé `saf`), livré le 14/11/2026 : lots 1 (calcul et export)
et 2 (écran et note) du cahier des charges du bureau. Le lot 3 (automatisation SCIA : XML, ESA_XML, assemblage de la
note) attend les essais S2 à S7 dans SCIA, faits par le bureau. Le lot 4 (15/11/2026 : efforts, ferraillage, plan
ferraillé, coupes, bordereau) est décrit au § 8 ; le lot 5 (15/11/2026, cahier des charges version 4 : SIMPLIFICATION —
principe d'armatures avec le moteur Dalles, efforts SCIA recopiés, note refaite, ferraillage détaillé RETIRÉ) au § 10.
Là où le § 8 et le § 10 diffèrent, le § 10 fait foi.*

## 1. Ce que fait le module

Une dalle en béton armé sur sol élastique sous les quatre pieds d'une grue à tour. On dessine la dalle (rectangle ou
contour), on pose le châssis (celui de la bibliothèque des châssis de grue, ou saisi), on donne les jeux de réactions,
et l'écran dit **tout de suite** la contrainte de contact face à la limite, pour chaque raideur de sol k. On joue sur la
dalle, les appuis et les plots en direct ; « Proposer » cherche la plus petite dalle qui tient (un clic l'applique,
« Annuler » revient) ; « Exporter le SAF » rend LE fichier SAF (le classeur Excel que SCIA Engineer importe, écrit avec
le premier k coché « SCIA ») ; les efforts lus dans SCIA se recopient dans le bloc 3 ; « Imprimer la note » sort la note
PDF, les notes SCIA jointes à la fin.

Depuis le lot 5, le module est une **vérification complémentaire de SCIA, par un calcul indépendant** (décision 11) :
« SCIA donne le niveau, le module donne la forme » — les efforts SCIA saisis recalent le principe d'armatures.

## 2. Les fichiers

| Où | Quoi |
|---|---|
| `saf/classeur.py` | l'écriture d'un classeur .xlsx avec la bibliothèque standard seule (valeurs, nombres, dates, tableaux nommés) |
| `saf/fondation_grue/geometrie.py` | polygones, pieds, plaques d'appui, plots, bandes d'intégration, contour décalé |
| `saf/fondation_grue/positions.py` | les positions des charges « qui tournent » (rotations, miroirs, doublons retirés) et les cas nommés « j.x » |
| `saf/fondation_grue/plaque.py` | la plaque mince (Kirchhoff, élément ACM à 12 ddl) sur sol de Winkler + Pasternak, en compression seule |
| `saf/fondation_grue/verification.py` | la vérification rapide : tous les cas, plusieurs k ; la recherche de « Proposer » |
| `saf/fondation_grue/chassis.py` | le châssis d'un projet : lu dans la bibliothèque (jamais modifiée) ou saisi |
| `saf/fondation_grue/projet.py` | le projet : valeurs par défaut, relecture stricte, modèle (contour, plaques, plots, bandes, cas, contrôles) |
| `saf/fondation_grue/proposition.py` | « Proposer » : par épaisseur, la plus petite dalle sous la limite |
| `saf/fondation_grue/ecriture.py` | les feuilles du SAF d'un projet, la règle des Id ; `ndc_type.json` : le fichier SAF du bureau (textes neutralisés) |
| `saf/fondation_grue/efforts.py` | les efforts ELS lus dans les solutions (moments aux nœuds, bandes intégrées sur la largeur exacte, enveloppes, le cas et le k de chaque valeur — sans torsion depuis le lot 5) |
| `saf/fondation_grue/armatures.py` | lot 5 : le principe d'armatures (l'annexe A, `principe`), avec le moteur du module Dalles pour m_Rd, A_s(), la hauteur de béton et l'effort tranchant |
| `saf/fondation_grue/scia.py` | lot 5 : les efforts SCIA recopiés, r = SCIA ÷ module, l'échelle du principe, l'écart |
| `grues/chassis/` | la bibliothèque des châssis de grue : la base (`donnees/chassis_grues.json`, 21 châssis), son moteur de dessin d'origine (`src/chassis_dessin.js`) et son test (`tests/verifier_bibliotheque.js`), tels que livrés ; `acces.py` traduit en Python ses fonctions d'accès, sans changer un résultat |
| `api_web/saf.py` | le module : l'état, la saisie (`modifier`), le sélecteur des châssis, le résultat, l'export, le carnet |
| `api_web/saf_saisie.py` | la colonne de gauche (quatre blocs sans sous-titre : Dalle et sol, Grue et charges, Ferraillage, Options) |
| `api_web/saf_resultat.py` | la colonne de droite (ligne d'étapes, cartes SOL, ÉQUILIBRE, CONTRÔLES, Proposer, les deux boutons), la barre du centre et les fenêtres de détail |
| `api_web/saf_armatures.py` | lot 5 : la carte ARMATURES, sa fenêtre, le tableau des efforts SCIA et son aide — les PHRASES, une fois, pour l'écran et la note |
| `api_web/dessin_saf.py` | le plan à l'échelle, ses gestes, ses cotes, la carte (contrainte, tassement, armatures) |
| `api_web/dessin_armatures.py` | lot 5 : la couche « Armatures » du plan (zones rouges, renforts, repères, légende) |
| `api_web/dessin_efforts_saf.py` | lot 5 : les diagrammes enveloppes M(s), V(s) de la note, dessinés en points (7,5 pt imprimés) |
| `api_web/note_saf.py` | la note de calcul PDF (lot 5, § 10) |
| `site/web/js/plan.js` | les gestes du plan (glisser, tourner, cote à taper, sommets), au doigt comme à la souris |
| `site/web/js/carnet.js` | le carnet d'un module (les grues enregistrées), dans le navigateur |
| `site/web/css/modules/saf.css` | la feuille du module |

Le moteur n'utilise que la bibliothèque standard et numpy (pas de scipy : le navigateur ne l'a pas). Le calcul de
référence indépendant, avec scipy, est `tests/fondation_grue_ref.py` (l'annexe A du cahier des charges, à l'identique).

## 3. Le calcul

- **Plaque** : élément rectangle ACM (w, ∂w/∂x, ∂w/∂y), grille régulière (maille 0,20 m par défaut depuis le lot 4) ; un élément est
  gardé si son centre est dans le contour ; un plot coulé en place épaissit les éléments dont le centre est dedans.
- **Sol** : un ressort k × surface tributaire à chaque nœud, plus C2 (Pasternak, k/10 par défaut, décision 2) ; en
  **compression seule** — les nœuds qui se soulèvent sont libérés, et on recalcule jusqu'à stabilité.
- **Résolution** : matrice bande résolue par blocs (LDLᵀ tridiagonale par blocs) et formule de Woodbury pour les nœuds
  libérés ; une dalle déjà factorisée est gardée en cache (déplacer la grue ne refactorise rien).
- **Charges** (ELS caractéristiques, comme les combinaisons du SAF) : poids propre γ·h, poids des plots, chaque réaction
  répartie sur sa surface d'appui — sous un plot préfabriqué, sur le carré de diffusion à 45° (côté du carré de même
  aire que la surface d'appui + 2 × hauteur du plot, borné par le plot ; lot 4) ; chaque charge est normalisée (sa somme
  vaut exactement la réaction).
- **Positions** : la grue tourne ; châssis carré : 4 rotations et 4 miroirs, rectangulaire : 2 et 2, doublons retirés.
  Les cas « 1.a, 2.a, 3.a, 1.b… » sont rangés comme dans le fichier du bureau.
- **À égalité** (deux positions que la symétrie rend égales, deux nœuds symétriques), c'est la PREMIÈRE qui gouverne
  (`verification.premier_max`, à 1e-9 près) : sur la dalle type, carrée sous un châssis carré centré, les huit positions
  du jeu 3 sont égales et le cas nommé est « 3.a ». Sans cette règle, le bruit de la virgule flottante choisissait
  « 3.f » en Python et « 3.g » dans le navigateur (trouvé par `tests/test_moteur_partage.py`).
- **Contrôles** : surfaces d'appui dans la dalle, distance au bord ≥ distance mini (une alerte, pas un refus), plots dans
  la dalle, séparés, couvrant leur surface d'appui, équilibre (réaction du sol = charges, à moins de 0,01 %).
- **Proposer** : pour chaque épaisseur (0,30 à 1,20 m par pas de 0,10 m par défaut), le plus petit agrandissement d, au
  pas de 5 cm, qui met σ max sous la limite pour TOUS les k cochés « SCIA » (borne cherchée en doublant, puis
  dichotomie) ; la ligne au plus petit volume de béton est mise en évidence ; un clic l'applique.

## 4. Le fichier SAF

Mêmes feuilles, même ordre, mêmes en-têtes et mêmes tableaux Excel nommés que le fichier SAF du bureau ; des valeurs
seulement (aucune formule), les nombres en cellules numériques, Created et Last update en cellules date (la date de la
page à 0 h : le moteur n'a pas d'horloge), les listes en texte séparé par « ; », les coordonnées arrondies au mm.
L'Id d'un objet du fichier du bureau (même feuille, même nom) est repris ; un objet nouveau reçoit un UUID v5 (espace de
noms `cd663d86-b1df-5f4f-b40e-3c708345d746`, nom « feuille/objet ») : le même objet a toujours le même Id, et le modèle
SCIA type reconnaît ses objets à la mise à jour. Une charge surfacique vaut R ÷ aire du polygone ÉCRIT (sommets arrondis
au mm), pour que la somme fasse exactement R. Depuis le lot 4 : UN seul fichier, « <Projet>.xlsx », écrit avec le
premier k coché « SCIA » (le bureau fait varier k dans SCIA ; la note rappelle les k à passer). Options : anneau D2
de 1, 2 ou 3 cm, arêtes internes sur les axes des bandes, combinaisons Nonlinear, platine ronde en carré de même aire
ou en 48 côtés.

## 5. L'écran

- **À gauche**, huit rubriques, une ouverte à la fois : Projet, Dalle (rectangle ou contour, épaisseur, béton), Sol
  (les valeurs de k, chacune cochée ou non « SCIA » ; « Reprendre » une note de « Rigidité du sol »), Grue (châssis de
  la bibliothèque ou saisi, appuis réels ou rectangle, plots, position, angle, distance mini, grues enregistrées),
  Réactions (des jeux P1 à P4, avec le schéma de numérotation), Bandes, Limite, et un panneau Options replié.
- **Au centre**, le plan à l'échelle, où l'on joue :
  - le châssis se **glisse** (accroches : centre et axes de la dalle, sommets et milieux des bords, grille de 5 cm,
    « plaque à la distance mini » d'un bord pour une dalle rectangulaire) ; l'étiquette dit e_x et e_y pendant le geste ;
  - il se **tourne** par sa poignée (accroches : 0°, 45°, 90°, et parallèle à chaque bord de la dalle) ;
  - une **cote se tape** : un clic sur L_x, L_y ou sur une cote de position pose un champ sur elle ;
  - un clic sur un **bord** le choisit comme repère : les cotes de position se mesurent depuis lui ;
  - les deux bords libres d'un rectangle se tirent pour l'agrandir ;
  - en **contour** : un clic dans le vide ajoute un sommet, un sommet se tire, deux appuis sur un bord y ajoutent un
    sommet, deux appuis sur un sommet le retirent ;
  - un clic sur le châssis ouvre la rubrique Grue, sur la dalle la rubrique Dalle ;
  - la **carte** : contrainte (vert → orange jusqu'à la limite, puis rouge), tassement, ou rien ; la position et le k
    affichés se choisissent au-dessus du plan, le plus défavorable par défaut.
- **À droite** : σ max en kg/cm² et en kPa face à la limite, le cas et le k qui gouvernent ; le tableau par k
  (k | σ max | tassement | décollé | case SCIA ; un clic sur un k l'affiche) ; σ moyen ; les contrôles ; les positions
  (liste repliée, un clic en affiche une) ; « Proposer » (barre de progression, « Arrêter ») ; le bouton unique
  « Exporter · n SAF + note », éteint avec sa raison écrite dessous quand l'export est impossible.

Mêmes fonctions aux trois formats (PC, tablette, téléphone) ; au doigt, chaque bouton fait 44 px au moins.

## 6. Ce qui sert aussi aux autres modules

- **Le carnet d'un module** (`CARNET = True` dans le module) : une liste qui vit dans le navigateur
  (`localStorage["stab.carnets.v1"]`, `js/carnet.js`) et part avec chaque requête ; le module la relit
  (`lire_carnet`) et la rend modifiée (`etat["carnet"]`) — ici, les grues enregistrées (châssis, appuis, plots,
  réactions ; 40 au plus), réutilisables d'un projet à l'autre. Le carnet n'est jamais dans la note.
- **Le plan qui se manipule** (`js/plan.js`) : le moteur écrit dans le dessin ce que chaque geste règle
  (`data-plan-glisser`, `data-plan-tourner`, `data-plan-cote`, `data-plan-clic`, `data-plan-sec`, `data-plan-bord`,
  `data-plan-fond`) ; au lâcher, la valeur part au moteur (« modifier »), qui la relit et redessine.
- `data-agir` (un bouton ou une case du résultat qui modifie la note), `data-suite` (une suite de lectures avec barre de
  progression), `data-reprendre` (choisir une note d'un autre module), le téléchargement d'un .zip
  (`commun.TYPES_BINAIRES`).
- **Le lien avec « Rigidité du sol »** (§ 9 du cahier des charges) : `sol.page_resultats(etat)` rend la page de
  résultats d'une note de sol ; « Reprendre » prend k bas et k haut de sa fondation ouverte (les valeurs distinctes,
  toutes cochées « SCIA ») et la note PDF joint cette page en annexe. Une note en mode abaque (sans k) est refusée.

## 7. Les tests

| Suite | Ce qu'elle prouve |
|---|---|
| `tests/fondation_grue_ref.py` | le calcul de référence de l'annexe A, joué tel quel : sa sortie est celle du cahier des charges (seules les durées changent) |
| `tests/test_saf_grue_moteur.py` | T1 à T13, le moteur face au calcul de référence cas par cas, la pureté (aucun scipy chargé), des mutations |
| `tests/test_saf_grue_export.py` | S1 (non-régression du fichier du bureau), S1b (plots), les règles d'écriture, les options, B1 à B5 côté moteur, la bibliothèque (Python face au moteur d'origine joué par node, sur les 21 châssis, et son propre test), des mutations |
| `tests/test_api_web_saf.py` | l'écran : la saisie et ses refus, la colonne de droite égale au moteur, le plan et ses gestes, B1 à B5 à l'écran, Proposer appliqué et Annuler, le lien sol, le carnet, S0 (le SAF à l'octet du moteur), U1 à U11, T18, les étapes, le collage, E1 à E4 à l'écran (lot 5), des mutations |
| `tests/test_saf_grue_efforts.py` | T14, T15, les règles des efforts (sans torsion), la parité avec le calcul de référence (6 cas), des mutations |
| `tests/test_saf_grue_armatures.py` | lot 5 : T19 (la sortie de l'annexe A, 8 cas), A1 à A4, E1 à E4 côté moteur, la parité avec `principe_armatures` (scipy), A2 (le sol jamais recalculé), la pureté, des mutations |
| `tests/test_saf_grue_note.py` | lot 5 : NC1 à NC8, A3 dans la note, la taille du texte des dessins, les données, des mutations |
| harnais, filtre « saf » | dans le vrai navigateur, aux trois formats : la page égale au moteur du dépôt après chaque pas, les gestes mesurés (souris et doigt), la carte peinte, Proposer, le SAF et la note téléchargés (les étapes cochées), les notes SCIA jointes, la carte Armatures, le collage d'Excel, Reprendre, les grues enregistrées, le contour, 44 px au doigt |

Les essais S2 à S7 se font dans SCIA, par le bureau ; leurs fichiers SAF sont produits par le module lui-même.

## 8. Lot 4 (15/11/2026) : efforts, ferraillage, plans

- **L'écran** (comme le module « Poutres béton armé ») : à gauche quatre blocs (Dalle et sol, Grue et charges, Armatures,
  Options) ; au centre une ligne — les onglets Plan, Ferraillage, Coupes ; la carte (Contrainte, Tassement, Moments, Sans
  carte), le cas et le k, « Enveloppe » en tête ; à droite des cartes sobres (SOL, CONTRÔLES, DALLE, BX1 à BY2, PLOTS), la
  ligne « Non vérifiés », le pied « Méthode du bureau · moteur poutre ». Dégradé « turbo » de 0 au maximum, la limite en
  trait noir sur la légende et en ligne noire sur le plan ; l'unité des contraintes de sol (kg/cm² par défaut) partout.
- **Les efforts** (`efforts.py`) se LISENT dans les solutions de la contrainte de contact (aucun calcul de plus) : moments
  aux coins moyennés aux nœuds, V au centre moyenné ; M(s) et V(s) intégrés sur la largeur EXACTE de chaque bande (bords
  interpolés, pas maille / 4), en enveloppe des cas et des k cochés ; V à d du bord des zones chargées (surface d'appui,
  carré de diffusion, plot coulé), d étant celui du moteur poutre ; moments de calcul m ± |mxy| au centre des éléments ;
  zones de renfort prolongées jusqu'au bord quand la lisière restante est plus étroite qu'une bande.
- **Le ferraillage** (`ferraillage.py`) appelle `beton.poutre.moteur.verifier` sans le modifier (une bande = une poutre
  b × h, h + plot au droit d'un plot coulé en face inférieure ; le treillis et les renforts locaux par mètre). Treillis
  Ø10/Ø12 e = 10/15 cm par face, le plus léger au total (treillis + renforts locaux + renforts de bandes) ou imposé ;
  renforts Ø12/16/20, le plus léger qui couvre ; renforts locaux là où le treillis ne suffit pas (rectangles allongés d'un
  recouvrement) ; C verticaux d'effort tranchant quand le moteur en demande ; U de fermeture ; cage des plots coulés ;
  bordereau (T, R, L, U, C, V), béton et ratio. Le poinçonnement : non vérifié (le logiciel n'en a pas selon la méthode
  du bureau) ; les brins verticaux sous le pied : « à dimensionner » ; l'arrachement : refusé à la saisie.
- **Choix faits seul, à revoir avec le bureau** : la face opposée la plus défavorable (Ø20) pour le terme « 0,25·A_s,req
  opposé » ; un renfort dans le plan des barres de sa direction, entre les barres du treillis (un tous les m
  intervalles, ou deux par intervalle si l'espacement libre du moteur poutre le permet) ; le lit d'une direction pris au
  centre de sa plus grosse barre (≤ 4 mm côté sécurité) ; les C d'effort tranchant à retours de 10 Ø ; recouvrement 50 Ø
  (le logiciel n'a pas de règle).
- **Plans** (`dessin_ferraillage.py`) : le plan ferraillé (treillis, renforts de bandes hachurés, renforts locaux, U,
  cages, repères des coupes ; fond « taux de travail du treillis seul » avec sa ligne noire à 100 %) et les coupes A-A
  (BX1) et B-B (BY1), à l'échelle, avec les repères du bordereau. La note PDF y ajoute quatre planches : Efforts
  (diagrammes M(s), V(s)), Ferraillage, Coupes, Bordereau.
- **Défaut trouvé par le harnais navigateur** : la vue gardait un k que la liste du sol n'avait plus (« Reprendre »
  pendant que k = 30 était affiché) — plus aucune carte ne se peignait. `projet.lire` rend désormais la vue à
  l'enveloppe ; `tests/test_api_web_saf.py` le tient (le contrôle rougit sans la correction).
- **Relecture adverse du lot 4 (15/11/2026)** — 15 constats, tous confirmés par un second relecteur chargé de les
  réfuter, regroupés en 10 défauts, tous corrigés et figés (« relecture 1 » à « relecture 13 » dans
  `test_saf_grue_ferraillage.py`, `test_api_web_saf.py`, `test_saf_grue_plans.py`, chacun avec sa mutation) :
  1. *(non conservatif)* dans la zone d'une bande, la direction PERPENDICULAIRE n'était contrôlée nulle part (une BX ne
     pose que des barres en x) : jusqu'à 180 % de travail affiché « Conforme ». Le contrôle des treillis et la hauteur
     de la zone courante se font désormais direction par direction (`zone_x`, `zone_y`) ; la « zone courante » des
     efforts reste celle de l'annexe A (hors de toutes les zones : les valeurs T14 du cahier). Dalle type : plus de
     renforts locaux, bordereau 2 702 kg au lieu de 2 628, hauteur de la zone courante 74 %.
  2. *(non conservatif)* les C d'effort tranchant s'arrêtaient à d du bord de la zone chargée, là où |V| est le plus
     fort (h = 0,35 m : 36 C posés pour 72 demandés). V à d fixe le CONTRÔLE ; les C vont partout où |V(s)| dépasse ce
     que la dalle reprend sans armatures, jusqu'au bord de la zone chargée.
  3. le treillis automatique retenait un candidat dont les renforts sont impossibles (ils « pesaient » 0 kg) : il
     choisit désormais le plus léger de ceux qui CONVIENNENT.
  4. τ en bande IV (le moteur poutre demande barres inclinées ET étriers) s'affichait « Conforme » : la bande est
     « à vérifier », la ligne « Barres inclinées » et « Non vérifiés » le disent.
  5. V à d illisible (aucune abscisse à d des zones chargées, plots coulés proches) valait 0 et passait pour conforme :
     il est « non vérifié ».
  6. le résumé de la note et les pastilles des planches ne disaient que le SOL : ils disent le plus défavorable du sol
     et du ferraillage (« À vérifier » pour une bande que le module ne peut pas conclure).
  7. la planche Ferraillage ne dessinait pas ses verdicts ; ils disent désormais « Contrôles du moteur poutre
     conformes » (jamais « Ferraillage conforme ») et, avec des plots coulés, « Plots : non vérifié — brins verticaux
     sous le pied à dimensionner ».
  8. « ≤ » était écrit même quand la vérification échoue : le signe suit le taux, écran et note (une seule source,
     `saf_ferraillage.signe`, `face_bande`, `ligne_tranchant`, `hauteur`).
  9. un renfort de bande impossible : l'écran inventait 999 %, la note écrivait « treillis seul » à 526 % — les deux
     disent « renfort impossible : augmenter h » avec le vrai taux.
  10. les coupes dessinaient le renfort de bande empilé sur le treillis (≈ 1,1 cm de d perdu sur chantier) : il est au
     niveau du lit CALCULÉ.
- **Choix faits seul, à revoir avec le bureau (relecture)** : aucun C sous la zone chargée elle-même (le pied, le plot
  coulé) — les brins sous le pied relèvent du poinçonnement, non vérifié ; une bande en τ bande IV n'est ni conforme ni
  non conforme mais « à vérifier ».
- **Barre du centre** : les menus se partagent la place à parts égales ; un choix qui ne tient pas prend son libellé
  court (σ, w, M, « Env. cas », « Env. k »…, `js/plan.js`, `data-courts`) ; si même le court ne tient pas (toile très
  étroite : petit portable 1 280 px avec trois menus), la barre passe exceptionnellement sur deux lignes plutôt que de
  couper un mot. Tablette 1 024 px : une ligne, libellés entiers. Le zoom de la toile est posé au bord de la partie VUE
  du dessin (il tombait sous la colonne des résultats au PC) et existe au téléphone (il y était caché).

## 9. Points signalés au bureau (14/11/2026)

1. **Le module « Bibliothèque des châssis de grue » n'existe pas encore comme écran du site** : la base, son moteur de
   dessin et son test ont été livrés ; le Générateur SAF lit la base par `grues/chassis/acces.py`, dont la parité avec
   le moteur d'origine est testée sur les 21 châssis. Le choix du châssis se fait dans le sélecteur du module.
2. Le fichier SAF du bureau est versionné **sans ses textes de projet** (`ndc_type.json` : le dépôt est public) ; les
   Id, feuilles, en-têtes et lignes fixes sont ceux du fichier.
3. « Reprendre » prend les k de la fondation **ouverte** de la note de sol choisie ; le module « Rigidité du sol »
   n'ayant pas de note PDF, l'annexe est sa page de résultats, mise en page par le module.
4. L'accroche « plaque à la distance mini » n'existe que pour une dalle rectangulaire ; la rotation s'accroche
   parallèlement à **chaque** bord (pas seulement au bord choisi comme repère).
5. La charge d'une surface d'appui dans le SAF vaut R ÷ aire du polygone écrit au mm : q diffère de −R ÷ 0,25 de moins
   de 0,5 % (B5).
6. **Durées mesurées** (navigateur sans écran, machine de travail) : déplacer la grue → résultat en ≈ 0,22 s ; changer
   la dalle → ≈ 0,45 à 0,6 s (cinq k à refactoriser) : au-dessus de la cible de 0,3 s pour ce geste-là ; « Proposer »
   aux valeurs par défaut (dix épaisseurs, trois k cochés, trois jeux) → ≈ 34 s, sous sa barre de progression
   (« Arrêter » l'interrompt) ; l'export en un clic → moins d'une seconde.
7. Les grues enregistrées vivent dans **ce navigateur** (comme le gabarit du bureau) : elles ne passent pas d'un
   appareil à l'autre.

8. **« Proposer » ne regarde que le SOL** (lot 1) : la dalle qu'il propose peut ne pas tenir en ferraillage — dalle type,
   la meilleure ligne (h = 0,30 m, 11,00 × 11,00 m) tient pour le sol (48,7 ≤ 49,0 kPa) mais la hauteur du ferraillage
   est à 126 %. Depuis la relecture du lot 4, le résumé de la note le dit (« Non conforme · ferraillage 126 % »). Faire
   entrer le ferraillage dans « Proposer » est une décision du bureau (à trancher).

## 10. Lot 5 (15/11/2026) : simplification — cahier des charges version 4

Le cahier des charges version 4 est le seul du lot 5 (il remplace les versions 1 à 3). La référence est son annexe A
(`tests/fondation_grue_ref.py`, remplacé tel quel ; sa sortie est celle du cahier, seules les durées changent).

- **Retiré** (§ 2) : le treillis automatique, la torsion, les C d'effort tranchant, les cages des plots, les
  recouvrements et la longueur maxi des barres, le bordereau, les onglets Ferraillage et Coupes, le plan ferraillé
  détaillé et les coupes, les sous-titres des blocs de gauche ; les fichiers `ferraillage.py`, `saf_ferraillage.py`,
  `dessin_ferraillage.py`, `note_saf_ferraillage.py` et leurs suites (R1 à R8, D1 à D3, T17).
- **L'écran** : la ligne d'étapes (Données · Vérification · SAF · Efforts SCIA · Note, cochées au bon moment : une étape
  faite se décoche quand les données qu'elle porte changent) ; quatre blocs sans sous-titre ; au centre la carte
  (Contrainte, Tassement, Armatures avec « Bas · Haut », Sans carte), le cas, le k ; à droite les cartes SOL, ÉQUILIBRE
  (résultante du sol = grue + dalle + plots, en kN), CONTRÔLES, ARMATURES, la ligne « Non vérifiés », Proposer (un clic
  applique, « Annuler »), « Exporter le SAF », « Imprimer la note », « Joindre les notes SCIA ». Un tableau copié d'Excel
  se colle dans les réactions (un jeu par ligne).
- **Plots** (§ 6) : la charge d'un pied sur un plot (préfabriqué ou coulé) est répartie sur le carré de diffusion à 45°
  depuis une plaque de 0,50 × 0,50 m (0,50 + 2 × hauteur, borné par le plot), centré sur le pied ; un plot coulé garde
  sa raideur (régions PL du SAF inchangées).
- **Le principe d'armatures** (§ 7, `armatures.py`) : l'ingénieur CHOISIT les treillis (Ø12 e 15 par défaut) ; le
  moteur du module Dalles (`beton/dalle/moteur.verifier`, sans le modifier) dit ce que chaque treillis reprend
  (m_Rd = A_s ÷ A_s,req(1 kNm/m), A_s,req étant linéaire) ; les renforts des bandes (sur toute la largeur de la bande)
  et les zones rouges / renforts locaux sont ceux de l'annexe A (T19) ; n = ⌈(A_s(m_dim) − A_s du treillis) × largeur ÷
  aire d'une barre⌉ avec A_s() du moteur ; la hauteur de béton (d_min ≤ d) et l'effort tranchant (τ ≤ τ_adm,I) par le
  moteur Dalles — les deux seuls cas, avec le sol, où le module dit d'épaissir.
- **Les efforts SCIA** (§ 8, `scia.py`) : six valeurs (trois quand tout est symétrique, case « x = y »), la valeur du
  module en gris dans le champ ; r = SCIA ÷ module multiplie les moments du module (le principe est refait) ; V SCIA va
  directement au moteur ; σ max : comparaison seule ; écart hors de 0,85 à 1,15 : pastille orange ; un champ vide reste
  vide (r = 1).
- **La note** (§ 10, `note_saf.py`) : garde en deux parties (la vérification complémentaire, une seule pastille ;
  l'annexe Note SCIA, jointe ou à joindre) ; la page Données, seule avec la bande grise ; puis les chapitres en flux à
  deux colonnes (un chapitre court partage la page avec le suivant ; l'en-tête de chaque page nomme ses chapitres) :
  Charges et combinaisons (lues dans les feuilles du SAF : LC1, R et q des lignes FF, surfaces, combinaisons, sol pour
  SCIA), Sol (tous les cas pour chaque k coché), Efforts (diagrammes M(s), V(s) en grand, le treillis en tirets, les
  cartes des moments avec la ligne à m_Rd), Principe d'armatures (deux plans, le tableau des renforts, les contrôles du
  moteur Dalles, « Principe d'armatures, pas un plan d'exécution. »), Comparaison avec SCIA, l'annexe sol. Le texte des
  dessins fait 7 pt au moins imprimé (le dessin est grossi jusqu'à ce que ce soit vrai). Les notes SCIA jointes vont à la
  fin du PDF (`pdf_reunion.reunir`) ; un PDF illisible n'empêche pas la note, l'écran le dit.

**Points signalés (choix faits seul, cahier « prends la solution la plus simple qui respecte le document »)**

1. L'écran du module est celui du site (établi compact), pas un écran Streamlit : l'audit du § 3 a été fait sur lui.
2. V à d se lit à d = h − 6 cm des zones chargées (l'annexe A) ; le moteur Dalles, lui, calcule d = 43 cm (Ø12, enrobage
   5 cm) pour la flexion : les deux valeurs coexistent, chacune là où le cahier la met.
3. Les champs γs, jeu et arrondi (options du moteur poutre) sont retirés : le moteur Dalles prend ses valeurs par défaut
   (γs = 1,5 ; jeu 1 cm ; arrondi 0,5 cm). h est passé au moteur arrondi au cm inférieur (d plus petit : côté sécurité).
4. Le moteur Dalles ne connaît ni C45/55, ni C50/60, ni h > 1,00 m : les armatures y sont « non vérifiées » (à
   vérifier, jamais conformes).
5. Le moteur Dalles n'a pas de règle d'ancrage : 50 Ø par défaut (Options) ; ni d'espacement mini : celui du moteur
   Poutres (EC2 8.2(2)).
6. Le Ø des renforts (« Auto ») est choisi face par face et direction par direction : le plus léger de Ø12, Ø16, Ø20 (à
   poids égal le plus gros ; un espacement trop serré en dernier). Châssis tourné : une bande prend le m_Rd le plus
   faible des deux directions de sa face.
7. La carte Armatures montre toujours l'enveloppe (le principe) : « Bas · Haut » remplace les listes cas et k.
8. Positions symétriques à égalité : le premier cas dans l'ordre de l'annexe (jeu, puis lettre) donne la valeur ; la
   parité avec la référence accepte un cas symétrique de même valeur.
9. Partagés avec les autres modules : le type MIME .xlsx (`commun.TYPES_BINAIRES`), trois crochets JavaScript
   génériques de l'établi (`data-coller`, `data-apres`, `data-joindre`) et `etabli_compact.etapes` (la ligne d'étapes).
10. La note : les grands tableaux tiennent dans une colonne (en-têtes courts R1, q1…) plutôt que sur toute la page ;
    sur les plans du principe, les renforts locaux sont en contour (tirets), sans barres ni repère — le tableau des
    renforts les dit — pour rester lisibles imprimés.

**Relecture indépendante (15/11/2026) : ce qu'elle a trouvé et ce qui a été corrigé** — cinq relecteurs, chaque défaut
contre-vérifié par un second ; chaque correction est figée par un contrôle qui rougit sans elle (une mutation le prouve).

- *Non conservatif* : (1) châssis à 90° : le sens d'une bande se tirait de son NOM — les barres en x n'étaient plus
  contrôlées au point dans les bandes BX devenues verticales ; le sens vient maintenant de la direction de la bande
  (`armatures.axe_bande`, l'axe le plus proche ; à 45° tout juste, celui du nom comme l'annexe A — à moins de 45°, rien ne
  change par rapport à l'annexe) ; l'échelle SCIA du bas suit le nom (r des bandes BX pour BX), celle du haut et celle de
  la carte au point suivent le sens des barres (`echelle_carte`). (2) A_s,min est lu au cm SUPÉRIEUR de h (d reste au cm
  inférieur) : 0,509 m donnait un treillis « au-dessus du minimum » à tort. (3) r = |SCIA| ÷ |module| : un moment tapé
  avec le signe opposé effaçait les renforts de la face ; c'est maintenant une alerte. (4) « x = y » n'est cochée d'office
  que si les appuis, les plots et les ressorts du sol sont symétriques par l'échange x ↔ y. (5) Les valeurs SCIA sont liées
  au modèle pour lequel elles ont été tapées (`scia.signature`, gardée dans `scia.base`) : les données changent, elles ne
  servent plus (« à ressaisir »), l'étape 4 se décoche, retaper une valeur efface les autres. (6) Le collage Excel est tout
  ou rien (plus de lignes que de jeux, case vide : refusé avec la raison).
- *Faux / écarts au cahier* : σ max SCIA en écart compte dans les alertes et dans « saisi » ; la note ne plante plus sur
  une valeur SCIA sans valeur du module (« SCIA seul ») ; les valeurs écrites sous les cartes des moments sont × le rapport
  SCIA ; le carré de diffusion d'un châssis tourné donne ses côtés ; le chapitre Efforts reste hors du moteur Dalles (sans
  trait m_Rd) ; la page Données tient en une colonne large ; un titre part avec son premier diagramme ; « Tous les cas » :
  trois k par tableau ; plus aucun texte sur un autre dans la note (repères x et y décalés le long des barres, liseré blanc
  des diagrammes, valeur pointée et plot dans la marge des cartes imprimées, distances plaque–bord seulement sur la page
  Données) ; l'étape « SAF » ne se décoche plus quand on change un treillis ; la carte ARMATURES dit l'acier demandé, le U
  de fermeture (le treillis le plus FORT, A_s au mètre) et « Ferraillage du plot : à prévoir (non dimensionné) » ; « Non
  vérifiés » garde le poinçonnement et les plots hors du moteur Dalles ; l'aide « Dans SCIA » donne la limite de chaque
  carte et de chaque face ; les alertes SCIA (§ 8.2) sont écrites à l'écran ; un renfort local est coupé par le contour réel
  (sur les lignes de ses barres) — l'annexe A a le même défaut, son principe reste celui de l'annexe, la coupe se fait dans
  `armatures.calculer` (`dans_dalle`).
- *Tests renforcés* : T11 à « 0,0000 % » (5·10⁻⁷, au lieu de 0,01 %), A3 (largeur lue sur le rectangle, barres liées à
  leur repère), NC6 (la ligne M−), U8 (la barre du centre sur une ligne à la tablette, harnais).

## 11. Lot 6 (15/11/2026) : le calcul des bandes

Demande du propriétaire : « une note de calcul au droit des bandes, aux appuis », avec ce que le treillis ne couvre pas et
les renforts inférieurs et supérieurs. Le principe d'armatures (§ 10) calculait déjà les renforts ; le lot 6 les ÉCRIT.

- **La section de calcul de chaque bande** (`saf/fondation_grue/sections.py`) : le moteur du module Dalles
  (`beton/dalle/moteur.verifier`, sans le modifier) calcule la bande comme UNE section b × h (b = 50 ou 100 cm, h au cm
  inférieur), avec ce que le principe y pose : le treillis de chaque face et le renfort qui gouverne (« n barres » dans la
  bande, à l'axe du treillis de sa direction) ; ses moments M⁺ (celui du renfort, le plus fort de son tronçon) et M⁻ (le
  plus fort de la bande, ou celui de son renfort) pris ENSEMBLE — ils ne sont pas au même endroit de la bande : côté
  sécurité —, V à d. Ses vérifications : la hauteur, les deux faces de la direction de la bande, τ.
- **Elle redit le principe** (test L6-2, six dalles) : l'acier demandé est A_s,néc du renfort × b, l'acier posé
  (treillis + n barres) × b, le même τ ; h non entier (0,505 m) : A_s,min au cm supérieur, comme le principe (L6-5).
- **Elle le contrôle** : le principe dimensionne chaque face SEULE ; le moteur Dalles applique toute sa méthode, dont le
  minimum 0,25·A_s,req de la face opposée et le maximum 0,04·b·h. Une face qu'il dit non conforme pour une raison que le
  principe ne voit pas est une alerte ROUGE (« Bande BY1, armature supérieure Y : sous le minimum 0,25·A_s,req de la face
  opposée (moteur Dalles) — renforcer ou épaissir ») ; la hauteur, τ et le minimum de la dalle, déjà dits par le principe,
  ne sont pas redits. *Défaut non conservatif trouvé ainsi* : dalle 10,70 × 0,50, réactions × 2,05 — la hauteur tient
  (99 %) mais le haut des bandes BY est sous le minimum de la face opposée ; le principe seul le disait conforme en flexion
  (L6-4). Sur la dalle de contrôle, rien ne change.
- **La couverture de M(s)** (`dessin_efforts_saf.couverture`) : M⁺ et M⁻ de la bande, et ce que reprennent ses armatures —
  le treillis, puis sur chaque renfort treillis + renfort (m_Rd = A_s posée ÷ A_s,req(1 kN·m/m), le moteur Dalles étant
  linéaire), atteint en rampe sur l'ancrage de chaque extrémité (au bord de la dalle : tout de suite). La courbe des moments
  reste partout dessous (L6-2, six dalles). Dans la zone grise d'un plot coulé, M n'est pas comparé : le plot est hors du
  principe, son ferraillage est à prévoir (la légende le dit).
- **L'écran** : une fenêtre de calcul par direction (« bd-x », « bd-y ») — les lignes Flexion, Hauteur et Effort tranchant
  de la carte ARMATURES l'ouvrent : les hypothèses du module Dalles, le tableau de ses vérifications, la légende et le
  diagramme de couverture, puis le détail de chaque vérification ; deux bandes pareilles (« BX1 = BX2 ») une fois, la plus
  chargée ; différentes (la grue décalée), chacune. Les formules sont celles du module Dalles (`api_web/dalle._formule`),
  une seule écriture (`api_web/saf_bandes.py`).
- **La note** : le chapitre « Calcul des bandes », après le principe d'armatures — un bloc par section écrite, ses efforts,
  ses armatures, le diagramme de couverture, chaque vérification avec ses formules et son verdict.
- **Point signalé au bureau** : la hauteur s'écrit de deux façons équivalentes — le principe (cahier § 7.2) :
  d_min = h_u,min ≤ d = h − y_G, taux d_min ÷ d (68 % sur la dalle de contrôle) ; la section du module Dalles :
  h_min = h_u,min + y_G ≤ h, taux h_min ÷ h (72 %). Le verdict est le même ; la fenêtre et la note le disent. Le décalage
  de la courbe des moments (a_l, EC2 9.2.1.3) n'est pas appliqué au principe (l'annexe A ne le fait pas) : il reste dans
  les « Non vérifiés » (les ancrages).
