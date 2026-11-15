# Générateur SAF — premier outil : la fondation de grue

*Module du site « Générateur SAF » (page `generateur-saf`, clé `saf`), livré le 14/11/2026 : lots 1 (calcul et export)
et 2 (écran et note) du cahier des charges du bureau. Le lot 3 (automatisation SCIA : XML, ESA_XML, assemblage de la
note) attend les essais S2 à S7 dans SCIA, faits par le bureau. Le lot 4 (15/11/2026 : efforts, ferraillage, plan
ferraillé, coupes, bordereau, et les corrections d'interface) est décrit au § 8.*

## 1. Ce que fait le module

Une dalle en béton armé sur sol élastique sous les quatre pieds d'une grue à tour. On dessine la dalle (rectangle ou
contour), on pose le châssis (celui de la bibliothèque des châssis de grue, ou saisi), on donne les jeux de réactions,
et l'écran dit **tout de suite** la contrainte de contact face à la limite, pour chaque raideur de sol k. On joue sur la
dalle, les appuis et les plots en direct ; « Proposer » cherche la plus petite dalle qui tient ; « Exporter » rend en
un clic un fichier .zip : un fichier SAF par k coché « SCIA » (le classeur Excel que SCIA Engineer importe) et la note
de calcul PDF.

La vérification rapide est une **estimation** pour fixer la dalle avant SCIA, qui reste la référence (décision 4 du
cahier des charges) : pas de calage sur SCIA.

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
| `saf/fondation_grue/efforts.py` | lot 4 : les efforts ELS lus dans les solutions (champs aux nœuds, bandes intégrées sur la largeur exacte, enveloppes, moments de calcul avec torsion, zones de renfort) |
| `saf/fondation_grue/ferraillage.py` | lot 4 : le ferraillage, avec le moteur du module « Poutres béton armé » (treillis, renforts de bandes et locaux, effort tranchant, U, cages, bordereau) |
| `grues/chassis/` | la bibliothèque des châssis de grue : la base (`donnees/chassis_grues.json`, 21 châssis), son moteur de dessin d'origine (`src/chassis_dessin.js`) et son test (`tests/verifier_bibliotheque.js`), tels que livrés ; `acces.py` traduit en Python ses fonctions d'accès, sans changer un résultat |
| `api_web/saf.py` | le module : l'état, la saisie (`modifier`), le sélecteur des châssis, le résultat, l'export, le carnet |
| `api_web/saf_saisie.py` | la colonne de gauche (quatre blocs, lot 4) |
| `api_web/saf_resultat.py` | la colonne de droite (cartes SOL et CONTRÔLES, tableau par k, positions, Proposer, export), la barre du centre et les fenêtres de détail |
| `api_web/saf_ferraillage.py` | lot 4 : les cartes DALLE, BX1 à BY2, PLOTS, le bordereau dépliable et leurs fenêtres de calcul |
| `api_web/dessin_saf.py` | le plan à l'échelle, ses gestes, ses cotes, la carte (contrainte, tassement, moments) |
| `api_web/dessin_ferraillage.py` | lot 4 : le plan ferraillé (avec le fond « taux du treillis ») et les coupes A-A et B-B |
| `api_web/note_saf.py`, `api_web/note_saf_ferraillage.py` | la note de calcul PDF ; ses planches du ferraillage (lot 4) |
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
| `tests/test_api_web_saf.py` | l'écran : la saisie et ses refus, la colonne de droite égale au moteur, le plan et ses gestes, B1 à B5 à l'écran, Proposer, le lien sol, le carnet, l'export à l'octet du moteur, la note PDF, U1 à U9 et R7, R8 à l'écran (lot 4), des mutations |
| `tests/test_saf_grue_efforts.py` | lot 4 : T14, T15, T17, les règles des efforts, la parité avec le calcul de référence (6 cas), des mutations |
| `tests/test_saf_grue_ferraillage.py` | lot 4 : R1 à R8 (R1 rejoue le moteur poutre sur tous les éléments), l'effort tranchant, l'arrachement, la pureté, des mutations |
| `tests/test_saf_grue_plans.py` | lot 4 : D1 à D3 (plan ferraillé, coupes, note PDF), les onglets, le bordereau à l'écran, des mutations |
| harnais, filtre « saf » | dans le vrai navigateur, aux trois formats : la page égale au moteur du dépôt après chaque pas, les gestes mesurés (souris et doigt), la carte peinte, Proposer, l'export .zip, Reprendre, les grues enregistrées, le contour, 44 px au doigt |

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
