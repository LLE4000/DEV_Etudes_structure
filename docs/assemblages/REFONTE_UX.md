# Refonte UX / UI / information design — doubles cornières d'âme

Point de départ : commit `52b5e4e` (module fonctionnel, parité prouvée). Le
moteur, les vérifications, les benchmarks et les exports texte ne changent
pas. Ce document est la réflexion **avant** le code ; les décisions sont
motivées, et ce qui contredit une idée initiale du bureau est dit
explicitement.

---

## 1. Diagnostic de l'écran actuel

| Constat | Cause | Effet |
|---|---|---|
| 84 champs empilés, un par ligne, dans 14 accordéons | transcription fidèle du formulaire HTML | 4 à 5 écrans de défilement pour un cas courant |
| Deux entrées pour la même géométrie (formulaire **et** dessin, plus le panneau « Cotes ») | prudence de migration | le bureau ne sait plus « où » est la valeur |
| Le dessin est dans un onglet, sous un bandeau de trois lignes | ordre hérité du HTML | l'objet n'est pas au centre |
| Bandeau d'état verbeux (« Benchmark du module : VALIDÉ ») | reprise intégrale | bruit sur l'information de niveau 1 |
| 38 blocs de vérification empilés, un encadré chacun | reprise des cartes béton | lecture pénible, on ne voit pas ce qui gouverne |
| Note PDF sur 2 planches + garde | mise en page béton réutilisée telle quelle | les bonnes informations, dispersées |
| Notations « ln », « dnt », « déc. » | vocabulaire de l'outil | pas celui de l'Eurocode |

## 2. Principes retenus

1. **Une seule entrée par paramètre.** Ce qui est géométrique se modifie
   **sur le dessin** (et seulement là) ; ce qui ne l'est pas — profilés,
   nuances, cornière, boulons, catégorie, efforts, options — se modifie
   dans une **carte compacte** à côté du dessin (et seulement là). Le
   panneau « Cotes » disparaît de l'écran normal ; il ne revient qu'en repli
   si le dessin interactif est désactivé.
2. **Le dessin est le centre.** Il est à gauche, en pleine hauteur d'écran,
   toujours visible ; toutes les cotes modifiables y sont affichées par
   défaut (pas seulement les « principales »).
3. **Un objet = un bloc.** Profilés, cornières, boulons, efforts : quatre
   cartes de deux lignes, pas quatorze accordéons.
4. **Essentiel / expert.** L'écran normal ne montre que ce qu'on modifie
   dans 90 % des cas ; le reste est sous « Paramètres avancés » et
   « Identification », repliés.
5. **Le dessin diagnostique.** Une géométrie impossible se voit en rouge
   sur le dessin et se lit en une ligne : « Impossible : zc + hc = 310 mm >
   249 mm disponibles ».
6. **Lecture à trois niveaux**, à l'écran comme sur la note : 5 secondes
   (statut, taux, dimensionnant), 30 secondes (taux par élément), contrôle
   (formule, substitution numérique, Ed / Rd, référence).
7. **Ne rien perdre.** Les 38 vérifications restent calculées et
   consultables ; elles sont regroupées et tabulées, pas supprimées.

## 3. La question posée : « tout par le dessin ? »

**Réponse : non, pas tout — mais toute la géométrie, oui, et uniquement là.**

Pourquoi pas tout : une nuance d'acier, une classe de boulon, une catégorie
d'assemblage, un type de trou, un effort, un coefficient partiel n'ont pas de
place naturelle sur un dessin ; les cacher dans des fenêtres surgissantes
rend la définition du cas invisible d'un coup d'œil, et l'ingénieur veut
*lire* le cas (une carte de données) autant que le *voir* (le dessin).
Les logiciels de calcul aboutis font exactement cela : un grand graphique,
et une carte de propriétés compacte à côté.

Pourquoi la géométrie uniquement sur le dessin : c'est ce qui supprime la
double entrée. Les entraxes, pinces, longueur et position des cornières,
grugeage, jeu, gorges de cordons se touchent sur le dessin — ce qu'on voit
est ce qu'on modifie. Le nombre de rangées se règle aussi sur le dessin
(**+ / −** à côté de chaque groupe) et se lit dans la carte « Boulons »
(c'est un décompte, pas une cote : il a sa place aux deux endroits, la valeur
reste unique).

## 4. Architecture de l'écran

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ Assemblage poutre–poutre — doubles cornières d'âme        ◀ Assemblages · 🏠 · 🔄 · 💾 · 📂 · 📄 │
│ ● ASSEMBLAGE VÉRIFIÉ — 74 % · pression diamétrale, âme secondaire                          │
│   Boulons 35 % · Cornières 28 % · Poutre secondaire 74 % · Poutre principale 14 %          │
│ ⛔ Impossible : zc + hc = 310 mm > 249 mm disponibles            [Localiser]  (s'il y a lieu)│
├───────────────────────────────────────────────┬──────────────────────────────────────────┤
│  DESSIN (élévation + plan, cotes cliquables,  │  Vérification | Prédimensionnement       │
│  + / − sur les groupes de boulons,            │  PROFILÉS   principale [HEA 400][S355]    │
│  rouge = en cause)                            │             secondaire [HEA 300][S355]    │
│                                               │  CORNIÈRES  [L100x100x10][S355][orient.]  │
│  niveau : ○ simple ● édition ○ complet        │             fixation P [boulonnée] S [..]  │
│                                               │  BOULONS    [M20][8.8][trou][cat.]        │
│                                               │             S : n1 [3] × n2 [1]  P : …     │
│                                               │  EFFORTS    VEd [125] NEd [0] HEd [0] MEd  │
│                                               │  ▸ Paramètres avancés   ▸ Identification  │
├───────────────────────────────────────────────┴──────────────────────────────────────────┤
│  Vérifications | Prédim | Note | Benchmark | Méthode                                      │
│  BOULONS      table Ed · Rd · η · réf.       CORNIÈRES  matrice aile A | aile B | max     │
│  POUTRE SECONDAIRE  table                    POUTRE PRINCIPALE  table                     │
│  ▸ Formules et substitutions numériques (2 lignes par vérification)                       │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

- **Carte compacte** : quatre blocs, deux lignes chacun ; les dimensions des
  profilés « Personnalisé » n'apparaissent que si ce choix est fait.
- **Paramètres avancés** (repliés) : rayon de grugeage, maintien au
  déversement, d0 imposé, filetage, μ / ks / ELS, coefficients γ, options du
  modèle (excentricité P, facteur 0,8, βLf, exposition), acier
  personnalisé, règles du prédimensionnement, dessin interactif.
- **Identification** (repliée) : projet, repère, rédacteur, date — pour la
  note.
- **Dessin** : niveau par défaut « édition » = toutes les cotes
  modifiables + quelques cotes calculées utiles (z, p3, bA, bB) ; « simple »
  et « complet » restent disponibles.
- **Alertes** : une ligne courte par alerte, construite sans recalcul à
  partir des données de l'alerte (valeur, limite, référence) ; « Localiser »
  met en rouge cotes et éléments, déplie les champs en cause.

Interaction sur le dessin (composant, sans compilation) :

| Geste | Effet |
|---|---|
| clic sur une cote jaune | saisie sur place, Entrée valide → état unique → recalcul |
| clic sur **+** / **−** d'un groupe de boulons | n1 ± 1 → le dessin se redessine, la carte suit |
| clic sur l'étiquette « 3 × 1 » d'un groupe | petite fenêtre : n1, n2, p1, e1 (+ e2,b ou g) |
| clic sur « Localiser » (alerte) | cotes et pièces en cause en rouge, champs marqués |

## 5. Notations (dessin, écran, note)

EN 1993-1-8 Tableau 3.3 partout où il s'applique ; profilés EN 1993-1-1 ;
pour les grandeurs que l'Eurocode ne nomme pas, les conventions des guides
SCI / MSB, avec une légende sur la note.

| Ancien | Nouveau | Sens |
|---|---|---|
| e1, p1, e2, p2 | e1, p1, e2, p2 | pinces et entraxes (Tab. 3.3) |
| e2,b / e1,b | e2,b / e1,b | pinces dans l'âme de la poutre portée (indice b) |
| gA | gA | trusquinage de l'aile A (talon → file) |
| Lc | **hc** | hauteur des cornières |
| zc | zc | dessus de la poutre portée → dessus des cornières |
| ln | **c** | longueur du grugeage |
| dnt / dnb | **dc,sup / dc,inf** | profondeur du grugeage |
| gh | gh | jeu horizontal entre l'âme porteuse et l'about |
| déc. | **Δz** | décalage des dessus de semelles |
| z, p3, a, lh | z, p3, a, ℓh | excentricité, entraxe des files des deux cornières, gorge, retours |

Les clés de calcul (`LC_u`, `l_n`, `d_nt`…) ne changent pas : la parité et
les fichiers `.json` existants restent valables.

## 6. Vérifications : regroupées par élément physique

À l'écran comme sur la note, cinq tableaux compacts :

- **Boulons** — cisaillement S, cisaillement P, traction P, interaction,
  glissement S / P ;
- **Cornières** — une **matrice** : lignes = type de vérification (pression
  diamétrale, cisaillement brut, cisaillement net, rupture de bloc,
  flexion, traction nette, bloc en traction, tronçon en T), colonnes =
  aile A (support) · aile B (portée) · **max** ;
- **Poutre portée** — pression diamétrale, cisaillement brut / net, bloc,
  grugeage (flexion, 2ᵉ file, stabilité c, stabilité dc), traction ;
- **Poutre porteuse** — pression diamétrale, cisaillement local brut / net ;
- **Cordons** (si soudé) — effort au point critique, gorge, longueur.

Colonnes : Ed · Rd · η · réf. courte. Référence courte : « EC3 » = EN 1993-1-8
sauf indication (« EC3-1-1 », « MSB », « P358 », « ECCS 126 »), en pied de
page — jamais « EN 1993-1-8 » répété vingt fois.

**Formules** : pour chaque vérification essentielle, deux lignes au plus —
*formule = substitution numérique = résultat* puis *Ed / Rd → η* — composées
uniquement avec les nombres du moteur, et **testées** : chaque substitution
imprimée est recalculée à partir de ses propres nombres et doit redonner la
valeur du moteur (la leçon de l'audit béton du 21/09).

## 7. La note : une page A4 paysage

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│ Bureau d'Études Valens   Assemblage poutre–poutre — doubles cornières d'âme   date · indice │
│ ● VÉRIFIÉ 74,0 %  dimensionnant : pression diamétrale, âme portée (Tab. 3.4)                 │
│   Boulons 35 % · Cornières 28 % · Portée 74 % · Porteuse 14 %      projet · repère · rédacteur │
│ HEA 400 S355 · HEA 300 S355, grugée 50 × 150 · 2 L100x100x10 S355, hc 190 · M20 8.8 cat. A …  │
├──────────────────────┬────────────────────────────────┬───────────────────────────────────┤
│ ÉLÉVATION            │ BOULONS        Ed  Rd  η  réf. │ POUTRE PORTÉE     Ed  Rd  η  réf. │
│  (dessin coté)       │ CORNIÈRES   aile A | aile B|max│ POUTRE PORTEUSE   Ed  Rd  η  réf. │
│ VUE EN PLAN          │ CORDONS (si soudé)             │ PINCES ET ENTRAXES (Tab. 3.3)     │
│ HYPOTHÈSES (4 lignes)│ FORMULES ET SUBSTITUTIONS …    │ … suite des formules              │
│                      │                                │ CONCLUSION                        │
├──────────────────────┴────────────────────────────────┴───────────────────────────────────┤
│ EC3 = EN 1993-1-8 sauf indication · notations : Tab. 3.3 ; c, dc grugeage ; hc, zc cornières … │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

- même charte (`01_encre`, Poppins / Carlito, chip d'état, filets, encadrés
  de verdict), mêmes primitives `ndc_pdf` ; corps minimal inchangé (6,0 pt) ;
- les formules occupent la place restante et s'ajustent (toutes les
  essentielles avec substitution → sans valeurs intermédiaires → la
  dimensionnante et les NON OK seules) ; les **tableaux ne s'ajustent
  jamais** ;
- **rapport détaillé** conservé (garde + synthèse + développement, 3 pages)
  pour qui veut toutes les démonstrations ; la note d'une page est celle du
  bouton « 📄 Générer PDF ».

## 8. Ce qui ne change pas

Moteur (`moteur.py`, `benchmark.py`, bibliothèques), export texte, parité
(31 cas, oracle Node, dessins au niveau des coordonnées), enregistrement /
chargement `.json` (mêmes clés), registre et page de sélection.

## 9. Protocole de non-régression

1. Sauvegarde : l'état avant refonte est le commit `52b5e4e` (branche
   poussée) ; les 31 cas de référence et le benchmark sont figés dans
   `acier/reference/`.
2. Après refonte, pour chaque cas de référence : Ed, Rd, η, statut,
   alertes, Tableau 3.3, prédimensionnement comparés automatiquement au
   corrigé — le test imprime **« RÉGRESSION CALCUL : 0 différence »** ou
   liste chaque différence.
3. Les substitutions numériques de la note sont vérifiées par un test qui
   les recalcule.
4. L'étalon béton (`ndc_pdf`) reste identique au pixel.
5. Recette : AppTest sur l'application réelle + contrôle en navigateur du
   dessin (cote, + / −, fenêtre de groupe, alerte).

---

## 10. Réalisé le 21/09/2026 — et ce qui diffère du plan

| Prévu (§ ci-dessus) | Réalisé | Où |
|---|---|---|
| Notations Eurocode sur le dessin | `NOTATION_EC` (hc, c, dc,sup, dc,inf, Δz, ℓh) ; parité des dessins rejouée en notation du moteur (`Options(notation={})`) | `schemas.py`, `tests/test_assemblages_schemas.py` |
| Notations dans les textes du moteur (alertes, formules) | une seule fonction de vue `notation.ec()` ; le moteur et l'export texte ne changent pas | `notation.py`, `tests/test_assemblages_formules.py` §2 |
| Toute la géométrie sur le dessin, et seulement là | mode « Édition » (défaut) : toutes les cotes modifiables posées ; étiquettes pour les valeurs nulles (dc, Δz, ℓh) ; la carte ne porte aucun champ de géométrie (test) ; panneau « Cotes » seulement en repli | `schemas.Options.editables`, `ecran_saisie.GEOMETRIE_DESSIN` |
| + / − et fenêtre de groupe | poignées `data-action="n1S_u:+1"`, étiquette `data-group` ; fenêtre n1, n2, p1, (p2), e1, e2,b / gA ; message unique `{changes:{…}}` | `svg_cliquable/static/index.html`, `ecran_resultats._traiter_clic` |
| Carte compacte essentiel / expert | 5 blocs (mode et fixations, profilés, cornières, boulons, efforts) de 1 à 3 lignes, titre à gauche ; « Paramètres avancés » et « Identification » repliés ; aides longues en infobulle | `ecran_saisie.BLOCS / AVANCES` |
| Statut une ligne, taux par élément, alertes courtes | `bloc_statut_ligne`, `bloc_alerte_ligne` ; titres courts par identifiant d'alerte + première phrase chiffrée ; « Benchmark du module » n'apparaît plus que dans l'onglet Benchmark | `ui_commun.py`, `notation.ligne_alerte` |
| Cinq tableaux + matrice des cornières | `synthese.TABLES`, `matrice_cornieres` (aile A · aile B · max), références courtes ; chaque vérification active apparaît exactement une fois (test 31 cas) | `synthese.py` |
| Formules en deux lignes avec substitution testée | gabarits Python → forme littérale et forme numérique ; **2 284 substitutions réévaluées sur les 31 cas, 0 écart** | `formules.py`, test §1 |
| Note d'une page A4 paysage | bandeau (statut, η max, dimensionnante, taux par élément, alertes), ligne de données, trois colonnes (dessins + hypothèses ; boulons + cornières + cordons ; poutres + pinces + conclusion), formules en flux (sélection gloutonne par priorité : dimensionnante, non vérifiées, une par élément, puis les essentielles), légendes en pied ; **31 cas : une page, 0 débordement, corps 6,0 / 6,5 pt** | `note.py`, `tests/test_assemblages_note.py` |
| Rapport détaillé conservé | inchangé (3 pages, notation Eurocode sur ses dessins), bouton de l'onglet Note | `rapport.py` |
| Régression calcul | `tests/test_assemblages_regression.py` imprime **« RÉGRESSION CALCUL : 0 différence »** (31 cas, 23 977 grandeurs, 5 exports texte) | — |

Complément du 21/09/2026 — **style du dessin** (`Options.realiste`, par
défaut à l'écran et sur la note ; le test de parité des dessins rejoue la
géométrie du HTML avec `realiste=False`) :

- congés réels : âme–semelle de la poutre principale (rayon `r` du profilé,
  section en I d'une seule pièce), congé de racine `r` et arrondis de bout
  `r/2` des cornières en plan (convention EN 10056-1, dessin seulement),
  rayon du grugeage `r_n` (jusque-là « information d'exécution » non
  dessinée) ;
- hachures à 45° des parties coupées (section de la principale en
  élévation ; âmes et cornières en plan), jamais des parties vues ;
- cordons d'angle à leur taille : bande de largeur a·√2 le long du bout de
  l'aile B avec ses retours (élévation), triangles a·√2 dans l'angle
  aile–âme (plan) ;
- rondelles (`dw` de la bibliothèque) autour des trous du groupe S.

## 11. V2 du 21/09/2026 — retour du bureau (« 40/100 : trop de champs »)

Le bureau : « à quoi ça sert d'avoir à droite autant de paramètres à remplir
alors qu'on pourrait le faire sur le dessin ? Si on sélectionne la poutre
secondaire, on a les étiquettes où on peut modifier tous ses paramètres…
cotations de dessin limite professionnelles… on pourrait déplacer la
poutrelle horizontalement. »

Réalisé :

- **la colonne de saisie disparaît.** Hors replis, il ne reste qu'UN
  réglage à l'écran : le mode (VÉRIFICATION / PRÉDIMENSIONNEMENT), au-dessus
  du dessin. « Paramètres avancés » (réduits : d0 imposé, coefficients γ,
  options du modèle, acier personnalisé, règles du prédimensionnement) et
  « Identification » sont repliés sous le dessin. La carte complète ne
  revient qu'en repli, si le dessin interactif est désactivé ;
- **chaque pièce se sélectionne** (elle s'illumine) et ouvre son panneau
  complet : poutre portée (profilé, nuance, dimensions personnalisées,
  maintien, rayon de grugeage), poutre porteuse, cornières (dont
  l'orientation ET les deux fixations), boulons (diamètre, classe, trou,
  catégorie, filetage, μ / ks selon la catégorie), groupes S et P
  (rangées, files, entraxes, pinces), cordons (gorge, retours), efforts
  (étiquette « VEd 125 » sur l'âme → VEd, NEd, HEd, MEd). En
  prédimensionnement, les champs pilotés ◆ y sont désactivés ;
- **la poutre portée se déplace à la souris** : le glissement horizontal
  pilote le jeu gh (mm entiers, bulle « gh = 43 mm » pendant le geste, la
  pièce suit en fantôme, le calcul repart au relâchement) ;
- **cotation de dessin propre** : plus de fond jaune permanent — halo blanc
  discret sous chaque étiquette, bleu = modifiable (jaune au survol
  seulement), violet italique = calculée, lignes de cote fines (0,8),
  lignes d'attache très fines (0,5), tangentes des congés âme–semelle de la
  poutre portée en trait fin (tf + r) ; **traits d'axe normalisés** (mixte
  fin) par rangée et par file du groupe S et pour les boulons P — à la
  place des croix et des tirets rouges du HTML, le rouge restant réservé
  aux alertes (l'axe passe en rouge épais quand une alerte le désigne) ;
- **note d'une page sans doublon** : le cartouche sous l'élévation est
  retiré de la note (il répétait la ligne de données et les hypothèses — il
  reste à l'écran, où il est seul), la conclusion ne répète plus le taux
  maximal ni la dimensionnante (déjà au bandeau), l'hypothèse du grugeage
  ne répète plus le MEd du tableau.

Une entrée par paramètre, mise à jour : la géométrie sur le dessin
(cotes + glissement), tout le reste dans les panneaux de pièce ; les
avancés réduits ne portent plus rien de ce que les panneaux couvrent.

Complément (même jour, retour « la semelle est mal faite ») :

- **semelle corrigée** : le contour de la section en I omettait le coin
  sous la semelle supérieure gauche — l'aile partait en biseau du congé
  vers le bout. Sommet rétabli, test qui vérifie les quatre coins ;
- **symboles de soudure EN 22553** : flèche vers le cordon, ligne de
  référence, triangle du cordon d'angle, désignation « a … » à gauche du
  triangle — c'est elle l'étiquette éditable (le clic sur le symbole ouvre
  aussi le panneau des cordons). Un symbole en élévation pour les ailes B,
  un en plan pour les ailes A ;
- **glissement à deux axes** : la poutre portée se déplace aussi
  verticalement — horizontal = jeu gh, vertical = décalage Δz, bulle
  « gh = 36 · Δz = 20 mm » pendant le geste ;
- **arêtes cachées en plan** (trait interrompu fin) : bord de la semelle
  de la porteuse côté attache et bords de semelle de la portée — le
  dégagement du grugeage se lit d'un coup d'œil.

Écarts assumés par rapport aux idées initiales :

- la matrice des cornières de la note ne porte pas de colonne « Réf. » :
  les références, communes aux deux ailes, sont sur une ligne sous la
  matrice (une colonne les repliait sur trois à cinq lignes) ;
- les formules de la note ne sont pas « toutes les essentielles » sur les
  cas chargés : la place restante est remplie par priorité *(depuis la
  finalisation §12, la page ne l'écrit plus — le rapport détaillé reste le
  développement complet)* ;
- l'onglet « Schéma et géométrie » disparaît : le dessin est toujours
  visible, au-dessus des onglets ; « Paramètres retenus » et le Tableau 3.3
  sont dans l'onglet Vérifications ;
- les libellés d'options longs (orientation de la cornière) sont abrégés à
  l'affichage seulement (`ecran_saisie.AFFICHAGE`) ; la valeur enregistrée
  ne change pas.

## 12. Finalisation du 21/09/2026 — colonne figée, plan de principe

Dernier retour du bureau, réalisé le même jour.

**Écran en deux colonnes, dessin FIGÉ.** À gauche, l'élévation puis la vue
en plan, EMPILÉES ; la colonne est `position: sticky` (CSS injecté,
`interface._CSS_STICKY`) : le schéma reste à l'écran pendant qu'on fait
défiler les paramètres et les vérifications à droite. La colonne est
reconnue par son contenu (`:has` sur le conteneur `asm_col_dessin`), jamais
par sa position dans le DOM ; sous 641 px (téléphone), Streamlit empile les
colonnes et le sticky est retiré. Le padding bas du conteneur principal
(10 rem par défaut) est ramené à 1,5 rem : sans cela, la colonne « se
gare » sous le haut de l'écran en fin de page (mesuré : −104 px, soit
exactement `bas du parent − hauteur de la colonne`). À droite : la carte
complète par objet (mode, profilés, cornières, boulons + **visserie**,
efforts, avancés repliés, identification), puis les onglets. Le repli
« dessin interactif désactivé » garde les mêmes colonnes : dessins
statiques à gauche, panneau des cotes en tête de droite. Le cartouche
texte sous l'élévation disparaît de l'écran (redondant avec la carte, il
allongeait la colonne figée) ; la parité continue de le rejouer.

**Visserie.** « Parfois on met un écrou, deux écrous, une rondelle… » :
champ libre « Par boulon » (bloc BOULONS), par défaut « 1 rondelle +
1 écrou ». C'est une annotation de FABRICATION, hors moteur et hors CLES
(la parité fige les 84 entrées du corrigé) : clé de session
`asm_visserie`, enregistrée dans le JSON (`visserie` à la racine du
payload), relue à l'ouverture, portée au cartouche du plan de principe.

**Page 1 allégée.** En-tête compact en trois lignes — objets
(Principale | Secondaire | Cornières), attaches (Boulons | Groupe S |
Groupe P), efforts (VEd | NEd | HEd | MEd) — sans géométrie détaillée
(elle est COTÉE page 2). Hypothèses d'une ligne chacune, références
abrégées (« MSB P5 §4.2.1.1 » — `notation._REMPLACEMENTS`). Pied en deux
lignes : références puis notations. Phrases supprimées : l'ancienne ligne
de géométrie (e1/p1/e2/hc/zc/gA/p3/grugeage « (mm) »), « Autres
vérifications : formules dans le rapport détaillé. », les hypothèses
longues (« répartition élastique… », « interaction quadratique… »), la
légende Notations multiligne.

**Page 2 « PLAN DE PRINCIPE ».** Trois vues — élévation, vue en plan, et
une VUE DE DROITE nouvelle (face de l'âme porteuse, ailes A en vraie
grandeur, perçage du groupe P, section d'about de la portée) — à la MÊME
échelle normalisée, la PLUS GRANDE de la série 1:1, 1:2, 1:2,5, 1:5, 1:10…
qui fait tenir une disposition (cinq candidates, centrées ; à échelle
égale, celle qui remplit le mieux la feuille). Les vues sont RECONSTRUITES
à chaque échelle candidate avec une police imposée
(`Options.fs_force = TEXTE_MM × dénominateur`) : le texte des cotes fait
2,4 mm sur le papier quelle que soit l'échelle — même hauteur de texte sur
les trois vues, comme sur un plan. « Échelle 1:5 » affichée en tête et au
cartouche (six cases : assemblage, poutres, cornières hc·zc, fixations +
visserie, date·indice, échelle). Cotes réparties SANS doublon
(`EXCLURE_ELEVATION`, `EXCLURE_PLAN`, liste blanche `FABRICATION_CALC`) ;
renvois de perçage « 3×Ø22 » (groupe S) et « 6×Ø22 » (groupe P), rayon du
grugeage « r 10 », cote `zt` (dessus de la porteuse → première rangée P :
la référence de perçage de l'âme porteuse).

**Lignes de rupture (ISO 128).** Un profil coupé ne se termine plus par un
bord franc : le bord EST une ligne de rupture (zigzag inséré dans le
contour, `_rupture` / `_rect_rompu`) — élévation : bout de la portée ;
plan : âme porteuse en haut et en bas, âme portée à droite ; vue de
droite : les deux côtés de la porteuse. À l'écran comme sur la note
(rendu réaliste) ; la parité (realiste=False) garde la géométrie du HTML.

Contrôle final : `python3 lancer_tests.py` — **613 OK, 0 échec, 18/18
suites vertes** ; parité dessins 372 vues, 0 divergente ; « RÉGRESSION
CALCUL : 0 différence » ; benchmark VALIDÉ. Recette navigateur : desktop
1600×1000 et tablette 834×1112, position de l'élévation mesurée avant et
après défilement (épinglée sous l'en-tête), glissement gh rejoué.
