# Charte d'ergonomie de l'établi — ce que la révision des dalles a décidé, pour tous les modules

*Ouverte le 13/11/2026, à la demande du propriétaire : « tu peux noter tous les changements d'ergonomie visuelle qu'on a
pris dans DAL, pré-DAL, pour l'appliquer à poutre et puis partout ». Chaque règle dit d'où elle vient, où elle est déjà
appliquée, et ce qu'il reste à faire ailleurs. Une règle s'applique aux trois formats (PC, tablette, téléphone) ; seule
la disposition change d'un format à l'autre.*

*Le détail de chaque lot des dalles : `docs/site/DALLES_FUSION.md` § 8.*

## 1. Les règles

| N° | Règle | Origine | Appliquée | À faire |
|---|---|---|---|---|
| E1 | **Une information est encodée à UN endroit.** Le résultat dit ce qui tient ; il ne propose plus de modifier (pas de sélecteur ni d'étoile dans les lignes du résultat). | DAL1 | dalle, prédalle, poutre (lot 4) | les autres modules |
| E2 | **Pas de message qui répète ce que le rouge dit déjà.** La raison d'une non-conformité est dans la LIGNE de la vérification (« d = −1 cm ≤ 0 · nappe hors de l'épaisseur »), pas dans un message en double dessous. Ce qui n'est PAS vérifié : une seule phrase sous le résultat ; le périmètre entier dans les fenêtres de calcul. | DAL1 | dalle, prédalle, poutre (lot 4 : « Non vérifiées » en une phrase) | chaque module |
| E3 | **La ligne de l'élément** : le nom (modifiable d'un clic) et la liste à gauche ; le cadenas, copier, supprimer à DROITE. | DAL2 | dalle, poutre (lot 4 : le cadenas) | les modules à colonne compacte |
| E4 | **L'état de chaque élément** se lit à la couleur de son nom (liste et ligne) : vert conforme, rouge non conforme, neutre sans effort. | DAL2 | dalle, poutre (lot 4) | tous |
| E5 | **Annuler, rétablir, replier les données** en haut à gauche du DESSIN au PC et à la tablette ; au téléphone, dans l'en-tête des données. | DAL2 | dalle, poutre (lot 4) (`construire.OUTILS_AU_DESSIN`) | ajouter les modules à OUTILS_AU_DESSIN un par un, en relisant leur ligne |
| E6 | **Des symboles, pas des mots répétés** : « b », « h », « c » ; le mot (« largeur de bande », « épaisseur », « enrobage ») au survol ; un ⓘ seulement quand l'explication est utile (b : la bande de calcul de 1 m). Les symboles en italique, comme dans les formules. Un choix évident n'a pas de libellé (« Dalle pleine · Prédalle »). Les symboles restent ceux des normes : **b minuscule** (confirmé le 13/11). | DAL2 | dalle, poutre (lot 4 : b, h, c) | tous |
| E7 | **Les efforts en tableau** : une ligne par direction, une colonne par sens (M⁺ face inférieure tendue, M⁻ face supérieure tendue), V à côté ; valeurs **sans signe**, refus expliqué d'un négatif ; les unités dans les champs. | DAL3 | dalle, prédalle, poutre (lot 4 : M⁺, M⁻, V) | — |
| E8 | **Un champ numérique se vide quand on y entre** : sa valeur en gris, on tape directement ; quitté sans rien taper, il la reprend et rien n'est envoyé. | DAL3 | **tout le site** (`js/etabli.js`) | — |
| E9 | **Les flèches de pas (variante B)** dans tout champ de DIMENSION : deux petites flèches dans le champ au PC, côte à côte et à la hauteur du champ au doigt ; le pas défini avec le champ (à côté de son minimum et de son maximum), donné par le propriétaire. Pas de flèches sur les efforts ni sur les coefficients. | maquette du 13/11, choix B | l'axe des couches des dalles (DAL5c : 0,5 cm, ▲ monte, ▼ descend ; `plaque_compact.fleches`, le style `.et-pas` / `.et-fl`) | les autres dimensions (h, b, c, h_p…) dès que le propriétaire confirme leurs pas (§ 2), puis tous les modules, du même composant |
| E10 | **Le menu « ⋯ » d'une section** : « Dupliquer », « Supprimer » ; pas de verrou par section. | DAL1 | dalle, poutre (lot 4) | — |
| E11 | **Les options** : la méthode en une ligne ; chaque réglage avec une infobulle courte ; un réglage commun à la note reste dans les options de la note. | DAL1 | dalle, prédalle | les autres modules |
| E12 | **Un dessin nomme chaque objet UNE fois** : l'en-tête d'une vue = le titre de l'élément et la bande des sollicitations, rien d'autre ; le titre d'une coupe dit la direction (« Direction Y · principale ») ; une légende commune sous les coupes, où un treillis est une entrée et chaque renfort garde son repère (« R1 Y ») et son annotation. | DAL5a | dalle, prédalle ; poutre (lot 4 : plus la ligne des lits) | poutre : la ligne des étriers reste dans l'en-tête tant que le dessin ne dit pas leur Ø et leur pas (« C1 » seul) ; les autres dessins |
| E13 | **Les mots des armatures** : « Armature inférieure », « Armature supérieure » — jamais « nappe », ni à l'écran, ni dans les dessins, les messages, les aides au survol ou la note PDF ; le troisième type d'armature s'appelle **« Mixte »** (le survol dit « Treillis soudé et barres de renfort »). | DAL9 | dalle, prédalle (et leurs deux moteurs) | tout module qui dirait « nappe » |
| E14 | **Une seule rubrique ouverte à la fois** dans la colonne des données : ouvrir une rubrique referme les autres (une rubrique fermée garde sa ligne de résumé) ; le dessin qui amène à une pièce ouvre sa rubrique et referme toutes les autres ; un appui sur la rubrique ouverte la referme. | UI1 | **tout le site** (`js/etabli.js`) | — |
| E15 | **Une discipline commune au béton armé** : poutre, dalle et prédalle ont la même organisation et les mêmes règles — les armatures se saisissent dans la colonne de gauche SEULEMENT (la colonne de droite ne montre que des résultats), les mêmes mots, les mêmes boutons, la même règle de position des barres (jeu de pose, arrondi, dessin à la position réelle). Un changement demandé pour la dalle passe aussi dans la poutre, dans le même lot. | demande du 13/11 | dalle, prédalle ; poutre (règle de position : lot 3 ; organisation, mots et boutons : lot 4, UNI4) | la note PDF de la poutre (ses mots, la hauteur détaillée) — `SUITE_DU_CHANTIER.md` |

### Décidé le 13/11/2026 — la note PDF et la ligne Hauteur

- La note PDF de la dalle garde sa mise en page (« je préférais l'ancienne note ») : symbole d'une ligne clé-valeur
  collé à sa valeur, « = » d'une accolade à côté de leur calcul — `DALLES_FUSION.md` § 8.8.
- À l'écran, une somme se DÉTAILLE dans la colonne des résultats : « h_min = 12,6 + 3,5 = 16,1 cm ≤ h » (la part
  calculée + l'enrobage) — dalle et prédalle ; la poutre, si le propriétaire le veut.

### Décidé le 13/11/2026 — les mots des armatures (DAL9)

- « Je ne veux plus les mots nappe » : « Armature inférieure », « Armature supérieure » partout où un texte se voit
  (E13). Les CLÉS internes gardent leur nom (chemin « nappes », code « nappes_confondues », attribut data-nappe) : elles
  ne s'affichent pas.
- Le troisième type d'armature s'appelle « Mixte » : « Treillis + renforts » passait sur deux lignes ; « Mixte » se
  comprend (le treillis plus des barres de renfort) et le survol le dit en entier — `DALLES_FUSION.md` § 8.9.

### Décidé le 13/11/2026 (soir) — la discipline commune (E15)

- « Toutes nos règles de jeu doivent être aussi appliquées dans poutre » : la règle de position décidée pour les dalles
  (jeu de pose 1 cm, arrondi 0,5 cm, « façon A », dessin à la position réelle, calcul détaillé) vaut aussi pour la
  poutre.
- « On ne doit pas pouvoir encoder les armatures dans la fenêtre de droite » : la poutre suit l'organisation de la
  dalle et de la prédalle — les armatures à gauche seulement, la colonne de droite pour les résultats.
- Tout petit changement demandé pour les dalles suit dans la poutre : il ne se livre pas sans elle.

### Fait le 13/11/2026 (lot 4, UNI4) — la poutre comme la dalle

- **À droite, des résultats seulement** : la ligne d'une armature NOMME les barres posées (« 2Ø16 »), sans bouton ; le
  choix des armatures s'ouvre dans la rubrique « Armatures », par le bouton de la LISTE de chaque face (l'icône des dalles,
  plus l'ampoule).
- **Les mêmes mots, écrits une fois** (`api_web/etabli_compact.py`, « les mots communs du béton armé ») : « Armature
  inférieure / supérieure » ; « A_s,néc = 588 > A_s = 402 mm² » (plus « A_s,requis », « A_s,posé ») ; la hauteur
  détaillée « h_min = 36,8 + 6 = 42,8 cm > h = 40 cm » ; « Non vérifiées : la flèche et la fissuration. » sous les blocs ;
  à gauche « A_s néc. » en tête de chaque face et « A_s posé » à son pied ; « Caractéristiques » et les symboles b, h, c ;
  les efforts en tableau « M⁺ · inf. tendue | M⁻ · sup. tendue | V » ; la ligne « Méthode du bureau · ELS · z = 0,9 d »
  et le résumé « γs · jeu de pose · arrondi » des Options. La dalle et la prédalle lisent les mêmes fonctions : leur écran
  est resté identique à l'octet (contrôlé).
- **Les mêmes boutons** : le CADENAS dans la ligne de la poutre (une poutre verrouillée se lit, rien ne s'y écrit ; la
  copie d'une poutre verrouillée l'est aussi) ; « Annuler », « Rétablir », « Replier » en haut à gauche du dessin (PC,
  tablette ; l'en-tête des données au téléphone) ; l'état de chaque poutre dit par la couleur de son nom.
- **Le dessin** : l'en-tête de la coupe ne répète plus les lits (le dessin les nomme, « L1 : 2Ø16 ») ; la ligne des
  étriers reste (le dessin ne dit ni leur Ø ni leur pas). Toucher une barre ouvre « Armatures » et déplie la ligne de son
  lit (c'était déjà le cas ; le parcours du navigateur le contrôle désormais sur une barre).
- y_G imposé ne se montre que dès deux lits sur la face, ou imposé (d'un lit seul, y_G est sa position).

## 2. Les pas des flèches (E9) — confirmés par le propriétaire (13/11/2026)

Confirmés tels quels le 13/11/2026 (« c'est bon comme tu as fait ; dans le pire des cas, on les changera »).

| Grandeur | Pas proposé | Remarque |
|---|---|---|
| Épaisseur d'une dalle h | 1 cm | |
| Bande de calcul b | 5 cm | 100 cm presque toujours |
| Enrobage c | 1 cm | le propriétaire : « un enrobage, ce sera du 1 cm » |
| Épaisseur d'une prédalle h_p | 1 cm | |
| Position d'une couche d'armatures | 0,5 cm | aussi le réglage de la position verticale (§ 3) ; la position automatique est un cran (§ 3, point 2) |
| Jeu de pose | 0,5 cm | (c'était le « jeu du premier lit » : depuis le 13/11/2026, il s'ajoute à la position de chaque couche, dans le calcul) |
| Largeur et hauteur d'une poutre | 5 cm | le propriétaire : « plutôt du 5 cm » |

## 3. La position verticale des armatures — la proposition du 13/11/2026, acceptée (« Ok lot 5 »)

1. Par défaut, des niveaux par RÈGLE (DAL5) : les fils d'un treillis en contact ; les renforts posés directement sur le
   treillis (côté intérieur) ; la direction secondaire posée sur la principale. **Fait (DAL5b, 13/11/2026)** : la règle et
   ses chiffres, `docs/site/DALLES_FUSION.md` § 8.6.
2. Pour changer un niveau : toucher la couche (dans le dessin ou dans « Armatures ») ; les flèches ▲▼ de son axe la
   montent ou la descendent de 0,5 cm ; le dessin et d suivent aussitôt ; « Annuler » défait ; ↺ rend le niveau par défaut.
   Une couche ne traverse ni une autre couche ni l'enrobage. **Fait (DAL5c, 13/11/2026)** : les flèches dans le champ de
   l'axe, et sur le dessin la barre de la couche choisie (▲ ▼ ↺) — `docs/site/DALLES_FUSION.md` § 8.7. **Revu (lot 3,
   13/11/2026)** : le champ dit la position RÉELLE (« Position », la règle E15) ; la position automatique, réelle, tombe
   rarement sur la grille de 0,5 cm (3,6 ; 4,7 ; 5,6 cm) — elle est un CRAN : un pas qui l'atteint ou la franchit s'y
   arrête, et ▲ puis ▼ ramènent la couche où elle était (`DALLES_FUSION.md` § 9).
3. Glisser une couche à la souris : plus tard, au PC seulement, comme raccourci (aimanté à 0,5 cm) — la donnée reste
   l'axe de la couche, que l'on lit en chiffres ; au téléphone, une barre de 1 cm dans un dessin qu'on déplace et zoome
   ne s'attrape pas proprement.
