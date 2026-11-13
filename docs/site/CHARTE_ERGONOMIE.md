# Charte d'ergonomie de l'établi — ce que la révision des dalles a décidé, pour tous les modules

*Ouverte le 13/11/2026, à la demande du propriétaire : « tu peux noter tous les changements d'ergonomie visuelle qu'on a
pris dans DAL, pré-DAL, pour l'appliquer à poutre et puis partout ». Chaque règle dit d'où elle vient, où elle est déjà
appliquée, et ce qu'il reste à faire ailleurs. Une règle s'applique aux trois formats (PC, tablette, téléphone) ; seule
la disposition change d'un format à l'autre.*

*Le détail de chaque lot des dalles : `docs/site/DALLES_FUSION.md` § 8.*

## 1. Les règles

| N° | Règle | Origine | Appliquée | À faire |
|---|---|---|---|---|
| E1 | **Une information est encodée à UN endroit.** Le résultat dit ce qui tient ; il ne propose plus de modifier (pas de sélecteur ni d'étoile dans les lignes du résultat). | DAL1 | dalle, prédalle | poutre (son résultat propose encore les armatures), les autres modules |
| E2 | **Pas de message qui répète ce que le rouge dit déjà.** La raison d'une non-conformité est dans la LIGNE de la vérification (« d = −1 cm ≤ 0 · nappe hors de l'épaisseur »), pas dans un message en double dessous. Ce qui n'est PAS vérifié : une seule phrase sous le résultat ; le périmètre entier dans les fenêtres de calcul. | DAL1 | dalle, prédalle | poutre, puis chaque module |
| E3 | **La ligne de l'élément** : le nom (modifiable d'un clic) et la liste à gauche ; le cadenas, copier, supprimer à DROITE. | DAL2 | dalle | poutre et les modules à colonne compacte |
| E4 | **L'état de chaque élément** se lit à la couleur de son nom (liste et ligne) : vert conforme, rouge non conforme, neutre sans effort. | DAL2 | dalle | poutre, puis tous |
| E5 | **Annuler, rétablir, replier les données** en haut à gauche du DESSIN au PC et à la tablette ; au téléphone, dans l'en-tête des données. | DAL2 | dalle (`construire.OUTILS_AU_DESSIN`) | ajouter les modules à OUTILS_AU_DESSIN un par un, en relisant leur ligne |
| E6 | **Des symboles, pas des mots répétés** : « b », « h », « c » ; le mot (« largeur de bande », « épaisseur », « enrobage ») au survol ; un ⓘ seulement quand l'explication est utile (b : la bande de calcul de 1 m). Les symboles en italique, comme dans les formules. Un choix évident n'a pas de libellé (« Dalle pleine · Prédalle »). Les symboles restent ceux des normes : **b minuscule** (confirmé le 13/11). | DAL2 | dalle | poutre (b, h, enrobage…), puis tous |
| E7 | **Les efforts en tableau** : une ligne par direction, une colonne par sens (M⁺ face inférieure tendue, M⁻ face supérieure tendue), V à côté ; valeurs **sans signe**, refus expliqué d'un négatif ; les unités dans les champs. | DAL3 | dalle, prédalle | poutre (ses efforts par section) |
| E8 | **Un champ numérique se vide quand on y entre** : sa valeur en gris, on tape directement ; quitté sans rien taper, il la reprend et rien n'est envoyé. | DAL3 | **tout le site** (`js/etabli.js`) | — |
| E9 | **Les flèches de pas (variante B)** dans tout champ de DIMENSION : deux petites flèches dans le champ au PC, côte à côte et à la hauteur du champ au doigt ; le pas défini avec le champ (à côté de son minimum et de son maximum), donné par le propriétaire. Pas de flèches sur les efforts ni sur les coefficients. | maquette du 13/11, choix B | l'axe des couches des dalles (DAL5c : 0,5 cm, ▲ monte, ▼ descend ; `plaque_compact.fleches`, le style `.et-pas` / `.et-fl`) | les autres dimensions (h, b, c, h_p…) dès que le propriétaire confirme leurs pas (§ 2), puis tous les modules, du même composant |
| E10 | **Le menu « ⋯ » d'une section** : « Dupliquer », « Supprimer » ; pas de verrou par section. | DAL1 | dalle | poutre (« Copier la section ») |
| E11 | **Les options** : la méthode en une ligne ; chaque réglage avec une infobulle courte ; un réglage commun à la note reste dans les options de la note. | DAL1 | dalle, prédalle | les autres modules |
| E12 | **Un dessin nomme chaque objet UNE fois** : l'en-tête d'une vue = le titre de l'élément et la bande des sollicitations, rien d'autre ; le titre d'une coupe dit la direction (« Direction Y · principale ») ; une légende commune sous les coupes, où un treillis est une entrée et chaque renfort garde son repère (« R1 Y ») et son annotation. | DAL5a | dalle, prédalle | poutre (son en-tête répète ses aciers et ses étriers), les autres dessins |

## 2. Les pas des flèches (E9) — à confirmer par le propriétaire

Le propriétaire donnera les pas ; ceux-ci sont la proposition de départ.

| Grandeur | Pas proposé | Remarque |
|---|---|---|
| Épaisseur d'une dalle h | 1 cm | |
| Bande de calcul b | 5 cm | 100 cm presque toujours |
| Enrobage c | 1 cm | le propriétaire : « un enrobage, ce sera du 1 cm » |
| Épaisseur d'une prédalle h_p | 1 cm | |
| Axe d'une couche d'armatures | 0,5 cm | aussi le réglage de la position verticale (§ 3) |
| Jeu du premier lit | 0,5 cm | |
| Largeur et hauteur d'une poutre | 5 cm | le propriétaire : « plutôt du 5 cm » |

## 3. La position verticale des armatures — la proposition du 13/11/2026, acceptée (« Ok lot 5 »)

1. Par défaut, des niveaux par RÈGLE (DAL5) : les fils d'un treillis en contact ; les renforts posés directement sur le
   treillis (côté intérieur) ; la direction secondaire posée sur la principale. **Fait (DAL5b, 13/11/2026)** : la règle et
   ses chiffres, `docs/site/DALLES_FUSION.md` § 8.6.
2. Pour changer un niveau : toucher la couche (dans le dessin ou dans « Armatures ») ; les flèches ▲▼ de son axe la
   montent ou la descendent de 0,5 cm ; le dessin et d suivent aussitôt ; « Annuler » défait ; ↺ rend le niveau par défaut.
   Une couche ne traverse ni une autre couche ni l'enrobage. **Fait (DAL5c, 13/11/2026)** : les flèches dans le champ de
   l'axe, et sur le dessin la barre de la couche choisie (▲ ▼ ↺) — `docs/site/DALLES_FUSION.md` § 8.7.
3. Glisser une couche à la souris : plus tard, au PC seulement, comme raccourci (aimanté à 0,5 cm) — la donnée reste
   l'axe de la couche, que l'on lit en chiffres ; au téléphone, une barre de 1 cm dans un dessin qu'on déplace et zoome
   ne s'attrape pas proprement.
