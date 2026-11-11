# Brief pour la conversation RBAR — dessiner dans RBAR un ouvrage calculé par le site du bureau

*Écrit le 11/11/2026 par la conversation qui développe le site de calcul du bureau (dépôt `DEV_Etudes_structure`, module
« Fondations de grues »). Le propriétaire joint à ce message le plan d'armatures de référence : PDF « ARMATURE FONDATION
GRUE TOUR GT1 », plan N° GWW VAL COF 001, indice A du 22-04-25. Lis ce document en entier avant de toucher au dépôt.*

## 1. La demande, en bref

Le bureau calcule ses fondations de grue à tour sur un site interne. Le site exporte déjà le résultat : dimensions,
pieux, armatures choisies. On veut que RBAR, à partir de ce fichier, fasse le plan de coffrage et d'armatures presque
tout seul (« en deux clics »), au niveau du plan de référence joint.

Trois exigences du propriétaire :

1. **La fondation de grue n'existe pas dans RBAR.** RBAR sera vendu hors de la société ; le site, lui, reste interne.
   Un client de RBAR ne doit pas « avoir » la fondation de grue. C'est le site qui envoie les données ; RBAR les
   dessine sans savoir ce que c'est (§ 3).
2. **Le plan de référence est la base** : même nombre de vues, vues identiques, tout est indiqué. Qualité au minimum
   égale à celle du PDF ; tu peux l'améliorer (§ 4, § 9).
3. **Chaque barre dit d'où elle vient** dans la note de calcul : son poste et ses cellules du classeur (§ 6).

## 2. Qui fait quoi

- **Toi** (conversation RBAR) : tout ce qui se passe dans RBAR.
- **Moi** (conversation du site) : le fichier exporté par le site. Je ne touche pas au dépôt RBAR.
- Nous ne nous parlons pas directement : **le propriétaire fait passer les messages**. S'il te manque un paramètre,
  ne l'invente pas et ne le déduis pas du plan : écris une courte « demande au site » (nom voulu, type, unité, sens).
  Je l'ajoute et je renvoie un fichier d'exemple.
- Les règles de ton dépôt (CLAUDE.md, fiches) restent maîtresses pour tout ce qui est interne à RBAR. Si l'une d'elles
  contredit une exigence de ce document (par exemple « une action = un endroit évident » face à un import caché),
  c'est au propriétaire de trancher : signale-le, ne choisis pas à sa place.

## 3. Exigence n° 1 — la fondation de grue n'existe pas dans RBAR

Ce que le propriétaire a dit : RBAR sera vendu à d'autres sociétés ; le site est l'outil interne du bureau ; « les
gens qui ont RBAR n'ont pas la fondation de grue ; c'est mon site qui envoie les données pour la construire ».

Ce que cela veut dire concrètement :

- **Aucun savoir « grue » dans RBAR** : ni commande, ni bouton, ni modèle livré, ni texte, ni aide, ni exemple qui
  parle de fondation de grue, de châssis, de pied de grue. Les formules et les classeurs restent sur le site ; le
  choix des pièces, de leur place et de leurs barres aussi.
- **La porte d'entrée dans RBAR est générale** : « construire un ouvrage décrit par un fichier ». RBAR y lit des
  poutres, des massifs, des pieux, des barres, des étriers, des vues et des textes — rien d'autre. Les titres des vues
  (« GT1 : Détail Poutres transversales »), les noms des pièces, les remarques, le bloc « Grue GT1 » sont des
  **textes du fichier**, jamais des textes de RBAR.
- **Cette porte est cachée aux clients** : le propriétaire parle d'« un truc caché, interne à nous ». Le mécanisme
  est à toi (une piste : un réglage du profil du bureau, à condition que ce profil ne parte pas chez les
  clients). Si tu penses qu'un import général visible de tous sert mieux le produit, c'est une décision du
  propriétaire : pose-lui la question.
- **Ce qui se vérifie** : dans le paquet livré (DLL, bundle, configuration neutre, aide, guide), une recherche de
  « grue » ne trouve rien qui vienne de ce travail ; et un RBAR installé chez un client ne propose pas l'import.
- Le plan de référence et le fichier d'exemple portent un vrai projet et le nom du bureau : ils servent à tes essais,
  ils ne partent ni dans le paquet livré ni dans un exemple fourni aux clients.

La limite, pour que ce soit dit : un fichier reste un fichier. Quelqu'un qui aurait à la fois l'import et un fichier
produit par le site pourrait redessiner cet ouvrage-là. Ce qui est protégé, c'est ce qui fabrique le fichier — le
calcul et le choix des pièces —, et cela ne sort pas du site.

**Conséquence pour le format** : le fichier d'aujourd'hui (§ 5) parle le vocabulaire du site (`stab.fondation_grue`,
`support`, `semelle`, `pieds`, `perfos`…). **N'écris pas ces mots dans le code de RBAR.** Dis-moi plutôt ce que RBAR
veut lire (§ 8) : j'écrirai, à partir des mêmes données, un fichier neutre à ce format.

## 4. Le plan de référence, vue par vue

Ce que je lis sur le PDF (une seule feuille). Le PDF fait foi : corrige-moi si je lis mal.

| Groupe (titre sur le plan) | Échelle | Vues |
|---|---|---|
| GT1 : Détail Poutres transversales | 1/25 | Vue en plan · Élévation · Coupe AA |
| GT1 : Détail Poutres longitudinales | 1/25 | Vue en plan · Élévation · Coupe BB |
| GT1 : Détail Nœud Poutre longitudinale – Poutre transversale | 1/5 | une coupe du croisement des deux poutres |
| GT1 : Implantation | 1/50 | le plan d'ensemble |

Soit **huit vues dessinées**, plus :

- le **bordereau** : 12 repères ; colonnes N°, Élément (la forme), Diam, Nombre (deux facteurs), Total, Distance HOH,
  a, b, c, d, L., L. totale, P.U., P. total, nuance, Remarques (vide) ; en tête les quatre formes cotées (barre
  droite, L, U, étrier) ; total 2 748,31 kg ;
- le bloc « Armatures – Légende » (inférieures : trait plein ; supérieures : tireté) ;
- le bloc « Grue GT1 » (type, châssis, flèche, hauteur sous crochet, charge maximale par pied) ;
- le bloc « Matériaux » (béton, acier, enrobage — « distance entre bord béton et extérieur barres ») ;
- une « Remarque » (relevé des pieux, de la fondation et de la position de la grue à transmettre avant bétonnage) ;
- les « REMARQUES — Placement des barres » : 1. poser les barres inférieures dans les poutres longitudinales ;
  2. poser la cage d'armatures des poutres transversales ; 3. poser les barres supérieures des poutres longitudinales ;
- le cartouche (projet, titre, n° de plan, indice, date, modifications, échelle « 1/50 et 1/20 »).

Ce que montrent les vues :

- **Transversales, vue en plan** : la poutre de 750 × 140 posée sur les deux poutres longitudinales (cotes
  100 / 550 / 100 et 270) ; barres supérieures en tireté, inférieures en trait plein ; les perfos (12) et les barres
  latérales (11).
- **Transversales, élévation** : les trois zones d'étriers cotées 200 / 350 / 200, leurs étiquettes, les pieux sous
  les extrémités, le repère de coupe A-A.
- **Coupe AA** : la section 140 × 70, ses étriers (6) et (7), ses barres (9), (10), (11).
- **Longitudinales, vue en plan** : la poutre de 270 × 100 sur ses deux pieux (cercles), barres (1) à (5).
- **Longitudinales, élévation** : hauteur 75, pieux cotés 50 / 170 / 50, la section de la poutre transversale vue
  en gris, le repère de coupe B-B.
- **Coupe BB** : la section de 100 de large, barres (1) à (4), le pieu dessous.
- **Nœud, 1/5** : la coupe de la poutre longitudinale au croisement. Les deux cages sont en deux couleurs : la
  longitudinale en rouge — (1), (2), (3), (4), (5) —, la transversale en bleu — (6) et (7), (9), (10), (11). Des
  cotes d'enrobage y figurent (4 et 3,1). J'y lis que le dessous de la poutre transversale est 5 cm plus haut que celui de la poutre
  longitudinale, les deux dessus étant au même niveau : **à confirmer sur le PDF**.
- **Implantation, 1/50** : les 8 pieux, les 4 poutres longitudinales, les 2 poutres transversales, le châssis de la
  grue en traits fins avec ses diagonales (« Grue Tour »), une étiquette par famille de poutres.

Chaque groupe de barres porte une étiquette — quantité, Ø, pas, repère — et une **miniature cotée de sa forme** :

- « 2(poutres)x9Ø20 (10) », miniature 50 / 742 / 50 ;
- « 2(poutres)x2x21Ø12-10 (6) », « 4(poutres)x4x23Ø12-12,5 (1) » ;
- « 2(pieux)x4(poutres)x2x10Ø12 (5) Étriers de renfort » ;
- « Perfos en 2ème phase - 4(poutres)x3Ø25 (12) », suivi d'une mention d'ancrage chimique.

Les armatures des pieux sont dessinées en gris dans les élévations ; elles ne sont pas au bordereau.

**Vocabulaire** : le plan appelle « poutre transversale » ce que la note de calcul appelle « poutre support châssis »,
et « poutre longitudinale » ce qu'elle appelle « semelle sur pieux ».

## 5. Ce que le site envoie aujourd'hui

Ceci est en ligne et vérifié par les tests du site. Rien n'a encore été essayé dans AutoCAD.

### 5.1 Les fichiers

| Fichier téléchargé | Contenu |
|---|---|
| `<nom> - RBAR.json` | le fichier d'échange, décrit ci-dessous (`format` : `stab.fondation_grue`, `version` : 2) |
| `<nom> - plan de principe.dxf` | DXF R12, unités cm : contours, axes, pieux en cercles au diamètre réel, pieds de la grue ; calques `STAB_BETON`, `STAB_PIEUX`, `STAB_AXES`, `STAB_GRUE`, `STAB_COTES`, `STAB_TEXTES` |
| `<nom> - armatures.csv` | une ligne par poste d'armature (les mêmes données que le bloc `armatures`) |
| `<nom> - cas de la grue.csv` | les tableaux de cas du calcul : inutile au dessin |

Le site **refuse de produire le fichier RBAR** tant qu'une vérification de la note est « PAS OK » : on ne transmet pas
au dessin une armature que la note refuse. Les armatures exportées sont toujours les armatures **choisies** dans la
note, celles que le calcul a vérifiées ; rien n'est recalculé à l'export.

### 5.2 Les paramètres du fichier `RBAR.json`

Longueurs en **mm**, sauf le bloc `pieces` (en **cm**, comme la note). Diamètres en mm. Sections en mm².

**`note`** — l'identification (des textes, tels que saisis)

| Paramètre | Sens | Greenwood |
|---|---|---|
| `nom` | nom de la fondation dans la note | `F1` |
| `projet` | projet (cellule C3 du classeur) | `GWW4 GT 11-04-25` |
| `grue` | type de grue (C5) | `WOLF 7534` |
| `schema` | `semelles8`, `cadre4` ou `diagonales8` (§ 5.4) | `semelles8` |
| `libelle` | le schéma en clair | |
| `classeur` | la fiche de calcul suivie : `BASE`, `VAR4` ou `VAR8` | `BASE` |
| `statut` | verdict de la note : `ok`, `warn` ou `nok` | `warn` |
| `date`, `indice`, `chantier` | l'identification de la note du site (vide si non saisie) | |

**`materiaux`** — `beton` (« C30/37 »), `acier` (« BE500 »).

**`modelesPoutre`** — au format que RBAR importe déjà (palette › liste des modèles › « Importer JSON… »). Un modèle
par pièce **et par zone d'étriers**, parce qu'un modèle RBAR porte un seul nombre de brins.

| Paramètre | Sens |
|---|---|
| `nom` | « F1 — Poutre support châssis 140x70 — extrémités » |
| `bas` `{n, d}` | barres inférieures, premier groupe : nombre, Ø |
| `basSuivants` `[{n, d}]` | second groupe de barres inférieures, s'il y en a un |
| `haut` `{n, d}` | barres supérieures, premier groupe |
| `hautSuivants` `[{n, d}]` | second groupe de barres supérieures, s'il y en a un |
| `lat` `{nParFace, d}` | barres latérales ; `nParFace` vaut 0 : le calcul ne fixe pas leur nombre (§ 7) |
| `etrier` `{d, espacement, brins}` | étriers de la zone : Ø, pas (mm), nombre de brins |
| `enrobage` | mm, quand la fiche de calcul en a un |
| `domaine` `{largeurMin_mm, largeurMax_mm, hauteurMin_mm, hauteurMax_mm}` | la section pour laquelle le modèle vaut (min = max) |
| `stab` | la traçabilité, à garder telle quelle : `module`, `note`, `piece`, `nombre_de_pieces`, `zone`, `section_cm` `[b, h]`, `etats` `{inf, sup, etriers}`, `saisies` `{inf, sup, etriers}` (les cellules du classeur) |

**`pieux`** — les types de pieux, au format de `types.json` : `code` (« PX53 »), `nom`, `diametre_mm`, `charge_t`,
`etat` (« ELS »).

**`fondations`** — le gabarit de massif, au format de `types.json` : `code` (« MA75 »), `sorte` (« massif »),
`hauteur_mm`, `debord_mm`, `nom`.

**`pieces`** — les pièces de la note, en **cm**

| Paramètre | Sens |
|---|---|
| `cle` | `support`, `semelle`, `poutre`, `secondaire`, `primaire` ou `pieux` |
| `nom` | « Poutre support châssis », « Semelle sur pieux », « Pieux »… |
| `nombre` | nombre de pièces identiques |
| `b`, `h` | largeur et hauteur de la section |
| `portee` | portée de calcul (d'axe en axe des appuis) |
| `longueur` | longueur de la pièce (semelle, poutre primaire) |
| `D`, `entraxe`, `capacite`, `unite_capacite` | pour `pieux` : diamètre, entraxe, capacité (« T ») |

**`implantation`** — la géométrie en plan, en **mm**

| Paramètre | Sens |
|---|---|
| `unite` | `mm` |
| `repere` | « origine au centre du châssis ; x vers la droite, y vers le haut » |
| `pieds` `[{nom, x, y}]` | les quatre appuis de la grue, A, B, C, D |
| `pieux` `[{nom, x, y, diametre}]` | chaque pieu, P1, P2… |
| `pieces` `[{cle, nom, largeur, contour, axe}]` | chaque pièce : son contour (4 points) et son axe (2 points) |
| `paroi` `{x, y0, y1, D}` ou `null` | la paroi de pieux sécants, quand un côté s'y appuie |
| `conventions` | ce que le dessin en plan suppose, en clair |

**`armatures`** — toutes les armatures choisies, une entrée par poste

| Paramètre | Présent pour | Sens |
|---|---|---|
| `element` | tous | `support`, `semelle`, `poutre`, `secondaire`, `primaire` ou `perfos` |
| `nom_element` | tous | le nom de la pièce dans la note |
| `poste` | tous | `inf`, `sup`, `etriers`, `etriers_extremites`, `etriers_centre`, `etriers_appui`, `etriers_travee`, `laterales`, `renfort_pieu`, `perfos` |
| `designation` | tous | le poste en clair (« Armatures inférieures ») |
| `genre` | tous | `barres`, `etriers`, `prescription`, `renfort` ou `perfos` |
| `texte` | tous | l'armature telle que la note l'écrit (« 9Ø20 », « 6 brins Ø12 – 100 mm ») |
| `etat` | tous | `ok`, `warn`, `nok`, ou `info` pour une prescription |
| `saisie` | tous | d'où vient la valeur dans la note (« support/inf ») ; `null` pour une prescription |
| `groupes` `[{n, d}]` | `barres` | un ou deux groupes : nombre, Ø |
| `section`, `requis` | `barres`, `renfort`, `perfos` | section choisie et section requise par le calcul, mm² |
| `brins`, `d`, `pas` | `etriers` | nombre de brins, Ø, pas choisi (mm) |
| `zone` | `etriers` | « extrémités », « partie centrale », « toute la longueur », « appuis », « travée » |
| `pas_calcule` | `etriers` | le pas maximal que donne le calcul (mm) |
| `d` | `prescription` | Ø prescrit ; le reste est dans `texte` (« 2Ø12 tous les 30 à 40 cm ») |
| `brins`, `d`, `brins_base` | `renfort` | étriers de renfort au droit de chaque pieu : brins ajoutés, Ø, brins des étriers de base qui comptent déjà |
| `n`, `d` | `perfos` | par plaque d'appui : nombre, Ø |

**`controle`** — ce que le site montre à l'ingénieur avant l'envoi

| Paramètre | Sens |
|---|---|
| `lignes` `[{element, poste, valeur, etat, cellules, genre}]` | une ligne par armature, avec ses **cellules du classeur** (« G129, G130, G131, G132 ») |
| `manques` | ce que le calcul ne fixe pas (§ 7), en clair |
| `bloquants`, `pret` | les vérifications « PAS OK » ; `pret` est faux s'il y en a |
| `a_regarder` | les vérifications « LIMITE » ou « À VÉRIFIER » |
| `ecarts` | les écarts relevés dans le classeur (information pour l'ingénieur) |

**`limites`** — trois phrases sur ce que RBAR ne lit pas encore dans ce fichier.

### 5.3 Les ajouts de la version 2

Le fichier ne change que par **ajouts** : un paramètre publié ne change ni de nom ni de sens. La version 2
(11/11/2026) ajoute ce que le plan de référence montre et que la version 1 ne portait pas.

| Paramètre | Sens | Greenwood |
|---|---|---|
| `note.fleche` | flèche de la grue, texte saisi (vide si la fiche ne l'a pas) | `55 m` |
| `note.hauteur_crochet` | hauteur sous crochet, texte saisi | `47,7` |
| `note.dimensions_chassis` | châssis, texte saisi | `6m x 6m` |
| `note.niveau_beton` | niveau du béton, texte saisi | `76,85` |
| `materiaux.enrobage_mm` | enrobage (ou `null` si la fiche n'en a pas) | `40` |
| `charges.max_par_pied_kN` | la plus grande réaction verticale d'un pied de la grue, tous cas confondus (nouveau bloc `charges`) | `1004` |
| `armatures[].cellules` | les cellules du classeur de ce poste, en plage (« G129:G132 ») ; vide pour une prescription | |
| `armatures[].origine` | le texte prêt à écrire qui dit d'où vient l'armature : « F1 · Poutre support châssis · Armatures inférieures · classeur G129:G132 » | |

### 5.4 Les trois schémas

Greenwood est le premier (`semelles8`) et le seul qui ait un plan de référence. Les deux autres passent par le même
fichier : la porte de RBAR doit rester générale.

| `schema` | Ouvrage | Pièces (`cle`) |
|---|---|---|
| `semelles8` | 8 pieux — 2 poutres support sur 4 semelles de 2 pieux ; variante : un côté sur une paroi de pieux sécants | `support`, `semelle`, `pieux` |
| `cadre4` | 4 pieux — 4 poutres en cadre | `poutre`, `pieux` |
| `diagonales8` | 8 pieux — poutres primaires à 45° sur 2 pieux, et poutres secondaires | `secondaire`, `primaire`, `pieux` |

### 5.5 Le fichier Greenwood, complet et réel

C'est le fichier `F1 - RBAR.json` que le site produit pour Greenwood (version 2), les armatures choisies étant
**celles du plan de référence** : 9Ø20 et 6Ø16 dans la poutre support, 6Ø25 en bas de la semelle. Tu peux donc
comparer ton dessin au PDF barre pour barre (une exception : le repère 5, § 6). Garde-le comme fichier d'essai.

```json
{
 "format": "stab.fondation_grue",
 "version": 2,
 "note": {
  "nom": "F1",
  "projet": "GWW4 GT 11-04-25",
  "grue": "WOLF 7534",
  "schema": "semelles8",
  "libelle": "8 pieux — 2 poutres support sur 4 semelles de 2 pieux",
  "classeur": "BASE",
  "statut": "warn",
  "date": "",
  "indice": "",
  "chantier": "",
  "fleche": "55 m",
  "hauteur_crochet": "47,7",
  "dimensions_chassis": "6m x 6m",
  "niveau_beton": "76,85"
 },
 "limites": [
  "RBAR importe les modèles de poutre (« modelesPoutre ») : palette › liste des modèles › Importer JSON. Il ne lit pas encore le reste de ce fichier (géométrie, pieux, armatures de renfort, perfos).",
  "Un modèle RBAR porte un seul nombre de brins : deux zones d'étriers différentes font deux modèles, donc deux coupes.",
  "Formes, retours, ancrages, recouvrements et longueur des zones d'étriers ne viennent pas du calcul : RBAR applique le standard du bureau."
 ],
 "modelesPoutre": [
  {
   "nom": "F1 — Poutre support châssis 140x70 — extrémités",
   "bas": {"n": 9, "d": 20},
   "haut": {"n": 6, "d": 16},
   "lat": {"nParFace": 0, "d": 12},
   "etrier": {"d": 12, "espacement": 100.0, "brins": 6},
   "enrobage": 40.0,
   "domaine": {"largeurMin_mm": 1400.0, "largeurMax_mm": 1400.0, "hauteurMin_mm": 700.0, "hauteurMax_mm": 700.0},
   "stab": {"module": "fondation_grue", "note": "F1", "piece": "support", "nombre_de_pieces": 2, "zone": "extrémités", "section_cm": [140.0, 70.0], "etats": {"inf": "ok", "sup": "ok", "etriers": "ok"}, "saisies": {"inf": "G129, G130, G131, G132", "sup": "G136, G137, G138, G139", "etriers": "G145, G146, G147"}}
  },
  {
   "nom": "F1 — Poutre support châssis 140x70 — partie centrale",
   "bas": {"n": 9, "d": 20},
   "haut": {"n": 6, "d": 16},
   "lat": {"nParFace": 0, "d": 12},
   "etrier": {"d": 12, "espacement": 300.0, "brins": 4},
   "enrobage": 40.0,
   "domaine": {"largeurMin_mm": 1400.0, "largeurMax_mm": 1400.0, "hauteurMin_mm": 700.0, "hauteurMax_mm": 700.0},
   "stab": {"module": "fondation_grue", "note": "F1", "piece": "support", "nombre_de_pieces": 2, "zone": "partie centrale", "section_cm": [140.0, 70.0], "etats": {"inf": "ok", "sup": "ok", "etriers": "ok"}, "saisies": {"inf": "G129, G130, G131, G132", "sup": "G136, G137, G138, G139", "etriers": "G152, G153, G154"}}
  },
  {
   "nom": "F1 — Semelle sur pieux 100x75",
   "bas": {"n": 6, "d": 25},
   "haut": {"n": 4, "d": 16},
   "lat": {"nParFace": 0, "d": 12},
   "etrier": {"d": 12, "espacement": 125.0, "brins": 4},
   "enrobage": 40.0,
   "domaine": {"largeurMin_mm": 1000.0, "largeurMax_mm": 1000.0, "hauteurMin_mm": 750.0, "hauteurMax_mm": 750.0},
   "stab": {"module": "fondation_grue", "note": "F1", "piece": "semelle", "nombre_de_pieces": 4, "zone": "toute la longueur", "section_cm": [100.0, 75.0], "etats": {"inf": "ok", "sup": "ok", "etriers": "ok"}, "saisies": {"inf": "G175, G176, G177, G178", "sup": "G181, G182, G183, G184", "etriers": "G189, G190, G191"}}
  }
 ],
 "pieux": [{"code": "PX53", "diametre_mm": 530.0, "etat": "ELS", "charge_t": 75.0, "nom": "Pieu Ø53 — 75 T (ELS)"}],
 "fondations": [{"code": "MA75", "sorte": "massif", "hauteur_mm": 750.0, "debord_mm": 235.0, "nom": "Massif sur pieux — h. 75, débord 23,5"}],
 "materiaux": {"beton": "C30/37", "acier": "BE500", "enrobage_mm": 40.0},
 "charges": {"max_par_pied_kN": 1004.0},
 "pieces": [
  {"cle": "support", "nom": "Poutre support châssis", "nombre": 2, "b": 140.0, "h": 70.0, "portee": 650.0},
  {"cle": "semelle", "nom": "Semelle sur pieux", "nombre": 4, "b": 100.0, "h": 75.0, "longueur": 270.0, "portee": 170.0},
  {"cle": "pieux", "nom": "Pieux", "nombre": 8, "D": 53.0, "entraxe": 170.0, "capacite": 75.0, "unite_capacite": "T"}
 ],
 "implantation": {
  "unite": "mm",
  "repere": "origine au centre du châssis ; x vers la droite, y vers le haut",
  "pieds": [{"nom": "A", "x": -3000.0, "y": 3000.0}, {"nom": "B", "x": 3000.0, "y": 3000.0}, {"nom": "C", "x": 3000.0, "y": -3000.0}, {"nom": "D", "x": -3000.0, "y": -3000.0}],
  "pieux": [
   {"nom": "P1", "x": -3250.0, "y": 3850.0, "diametre": 530.0},
   {"nom": "P2", "x": -3250.0, "y": 2150.0, "diametre": 530.0},
   {"nom": "P3", "x": -3250.0, "y": -2150.0, "diametre": 530.0},
   {"nom": "P4", "x": -3250.0, "y": -3850.0, "diametre": 530.0},
   {"nom": "P5", "x": 3250.0, "y": 3850.0, "diametre": 530.0},
   {"nom": "P6", "x": 3250.0, "y": 2150.0, "diametre": 530.0},
   {"nom": "P7", "x": 3250.0, "y": -2150.0, "diametre": 530.0},
   {"nom": "P8", "x": 3250.0, "y": -3850.0, "diametre": 530.0}
  ],
  "pieces": [
   {"cle": "support", "nom": "Poutre support 1", "largeur": 1400.0, "contour": [[-3750.0, 2300.0], [3750.0, 2300.0], [3750.0, 3700.0], [-3750.0, 3700.0]], "axe": [[-3250.0, 3000.0], [3250.0, 3000.0]]},
   {"cle": "support", "nom": "Poutre support 2", "largeur": 1400.0, "contour": [[-3750.0, -3700.0], [3750.0, -3700.0], [3750.0, -2300.0], [-3750.0, -2300.0]], "axe": [[-3250.0, -3000.0], [3250.0, -3000.0]]},
   {"cle": "semelle", "nom": "Semelle 1", "largeur": 1000.0, "contour": [[-3750.0, 1650.0], [-2750.0, 1650.0], [-2750.0, 4350.0], [-3750.0, 4350.0]], "axe": [[-3250.0, 1650.0], [-3250.0, 4350.0]]},
   {"cle": "semelle", "nom": "Semelle 2", "largeur": 1000.0, "contour": [[-3750.0, -4350.0], [-2750.0, -4350.0], [-2750.0, -1650.0], [-3750.0, -1650.0]], "axe": [[-3250.0, -4350.0], [-3250.0, -1650.0]]},
   {"cle": "semelle", "nom": "Semelle 3", "largeur": 1000.0, "contour": [[2750.0, 1650.0], [3750.0, 1650.0], [3750.0, 4350.0], [2750.0, 4350.0]], "axe": [[3250.0, 1650.0], [3250.0, 4350.0]]},
   {"cle": "semelle", "nom": "Semelle 4", "largeur": 1000.0, "contour": [[2750.0, -4350.0], [3750.0, -4350.0], [3750.0, -1650.0], [2750.0, -1650.0]], "axe": [[3250.0, -4350.0], [3250.0, -1650.0]]}
  ],
  "paroi": null,
  "conventions": ["les poutres support sont dessinées jusqu'à la face extérieure des semelles (le calcul s'arrête aux axes)"]
 },
 "armatures": [
  {
   "element": "support",
   "nom_element": "Poutre support châssis",
   "poste": "inf",
   "designation": "Armatures inférieures",
   "genre": "barres",
   "groupes": [{"n": 9, "d": 20}],
   "texte": "9Ø20",
   "section": 2827.4333882308138,
   "requis": 2458.218477520011,
   "etat": "ok",
   "saisie": "support/inf",
   "cellules": "G129:G132",
   "origine": "F1 · Poutre support châssis · Armatures inférieures · classeur G129:G132"
  },
  {
   "element": "support",
   "nom_element": "Poutre support châssis",
   "poste": "sup",
   "designation": "Armatures supérieures",
   "genre": "barres",
   "groupes": [{"n": 6, "d": 16}],
   "texte": "6Ø16",
   "section": 1206.3715789784806,
   "requis": 614.5546193800028,
   "etat": "ok",
   "saisie": "support/sup",
   "cellules": "G136:G139",
   "origine": "F1 · Poutre support châssis · Armatures supérieures · classeur G136:G139"
  },
  {
   "element": "support",
   "nom_element": "Poutre support châssis",
   "poste": "etriers_extremites",
   "designation": "Étriers aux extrémités",
   "genre": "etriers",
   "brins": 6,
   "d": 12,
   "pas": 100.0,
   "zone": "extrémités",
   "texte": "6 brins Ø12 – 100 mm",
   "pas_calcule": 124.85834568264556,
   "etat": "ok",
   "saisie": "support/etriers_extremites",
   "cellules": "G145:G147",
   "origine": "F1 · Poutre support châssis · Étriers aux extrémités · classeur G145:G147"
  },
  {
   "element": "support",
   "nom_element": "Poutre support châssis",
   "poste": "etriers_centre",
   "designation": "Étriers en partie centrale",
   "genre": "etriers",
   "brins": 4,
   "d": 12,
   "pas": 300.0,
   "zone": "partie centrale",
   "texte": "4 brins Ø12 – 300 mm",
   "pas_calcule": 621.6023431552678,
   "etat": "ok",
   "saisie": "support/etriers_centre",
   "cellules": "G152:G154",
   "origine": "F1 · Poutre support châssis · Étriers en partie centrale · classeur G152:G154"
  },
  {
   "element": "semelle",
   "nom_element": "Semelle sur pieux",
   "poste": "inf",
   "designation": "Armatures inférieures",
   "genre": "barres",
   "groupes": [{"n": 6, "d": 25}],
   "texte": "6Ø25",
   "section": 2945.243112740431,
   "requis": 2243.1941763170084,
   "etat": "ok",
   "saisie": "semelle/inf",
   "cellules": "G175:G178",
   "origine": "F1 · Semelle sur pieux · Armatures inférieures · classeur G175:G178"
  },
  {
   "element": "semelle",
   "nom_element": "Semelle sur pieux",
   "poste": "sup",
   "designation": "Armatures supérieures",
   "genre": "barres",
   "groupes": [{"n": 4, "d": 16}],
   "texte": "4Ø16",
   "section": 804.247719318987,
   "requis": 560.7985440792521,
   "etat": "ok",
   "saisie": "semelle/sup",
   "cellules": "G181:G184",
   "origine": "F1 · Semelle sur pieux · Armatures supérieures · classeur G181:G184"
  },
  {
   "element": "semelle",
   "nom_element": "Semelle sur pieux",
   "poste": "etriers",
   "designation": "Étriers",
   "genre": "etriers",
   "brins": 4,
   "d": 12,
   "pas": 125.0,
   "zone": "toute la longueur",
   "texte": "4 brins Ø12 – 125 mm",
   "pas_calcule": 152.51587364958493,
   "etat": "ok",
   "saisie": "semelle/etriers",
   "cellules": "G189:G191",
   "origine": "F1 · Semelle sur pieux · Étriers · classeur G189:G191"
  },
  {
   "element": "support",
   "nom_element": "Poutre support châssis",
   "poste": "laterales",
   "designation": "Armatures latérales",
   "genre": "prescription",
   "d": 12,
   "texte": "2Ø12 tous les 30 à 40 cm",
   "etat": "info",
   "saisie": null,
   "cellules": "",
   "origine": "F1 · Poutre support châssis · Armatures latérales · prescription du classeur"
  },
  {
   "element": "semelle",
   "nom_element": "Semelle sur pieux",
   "poste": "laterales",
   "designation": "Armatures latérales",
   "genre": "prescription",
   "d": 12,
   "texte": "2Ø12 tous les 30 à 40 cm",
   "etat": "info",
   "saisie": null,
   "cellules": "",
   "origine": "F1 · Semelle sur pieux · Armatures latérales · prescription du classeur"
  },
  {
   "element": "semelle",
   "nom_element": "Semelle sur pieux",
   "poste": "renfort_pieu",
   "designation": "Étriers de renfort au droit de chaque pieu",
   "genre": "renfort",
   "brins": 16,
   "d": 12,
   "brins_base": 5,
   "texte": "16 brins Ø12 + 5 brins des étriers de base",
   "section": 2375.0440461138837,
   "requis": 1914.6683774530056,
   "etat": "ok",
   "saisie": "renfort",
   "cellules": "G234:G236",
   "origine": "F1 · Semelle sur pieux · Étriers de renfort au droit de chaque pieu · classeur G234:G236"
  },
  {
   "element": "perfos",
   "nom_element": "Plaques de répartition",
   "poste": "perfos",
   "designation": "Perfos par plaque",
   "genre": "perfos",
   "n": 3,
   "d": 25,
   "texte": "3Ø25 par plaque (calcul : 1Ø25 par pied)",
   "section": 490.8738521234052,
   "requis": 286.6666666666667,
   "etat": "ok",
   "saisie": "perfos",
   "cellules": "G258:G259",
   "origine": "F1 · Plaques de répartition · Perfos par plaque · classeur G258:G259"
  }
 ],
 "controle": {
  "lignes": [
   {"element": "Poutre support châssis", "poste": "Armatures inférieures", "valeur": "9Ø20", "etat": "ok", "cellules": "G129, G130, G131, G132", "genre": "barres"},
   {"element": "Poutre support châssis", "poste": "Armatures supérieures", "valeur": "6Ø16", "etat": "ok", "cellules": "G136, G137, G138, G139", "genre": "barres"},
   {"element": "Poutre support châssis", "poste": "Étriers aux extrémités", "valeur": "6 brins Ø12 – 100 mm", "etat": "ok", "cellules": "G145, G146, G147", "genre": "etriers"},
   {"element": "Poutre support châssis", "poste": "Étriers en partie centrale", "valeur": "4 brins Ø12 – 300 mm", "etat": "ok", "cellules": "G152, G153, G154", "genre": "etriers"},
   {"element": "Semelle sur pieux", "poste": "Armatures inférieures", "valeur": "6Ø25", "etat": "ok", "cellules": "G175, G176, G177, G178", "genre": "barres"},
   {"element": "Semelle sur pieux", "poste": "Armatures supérieures", "valeur": "4Ø16", "etat": "ok", "cellules": "G181, G182, G183, G184", "genre": "barres"},
   {"element": "Semelle sur pieux", "poste": "Étriers", "valeur": "4 brins Ø12 – 125 mm", "etat": "ok", "cellules": "G189, G190, G191", "genre": "etriers"},
   {"element": "Poutre support châssis", "poste": "Armatures latérales", "valeur": "2Ø12 tous les 30 à 40 cm", "etat": "info", "cellules": "", "genre": "prescription"},
   {"element": "Semelle sur pieux", "poste": "Armatures latérales", "valeur": "2Ø12 tous les 30 à 40 cm", "etat": "info", "cellules": "", "genre": "prescription"},
   {"element": "Semelle sur pieux", "poste": "Étriers de renfort au droit de chaque pieu", "valeur": "16 brins Ø12 + 5 brins des étriers de base", "etat": "ok", "cellules": "G234, G235, G236", "genre": "renfort"},
   {"element": "Plaques de répartition", "poste": "Perfos par plaque", "valeur": "3Ø25 par plaque (calcul : 1Ø25 par pied)", "etat": "ok", "cellules": "G258, G259", "genre": "perfos"}
  ],
  "manques": [
   "Armatures latérales : le classeur prescrit « 2Ø12 tous les 30 à 40 cm » sans en fixer le nombre de rangées — à poser par le dessinateur (elles partent dans le tableau, pas dans le modèle RBAR).",
   "Longueur des zones d'étriers (extrémités, partie centrale) : non fixée par le calcul.",
   "Formes, retours, ancrages et recouvrements des barres : non fixés par le calcul (standard du bureau).",
   "Étriers de renfort au droit des pieux et perfos : hors du modèle de poutre de RBAR — à poser d'après le tableau des armatures."
  ],
  "bloquants": [],
  "pret": true,
  "a_regarder": [],
  "ecarts": ["E3 — Pieu vérifié sur l'effort tranchant, pas sur sa réaction", "E7 — Bras de levier des étriers", "E7 — Bras de levier des étriers", "E9 — Hauteur calculée sans le moment horizontal"]
 }
}
```

## 6. D'où vient chaque barre du plan

Pour chaque repère du bordereau de référence : le paramètre du fichier qui le commande, les cellules du classeur de
calcul, puis ce que le calcul fixe et ce qu'il ne fixe pas.

| Rep. | Sur le plan | Pièce de la note | Paramètre du fichier (`armatures`) | Cellules | Le calcul fixe | Le calcul ne fixe pas |
|---|---|---|---|---|---|---|
| 1 | U Ø12 – 12,5 ; 4 × 4 × 23 = 368 ; 65 / 60 / 65 | semelle | `semelle` · `etriers` : `brins` 4, `d` 12, `pas` 125 | G189:G191 | 4 brins, Ø12, pas 125 | la forme (des U assemblés par deux, d'après la miniature), le nombre 23, les cotes |
| 2 | Ø16 ; 4 × 4 = 16 ; 60 / 262 / 60 | semelle | `semelle` · `sup` : `n` 4, `d` 16 | G181:G184 | 4Ø16 | les retours, la longueur |
| 3 | Ø25 ; 4 × 6 = 24 ; 60 / 262 / 60 | semelle | `semelle` · `inf` : `n` 6, `d` 25 | G175:G178 | 6Ø25 | les retours, la longueur |
| 4 | L Ø12 ; 4 × 4 × 2 = 32 ; 80 / 160 | semelle | `semelle` · `laterales` : `d` 12, `texte` | — | le Ø et la règle « 2Ø12 tous les 30 à 40 cm » | le nombre de rangées, la forme, le recouvrement |
| 5 | U Ø12 ; 2 × 4 × 2 × 10 = 160 ; 65 / 70 / 65 | semelle, au droit de chaque pieu | `semelle` · `renfort_pieu` : `brins` 16, `d` 12, `brins_base` 5 | G234:G236 | 16 brins Ø12 en plus de 5 brins de base | la forme et le nombre de barres — voir ci-dessous |
| 6 | étriers Ø12 – 10 ; 8 × 21 = 168 ; 62 × 55 | poutre support | `support` · `etriers_extremites` : `brins` 6, `d` 12, `pas` 100 | G145:G147 | 6 brins, Ø12, pas 100 | la longueur de la zone (200), le découpage en cadres |
| 7 | étriers Ø12 – 10 ; 4 × 21 = 84 ; 62 × 70 | poutre support | le même poste que le repère 6 | G145:G147 | (les 6 brins = deux cadres du rep. 6 et un du rep. 7) | |
| 8 | étriers Ø12 – 30 ; 4 × 12 = 48 ; 62 × 95 | poutre support | `support` · `etriers_centre` : `brins` 4, `d` 12, `pas` 300 | G152:G154 | 4 brins, Ø12, pas 300 | la longueur de la zone (350), le découpage en cadres |
| 9 | Ø16 ; 2 × 6 = 12 ; 50 / 742 / 50 | poutre support | `support` · `sup` : `n` 6, `d` 16 | G136:G139 | 6Ø16 | les retours, la longueur |
| 10 | Ø20 ; 2 × 9 = 18 ; 50 / 742 / 50 | poutre support | `support` · `inf` : `n` 9, `d` 20 | G129:G132 | 9Ø20 | les retours, la longueur |
| 11 | L Ø12 ; 2 × 4 × 2 = 16 ; 50 / 410 | poutre support | `support` · `laterales` : `d` 12, `texte` | — | le Ø et la règle « 2Ø12 tous les 30 à 40 cm » | le nombre de rangées, la forme, le recouvrement |
| 12 | Ø25 droit ; 4 × 3 = 12 ; L = 60 | plaques d'appui de la grue | `perfos` · `perfos` : `n` 3, `d` 25 | G258:G259 | 3Ø25 par plaque | la longueur, le mode d'ancrage |

Les cellules sont celles de la « Fiche de calcul » du classeur ; le fichier les donne dans `controle.lignes[].cellules`
et dans `modelesPoutre[].stab.saisies`.

**Le repère 5 ne se compare pas tel quel.** Le classeur compte des **brins** (16 brins Ø12 ajoutés aux 5 brins des
étriers de base) ; le plan pose **2 × 10 U** par pieu. Je ne sais pas comment le dessinateur est passé de l'un à
l'autre : c'est une question à poser au propriétaire, pas une règle à deviner.

**Le plan de référence et le classeur d'origine ne disent pas la même chose sur quatre postes.** L'exemple du classeur
Greenwood porte 8Ø20 et 4Ø16 dans la poutre support et 8Ø20 en bas de la semelle ; le plan porte 9Ø20, 6Ø16 et 6Ø25 ;
le quatrième poste est le repère 5. Le dessinateur a mis plus d'acier que le calcul n'en demandait. Règle que je
propose, **pas encore confirmée par le propriétaire** : RBAR dessine ce que dit le fichier, donc ce que la note a
vérifié ; si le dessin doit changer, l'ingénieur change le choix dans la note, qui revérifie, et réexporte.

## 7. Ce que le calcul ne fixe pas

Le fichier ne le porte donc pas. Pour chaque point, soit RBAR applique le standard du bureau, soit tu me demandes un
paramètre.

- Les **formes**, les retours, les crochets, les ancrages et les recouvrements des barres.
- La **longueur des barres**. Sur le plan : 742 = 750 − 2 × 4 d'enrobage ; 262 = 270 − 2 × 4 ; hauteur des étriers
  62 = 70 − 2 × 4.
- La **longueur des zones d'étriers** (200 / 350 / 200 sur le plan). Le calcul donne un pas « aux extrémités » et un
  pas « en partie centrale », sans dire où l'on passe de l'un à l'autre.
- Le **découpage des brins en cadres** : 6 brins = deux cadres de 55 et un de 70 ; 4 brins = deux cadres de 95 ;
  dans la semelle, des U en deux moitiés.
- Le **nombre de rangées de barres latérales** : le classeur prescrit « 2Ø12 tous les 30 à 40 cm », rien de plus.
- Les **niveaux** : le calcul exige seulement que la semelle soit au moins 5 cm plus haute que la poutre support,
  pour que les deux cages se croisent.
- Les **armatures des pieux** (en gris sur le plan) et l'ordre de pose.
- Tout **texte de chantier** : le produit d'ancrage des perfos, la remarque sur le relevé. RBAR ne doit pas les
  porter ; ils viendront du fichier ou du dessinateur.

## 8. Ce que j'attends de toi en premier : l'entrée que RBAR veut lire

C'est toi qui lis : c'est toi qui définis l'entrée. Elle doit être **neutre** (§ 3) et générale : des éléments
(poutre, massif, pieu, paroi), leurs dimensions, leur place, leurs barres et leurs étriers, puis les vues à composer
et les textes à écrire. Le plus simple pour moi est une forme proche de ce que RBAR sait déjà enregistrer.

Ce que le site sait et peut y écrire :

- toute la géométrie en plan (contours, axes, pieux, paroi), les sections, les longueurs, les nombres de pièces ;
- toutes les armatures choisies, avec leur origine (§ 5.3) ;
- les noms des pièces, dans le vocabulaire de la note ou dans celui du plan (« Poutres transversales ») ;
- la liste des vues à produire, avec leurs titres et leurs échelles, et les textes des blocs (grue, matériaux,
  remarques) : s'ils sont dans le fichier, RBAR n'a pas à les connaître ;
- les réactions et les charges, si un bloc du plan doit les montrer.

Renvoie-moi, par le propriétaire, le format que tu veux (noms, types, unités, un exemple). Je produis alors le fichier
Greenwood à ce format, et je le fais vérifier par les tests du site.

## 9. Le résultat attendu

Ce que le propriétaire regardera, sur Greenwood :

1. À partir du seul fichier, RBAR produit la feuille : les huit vues du § 4, le bordereau, la légende, les blocs de
   texte, le cartouche ; aux mêmes échelles (1/25, 1/5, 1/50).
2. Chaque groupe de barres est étiqueté comme sur le PDF : quantité, Ø, pas, repère, miniature cotée.
3. Les quantités et les Ø sont ceux du fichier. Là où le calcul ne fixe rien (§ 7), RBAR applique le standard du
   bureau, et on peut retrouver quelle règle a donné quelle valeur.
4. **Chaque barre dit d'où elle vient** dans la note. L'endroit est à toi ; la colonne « Remarques » du bordereau de
   référence est vide, c'est une place possible. Le propriétaire a dit : « tout doit être indiqué ».
5. Rien de propre aux grues dans le produit livré (§ 3).
6. Qualité au minimum égale à celle du PDF. Tu peux améliorer ; dis ce que tu améliores et pourquoi.

Le propriétaire estime que le style de RBAR et celui du PDF sont déjà presque les mêmes : l'écart à combler est
surtout dans ce qui est indiqué, et dans le fait que tout se construise depuis le fichier.

## 10. Ce que j'ai vérifié dans RBAR, et ce que je n'ai pas vérifié

Lu dans ton dépôt en lecture seule, au commit `10e2245` (version 0.4.25, 06/11/2026). Tu es mieux placé que moi :
corrige ce qui a bougé.

- « Importer JSON… » de la palette lit `modelesPoutre` (aussi `modelesColonne`, `modelesDalle`) : ce sont des
  **modèles d'armatures**, pas des éléments.
- `RBARPIEU › Objets` change des cercles en pieux du type courant ; `types.json` porte les types de pieux et les
  gabarits de massifs.
- **Aucune commande ne construit un élément depuis un fichier.** `docs/product/BILAN-0810.md` range « Du calcul au
  dessin » parmi les travaux à venir.

Non vérifié : je n'ai fait tourner ni RBAR ni AutoCAD. Le fichier du site est contrôlé par les tests du site
(mêmes valeurs que le calcul, format relu sur ton code), jamais par un import réel.
