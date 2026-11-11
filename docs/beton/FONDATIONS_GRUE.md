# Fondations de grues — le module

Le module « Fondations de grues » du site (famille Béton, adresse `fondations-grues`) reprend les quatre classeurs du
bureau « Fondation grue tour sur pieux ». Ce document dit ce qu'il fait, comment il est construit, ce qui prouve qu'il
calcule comme les classeurs, et ce qu'il ne couvre pas. L'étude qui l'a précédé — la lecture des quatre classeurs,
feuille par feuille, les écarts relevés et la correspondance cellule par cellule — est dans
[`FONDATIONS_GRUE_ETUDE.md`](FONDATIONS_GRUE_ETUDE.md).

## 1. La règle : le classeur, à la lettre

Les classeurs ont été validés par le bureau. Le module n'en change rien : mêmes formules, mêmes hypothèses, mêmes
unités, mêmes arrondis, mêmes résultats intermédiaires. Rien n'est repris du module « Poutres béton armé » du site,
parce que ses formules ne sont pas celles des classeurs (étude § 7).

Trois conséquences :

- **le verdict est celui du classeur** : chaque vérification porte l'état que le classeur lui donne (OK, LIMITE,
  PAS OK, A VERIFIER), et la couleur de l'écran le suit ;
- **un écart n'est jamais corrigé en silence** : là où la lecture a trouvé une formule discutable, le module calcule
  comme le classeur et le SIGNALE, avec les deux chiffres (bloc « Écarts relevés dans le classeur ») ;
- **chaque ligne de calcul dit d'où elle vient** : la cellule du classeur dont elle reproduit la valeur est écrite au
  bout de la ligne.

## 2. Les trois schémas

| Schéma | Classeur | Pièces calculées |
|---|---|---|
| 8 pieux — 2 poutres support sur 4 semelles de 2 pieux (ou, du côté b, une paroi de pieux sécants) | BASE (et Greenwood, mêmes formules) | poutre support châssis, semelle sur pieux, pieux, renfort au droit du pieu, effort horizontal |
| 4 pieux — 4 poutres en cadre | variante 4 pieux | poutre, pieux, renfort, effort horizontal |
| 8 pieux — poutres primaires à 45° sur 2 pieux et poutres secondaires | variante 8 pieux | poutre secondaire, poutre primaire, renfort, effort horizontal |

« Reprendre un classeur » (rubrique 1) charge les valeurs d'un des quatre classeurs : c'est le moyen de comparer le
module au classeur, ligne par ligne.

## 3. Les tolérances d'exécution : une poutre sur deux appuis

Un pieu peut être exécuté à côté de sa place. Les classeurs en tiennent compte par de grands tableaux ; le module
montre la même chose comme on la dessinerait à la main (`beton/fondation_grue/statique.py`) :

- chaque poutre et chaque semelle est **une poutre sur deux appuis** ;
- pour chaque **position de tolérance** (le pieu décalé de e d'un côté, de l'autre ; pour la poutre support : les
  pieux de la semelle A décalés, ceux de l'appui B, le châssis posé décalé de + e, de − e), les appuis et les charges
  sont placés là où cette position les met ;
- chaque position donne sa courbe de moment et sa courbe d'effort tranchant ; l'**enveloppe** est tracée en trait
  fort, ses extremums sont cotés — valeur et abscisse EXACTES (le sommet de M est là où l'effort tranchant s'annule ;
  le classeur, lui, lit M tous les 5 cm : écart E8) ;
- la valeur que le classeur retient est repérée par un tireté : quand elle diffère de l'extremum exact, on le voit
  (c'est le cas de la variante 4 pieux : 72,6 kN·m au classeur, 76,3 kN·m sur la courbe — écart E16).

Les abscisses sont comptées depuis l'appui A à sa place prévue ; pour une pièce posée sur deux pieux (la semelle, la
poutre primaire), depuis son milieu — la légende de chaque dessin le dit.

Cette statique est écrite indépendamment de la transcription des classeurs ; les tests vérifient que les deux donnent
les mêmes réactions, les mêmes moments et les mêmes efforts tranchants.

## 4. L'écran

Il a la disposition des Poutres béton armé (l'établi compact).

- **À gauche**, six rubriques courtes : 1 Fondation (le schéma, le côté b, le chantier, la grue, « reprendre un
  classeur »), 2 Matériaux, 3 Géométrie, 4 Tolérances d'exécution, 5 Charges de la grue (pour les 8 pieux sur
  semelles : un cas par carte, la réaction de chaque pied A, B, C, D), 6 Armatures choisies.
- **Au centre**, la vue en plan, redessinée à chaque saisie (pieux, semelles, poutres, pieds de la grue, châssis,
  cotes), puis une vue « M et V » par poutre et par semelle.
- **À droite**, le verdict, un bloc par partie du classeur, les contrôles de géométrie, les écarts, puis l'export.
  Chaque ligne ouvre sa fenêtre de calcul.

Une saisie hors du domaine du classeur est refusée avec un message, et la fondation reste comme elle était.

## 5. L'export vers RBAR (AutoCAD)

RBAR dessine, il ne dimensionne pas : il attend des armatures toutes choisies. Le module lui donne celles de la note,
sans rien recalculer (`beton/fondation_grue/export.py`).

| Fichier | Contenu | Usage |
|---|---|---|
| RBAR `.json` | les **modèles de poutre** au format que RBAR importe déjà (palette › modèles › « Importer JSON… ») : un modèle par pièce et par zone d'étriers — barres basses et hautes, second groupe en 2ᵉ lit, étriers (Ø, pas, brins), enrobage quand le classeur en a un ; le type de pieu et de massif ; l'implantation en mm ; le tableau des armatures | le dessinateur importe les modèles, puis dessine |
| Plan `.dxf` | le plan de principe (DXF R12, en cm) : contours, axes, pieux en cercles au diamètre réel, pieds de la grue | s'ouvre dans AutoCAD ; `RBARPIEU › Objets` reprend les cercles |
| Armatures `.csv` | une ligne par poste d'armature, avec la section requise, la section choisie et les cellules de saisie | contrôle, bordereau |
| Cas `.csv` | toutes les colonnes des grands tableaux (4 positions × 6 cas × 8 orientations) | comparer aux feuilles du classeur |

**Avant d'envoyer**, « Contrôle » montre ce qui part, poste par poste, ce que le calcul ne fixe pas et ce que RBAR ne
reprend pas d'un fichier. **L'export RBAR est refusé** tant qu'une vérification du classeur est « PAS OK » : on ne
transmet pas au dessin une armature que la note refuse.

**Le contrat du fichier RBAR** est [`FONDATIONS_GRUE_RBAR.md`](FONDATIONS_GRUE_RBAR.md) : chaque paramètre, son sens,
son unité, et un fichier d'exemple complet (Greenwood, aux armatures de son plan d'armatures). C'est le document remis
à la conversation qui développe RBAR. Le fichier est versionné et ne change que par **ajouts** ; la version 2
(11/11/2026) ajoute ce qu'un plan écrit et que la version 1 ne portait pas — les textes de la grue (flèche, hauteur
sous crochet, châssis, niveau), l'enrobage, la plus grande charge par pied — et, pour chaque armature, **d'où elle
vient** : ses cellules du classeur (`cellules`) et le texte prêt à écrire sur le plan (`origine`).

**Ce qui reste hors de RBAR, par décision du propriétaire** : RBAR sera vendu hors du bureau, ce site reste interne.
La fondation de grue ne doit donc pas exister dans RBAR — ni commande, ni modèle, ni texte. C'est le fichier du site
qui porte tout ; RBAR recevra une porte d'entrée générale (« construire un ouvrage décrit par un fichier »), et le
vocabulaire du fichier d'aujourd'hui (`support`, `semelle`, `pieds`…) ne doit pas s'écrire dans son code. Le format
neutre que RBAR lira est à définir par sa conversation ; le site s'y conformera.

Ce que l'export ne peut pas donner, parce que le calcul ne le fixe pas : les longueurs des zones d'étriers, les
formes, les retours, les ancrages et les recouvrements (standard du bureau), le nombre de rangées d'armatures
latérales. Et ce que RBAR ne lit pas encore d'un fichier : la géométrie complète d'un élément, les étriers de renfort
au droit des pieux, les perfos — ils sont dans le tableau, à poser par le dessinateur.

## 6. La note de calcul PDF

Le bouton « Note PDF » fait la note de TOUTE la note du site (`api_web/note_fondation_grue.py`, sur le kit `ndc_pdf` des
notes Poutre et Dalle : même page de garde, mêmes polices). Elle ne calcule rien : elle imprime le résultat que l'écran
affiche (`moteur.verifier`), avec les mots et les nombres de l'écran.

| Pages | Contenu |
|---|---|
| Garde (A4 portrait) | projet, partie, date, indice ; une ligne par fondation : son schéma, son taux, ce qui gouverne, son statut, sa page |
| Calcul (A4 paysage, 2 ou 3 pages) | les hypothèses (la méthode du classeur, les matériaux et leurs contraintes) ; les données saisies, **chacune avec sa cellule du classeur** ; les charges de la grue ; un bloc par partie du classeur — ses sollicitations, chaque vérification avec ses formules « symbole = expression = valeurs = résultat » et la cellule dont chacune reproduit la valeur, son verdict (l'état que le classeur donne), le cisaillement et sa zone ; les contrôles de géométrie ; les écarts relevés (les deux chiffres) ; le récapitulatif ; ce que la note ne couvre pas ; la conclusion |
| Diagrammes M et V | pour chaque poutre et chaque semelle : le dessin de l'écran (une rangée et une courbe par position de tolérance, l'enveloppe, la valeur du classeur en tireté) et le tableau des positions (portée, réactions, moments, efforts tranchants, l'enveloppe, ce que le classeur retient) |
| Plan de principe (la dernière page) | la vue en plan à une **échelle normalisée** (1:50, 1:75…), les pièces, les armatures choisies avec leurs cellules (ce qui part vers RBAR), ce que le calcul ne fixe pas, le cartouche — ni plan de coffrage, ni plan d'armatures |

La mise en page ne tronque jamais : les blocs du calcul sont mesurés puis posés, colonne après colonne ; un titre ne
reste pas seul en bas d'une colonne. Le nombre de pages est celui de la taille normale du texte ; la taille retenue est
la plus grande qui tient dans ce nombre de pages. Les dessins sont les scènes de l'écran, peintes telles quelles (un
seul dessin pour l'écran et pour la note).

## 7. Ce qui prouve la fidélité

| Suite | Ce qu'elle tient |
|---|---|
| `tests/test_fondation_grue_moteur.py` | les 541 cellules calculées des quatre fiches et environ 7 800 valeurs des grands tableaux, reproduites à 10⁻⁹ près ; la statique indépendante ; les formules imprimées — leur gabarit réévalué, et leur ÉCRITURE : la chaîne imprimée de 1 236 formules (les quatre classeurs et 24 variantes) relue caractère par caractère ; l'enveloppe exacte de la poutre support ; 39 + 8 mutations vues |
| `tests/test_api_web_fondation_grue.py` | ce que le site affiche, dessine et exporte = ce que le moteur calcule, sur 14 états ; aucun texte d'un dessin hors de son cadre ni sur un autre ; le fichier RBAR en version 2 ; les refus ; l'échappement ; 57 mutations vues |
| `tests/test_note_fondation_grue.py` | la note PDF relue par pdftotext sur 13 états et une grille de 40 fondations : tout ce que le moteur calcule y est, à la lettre (formules, cellules, verdicts, écarts, positions de tolérance, armatures), les dessins de l'écran, l'échelle du plan, la mise en page sans débordement, l'identification, les conditions du navigateur ; 21 mutations vues |
| `tests/test_fondation_grue_contrat_rbar.py` | le document du fichier RBAR et le fichier disent la même chose : l'exemple du document est celui que le site produit, chaque paramètre exporté y est nommé, le tableau des 12 repères du plan Greenwood dit les cellules et les armatures du fichier, rien de la version 1 n'a disparu ; 14 mutations vues |
| `tests/test_api_web_modules.py fondation_grue` | le contrat commun des modules du site |
| `site/web/verifier_site.js` (`module:fondation_grue`, `formats:fondation_grue`, `note_pdf`) | l'écran dans le navigateur : PC, tablettes, téléphone ; la note PDF téléchargée d'un clic |

Les valeurs de référence (`tests/ref_fondation_grue.json`) sont tirées des classeurs par
`tools/transcrire_fondation_grue.py`.

**L'écriture des formules** (`beton/fondation_grue/ecriture.py`) est la même dans la fenêtre de calcul et dans la note :
« symbole = expression = valeurs = résultat », les valeurs à cinq chiffres significatifs (un calcul refait à la main avec
les nombres imprimés retombe sur le résultat imprimé), la cellule du classeur au bout de la ligne. Le 11/11/2026, la
relecture de l'écran a trouvé trois écritures fausses sous des résultats justes : « × 10³ » imprimé « × 100 » quand la
part de V intervenait, « H_pied » écrit « H_πed », un tiret pour le signe moins de « −P·e ». La cause était dans la
brique partagée qui écrit les substitutions (`acier/assemblages/poutre_poutre/doubles_cornieres/formules.py`) ; elle
est corrigée là, et le test relit désormais la chaîne IMPRIMÉE — l'ancien contrôle ne réévaluait que le gabarit.

La relecture des pages de la note, calculs refaits à la main, a encore trouvé quatre écritures à reprendre (aucune ne
changeait un résultat) : au droit du pieu, la largeur de la hauteur calculée était écrite « b » alors que le classeur y
met le diamètre du pieu (elle s'écrit D) ; le moment maximal de la poutre support était coté à un point de grille
(« x = 5,21 m » pour un sommet en 5,18 m) ; le tranchant du classeur portait le signe du côté où il est tracé
(« V max = −820,8 ») ; la dernière graduation pouvait toucher le titre de l'axe (« 6x (m) »). Chacune a son contrôle
et sa mutation.

## 8. Ce qui n'est pas couvert

- **Toute géométrie hors des trois schémas** : pieux inclinés, nombre de pieux différent, châssis rectangulaire,
  semelle sur trois pieux. Le module refuse plutôt que d'extrapoler.
- **Ce que les classeurs ne calculent pas** : le soulèvement d'un pieu, le poinçonnement, les longueurs d'ancrage, la
  fissuration, la répartition de l'effort horizontal entre les pieux (`moteur.NON_COUVERT` : la note l'imprime).
- **Les seize écarts de l'étude** (§ 5) : signalés, non corrigés. Six questions restent posées au bureau (étude § 11).
