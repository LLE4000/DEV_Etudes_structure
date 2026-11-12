# Fondations de grues — le module

Le module « Fondations de grues » du site (famille Béton, adresse `fondations-grues`) reprend les quatre classeurs du
bureau « Fondation grue tour sur pieux ». Ce document dit ce qu'il fait, comment il est construit, ce qui prouve qu'il
calcule comme les classeurs, ce qu'il ne couvre pas, et les améliorations à l'étude (§ 9). L'étude qui l'a précédé — la lecture des quatre classeurs,
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
  semelles : un cas par carte, la réaction de chaque pied A, B, C, D), 6 Armatures choisies ; et, pour les 8 pieux sur
  semelles, une septième, secondaire : 7 **Options de conception** — ce que le plan d'armatures demande et que le
  calcul ne fixe pas (retours, zones d'étriers serrés, latérales, coupes, cages, perfos, niveaux, textes, mise en
  page). Elles ne changent aucun résultat ; valeurs de Greenwood par défaut ([`FONDATIONS_GRUE_OUVRAGE.md`](FONDATIONS_GRUE_OUVRAGE.md)).
- **Au centre**, la vue en plan, redessinée à chaque saisie (pieux, semelles, poutres, pieds de la grue, châssis,
  cotes), puis une vue « M et V » par poutre et par semelle.
- **À droite**, le verdict, un bloc par partie du classeur, les contrôles de géométrie, les écarts, puis le bouton
  **« Exporter »** (§ 5). Chaque ligne ouvre sa fenêtre de calcul.

Une saisie hors du domaine du classeur est refusée avec un message, et la fondation reste comme elle était.

## 5. L'export vers RBAR (AutoCAD)

RBAR dessine, il ne dimensionne pas : il attend des armatures toutes choisies. Le module lui donne celles de la note,
sans rien recalculer (`beton/fondation_grue/export.py`, `beton/fondation_grue/ouvrage.py`).

**Un seul bouton, « Exporter »** (12/11/2026, demande de la conversation RBAR) : son menu commence par « Contrôle avant
export », puis une ligne par fichier, son titre et en clair ce qu'il produit ; une ligne qui ne peut pas partir est
éteinte et dit pourquoi. Rien n'a disparu des exports d'avant.

| Fichier | Contenu | Usage |
|---|---|---|
| **DWG pour AutoCAD** (`.json`, format « rbar.ouvrage ») | l'ouvrage complet au format que RBAR lit : pieux, poutres, armatures de la note, d'où vient chaque barre, options de conception, mise en page des deux feuilles, cartouche — [`FONDATIONS_GRUE_OUVRAGE.md`](FONDATIONS_GRUE_OUVRAGE.md) | dans AutoCAD : commande `RBAROUVRAGE`, puis choisir ce fichier (étape 1 ; le DWG fait sans AutoCAD et les plans PDF viendront avec le moteur de RBAR) — 8 pieux sur semelles seulement |
| Modèles de poutre pour RBAR (`.json`) | les **modèles de poutre** au format que RBAR importe déjà (palette › modèles › « Importer JSON… ») : un modèle par pièce et par zone d'étriers — barres basses et hautes, second groupe en 2ᵉ lit, étriers (Ø, pas, brins), enrobage quand le classeur en a un ; le type de pieu et de massif ; l'implantation en mm ; le tableau des armatures | le dessinateur importe les modèles, puis dessine |
| Plan `.dxf` | le plan de principe (DXF R12, en cm) : contours, axes, pieux en cercles au diamètre réel, pieds de la grue | s'ouvre dans AutoCAD ; `RBARPIEU › Objets` reprend les cercles |
| Armatures `.csv` | une ligne par poste d'armature, avec la section requise, la section choisie et les cellules de saisie | contrôle, bordereau |
| Cas `.csv` | toutes les colonnes des grands tableaux (4 positions × 6 cas × 8 orientations) | comparer aux feuilles du classeur |

**Avant d'envoyer**, « Contrôle avant export » montre ce qui part — chaque valeur et d'où elle vient —, ce qui est à
regarder, ce qui manque et ce qui bloque. **Les fichiers pour RBAR sont refusés** tant qu'une vérification du classeur
est « PAS OK » : on ne transmet pas au dessin une armature que la note refuse.

**Le contrat du fichier RBAR** est [`FONDATIONS_GRUE_RBAR.md`](FONDATIONS_GRUE_RBAR.md) : chaque paramètre, son sens,
son unité, et un fichier d'exemple complet (Greenwood, aux armatures de son plan d'armatures). C'est le document remis
à la conversation qui développe RBAR. Le fichier est versionné et ne change que par **ajouts** ; la version 2
(11/11/2026) ajoute ce qu'un plan écrit et que la version 1 ne portait pas — les textes de la grue (flèche, hauteur
sous crochet, châssis, niveau), l'enrobage, la plus grande charge par pied — et, pour chaque armature, **d'où elle
vient** : ses cellules du classeur (`cellules`) et le texte prêt à écrire sur le plan (`origine`).

**Ce qui reste hors de RBAR, par décision du propriétaire** : RBAR sera vendu hors du bureau, ce site reste interne.
La fondation de grue ne doit donc pas exister dans RBAR — ni commande, ni modèle, ni texte. C'est le fichier du site
qui porte tout ; RBAR recevra une porte d'entrée générale (« construire un ouvrage décrit par un fichier »), et le
vocabulaire du fichier d'aujourd'hui (`support`, `semelle`, `pieds`…) ne doit pas s'écrire dans son code. Ce format
neutre existe depuis le 11/11/2026 : « rbar.ouvrage », défini par la conversation RBAR ; le site l'écrit depuis le
12/11/2026 (ligne « DWG pour AutoCAD »).

Ce que le calcul ne fixe pas — les longueurs des zones d'étriers, les retours, le nombre de rangées de barres
latérales, les cages de renfort dessinées, les longueurs des perfos, les niveaux — vient des **options de
conception** de la note (rubrique 7), aux valeurs de Greenwood tant que le bureau n'a pas donné les siennes : la liste
des questions est dans [`FONDATIONS_GRUE_OUVRAGE.md`](FONDATIONS_GRUE_OUVRAGE.md) § 5.

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
| `tests/test_fondation_grue_ouvrage.py` | le fichier « rbar.ouvrage » : pour Greenwood, c'est l'exemple de RBAR à ses écarts voulus près ; il tient les règles du format de RBAR (recopiées en contrôles, chaque contre-exemple vu) ; chaque option de conception change le fichier là où elle le dit et RIEN du calcul ; le contrôle avant export et les blocages ; des mutations du fichier et des options, toutes vues. Le passage dans le vrai lecteur de RBAR est une mesure : `docs/site/mesures/2026-11-12_fichier_rbar_lu_par_rbar.md` |
| `tests/test_fondation_grue_contrat_rbar.py` | le document du fichier RBAR et le fichier disent la même chose : l'exemple du document est celui que le site produit, chaque paramètre exporté y est nommé, le tableau des 12 repères du plan Greenwood dit les cellules et les armatures du fichier, rien de la version 1 n'a disparu ; 14 mutations vues |
| `tests/test_api_web_modules.py fondation_grue` | le contrat commun des modules du site |
| `site/web/verifier_site.js` (`module:fondation_grue`, `formats:fondation_grue`, `note_pdf`, `exporter`) | l'écran dans le navigateur : PC, tablettes, téléphone ; la note PDF téléchargée d'un clic ; le menu « Exporter » ouvert, mesuré tel que l'œil le voit, et le fichier « rbar.ouvrage » téléchargé |

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
- **Le fichier « rbar.ouvrage »** : seulement le schéma 8 pieux sur semelles, des semelles des deux côtés ; la charge
  horizontale des pieux, l'historique des indices et une partie du cartouche n'y sont pas ; les options de conception
  attendent les valeurs du bureau ([`FONDATIONS_GRUE_OUVRAGE.md`](FONDATIONS_GRUE_OUVRAGE.md) § 5 et § 6).

## 9. Chantiers à venir — améliorations du classeur et des exports

### 9.1 Une géométrie propre à chaque pièce : relevé du géomètre, poutres inégales

*Réflexion de l'ingénieur du 12/11/2026 — à l'étude, pas encore décidée ni commencée.*

**Le besoin.** Le classeur, et donc le module, supposent UNE géométrie, symétrique : les deux poutres support
identiques (mêmes distances a et b, même section), les quatre semelles identiques, une tolérance d'exécution e posée
partout (§ 3). Trois cas réels en sortent :

- **le relevé du géomètre** : les pieux sont faits, le géomètre envoie leurs positions réelles. On veut remettre la note
  à jour avec ces positions — deux pieux d'une petite semelle écartés de 2,00 m au lieu de 1,70 m ; les deux pieux d'une
  semelle exécutés plus loin dans le sens de la poutre support, qui allongent sa portée — en gardant une tolérance,
  parce qu'un relevé n'est pas exact ;
- **une seule poutre touchée** : deux pieux bien placés sous une poutre, deux trop loin sous l'autre ; on veut pouvoir
  changer les armatures de cette poutre seulement ;
- **la conception** : une poutre courte et une plus longue, pour passer au-dessus d'un obstacle, au lieu de deux
  grandes poutres.

**Le principe proposé.**

1. On saisit l'écart mesuré de chacun des huit pieux par rapport à sa position théorique. Le module en déduit, pièce par
   pièce, l'entredistance réelle des pieux de chaque semelle, la position réelle de la poutre sur la semelle et la
   portée réelle de chaque poutre support.
2. Les positions de tolérance d'aujourd'hui (pieux décalés de e, châssis décalé de ± e) deviennent « position mesurée ±
   tolérance résiduelle » (l'imprécision du relevé). Le châssis de la grue, posé après les pieux, garde sa propre
   tolérance. Même mécanique qu'aujourd'hui, d'autres chiffres.
3. Chaque pièce est vérifiée avec SA géométrie : les poutres 1 et 2 séparément (chacune ses armatures), chacune des
   quatre semelles, chacun des huit pieux avec sa part réelle de la charge (quand la poutre n'arrive plus au milieu de
   la semelle, ses deux pieux ne portent plus la même charge). Les charges de la grue continuent de tourner sur chaque
   poutre : chaque cas joué dans ses huit orientations (étude § 3.3).

*En attendant* (deux poutres inégales) : deux fondations dans la note, la seconde avec a ou b allongé, et le pire des
deux retenu pièce par pièce — chaque calcul fait tourner tous les cas sur SA poutre. Limites : le plan reste dessiné
symétrique, et la ligne « R_B respectif » (l'autre poutre, même cas) suppose deux poutres égales.

**Ce qu'il faut surveiller.**

- Deux poutres différentes ne fléchissent pas pareil. Le tableau du fabricant suppose des appuis rigides : une poutre
  nettement plus souple fait reporter de la charge sur l'autre par le châssis. Comparer au moins la différence de flèche
  à l'inclinaison que le fabricant admet.
- Un seul pieu décalé — et non les deux d'une semelle — fait tourner la semelle : de la torsion, que le classeur ne
  calcule pas. À signaler au minimum.
- La note dit qu'elle est faite sur relevé : la date, le géomètre, les écarts mesurés, la tolérance résiduelle, et ce qui
  change par rapport à la note d'origine. Le plan montre les positions réelles en trait plein, les théoriques en tireté.

**Ce qu'il demande.**

- L'accord du bureau : c'est une sortie du classeur validé, qui ne connaît qu'une géométrie symétrique et une
  tolérance (règle du § 1).
- Le garde-fou : avec une géométrie identique et la tolérance d'origine, le module redonne les chiffres du classeur à
  l'identique (les suites du § 7) ; les cas dissymétriques se valident par un calcul fait à la main. La statique
  indépendante (`beton/fondation_grue/statique.py`, une poutre sur deux appuis) calcule déjà n'importe quelle portée.
- Un effort moyen : le moteur, l'écran (une rubrique « Relevé du géomètre »), la note PDF et le plan.

### 9.2 Trois exports à terme, dont un Excel qui remplit un classeur type

*Décision du propriétaire du 12/11/2026 (soir) — pour plus tard ; rien n'est retiré aujourd'hui.*

**Aujourd'hui, on garde toutes les lignes du menu « Exporter »** (plan de principe .dxf, armatures .csv, cas .csv,
ancien fichier RBAR) : la conversation RBAR travaille en parallèle, rien d'utile ne doit disparaître pendant ce temps.

**À terme, trois exports** : le **DWG** (le fichier « rbar.ouvrage », à ouvrir dans AutoCAD avec RBAR), le **PDF**
(la note de calcul, et les plans quand le moteur de RBAR sera là — § 9.3), l'**Excel**.

**L'export Excel remplit un classeur TYPE**, pour faire le lien avec l'ancien fichier :

1. le bureau prépare le classeur type : chaque valeur à encoder y est vidée — elle vaut 0 — et sa case est mise sur un
   fond de couleur vive (jaune ou vert fluo) : on voit d'un coup d'œil toutes les cases que l'export remplit ;
2. l'export du site écrit, dans une copie de ce classeur, les saisies de la note dans ces cases-là, et nulle part
   ailleurs ; les formules du classeur font le reste ;
3. **les cases colorées sont le contrat** : un test lit le classeur type, relève toutes les cases de cette couleur et
   vérifie que l'export les remplit TOUTES, et AUCUNE autre (une case oubliée, ou une formule écrasée, se voit tout de
   suite) ; puis il compare les résultats du classeur rempli à ceux du site.

Ce qu'il demande : le classeur type, fourni par le bureau (l'adresse de chaque case est déjà connue du site :
`beton/fondation_grue/classeur.py`, les cellules de saisie des quatre classeurs) ; l'écriture se fait avec la
bibliothèque standard (le fichier Excel est une archive de fichiers XML), sans rien charger d'un tiers, réseau coupé.

### 9.3 L'aperçu des plans dans l'écran

*Idée du propriétaire du 12/11/2026 : voir le plan de coffrage et le plan d'armatures avant de les exporter, et les
réadapter s'ils ne vont pas.*

**La place proposée : au centre**, à côté de « Plan », deux onglets « Coffrage » et « Armatures » qui montrent les
feuilles dessinées par RBAR. On change une option de conception à gauche, l'aperçu se redessine ; le PDF se télécharge
depuis « Exporter ». La gauche reste la saisie, la droite les résultats.

**Ce qu'il attend** : le moteur de plans de RBAR qui tourne dans le navigateur (WebAssembly), et — pour l'aperçu — chaque
feuille rendue aussi en image (SVG ; son noyau sait le faire : `OuvrageSheetSvg.Render`, celui de la mesure du
12/11/2026). Demandé à la conversation RBAR le 12/11. Le moteur existe depuis le 12/11 au soir comme programme .NET
(`rbar-plans`, RBAR 0.4.37) : sur le fichier Greenwood du site, les deux plans en PDF, aucune note
(`docs/site/mesures/2026-11-12_plans_pdf_par_rbar_plans.md`) ; il ne manque que sa version pour le navigateur. Ensuite, du côté du site : ses fichiers copiés dans le site
(chacun sous 25 Mo, aucun CDN), chargés au premier aperçu dans un travailleur (l'écran ne se fige pas), la ligne
« Plans PDF », les deux onglets, et les mesures aux trois formats (taille, durée au téléphone).

### 9.4 Le retour depuis AutoCAD : réimporter le fichier de l'ouvrage

*Annoncé par la conversation RBAR le 12/11/2026 (son lot suivant) : RBAR réécrira le fichier « rbar.ouvrage » depuis le
DWG quand le dessinateur aura changé des barres, avec l'indice suivant (C après B) et la description du changement.*

**Le principe proposé** : le site relit ce fichier (le même format), compare ses armatures à celles de la note et montre
chaque différence ; sur accord, il les reprend dans la note, qui les REVÉRIFIE avec le classeur. La note reste la
source : ce qui apparaît dans RBAR est ce qui apparaît dans la note (règle du propriétaire du 11/11/2026) — une barre
changée dans AutoCAD n'est jamais acceptée sans être revérifiée.

**Ce qu'il demande** : l'historique des indices dans la note (aujourd'hui, le site ne garde que l'indice courant ;
`revisions` n'en porte qu'un), et un import qui refuse ce qu'il ne sait pas relier à la note (une poutre, un poste
inconnus), en le disant.

