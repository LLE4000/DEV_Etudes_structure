# Site « Stab Méthodes » — cahier des charges

Ce que le bureau a demandé pour le site (stab-methodes.be), et les réponses retenues. Les maquettes qui le
montrent sont dans `site/maquettes/` (voir son README). Mis à jour le 06/11/2026 (relecture des maquettes :
navigation, paramètres, résultat sur deux lignes, fenêtre de calcul, rapports multi-modules, poutres béton armé).

## 1 · Principes

- Le site remplace Streamlit comme écran principal ; **Streamlit reste en secours**, sur le même moteur.
- **Le calcul est celui du moteur Python du dépôt**, exécuté dans le navigateur : les mêmes chiffres, les mêmes
  formules, la même note PDF que Streamlit.
- **Gratuit** : le site sur Cloudflare Pages, les données sur Supabase, le domaine chez OVH.
- **Le PC d'abord.** Le téléphone n'est pas prioritaire : il doit fonctionner, sans plus d'effort.
- Premier module : le garde-corps (il fixe le style) ; ensuite le menu Béton, en commençant par les poutres
  (maquette faite : `site/maquettes/pages/poutre.html`). **Avant le site**, le calcul des poutres doit sortir de
  l'écran Streamlit (`modules/poutre.py` mêle calcul et écran) pour devenir un moteur aux formules nommées et
  testées, comme celui du garde-corps : le site ne fait que l'appeler.
- **Homogénéité** : tous les modules sont bâtis avec les mêmes composants (cadres, lignes de résultat, fenêtre de
  calcul, sélecteur) ; un module ne réinvente pas un geste qu'un autre a déjà.

## 1 bis · Navigation

- Une barre de **cinq icônes** : Accueil, Projets, Modules, Équipe, Paramètres ; chacune nommée par une bulle au
  survol. Rien d'autre dans la barre.
- **Deux navigations qui ne se mélangent pas** : « Projets » mène aux chantiers et à leurs dossiers ; « Modules »
  ouvre le volet des modules, classés par famille (Béton, Acier, Autres).
- Le **fil d'Ariane** (chantier › dossier › note) est cliquable.
- Au téléphone : les **cinq mêmes entrées**, en bas de l'écran, dans le même ordre.
- **Nouvelle note** : on choisit le chantier, le dossier, le nom, puis le module ; l'établi du module s'ouvre.

## 2 · Accès

- Le site est **privé**. Seules les **adresses e-mail invitées** par un administrateur y entrent.
- La personne invitée reçoit un lien et **choisit son mot de passe**. Rien d'autre à faire.
- Un administrateur peut retirer une personne ; elle n'entre plus, ses notes restent au bureau.

## 3 · Rôles et droits

| | Administrateur | Membre |
|---|---|---|
| Lire les notes des chantiers | oui | oui |
| Modifier une note | **toutes** | les siennes, et celles qu'on lui partage |
| Inviter, retirer, nommer administrateur | oui | non |
| Valider un module | oui | non |

- **Plusieurs administrateurs** (par exemple le chef de service et l'auteur du site).
- **Par défaut, seul l'auteur d'une note la modifie** (et les administrateurs).
- **Partager** : l'auteur d'une note ou d'un dossier peut autoriser un collègue à la modifier.
- **Dossier personnel** : visible par son propriétaire et les administrateurs ; il l'ouvre à qui il veut, en lecture
  ou en modification.

## 4 · Validation des modules

- Un administrateur pose une **petite pastille « validé »** sur un module dont il a relu le principe (avec son nom
  et la date). Elle se voit sur l'accueil et dans le module ; « à valider » sinon.
- La pastille porte la **version du module** : une nouvelle version du module repasse « à valider ».
- La pastille dit que la **méthode** est validée ; chaque note reste à relire par son auteur.

## 5 · Projets, notes et indices

- Chantier › dossiers › rapports. Chaque rapport garde ses **indices A, B, C…** avec leur historique.
- Statut d'un indice : **Brouillon**, **Diffusé**, **Old** (remplacé).
- Un indice **Old s'imprime grisé, avec le filigrane OLD** (tampon en travers de chaque page).
- Une note peut contenir **plusieurs éléments** (plusieurs garde-corps, plusieurs poutres…) : la liste est en tête
  des données, on passe de l'un à l'autre en un clic, on en ajoute un en un clic.
- Export et import d'un rapport en fichier **`.stab`**.

## 5 bis · Module, élément, rapport

- Un **module** est un type de calcul (Poutres béton armé, Garde-corps…). Un **élément** est une pièce calculée
  (la poutre V1, le garde-corps G1). Un **rapport** (la note) rassemble des éléments, **de plusieurs modules** :
  5 poutres, 3 dalles et 2 prédalles dans le même rapport.
- Le rapport imprimé a **une seule page de garde**, un **sommaire**, une **pagination continue** et un seul gabarit
  **A4 paysage** ; ses indices (A, B, C…) restent ceux du rapport.

## 5 ter · Paramètres

- La page **Paramètres** porte les **valeurs du bureau** (unités, format des notes, indice old, impression), les
  **matériaux par défaut** (béton, armatures, acier des profilés, E) et les **valeurs de chaque module** (poutres :
  γ_s, enrobage, jeux, armatures de montage ; garde-corps : méthode, σ_adm = f_y / 1,5, flèche, γ_M0, γ_M2).
- **Seuls les administrateurs** changent les valeurs du bureau.
- Une note peut changer une valeur **pour elle seule** : le champ porte « modifié » et la **valeur du bureau reste
  écrite dessous** ; les valeurs du bureau ne changent pas. Une valeur venue du bureau ne porte pas d'étiquette.
- Les valeurs s'affichent **telles quelles** : γ_M2 = 1,25 (jamais arrondi comme un taux).

## 6 · Impression

- Le bouton **Imprimer** ouvre la note prête à imprimer ; le navigateur **retient la dernière imprimante** utilisée.
- Un site ne peut pas choisir une imprimante tout seul (sécurité des navigateurs). Pour imprimer **sans aucune
  fenêtre**, un raccourci Chrome avec l'option `--kiosk-printing` envoie directement à l'imprimante par défaut de
  Windows — un réglage par PC, facultatif.

## 7 · Ne jamais perdre une note

- **Enregistrement automatique** à chaque modification (« Enregistré il y a 2 min »).
- **Historique** : chaque version enregistrée peut être rouverte ; on revient en arrière en un clic.
- **Corbeille** de 30 jours pour une note supprimée.
- **Sauvegarde quotidienne** de toute la base, et export `.stab` à tout moment.
- Si un calcul plante, la note garde ses **données** : on la rouvre, on corrige, on relance.

## 8 · L'écran d'un module

- **La toile du dessin occupe toute la largeur.** Les données (à gauche) et le résultat (à droite) flottent dessus,
  dans deux cadres ; on les **replie** pour donner toute la place au dessin.
- **Résultat** : une vérification par ligne, **sur deux lignes** — le nom et la pièce (profilé, armatures), puis la
  jauge et le taux ; les lignes se groupent par élément ou par section ; la ligne qui gouverne est marquée.
- **La ligne entière ouvre le calcul** : une grande fenêtre (la toile moins la colonne Résultat, environ 75 % de la
  largeur), avec en tête les hypothèses (matériaux, méthode), puis un tableau **Vérification · Formule · Valeur ·
  Limite · Taux** (valeurs alignées à droite), puis le détail (formule = valeurs = résultat). Au téléphone, elle monte
  du bas sur toute la largeur, sans la colonne Formule.
- **La pièce ouvre le sélecteur** (profilés, dispositions d'armatures) : trié par masse ou par section d'acier ; le
  retenu (✓), l'optimal (★ le plus léger qui tient), quatre au-dessus et au-dessous ; la molette fait défiler, un clic
  choisit ; au survol, les propriétés (formé à froid EN 10219 ou fini à chaud EN 10210, A, I, W). Quand un plus léger
  tient, une ★ le signale sur la ligne du résultat.
- **Statuts** : Conforme, Non conforme, **À compléter** — un élément dont les données manquent n'est jamais
  « conforme ».
- **Méthode** : ELS (méthode du bureau, σ_adm = f_y / 1,5 = 235 / 1,5 = 156,7 N/mm² en S235) ou Eurocode (γ_M0,
  γ_M2) ; c'est un choix de la note, les deux ne se mélangent pas.
- Code couleur des taux : rouge au-dessus de 100 %, vert foncé de 95 à 100 %, vert pâle en dessous. Le taux
  **affiché** garde la couleur de sa bande : 94,98 % s'affiche 94 % (vert pâle), jamais « 95 % » en vert pâle.
- La toile a un **fond uni** ; les deux cadres se replient ; le dessin passe en plein écran.

## 9 · Noms des modules

Le titre seul, sans explication : il dit ce que fait le module.

- **Béton** : Poutres béton armé · Dalles béton armé · Prédalles · Charges sur plancher · Taux d'armature ·
  Enrobage · Maturité du béton · Tableau des armatures
- **Acier** : Choix de profilé · Catalogue des profilés · Cornières · Flambement · Assemblages métalliques · Soudures
- **Autres** : Garde-corps · Poutres bois · Rigidité du sol

D'autres changements de noms suivront ; cette liste est la référence.
