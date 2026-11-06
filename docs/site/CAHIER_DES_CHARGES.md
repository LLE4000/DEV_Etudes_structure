# Site « Stab Méthodes » — cahier des charges

Ce que le bureau a demandé pour le site (stab-methodes.be), et les réponses retenues. Les maquettes qui le
montrent sont dans `site/maquettes/` (voir son README). Mis à jour le 06/11/2026.

## 1 · Principes

- Le site remplace Streamlit comme écran principal ; **Streamlit reste en secours**, sur le même moteur.
- **Le calcul est celui du moteur Python du dépôt**, exécuté dans le navigateur : les mêmes chiffres, les mêmes
  formules, la même note PDF que Streamlit.
- **Gratuit** : le site sur Cloudflare Pages, les données sur Supabase, le domaine chez OVH.
- **Le PC d'abord.** Le téléphone n'est pas prioritaire : il doit fonctionner, sans plus d'effort.
- Premier module : le garde-corps (il fixe le style) ; ensuite le menu Béton, en commençant par les poutres.

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
- **Résultat simplifié** : une ligne par élément — nom, profilé, taux, jauge. Le détail (critère, cas, l'optimal)
  s'ouvre au clic ; il est ouvert d'emblée pour l'élément qui gouverne.
- **Sélecteur de profilés** : trié par masse (l'ordre de grandeur) ; on voit le profilé retenu (✓), l'optimal
  (★ le plus léger qui tient), quatre profilés au-dessus et au-dessous ; la molette fait défiler, un clic choisit.
- **Voir le calcul** : les formules de la note (symbole = formule = valeurs = résultat), élément par élément.
- Code couleur des taux : rouge au-dessus de 100 %, vert foncé de 95 à 100 %, vert pâle en dessous.

## 9 · Noms des modules

Le titre seul, sans explication : il dit ce que fait le module.

- **Béton** : Poutres béton armé · Dalles béton armé · Prédalles · Charges sur plancher · Taux d'armature ·
  Enrobage · Maturité du béton · Tableau des armatures
- **Acier** : Choix de profilé · Catalogue des profilés · Cornières · Flambement · Assemblages métalliques · Soudures
- **Autres** : Garde-corps · Poutres bois · Rigidité du sol

D'autres changements de noms suivront ; cette liste est la référence.
