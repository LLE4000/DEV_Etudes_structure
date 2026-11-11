# Projets, équipe et journal d'activité

*Écrit le 11/11/2026, avec le lot. Demande de l'ingénieur, le même jour : « dans le menu principal, il y a Projets :
c'est là qu'on détermine un projet — le nom du chantier, pas l'adresse […] ça doit être très facile de créer des
projets, et il peut y avoir des sous-projets […] Équipe : un peu comme quand les gens s'enregistrent, avec leur adresse
mail […] on a des initiales, une, deux ou trois, ça pourrait arriver qu'il y en ait 3, 4 ou 5 — moi, c'est LLE — qui
apparaissent dans un rond […] un journal des activités : le cercle LLE a rajouté une note de calcul […] sur la page
d'accueil, ça doit être très bien intégré. »*

## 1 · Ce qui est livré

| Où | Quoi |
|---|---|
| Le menu (les 23 pages qui ont la barre, les onglets du téléphone) | **Projets** et **Équipe** sont des liens (ils disaient « bientôt ») ; Paramètres attend encore les comptes. |
| **Accueil** | La pastille de la personne et « Bonjour Louis » ; la carte **Projets** (un champ qui en crée un — le nom, Entrée —, les six plus récemment actifs, « Sans projet », « Tout voir ») ; la carte **Activité** (les six dernières lignes du journal, chacune avec la pastille de son auteur ; « Qui êtes-vous ? » tant que personne ne s'est présenté) ; les notes à reprendre disent leur projet et leur auteur. |
| **Projets** (`projets`, `projets?p=<projet>&s=<sous-projet>`, `projets?sans=1`) | L'arbre des projets et de leurs sous-projets — son champ **cherche et crée** —, les notes du projet groupées par sous-projet, les faits et l'activité du projet. Renommer en écrivant par-dessus le titre ; ranger une note par un menu ; archiver ; supprimer ce qui est vide. |
| **Équipe** (`equipe`) | Ma fiche (nom, initiales, e-mail, teinte de la pastille), les membres, l'ajout d'un collègue, le journal entier (filtrable par membre). |
| « Nouvelle note » (accueil et projets : la même fenêtre) | **Projet**, **Sous-projet**, nom, module ; les projets et sous-projets connus sont proposés ; depuis un sous-projet, tout est déjà écrit. |

## 2 · Les choix

1. **Tout est enregistré dans le navigateur**, comme les notes. Le site n'a ni serveur ni comptes (ils viendront avec
   Supabase, cahier des charges § 2 et 3) : construire les écrans sur une fausse base aurait été les refaire. Ce lot
   donne donc les écrans, les gestes et le format des données ; le partage entre les postes est l'étape d'après, et
   l'écran le **dit** (carte « Où sont mes notes ? », carte « Où est enregistrée l'équipe ? »).
2. **Un projet est le nom d'un chantier, rien d'autre.** Pas d'adresse, pas de client, pas de fiche à remplir : on
   écrit le nom, Entrée. Le projet est le champ `chantier` que les notes portent depuis le premier jour, le
   sous-projet leur champ `dossier` : **toutes les notes existantes sont déjà rangées**, les fichiers `.stab` restent
   compatibles dans les deux sens, aucune migration.
3. **Le lien est le NOM**, comparé sans la casse ni les espaces de trop (`cle()`). Renommer un projet réécrit donc le
   nom dans toutes ses notes, sans changer leur date ; deux projets ne peuvent pas porter le même nom (pas de fusion
   par mégarde).
4. **On ne perd jamais une note en supprimant** : un projet ou un sous-projet qui a encore des notes ne se supprime
   pas, et dit pourquoi. Archiver range un projet sans rien effacer.
5. **Les initiales** : de 1 à 5 lettres ou chiffres, en capitales, uniques dans l'équipe. Elles sont **proposées**
   d'après le nom, selon l'usage du bureau — l'initiale du prénom et deux lettres du nom (Louis Lejeune → LLE) ;
   trois mots et plus, l'initiale de chacun (Jean-Marc Dupont → JMD, Anne Van den Broeck → AVDB) — et se corrigent.
6. **La signature est une copie du moment.** Une note garde les initiales de qui l'a créée (`auteur`) et de qui l'a
   modifiée en dernier (`par`) ; une ligne du journal, celles de son auteur. Changer ses initiales ou quitter l'équipe
   ne réécrit pas le passé. La signature ne porte **jamais** l'e-mail.
7. **Le journal dit ce qui compte, une fois.** Créer, renommer, ranger, supprimer, exporter, importer une note ; créer,
   renommer, archiver, supprimer un projet ou un sous-projet ; arriver dans l'équipe, y ajouter ou en retirer quelqu'un.
   Modifier une note dix fois en une demi-heure laisse UNE ligne (et aucune si on vient de la créer). 400 lignes au
   plus. **Le journal ne peut pas faire échouer une note** : il s'écrit après elle, et un journal qui ne s'enregistre
   pas ne lève rien.

## 3 · Les données (dans `localStorage`)

| Clé | Forme | Script |
|---|---|---|
| `stab.equipe.v1` | `{ moi: <id>, membres: { <id>: { id, nom, email, initiales, couleur (0–7), cree } } }` | `js/equipe.js` |
| `stab.projets.v1` | `{ <cle>: { nom, cree, par (signature), archive, sous: [noms] } }` — le REGISTRE : il fait exister un projet ou un sous-projet vide, et garde l'ordre des sous-projets | `js/projets.js` |
| `stab.journal.v1` | `[ { t, q (signature), k (le geste), n: { id, nom, m }, p, s, de, x } ]`, les plus récentes d'abord | `js/journal.js` |
| `stab.notes.v1` | (inchangée) + `auteur`, `par` : des signatures `{ id, ini, nom, c }` | `js/notes.js` |

- **Le modèle affiché** (`projets.modele()`) réunit le registre et les notes : un projet que seules des notes nomment
  existe quand même (il entre au registre au premier geste, sans date de création inventée).
- **Une mémoire abîmée n'est jamais écrasée sans copie** (`js/memoire.js`, la règle de `notes.js`) : son texte est
  d'abord gardé sous `<clé>.illisible.<heure>` ; si la copie échoue, rien n'est écrit et le geste est refusé.
- **Ce qui vient d'ailleurs est nettoyé** : la signature d'un `.stab` reçu ou d'un journal réécrit à la main ne garde
  que des initiales valides, un nom borné, une teinte de la palette (`signaturePropre`).
- Copies liées par `tests/test_site_bureau.py` (R2) : les pages du bureau (`construire.PAGES_BUREAU` ↔ `app.js`),
  les huit teintes (`equipe.js` ↔ `css/bureau.css`), les longueurs (5 initiales, 80 caractères) entre données et champs.

## 4 · Les écrans

Trois écrans montés par l'application (`js/app.js`), sans moteur de calcul : ils sont prêts dès qu'ils sont montés.
Ce qu'ils partagent — une ligne du journal, la pastille, « Qui êtes-vous ? », le champ qui crée à Entrée, la ligne
d'une note, la fenêtre « Nouvelle note » — vit dans `js/vues_bureau.js` et `js/nouvelle.js` (R1). La feuille :
`css/bureau.css`, par-dessus la charte validée en maquette (`.avatar`, `.feed`, `.site`, `.proj`, `.team`…).

### 4.1 · L'accueil (`js/accueil.js`)

À droite des modules (dessous à la tablette et au téléphone) : **Projets**, puis **Activité**. Un projet se crée d'ici
sans quitter la page. Tant que personne ne s'est présenté, la carte Activité demande « Qui êtes-vous ? » : le prénom et
le nom suffisent, les initiales se proposent à mesure dans leur pastille.

### 4.2 · Projets (`js/ecran_projets.js`)

- **PC** : l'arbre | les notes | le projet et son activité. **Tablette** : l'arbre | les notes, l'activité dessous.
  **Téléphone** : la liste des projets (son champ en tête), puis la page d'un projet avec « ‹ Projets ».
- La sélection vit dans l'**adresse** : un nœud de l'arbre est un lien, « retour » revient, une adresse se partage.
- **Une seule barre pour les trois formats** : le fil d'Ariane, « Importer .stab » et « Nouvelle note » restent
  atteignables au téléphone (la charte y cache la barre des autres pages).
- La table des notes s'adapte à la largeur de SA colonne (requête de conteneur) : cinq colonnes, quatre, puis une
  fiche de deux lignes par note.

### 4.3 · Équipe (`js/ecran_equipe.js`)

Ma fiche se corrige sur place (chaque champ s'enregistre quand on en sort ; un refus dit pourquoi et rend l'ancienne
valeur). « Ce n'est pas vous ? » passe le poste à un autre membre. L'horloge d'un membre ne montre que son activité.

## 5 · Sécurité et vie privée

- **Tout nom s'affiche comme du texte** (projet, sous-projet, personne, note) : contrôlé dans le vrai navigateur avec
  un nom « `<img onerror>` », du titre au journal.
- **L'e-mail** est facultatif, reste dans le navigateur, et n'entre dans aucune signature ni aucun `.stab`.
- **L'adresse de connexion** : derrière Cloudflare Access, le site sait qui est connecté
  (`/cdn-cgi/access/get-identity`, sur le site même : `connect-src 'self'`, rien d'un tiers). Elle ne sert qu'à
  **préremplir** le champ e-mail quand on commence à se présenter ; son absence (le serveur local, une réponse qui
  n'est pas du JSON) ne dit rien et n'empêche rien. **Non vérifié en production** : le préremplissage n'a été joué
  qu'avec une réponse simulée.
- Aucun formulaire n'a d'action (`form-action 'none'`), aucun geste n'est écrit dans une page.

## 6 · Les contrôles

- `tests/test_site_bureau.py` — 78 contrôles : les données jouées dans node (initiales, fiches refusées, signature,
  journal qui ne se répète pas, projets, renommage, suppressions refusées, mémoire abîmée, `.stab` piégé), les pages,
  les copies liées, le contraste des huit teintes (≥ 4,5), et **18 mutations** qui rejouent le fichier entier sur une
  copie abîmée des sources.
- `site/web/verifier_site.js`, filtre **`bureau`** — 107 contrôles, aux quatre formats, dans un navigateur neuf : le
  parcours entier (se présenter, un projet, deux sous-projets, une note créée depuis le sous-projet, la retrouver,
  ranger, renommer, refuser de supprimer, le nom piégé, l'équipe, le retour à l'accueil, le rechargement) et ce que
  l'œil voit (rien ne dépasse de l'écran, pastilles rondes et lisibles, cibles de 40 px au doigt).
- `tests/test_app_navigation.py` (les routes `projets` et `equipe`, les écrans sans effet au chargement, quatre
  mutations), `tests/test_site_web.py` et `tests/test_apercus.py` (24 pages) suivent.

**Ce que les photos et le harnais ont trouvé pendant le lot** (corrigé, figé par un contrôle) :

1. *Au téléphone, un projet s'ouvrait sans ses notes* : la charte, écrite pour la maquette, cache « .list .tablecard »
   sous 700 px. Vu sur une photo ; le harnais exige désormais que la ligne de la note soit VUE, dans l'écran.
2. *La page Équipe faisait 413 px pour 390* (deux colonnes gardées au téléphone).
3. *L'arbre décalé et barré de traits* : une règle « .chantier » de l'ancien accueil visait aussi les nœuds
   « .node.chantier ». Règle retirée.
4. *Une pile de pastilles mangeait la dernière lettre des initiales* (recouvrement de la charte) : côte à côte.
5. *Un champ de 44 px dont la saisie n'en faisait que 23* : on touchait à côté. La saisie prend la hauteur du cadre.
6. *Une erreur 404 à chaque ouverture de l'accueil* (l'adresse de connexion demandée pour rien hors de Cloudflare) :
   elle n'est plus demandée qu'au moment où l'on commence à se présenter.
7. **Piège du harnais** : photographier une page plus haute que l'écran fait perdre à Chromium l'émulation tactile
   jusqu'au document suivant ; une page atteinte sans rechargement juste après était mesurée « à la souris ». Chaque
   mesure dit maintenant si le navigateur est au doigt, et le contrôle l'exige.

## 7 · Ce qui reste ouvert

1. **Le partage entre les postes** : chaque navigateur a ses projets, son équipe, son journal. Deux collègues ne se
   voient pas encore ; un `.stab` transporte une note avec les initiales de son auteur. C'est l'objet des comptes.
2. **Les invitations par e-mail, les rôles et les droits** (cahier des charges § 3) : pas dans ce lot. « Ajouter un
   collègue » l'inscrit sur ce poste, sans rien lui envoyer.
3. **Qui je suis n'est pas prouvé** : « Ce n'est pas vous ? » change de personne sans mot de passe. Sur un poste
   partagé, le journal dit ce que le poste a déclaré. Les comptes le régleront.
4. Le journal ne note pas encore « a produit la note PDF » (le geste est prévu : `note.pdf`).
5. Un projet n'a ni indice, ni statut, ni rapport assemblé (cahier des charges § 5 et 5 bis) : ce sont les lots suivants.
