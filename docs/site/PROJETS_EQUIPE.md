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
| Le menu (les pages qui ont la barre, les onglets du téléphone) | **Projets**, **Équipe** et, depuis le lot 6 (14/11/2026), **Paramètres** sont des liens (ils disaient « bientôt »). |
| **Accueil** | La pastille de la personne et « Bonjour Louis » ; la carte **Projets** (un champ qui en crée un — le nom, Entrée —, les six plus récemment actifs, « Sans projet », « Tout voir ») ; la carte **Activité** (les six dernières lignes du journal, chacune avec la pastille de son auteur ; « Qui êtes-vous ? » tant que personne ne s'est présenté) ; les notes à reprendre disent leur projet et leur auteur. |
| **Projets** (`projets`, `projets?p=<projet>&s=<sous-projet>`, `projets?sans=1`) | L'arbre des projets et de leurs sous-projets — son champ **cherche et crée** —, les notes du projet groupées par sous-projet, les faits et l'activité du projet. Renommer en écrivant par-dessus le titre ; ranger une note par un menu ; archiver ; supprimer ce qui est vide. |
| **Équipe** (`equipe`) | Ma fiche (nom, initiales, e-mail, teinte de la pastille), les membres, l'ajout d'un collègue, le journal entier (filtrable par membre). |
| **Rôles et modules vérifiés** (§ 8, ajouté le 11/11/2026) | Administrateur ou Membre ; un administrateur déclare un module **vérifié** : la marque se voit sur l'accueil, dans le volet des modules, dans « Nouvelle note » et au pied du résultat du module, avec son nom et la date ; elle tombe d'elle-même si le calcul du module change. La **recherche** de l'accueil (Ctrl K), « Tout voir », la pastille du poste au pied de la barre. |
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
| `stab.gabarit.v1` | (lot 6) `{ modules: { <module>: { <option>: valeur } }, par (signature), t }` — les ÉCARTS à l'usine du gabarit du bureau, qu'un administrateur fixe (§ 9) | `js/gabarit.js` |

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

- `tests/test_site_bureau.py` — 137 contrôles (78 avant le § 8) : les données jouées dans node (initiales, fiches
  refusées, signature, journal qui ne se répète pas, projets, renommage, suppressions refusées, mémoire abîmée, `.stab`
  piégé ; puis les rôles, les vérifications, l'empreinte d'un module, la recherche), les pages, les copies liées, le
  contraste des huit teintes (≥ 4,5), et **37 mutations** (18 avant le § 8) qui rejouent le fichier entier sur une copie
  abîmée des sources.
- `site/web/verifier_site.js`, filtre **`bureau`** — 151 contrôles (107 avant le § 8), aux quatre formats, dans un
  navigateur neuf : le parcours entier (se présenter, un projet, deux sous-projets, une note créée depuis le
  sous-projet, la retrouver, ranger, renommer, refuser de supprimer, le nom piégé, l'équipe, les rôles, la vérification
  de deux modules, le retour à l'accueil, la recherche, le module ouvert, le rechargement) et ce que l'œil voit (rien ne
  dépasse de l'écran, pastilles rondes et lisibles, cibles de 40 px au doigt).
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
2. **Les invitations par e-mail et les droits fins** (cahier des charges § 3 : partager une note, un dossier
   personnel) : pas encore. « Ajouter un collègue » l'inscrit sur ce poste, sans rien lui envoyer. Les deux RÔLES
   (Administrateur, Membre) existent depuis le § 8 — déclarés, pas prouvés (point 3).
3. **Qui je suis n'est pas prouvé** : « Ce n'est pas vous ? » change de personne sans mot de passe. Sur un poste
   partagé, le journal dit ce que le poste a déclaré. Les comptes le régleront.
4. Le journal ne note pas encore « a produit la note PDF » (le geste est prévu : `note.pdf`).
5. Un projet n'a ni indice, ni statut, ni rapport assemblé (cahier des charges § 5 et 5 bis) : ce sont les lots suivants.

## 8 · Les rôles, les modules vérifiés et les détails de l'accueil (11/11/2026, à la suite)

*Demande de l'ingénieur, le même jour : « il y avait tous les petits détails. Par exemple […] quand un module est
vérifié, un des administrateurs peut mettre vérifié. Et comme ça, c'est un truc qui devient un peu certifié. Et donc, on
sait qu'on pourra les utiliser sans se poser de questions et sans revérifier. […] pour que la page d'accueil soit
complète, plus agréable aussi. »* C'est le § 4 du cahier des charges (« Validation des modules »), dit avec le mot de
l'ingénieur : **vérifié**.

### 8.1 · Les rôles (`js/equipe.js`)

- Deux rôles : **Administrateur** et **Membre** (`ROLES`, la source unique des libellés ; la charte les peint).
- **Le premier inscrit du poste est administrateur** : c'est lui qui met le bureau en place. Les suivants sont membres.
- **Seul un administrateur change un rôle** (`nommer`) — le menu « Rôle » de la fiche d'un collègue n'existe que pour lui.
- **Il y a toujours un administrateur** : le dernier ne peut pas se retirer le rôle (« nommez-en un autre d'abord ») ;
  s'il quitte l'équipe, ou si l'équipe date d'avant les rôles, le plus ancien inscrit le devient — et c'est ÉCRIT dans
  le registre, pas seulement lu.
- Un rôle mal écrit dans le registre (réécrit à la main, piégé) vaut « membre ».

### 8.2 · Un module vérifié (`js/verifications.js`, `stab.verifications.v1`)

- Un administrateur déclare qu'il a **relu la méthode** d'un module (page Équipe, carte « Vérification des modules » :
  « Vérifier », avec une confirmation qui dit ce qu'on affirme). La vérification porte **sa signature, la date et la
  version du module** ce jour-là.
- **Trois états**, nommés par `etat()` : `verifie` (vérifié, et le calcul n'a pas changé), `change` (vérifié, mais le
  calcul a changé depuis : « à revérifier ») et `non` (« à vérifier »).
- **La marque ne se montre que pour `verifie`** (`js/marques.js`). Elle est écrite CACHÉE dans les pages
  (`construire.marque_verifie`) : sans le script, ou avant lui, aucun module ne se dit vérifié.
- **Où on la voit** (la même, depuis un composant partagé — R1) : la tuile du module sur l'accueil, sa ligne du volet
  des modules (toutes les pages), son bouton dans « Nouvelle note », son résultat dans la recherche, et **dans le
  module**, au pied du cadre Résultat : « Module vérifié · LLE · 11/11/2026 », « Module à vérifier » ou « Module à
  revérifier ». Cette ligne se LIT : ce n'est pas un bouton (au doigt, un bouton du résultat doit faire 44 px — il
  aurait pris une ligne au résultat).
- **Ce que la marque dit, et ce qu'elle ne dit pas** : la MÉTHODE du module a été relue par un administrateur ;
  **chaque note reste à relire par son auteur** (une saisie fausse donne un résultat faux dans un module vérifié).
  L'infobulle et la page Équipe le rappellent.
- Seul un administrateur vérifie ou **retire** une vérification ; chaque geste laisse sa ligne au journal
  (`module.verifie`, `module.retire`, et `membre.admin` / `membre.membre` pour les rôles).
- **ARGENT puis OR** (demande du propriétaire, 14/11/2026 : « deux personnes qui valident un module, c'est plus sûr
  qu'une ») : la marque est **argentée** quand un administrateur a vérifié la version du jour, **dorée** quand **deux
  administrateurs différents** l'ont vérifiée — « Module vérifié deux fois · LLE, MDU · 14/11/2026 », les deux noms dans
  l'infobulle et sur la page Équipe. Les règles, écrites dans `etat()` et `verifier()` :
  - le registre garde **une ligne par administrateur** (`{ <module>: [{ par, t, version }, …] }`) ; l'ancienne forme (une
    seule ligne) se lit encore, comme une marque argentée ;
  - une personne ne compte **qu'une fois** (son identifiant ; à défaut, ses initiales et son nom) : revérifier remplace
    sa ligne, la marque reste argentée ;
  - seules les signatures de la **version du jour** comptent : quand le calcul change, les deux tombent ensemble
    (« à revérifier ») ; revérifier repart de l'argent et efface les lignes périmées ;
  - **« Vérifier aussi »** n'est proposé qu'à un administrateur qui n'a pas encore signé une marque argentée ; « Retirer »
    efface tout, or compris ;
  - les couleurs sont des jetons de la charte (`--argent`, `--or` pour le texte, lus à 4,5 contre 1 ; `--argent-vif`,
    `--or-vif` pour le pictogramme seul, à 3 contre 1 — dans les deux thèmes, un test les mesure).

### 8.3 · La version d'un module : l'empreinte de son calcul (`construire.empreinte_module`, `js/versions.js`)

Une vérification vaut pour UNE version : il fallait donc que « la version du module » soit un fait, pas un numéro
qu'on pense à changer. C'est l'**empreinte des fichiers que le module charge pour calculer**.

- La construction du site rejoue déjà le parcours de chaque module pour savoir quels fichiers il charge (le relevé du
  paquet du moteur, `construire.releve`) ; elle garde maintenant la liste **par module** et en tire `sha256[:16]`.
- **Ce qui la fait changer** : une formule, un coefficient, une donnée (une table JSON), l'adaptateur du module, un
  fichier en plus ou en moins. La marque tombe alors d'elle-même : le module redevient « à revérifier ».
- **Ce qui ne la fait pas changer** : un commentaire ou une ligne vide dans un fichier Python (`_calcul_seul`), et la
  plomberie que TOUS les modules partagent (`HORS_VERSION` : le routeur, le registre, les briques d'écran) — sinon
  retoucher un écran ferait tomber toutes les vérifications du bureau.
- **Le choix est du côté sûr** : un fichier partagé par plusieurs modules de calcul (les classes de béton, par
  exemple) fait tomber la marque de chacun d'eux ; une docstring modifiée aussi (c'est une ligne de code pour
  l'empreinte). On revérifie parfois pour rien ; on ne garde jamais une marque sur un calcul qui a bougé.
- `js/versions.js` est ÉCRIT par la construction (ne pas le modifier) ; `tests/test_site_web.py` le lie au relevé :
  un fichier du calcul modifié sans reconstruire le site y rougit.

### 8.4 · Les détails de l'accueil

- **La recherche** (`js/accueil.js` : `indexRecherche`, `chercher`) : un module, une bibliothèque, un projet, un
  sous-projet, une note de ce navigateur ; sans accents ni capitales ; plusieurs mots, chacun doit s'y trouver ; huit
  résultats au plus, d'abord ce dont le nom commence par la recherche. Au clavier : flèches, Entrée, Échap ; **Ctrl K**
  y ramène. Elle ne propose pas les modules masqués.
- **« Tout voir »** au-dessus des notes à reprendre (vers la page Projets).
- **La légende** « [argentée] vérifié par un administrateur [dorée] par deux » au-dessus des modules (« vérifié · par
  deux » en tête du volet).
- **La pastille de la personne du poste au pied de la barre**, sur toutes les pages (PC et tablettes) : elle mène à
  l'équipe. Au téléphone, la barre n'existe pas : la pastille est en tête de l'accueil et l'onglet Équipe y mène.
  (C'est pourquoi les règles de la pastille sont passées de `css/bureau.css` à `css/site.css`.)
- La page Équipe dit **« Qui peut faire quoi »** (Administrateur, Membre, Module vérifié).

### 8.5 · Les contrôles du § 8

- `tests/test_site_bureau.py`, « 3 bis » : les rôles (premier inscrit, toujours un administrateur, refus, équipe
  d'avant les rôles), les vérifications (qui, quoi, les trois états, le registre piégé ou abîmé, le journal),
  l'empreinte (un commentaire ne la change pas, un coefficient si, la plomberie non), la recherche ; les pages (la
  marque cachée partout, une par module et la sienne, la légende, le gabarit de l'établi, la barre) ; 19 mutations de
  plus.
- `tests/test_site_web.py` : `js/versions.js` est celui que le générateur écrit ; la version d'un module vient de SES
  fichiers.
- Harnais, filtre `bureau` (11 contrôles de plus par format) : un administrateur vérifie deux modules, un membre ne le
  peut pas, le calcul d'un des deux « change » (son empreinte est réécrite dans le registre) et sa marque tombe partout
  — page Équipe, accueil, volet, « Nouvelle note », pied du module ; les rôles (nommer, reprendre, refus du dernier
  administrateur) ; la recherche ; la pastille de la barre. **Sept mutations du site** le font rougir (la règle
  « hidden », la marque gardée pour un module à revérifier, les boutons montrés à un membre, les cibles de 44 px, la
  ligne du module sans nom ni date, la recherche sans accents, la barre sans pastille).

**Ce que les contrôles ont trouvé pendant le § 8** : la charte affiche tout « .who2 > div span » — l'écusson du rôle,
prévu pour le téléphone, se voyait aussi au PC sous le nom (vu sur une photo ; la règle est plus précise, un contrôle
la tient) ; les classes de la ligne du module n'avaient pas le préfixe « et- » (convention B.2.6, relevée par
`test_site_web`).

### 8.6 · Les limites du § 8 (dites à l'écran)

1. **Par poste, comme le reste** : un module vérifié sur ce navigateur ne l'est pas sur celui d'un collègue. La marque
   deviendra celle du bureau avec les comptes.
2. **Un rôle est une DÉCLARATION** : « Ce n'est pas vous ? » change de personne sans mot de passe ; sur un poste
   partagé, n'importe qui peut prendre la place de l'administrateur. La marque dit ce que le poste a déclaré.
3. La note PDF ne dit pas encore si son module était vérifié le jour où elle a été produite (ni argent, ni or).
4. Le mot : « **vérifié** » (celui de l'ingénieur) ; le cahier des charges et la maquette disaient « validé ».

## 9 · La page PARAMÈTRES et le gabarit du bureau (lot 6, 14/11/2026)

*Décision du propriétaire du 13/11/2026 : les options par défaut de chaque module, que le responsable fixe ; chaque note
peut changer une valeur au cas par cas. Le détail : `docs/site/MODELES_DIMENSIONNEMENT.md` § 7.*

- **La page** (`parametres`, `js/ecran_parametres.js`) : « Le site » (le thème : Clair · Sombre · Comme l'appareil — le
  bouton du rail et celui de l'en-tête du téléphone posent le même, `commun.poserTheme`), puis le **gabarit du bureau** :
  une carte par module (`construire.MODULES_GABARIT`, liée au registre du moteur par `tests/test_gabarit.py`), sa
  rubrique Options **rendue par le moteur** — c'est la seule page du bureau qui DEMANDE le moteur
  (`app.js BUREAU_AVEC_MOTEUR`). Elle lit ses champs comme une note (`commun.valeurSaisie`, `viderAuFocus`,
  `segmentAEnvoyer` : une source pour les deux).
- **Les rôles** : un administrateur change le gabarit ; un membre le lit, ses contrôles éteints ; personne présenté : la
  page renvoie à l'Équipe (le premier inscrit du poste est administrateur, § 8.1).
- **Le journal** : `gabarit.change` (« a changé le gabarit du bureau : <module> », une ligne par visite) et
  `gabarit.usine` (« a rendu aux valeurs d'usine le gabarit : <module> »).
- **Les limites** (dites sur la page) : le gabarit est celui de CE navigateur, comme l'équipe — il deviendra celui du
  bureau avec les comptes ; une note neuve le reçoit, une note existante jamais.

