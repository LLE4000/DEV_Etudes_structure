# Les tests : comment on les lance, et pourquoi on ne rejoue pas tout

*09/11/2026 — demande de l'ingénieur : « arrête de relancer systématiquement la suite complète ; vérifier à chaque
fois le catalogue alors qu'on n'y touche pas, c'est bête ».*

## Les suites

Chaque fichier `tests/test_*.py` est une SUITE : un script qui imprime ses contrôles (`OK` / `KO`), termine par
`RÉSULTAT : n OK, m échec(s)` et rend un code de sortie. Une suite se lance seule :

    /usr/bin/python3.12 tests/test_api_web_dalle.py

## Le lanceur : `lancer_tests.py`

    python3 lancer_tests.py                 # les suites à rejouer
    python3 lancer_tests.py assemblages     # celles dont le nom contient « assemblages »
    python3 lancer_tests.py --tout          # toutes, sans regarder le témoin (avant une livraison qui en doute)
    python3 lancer_tests.py -j 1            # une à la fois (défaut : un processus par cœur)
    python3 lancer_tests.py --pourquoi      # dit pour chaque suite pourquoi elle est rejouée

Il joue les suites EN PARALLÈLE (un processus par cœur, les plus longues d'abord) et ne rejoue que celles dont une
DÉPENDANCE a changé depuis leur dernier vert. Le reste est dit `INCHANGÉ`, avec la date de son vert : ce vert vaut
toujours, puisque pas un octet de ce que la suite avait lu n'a bougé.

### Le suivi des dépendances

Pendant qu'une suite tourne, `outils/suivi_tests/sitecustomize.py` (posé sur `PYTHONPATH` par le lanceur, inerte
sans les variables `STAB_SUIVI` et `STAB_SUIVI_RACINE`) note ce qu'elle lit dans le dépôt :

- les MODULES chargés (`sys.modules` à la fin : imports directs, différés, par chaîne) ;
- les FICHIERS ouverts en lecture (événement d'audit `open` : json, html, svg, zip, gels, docs…) ;
- les DOSSIERS listés par la suite (`os.listdir`, `os.scandir`, donc `glob`, `os.walk`, `pathlib`) — un fichier
  ajouté ou retiré y change la liste ; les listages que le mécanisme d'import fait pour lui-même ne comptent pas
  (ajouter un test ne rejoue pas toutes les suites) ;
- les SOUS-PROCESSUS (`subprocess.Popen`). Un fils Python ordinaire hérite du suivi et écrit sa propre ligne (ses
  lectures comptent). Un fils que le suivi ne peut pas lire — `python -I` / `-E` / `-S`, un environnement refait sans
  `PYTHONPATH`, `node`, `pdftotext`, `git`… — rend la suite NON SUIVIE : elle est toujours rejouée, et le lanceur le
  dit (`· non suivie, toujours rejouée`).

Le TÉMOIN d'une suite verte (`.temoin_tests/<suite>.json`, jamais versionné, propre à chaque arbre de travail)
garde l'empreinte SHA-256 de chaque fichier lu et les noms de chaque dossier listé, le Python, la durée, le compte.

### Ce qui rejoue une suite (tout doute rejoue)

- jamais vue verte (nouvelle, ou rouge au dernier passage : un rouge efface le témoin) ;
- un fichier lu dont l'empreinte diffère, ou qui a disparu ; un dossier listé dont les noms diffèrent ;
- un Python différent ; une suite non suivie ; `--tout` ;
- un fils `pdftotext` d'une autre version que celle gardée au témoin (depuis le 14/11/2026, ci-dessous).

### Ce que le suivi ne voit pas (et donc ce que `--tout` garde pour rôle)

- une suite qui teste l'EXISTENCE d'un fichier sans l'ouvrir ni lister son dossier (`os.path.exists`) ;
- ce qu'un sous-processus non suivi lit (la suite est alors rejouée de toute façon) ;
- l'environnement hors du dépôt (une bibliothèque mise à jour, `node`) : `--tout` quand il change.

### `pdftotext`, `pdffonts`, `pdfinfo` : des outils PURS, suivis (14/11/2026)

Une suite qui relit sa note PDF lance `pdftotext` (ou `pdffonts`, `pdfinfo`, de la même famille poppler) sur l'entrée
standard (le PDF qu'elle vient d'écrire) ou sur un fichier nommé. Un tel fils ne lit que ce qu'on lui donne : il ne rend
plus la suite « non suivie » (`OUTILS_PURS` dans `lancer_tests.py`). Un fichier DU DÉPÔT nommé sur sa ligne de commande devient une dépendance ; sa version
(« pdftotext version 24.02.0 ») est gardée au témoin, et une autre version rejoue la suite. Les seize suites qui
relisent leur note PDF ne sont plus rejouées quand rien de ce qu'elles lisent n'a changé. Restent « non suivies » : les fils
`node` (le moteur partagé, la page simulée) et les Python lancés avec `-I` (la pureté du moteur, sans le suivi).
Avant une LIVRAISON, `python3 lancer_tests.py` suffit si le témoin date de la même machine ; `--tout` reste la preuve
complète quand on en doute (après un changement d'environnement, ou une fois par lot important).

### La preuve

`tests/test_lancer_tests.py` fabrique un petit dépôt et vérifie le lanceur et le suivi : ce qui est joué, ce qui est
inchangé, ce qu'un fichier de données, un dossier listé, un module ou un nouveau test rejouent, le rouge qui
devient vert et l'inverse, `--tout`, `-j 1`, le filtre — et trois MUTATIONS (sans `open`, sans l'appariement des fils
Python, sans `os.listdir`, le suivi laisserait passer un changement) ; depuis le 14/11/2026, le suivi de `pdftotext` (une
suite qui le lance sur l'entrée standard est suivie, un PDF du dépôt nommé est une dépendance, une autre version rejoue)
et sa mutation (sans les outils purs, la suite redevient « non suivie »).

## Le parcours du navigateur EN MORCEAUX : `outils/parcours_parallele.py` (14/11/2026)

    python3 outils/parcours_parallele.py http://127.0.0.1:8770 <dossier>                 # tout le parcours, 3 à la fois
    python3 outils/parcours_parallele.py … --reference <journal d'un passage complet>    # et chaque contrôle retrouvé

Le passage complet (`node site/web/verifier_site.js <base> <captures>`, sans filtre) joue ses sections l'une après
l'autre dans un seul navigateur : ≈ 31 min. Chaque section a déjà son FILTRE, qui la joue seule dans son propre
navigateur (« garde », « formats », « plaques »…). Le lanceur joue les filtres EN MÊME TEMPS (`-j`, 3 par défaut, les plus
longs d'abord d'après le dernier passage) et additionne. Ce qui garantit qu'on ne perd rien :
- `tests/test_parcours_parallele.py` lie la liste des morceaux aux gardes du parcours : une section ajoutée sans
  morceau, ou un morceau qui n'est pas un filtre, fait rougir le contrôle (mutations comprises) ; la seule partie que
  le passage complet joue seul (« demarrage ») est un morceau ;
- chaque morceau rend son code de sortie (un rouge, un « non mesuré » font rougir le tout) et vérifie l'empreinte de
  construction comme le passage complet ;
- `--reference` : chaque contrôle d'un passage complet (son nom, les mesures d'un nom — « 18 pages, 3,8 s » —
  neutralisées) doit se retrouver dans l'union des morceaux ; un nom absent est dit, et le passage rougit.

Mesure (14/11/2026, le site du lot 4, 4 processeurs) : 18 morceaux, 3 à la fois — **1 015 s (17 min)**, contre 1 860 s
(31 min) pour le passage complet du même site ; 18/18 verts, les 3 296 noms de contrôle du passage de référence tous
retrouvés. Le compte (3 509) dépasse celui du passage complet (3 346) : chaque morceau rejoue les contrôles de
l'accueil, au début et à la fin. Les plus longs : « formats » 481 s, « demarrage » 395 s, « ouverture » 330 s ; les
durées sont gardées (`.temoin_tests/parcours.json`) et le passage suivant commence par les plus longs. Le passage
complet d'une traite reste disponible (`node site/web/verifier_site.js <base> <captures>`).

**Les morceaux qui mesurent des TEMPS sont joués SEULS, après les autres** (`SEULS` : « sonde », « demarrage » —
14/11/2026). Le passage du badge argent/or les avait joués avec deux autres navigateurs : charge de 7 sur 4 cœurs, et
quatre rouges — deux « temps : non mesurable » (le morceau refuse de conclure sur une machine chargée), la fermeture au
téléphone (une attente fixe de 600 ms après la réponse) et l'ordre des marques de la sonde ; rejoués SEULS, machine au
calme : 179/0 et 26/0. Ce n'était ni le site ni le harnais, mais le lanceur qui les jouait mal ; un contrôle le tient
désormais (`tests/test_parcours_parallele.py` : « demarrage » commence quand les autres ont fini, et rien ne tourne avec
lui ; mutation : aucun morceau seul). Les morceaux se nomment aussi après une option (`… -j 1 sonde demarrage`).
Mesure du même passage : 16 morceaux en 864 s, 3 à la fois, puis les deux morceaux seuls en 436 s — **1 300 s (22 min)**
au lieu de 1 860 s d'une traite.

### Mesure

Avant (lot 3, 09/11/2026) : 103 suites, une à la fois, 1 845 s (le catalogue 512 s, les pieds de poteau 355 s, les
modules 153 s). Premier passage du lanceur (09/11/2026, 02:08 UTC, quatre agents de code au travail à côté) : 105
suites en 757 s sur 4 processus, 8411 OK ; 20 suites non suivies (node, pdftotext, python -I), toujours rejouées ; le
premier passage a trouvé un défaut du suivi lui-même (il se comptait parmi les modules du dépôt : corrigé, il s'efface
de `sys.modules` au premier événement). La mesure d'un passage « rien n'a bougé » sera notée au prochain lot.
