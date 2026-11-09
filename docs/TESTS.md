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
- un Python différent ; une suite non suivie ; `--tout`.

### Ce que le suivi ne voit pas (et donc ce que `--tout` garde pour rôle)

- une suite qui teste l'EXISTENCE d'un fichier sans l'ouvrir ni lister son dossier (`os.path.exists`) ;
- ce qu'un sous-processus non suivi lit (la suite est alors rejouée de toute façon) ;
- l'environnement hors du dépôt (une bibliothèque mise à jour, `node`, `pdftotext`) : `--tout` quand il change.
Avant une LIVRAISON, `python3 lancer_tests.py` suffit si le témoin date de la même machine ; `--tout` reste la preuve
complète quand on en doute (après un changement d'environnement, ou une fois par lot important).

### La preuve

`tests/test_lancer_tests.py` fabrique un petit dépôt et vérifie le lanceur et le suivi : ce qui est joué, ce qui est
inchangé, ce qu'un fichier de données, un dossier listé, un module ou un nouveau test rejouent, le rouge qui
devient vert et l'inverse, `--tout`, `-j 1`, le filtre — et trois MUTATIONS (sans `open`, sans l'appariement des fils
Python, sans `os.listdir`, le suivi laisserait passer un changement).

### Mesure

Avant (lot 3, 09/11/2026) : 103 suites, une à la fois, 1 845 s (le catalogue 512 s, les pieds de poteau 355 s, les
modules 153 s). Premier passage du lanceur (09/11/2026, 02:08 UTC, quatre agents de code au travail à côté) : 105
suites en 757 s sur 4 processus, 8411 OK ; 20 suites non suivies (node, pdftotext, python -I), toujours rejouées ; le
premier passage a trouvé un défaut du suivi lui-même (il se comptait parmi les modules du dépôt : corrigé, il s'efface
de `sys.modules` au premier événement). La mesure d'un passage « rien n'a bougé » sera notée au prochain lot.
