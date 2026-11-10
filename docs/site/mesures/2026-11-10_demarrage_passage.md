# Démarrage de l'établi — le passage d'un module à l'autre (10/11/2026)

Mesure intermédiaire, prise à la demande de l'ingénieur (« le message “Python démarre dans le navigateur…” revient à
chaque écran ; est-ce un redémarrage du moteur ou seulement un affichage répété ? »), avec le script et le protocole du
lot 0 (`site/web/outils/mesurer_demarrage.js`, `PLAN=quatre`, `REP=3`, profil Chromium neuf sur disque à chaque tirage,
médiane de 3). Elle ne remplace ni la mesure « avant » du 07/11 (`2026-11-07_demarrage_avant.md`) ni la mesure « après »
que le lot S4 de `REFONTE_ETABLI.md` devra verser.

**Réponse.** C'est un REDÉMARRAGE. Chaque module est une page HTML à part avec son propre travailleur ; passer d'un
module à l'autre recharge le document, donc relance Python en entier (`loadPyodide`, déballage du paquet du moteur,
imports). Le passage coûte autant qu'une deuxième visite du même module, et la part de Python dans ce temps est de 75 à
85 %. Le voile `.demarrage` couvre tout l'établi jusqu'à la première réponse, alors que le squelette de la page est
peint en ≈ 0,1 s.

## 1. Conditions

- Code `2fd6dd6` (site reconstruit `9fe3d3a`, paquet du moteur `c619f3ed232f`, 270 fichiers, 1 710 ko), Pyodide 0.29.5,
  numpy chargé pour les seuls modules du manifeste `avec_numpy` (plancher et age ici ; pas poutre ni assemblages).
- Même machine que le 07/11 (Xeon 2,1 GHz, 4 cœurs), Chromium de Playwright, serveur local `serveur_local.py 8771`
  sans compression. Charge moyenne sur une minute pendant les tirages : 2,06 (médiane). 72 relevés, 0 erreur.
- Un vrai téléphone ou une tablette sur internet n'est PAS mesuré ici : le téléphone est émulé (390 × 844, agent
  Android) sans ralentir le processeur, et le réseau est local. L'ingénieur observe ≈ 10 s sur tablette en ligne ; la
  structure du coût est la même, la vitesse du processeur et le réseau (8,6 Mo de WebAssembly à la première visite)
  font la différence.

## 2. Temps jusqu'au premier rendu (ms, médiane de 3)

| Module | Froid PC | 2ᵉ visite PC | Passage PC | Froid tél. | 2ᵉ visite tél. | Passage tél. |
|---|---|---|---|---|---|---|
| poutre | 2 546 | 1 894 | 1 851 | 2 542 | 1 958 | 1 841 |
| assemblages | 3 153 | 2 503 | 2 460 | 3 073 | 2 457 | 2 573 |
| plancher | 3 584 | 2 961 | 2 819 | 3 541 | 2 933 | 2 821 |
| age | 3 335 | 2 818 | 2 765 | 3 346 | 2 655 | 2 724 |

La colonne « passage » (vers ce module, depuis le précédent du cycle poutre → assemblages → plancher → age → poutre,
même profil) est égale à la 2ᵉ visite : rien n'est réutilisé d'un module à l'autre.

## 3. Où va le temps (2ᵉ visite, PC, ms, médianes)

| Module | loadPyodide (froid → 2ᵉ) | numpy | import du module | premier calcul |
|---|---|---|---|---|
| poutre | 2 044 → 1 438 | — | 82 | 27 |
| assemblages | 1 978 → 1 495 | — | 536 | 98 |
| plancher | 2 020 → 1 445 | 453 | 283 | 46 |
| age | 2 004 → 1 539 | 444 | 92 | 46 |

Le cache du navigateur (code WebAssembly compilé, HTTP) ne fait gagner que ≈ 0,5 s sur `loadPyodide` : l'interpréteur
et sa bibliothèque standard se réinitialisent à chaque démarrage. numpy ajoute ≈ 0,45 s là où il est chargé. Le calcul
lui-même est négligeable (≤ 0,1 s).

## 4. Ce que ça dit pour la suite

- Le gain n'est pas dans le calcul ni dans le réseau local, il est dans le NOMBRE de démarrages : un par session au
  lieu d'un par module. C'est le lot S4 de `REFONTE_ETABLI.md` (une seule application, un seul travailleur gardé en vie,
  navigation sans rechargement, modules importés à la demande et conservés, préchauffage à l'approche d'une tuile,
  aperçus précalculés pour afficher l'écran en moins de 300 ms, voile retiré). Attendu : le passage d'un module à
  l'autre tombe de ≈ 2 à 3 s ici (≈ 10 s observés sur tablette en ligne) à l'import du module, 0,1 à 0,5 s ; le premier
  démarrage de la session reste.
- Sans le lot, un gain de perception seulement : retirer le voile et montrer l'établi (champs inertes) pendant que
  Python démarre.
- Hors périmètre, comme écrit en D8 de la refonte : l'instantané mémoire de Pyodide (API privée) ; un service worker
  ne garde pas Python en vie.

Relevés bruts : `scratchpad` de la session (`mesure_2026-11-10.json`), non versés ; le script les régénère.
