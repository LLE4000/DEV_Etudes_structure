# Démarrage de l'établi — mesure « avant » la refonte (07/11/2026)

Mesure de référence du lot 0 de la refonte de l'établi (`docs/site/REFONTE_ETABLI.md`, B.4). La mesure « après »
(lots S4 et V) se prendra avec le même script et le même protocole, et se versera à côté, sans réécrire celle-ci.

**En bref.** Ouvrir un calcul prend de 1,4 à 2,8 s sur cette machine (5 s pour le garde-corps), et ce temps se refait
à chaque page : le passage d'un module à l'autre coûte autant qu'une deuxième visite. L'essentiel est le démarrage de
Python (`loadPyodide` : 1,15 à 1,34 s en 2ᵉ visite ou au passage, 1,57 à 1,68 s à froid). numpy ajoute environ
0,65 s aux huit modules qui le chargent (≈ 0,29 s d'attente après le déballage, puis ≈ 0,35 s d'import). L'import du
module va de 26 ms (poutre) à 426 ms (assemblages), et le premier calcul du garde-corps prend à lui seul 2,6 s.

## 1. Ce qui a été mesuré

- **Code** : `a5230d9`, c'est-à-dire `e3a0ca0` (le site avant la refonte), plus le gel du lot 0 (des tests seulement),
  plus la sonde du démarrage (`site/web/js/sonde.js`), ouverte par `?sonde=1`. Fermée, elle ne change rien
  (`verifier_site.js`, filtre `sonde`) ; ouverte, elle date les étapes sans allonger le chemin du premier rendu.
  Paquet du moteur `86b3585a392c` (248 fichiers, 1 495 ko), Pyodide 0.29.5.
- **Machine** : Xeon 2,1 GHz, 4 cœurs ; Chromium 153.0.8010.12 piloté par Playwright ; Node 22.22. Le site est servi
  par `site/web/outils/serveur_local.py`, avec la politique de sécurité de `_headers` et sans compression. D'autres
  sessions tournaient sur la machine : la charge moyenne sur une minute, relevée à chaque tirage, allait de 1,3 à 3,0.
  Pendant le plan « quatre », un Chromium a aussi été lancé une seconde, hors du verrou des navigateurs, pour relever
  sa version.
- **Script** : `site/web/outils/mesurer_demarrage.js` (`PLAN=quatre`, puis `PLAN=dix-huit`).

## 2. Protocole

- **Médiane de 3 tirages.** Les tableaux donnent la médiane ; le tableau 1 donne aussi, entre parenthèses, le plus
  petit et le plus grand des trois tirages.
- **Profil Chromium sur disque**, neuf à chaque tirage (`launchPersistentContext`), comme un vrai navigateur. Un
  contexte privé fausserait la mesure : il ne garde pas en cache le `.wasm` de 8,6 Mo et le retélécharge à chaque
  visite.
- **Trois scénarios**, pour poutre, assemblages, plancher (charges sur plancher) et age (maturité du béton) :
  - **froid** : première visite, profil neuf, rien en cache ;
  - **2ᵉ visite** : le même module, rouvert dans le même profil (cache HTTP et cache de code WebAssembly) ;
  - **passage** : le module suivant, dans le même profil, selon le cycle poutre → assemblages → plancher → age →
    poutre. La ligne d'un module donne le passage **vers** ce module.
- **Deux formats** :
  - PC 1440×900 ;
  - téléphone émulé : 390×844, densité 3, tactile, agent Android. Le processeur n'est pas ralenti : le
    ralentissement de Chromium n'atteint pas le travailleur, où tourne Python (rapport d'enquête « démarrage »).
- **2ᵉ visite des 18 modules**, au PC seulement : pour chaque module, un profil neuf, une première visite, puis la
  seconde, seule retenue.
- **Chaque visite** ouvre l'établi sans note (`/<page>?sonde=1`), donc l'action « nouveau ». Le temps est compté depuis
  le début de la navigation.
- **Le premier rendu** est posé deux images (`requestAnimationFrame`) après l'application de la première réponse :
  c'est la marque `stab:premier_rendu`.
- Le serveur est local et répond sans délai : le travailleur reçoit 15,2 Mo à froid et 600 octets ensuite (médianes),
  mais le téléchargement y pèse peu. L'écart entre le froid et la 2ᵉ visite (≈ 0,4 s) vient surtout du démarrage de
  Python : « init. Python » prend de 1,43 à 1,54 s à froid, contre 1,07 à 1,19 s en 2ᵉ visite.

## 3. Temps jusqu'au premier rendu (ms, médiane, et entre parenthèses le minimum et le maximum)

| Module | Froid PC | 2e visite PC | Passage PC | Froid tél. | 2e visite tél. | Passage tél. |
|---|---|---|---|---|---|---|
| poutre | 2001 (1966–2021) | 1668 (1562–1675) | 1468 (1399–1550) | 1955 (1865–2300) | 1594 (1459–1622) | 1378 (1377–1457) |
| assemblages | 2493 (2457–2494) | 1975 (1893–1992) | 2064 (1998–2134) | 2420 (2390–2431) | 1977 (1957–1983) | 1963 (1956–1966) |
| plancher | 2802 (2791–2903) | 2357 (2321–2413) | 2281 (2272–2326) | 2838 (2774–2868) | 2323 (2259–2460) | 2256 (2149–2276) |
| age | 2742 (2652–2804) | 2244 (2177–2314) | 2244 (2032–2260) | 2638 (2551–2652) | 2177 (2073–2207) | 2143 (1977–2159) |

Avant Python, la page est rapide (médianes au PC, en 2ᵉ visite) :
- le HTML arrive en 2 ms ;
- le script de l'établi commence 16 ms plus tard (43 ms à froid) ;
- le travailleur démarre 20 ms après sa création ;
- la première peinture du squelette (FCP) a lieu à 104 ms (126 ms à froid).

Tout le reste est Python.

## 4. Décomposition (ms, médianes)

Ce que mesure chaque colonne :
- **wasm** : `WebAssembly.instantiateStreaming` du cœur Pyodide (téléchargement, compilation, instanciation).
- **init. Python** : de la fin de cette instanciation à la fin de `loadPyodide` (l'interpréteur et sa bibliothèque
  standard, qui est recompilée à chaque démarrage).
- **loadPyodide** : le total des deux.
- **numpy (chargement)** : `loadPackage("numpy")`. Il se fait en parallèle du paquet ; **attente numpy** est ce qui
  reste à attendre une fois le paquet déballé.
- **empreinte**, **déballage** : la vérification SHA-256 du paquet, puis `unpackArchive`.
- **routeur** : l'import de `api_web.routeur`.
- **import numpy**, **import du module** : faits à la première requête (la sonde les sépare du calcul ; sans elle, le
  routeur les fait au même moment).
- **premier calcul** : la requête « nouveau » (avec, s'il y a lieu, sa reprise quand le guet de numpy se déclenche).
- **vers la page** : le passage du message du travailleur à la page.
- **application** : la pose des fragments dans le DOM.
- **peinture** : deux images.
- **premier rendu** : le temps total depuis le début de la navigation.

### PC 1440×900

| Module | Scénario | wasm | init. Python | loadPyodide | numpy (chargement) | empreinte | déballage | attente numpy | routeur | import numpy | import du module | premier calcul | vers la page | application | peinture | premier rendu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| poutre | Froid | 57 | 1507 | 1654 | — | 2 | 68 | 2 | 58 | — | 28 | 33 | 1 | 37 | 31 | 2001 |
| poutre | 2e visite | 62 | 1193 | 1344 | — | 2 | 63 | 2 | 56 | — | 29 | 31 | 2 | 8 | 82 | 1668 |
| poutre | Passage | 55 | 1066 | 1152 | — | 2 | 64 | 2 | 55 | — | 27 | 30 | 1 | 18 | 78 | 1468 |
| assemblages | Froid | 56 | 1469 | 1622 | — | 2 | 66 | 2 | 57 | — | 440 | 109 | 2 | 50 | 30 | 2493 |
| assemblages | 2e visite | 64 | 1111 | 1261 | — | 2 | 61 | 2 | 50 | — | 419 | 74 | 2 | 25 | 28 | 1975 |
| assemblages | Passage | 51 | 1104 | 1187 | — | 2 | 64 | 3 | 58 | — | 474 | 115 | 2 | 29 | 44 | 2064 |
| plancher | Froid | 54 | 1454 | 1596 | 377 | 2 | 71 | 303 | 56 | 362 | 248 | 37 | 1 | 42 | 20 | 2802 |
| plancher | 2e visite | 63 | 1071 | 1212 | 348 | 2 | 61 | 285 | 52 | 361 | 232 | 34 | 1 | 17 | 49 | 2357 |
| plancher | Passage | 54 | 1046 | 1145 | 353 | 2 | 64 | 286 | 52 | 356 | 234 | 21 | 1 | 26 | 42 | 2281 |
| age | Froid | 58 | 1535 | 1684 | 369 | 2 | 66 | 300 | 53 | 364 | 77 | 40 | 3 | 39 | 29 | 2742 |
| age | 2e visite | 63 | 1104 | 1277 | 359 | 2 | 62 | 293 | 53 | 353 | 74 | 35 | 1 | 12 | 81 | 2244 |
| age | Passage | 54 | 1076 | 1161 | 359 | 2 | 64 | 292 | 51 | 358 | 74 | 34 | 1 | 24 | 51 | 2244 |

### Téléphone émulé 390×844

| Module | Scénario | wasm | init. Python | loadPyodide | numpy (chargement) | empreinte | déballage | attente numpy | routeur | import numpy | import du module | premier calcul | vers la page | application | peinture | premier rendu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| poutre | Froid | 56 | 1495 | 1638 | — | 2 | 75 | 3 | 58 | — | 27 | 34 | 1 | 31 | 6 | 1955 |
| poutre | 2e visite | 65 | 1166 | 1340 | — | 2 | 66 | 3 | 58 | — | 28 | 31 | 0 | 6 | 14 | 1594 |
| poutre | Passage | 52 | 1063 | 1149 | — | 2 | 60 | 2 | 52 | — | 24 | 27 | 0 | 15 | 15 | 1378 |
| assemblages | Froid | 54 | 1466 | 1618 | — | 2 | 65 | 2 | 56 | — | 442 | 113 | 2 | 43 | 3 | 2420 |
| assemblages | 2e visite | 65 | 1073 | 1236 | — | 2 | 62 | 2 | 52 | — | 458 | 74 | 3 | 30 | 9 | 1977 |
| assemblages | Passage | 52 | 1126 | 1212 | — | 2 | 68 | 2 | 57 | — | 432 | 109 | 3 | 25 | 6 | 1963 |
| plancher | Froid | 57 | 1489 | 1633 | 359 | 2 | 68 | 288 | 54 | 368 | 234 | 38 | 1 | 40 | 10 | 2838 |
| plancher | 2e visite | 61 | 1081 | 1221 | 342 | 2 | 61 | 279 | 55 | 359 | 241 | 34 | 1 | 13 | 16 | 2323 |
| plancher | Passage | 52 | 1078 | 1194 | 339 | 2 | 61 | 276 | 48 | 344 | 222 | 18 | 1 | 21 | 7 | 2256 |
| age | Froid | 55 | 1426 | 1568 | 359 | 2 | 66 | 290 | 52 | 367 | 74 | 39 | 1 | 26 | 5 | 2638 |
| age | 2e visite | 61 | 1083 | 1227 | 355 | 2 | 68 | 285 | 51 | 353 | 74 | 34 | 1 | 6 | 18 | 2177 |
| age | Passage | 54 | 1084 | 1182 | 340 | 2 | 62 | 276 | 50 | 346 | 77 | 35 | 1 | 9 | 10 | 2143 |

## 5. Les 18 modules en 2ᵉ visite, au PC (ms, médianes)

- **requête ordinaire** : `calculer` sur l'état rendu, médiane de 10 appels, juste après la première réponse.
- **registre.cles()** : la relecture du registre que `registre.module` refait à chaque requête, moyenne de 5 appels.

| Module | premier rendu | loadPyodide | numpy (chargement) | import numpy | import du module | premier calcul | application | peinture | réponse (octets) | requête ordinaire | registre.cles() |
|---|---|---|---|---|---|---|---|---|---|---|---|
| age | 2224 (2195–2258) | 1238 | 353 | 351 | 73 | 36 | 12 | 63 | 54 ko | 14,4 | 8,1 |
| armatures | 1519 (1515–1590) | 1243 | — | — | 27 | 38 | 18 | 52 | 50 ko | 18,5 | 8,5 |
| assemblages | 1944 (1923–2076) | 1209 | — | — | 426 | 80 | 24 | 40 | 217 ko | 63,5 | 8,2 |
| bois | 1593 (1590–1622) | 1282 | — | — | 31 | 37 | 14 | 60 | 71 ko | 15,5 | 8,2 |
| catalogue | 2260 (2246–2328) | 1188 | 349 | 353 | 61 | 119 | 51 | 20 | 121 ko | 57,5 | 7,8 |
| corniere | 2366 (2240–2459) | 1218 | 346 | 349 | 145 | 95 | 13 | 55 | 117 ko | 57,3 | 8,2 |
| dalle | 1623 (1520–1706) | 1244 | — | — | 50 | 64 | 15 | 36 | 71 ko | 38,3 | 8,6 |
| donnees_beton | 1729 (1617–1730) | 1352 | — | — | 90 | 30 | 8 | 42 | 16 ko | 10,7 | 7,8 |
| enrobage | 1578 (1475–1601) | 1257 | — | — | 30 | 28 | 11 | 54 | 30 ko | 10,1 | 9,0 |
| flambement | 2243 (2229–2412) | 1295 | 361 | 345 | 55 | 51 | 22 | 37 | 81 ko | 24,0 | 10,1 |
| garde_corps | 4873 (4865–5109) | 1188 | 345 | 348 | 210 | 2632 | 23 | 41 | 285 ko | 195,6 | 8,1 |
| plancher | 2428 (2263–2471) | 1241 | 373 | 355 | 229 | 22 | 18 | 45 | 84 ko | 17,8 | 8,5 |
| poutre | 1545 (1468–1605) | 1272 | — | — | 26 | 28 | 8 | 26 | 23 ko | 10,5 | 8,2 |
| predalle | 1643 (1629–1743) | 1235 | — | — | 61 | 105 | 16 | 83 | 83 ko | 78,4 | 8,1 |
| profil | 2232 (2106–2402) | 1204 | 350 | 348 | 91 | 22 | 21 | 43 | 60 ko | 16,0 | 8,6 |
| sol | 1661 (1564–1743) | 1206 | — | — | 146 | 44 | 15 | 48 | 74 ko | 23,3 | 8,4 |
| soudures | 2397 (2283–2434) | 1217 | 350 | 358 | 251 | 40 | 17 | 16 | 102 ko | 26,9 | 8,2 |
| taux | 1560 (1546–1612) | 1241 | — | — | 44 | 29 | 6 | 62 | 18 ko | 9,7 | 8,6 |

## 6. Ce qu'on en lit

1. **Python domine.** `loadPyodide` représente de la moitié du premier rendu (plancher, cornière, soudures) à plus
   des quatre cinquièmes (poutre, armatures), et le quart seulement pour le garde-corps. Au PC, il prend 1 344 ms sur
   1 668 pour la poutre en 2ᵉ visite, et 1 654 ms sur 2 001 à froid. L'instanciation du WebAssembly n'en prend que
   51 à 65 ms ; le reste est le démarrage de l'interpréteur.
2. **Le passage d'un module à l'autre ne réutilise rien.** Il coûte autant qu'une 2ᵉ visite : poutre 1 468 ms,
   assemblages 2 064, plancher 2 281, age 2 244 au PC. Chaque page refait le démarrage de Python. Le seuil du lot S4
   (« passage ≤ 0,5 × froid ») se mesurera contre ces chiffres.
3. **numpy coûte environ 0,65 s** aux huit modules qui le chargent d'emblée (age, catalogue, corniere, flambement,
   garde_corps, plancher, profil, soudures) :
   - 276 à 303 ms d'attente après le déballage ;
   - puis 344 à 368 ms d'import à la première requête.
4. **L'import du module.**
   - Il varie beaucoup : assemblages 419 à 474 ms ; soudures 251 ; plancher 222 à 248 ; garde-corps 210 ;
     cornière et sol 145 à 146.
   - Les autres modules restent sous 100 ms, la poutre de 24 à 29 ms.
5. **Le garde-corps.** Son premier calcul prend 2 632 ms dans Pyodide, pour un premier rendu de 4,9 s en 2ᵉ visite ;
   une requête ordinaire ne prend ensuite que 196 ms. En CPython, sur la même machine, le premier « nouveau » prend
   1 388 ms et le second 98 ms (mesuré ici, après l'import du module : 104 ms). Ce coût de premier appel n'est pas un
   import, et sa cause n'a pas été cherchée (SUPPOSÉ : des tables ou des caches construits au premier calcul). Le
   préchauffage du lot S4 devra le couvrir.
6. **`registre.cles()`** coûte 7,8 à 10,1 ms à chaque requête dans Pyodide, ce qui confirme le défaut relevé par
   l'enquête (8 à 11 ms). Le registre paresseux du lot S4 le traite.
7. **Les réponses « nouveau » pèsent** de 16 ko (données béton) à 285 ko (garde-corps).
8. **Cohérence avec l'enquête.** Le rapport d'enquête « démarrage » a mesuré le même code, le même jour, sur la même
   machine, avec une instrumentation non gardée. Ses premiers rendus à froid au PC étaient : poutre 2 077, assemblages
   2 598, plancher 2 792, age 2 745 ms. Ceux-ci sont de 2 001, 2 493, 2 802 et 2 742 ms : l'écart reste dans ±4 %.

## 7. Ce qui reste SUPPOSÉ

- **Un vrai téléphone** serait 2 à 4 fois plus lent ; la mesure sur appareil réel s'affichera dans Paramètres
  (lot C).
- **Le réseau réel** : le serveur local ne compresse pas et répond sans délai. L'enquête a mesuré, à 20 Mbit/s et
  40 ms, 7,8 à 10,3 s pour la première visite.
- **La charge de la machine** (de 1,3 à 3,0) a pu élargir les écarts entre tirages ; les médianes en atténuent l'effet.
