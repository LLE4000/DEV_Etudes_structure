# Démarrage de l'établi — APRÈS le lot S4 (une application, un moteur) (10/11/2026)

Mesure « après » du lot S4 (`docs/site/REFONTE_ETABLI.md`, S4-D10), prise avec le script et le protocole du lot 0
(`site/web/outils/mesurer_demarrage.js`, `PLAN=quatre`, `REP=3`, profil Chromium neuf sur disque à chaque tirage,
médiane de 3), plus les deux scénarios que le lot rend possibles : le passage DANS l'application et l'ouverture depuis
l'accueil. Elle se lit contre la mesure du même jour, d'avant le lot : `2026-11-10_demarrage_passage.md`.

**Réponse.** Passer d'un module à l'autre DANS l'application ne redémarre plus Python : 0,11 à 0,86 s selon le
module, contre 1,8 à 2,8 s avant, et aucun rechargement de document sur 48 passages mesurés. L'interface est à
l'écran en ≈ 0,1 s dans tous les cas (plus de voile) ; un lien direct, un F5 ou la première visite de la session
démarrent encore Python, comme avant, mais montrent l'écran tout de suite et l'aperçu du calcul par défaut en moins de
0,3 s (contrôlé par le harnais, filtre « demarrage »).

## 1. Conditions

- Code `0c521f2` (site version `0b1b45c633132668`, paquet du moteur `68ac7eaef08c`, 270 fichiers, 1 710 ko), Pyodide
  0.29.5 ; même machine que la mesure d'avant (Xeon 2,1 GHz, 4 cœurs), Chromium de Playwright, serveur local
  `serveur_local.py 8771` sans compression. 120 relevés, 0 erreur ; charge moyenne sur une minute 1,1 à 2,0 (médiane
  1,55 ; avant : 2,06).
- Scénarios : « froid » (1re visite, profil neuf) ; « 2e visite » (même module, cache chaud) ; « lien direct » (vers le
  module suivant du cycle poutre → assemblages → plancher → age, par l'adresse : un document neuf) ; « dans
  l'application » (de l'établi précédent au module, par le volet des modules au PC, par l'accueil au téléphone) ;
  « depuis l'accueil » (lien Accueil, moteur prêt, puis la tuile du module suivant). Pour les deux derniers, le temps
  est compté DEPUIS LE CLIC ; le script compte les navigations de document (0 partout) et relit la sentinelle du
  document (gardée partout).
- Le téléphone est émulé (390 × 844, agent Android) sans ralentir le processeur, et le réseau est local. Un vrai
  téléphone ou une tablette en ligne ne sont PAS mesurés ici (voir § 4).

## 2. Temps jusqu'au premier calcul affiché (ms, médiane de 3)

| Module | Froid PC (avant → après) | 2ᵉ visite PC | Lien direct PC | **Dans l'application PC** | Depuis l'accueil PC |
|---|---|---|---|---|---|
| poutre | 2 546 → 2 286 | 1 894 → 1 643 | 1 851 → 1 562 | **141** | 138 |
| assemblages | 3 153 → 2 782 | 2 503 → 2 153 | 2 460 → 2 227 | **554** | 586 |
| plancher | 3 584 → 3 071 | 2 961 → 2 484 | 2 819 → 2 549 | **343** | 1 024 |
| age | 3 335 → 2 972 | 2 818 → 2 404 | 2 765 → 2 391 | **835** | 119 |

| Module | Froid tél. (avant → après) | 2ᵉ visite tél. | Lien direct tél. | **Dans l'application tél.** | Depuis l'accueil tél. |
|---|---|---|---|---|---|
| poutre | 2 542 → 2 186 | 1 958 → 1 704 | 1 841 → 1 610 | **111** | 109 |
| assemblages | 3 073 → 2 748 | 2 457 → 2 050 | 2 573 → 2 120 | **543** | 573 |
| plancher | 3 541 → 3 183 | 2 933 → 2 683 | 2 821 → 2 478 | **311** | 917 |
| age | 3 346 → 2 842 | 2 655 → 2 490 | 2 724 → 2 310 | **860** | 111 |

Avant le lot, « dans l'application » n'existait pas : tout passage était un lien direct (colonne « lien direct »,
avant).

- **Dans l'application**, ce qui reste est l'import du module la première fois dans la session (assemblages
  ≈ 0,45 s, plancher ≈ 0,25 s) et numpy la première fois qu'un module qui s'en sert est ouvert (≈ 0,4 s, une fois par
  session : age, plancher) ; le calcul lui-même est de 10 à 80 ms. Un module déjà ouvert dans la session revient en
  ≈ 0,1 s (poutre). Le préchauffage à l'approche d'une tuile fait cet import pendant le survol, avant le clic.
- **Premier affichage** (le « first contentful paint » d'un document neuf) : médiane 88 ms au PC, 52 ms au
  téléphone (au plus 132 ms). Avant, le voile couvrait l'établi jusqu'à la première réponse du moteur (1,8 à 3,6 s).
- **Lien direct, froid, 2ᵉ visite** : −8 à −18 % ; c'est le registre paresseux (`registre.module(clé)` ne lit plus
  les 54 fichiers de `api_web/`) et une machine moins chargée qu'au relevé d'avant. Python redémarre toujours dans un
  document neuf : c'est attendu (D8 : l'instantané mémoire de Pyodide reste hors périmètre).

## 3. Où va le temps d'un démarrage (2ᵉ visite, PC, ms, médianes)

| Module | loadPyodide | numpy | import du module | premier calcul |
|---|---|---|---|---|
| poutre | 1 299 | — | 68 | 14 |
| assemblages | 1 324 | — | 453 | 72 |
| plancher | 1 308 | 416 | 250 | 20 |
| age | 1 325 | 441 | 67 | 12 |

Comme avant, le démarrage de l'interpréteur (≈ 1,3 s en cache chaud) domine ; le lot ne le paie plus qu'une fois par
session au lieu d'une fois par écran. numpy part en parallèle du déballage du paquet pour le module de la page
(`annoncer`, contrôlé par le filtre « sonde » : numpy_debut avant deballage_fin).

## 4. Ce qui n'est pas mesuré ici, et la recette à faire à la main

L'ingénieur observait ≈ 10 s par module sur tablette en ligne. La structure du coût est la même qu'ici : le
démarrage de Python, qui ne se paie plus qu'une fois par session ; le passage dans l'application ne coûte plus que
l'import du module et le calcul — SUPPOSÉ de l'ordre de 0,3 à 2 s sur une tablette, à un facteur 2 à 4 du PC, au lieu
de ≈ 10 s. Safari ne se prouve pas dans Chromium ; recette sur iPad et iPhone (Safari), à faire et à noter ici :

1. Ouvrir l'accueil, attendre 5 s ; toucher « Poutres » : l'écran est là tout de suite, le bandeau « Le moteur de
   calcul démarre » disparaît à la première réponse.
2. Revenir à l'accueil (lien Accueil), toucher « Assemblages », puis « Charges sur plancher » : aucun bandeau, aucun
   écran blanc ; le calcul s'affiche en moins d'une seconde ou deux.
3. Retour arrière du navigateur (geste) deux fois : les écrans reviennent, au même endroit.
4. Mode avion, toucher un module déjà ouvert : il s'ouvre (copie gardée) ; un module jamais ouvert : le message
   « Pas de réseau : cet écran ne peut pas s'ouvrir sans connexion ; l'écran ouvert reste utilisable », l'écran
   courant calcule toujours.
5. Saisir une valeur puis toucher aussitôt Accueil : la carte de la note dit « enregistrement en cours », puis le
   nouveau taux.

Relevés bruts : `scratchpad` de la session (`mesure_apres.json`, 120 relevés), non versés ; le script les régénère.
