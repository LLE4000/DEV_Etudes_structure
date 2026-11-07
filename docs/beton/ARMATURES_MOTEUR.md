# Tableau des armatures — le moteur sans Streamlit

*Livré le 07/11/2026. Moteur `beton/armatures/` : bibliothèque standard seule (`math`), Python 3.12 et 3.13 (donc
Pyodide 0.29.5). Il recalcule le tableau que l'écran Streamlit `modules/tableau_armatures.py` affiche — une IMAGE,
« Tableau armature.png » — et le met au service d'une question (une section requise). L'écran Streamlit reste la
**référence** ; la preuve de parité est `tests/test_armatures_moteur.py` ; le module du site est
`api_web/armatures.py` (`MODULE = "armatures"`, établi `tableau-armatures`).*

> **Défauts NON CONSERVATIFS de l'écran Streamlit (l'image), légers — reproduits à l'affichage, jamais dans les
> comparaisons du site :**
> 1. **Les sections sont arrondies au mm² le plus proche** : 84 des 168 cases de section sont arrondies VERS LE HAUT
>    (de 0,5 mm² au plus). En relatif, c'est au plus **+0,8 %** (2Ø6 = 56,55 mm² écrit 57 ; 1Ø10 = 78,54 écrit 79,
>    +0,6 % ; 2Ø8 = 100,53 écrit 101, +0,5 %) ; au-delà de 150 mm², moins de 0,3 %. Un projeteur qui retient 1Ø40
>    (« 1 257 ») pour 1 257 mm² requis a 1 256,6 mm². Le site compare toujours la section EXACTE au besoin : 1Ø40 ne
>    suffit pas pour 1 257 mm², et l'écran écrit alors « 1 256,6 mm² < 1 257 mm² » (jamais « 1 257 < 1 257 »).
> 2. **L'en-tête 100/n à une décimale** : poser les barres à 16,7 cm (colonne 6) donne 0,2 % de section de moins que
>    la case, à 14,3 cm (colonne 7) 0,1 % de moins (§ 7, point 2).

## 1 · Ce qui est livré

| Pièce | Rôle |
|---|---|
| `beton/armatures/tableau.py` | les lignes et colonnes de l'image (12 Ø ; 1 à 10 barres ; 15, 30, 35, 40 cm), ses formules (π·Ø²/4, ×n, ×100/e, ρ = 7 860 kg/m³), son arrondi ; `tableau()` rend le tableau complet, valeurs exactes et valeurs écrites |
| `beton/armatures/moteur.py` | un **besoin** (section requise, critères), la **disposition retenue** (section, masse, entraxe, formules), la **plus économique**, les **candidats** triés (le sélecteur), le **tableau marqué** — et le **format complet** (entrée, sortie) dans sa docstring |
| `beton/armatures/dessins.py` | deux dessins SVG à l'échelle (mm) : un mètre de nappe (vue en plan et coupe A-A, cotes), les douze diamètres (échelle 2:1, section et masse de chacun) |
| `api_web/armatures.py` | le module du site (repères de la note, saisie, résultat, fenêtres de calcul, sélecteur, vue Tableau) |
| `site/web/css/modules/armatures.css` | la feuille propre à la vue Tableau (en-têtes et colonne des Ø figés, surlignages) |
| `tests/test_armatures_moteur.py` | parité avec l'écran et l'image, mutations, contrôles à la main, recherche, normalisation et bornes, format, dessins, pureté (120 contrôles, 1,6 s) |
| `tests/test_api_web_armatures.py` | l'écran du site : la vue Tableau case par case, résultat, fenêtres, saisies et refus, sélecteur, repères, échappement, comparaisons honnêtes, seize dispositions relues sur l'image (106 contrôles) |

## 2 · L'API

```python
from beton.armatures import moteur as M, tableau as T

T.tableau() -> dict                     # le tableau de l'image : valeurs exactes et écrites
T.aire(d), T.section(n, d), T.section_par_metre(d, e), T.masse_lineique(d), T.espacement(n), T.arrondi(x)
M.nouveau_besoin(nom="Repère 1") -> dict
M.normaliser(besoin) -> dict            # complété et contrôlé ; ValueError claire hors du domaine
M.disposition(besoin) -> dict           # la retenue (besoin normalisé)
M.candidats(besoin, avec_retenue=False) -> list    # les cases dans les critères, triées
M.optimale(besoin) -> dict | None       # la plus économique qui suffit
M.calculer(besoin) -> dict              # tout : retenue, plus économique, tableau marqué, avertissements
```

**Entrée** (une clé absente prend son défaut ; une clé inconnue est refusée) :

```json
{"nom": "Repère 1", "mode": "barres", "As_req": 400.0, "d": 12, "n": 4, "espacement": "e15", "e_libre": 15.0,
 "d_min": 6, "d_max": 40, "n_min": 1, "n_max": 10, "e_min": 10.0, "e_max": 40.0}
```

`mode` : `barres` (une section en mm², n barres) ou `nappe` (une section par mètre en mm²/m, à espacement constant).
`espacement` : une colonne du tableau — `n1` … `n10` (n barres par mètre, e = 100/n cm), `e15`, `e30`, `e35`, `e40`
(les « espacements fixes donnés ») — ou `libre` (`e_libre`, en cm). `As_req = 0` : pas de recherche, le tableau seul.

Bornes (refus avec un message, le repère reste intact) : `0 ≤ As_req ≤ 1 000 000` (mm² ou mm²/m ; « -0 » est relu 0) ;
`1 ≤ n ≤ 200` ; `0 < e_libre ≤ 100 cm` (une barre par mètre au moins : au-delà, le mètre de nappe dessiné n'avait plus
aucune barre) ; nom d'un repère de 120 caractères au plus. Un espacement libre s'écrit au centième (« Ø12 · 16,75 cm » :
une décimale l'aurait écrit 16,8, une autre nappe) ; une colonne garde l'écriture de son en-tête (« 16,7 »).

**Sortie** de `calculer` (JSON strict) : `besoin`, `disposition` (`nom`, `As`, `As_arrondi`, `unite`, `masse` et
`unite_masse` — kg/m pour des barres, kg/m² pour une nappe —, `e`, `jour` = e − Ø, `barres_par_metre`, `aire`,
`masse_barre`, `colonne`, `dans_tableau`, `dans_criteres`, `taux` = A_s,req / A_s, `suffit`, `formules`), `optimale`
(même forme, ou `null`), `suffisantes` (nombre de cases qui suffisent dans les critères), `tableau` (chaque case :
valeur exacte, valeur écrite, utile au mode, dans les critères, suffit, optimale, retenue), `avertissements`
(`insuffisante`, `hors_tableau`, `chevauchement`, `aucune`).

## 3 · La méthode (celle de l'image)

L'image porte, pour douze diamètres Ø6 à Ø40 :

| Colonnes | Grandeur | Formule | Arrondi |
|---|---|---|---|
| 1 … 10 | n barres (mm²) — **ou** n barres par mètre, e = 100/n cm (mm²/m) ; l'en-tête écrit 100,0 · 50,0 · 33,3 … 10,0 | A = n·π·Ø²/4 | au mm² |
| « poids kg/m » | masse d'un mètre de barre | m = ρ·π·Ø²/4·10⁻⁶, ρ = 7 860 kg/m³ (écrit dans l'en-tête) | 3 décimales |
| « espacement fixe donné [cm] » 15, 30, 35, 40 | section par mètre (mm²/m) | A/m = π·Ø²/4 · 100/e | au mm² |

Le moteur ajoute, avec les mêmes formules :

- **la disposition retenue** : Ø et n barres (section, masse n·m en kg/m, et la lecture « par mètre » de la même
  colonne : e = 100/n), ou Ø et un espacement (section par mètre, barres par mètre 100/e, masse de la nappe en kg/m²,
  jour entre barres e − Ø). Une colonne « n barres par mètre » se calcule n·a, comme l'image (et non a·100/e : 100/n
  n'est écrit qu'à une décimale) ; un nombre de barres au-delà de 10, ou un espacement libre, sortent du tableau :
  même formule, **signalés** (« valeur nouvelle sur le site — à valider par le bureau ») ;
- **la plus économique** pour une section requise : la plus petite section ≥ A_s,req parmi les cases du tableau dans
  les critères (Ø mini et maxi ; nombre de barres de 1 à 10, ou espacement mini et maxi). La masse étant
  proportionnelle à la section (même ρ), c'est aussi la plus légère. À section égale (2Ø16 et 8Ø8 : n·Ø² = 512 ;
  Ø16 à 40 cm et Ø8 à 10 cm), le moins de barres, puis le plus petit Ø ; la section est comparée au millionième de
  mm² pour qu'un dernier bit de flottant ne départage pas deux sections égales ;
- **le taux** A_s,req / A_s, coloré comme partout (`C.niveau`) : ce n'est pas une vérification de la note (le module
  est une **consultation**, statut « Consultation »), c'est la lecture du besoin.

## 4 · D'où vient chaque grandeur

| Grandeur | Source |
|---|---|
| les 12 Ø, les 10 nombres de barres, les 4 espacements fixes | l'image (lignes et en-tête), `tableau.DIAMETRES`, `NOMBRES`, `ESPACEMENTS_FIXES` |
| ρ = 7 860 kg/m³ | l'en-tête de l'image (« 7860 kg/m³ ») — l'usage est 7 850 : voir § 7 |
| 100/n écrit à une décimale | la seconde ligne de l'en-tête de l'image |
| arrondi au mm² (la moitié vers le haut), masse à trois décimales | l'image ; aucune valeur n'est à moins de 0,0033 mm² d'une moitié : le sens de l'arrondi des moitiés est sans effet (contrôlé) |
| la légende « Tableau des sections d’armatures en mm² et espacement fixe (cm) » | `st.image(…, caption=…)` de l'écran, `tableau.LEGENDE_IMAGE`, reprise comme légende de la vue Tableau |
| la plus économique, les critères, les dispositions hors du tableau, le jour entre barres | **nouveau sur le site**, mêmes formules ; les dispositions hors du tableau sont signalées à l'écran |

## 5 · La preuve de parité

L'écran Streamlit ne calcule rien : la preuve porte donc sur ce qu'il MONTRE. `tests/test_armatures_moteur.py` :

1. **L'écran** est exécuté avec un faux `st` qui enregistre au lieu d'afficher : un titre (« 📊 Tableau des sections
   d’armatures »), le bouton « 🏠 Accueil », UNE image — ce fichier, cette légende — et aucun champ de saisie. Il est
   exécuté aussi pour de vrai (`AppTest`), et les **octets que Streamlit sert** sont captés : il recode le PNG en
   JPEG (qualité 90). Le PNG est épinglé par son empreinte SHA-256 (`886dd825…`) : si le bureau remplace l'image, le
   test rougit et demande de la retranscrire.
2. **L'image est relue dans ses pixels** — le PNG du dépôt ET le JPEG servi : la grille est retrouvée par ses traits
   (12 rangées × 17 colonnes), chaque case est découpée en glyphes par projection, chaque glyphe est comparé aux
   modèles moyens des dix chiffres (les chiffres gras de la colonne des masses ont leurs modèles) : les **669
   glyphes** du corps sont lus comme la transcription écrite dans le test (pire rapport des distances au bon chiffre
   et au suivant : 0,33 pour le PNG, 0,34 pour le JPEG), et l'en-tête (1 … 10 ; 100,0 … 10,0 ; 7860 ; 15 30 35 40)
   aussi. Cinq **mutations** de la transcription (un seul chiffre faux : 1 018 → 1 019, 2,469 → 2,496, 3 142 → 3 143,
   28 → 29, 16,7 → 16,6) sont toutes dénoncées.
3. **Le moteur = la transcription**, au chiffre près : 120 sections de n barres, 12 masses, 48 sections par mètre, et
   l'en-tête ; le tableau marqué de `calculer` et chacune des 288 dispositions servies au sélecteur (120 en barres,
   168 en nappe) portent la valeur de leur case. Neuf **mutations du moteur** font toutes diverger des cases :
   ρ = 7 850 (10 écarts), π = 3,14 (97), troncature (84), arrondi supérieur (84), espacement fixe 45 cm (13), bande de
   1 000 cm (49), masse à deux décimales (12), Ø50 ajouté, aire π·Ø² (180). La mutation témoin — arrondir les
   moitiés au pair — ne change rien (aucune moitié dans le tableau).

Puis : sept contrôles **à la main** (36·π, 144·π, 25·π·100/15, 7 860 × 36·π × 10⁻⁶, 4 000·π, 100/3, 64·π·100/12) ;
la **plus économique** (ex æquo, critères, aucune, section requise égale à une case) et **600 besoins tirés au sort**
comparés à une recherche exhaustive écrite à part ; la **disposition** (barres, nappe, colonne « n barres par mètre »,
espacement libre retombant sur une colonne, hors du tableau, barres qui se chevauchent) ; 25 **refus** de
normalisation ; le **format** JSON strict ; les **dessins** (barres de la coupe à leur abscisse et à leur rayon, cotes,
échelle 2:1 des diamètres, textes échappés) ; la **pureté** : un sous-processus dont `sys.meta_path` refuse streamlit,
numpy, pandas, PIL, matplotlib, scipy, reportlab et fitz charge le moteur et ses dessins, calcule, sous Python 3.12
et 3.13, avec des résultats identiques au bit près.

`tests/test_api_web_armatures.py` relit la transcription dans le fichier précédent (source unique) et vérifie que la
**vue Tableau du site** porte, case par case, les 180 nombres de l'image et son en-tête.

## 6 · Ce que le site couvre, et ce qu'il ne couvre pas

| Écran Streamlit | Site |
|---|---|
| titre « 📊 Tableau des sections d’armatures » | le titre de la vue Tableau (`tableau.TITRE_ECRAN`, lié au titre de l'écran par le test de parité) |
| bouton « 🏠 Accueil » | la barre de navigation du site |
| l'image du tableau (180 nombres, en-tête) | la vue **Tableau** : les mêmes nombres, en texte (lisibles, copiables, imprimables), en-têtes et colonne des Ø figés, valeur exacte et taux au survol |
| la légende de l'image | écrite en tête du pied du tableau, visible (le cadre de la toile n'écrit, pour la première vue, que la phrase du résultat) |
| — | une **section requise** (mm² ou mm²/m) surligne les cases qui suffisent et propose la **plus économique** (★) ; un bouton la retient |
| — | une **disposition** (Ø et n barres, ou Ø et espacement, y compris libre) : section, masse, entraxe, jour, avec la fenêtre de calcul (formules substituées) et le sélecteur trié (✓ retenue, ★ plus économique) |
| — | des **critères** de recherche (Ø mini et maxi, nombre de barres, espacements) |
| — | deux dessins à l'échelle : un mètre de nappe (plan et coupe), les douze diamètres |
| — | plusieurs **repères** dans une note (une nappe de dalle, les barres d'une poutre…) |

Non couvert : la mise en relief du « 5 » de l'en-tête de l'image (en gras, sans explication : sans doute
l'espacement courant de 20 cm) ; les couleurs de l'image (bandes orangées) — le site a les siennes ; la note PDF (hors
périmètre du site, bouton grisé). Rien de ce que l'écran Streamlit offre ne manque.

## 7 · Défauts constatés dans l'écran Streamlit (non corrigés : le site reproduit et signale)

1. **ρ = 7 860 kg/m³** au lieu de 7 850 kg/m³ (EN 1991-1-1, tableau A.4 ; l'usage des tableaux de fournisseurs) : les
   masses de l'image sont plus fortes de 0,13 %. Côté sûr pour une estimation de poids, sans effet sur les sections.
   Le site garde 7 860 (parité) et l'écrit (pied du résultat, section 4 des données, fenêtre de calcul).
2. **L'en-tête 100/n à une décimale** : les colonnes 6 et 7 se lisent « 16,7 cm » et « 14,3 cm », mais leur section
   est celle de 6 et 7 barres par mètre (e = 16,67 et 14,29 cm). Placer réellement les barres à 16,7 cm donne 0,2 %
   de moins que la case (Ø20 : 1 881 au lieu de 1 885 mm²/m), à 14,3 cm 0,1 % de moins : **légèrement non
   conservatif**, négligeable. Le site écrit la même chose, mais calcule un espacement libre à sa vraie valeur.
3. **Une seule colonne pour deux lectures** : « n barres » (mm²) et « n barres par mètre » (mm²/m) partagent les
   colonnes 1 à 10, sans unité écrite ; les sections par mètre des « espacements fixes » n'ont pas d'unité non plus.
   Le site écrit les deux unités et la lecture du mode choisi.
4. **« poids kg/m »** : c'est une masse (kg/m), pas un poids (kN/m). Le site écrit « Masse ».
5. **Les espacements fixes** se limitent à 15, 30, 35 et 40 cm ; 12, 17,5, 22,5 cm… manquent (20 et 25 cm se lisent
   dans les colonnes 5 et 4). Le site calcule tout espacement, et le signale hors du tableau.
6. **L'image est servie recodée en JPEG** (qualité 90 : Streamlit convertit une image sans transparence) : de légers
   artefacts autour des chiffres ; la lecture des pixels du JPEG servi donne les mêmes 669 chiffres.
7. **Les sections arrondies au plus proche** (en tête de ce document) : la moitié des cases surestime la section,
   de 0,8 % au plus. Le site écrit les mêmes nombres dans le tableau, mais compare la section exacte.
8. `st.image(…, use_container_width=True)` est **déprécié** (Streamlit 1.66 : « will be removed after 2025-12-31 ») ;
   l'écran affichera un avertissement, puis une erreur, à la prochaine mise à jour de Streamlit.

Aucune saisie : rien n'est « ramené en silence » dans l'écran. Sur le site, toute saisie hors domaine est refusée
avec un message (section requise négative, Ø hors du tableau, nombre de barres non entier ou hors de 1 à 200,
espacement nul ou supérieur à 100 cm, section requise au-delà de 1 000 000, nom de plus de 120 caractères, critères
mini > maxi) et le repère reste intact.

## 8 · Relecture indépendante du 07/11/2026

Ce que la relecture a trouvé et corrigé (chaque correction est figée par un contrôle qui rougit si on la retire) :

- **« 1 257 mm² < 1 257 mm² requis »** : la section retenue était écrite au mm² face au besoin, même quand l'arrondi
  renversait la comparaison (1Ø40 = 1 256,64 mm²). `moteur.section_txt` l'écrit au dixième (ou plus) dans ce cas : en-tête,
  ligne « Retenue », fenêtre de calcul (limite et η), message, propriétés du sélecteur.
- **Section requise sans espace des milliers** (« A_s,req = 1257 mm² » à côté de « 1 257 ») : écrite comme le tableau.
- **Saisies absurdes acceptées** : A_s,req = 1e300 donnait un taux de 300 chiffres ; « -0 » s'affichait « -0 » ;
  un espacement libre de 2,5 m laissait le mètre de nappe sans barre ; un nom sans limite. Bornes ajoutées (§ 2) ; un
  taux au-delà de 999 % s'écrit « > 999 % ».
- **Espacement libre arrondi dans son nom** (16,75 cm écrit « 16,8 ») : écrit au centième.
- **La légende de l'image n'était visible nulle part** (le cadre de la toile remplace la légende de la première vue par
  la phrase du résultat) : elle ouvre le pied du tableau ; le titre de la vue est celui de l'écran Streamlit.
- **Tablette (1 024 px)** : « ★ Ø8 · 12,5 cm · 402 mm²/m » était coupé derrière « Plus économique » (167 px pour 184) :
  la ligne s'appelle « Économique » ; la légende du tableau sortait de la feuille : elle passe à la ligne ; la grande
  pastille de l'en-tête se laissait écraser par un nom long : elle garde sa largeur, le nom passe à la ligne.
- **Dessins** : « COUPE A-A » était barré par l'axe de la première barre ; les textes du mètre de nappe étaient
  illisibles au téléphone (6 px) : agrandis ; « échelle 2:1 » des diamètres était faux à l'écran (le dessin est
  redimensionné) : le trait de 20 mm fait foi, « tous à la même échelle ».
- Unités en capitales dans les titres de groupe (« E = 15 CM ») : retirées ; masse des barres écrite au gramme partout.

Chiffres : 16 dispositions (barres et nappe, coins du tableau, colonnes 100/n, espacements fixes, espacement libre
retombant sur une colonne) et les 12 rangées de la vue Tableau ont été relues **sur l'image elle-même**, à l'œil, et
comparées à ce que le site affiche (routeur, fragments HTML) : 77 contrôles, aucun écart.
