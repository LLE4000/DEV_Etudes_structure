# Données béton — l'aide-mémoire du bureau (module du site)

Demande de l'ingénieur (07/11/2026) : « une page où on a toutes les données pour le béton, facilement accessibles : le
tableau d'armatures, les longueurs d'ancrage, de recouvrement (il donnera ses tableaux plus tard)… ».

Le module **« Données béton »** (clé `donnees_beton`, établi `donnees-beton`, famille Béton de l'accueil) est une
**consultation** (statut « Consultation », niveau `i`, aucun verdict) : on choisit une rubrique, la toile montre son
tableau, le résultat la fiche de la ligne choisie et sa source. **Rien n'y est recopié** : chaque rubrique lit la source
qui fait foi dans le dépôt ; les seules grandeurs ajoutées sont des fonctions nommées, confrontées au tableau publié.

## 1 · Les fichiers

| Fichier | Rôle |
|---|---|
| `api_web/donnees_beton.py` | le module du site : état (des consultations), saisie, tableaux de la toile, fiche, fenêtre « d'où viennent les valeurs », sélecteur de la ligne. Il MET EN FORME, il ne calcule rien |
| `beton/donnees/ec2.py` | les grandeurs de l'EN 1992-1-1 ajoutées : f_cm, f_ctm, f_ctk,0,05, f_ctk,0,95, E_cm (tableau 3.1), E_s, γ_s = 1,15, ε_yd — des fonctions et des constantes NOMMÉES, avec leur source |
| `beton/donnees/consultation.py` | les rubriques « classes » et « aciers » : chaque valeur lue à sa source (fichier, moteur des poutres, `ec2.py`) |
| `beton/donnees/ancrages.py` | la lecture et le contrôle du fichier des ancrages et recouvrements (aucune valeur) |
| `beton/donnees/ancrages_recouvrements.json` | **le fichier à remplir** par le bureau : vide (« Tableaux du bureau à venir ») |
| `site/web/css/modules/donnees_beton.css` | la feuille du module (tableaux à en-têtes figés, fiche, verdict) |
| `site/maquettes/construire.py` | la clé `donnees_beton` ajoutée à la famille Béton de `FAMILLES`, et son pictogramme (`MOD`) |
| `tests/test_donnees_beton_moteur.py` | le moteur face au tableau 3.1 publié, au fichier des classes, aux formules ; le fichier des ancrages ; mutations |
| `tests/test_api_web_donnees_beton.py` | ce que l'écran affiche = la source, lue SANS le module ; formules réévaluées ; refus ; échappement ; mutations |

## 2 · Les rubriques et leur source unique

| Rubrique | Ce que la toile montre | La source (unique) |
|---|---|---|
| **Classes de béton** | les classes du bureau, colonnes « EN 1992-1-1 » (f_ck, f_ck,cube, f_cm, f_ctm, f_ctk,0,05, f_ctk,0,95, E_cm) ou « Bureau » (α_b, μ_B500, μ_B400, τ_adm,I/II/IV) | `beton_classes.json` lu par `beton/poutre/entree.classes_beton` (C20/25 à C40/50) ; tableau 3.1 : `beton/donnees/ec2.py` ; τ_adm = k·f_ck,cube / 1,05 : `beton/poutre/moteur.materiaux` (k et 1,05 : `BANDES_TAU`, `GAMMA_TAU`) |
| **Aciers d'armature** | B400, B500 : f_yk ; f_yd du bureau (γ_s = 1,5) ; f_yd de l'EN 1992-1-1 (γ_s = 1,15) ; E_s ; ε_yd | nuances : `beton/poutre/entree.NUANCES` ; γ_s du bureau : `beton/poutre/entree.DEFAUTS_POUTRE` ; f_yd = f_yk/γ_s : `beton/poutre/moteur.materiaux` (pour les deux γ_s) ; E_s, γ_s = 1,15, ε_yd : `ec2.py` ; ρ = 7 860 kg/m³ : `beton/armatures/tableau.MASSE_VOLUMIQUE` |
| **Tableau des armatures** | le tableau du bureau : n = 1 à 10 barres (ou par mètre, e = 100/n), la masse, les espacements fixes 15 · 30 · 35 · 40 cm ; le dessin des diamètres à l'échelle | `beton/armatures/tableau.tableau()` (l'image « Tableau armature.png », relue au chiffre près par `tests/test_armatures_moteur.py`) ; le dessin : `beton/armatures/dessins.diametres` |
| **Enrobages · Durabilité** | c_min,dur (classe structurale × exposition), la classe proposée (●) ; c_nom de la case choisie ; la coupe d'enrobage | `beton/enrobage/donnees.py` (C_MIN_DUR, CLASSE_STRUCTURALE_DEFAULT, DELTAS_DEV, EXPOSITION_INFO) ; c_nom et les avertissements : `beton/enrobage/moteur.calculer` (une dalle sans étrier) ; la coupe : `api_web/dessin_enrobage.coupe` |
| **Enrobages · Feu** | la distance à l'axe a : dalle (5.8), voile (5.4), poutre (5.5) et poteau (5.2a) avec b_min ; les cases à vérifier en rouge | `beton/enrobage/donnees.py` (FEU_*) ; les cases rouges : les avertissements `tableau_poteau`, `tableau_voile` du moteur de l'enrobage |
| **Ancrages et recouvrements** | « Tableaux du bureau à venir » — puis, une fois le fichier rempli, chaque tableau tel quel | `beton/donnees/ancrages_recouvrements.json` (§ 6) |

Le cadre **Données** choisit la rubrique (menu) et ses choix (segments et menus) ; sa section 3 nomme les sources. Le
**Résultat** : la valeur clé en grand (f_ck, f_yk, a, c_nom, R), la ligne ouverte (elle ouvre le sélecteur), la
ligne « D'où viennent les valeurs » (elle ouvre la fenêtre : chaque valeur avec sa formule et ses valeurs substituées,
ou son fichier), les messages, la fiche de la ligne choisie. Une note porte plusieurs **consultations** (ajouter,
copier, supprimer) : une par question qu'on se pose.

## 3 · Le tableau 3.1 de l'EN 1992-1-1 (`beton/donnees/ec2.py`)

    f_cm = f_ck + 8 ; f_ctm = 0,30·f_ck^(2/3) (≤ C50/60), 2,12·ln(1 + f_cm/10) au-delà ;
    f_ctk,0,05 = 0,7·f_ctm ; f_ctk,0,95 = 1,3·f_ctm ; E_cm = 22·(f_cm/10)^0,3 GPa

**La preuve** : `tests/test_donnees_beton_moteur.py` transcrit le tableau 3.1 imprimé (14 classes, C12/15 à
C90/105) et vérifie que les fonctions, arrondies comme lui, redonnent ses 70 cases. **Un seul écart, celui du tableau
imprimé** : f_ctk,0,05 de C60/75 vaut 0,7 × 4,355 = 3,05 MPa (3,0 arrondi), le tableau imprime 3,1 (0,7 × 4,4, le f_ctm
déjà arrondi). Il est déclaré et figé : tout autre écart rougit. Hors des classes du bureau (C20/25 à C40/50).

Le site AFFICHE ces valeurs arrondies comme le tableau publié (f_ctm au 0,1 MPa, E_cm au GPa) ; la fenêtre donne la
valeur exacte (2,896 MPa, 32,84 GPa) avec sa formule. Le f_ctm de la méthode du bureau (`beton/poutre/moteur`,
`K_FCTM = 0,30`) est celui de `ec2.fctm` au bit près pour chaque classe du bureau : les deux écritures sont liées par
le test (R2 : une copie tenue est une vue).

## 4 · Les aciers : d'où viennent les valeurs

- **f_yk** : les nuances des modules béton, B400 et B500 (`beton/poutre/entree.NUANCES`, « Qualité acier (B) »).
- **γ_s = 1,5** : le « coefficient acier ELS » des modules Poutre, Dalle et Prédalle (méthode ancienne du bureau,
  réglable de 1,0 à 2,0 — l'aide de l'écran Streamlit cite déjà 1,15 en exemple).
- **γ_s = 1,15, E_s = 200 GPa** : ABSENTS du dépôt pour l'acier d'armature avant ce module (l'acier de construction a
  `E_ACIER = 210 000` dans les pieds de poteau, ce n'est pas le même). Ils sont écrits UNE fois dans `ec2.py`, avec
  leur source (tableau 2.1N, §3.2.7(4)). γ_s = 1,15 est la valeur recommandée : **l'annexe nationale belge est à
  confirmer par le bureau** (le site le dit dans le résultat).
- **f_yd = f_yk / γ_s** n'est pas refait : c'est `beton/poutre/moteur.materiaux`, appelé avec chacun des deux γ_s.

## 5 · Les enrobages : ce que le site relaie

Les tables sont celles de l'écran Enrobage, avec ses écarts à la norme relevés à l'extraction
(`docs/beton/ENROBAGE_MOTEUR.md` § 6) — **reproduits, et dits** :
- D4 (colonnes X0 et XC1, 5 mm de plus que le tableau 4.4N, côté sûr) : au pied du tableau de durabilité ;
- D3 (classe proposée sous S4) : la consultation s'ouvre en **S4** (« référence 50 ans ») ; la classe proposée par
  l'écran est marquée ● et nommée dans la fiche ; sous S4, l'avertissement du moteur s'affiche ;
- D1 (poteaux, tableau 5.2a) et D2 (voiles R180, R240) : les cases sont **en rouge** avec « ! », et l'avertissement
  du moteur s'affiche en rouge — ce sont ses messages, pas une copie.

## 6 · Ancrages et recouvrements : le fichier à remplir

Le fichier `beton/donnees/ancrages_recouvrements.json` est **vide** : aucune valeur n'est proposée tant que
l'ingénieur n'a pas fourni ses tableaux. Il suffit d'y verser un tableau pour qu'il apparaisse sur le site (aucune
ligne de code), puis de reconstruire le site (`python3 site/web/construire.py`) et de relancer
`tests/test_donnees_beton_moteur.py`.

Un tableau (exemple de **forme** : les cases `null` sont des cases vides, à remplacer par les valeurs du bureau) :

```json
{"cle": "ancrage_droit_b500",
 "titre": "Ancrage droit · B500 · bonne adhérence",
 "grandeur": "l_bd",
 "unite": "cm",
 "source": "Tableau du bureau, révision du …",
 "conditions": ["Bonne adhérence", "Barres droites"],
 "colonnes": {"titre": "Ø", "unite": "mm", "valeurs": [8, 10, 12, 16, 20, 25]},
 "lignes": [{"libelle": "C25/30", "valeurs": [null, null, null, null, null, null]},
            {"libelle": "C30/37", "valeurs": [null, null, null, null, null, null]}],
 "notes": []}
```

- `cle` : lettres, chiffres, « _ », « - » (40 au plus), unique dans le fichier ;
- `grandeur` : le symbole, indices après « _ » (l_bd, l_0, l_b,rqd) ; `unite` : « mm », « cm », « m » ou « Ø » (une
  longueur exprimée en diamètres) ;
- `source` est **obligatoire** : d'où vient le tableau ;
- chaque ligne a **une valeur par colonne** : un nombre, ou `null` (case vide, affichée « — ») ;
- `conditions` et `notes` : facultatives (textes).
- deux lignes d'un même tableau ont des **libellés différents** : l'écran désigne une ligne par son libellé (le
  sélecteur du résultat) ; deux libellés égaux sont refusés.

Une forme fautive (ligne trop courte, valeur qui n'est pas un nombre, unité inconnue, source manquante, clé inconnue
ou en double, libellé en double, version inconnue) est refusée par `ancrages.valider` avec l'endroit fautif ; le site affiche alors
l'erreur **en rouge** — dans la toile, le cadre Données, le verdict et la liste de la note (« Fichier des ancrages
illisible », avec l'endroit fautif) —, jamais un tableau tronqué ni « à venir ». Une note qui visait un tableau retiré du fichier montre le premier.

## 7 · Constats (écrits ici, rien n'est corrigé en silence)

1. **`tau_lim` de `beton_classes.json` n'est lu par aucun moteur** (0,60 à 0,85 N/mm²) : les τ_adm des modules Poutre,
   Dalle et Prédalle sont calculés depuis f_ck,cube (k·f_ck,cube/1,05 ; C30/37 : 0,56 · 1,13 · 2,26 N/mm²). Le site
   affiche la valeur du fichier avec ce message. À trancher par le bureau : la retirer, ou dire ce qu'elle est.
2. **E_s et le γ_s de l'EN 1992-1-1 manquaient** pour l'acier d'armature (§ 4).
3. **ρ = 7 860 kg/m³** dans le tableau des armatures du bureau (l'usage est 7 850 ; les masses sont plus fortes de
   0,13 %) — déjà écrit dans `beton/armatures/tableau.py`, rappelé dans la rubrique Aciers.

## 8 · Besoins communs (signalés, non bricolés)

- Un **clic sur une ligne d'un tableau de la toile** ne choisit rien : l'établi n'écoute les `data-chemin` que dans
  le cadre Données. La ligne se choisit dans les Données ou par le sélecteur du résultat.
- `C.ligne_res` **sans jauge** et `C.fenetre_calcul` **sans tableau de vérifications** : retirés ici comme le fait le
  catalogue des profilés (déjà signalé).
- Pas de **téléchargement** du tableau (CSV) : il viendrait par `C.bouton_telecharger` si le bureau le demande.

## 9 · La relecture adverse du 07/11/2026 (corrigé, figé par `tests/test_api_web_donnees_beton.py` § 10)

Chaque défaut a été montré avant sa correction (le contrôle rougissait sur le code d'origine) et chaque correction a sa
mutation (la retirer fait rougir le contrôle).

1. **Non conservatif — un booléen lu comme un choix** : `False == 0` en Python ; `"delta_dev": false` d'un fichier
   .stab abîmé devenait Δc_dev = 0 mm (c_nom de 30 à 20 mm), sans un mot ; `"ligne": true` ouvrait la ligne 2. Un
   booléen n'est plus jamais un choix (relu ou saisi).
2. **La fiche et le verdict du feu** écrivaient sans réserve les distances que le moteur de l'enrobage dit « à
   vérifier dans la norme » (poteau, voile R180 et R240) — la toile seule les montrait en rouge, et au téléphone on lit
   le résultat. Elles sont désormais en rouge, marquées « ! », dans la fiche, le verdict, la phrase et le sélecteur.
3. **Le fichier des ancrages illisible** : la toile et le cadre Données disaient « à venir… aucune valeur proposée en
   attendant », comme si rien n'avait été versé (le résultat seul le disait) ; la liste de la note disait « à venir »
   même sous un tableau affiché. Tout dit maintenant « illisible » avec l'erreur, ou le tableau et la ligne ouverts.
4. **Deux lignes de même libellé** dans un tableau d'ancrage : la seconde ne se choisissait jamais. Refusé (§ 6).
5. **Les noms** : sans borne (un nom d'un million de caractères rendait une réponse de 4 Mo), une liste devenait
   « [1, 2] », U+202E retournait l'affichage de la liste. Un nom est un texte de 80 caractères au plus, sans caractère
   invisible ni de commande.
6. **Les requêtes abîmées** : une rubrique en liste rendait `TypeError : unhashable type` ; un index illisible
   `invalid literal for int()` ; « supprimer -5 » supprimait la PREMIÈRE consultation (l'index était ramené dans la
   liste). Refus en français, rien n'est ouvert ni supprimé au hasard.
7. **Les notes abîmées relues en silence** (un état qui n'est pas un objet, « consultations » vide mais pas une liste,
   une clé inconnue, un tableau d'ancrage qui n'est pas un texte, une ligne négative) : erreur claire, comme le contrat.
8. **Lisibilité** : « Δc_dev = » en fin de ligne et « 10 mm » à la suivante dans les verdicts (le « = » est lié par des
   espaces insécables) ; « EN 1992- » / « 1-1 » dans la légende des vues au téléphone (la légende est du texte : un
   gluon U+2060 suit chaque trait d'union) ; « f_yd », « E_s » à souligné dans un sous-titre.
9. **Le jargon de développeur à l'écran** : les sources se nommaient par des chemins de code
   (`beton/poutre/moteur.materiaux`, `C_MIN_DUR, CLASSE_STRUCTURALE_DEFAULT`, `tests/test_…py`). Elles se disent
   maintenant dans les mots du bureau (« le calcul du module Poutre », « les tableaux de l'écran Enrobage ») ; restent
   nommés les fichiers que le bureau ouvre lui-même (`beton_classes.json`, le fichier des ancrages à remplir) et la
   page de documentation de son format. Ce document-ci garde les chemins (§ 1, § 2).

## 10 · Les contrôles

```bash
python3 tests/test_donnees_beton_moteur.py          # tableau 3.1, fichier des classes, aciers, ancrages, mutations
python3 tests/test_api_web_donnees_beton.py         # écran = source, formules, refus, échappement, mutations
python3 tests/test_api_web_modules.py donnees_beton # le contrat commun des modules
python3 tests/test_site_web.py                      # le paquet du moteur, la famille de l'accueil
node site/web/verifier_site.js http://127.0.0.1:<port> <dossier> module:donnees_beton
node site/web/verifier_site.js http://127.0.0.1:<port> <dossier> formats:donnees_beton
```
