# Projet d'assemblages métalliques — organisation et règles

*Plan validé le 30/10/2026, livré par lots. Lot 1 : page Projet, liste,
création, menu ⋯, sélecteur, fichier projet. Lot 2 : rapport (page de garde +
notes assemblées). Lot 3 : tableau récapitulatif, « créer à partir de »,
verrou 🔒.*

## 1 · Le principe

Le module « Assemblages métalliques » fonctionne comme Poutre et Dalle : **un
fichier = un projet**, qui contient plusieurs assemblages. La page du module
est la page PROJET ; l'écran d'un assemblage reste l'écran de son type, sans
réécriture.

```
Assemblages métalliques          ❔
🏠 · 🔄 · 💾 · 📂 · 📄 Générer PDF (lot 2)
Informations sur le projet       ➕
── Assemblages (3) · 1 non vérifié
A1  Platine d'about     ● vérifié · η = 82 %
    HEA 300 sur HEA 400 · V = 125 kN
    [Ouvrir]  [⋯]
A2  Plat d'âme soudé    ● non vérifié · η = 104 %
[ ➕ Nouvel assemblage ]
```

- **➕ Nouvel assemblage** ouvre le catalogue du registre avec un bouton
  « Créer » par type. L'assemblage naît avec ses valeurs par défaut, reçoit le
  premier repère libre (A1, A2, …) et s'ouvre aussitôt. Projet vide → le
  catalogue s'affiche directement : le premier assemblage reste à un clic.
- Chaque ligne : repère, type, profilés et efforts, pastille aux couleurs du
  PDF (vert vérifié · ocre sous réserve d'alertes · rouge non vérifié), taux
  maximal et vérification dimensionnante.
- **⋯** regroupe le repère (modifiable), 📋 Dupliquer, ↑ ↓ (ordre du rapport)
  et 🗑️ Supprimer, avec confirmation en ligne.
- Dans l'écran d'un assemblage : `◀ Projet · 🏠 · 🔄 · 💾 · [A2 ▾] · 📄`. Le
  sélecteur **[A2 ▾]** passe d'un assemblage à l'autre sans repasser par la
  liste ; 💾 enregistre tout le projet ; 📄 reste la note de cet assemblage.
- Le nom d'un assemblage est le champ **« Repère de l'assemblage »** de son
  écran (`values["id_rep"]`) : une seule source, pas de second champ.

## 2 · Où vivent les choses

| Quoi | Où |
|---|---|
| Modèle du projet (liste, repères, dupliquer, ordre, fichier) — sans Streamlit | `acier/projet.py` |
| Projet dans la session : ouvrir, fermer, synchroniser, barre partagée, infos projet | `acier/projet_ui.py` |
| Page Projet (liste, catalogue, menu ⋯, 💾 📂) | `modules/assemblages.py` |
| Barre d'outils des **sept** écrans (◀ Projet, 💾, sélecteur / 📂 seul) | `acier/assemblages/poutre_poutre/*/interface.py`, `acier/assemblages/poutre_poteau/*/interface.py` (préfixes `fpc_`, `dcc_`, `pac_` — famille livrée le 02/11/2026), `acier/assemblages/pied_poteau/platine_beton/interface.py` (préfixe `ppb_`, lot B du pied) |
| Registre des familles et des types (catalogue, vignettes, noms courts) | `acier/registre.py` — **3 familles, 7 types, tous « disponible »** |
| Appel de synchronisation au début de chaque run | `streamlit_app.py` |
| Tests | `tests/test_projet_assemblages.py` |

**Clés de session** : `prj_projet` (le dict du projet), `prj_ouvert` (id de
l'entrée ouverte), `asm_courant` (type de l'écran ouvert, inchangé),
`prj_*` transitoires (catalogue, confirmations, uploader). Les informations
du projet sont les clés communes à Poutre/Dalle (`nom_projet`, `partie`,
`date`, `indice`, `chk_infos_projet`).

**Fichier projet** :

```json
{"version": "projet-assemblages-1.0",
 "projet": {"nom_projet": "…", "partie": "…", "date": "…", "indice": "…"},
 "assemblages": [{"id": 1, "type": "platine_about", "repere": "A1",
                  "values": {…}, "visserie": "…"}]}
```

Nom : `AAA_Assemblages Partie#Indice_Date.json`, comme la note Poutre. 📂
accepte aussi un **fichier d'assemblage seul** (format des modules avant le
projet : il s'ajoute à la liste) et un fichier brut de l'outil HTML d'origine
(type détecté par ses clés).

## 3 · La règle de synchronisation

L'écran d'un type garde sa source unique (ses clés `<préfixe><clé>`) ;
l'entrée du projet en est une **vue**, recopiée **au début de chaque run**
(`streamlit_app.py` → `assemblages.synchroniser_projet()`), tant que l'état
du type est **vivant**.

Pourquoi au début du run : Streamlit applique la saisie de l'utilisateur
avant de lancer le script. La valeur tapée dans le même geste que le clic 🏠
est donc déjà dans l'état quand la page Accueil démarre — et c'est à la fin de
ce run-là que Streamlit efface les clés des widgets non rendus.

Pourquoi « vivant » : après la page Accueil, les clés de l'écran n'existent
plus ; `_init_etat` les recréerait avec les valeurs par défaut, et la recopie
suivante écrirait ces défauts dans l'entrée. Quand l'état n'est pas vivant et
que la page des assemblages va se rendre, c'est donc **l'entrée qui
ré-hydrate l'écran**. Le 🔄 d'un écran, qui efface les mêmes clés, appelle
`projet_ui.entree_reinitialisee()` pour que l'entrée prenne les défauts
(repère conservé) avant cette ré-hydratation.

## 4 · Pièges verrouillés (un test chacun)

1. Les doubles cornières utilisent les clés `asm_*` et leur 🔄 efface tout
   `asm_*`. Le projet vit sous `prj_*` : il survit.
2. Deux entrées d'un même type partagent un écran. À l'ouverture d'une
   entrée, la note PDF déjà générée et l'alerte sélectionnée sont oubliées
   (la note de A1 ne part pas sous le nom de A4) ; le **dernier clic du
   dessin est conservé** — le composant garde son dernier message, l'oublier
   le ferait rejouer sur la nouvelle entrée.
3. (Lot 2) La page de garde compte une page par élément : les numéros de page
   seront calculés sur les notes réellement produites, le sommaire pourra
   continuer sur une deuxième page, et les rapports Poutre/Dalle doivent
   sortir identiques.

## 5 · Le rapport du projet (lot 2)

`acier/rapport_projet.py` — sans Streamlit, `generer_rapport(projet, infos)`
rend les octets du PDF :

1. **Page(s) de garde** : la garde des notes Poutre/Dalle
   (`ndc_pdf.styles.Style.garde_lignes`, dont `garde()` est désormais un cas
   particulier — dessin inchangé, vérifié par `tests/test_export_pdf_ndc.py`) :
   bandeau « Bureau d'Études Valens · Note de calcul », titre « Assemblages
   métalliques », cartouche du projet, sommaire : repère — type, profilés et
   efforts, pastille (VÉRIFIÉ · SOUS RÉSERVE · NON VÉRIFIÉ · CALCUL IMPOSSIBLE),
   numéro de page. Au-delà de onze entrées le sommaire **continue** sur une
   deuxième page de garde et les numéros de page en tiennent compte.
2. **La note de chaque assemblage**, produite par `generer_note` de son type,
   sans modification ; son cartouche porte « partie · repère ». Les pages sont
   **comptées sur les PDF produits** (pymupdf), jamais supposées.
3. **Assemblage** avec pymupdf : un **signet** par assemblage, métadonnées du
   document. Une entrée dont le calcul échoue donne une page d'erreur et une
   pastille « calcul impossible » : le rapport sort quand même.

Sur la page Projet, **📄 Générer PDF** (désactivé si le projet est vide) puis
« ⬇️ Télécharger le rapport PDF », nommé `AAA_Assemblages Partie#Indice_Date.pdf`.
Le bouton de téléchargement disparaît dès que le projet change après la
génération (empreinte du fichier projet) : pas de rapport périmé.

## 6 · Lot 3 : récapitulatif, variante, verrou

- **Récapitulatif** : une page portrait après la garde (même bandeau), un
  tableau — n°, repère, type, profilés · efforts, η max, vérification
  dimensionnante, pastille d'état. Il est en tête du sommaire (ligne non
  numérotée) et a son signet ; les numéros de page des notes en tiennent
  compte. *Depuis le 02/11/2026, chaque assemblage tient sur **deux lignes**
  (20 par page) : le type, le taux et la pastille sur la première, les
  profilés, les efforts et la dimensionnante sur la seconde, en retrait.
  Six colonnes de texte et une pastille ne tenaient pas sur une A4 portrait —
  le type, les profilés ET la dimensionnante sortaient tous les trois coupés
  d'un « … », et un récapitulatif qui perd l'effort repris ne récapitule
  rien.*
- **🧬 Variante** (menu ⋯) : « créer à partir de » — copie « X (variante) »
  ouverte aussitôt, pour comparer deux solutions dans le même projet.
- **🔒 Verrou** (menu ⋯, et bouton 🔓 dans l'écran) : l'entrée ne se modifie
  plus — repère et suppression bloqués dans la liste ; dans l'écran, saisies
  et visserie désactivées, dessin rendu statique, 🔄 désactivé, bandeau de
  rappel. La garantie n'est pas l'interface : pour une entrée verrouillée,
  la synchronisation ne recopie jamais l'écran dans l'entrée et ré-hydrate
  l'écran depuis l'entrée à chaque run — une saisie forcée rebondit. Une
  copie naît déverrouillée ; le verrou est dans le fichier projet.

## 7 · Les SEPT types du catalogue (02/11/2026)

Le registre porte trois familles et sept assemblages, tous **disponibles** ;
la page Projet lit le registre et n'est pas modifiée quand on en ajoute un.

| Famille | Type | `id` | préfixe | nom court de la barre |
|---|---|---|---|---|
| Poutre – Poutre | Doubles cornières d'âme | `doubles_cornieres` | `asm_` | Doubles cornières |
| | Plat d'âme soudé (fin plate) | `plat_ame` | `fpl_` | Plat d'âme |
| | Platine d'about (hauteur partielle) | `platine_about` | `pab_` | Platine d'about |
| Poutre – Poteau | Plat d'âme soudé sur poteau | `plat_ame_poteau` | `fpc_` | Plat d'âme/poteau |
| | Doubles cornières d'âme sur poteau | `doubles_cornieres_poteau` | `dcc_` | Cornières/poteau |
| | Platine d'about sur poteau | `platine_about_poteau` | `pac_` | Platine/poteau |
| Pied de poteau | Platine de pied de poteau sur béton | `platine_beton` | `ppb_` | Pied de poteau |

**La ligne de liste** (`acier/projet.description`, source unique du sommaire
du rapport, du récapitulatif et de la liste de la page Projet) dit
« profilés · efforts ». Les trois types poutre–poteau y ajoutent le **cas
d'appui** — « IPE 400 sur HEB 300 (sur l'aile) · V = 200 kN » — parce que deux
entrées qui ne diffèrent que par l'appui donnaient sinon deux lignes
identiques. La clé reconnue est `appui` avec les valeurs « Aile du poteau » /
« Âme du poteau » : la clé `appui` du pied de poteau porte un tout autre choix
(le modèle d'aire efficace) et sa ligne ne change pas.

Un projet **MIXTE** aux sept types est tenu par
`tests/test_projet_assemblages.py` §10–11 (création, enregistrement,
relecture par la normalisation de chaque module, catalogue, ouverture des sept
écrans) et par `tests/test_rapport_projet.py` §5 (16 pages, 8 signets,
récapitulatif à deux lignes non coupé).

## 8 · Compatibilité

- Un écran ouvert sans entrée (tests existants, `asm_courant` posé
  directement) garde le comportement d'avant : 💾 et 📂 de l'assemblage seul,
  pas de sélecteur ; ◀ Projet mène à la page projet.
- Les clés des boutons (`asm_btn_open_<type>`, `<préfixe>btn_retour`,
  `<préfixe>btn_save_dl`, …) sont inchangées : le harnais existant les lit.
