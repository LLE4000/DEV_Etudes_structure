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
| Barre d'outils des trois écrans (◀ Projet, 💾, sélecteur / 📂 seul) | `acier/assemblages/poutre_poutre/*/interface.py` |
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

## 6 · Compatibilité

- Un écran ouvert sans entrée (tests existants, `asm_courant` posé
  directement) garde le comportement d'avant : 💾 et 📂 de l'assemblage seul,
  pas de sélecteur ; ◀ Projet mène à la page projet.
- Les clés des boutons (`asm_btn_open_<type>`, `<préfixe>btn_retour`,
  `<préfixe>btn_save_dl`, …) sont inchangées : le harnais existant les lit.
