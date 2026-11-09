# Dalles béton armé — fusion des interfaces Dalle et Prédalle, nappes et niveaux

*Conception du 09/11/2026, demande de l'ingénieur (prompt « Modifications du module Dalles béton armé — ergonomie,
fusion Prédalles et armatures »). Ce document est la référence des cinq lots FUS1 à FUS5 ; il est mis à jour avec ce
qui est construit.*

## 0. Ce qui ne bouge pas

- Les deux moteurs restent séparés et INCHANGÉS : `beton/dalle/moteur.py` et `beton/predalle/moteur.py` (ni une
  formule, ni une constante, ni un arrondi). Le site choisit lequel appeler. Les anciens écrans Streamlit
  (`modules/dalle.py`, `modules/predalle.py`) restent tels quels.
- L'état d'une dalle garde ses QUATRE faces (`inf_x`, `inf_y`, `sup_x`, `sup_y`), chacune une couche de base et ses
  renforts : c'est ce que les moteurs lisent. La « nappe » (§ 3) est une VUE d'écran sur deux faces.
- Sections multiples, barres X/Y indépendantes, renforts, positions imposées ou automatiques, plans et coupes,
  résultats à droite, note PDF (celle qui existe), .stab, import Streamlit, exports : tout reste atteignable.
  *(Révision PDF-D du 09/11/2026 : la note PDF se fait désormais sur le site, avec les chiffres du site — § 6.)*

## 1. Colonne de gauche (FUS5)

Aujourd'hui (`api_web/etabli_compact.arbre`, `construire.compact`) : « DONNÉES » (titre du panneau), « DALLES (1) »
(en-tête de l'arborescence), « Dalle 1 » (la ligne de l'élément, nom modifiable), « 1 — Dalle » (rubrique 1).

Cible :
- plus de titre « DONNÉES » ni d'en-tête « DALLES (n) » dans les modules à établi compact qui le demandent (paramètre du
  composant partagé : la dalle l'utilise ; la poutre garde son arbre tant que l'ingénieur n'a pas validé — un seul
  paramètre à basculer) ;
- une seule dalle : les rubriques commencent tout en haut, sans ligne « Dalle 1 » ;
- plusieurs dalles : un SÉLECTEUR compact en tête (une ligne : les noms en segments ou un menu, la dalle ouverte
  marquée) avec le menu « ⋯ » (Ajouter, Copier, Supprimer — jamais la dernière) ; les actions passent par les chemins
  existants (`gerer_elements` : activer / ajouter / dupliquer / supprimer) ;
- le NOM se modifie dans l'EN-TÊTE DES RÉSULTATS à droite (`E.tete_resultats` : le nom devient un champ
  `data-chemin="nom" data-sorte="texte"`, même mécanisme que la ligne d'arborescence d'aujourd'hui, verrou = lecture
  seule) ; le même chemin « nom » qu'avant, donc résultats, résumé, .stab, onglets téléphone et navigation restent
  synchronisés sans rien d'autre ;
- rubrique 1 : « 1 — Caractéristiques » (clé `dl`, inchangée) ; 2 — Sections et sollicitations ; 3 — Armatures ;
  4 — Options.

**Construit (FUS5, 09/11/2026)** — les noms exacts :
- `api_web/etabli_compact.arbre(genre, elements, actif, corps, marque, compact=False)` : `compact=False` rend le HTML
  d'avant à l'octet (gel : `tests/test_etabli_compact.py`) ; `compact=True` → `arbre_compact(…)` :
  `<div class="et-arbre et-arbre-c[ et-seul]" data-<marque>="<actif>">` puis, UN élément : `<div class="et-elsel seul">`
  qui ne porte que le menu « ⋯ » (`.et-mn`, `data-menu`), posé par `css/etabli.css` dans l'angle, sur le bord droit de
  l'en-tête de la première rubrique (`.et-elsel.seul + .sec > .sechd { padding-right }`) ; PLUSIEURS : `<div
  class="et-elsel">` = `.seg[role=group]` d'un `button[data-activer="i"][aria-pressed]` par élément (`.on` l'ouvert, un
  cadenas si verrouillé, `<span class="nom">`), `<span class="n">` le compte, puis le menu « ⋯ » ; dans les deux cas les
  rubriques (`corps`) suivent directement. Le menu : `data-ajouter="1"` (l'action « ajouter » existante), `data-renommer`
  (éteint si verrouillé), `data-dupliquer-index="<actif>"`, `data-supprimer="<actif>" data-confirmer="…"` (éteint quand
  l'élément est seul) — les attributs que `js/etabli.js` lisait déjà, pas un de plus. Plus de `.et-arbre-h`, de `.et-pt`,
  de `.et-pt-add` ni de champ `data-chemin="nom"` à gauche.
- `api_web/etabli_compact.tete_resultats(nom, dims, chemin_nom=None, verrou=False)` : sans `chemin_nom`, le HTML d'avant
  ; avec (`"nom"`) : `<div class="et-rp et-rp-c"><input data-chemin="nom" data-sorte="texte" class="et-rp-nom" …>
  <span>dims</span></div>` ; `verrou` : `<b>` avec un cadenas, pas de champ. `js/etabli.js` écoute `focusout` et Entrée
  sur `#resultat` comme sur `#donnees` (« modifier »), garde la frappe en cours quand le cadre est refait
  (`saisieEnCours` / `reprendreSaisie`), et « Renommer » (`renommer()`) choisit ce champ — au téléphone l'onglet
  Résultat s'ouvre, au PC la colonne repliée se rouvre ; `js/compact.js` règle sa largeur au texte (`ajusterNoms`) et
  garde le segment ouvert sous les yeux (`montrerOuvert`).
- `site/web/construire.py` : `ARBRE_COMPACT = ("dalle",)` — l'en-tête du panneau (`compact(…, arbre_compact=True)`)
  n'écrit plus `<span class="h3">Données</span>` (Annuler, Rétablir restent à droite : `.phead > .calcule:first-child {
  margin-right: auto }`) ; la poutre garde le titre et son arbre. `tests/test_site_web.py` lie ce tuple à ce que le
  moteur rend (`et-arbre-c`, `et-rp-c`) : un module ajouté d'un seul côté rougit.
- Harnais : `module:dalle` rejoue aussi les contrôles « plaques dalle » (colonne compacte, Copier → sélecteur, renommer
  dans la tête des résultats → sélecteur, confirmation du menu et note enregistrée, « Renommer » du menu → curseur dans
  le champ, segment → autre dalle, Supprimer → une seule, type Prédalle → `h_pre` et complément, téléphone : « Renommer »
  → onglet Résultat) ; `formats:dalle` lit la colonne par `#donnees .et-arbre .sec.open` et le nom long par
  `#resultat .et-rp.et-rp-c`.

**Révision du 09/11/2026 (COL1, retour de l'ingénieur sur la capture)** — ce qui est REMPLACÉ ci-dessus :
« le nom de la dalle a complètement disparu ; il y a les trois petits points ; ce qui aurait été bien, c'est de garder
le nom : un clic dessus, on renomme ; à côté, dupliquer ou effacer ; et la ligne au-dessus (Annuler, Rétablir) est
vide ». Puis : « applique la même chose à la poutre ».
- La LIGNE de l'élément ouvert (`etabli_compact.arbre_compact` : `<div class="et-elsel" data-elements="n">`) porte le
  NOM en champ (`input.et-elt-nom[data-chemin="nom"]` : un clic et on le renomme ; Entrée valide, Échap rend l'ancien
  nom, un nom vide est refusé ; verrouillé, `span.et-elt-nom.lu` et un cadenas), la LISTE des éléments (un menu
  `.et-mn.et-elt-liste` : `button.et-elt-i[data-activer="i"]` par élément avec ses dimensions, l'ouvert coché, puis
  `data-ajouter` « Ajouter une … » ; le compte sur le bouton dès deux), COPIER (`.et-elt-b[data-dupliquer-index]`) et
  SUPPRIMER (`.et-elt-b.danger[data-supprimer][data-confirmer]`, éteint quand l'élément est seul). Plus de segments, de
  menu « ⋯ » d'élément ni de `data-renommer` dans cette ligne.
- `js/etabli.js` POSE cette ligne dans l'EN-TÊTE du panneau de gauche, à côté d'« Annuler » et « Rétablir »
  (`construire.compact(…, arbre_compact=True)` y met l'emplacement `<div class="et-elt">` à la place du titre
  « Données ») ; ses écouteurs (nom, actions) et ceux de `js/compact.js` (le menu) y sont branchés. La ligne vide d'avant
  n'existe plus : rien ne prend de hauteur en plus des rubriques.
- Le nom ne se modifie plus dans la tête des RÉSULTATS : `tete_resultats(nom, dims)` redevient un titre (un seul endroit
  pour renommer).
- La POUTRE prend la même colonne : `site/web/construire.ARBRE_COMPACT = ("dalle", "poutre")`, `api_web/poutre.arbre`
  passe `compact=True` (son arborescence `et-pt` n'est plus rendue ; `arbre(…, compact=False)` reste, à l'octet, pour
  l'ancien module Prédalles servi aux anciens liens).
- Les types de dalle s'écrivent « Dalle CSP » et « Prédalle + dalle CSP » (CSP : coulé sur place ; l'aide du champ le
  dit en entier).

## 2. Un seul module « Dalles béton armé » (FUS2, FUS5)

### 2.1 L'état

`{"version": 2, "dalles": [dalle…], "actif": i}` ; chaque dalle porte `"type": "coulee" | "predalle"`.
- `coulee` : la dalle de `beton/dalle/entree.normaliser` (+ `verrou` du site), sans `h_pre`.
- `predalle` : la prédalle de `beton/predalle/entree.normaliser` (+ `verrou`), avec `h_pre`.
Relecture (`api_web/dalle.normaliser`) :
- `{"dalles": […]}` sans `type` (note d'avant) → `coulee` ;
- `{"predalles": […]}` (note de l'ancien module Prédalles, mémoire du navigateur ou .stab) → chaque prédalle devient
  une dalle `type = predalle`, `actif` conservé ; RIEN n'est perdu, les chiffres sont ceux de l'ancien module (même
  moteur, mêmes positions : § 4.3) ;
- une note qui a les deux clés : refusée avec un message clair (abîmée), jamais remplacée par une note neuve.
Le fichier Streamlit `dalle_ba.json` et celui des prédalles s'importent l'un et l'autre (les `fichier.py` des deux
moteurs, inchangés) : le type est déduit du fichier.

### 2.2 Le sélecteur et les champs

Rubrique « 1 — Caractéristiques », première ligne : segments « Dalle CSP » (défaut) · « Prédalle + dalle CSP » (CSP :
coulé sur place — libellés courts demandés le 09/11/2026 ; d'abord « Dalle coulée en place » · « Prédalle + béton coulé
en place ») (chemin `["type"]`, valeurs `coulee` / `predalle`).
- Dalle coulée : les champs d'aujourd'hui (Béton, Acier, b, h, c, Principale).
- Prédalle : en plus, « h<sub>p</sub> prédalle » (épaisseur préfabriquée, `["h_pre"]`, bornes `EN_P.H_PRE_MIN` à
  `h_pre_max(h)`) et le COMPLÉMENT coulé en place en lecture seule : `h − h_pre` (recalculé à chaque saisie, jamais
  stocké : R2). Les refus géométriques existants de `api_web/predalle._ecrire` (« au moins 1 cm coulé en place »…)
  s'appliquent tels quels.
Changer de type : `coulee → predalle` ajoute `h_pre` (défaut `EN_P.DEFAUTS_PREDALLE["h_pre"]`, borné à `h_pre_max(h)`),
`predalle → coulee` retire `h_pre` ; sections, faces, couches, positions imposées, noms : conservés.

### 2.3 Les moteurs et ce qui en dépend

`api_web/dalle._calculer(d)` appelle `beton.dalle.moteur` si `coulee`, `beton.predalle.moteur` si `predalle`. Les
avertissements propres à la prédalle (section finale / phase provisoire, limitations, périmètre `PERIMETRE`) viennent
de `api_web/predalle.py`, réemployé comme AIDE du type prédalle (ses fonctions `_avertissements`, `_hypotheses`,
`PERIMETRE`, `_pose`, `REGLES_TXT`, `dessin_predalle`), pas recopié. Les dessins : `dessin_dalle` pour `coulee`,
`dessin_predalle` pour `predalle` (même interface `svg_coupe/svg_plan/svg_toutes`).

**Construit (FUS2, 09/11/2026)** — `api_web/dalle.py`, les noms exacts :
- L'ÉTAT : `VERSION = 2` ; `CLES_SITE = ("verrou", "type", "empilement")` — les trois clés du site portées par CHAQUE
  dalle (« empilement » est « par note » comme γs et le jeu : écrite sur toutes les dalles à la fois) ; la même liste
  vit dans `dalle_positions.CLES_SITE`, et `tests/test_api_web_dalles_fusion.py` lie les deux copies (R2).
  `TYPES_DALLE = (("coulee", "Dalle CSP"), ("predalle", "Prédalle + dalle CSP"))`, `TYPES_TXT`.
  `est_predalle(d)` ; `_EN(d)` / `_MO(d)` / `_M(d)` : l'entrée, le moteur et le module « formules mises en mots » du type
  (`_M` rend `api_web.predalle` pour une prédalle : `plaque_compact.bloc` y lit `_formule`, `_ligne_tableau`…).
  `_site(d, verrou=False, type_="coulee", empilement=False)` normalise par l'entrée du type puis pose les trois clés ;
  `_sans_site(d)` (pour RE-normaliser, jamais pour le moteur) ; `_resite(d, **modifs)` ; `nouvel_etat()` : une dalle
  coulée, `empilement=True`. `normaliser(etat)` relit `{"dalles"}` (type absent → coulee, empilement absent → faux),
  `{"predalles"}` (→ predalle, `actif` conservé), et REFUSE (ValueError, message en français) : les deux clés ensemble,
  une liste qui n'en est pas une, un `type` inconnu, un `type ≠ predalle` dans `predalles`, `verrou`/`empilement` non
  booléens, `h_pre` sur une dalle coulée (clé inconnue du moteur des dalles), une peau impossible. `_bornes_site`
  applique aussi `AP.NOM_MAX` (60) aux deux types. `gerer_elements("ajouter")` : le type de la dalle ouverte, le nom
  « Prédalle k » ou « Dalle k », γs, jeu et empilement de la note.
- La SAISIE (`modifier` → `_ecrire(etat, d, chemin, valeur)`) : `["type"]` → `_changer_type` (coulee → predalle :
  `h_pre = min(max(6, H_PRE_MIN), h_pre_max(h))` ; predalle → coulee : `h_pre` retirée ; tout le reste gardé) ;
  `["h_pre"]` → `AP.peau(d, valeur)` (refusée sur une dalle coulée, avec le chemin à suivre) ; `["h"]` →
  `AP.epaisseur_totale(d, valeur)` pour une prédalle (« au moins h_pre + 1 cm »), `_nombre("h")` sinon ;
  `["empilement"]` → toutes les dalles de la note (refusé si la dalle ouverte est verrouillée, comme γs) ; `["nom"]` et
  `sections/i/nom` → `AP.nom_saisi` ; `sections/i/nappes/…` → `NAP.ecrire(d, chemin, valeur)` (ValueError → message,
  dalle intacte) ; `…/dist` → `_auto_ou_impose(…, POS.axe_auto(d, s, f, j))` (`s` : la SECTION, dict) ; `…/yG` →
  `POS.yG_calcule(d, s, f)`. Les refus de l'aide prédalle sont `AP.Refus` (attrapés comme `_Refus`, message tel quel).
  `choisir_section` : un treillis choisi sur une nappe qui était un treillis unique (`NAP.mode(s, cote, principale)`)
  pose `NAP.transposee(des)` sur l'autre face (demande FUS3).
- L'ÉCRAN : `_caracteristiques(d)` = `C.ligne("Type", C.segments(["type"], …), …, AIDE_TYPE, full=True)` + Béton | Acier
  + b | h (« h totale » pour une prédalle) + [prédalle : « h<sub>p</sub> prédalle » `C.num(["h_pre"], …,
  mini=EN_P.H_PRE_MIN, maxi=EN_P.h_pre_max(h))` | « coulé h − h<sub>p</sub> » en `<span class="dl-ro">`, lu, jamais
  stocké] + c | Principale ; titre `PC.formulaire("dl", "Caractéristiques", résumé, …)`, résumé « Coulée en place · C30/37
  · B500 · c = 3 cm » ou « Prédalle h<sub>p</sub> = 6 cm · C30/37 · B500 ». `_formulaire_verrouille` dit le type et
  « prédalle · coulé en place h_p + (h − h_p) cm ». `_options(d)` : la méthode du moteur du type, γs, jeu (le texte de
  la prédalle pour une prédalle), la case `["empilement"]` « Nappes empilées : direction secondaire posée sur la
  principale » (`AIDE_EMPILEMENT`), l'import « dalle_ba.json ou le fichier de l'écran Prédalle ». Rubrique 3 :
  `NAP.html(EN_, i, s, rs, verifs, rs["principale"], _aide_axe(d), _aide_renfort(d), False)` — `_aide_axe` rend
  `AIDE_AXE` (chaîne) pour une dalle coulée, et pour une prédalle la fonction `(f, j, c, g) → {"details":
  AP.pose_couche(c, g), "aide_axe": AP.POSES_AIDE[g["pose"]], "dit_auto": False}` ; `_aide_renfort` de même (« posées
  sur la prédalle » aux faces inférieures). `arbre` → `E.arbre(E.DALLE, elements, actif, formulaire, "dalle",
  compact=True)`, dims « · b × h cm[, prédalle h_p cm] ».
- Le RÉSULTAT : `_calculer(d) = POS.annoter(_MO(d).verifier(POS.pour_moteur(d)), d)` ; `_optimaux` et `selecteur` →
  `POS.candidats_base(d, i, f, fams)`, la ligne d'un candidat par `AP.description_candidat(c, axe)` (l'axe et la pose
  dits pour une prédalle) ; `_dims(d)` « 100 × 22 cm, prédalle 6 cm · C30/37 · B500 » ; `_avertissements_de(d, r, i)`
  → `AP.avertissements` / `_avertissements` ; hypothèses `AP.hypotheses(d, m)` ; `pied_calcul=C.aide(AP.PERIMETRE)` et le
  périmètre sous le résultat ; pied « moteur prédalle v1.1 » / « moteur dalle v2.2 » ; `E.tete_resultats(C.esc(nom),
  dims, chemin_nom=["nom"], verrou=d["verrou"])` ; `PC.vues(r["sections"], r["predalle"] | r["dalle"], DP | DD, "pd" |
  "dl", dims, AP.LEGENDES | LEGENDES)` ; `resume_note` et le statut de la note avec le moteur et le module de CHAQUE
  dalle. `_ouvrir_streamlit` : la clé « predalles » du JSON choisit `FI_P.lire`, sinon `FI.lire` ; le type est celui
  du fichier, `empilement=False`, `_bornes_site` ; « repris (n prédalle(s)) ».
- `api_web/predalle.py` (le module « predalle » reste servi, et rend le même HTML) : renommés sans tiret bas, alias
  gardés — `Refus`, `saisi`, `nom_saisi`, `bornes_site`, `peau`, `axe_auto`, `yG_calcule`, `pose_couche`, `hypotheses`,
  `avertissements` ; extraits : `epaisseur_totale(p, valeur)`, `description_candidat(c, axe="")` ; pour l'empilement
  (FUS4) : `pose_couche` dit « axe auto · posée sur la base principale (e_p + ½ Ø_p + ½ Ø_s) » quand
  `geo_c["empilee"]`, et `_formule` écrit la formule « e » d'une couche `empilee` « e_p + ⌈Ø_p/2⌉ + ⌈Ø_s/2⌉ · posée sur
  la base principale » (termes `e_p`, `demi_p`, `demi_s` posés par `dalle_positions.annoter`).
- `PARCOURS` : les requêtes d'avant plus celles du type prédalle sur une note `_NOTE_P` (`{"predalles": [nouvelle
  prédalle]}`) : `h_pre` 21 / 3, `h` 6, `type` coulee, sélecteur, choisir, ajouter ; l'import des DEUX fichiers.
- *Révision du 09/11/2026 (NAP, § 4.3) : la clé « empilement », son chemin, sa case et le paramètre `empilement` de
  `_site` n'existent plus ; `CLES_SITE = ("verrou", "type")`, `VERSION = 3`, `CLES_ANCIENNES = ("empilement",)` (relue sans
  effet) ; l'avis d'ouverture `CLE_AVIS` / `AVIS`.*
- Preuves : `tests/test_api_web_dalles_fusion.py` (57 contrôles dont 5 mutations), `tests/test_api_web_dalle.py` (135 :
  le type coulée, l'option empilement décochée par son aide `nouveau()` — la parité avec `beton/dalle`),
  `tests/test_api_web_predalle.py` (103, inchangé), les deux moteurs (169, 152, inchangés). Le gel
  `tests/test_reference_refonte.py` : prédalle vert ; les 9 cas dalle rougissent — état v2 + `type` + `empilement`,
  champs et menus des nappes et du type, et les CHIFFRES de la direction secondaire (3,5 → 4,5 cm) parce qu'une note
  neuve a ses nappes empilées (§ 4.3, voulu) : regel `--accepter` à faire dans un commit distinct, pas ici.

### 2.4 L'accueil et les pages

- `site/web/construire.py` et `site/maquettes/construire.py` : une seule carte « Dalles béton armé » ; `predalle`
  rejoint `MASQUES` (page servie, plus listée — comme les modules de consultation du lot BIB3) ; `predalle.html`
  devient une page de REDIRECTION vers `dalle.html` en gardant l'adresse (`?note=…`, `?nom=…`) : une ancienne note
  s'ouvre dans le module fusionné en mode prédalle.
- `js/notes.js` : le module « predalle » d'une note en mémoire ou d'un .stab est servi par la page `dalle` (table des
  alias) ; l'état est relu par `api_web/dalle.normaliser` (§ 2.1). Le module `predalle` du registre reste servi
  (anciens liens, harnais) tant que la redirection n'est pas prouvée.

**Construit (FUS5, 09/11/2026)** :
- `site/web/construire.py` : `REDIRECTIONS = {"predalle": "dalle"}` (la SEULE table d'alias ; `js/modules.js` la porte
  — `MODULES.predalle = {page, nom, famille, masque: true, vers: "dalle"}` — et `js/notes.js` la lit :
  `moduleServi(module)`, `pageDe(module)`, employés par `js/accueil.js` (cartes « Reprendre », « Nouvelle note »,
  import .stab) et `js/etabli.js` (import .stab, note rouverte)). `page(cle)` écrit `page_redirection(cle)` pour un
  module absorbé : `predalle.html` = `<meta http-equiv="refresh" content="1; url=dalle">`, `<body data-page="redirection"
  data-module="predalle" data-vers="dalle">`, une carte « Le module « Prédalles » a rejoint « Dalles béton armé » » avec
  le lien de repli, et `<script src="js/redirection.js">` qui fait `location.replace(vers + location.search +
  location.hash)` — jamais de script en ligne (la politique de sécurité), rien d'un tiers (R5). `ETABLI_COMPACT =
  ("poutre", "dalle")`.
- `site/maquettes/construire.py` : `MASQUES = ("donnees_beton", "armatures", "enrobage", "predalle")`, et `listes(mods)`
  filtre les masqués dans l'accueil, le volet « Modules » et « Nouvelle note » des maquettes comme du site.
- `js/etabli.js` : une note dont `moduleServi(note.module) === MODULE` est RELUE (pas recréée) ; la requête part avec
  `module: "dalle"` et l'état tel quel (`{"predalles": …}`) ; à la première réponse, `note.module` devient `"dalle"` et
  la note s'enregistre dans la forme rendue par le moteur (`{"version": 2, "dalles": [… type: "predalle"]}`).
- Harnais `module:predalle` : le contrôle commun constate la redirection (adresse finissant sur `dalle`, `?nom=` et
  `?chantier=` repris) ; « plaques predalle » : une ancienne note `{predalles: [{nom, h_pre: 6}]}` semée en mémoire —
  sa carte de l'accueil vise `dalle?note=…`, `predalle?note=…` arrive sur `dalle?note=…`, le moteur reçoit `dalle` +
  `{predalles}`, la note s'ouvre en prédalle (`h_pre` 6, nom « Ancienne P1 ») et devient `module: "dalle"` en mémoire.

## 3. Les armatures par NAPPE (FUS3)

Un treillis soudé est une nappe bidirectionnelle. Nomenclature `Øl/Øt/el/et` (`beton/treillis.py`) : porteurs Øl/el
dans la direction CALCULÉE de la face ; répartition Øt/et non comptés. Donc une nappe « un treillis T = a/b/ea/eb,
porteurs selon X » s'écrit, dans l'état INCHANGÉ : face `inf_x` base treillis `a/b/ea/eb`, face `inf_y` base
treillis `b/a/eb/ea` (la TRANSPOSÉE). Le moteur compte a/ea en X et b/eb en Y : exactement les deux armatures du
treillis, selon l'orientation. (Les dix treillis standards sont symétriques ; la désignation accepte un treillis
dissymétrique, ex. `10/12/100/150`.)

**Construit (FUS3, 09/11/2026) : `api_web/dalle_nappes.py`, `tests/test_api_web_dalle_nappes.py` (64 contrôles, 3
mutations), bloc CSS « nappes de dalle » en fin de `site/web/css/etabli.css`.** Appelé par `api_web/dalle._ecrire`
(`chemin[2] == "nappes"` → `NAP.ecrire(d, chemin, valeur)`, `ValueError` → refus avec le message) et par
`dalle.formulaire` (`NAP.html(EN, i, s, rs, verifs, principale, aide_axe, aide_renfort, verrou)` pour chaque section ;
`aide_axe` et `aide_renfort` : un texte, ou une fonction — la prédalle y pose la pose de chaque couche).

Lecture de l'état — pas de champ nouveau, le mode se DÉDUIT (`NAP.mode(s, cote, principale=None)` →
`{"mode", "porteurs", "treillis", "renforts": {"x": n, "y": n}, "coherent"}`) :
- deux bases treillis dont l'une est la TRANSPOSÉE de l'autre → `treillis`, ou `treillis_renforts` dès qu'une des deux
  faces porte un renfort ;
- deux bases qui ne sont pas des treillis (barres Ø/e, n barres) → `barres` ;
- sinon (deux treillis différents, ou treillis d'un côté et barres de l'autre : une ancienne note) → `incoherent`.
L'ORIENTATION des porteurs se déduit aussi, car l'état ne la stocke pas (« a/b/ea/eb porteurs X » et « b/a/eb/ea
porteurs Y » sont le MÊME état) : les porteurs sont la direction dont la base porte la plus forte section au mètre
(Øl/el) ; à égalité — les dix treillis standards sont symétriques — la direction principale. La désignation affichée
est la base de cette face. Une note écrite « 8/10/100/150 porteurs X » se relit donc « 10/8/150/100 porteurs Y » : même
armature, même pose, et la ligne X/Y le dit sans ambiguïté.

Écran, rubrique « 3 — Armatures », pour chaque nappe (`.et-nap`, « Nappe inférieure » puis « Nappe supérieure ») :
- segments « Treillis soudé · Barres · Treillis + renforts » (chemin `["sections", i, "nappes", "inf"|"sup", "mode"]`) ;
  en mode incohérent aucun n'est marqué ;
- mode TREILLIS : UN sélecteur de treillis (`…, "treillis"` ; les dix de la base, plus la désignation courante si elle
  n'y est pas) et l'orientation des porteurs (`…, "porteurs"`, segments X / Y, la principale marquée « · princ. » ;
  GRISÉS pour un treillis symétrique, où l'orientation est sans effet) ; la ligne « X : Ø10 e 100 · Y : Ø12 e 150 »
  dit ce que chaque direction reçoit ; puis le détail de chaque direction (la principale d'abord, `plaque_compact.nappe`)
  où la base se LIT (type et désignation en `.et-lu`, « réglés par la nappe ») mais garde son axe (vide = automatique,
  « ↺ »), ses renforts et son y_G ; la clé `data-cible` de chaque couche est inchangée (le dessin la sélectionne) ;
- mode BARRES : exactement l'écran d'aujourd'hui par direction (`plaque_compact.nappe` + `couche`), chacune ses renforts ;
- mode TREILLIS + RENFORTS : le treillis comme ci-dessus, les renforts de chaque direction réglables comme aujourd'hui,
  « + Renfort » dans chaque direction ;
- mode INCOHÉRENT : l'écran par direction comme aujourd'hui, PLUS l'avertissement (`.et-avert`) « Deux armatures de base
  différentes en X et en Y — ce n'est pas un treillis unique » et le bouton « En faire un treillis » (chemin `mode`,
  valeur `treillis`) ; on ne peut donc plus croire à un treillis unique quand il y en a deux ;
- verrouillée : tout se lit, aucun bouton ne modifie (les pièces gardent leur `data-cible`).

Écriture (`NAP.ecrire`) — elle ne touche QUE le type et la désignation des deux bases ; renforts, axes imposés, y_G et
les réglages rangés (Ø, espacement, nombre) restent, et la dalle écrite est dans le domaine de `EN.normaliser` :
- `mode = treillis` : si l'une des bases est déjà un treillis on la garde (à deux treillis différents, celle de la
  direction principale) et sa transposée va sur l'autre ; sinon `TR.TREILLIS_DEFAUT` sur les deux. Les renforts ne sont
  JAMAIS retirés en silence : depuis des barres qui ont un renfort, le mode déduit devient « Treillis + renforts » ; et
  depuis « Treillis + renforts », « Treillis soudé » seul est REFUSÉ avec le message qui renvoie à « Retirer ce renfort » ;
- `mode = barres` : chaque base treillis devient `barres` avec SON Ø et SON espacement ; une base déjà barres ou n barres
  reste ce qu'elle est ;
- `mode = treillis_renforts` : comme `treillis`, puis, si aucune des deux faces n'a de renfort, le premier renfort
  (3 Ø12, le défaut de « + Renfort ») s'ajoute sur la face des porteurs — le même geste que `faces/<face>/couches/+` ;
- `treillis = Øl/Øt/el/et` (lue par `TR.parse_designation`, canonisée) : sur la face des porteurs (ceux de la nappe si
  c'est déjà un treillis unique, sinon la direction principale), la transposée sur l'autre ;
- `porteurs = x | y` : les deux désignations s'échangent (refusé si la nappe n'est pas un treillis unique) ;
- refusés avec un message : désignation illisible, côté autre que `inf`/`sup`, champ inconnu, mode inconnu, section
  introuvable.
Le sélecteur « ampoule » (`dalle.choisir_section`) règle encore UNE face : en mode treillis, choisir un autre candidat
rend la nappe incohérente, ce que l'écran dit aussitôt (« En faire un treillis » répare) — le rendre conscient de la
nappe est une demande du rapport FUS3.

## 4. Les NIVEAUX des deux directions (FUS4)

### 4.1 La règle

Les deux directions d'une nappe se croisent : leurs axes sont à des hauteurs différentes. Direction principale au
plus près du parement tendu ; la secondaire empilée dessus, les barres en contact. Avec la convention d'arrondi du
moteur (demi-diamètre arrondi au 0,5 cm supérieur, `arrondi_demi_cm`) :
- base de la direction principale : l'axe automatique du moteur (inchangé) ;
- base de la direction secondaire : `axe principal + ⌈Ø_p/2⌉₀,₅ + ⌈Ø_s/2⌉₀,₅` (Ø_p, Ø_s : diamètres des fils ou
  barres des deux bases ; pour un treillis, Øl de chaque face) — aucune distance inventée *(révisé le 09/11/2026, S5 :
  « axe principal + ⌈Ø_p/2⌉ » est le dessus de la couche principale la plus haute du lit de contact, base ou renfort —
  § 4.3)* ;
- supérieures : la même règle depuis le parement supérieur (principale la plus proche du haut, secondaire en dessous) ;
- renforts : leur axe reste celui du moteur (imposé ou automatique), affiché et dessiné à sa valeur *(révisé le
  09/11/2026, S3 : un renfort de la direction SECONDAIRE se pose sur sa nappe, comme sa base — § 4.3)* ;
- prédalle, faces inférieures : le moteur sait déjà poser (base principale DANS la peau, autre base inférieure SUR la
  prédalle : `beton/predalle/moteur.pose`) — règle du moteur conservée telle quelle ; ses faces supérieures suivent la
  règle d'empilement comme la dalle.

**Construit (FUS4, `api_web/dalle_positions.py`)** — la règle exacte de `axe_auto(d, s, f, j)` :
- option `empilement` fausse ou absente, base principale (j = 0, direction = `dir_principale`), renfort (j > 0), face
  INFÉRIEURE d'une prédalle : **exactement** `distance_auto` du moteur du type (dalle : `MO_D.distance_auto(c, enrobage,
  jeu)` ; prédalle : `MO_P.distance_auto(c, j, f, dir_principale, enrobage, jeu, h_pre)`) ;
- option vraie, base de la direction SECONDAIRE (dalle coulée : inf. et sup. ; prédalle : sup. seulement) :
  `e_p + arrondi_demi_cm(Ø_p / 20) + arrondi_demi_cm(Ø_s / 20)`, où `e_p` est l'axe **effectif** de la base principale de
  la même face (son `dist` s'il est saisi, sinon l'automatique du moteur : la secondaire repose sur la principale là où
  elle est — posée sur l'automatique alors que la principale est saisie à 5 cm, elle reviendrait à son niveau), Ø en mm
  par `couche_diametre` (Øl d'un treillis), les deux arrondis étant ceux du moteur.

Exemple chiffré (c = 3 cm, jeu 0, principale Y, bases Ø12/150 en Y et treillis 10/10/100/100 en X, renfort 3 Ø12 en
inf. Y) : inf. Y = 3 + 1,0 = **4,0** ; inf. X = 4,0 + 1,0 + 0,5 = **5,5** ; sup. Y 4,0, sup. X 5,5 (depuis le dessus) ;
renfort 4,0 (moteur). La même en prédalle (h_pre = 6) : inf. Y 4,0 (dans la peau), inf. X 6 + 0,5 = 6,5 (sur la prédalle),
renfort 7,0 — règle du moteur ; sup. Y 4,0, sup. X 5,5 (empilées). Principale X : base X 3,5, base Y 3,5 + 0,5 + 1,0 =
5,0. Le d utile de la secondaire passe de 16,5 à 14,5 cm (h = 20) et l'avertissement « nappes_confondues » du moteur
disparaît : les bases ne sont plus au même niveau. Test : `tests/test_api_web_dalle_positions.py` (55 contrôles, dont
trois mutations en mémoire).

### 4.2 Sans toucher au moteur

Le moteur honore une position imposée (`dist`). Le site calcule l'axe « automatique du site » d'une couche
(`api_web/dalle_positions.axe_auto(d, s, f, j)`), et, quand il diffère de l'automatique du moteur (la base secondaire
empilée), il l'ÉCRIT dans la copie envoyée au moteur (`dalle_positions.pour_moteur(d)` : `dist` renseigné). L'état
garde `dist = None` (automatique), l'écran continue d'afficher « auto · 2,2 cm » avec la valeur du site, et le dessin
lit les positions que le moteur a effectivement utilisées (`r["sections"][i]["faces"][f]["couches"][j]["e"]`). Les
cotes des hauteurs utiles (d = h − y_G) sont celles du calcul.

**Construit** — l'interface de `api_web/dalle_positions.py`, que `api_web/dalle.py` appelle partout où il parlait d'axe
automatique ou de moteur :
- `axe_auto(d, s, f, j) -> float`, `empilee(d, s, f, j) -> bool`, `axes(d, s) -> {face: [axe effectif…]}` ;
- `pour_moteur(d) -> dict` : copie sans `verrou`, `type`, `empilement` ; `dist` écrit là où l'axe du site diffère de
  celui du moteur, et là seulement ; `d` jamais modifié. Option fausse : `verifier(pour_moteur(d)) == verifier(d)` à
  l'octet (prouvé en JSON pour la dalle et la prédalle) ;
- `yG_calcule(d, s, f)`, `h_utile(d, s, f)` : lus dans `beton/dalle/moteur._face` sur les couches aux axes effectifs
  (remplacent `_yG_calcule` de `dalle.py` et `predalle.py`, qui lisaient l'automatique du moteur) ;
- `annoter(r, d) -> r` : le résultat du moteur relu avec la notion d'automatique DU SITE — par couche `auto` (vide dans
  l'ÉTAT ; le moteur, lui, a reçu un axe imposé), `e_auto` (du site : le champ vide affiche « auto · 5,5 »),
  `e_auto_moteur` (gardé), `empilee` ; les formules « e » d'une prédalle reçoivent `auto`, `empilee` et les termes
  `e_p`, `d_p`, `demi_p`, `d_s`, `demi_s` (le terme « saisi » du moteur reste, pour que `predalle._formule` lise encore).
  Le calcul d'une dalle du site s'écrit `POS.annoter(MO.verifier(POS.pour_moteur(d)), d)` ; sans `annoter`, le
  résultat du moteur dit « saisi » d'un axe que le site lui a imposé (la légende du dessin de prédalle, le « ↺ », le
  « l'automatique serait ») ;
- `candidats_base(d, i, f, familles)` : le sélecteur. Option fausse : le moteur tel quel. Option vraie : le Ø d'un
  candidat déplace l'axe empilé (le sien pour une base secondaire : Ø8 à 5,5, Ø12 à 6,0, Ø25 à 6,5 ; celui de la
  secondaire posée dessus pour une base principale), ce que le moteur ne sait pas recalculer sur un axe imposé — la
  dalle est recomposée candidat par candidat et calculée par le moteur ; mêmes clés que le moteur (plus `e`), même tri ;
- `moteur(d)`, `type_dalle(d)`, `empilement(d)`, `principale(d)`.

### 4.3 L'option « nappes empilées » — RETIRÉE le 09/11/2026 (révision NAP ci-dessous)

**Révision du 09/11/2026 (NAP, décision de l'ingénieur, soir)** — « Les nappes, on les met toujours l'une sur l'autre ;
on n'est pas obligé d'avoir les deux options » ; « S3 : corriger, il se pose sur sa nappe, comme la base ». Ce qui est
REMPLACÉ dans ce paragraphe (texte d'avant gardé plus bas pour l'histoire) et dans les § 2.3, 4.1, 4.2 :
- **Plus d'option.** Ni case dans « 4 — Options », ni chemin `["empilement"]` (refusé : champ inconnu), ni clé d'état :
  `api_web/dalle.CLES_SITE = ("verrou", "type")` (= `dalle_positions.CLES_SITE`, liés par
  `tests/test_api_web_dalles_fusion.py`), `VERSION = 3`. TOUTES les notes sont empilées : note neuve, note d'avant (clé
  absente, fausse, vraie ou illisible — `CLES_ANCIENNES = ("empilement",)` : relue SANS EFFET, jamais une raison de refuser
  la note), note de l'ancien module Prédalles, fichier Streamlit importé (les deux écrans). La comparaison avec les
  anciennes valeurs Streamlit n'a plus cours.
- **L'avis d'ouverture.** Une note d'avant (version < 3) dont les chiffres changent (`dalle.chiffres_changes` : une couche
  posée par le site sur la nappe principale — pour une note dont l'option était cochée, seulement un renfort secondaire,
  S3) reçoit sous `CLE_AVIS` l'avis « Note d'avant le 09/11/2026 : les nappes sont désormais posées l'une sur l'autre (la
  direction secondaire, base et renforts, sur la principale) ; A_s,req peut différer de l'ancienne note. » ; un fichier
  Streamlit repris, l'avis « Fichier Streamlit : … l'écran Streamlit les posait au même niveau ; A_s,req peut différer du
  sien. » (`dalle.AVIS`). `dalle.resultat` le dit en TÊTE du résultat (sous le nom de la dalle : `C.message`, un avis comme
  ceux du moteur, pas une erreur rouge — le même composant aux trois formats ; au téléphone, l'onglet Résultat) et le
  RETIRE de l'état : la note rendue, que l'écran enregistre en version 3, ne le redit pas. Une note d'avant dont les axes
  secondaires sont tous saisis, une note du 09/11 cochée sans renfort secondaire : rien ne change, rien n'est dit.
- **La règle (`dalle_positions.axe_auto`)**, sans option : direction PRINCIPALE (base ET renforts) et faces inférieures
  d'une prédalle → `distance_auto` du moteur ; direction SECONDAIRE — sa base ET ses RENFORTS (S3) —, faces inférieures et
  supérieures d'une dalle coulée, faces supérieures d'une prédalle → `e_p + ⌈Ø_p/2⌉₀,₅ + ⌈Ø/2⌉₀,₅` (e_p : axe effectif de la
  base principale de la même face ; Ø_p son Ø, Øl d'un treillis ; Ø celui de la couche posée). Un axe saisi n'est jamais
  déplacé. Pourquoi la formule de la base vaut pour le renfort : les deux moteurs posent un renfort dans le LIT de la base
  de sa face (`distance_auto` ne dépend pas du rang, hors faces inférieures d'une prédalle) — son décalage relatif est nul
  et le reste une fois empilé (vérifié pour chaque Ø : `tests/test_api_web_dalle_positions.py` (b)).
- **Exemple chiffré** (c = 3 cm, jeu 0, principale Y, base Y Ø12/150, base X treillis 10/10/100/100) : base inf. X
  4,0 + 1,0 + 0,5 = 5,5 cm ; renfort 3 Ø12 en inf. X : 4,0 + 1,0 + 1,0 = **6,0** cm (avant : 3 + 1,0 = 4,0, au niveau de la
  base Y, sous sa propre nappe) ; renfort Ø25 : 6,5 ; renfort de la principale (inf. Y) : 4,0, inchangé. Prédalle h_p = 6 :
  renfort 3 Ø16 en sup. X : 4,0 + 1,0 + 1,0 = 6,0 (avant : 4,0) ; renfort inf. X : 6 + 1,0 = 7,0, INCHANGÉ (déjà sur la
  prédalle, dans le lit de sa base à 6,5 : la règle du moteur le pose ainsi, S3 n'y change rien).
- **Mesures S3** (le site, avant → après, la face du renfort ; sens de la sécurité : le renfort monte, d baisse,
  A_s,req augmente) : dalle 20 cm, principale Y, treillis 10/10/100/100, renfort 3 Ø12 inf. X, M_x,inf 25 : axe 4,0 → 5,0,
  d 15,65 → 15,35 cm, A_s,req 532,5 → 542,9 mm² (+2,0 %) ; renfort Ø25/150 inf. X, M_x,inf 60 : axe 4,5 → 5,5, d 15,50 →
  14,69, A_s,req 1 290,3 → 1 361,1 (+5,5 %) ; dalle 25 cm, principale X, base sup. X Ø16/150, renfort 4 Ø20 sup. Y,
  M_y,sup 40 : axe 4,0 → 6,0, d 20,42 → 19,19, A_s,req 652,9 → 694,7 (+6,4 %) ; prédalle 22 cm, renfort 3 Ø16 sup. X,
  M_x,sup 30 : axe 4,0 → 5,0, A_s,req 564,4 → 578,6 (+2,5 %) ; prédalle, renfort 3 Ø12 inf. X : 434,3 → 434,3 (inchangé).
- **Ce qui suit l'axe, sans code de plus** : la copie envoyée au moteur (`pour_moteur`), le champ « auto · 6 », la légende
  de la coupe (« X · 3 Ø12 · axe à 60 · sur la base Y »), les points et traits des coupes (lus dans `couches[j]["e"]`),
  la formule « e » d'une prédalle (`e_p + ⌈Ø_p/2⌉ + ⌈Ø_s/2⌉`, Ø_s celui du renfort), le sélecteur de la nappe de base
  (`candidats_base` : une base principale candidate déplace aussi les renforts secondaires posés sur elle), le résumé et
  le verdict. Contrôle croisé dessin ↔ calcul : `tests/test_api_web_dalles_fusion.py` (§ 7) et
  `tests/test_api_web_dalle_positions.py` (e). L'avertissement « nappes X et Y au même niveau » de la dalle coulée ne sort
  plus que d'un axe saisi, et le dit (« Rendre l'axe saisi à l'automatique (↺) pose la direction secondaire sur la
  principale »).
- L'ancien module « predalle » du registre (`api_web/predalle.py`, servi pour les anciens liens et le harnais, sa page
  redirigée vers les dalles) calcule toujours au même niveau : il n'est pas atteignable depuis l'écran (toute note
  « predalle » est relue par le module des dalles) ; ses 9 photos du gel restent identiques.

**Révision du 09/11/2026, nuit (S5 — relecture du 09/11 : la nappe secondaire repose sur la couche principale la plus
haute qu'elle croise)** — suite de la décision de l'ingénieur « les nappes toujours l'une sur l'autre ». Le défaut, non
conservatif et préexistant (FUS4) : quand la direction principale porte, dans le lit de sa base, un RENFORT plus épais
que la base (le cas du mode par défaut « Treillis + renforts » : 3 Ø12 sur un treillis Ø10), la nappe secondaire —
base et renforts S3 — était posée sur la base seule : ses barres traversaient le renfort, dans le calcul et sur la
coupe, et rien ne le disait. Ce qui est REMPLACÉ dans la règle ci-dessus (et aux § 4.1, 4.2, 4.4) :
- **La règle (`dalle_positions.axe_auto`)** : direction SECONDAIRE (dalle coulée inf. et sup. ; prédalle sup.), base et
  renforts → `max(e_k + ⌈Ø_k/2⌉₀,₅) + ⌈Ø/2⌉₀,₅`, le max sur les couches principales k de la même face qui sont dans le
  **lit de contact** (`lit_contact`), chacune à son axe EFFECTIF (saisi, sinon l'automatique du moteur), Ø_k le Ø de ses
  barres ou de ses fils porteurs (Øl d'un treillis). La couche qui porte (`appui`) est la plus haute ; à égalité, la
  base — la formule d'avant S5, au flottant près (aucune photo du gel ne bouge). La prédalle, faces INFÉRIEURES : la
  règle du type, inchangée (voir plus bas). Un axe saisi n'est jamais déplacé.
- **Le lit de contact**, défini sur les axes effectifs, en convention du moteur (dessous e − ⌈Ø/2⌉, dessus e + ⌈Ø/2⌉) :
  il part de la base principale ; une autre couche principale en est si son dessous ne dépasse pas le dessus du lit
  (contact ou chevauchement : la nappe secondaire, posée au moins sur ce dessus, ne peut pas passer dessous), et le
  dessus du lit monte au sien, jusqu'à ce que plus rien n'entre (un renfort qui chevauche un renfort du lit en est).
  **Tolérance** : `TOL_LIT` = 1e-6 cm, les arrondis des flottants, AUCUNE marge — un 2e lit principal posé sur la nappe
  secondaire a son dessous au moins 0,05 cm au-dessus du dessus du lit, même placé aux Ø réels (⌈Ø/2⌉ dépasse Ø/20 de
  0,45 cm au plus, fil de 5 mm au moins) : une marge l'avalerait. Avec les axes automatiques, le moteur pose toutes
  les couches principales au même dessous (enrobage + jeu) : toutes sont du lit ; seule une couche SAISIE peut en sortir.
  *Relecture du 09/11/2026 (nuit)* : une couche principale AUTOMATIQUE est du lit quoi qu'il arrive — elle est posée sur
  les cales du premier lit, elle ne peut pas être un 2e lit. Le lit d'avant la jugeait sur son seul dessous : sous une
  base SAISIE plus bas que l'automatique (Ø8 saisie à 3,4 avec un jeu premier lit de 1 cm : dessus 3,9), un renfort 3 Ø20
  automatique (5,0 : dessous 4,0) sortait du lit, et la nappe secondaire, posée sur la base à 4,4, le traversait sans
  rien dire : A_s,req 427,4 au lieu de 493,8 mm² (− 13,4 %). Corrigé (`lit_contact`), test (g) « relecture S5 » et
  mutation 11.
- **Cas limites, écrits et testés** (`tests/test_api_web_dalle_positions.py` (g)) : renfort principal SAISI dans un 2e
  lit (Ø25 à 9 cm, dessous 7,5 > 4,0) : il ne porte rien ; saisi au contact exact (Ø12 à 5,0, dessous 4,0 = dessus de
  la base) : il porte ; 0,01 cm plus haut : un autre lit ; chaîne (R1 Ø25 automatique, R2 Ø12 saisi à 6,5 qui chevauche
  R1) : R2 entre par R1 et porte (8,0), sans R1 il est hors du lit (4,5) ; base saisie plus haut que son renfort : la
  base porte ; plusieurs renforts : le plus haut, le premier à égalité ; demis arrondis égaux (Ø16 et Ø20) : rien ne
  change ; renfort treillis : par son fil ; jeu premier lit : il entre une fois, par la principale. **Ce que la règle ne
  dit pas** : une couche principale SAISIE dont le dessous tombe dans l'épaisseur de la nappe secondaire (au-dessus du
  lit, sous le dessus de la secondaire) n'est pas du lit et rien ne le signale — l'axe saisi est celui qu'on a voulu ;
  un avertissement serait possible (à trancher).
- **Prédalle, faces INFÉRIEURES** (vérifié et mesuré) : la base principale est dans la peau, la base secondaire ET les
  renforts principaux sur la prédalle à h_p + ⌈Ø/2⌉ — un renfort principal y croise la base secondaire au même niveau.
  Le moteur le DIT déjà (« croisement_sur_predalle » : « Saisir son axe pour en tenir compte ») et la question 11 de
  l'ingénieur (« empiler les couches basses de la prédalle ») est ouverte : S5 n'y touche pas. Mesure (h = 22, h_p = 6,
  renfort 3 Ø20 inf. Y, base inf. X treillis Ø10, M_x,inf = 20) : aujourd'hui axe 6,5, d 15,5, A_s,req 430,1 mm² et
  l'avertissement ; empilée sur le renfort (axe saisi 8,5 : 7,0 + 1,0 + 0,5), d 13,5, A_s,req 493,8 (+ 14,8 %).
- **Ce qui suit l'axe** : la copie envoyée au moteur (`pour_moteur`) ; `annoter` pose sur chaque couche empilée `appui`
  (le rang de la couche qui porte), `sur` (« la base Y », « le renfort Y », « le renfort R2 Y » : formule et pose) et
  `sur_court` (« la base Y », « renfort Y » : légendes — `nom_appui`, une source) ; la légende des coupes dit « axe à 55 ·
  sur renfort Y » (forme courte : « sur le renfort Y » débordait sur la colonne des supérieures, vu sur la capture S5 ;
  « sur la base Y » inchangé) ; la formule « e » d'une prédalle prend e_p, Ø_p, ⌈Ø_p/2⌉ de la couche qui porte et dit
  « posée sur le renfort Y » (« posée sur la base principale » inchangé) ; la pose de la couche aussi ; l'aide de l'axe ;
  le sélecteur (`candidats_base` : une base secondaire candidate repose sur le renfort ; une base principale candidate
  plus mince que le renfort ne fait plus descendre la secondaire, une plus épaisse la porte) ; le résumé, le verdict ;
  la note PDF (`note_dalle` reçoit `POS.axes` : planche, coupe, d, A_s,req — contrôlée à chaque note contre le moteur
  du site) ; l'avis d'ouverture d'une note d'avant dont l'option était cochée (`dalle.chiffres_changes` : sa base
  secondaire, que l'option posait sur la base principale, change dès qu'un renfort plus haut la porte).
- **Mesures** (le site, avant → après, la face secondaire ; c = 3, jeu 0, B500, γs 1,5, C30/37 ; sens de la sécurité :
  aucun axe ne descend, A_s,req ne baisse jamais) :

  | Cas | Axe(s) secondaire(s) | d (cm) | A_s,req (mm²) | Écart | État |
  |---|---|---|---|---|---|
  | CSP h 20, « Treillis + renforts » : treillis 10/10/100/100, renfort 3 Ø12 inf. Y, M_x,inf 20 | 4,5 → 5,5 | 15,50 → 14,50 | 430,1 → 459,8 | + 6,9 % | ok → ok |
  | idem, renfort 3 Ø25 | 4,5 → 6,5 | 15,50 → 13,50 | 430,1 → 493,8 | + 14,8 % | ok → ok |
  | CSP h 25, principale X : sup. X Ø10/150 + 4 Ø20 ; sup. Y Ø8/200 + 3 Ø12 (S3), M_y,sup 35 | 4,5 ; 5,0 → 5,5 ; 6,0 | 20,21 → 19,21 | 577,2 → 607,2 | + 5,2 % | **ok → NON CONFORME** (A_s 590,6) |
  | CSP h 20 : inf. Y Ø10/150 + 3 Ø12 + 2 Ø20 ; inf. X Ø8/200 + 3 Ø16 (S3), M_x,inf 25 | 4,5 ; 5,0 → 5,5 ; 6,0 | 15,15 → 14,15 | 550,2 → 589,1 | + 7,1 % | ok → ok |
  | Prédalle h 22 / h_p 6 : sup. Y treillis 8/8/150/150 + 3 Ø16, M_x,sup 25 | 4,5 → 5,5 | 17,50 → 16,50 | 476,2 → 505,1 | + 6,1 % | nok → nok |
  | Prédalle, renfort 3 Ø20 inf. Y (sur la prédalle) | 6,5 → 6,5 | 15,50 | 430,1 | 0 (règle du type) | ok, avertissement gardé |
  | CSP, renfort 3 Ø25 SAISI à 9 cm (2e lit) | 4,5 → 4,5 | 15,50 | 430,1 | 0 | ok |
  | CSP, inf. Y Ø16/150 + 3 Ø20 (demis égaux) | 5,5 → 5,5 | 14,50 | 459,8 | 0 | ok |

  Balayage (test (g)) : 513 couples base × renfort (chaque treillis et chaque Ø de base, chaque Ø de renfort ; dalle
  inf. et sup., prédalle sup.) — l'axe secondaire n'est jamais sous celui de la base seule ni à travers le renfort ;
  A_s,req secondaire monte dès que ⌈Ø_r/2⌉ > ⌈Ø_b/2⌉, égal sinon. Le gel (`tests/test_reference_refonte.py`, 239) ne
  bouge pas : aucun cas gelé n'a de renfort principal plus haut que sa base (aucun regel).
- **Ce qui n'est pas fait** : les notes ENREGISTRÉES sur le site entre l'intégration du 09/11 et S5 (version 3) dont un
  renfort principal dépasse la base se rouvrent avec leurs nouveaux chiffres (plus sûrs) SANS avis : un avis supposerait
  une version 4 de l'état (une marque de format) — non fait sans l'accord de l'ingénieur.

*Texte d'avant (FUS4, REMPLACÉ ci-dessus)* — Clé d'état `empilement` (vrai/faux), par note (comme γ_s et le jeu,
rubrique 4 — Options, case « Nappes empilées : direction secondaire posée sur la principale ») :
- note NEUVE : vrai ;
- note d'avant (sans la clé), note de l'ancien module Prédalles, fichier Streamlit importé : FAUX → positions
  automatiques du moteur, résultats identiques à l'octet (les gels `tests/test_reference_refonte.py` dalle/predalle et
  la parité Streamlit restent vrais sans regel) ;
- l'ingénieur peut la cocher sur une ancienne note : il voit alors les chiffres changer, en connaissance de cause.
Faux → `axe_auto` du site = celui du moteur, et le dessin montre les deux directions au même niveau (ce que le calcul
fait vraiment : le dessin ne ment jamais).

### 4.4 Le dessin

Coupes (`dessin_dalle._coupe`, `dessin_predalle`) : barres coupées en points/cercles, barres parallèles en traits ;
CHAQUE couche à son `e` réel (lu dans le résultat du moteur) — les deux directions à des niveaux distincts quand
l'empilement est actif, les renforts à leur axe ; cotes de d par direction = `h − y_G` du calcul ; prédalle : peau
hachurée, pose respectée. Plans : inchangés (X horizontales, Y verticales).

**Construit (FUS4)** — vérifié : les deux dessins lisaient DÉJÀ `couches[j]["e"]` du résultat pour chaque couche (points,
traits, cotes d'axes, ordre de recouvrement du plan) ; le « même plan » des deux directions venait du moteur, pas du
dessin. Ce qui change : la légende dit « posée sur la base Y » pour une couche que le site a empilée (`empilee`, posé par
`annoter` ; sans chiffre : les empreintes des vues gelées ne bougent pas), la légende de la prédalle ne dit plus « axe
saisi » que d'un axe saisi dans l'état, et la cote de d de la prédalle suit la face TENDUE de la direction coupée (sup.
quand son moment l'emporte), comme celle de la dalle — le même dessin pour les deux types du module fusionné. Les
pièces cliquables (`data-sel="s<i>.<face>-<j>"`, `s<i>.beton`) sont inchangées. Prouvé dans
`tests/test_api_web_dalle_positions.py` (e) : barres Y 15 mm sous les barres X avec l'option, écart de 5 mm (les Ø du
moteur) sans ; chaque couche à `h − e` ou `e` au dixième de mm ; et par une mutation (le dessin qui lirait l'automatique
du moteur ramène les deux directions au même plan : le contrôle rougit). Les 9 cas gelés dalle et 9 prédalle de
`tests/test_reference_refonte.py` restent identiques (option absente → fausse).

*Révision du 09/11/2026 (NAP)* : « quand l'empilement est actif » se lit « toujours » ; le renfort de la direction
secondaire est dessiné dans le lit de sa base (« · sur la base Y » dans la légende), à l'axe que le moteur a reçu.

*Révision du 09/11/2026, nuit (S5)* : la légende dit sur quoi la couche repose réellement — « · sur la base Y », ou
« · sur renfort Y » quand un renfort principal plus haut que la base la porte (forme courte : « sur le renfort Y »
débordait sur la colonne des supérieures, vu sur la capture). Prouvé à l'écran (harnais, filtre « nappes », PC et
téléphone) : un renfort principal 3 Ø20 sous la nappe X — le bas du trait de la base X au contact du haut des points
du renfort (écart mesuré 0,0 mm ; avant S5 : 10 mm dedans), la ligne de légende qui s'arrête avant la colonne des
supérieures (6,8 px au PC, 3,2 px au téléphone).

## 5. Tests ciblés (pas de suite complète ; ni catalogue ni assemblages)

- `tests/test_api_web_dalle.py`, `tests/test_api_web_predalle.py` (suivent), nouveau `tests/test_api_web_dalles_fusion.py`
  (type, h_pre et complément, refus géométriques, relecture des trois formes d'état, import Streamlit des deux
  fichiers, moteur choisi, avertissements prédalle présents, nappes : mode déduit, écriture transposée, incohérence
  signalée, positions empilées vs option fausse = chiffres identiques, dessin : niveaux distincts lus dans le SVG,
  mutations).
- `tests/test_dalle_moteur.py`, `tests/test_predalle_moteur.py` : inchangés et VERTS (preuve que les moteurs n'ont pas
  bougé) ; `tests/test_reference_refonte.py` : les 9 photos « predalle » restent vertes sans regel (relecture d'une note de
  prédalles, option fausse) ; les 9 photos « dalle » partent d'une note NEUVE (`nouveau`), donc empilée (§ 4.3) : elles
  sont REGELÉES avec `--geler --accepter="…"` — raison écrite dans `regels` (état version 2, clés `type`, `empilement`,
  chemins `nappes`, base secondaire posée sur la principale : d plus petit, h_min 3,5 → 4,5 cm, avis
  `nappes_confondues` disparu) — dans un commit DISTINCT de celui de la fusion.
- `tests/test_site_web.py`, `tests/test_site_maquettes.py`, `tests/test_api_web_modules.py` : suivent (un module de
  moins sur l'accueil, redirection).
- Harnais : `module:dalle`, `formats:dalle` (PC, tablette, téléphone), captures regardées ; `module:predalle` devient
  la redirection.
- *(PDF-D, 09/11/2026)* `tests/test_note_dalle.py` : la note PDF (§ 6) — parité, texte des planches, écran, arrondis,
  coupe, verdicts, observations et leur place (balayage des épaisseurs, aucun texte de la coupe chevauché), PDF lu par
  pdftotext, routeur, aller-retour par la lecture Streamlit, conditions du navigateur, MUTATIONS ;
  `tests/test_api_web_dalle.py` suit (la dalle a désormais `telecharger`).

## 6. La note PDF sur le site (S4, lot PDF-D, 09/11/2026)

*Décision de l'ingénieur du 09/11/2026 : « la note PDF de la dalle se fait SUR LE SITE et dit les chiffres du site ».
Avant, elle ne se faisait que dans Streamlit, qui pose X et Y au même niveau sans le dire (A_s,req ≈ 6,5 % trop faible
sur la direction secondaire) : le PDF était moins sûr que l'écran.*

### 6.1 Comment elle se fait

- `api_web/note_dalle.py`, sur le modèle de la poutre (`api_web/note_poutre.py`) : le générateur VALIDÉ de Streamlit,
  `modules/export_pdf_dalle.py` (mise en page `ndc_pdf`), n'est PAS réécrit ni modifié. `valeurs(d, did)` traduit une
  dalle du site (les deux types) en valeurs de l'écran Streamlit (clés `dal<i>_…`, `dal<i>_sec<j>_…`), et
  `export_pdf_dalle._collecter_resultats` refait le calcul (`_compute_section`, les formules de l'écran).
- LES AXES : chaque couche reçoit son axe EFFECTIF du site (`dalle_positions.axes` : nappes posées l'une sur l'autre,
  renfort secondaire S3 compris, règle de pose de la prédalle, axe saisi) comme une distance saisie (`dist_auto` faux,
  `dist_axe` écrit SANS arrondi — l'écran prédalle de Streamlit les figeait au centième) ; le centre de gravité imposé
  passe par `ycdg`, la peau d'une prédalle par `h_pre`. Le calcul de la planche est alors celui de l'écran.
- La traduction est l'INVERSE de la lecture d'un fichier Streamlit (`beton/dalle/fichier.lire`) : relues par elle, les
  valeurs redonnent chaque grandeur du site (aller-retour testé) ; les libellés de type viennent de `EN.LIBELLES_TYPES`
  (une source, l'inverse de `FI.TYPES_ECRAN`, lié par un test). Aucune fonction de l'écran Streamlit n'est recopiée.
- CONTRÔLE À CHAQUE NOTE (`note_dalle.ecarts`) : la planche est comparée au moteur du site (axes, y_G, d, A_s,req,
  A_s,min, A_s, états, h_u,min, h_min, τ, τ_adm, état de la section) ; un écart au-delà de 1e-9 (relatif) LÈVE : la note
  n'est pas produite, le message dit la dalle, la section et la grandeur. Jamais une note qui dit autre chose que
  l'écran.
- `api_web/dalle.telecharger(etat, "note_pdf", infos)` (même forme que la poutre ; le routeur passe {nom, chantier,
  indice, date} → page de garde : projet = chantier ou nom, partie = nom ou « Dalles en béton armé ») ; l'avis
  d'ouverture (`CLE_AVIS`) n'y entre pas. Une note, une planche par section de chaque dalle, dans l'ordre ; chaque dalle
  traduite à part (γs et jeu propres à la dalle si une note relue en porte de différents).

### 6.2 Ce que le site ajoute à la planche (le générateur ne le sait pas)

- l'ÉTAT de l'écran (`plaque_compact.etat_bloc`) : « Non vérifié » d'abord, « Données incomplètes » sans aucun effort
  (jamais « Vérifié »), « Vérifié » sinon ;
- les NIVEAUX DESSINÉS de la coupe = les axes du calcul : le générateur écartait deux axes trop proches pour le dessin
  (`_coupe_dalle_depuis_R._niveaux`) et dessinait un renfort, posé dans le plan de sa base, plus haut qu'il n'est
  compté ; comme la coupe de l'écran, chaque couche est à l'axe qui entre dans d ;
- le VERDICT d'une nappe non conforme alors que A_s ≥ A_s,néc : acier excessif (« A_s > A_s,max ») ou d utile ≤ 0 — le
  générateur y écrivait « A_s < A_s,requise », faux ;
- les OBSERVATIONS de l'écran, en texte (sans balisage) : les avertissements de la section (enrobage nul lu 3 cm,
  nappes au même niveau par un axe saisi, d utile ≤ 0, base hors de la prédalle, couches croisées sur la prédalle…) et,
  sur la première planche de chaque dalle, ceux de la dalle entière et le PÉRIMÈTRE de la prédalle (ce qui n'est pas
  vérifié : phase provisoire, cisaillement d'interface, flèche, fissuration — `api_web/dalle.texte_perimetre`, une
  source pour l'écran et la note). Où : sous les données du panneau de gauche (le style du générateur hérité, `_data`
  suivi des observations) ; si la place manque, la coupe cède le juste nécessaire sans descendre sous `coupe_min`
  (mesuré : sous elle, les étiquettes de la coupe se chevauchent — 225 pt à h = 20 cm, 280 pt à 60 cm) ; sinon les
  observations passent en fin de planche, et la planche qui déborde alors passe SEULE en trois colonnes, ses titres
  raccourcis (« Armatures sup. — principale » : ceux du générateur ne tiennent pas dans une colonne sur trois). Les
  24 cas du test tiennent en deux colonnes ; le dernier recours est testé à part.

### 6.3 La parité (`tests/test_note_dalle.py`)

- 24 cas, 30 sections (CSP et prédalle, une et plusieurs dalles et sections, γs différents, principale X, treillis par
  nappe, treillis + renforts, trois renforts, gros Ø, moments sur les deux faces, axe saisi et y_G imposé, enrobage nul,
  B400/γs 1,15/C20/25/bande de 50 cm, acier excessif, d ≤ 0, hauteur insuffisante et τ > τ_adm, nappes au même niveau,
  verrouillée, peau de 4 cm, couches croisées, renfort 4 Ø20 secondaire empilé), dont 53 couches empilées par le site sur
  130 : la planche rend les grandeurs du moteur du site au 1e-9 près ; son TEXTE écrit 1 415 chiffres de l'écran (h_u,min,
  h_min, d, A_s,req, A_s,min, A_s/m, A_s, A_s,néc, τ, τ_adm, efforts) dans l'ordre des nappes de l'écran ; l'écran montre les
  mêmes A_s,néc et A_s (240 chiffres) et le même état de chaque section.
- ARRONDIS : la note (générateur, `fn` : arrondi au pair sur la valeur binaire) écrit les mm² entiers, les cm (d, h_min,
  y_G) et les kN·m au dixième, les N/mm² au centième ; l'écran (`C.fr` : demi vers l'extérieur) les mm² entiers, d et
  y_G au centième dans la fenêtre de calcul, les efforts tels que saisis. Les deux n'écrivent un nombre différemment que
  s'il tombe EXACTEMENT à mi-chemin en binaire (aucun dans les cas) ; un effort saisi au centième (28,25 kN·m) s'écrit
  au dixième dans la note.

### 6.4 Ce que la note ne dit pas, ce qui n'est pas vérifié

- Le TAUX η de chaque vérification (l'écran l'affiche ; la note dit A_s ≥ A_s,néc, comme celle de Streamlit) ; le plan
  des nappes ; la formule d'axe de chaque couche (la note donne d = h − y_G) ; la mention « sur la base Y » (ou « sur
  renfort Y », S5) ; l'étoile du sélecteur. Aucun de ces manques ne contredit l'écran : la note reçoit les axes du site,
  S5 compris (`tests/test_note_dalle.py` : « S5 de bout en bout », calcul, planche, coupe et PDF lu par pdftotext, et
  sa mutation).
- DALLE CSP : la FLÈCHE et la FISSURATION ne sont pas vérifiées (ni à l'écran ni dans la note, comme dans Streamlit) ;
  l'écran ne le dit pas encore — le jour où `api_web/dalle.texte_perimetre` le dira pour la dalle CSP, la note le dira
  aussi (même source). La prédalle le dit (son périmètre). *Révision du 10/11/2026 : c'est fait — § 7.*
- Le bouton « Note PDF » de la dalle reste grisé tant que `site/web/construire.NOTE_PDF` ne nomme pas « dalle » et que le
  paquet de la note (`paquet_note`) n'a pas `modules/export_pdf_dalle.py` et `modules/treillis.py` (intégration). La note
  est prouvée dans les conditions du navigateur par un sous-processus calqué sur `_SONDE_NOTE` (fontTools, accélérateurs
  C, streamlit, numpy, matplotlib refusés ; Pillow factice ; les six polices du paquet) : aucun fichier de reportlab en
  plus de la poutre. Essayée aussi dans le VRAI Pyodide (Chromium, 09/11/2026), sur un site construit à part avec
  NOTE_PDF += « dalle » et le paquet étendu (sans toucher à `construire.py`) : « Note PDF » au PC (ruban) et au
  téléphone (barre du bas, menu) télécharge la note, CSP puis prédalle, en 1,1 s puis 0,2 s ; son texte (pdftotext) est
  IDENTIQUE à celui de la note faite en CPython pour le même état. Le harnais (`verifier_site.js`) ne la contrôle pas
  encore : c'est à l'intégration de l'y ajouter.
  *Révision du 09/11/2026 (intégration de nuit)* : c'est fait — `NOTE_PDF` nomme « dalle », le paquet de la note est
  relevé sur une note riche de la dalle (`construire.NOTES_RELEVE`, CSP puis prédalle) et porte les deux fichiers ; le
  harnais (filtre « note_pdf ») télécharge la note au PC, à la tablette et au téléphone et y retrouve les noms des
  dalles, h_min, A_s,néc et A_s que l'écran montre (docs/site/MODULES_SITE.md § 4 quinquies, « Le branchement »).
- Les fiches de la bibliothèque (BIB4) ne sont pas dans la note de la dalle (la dalle n'en a pas encore).

## 7. Petits chantiers du 10/11/2026 (dalle)

- **Nom de plus de 60 caractères** (`AP.NOM_MAX`) : à la RELECTURE (note enregistrée, `.stab`, fichier Streamlit), un nom
  de dalle ou de section trop long faisait refuser TOUTE la note (`_bornes_site`) — elle ne s'ouvrait plus. Il est
  désormais coupé à la limite (`dalle._couper_noms`) et l'avis le dit une fois, à l'ouverture (`CLE_NOMS`, comme
  `CLE_AVIS` : posé par `normaliser` ou la reprise Streamlit, dit et retiré par `resultat`, jamais dans la note PDF). La
  SAISIE garde son refus clair (`AP.nom_saisi` : « 60 caractères au plus (61 saisis) »).
- **Dalle CSP : ce qui n'est pas vérifié** — la flèche et la fissuration (`dalle.PERIMETRE_CSP`, une source pour l'écran
  et la note : `texte_perimetre`), dit comme la prédalle, par le même composant (`C.aide`) aux mêmes places (sous le
  résultat, au pied de chaque fenêtre de calcul) et sur la première planche de la note PDF (`note_dalle.observations`,
  sans toucher au générateur : la planche tient toujours en deux colonnes, 43 contrôles de `tests/test_note_dalle.py`).
- **L'avis d'ouverture au téléphone** : les avis (note d'avant, nom coupé) partent aussi dans la réponse (`avis`, une
  liste) ; là où le résultat ne se VOIT pas à l'ouverture — le téléphone s'ouvre sur l'onglet Données —, `js/etabli.js`
  dit le même texte dans le message de la page (`#et-alerte`, le composant des trois formats), qui reste jusqu'à ce qu'on
  le ferme. Au PC et à la tablette, le résultat se voit : l'avis n'y est pas répété.
