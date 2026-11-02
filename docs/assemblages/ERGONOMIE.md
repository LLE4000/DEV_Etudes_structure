# Charte ergonomique des écrans d'assemblage

Charte du 31/10/2026, amendée le même jour (§0), consolidée le 01/11/2026 (§17, étape 7 du plan) et ÉTENDUE le
02/11/2026 à la famille Poutre – Poteau (§19). Document de référence des **sept** modules et du calcul manuel ;
le code cite ses paragraphes. **Ce qui est écrit « les cinq écrans » dans le corps de la charte vaut désormais
pour les SEPT** : les trois attaches poutre–poteau ont été écrites à la charte, sans exception et sans
amendement — elles n'ont rien demandé de neuf (§19).

Charte finale (31/10/2026), issue de la proposition D « trois formats » (gagnante chez deux juges sur trois) greffée des idées retenues de A, B et C et corrigée des manques relevés par le contrôle de complétude. Modules : doubles cornières d'âme (asm_), plat d'âme soudé (fpl_), platine d'about partielle (pab_), platine de pied de poteau sur béton (ppb_, modes complet et manuel). Streamlit 1.66.1. Aucune clé de session, aucune valeur d'option enregistrée, aucun fichier projet, aucun octet de moteur ne change — à une exception décidée et testée : le défaut du critère de la platine manuelle (§12).

Sources : le retour du bureau du 31/10/2026 (cité dans REFONTE_UX.md §14), l'inventaire des quatre écrans (87 / 70 / 66 / 70 clés relevées, documents de conception hors dépôt), les captures « avant », le prototype de grille mesuré (lot 0), le code lu (lignes citées), le moteur relancé ce jour (compute(defaults()) : voir §12). Les nombres de cette charte sont mesurés ; les lignes de code citées sont celles du 31/10/2026 (amendement 6 : le harnais se relit au texte, pas au numéro).

## 0. Amendements du 31/10/2026 (décisions du chef de projet — ils s'imposent à la suite)

Les amendements 7, 8, 10 et 11 règlent la conduite de la mise en œuvre (exécutants, commits, captures, serveur de capture) ; ils sont gardés tels qu'ils ont été écrits, pour l'histoire des décisions.

1. **Tables du mode manuel (boulons, éléments, raidisseurs) : exception écrite à P1.** Une rangée de
   table peut porter jusqu'à 5 cases quand la 5e est le bouton 🗑️ (Type · Profilé · x · y · 🗑️). Le
   contrôle (6) « aucune ligne d'objet > 4 widgets » ne s'applique qu'aux lignes d'objet (`lg_`),
   pas aux rangées de table (`tr_`). Sous 1351 px la rangée se replie en 2 + 2 + 🗑️.
2. **Mortier tg = 0 (pied de poteau) : destination garantie.** tg reste une cote de la coupe A-A
   quand tg > 0 ; quand tg = 0 la fenêtre « Massif » (toujours cliquable) porte tg, et le panneau
   Cotes (dessin fixe) liste tg dans tous les cas. Un chk le fige (tg = 0 → `tg` dans la fenêtre
   Massif ET dans le panneau Cotes).
3. **Rédacteur (id_red) imprimé au cartouche et dans l'export texte des CINQ écrans** (note.py,
   manuel/note.py, texte.py du pied ; texte.py du plat d'âme et de la platine d'about). Un chk par
   module. C'est « le harnais a trouvé » : un champ saisi et jamais lu.
4. **Mode manuel, état de la bascule : les deux états restent dits.** Sous l'en-tête, une caption
   courte dans les deux cas : « Vérification complète : disponible pour cette disposition. » ou
   « Vérification complète : indisponible — <raison>. » (les mots « disponible » / « indisponible »
   que lisent test_platine_manuelle_ecran l.98-99, l.108, l.116, l.130 sont conservés).
5. **Tableau 3.3 replié** : test_pied_poteau_ecran l.134-138 (« Pinces – EN 1993-1-8 Tableau 3.3 » dans
   le texte) est classé « le harnais suit » : le titre de l'expander devient « Pinces et entraxes
   (Tab. 3.3) » et le test lit ce titre. Même règle pour tout titre de section replié.
6. **Citations de lignes de tests** : §15 cite des numéros parfois faux. Règle : le harnais est RELU
   avant d'être lancé (grep du texte, pas du numéro) ; chaque rouge est classé dans le message de
   commit.
7. **Ordre et parallélisme de la mise en œuvre** : étape 1 (kit) + étape 2 (plat d'âme, gabarit)
   ensemble, dans l'arbre de travail, par UN exécutant, vérifié par un relecteur (captures aux quatre
   largeurs, suites), corrigé, puis commité par le chef de projet. Ensuite trois exécutants EN
   PARALLÈLE dans des worktrees : platine d'about ; doubles cornières ; pied de poteau (complet +
   manuel + défaut élastique). Pendant cette phase, les fichiers du kit (acier/assemblages/carte.py,
   ui_commun.py, note_kit.py, resultats_commun.py, acier/projet_ui.py, acier/registre.py) sont GELÉS :
   un module qui a besoin d'une fonction de kit l'écrit localement (préfixe `_kit_` dans son module)
   et le signale dans son rapport ; le chef de projet consolide après la fusion.
8. **Commits** : rien n'est commité par les agents ; chaque étape se termine par un rapport (ce qui a
   changé, suites rejouées avec leurs comptes, rouges classés, captures produites avec leurs chemins).
   Le message de commit est préparé par l'agent dans un fichier scratchpad/ergo/commit_<étape>.txt
   (sans identifiant de modèle).
9. **Docs** : cette charte → docs/assemblages/ERGONOMIE.md, sans le paragraphe « État des captures »,
   avec ces amendements en §0, daté. Les « Écarts assumés » (√(Ac1/Ac0) hors note, etc.) y sont listés.
10. **Captures « après »** : protocole des captures « avant » (fenêtre agrandie à la hauteur du contenu
    Streamlit, PC 1440, tablette 1024, téléphone 390 ; + 1280 pour le palier intermédiaire), fichiers
    scratchpad/ergo/apres/apres_<module>_<largeur>.png et notes apres_<module>_note_p1.png / p2.png ;
    le script versionné tools/captures_assemblages.js mesure ce que l'œil voit (checkVisibility,
    scrollWidth == innerWidth, cases ≥ 120 px, titres dans leur rangée, barre d'outils ≤ 3 rangées).
11. **Serveur de capture** : port 8597, script de relance
    /tmp/claude-0/-home-user/092afadd-796a-54d4-b802-037ae1cad731/scratchpad/st_restart.sh (à relancer
    après chaque modification de code avant de capturer). Playwright et Chromium sont installés
    (require('playwright') fonctionne). Python : /usr/bin/python3.12.
12. **Rien d'autre ne change** : moteurs, clés, fichiers projet, notes (hors HYPOTHÈSES, cartouche
    Rédacteur, conclusion conditionnée du manuel), exports texte (hors Rédacteur), benchmarks (hors
    crit=CRIT_EC3 explicite dans les VAL-M).

## 1. Principes

- **P1 — Un objet = une ligne de 3 ou 4 cases, titre discret à gauche.** Jamais 5 widgets sur une ligne d'objet (mesuré : base flex 128 px, colonne 660 px à 1440 → 5 × 128 + 32 = 672 > 660). Jamais une ligne d'objet à une seule case : on fusionne ou on sort de la carte. Les lignes conditionnelles sont des continuations « ↳ » à cases fixes de 150 px, sous-titre gris dans la colonne des titres.
- **P2 — Un seul arbre Streamlit pour les trois formats ; seule la grille change, par une règle CSS du kit sur nos propres clés** (.st-key-<p>lg_*, bt_*, tr_*, th_*). Exception nommée : les tables du mode manuel rendent deux fois les libellés (en-tête de colonnes ≥ 1351 px, libellés de rangée en dessous) et en cachent un par CSS — mesuré par captures.js avec checkVisibility.
- **P3 — Un « ? » seulement s'il apprend quelque chose.** `_aide()` ne recopie plus jamais le libellé long de GROUPES ; un « ? » n'existe que pour une entrée de AIDES (≤ 160 caractères). Les conventions communes vont sur le TITRE du bloc (titre_bloc(help=)). Aucun « ? » sur Nuance, Profilé, Boulon, Classe, dimensions, efforts, identification.
- **P4 — Les choix sont courts à l'affichage, les valeurs brutes ne changent pas** (AFFICHAGE + format_func ; constantes MODE_*, CRIT_*, APPUI_*, PERIM_*, TYPE_*, FORME_*, NAT_*, VÉRIFICATION / PRÉDIMENSIONNEMENT inchangées, set_value(valeur brute) inchangé). Aucune option affichée > 28 caractères (toutes ≤ 18 dans le mapping).
- **P5 — Une forme par clé, partout** : LIBELLE[k] (≤ 22, unité comprise, capitale initiale) est lu par la carte, les fenêtres d'objet du dessin, le panneau Cotes et les chips d'alerte. Exceptions nommées : le pied de poteau a un LIBELLE_MODE (« Tige » en vérification complète / « Boulon » en calcul manuel ; « Visserie par tige » / « par boulon ») ; la grille 2D du pied dit « Files (x) · Rangées (y) » là où les poutre-poutre disent « Rangées n1 · Files n2 ».
- **P6 — La géométrie se règle sur le dessin, jamais dans la carte** (règle testée dans les quatre suites) ; le mode manuel rejoint la règle (hef, dh quittent la carte, §4.5). Une fenêtre d'objet est une VUE de la même clé, pas un second lieu : les doublons (a_w dans Plat/Platine, perim dans Efforts, a_cc dans Massif) sont retirés et le lieu qui reste est nommé.
- **P7 — Rien ne se perd, et un test le prouve** : chaque clé de CHAMPS ∪ GEOMETRIE_DESSIN ∪ {ui_niveau, ui_composant} a une destination ; les onglets, exports et boutons restent (§10) ; un texte retiré de l'écran va dans un « ? », un expander ou l'onglet Méthode, jamais nulle part.
- **P8 — Aucun nombre de mémoire** : tout chiffre de cette charte est mesuré (prototype, moteur relancé, code lu). Le harnais suit dans le même commit ; chaque rouge est classé « suit » ou « a trouvé » (§15).
- **P9 — Un seul signal par état** : le prédimensionnement = bande ocre + « ◆ » sur les libellés pilotés ; plus de caption sous le dessin ni de phrase ajoutée aux aides.
- **P10 — Même charte, même ordre, dans les cinq écrans** : EN-TÊTE (mode…) → PROFILÉS / POTEAU → pièce d'attache → BOULONS / TIGES → (MATÉRIAUX) → EFFORTS ELU ; puis « Paramètres avancés » et « Cartouche (note et plan) » repliés.

## 2. Grille et kit (mesuré sur proto_grille.py, 31/10/2026)

Colonne droite mesurée : 660 px à 1440, 576 à 1280, 440 à 1024, 358 à 390 (≈ (fenêtre − 176) × 0,528). Ligne d'objet = `st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=f"{p}lg_{bloc}_{n}")` ; titre dans `st.container(key=f"{p}bt_{bloc}", width=76)` sur la première ligne du bloc ; widgets `width="stretch"`. Règle CSS du kit (st.markdown, lisible par le harnais), quatre paliers :

| Fenêtre | Titre | Cases par rangée (ligne de 4) | Largeur | −/+ |
|---|---|---|---|---|
| ≥ 1351 | à gauche (76 px) | 4 | 130 (mesuré) | visibles |
| 1181–1350 | au-dessus | 4 (flex-basis 126) | 126–138 (4 × 126 + 24 = 528 ≤ 531 à 1181) | visibles (≥ 120) — à mesurer à 1280 et 1181 au lot 0 |
| 641–1180 | au-dessus | 2 (flex-basis 205) | 212 à 1024 (mesuré) | visibles |
| ≤ 640 | au-dessus | 2 (flex-basis 160) | 171 à 390 (mesuré) | visibles |

```
@media (max-width:1350px){ .st-key-<p>lg_* > [data-testid="stLayoutWrapper"]:has([class*="st-key-<p>bt_"]){flex:0 0 100% !important}
  .st-key-<p>lg_* > [data-testid="stElementContainer"]{flex-basis:126px !important} }
@media (max-width:1180px){ .st-key-<p>lg_* > [data-testid="stElementContainer"]{flex-basis:205px !important} }
@media (max-width:640px){ .st-key-<p>lg_* > [data-testid="stElementContainer"]{flex-basis:160px !important} }
```
- Rangée de 3 : cases de 187 px à 1440 (libellés ≤ 20). Rangée de 4 : 130 px (libellés ≤ 16). Continuation « ↳ » : widgets `width=150`, 3 par rangée à côté du sous-titre à 1440 (76 + 8 + 3 × 150 + 16 = 550 ≤ 660 ; 5 dimensions = 3 + 2, mesuré variante F), 2 par rangée ailleurs, jamais étirées ; une continuation peut n'avoir qu'une case (↳ PLAT, ↳ groupe de la platine d'about).
- Titre de bloc ≤ 11 caractères à 0,72 em ; « EFFORTS ELU » passe sur deux lignes dans 76 px (constaté proto2_1440) : un titre peut occuper deux lignes, jamais être coupé ; captures.js vérifie qu'il ne dépasse pas la hauteur de sa rangée (le rendu « sous la ligne » du proto, risque 12 de D, se règle au lot 0).
- EN-TÊTE de carte : `st.container(horizontal=True, wrap=True)` de segmented_control `width="content"` (+ Taux cible · Ø minimal · Ø maximal en prédim) ; peut passer sur 2 rangées (manuel : 3 segmented ≈ 650 px) ; captures.js exige seulement : aucun débordement, ≤ 2 rangées à 1440, ≤ 3 à 390.
- Paramètres avancés : même grille (lignes d'objet de 2, 3 ou 4 ; continuations ↳ à 150 px pour glissement, αcc).
- Tables du manuel (boulons, éléments, raidisseurs) : rangées = continuations à clé `{p}tr_{bloc}_{i}` à libellés visibles ; en-tête `{p}th_{bloc}` (st.markdown, help=) ; CSS : `@media (min-width:1351px){.st-key-<p>tr_* [data-testid="stWidgetLabel"]{display:none}} @media (max-width:1350px){.st-key-<p>th_*{display:none}}`. Tables ≥ 6 colonnes de l'onglet Vérifications : `tableau_html(..., defilant=True)` (overflow-x:auto). Onglets : `segmented_control(wrap=True)`.
- Seuils de tools/captures_assemblages.js (versionné, issu de tel_carte.js + mesure_proto.js), à 1440 / 1280 / 1024 / 390 : `scrollWidth == innerWidth` ; case de nombre ≥ 120 px ; 4 cases par rangée ≥ 1181, 2 en dessous ; barre d'outils 1 rangée ≥ 1024, ≤ 3 rangées (≤ 130 px) à 390 ; titre de bloc dans sa rangée ; bandeau et premier dessin dans les 844 premiers px après le titre à 390 ; onglets sans débordement ; captures apres_<module>_<format>.png.

**Mesuré au lot 0 et retenu (`carte.css_kit`, source unique ; écarts avec le tableau ci-dessus).** `gap="small"`
vaut 16 px dans un conteneur horizontal, pas 8 : les lignes du kit passent à 8 px (4 × 124 + 3 × 8 = 520 ≤ 523 px de
colonne à 1181 — la colonne y mesure 523 et non 531) ; base flex 124 (et non 126) au palier 1181–1350 ; à 1440 les
cases d'une ligne de 4 mesurent 138 px ; les continuations ont leur propre clé `<p>ls_` (cases fixes de 150 px) ;
« COEFFICIENTS » (12 lettres) se coupait dans 76 px : le sous-bloc des avancés s'appelle « COEFF. γ ». Les règles
ajoutées le 01/11/2026 (case ≤ 187 px, continuation alignée, titres au-dessus de leur rangée, panneau Cotes 4 / 3 / 2,
tables serrées) sont au §17.

## 3. Libellés, aides, options, vocabulaire

- Libellés : ≤ 16 (rangée de 4), ≤ 20 (rangée de 3, continuation), ≤ 22 partout ailleurs ; unité tirée de UNITE ; capitale initiale ; grugeage = c / dc,sup / dc,inf dans les trois modules (GROUPES de la platine d'about et du plat aligné, vue seulement) ; lh_P reçoit « ℓh ailes A (mm) » (plus de 46 caractères) ; « Rangées n1 · Files n2 » (poutre-poutre), « Rangées n1 (2 files) » (platine), « Files (x) · Rangées (y) » (pied).
- Options affichées (valeurs brutes inchangées) : Vérification / Prédimensionnement ; Grande aile côté → Principale / Secondaire ; Nature de NEd → ELU / Tying ; Mode → Calcul manuel / Vérif. complète ; Critère → Élastique / Plastique ; Appui → Bande c / Toute la platine ; Périmètre → EC3 seul / EC3 + EC2-4 ; Forme → I / H · Tube RHS / SHS ; Ancrage → Tête / plaque · Crosse · Scellée (ETA) ; types d'élément → Profilé I / H · Tube RHS / SHS · Tube CHS · Plat sur chant · Rond plein · Rectangle plein. Booléens : selectbox Oui / Non (pas de toggle : seconde clé et sept rouges pour sept champs rares).
- AIDES de champ (textes exacts, ≤ 160, = ceux du mapping complet) : mode_calc « Prédimensionnement : boulons, rangées et pièce d'attache proposés (onglet Prédim) puis vérifiés ; les champs ◆ sont alors pilotés par la proposition. » · trou « Surdimensionné : Fb,Rd × 0,8 (EN 1993-1-8 Tab. 3.4) et d0 à saisir dans Paramètres avancés › TROUS. » · cat « A : pression diamétrale. B : sans glissement à l'ELS. C : sans glissement à l'ELU (boulons précontraints 8.8 ou 10.9). » · k_ser « Rapport des efforts ELS / ELU ; 1,0 = hypothèse sécuritaire. » · r_n « Information d'exécution : non utilisé dans les résistances ; dessiné sur l'élévation. » · lt_ok (dc) « Poutre secondaire maintenue au déversement : condition de la règle de stabilité locale du grugeage (MSB P5 §4.2.5). » (fpl : « … ; un plat long sur poutre non maintenue demande un examen à part. » ; pab : « … ; Non : stabilité d'ensemble de la poutre grugée à justifier à part. ») · d0_u « Vide = d0 normal du boulon (Tab. 3.3) ; à saisir pour un trou surdimensionné. » · opt_exc « Non : pratique ECCS n°126 / MSB P5. Oui : variante conservatrice, M = (VEd/2)·eP par cornière. » · k_rot « 0,8 : MSB P5 §4.2.1.2 / SCI P358 ; 1,0 pour désactiver. » · opt_blf « βLf des assemblages longs, EN 1993-1-8 §3.8 (Lj > 15d). Les guides ne l'appliquent pas aux attaches d'âme ; Oui = variante conservatrice. » · expo « Assemblage exposé aux intempéries : active les pinces maximales 4t + 40 mm (Tab. 3.3). » · g_M1 « Instabilités (déversement du plat long). NBN EN 1993-1-1 ANB : 1,10 ; valeur recommandée de l'EN : 1,00. » · g_Mu « 1,1 : valeur des guides (MSB P5 §2.3, SCI P358) ; l'EN 1993-1-8 ne la fixe pas — à confirmer pour la Belgique. » · nat_N « ELU : fy/γM0 et γM2, combiné avec VEd. Tying : force de chaînage seule, fu/γMu (MSB P5 §2.3). » · ui_composant « Décoché : dessins fixes ; les cotes se modifient dans le panneau « Cotes » en tête de la colonne de droite. » · visserie « Rondelles et écrous par boulon, portés au cartouche du plan. Ex. : « 1 rondelle + 2 écrous (contre-écrou) », « 2 rondelles + 1 écrou ». » (pied complet : « Rondelles, écrou et contre-écrou par tige, portés au cartouche du plan de principe. ») · hp_u « Recommandation des guides : hp ≥ 0,6·h de la secondaire (maintien en torsion). » · tp_u (fpl) « Ductilité (rotule par ovalisation) : tp ≤ 0,5·d du boulon (SCI P358 / MSB P5). » ; (pab) « Platine mince : tp ≤ 0,36·d·√(fub/fy) (§6.4.2, ECCS n°126) ; 10 ou 12 mm usuels, moins de 8 mm déconseillé (MSB P5 §2.1). » · a_w (fpl) « Deux cordons, un par face du plat ; pleine résistance a ≥ tp·fy·βw·γM2/(2·fu·γM0) (§4.5.3.3). » ; (pab) « Deux cordons, un de chaque côté de l'âme ; pleine résistance 0,46 / 0,48 / 0,55·tw en S235 / S275 / S355 (MSB P5 §2.2). » · d_top « 0 = dessus des semelles au même niveau ; aussi par glissement vertical de la poutre secondaire. » · d_nt « 0 = semelle supérieure non grugée. » · d_nb « 0 = semelle inférieure non grugée. » · g_h « Jeu entre l'âme porteuse et l'about de la secondaire ; aussi par glissement horizontal de la poutre. » · p3_u « Symétrique par rapport à l'âme de la secondaire ; la rondelle doit dégager le cordon. » · gA_u « Trusquinage : talon de la cornière → 1re file. » · a_P « Cordon vertical en bout d'aile sur toute la hauteur hc. » · typ_a « Tête / plaque : hef et dh se règlent sur la coupe A-A. Crosse : adhérence. Scellée : valeurs de l'ETA (τRk, k1, NRk,s, VRk,s). » · a_cc « Coefficient de l'ANB belge pour fcd = αcc·fck/γc ; 0,85 par défaut, à confirmer. » · C_f « Frottement platine / mortier, déduit du cisaillement des tiges (V EC3). » · L_c « Hauteur du poteau pour la classification de rigidité du pied (EN 1993-1-8 §5.2.2.5), en information. » · xN « Point d'application de N, Mx, My, Vx, Vy ; (0 ; 0) = centre de la platine. Un élément excentré crée le moment N·e tout seul. » · mode (ppb) « Calcul manuel : boulons, éléments et raidisseurs posés librement, épaisseur requise comme à la main. Complète : grille régulière, Tab. 6.7, EN 1992-4. » · crit « Élastique : Von Mises ≤ fy/γM0 sans effet de levier — ≈ 1,22 × l'épaisseur plastique (√1,5), jusqu'à 1,73 (√3) avec levier. Plastique : tronçon en T Mpl. » · appui « Bande c autour des parois (EN 1993-1-8 §6.2.5, sûr) ou toute la platine (rigide : efforts des tiges comparables aux logiciels d'ancrage). » · perim « EC3 seul : platine, tiges, poteau, cordons et appui fjd décident ; béton d'ancrage (EN 1992-4) en information. EC3 + EC2-4 : il entre dans le verdict. » · tiges (x de la table) « Origine au centre de la platine, x vers la droite, y vers le haut (vue en plan). Chaque boulon se déplace aussi sur le dessin. » · elems « Élément soudé = profilé I/H, tube, plat, rond ou rectangle posé en (x ; y). Rotation et dimensions : dans sa fenêtre sur le dessin (ici si le dessin est fixe). » · raid « Plat vertical soudé sur la platine et l'élément, parti de (x ; y) dans le sens donné ; h sert aux coupes. »
- AIDES de BLOC (titre_bloc(help=) ou sous-titre ↳), textes exacts : EFFORTS ELU (dc) « Efforts dans la secondaire : NEd > 0 traction ; HEd perpendiculaire à l'âme ; MEd à la face de l'âme porteuse (articulé : 0, pris en valeur absolue). » ; (fpl) « Efforts dans la secondaire : NEd > 0 traction ; MEd à la face de l'âme porteuse (articulé : normalement 0, pris en valeur absolue). » ; (pab) « Efforts dans la secondaire : VEd tranchant ; NEd > 0 traction (boulons tendus, platine fléchie §6.2.4) ; sa nature, ELU ou tying, apparaît dès NEd > 0. » ; (ppb) « N > 0 compression ; Mx > 0 comprime le côté y > 0, My > 0 comprime x > 0 ; T = torsion. Appliqués en (xN ; yN), (0 ; 0) = centre de la platine. » · COEFFICIENTS (dc) « Valeurs recommandées de l'EN 1993-1-8 ; γM0 selon l'ANB belge (1,00). γM2 boulons : boulons, pression diamétrale, soudures ; γM2 nettes : sections nettes. » ; (fpl) « ANB belge : γM0 1,00 et γM1 1,10 (NBN EN 1993-1-1 ANB). γM2 boulons : boulons, pression diamétrale, soudures ; γM2 nettes : sections nettes, bloc. » ; (pab) « γM0 1,00 (ANB belge). γM2 boulons : boulons, pression diamétrale, soudures ; γM2 nettes : sections nettes, rupture de bloc ; γMu : tying seulement. » ; (ppb) « γM0 1,00 et γM2 1,25 (ANB belge) ; γc = γMc 1,50 ; αcc 0,85 (fcd = αcc·fck/γc) — hypothèse à confirmer sur le texte NBN. » · PRÉDIM « Pinces et entraxes proposés en multiples de d0 ; minima normatifs e ≥ 1,2·d0, p1 ≥ 2,2·d0 (Tab. 3.3). Règles de détail modifiables. » · ↳ AILES B « Ailes B sur l'âme de la secondaire (groupe S, deux plans). Soudées des deux côtés : assemblage peu déformable, à réserver à des cas particuliers. » · ↳ ETA « Valeurs de l'ETA de la tige scellée (état du béton, classe) ; k1 = kcr ou kucr (7,7 / 11,0 à défaut) ; NRk,s et VRk,s : 0 = calculé par l'outil. » · ÉLÉMENTS et RAIDISSEURS : textes elems / raid ci-dessus.
- Partent dans Méthode › « Données de bibliothèque » (ajouté aux trois modules poutre-poutre), sans tutoiement : l'aide de la nuance (« S355 : fu = 510 MPa (EN 1993-1-1 Tab. 3.1) ; « S355 (fu 490) » : valeur d'autres références, à caler sur l'ANB ») et celle de la cornière (« Dimensions usuelles EN 10056-1, à contrôler avec le catalogue du fournisseur »). Le texte des anciens boutons ❔ devient le premier paragraphe de Méthode, « Ce que l'outil vérifie ». Les notes mortes de GROUPES sont supprimées (grep avant).
- « ? » visibles dans la CARTE au cas par défaut (périmètre : en-tête + lignes d'objet + continuations visibles + « ? » de bloc ; hors tête du dessin et avancés) : cornières 5 (Mode, Trou, Catégorie, ↳ AILES B, EFFORTS ELU), plat 4, platine 4, pied complet 4 (Mode, Périmètre, Ancrage, EFFORTS ELU), manuel 7 (Mode, Critère, Appui, Ancrage, EFFORTS ELU, ÉLÉMENTS, x de la table). Avant : 22 / 14 / 22 / 24 / 24.

## 4. Les blocs (cas par défaut, PC ≥ 1351 ; sous 1181 chaque ligne se replie en 2 × 2, titre au-dessus)

### 4.1 Doubles cornières (asm_) — 7 lignes visibles, 21 cases
```
EN-TÊTE     [ Vérification | Prédimensionnement ] ?        (+ Taux cible · Ø minimal · Ø maximal en prédim)
PROFILÉS    | Principale ▾ HEA 400 | Nuance ▾ S355 | Secondaire ▾ HEA 300 | Nuance ▾ S355 |
  ↳ principale / ↳ secondaire (si Personnalisé) | h (mm) | b (mm) | tw (mm) | tf (mm) | r (mm) |   (150 px : 3 + 2)
CORNIÈRES   | Cornière ▾ L100x100x10 | Nuance ▾ S355 | Grande aile côté ▾ (cornière inégale ou Personnalisé seulement) |
  ↳ perso    | Grande aile (mm) | Petite aile (mm) | Épaisseur (mm) | Congé (mm) |
BOULONS     | Boulon ▾ M20 | Classe ▾ 8.8 | Trou ▾ Normal ? | Catégorie ▾ A ? |
  ↳ AILES A (groupe P)   | Fixation ▾ Boulonnée | Rangées n1 3 | Files n2 ▾ 1 |
  ↳ AILES B (groupe S) ? | Fixation ▾ Boulonnée | Rangées n1 3 | Files n2 ▾ 1 |
  ↳ glissement (cat ≠ A) | μ | ks (Tab. 3.6) | ELS / ELU ? (cat B) |
EFFORTS ELU ? | VEd (kN) 125 | NEd (kN) 0 | HEd (kN) 0 | MEd (kNm) 0 |
▸ Paramètres avancés   ▸ Cartouche (note et plan)
```
MODE ET FIXATIONS disparaît (mode en en-tête, fixations avec le groupe qu'elles gouvernent) ; VISSERIE sort au cartouche ; l'orientation n'apparaît que pour L100x75x8, L120x80x8, L120x80x10, L150x90x10, L150x100x10 ou Personnalisé (5 des 18 cornières, mesuré dans DB.cornieres) ; elle reste dans la fenêtre Cornières.

### 4.2 Plat d'âme (fpl_) — 6 lignes, 15 cases
```
EN-TÊTE     [ Vérification | Prédimensionnement ] ?
PROFILÉS    | Principale ▾ | Nuance ▾ | Secondaire ▾ | Nuance ▾ |      (+ ↳ perso)
  ↳ PLAT     | Nuance du plat ▾ S355 |                                   (une case de 150 px, alignée à gauche)
BOULONS     | Boulon ▾ | Classe ▾ | Trou ▾ ? | Catégorie ▾ ? |
  ↳ groupe   | Rangées n1 3 | Files n2 ▾ 1 |        ↳ glissement (cat ≠ A) | μ | ks | ELS / ELU ? |
EFFORTS ELU ? | VEd (kN) | NEd (kN) | MEd (kNm) |
```

### 4.3 Platine d'about (pab_) — 6 lignes, 13 cases
```
EN-TÊTE     [ Vérification | Prédimensionnement ] ?
PROFILÉS    | Principale ▾ | Nuance ▾ | Secondaire ▾ | Nuance ▾ |      ↳ PLATINE | Nuance platine ▾ |
BOULONS     | Boulon ▾ | Classe ▾ | Trou ▾ ? | Catégorie ▾ ? |          ↳ groupe | Rangées n1 (2 files) 3 |
  ↳ glissement (cat ≠ A) | μ | ks | ELS / ELU ? |
EFFORTS ELU ? | VEd (kN) 125 | NEd (kN) 0 | Nature de NEd ▾ ELU / Tying ? (seulement si NEd > 0) |
```

### 4.4 Pied de poteau — vérification complète (ppb_) — 20 cases
```
EN-TÊTE     [ Calcul manuel | Vérif. complète ] ?   [ EC3 seul | EC3 + EC2-4 ] ?
POTEAU      | Forme ▾ I / H | Profilé ▾ HEA 300 | Nuance ▾ S355 |     ↳ perso | h | b | tw | tf | r |  (RHS : h | b | t paroi)
MATÉRIAUX   | Nuance platine ▾ | Béton ▾ C30/37 | Fissuré ▾ Oui | Massif hc (mm) 600 |
TIGES       | Tige ▾ M24 | Classe ▾ 8.8 | Ancrage ▾ Tête / plaque ? |   ↳ grille | Files (x) 2 | Rangées (y) 2 |
  ↳ ETA ? (scellée) | τRk (MPa) | k1 cône | NRk,s (kN) | VRk,s (kN) |
EFFORTS ELU ? | NEd (kN) 300 | Mx,Ed (kNm) 50 | My,Ed (kNm) 30 |      ↳ | Vx,Ed (kN) | Vy,Ed (kN) | TEd (kNm) |
```
PÉRIMÈTRE (un champ) et son doublon dans la fenêtre Efforts disparaissent ; PLATINE et BÉTON fusionnent en MATÉRIAUX.

### 4.5 Pied de poteau — calcul manuel (ppb_, défaut)
```
EN-TÊTE     [ Calcul manuel | Vérif. complète ] ?  [ Élastique | Plastique ] ?  [ Bande c | Toute la platine ] ?
            caption « Vérification complète indisponible — <raison>. » seulement si la bascule est impossible
MATÉRIAUX   | Nuance platine ▾ | Béton ▾ C30/37 | Massif hc (mm) 600 |
BOULONS     | Boulon ▾ M24 | Classe ▾ 8.8 | Ancrage ▾ Tête / plaque ? |
  ↳ table    | n° | x (mm) ? | y (mm) | 🗑️ |  1 | −200 | −200 | 🗑️ | … (une rangée par boulon)
  ↳ actions  | ➕ Boulon | Poser une grille… ▾ (Files (x) · Rangées (y) · wx (mm) · wy (mm) · Poser la grille) |
ÉLÉMENTS ?  | Type ▾ Profilé I / H | Profilé ▾ HEA 300 | x (mm) 0 | y (mm) 0 | 🗑️ |
  ↳ É1       | Rotation (°) ▾ | dimensions |   seulement si Personnalisé, type sans catalogue, rotation ≠ 0 ou dessin fixe
  ↳ actions  | ➕ Élément soudé |
RAIDISSEURS ? | x | y | Sens ▾ | 🗑️ |  ↳ R1 | L (mm) | t (mm) | h (mm) |  (en-tête seulement s'il y en a) · ➕ Raidisseur
EFFORTS ELU ? | NEd (kN) | Mx,Ed (kNm) | My,Ed (kNm) |      ↳ | Vx,Ed (kN) | Vy,Ed (kN) | TEd (kNm) |
```
hef et dh quittent la carte : cote de la coupe A-A dès qu'un boulon existe (manuel/schemas.py l.381-386 : « if pos », dh si > 0 → étiquette « dh 0 » éditable comme « Δz 0 »), fenêtre Boulons, fenêtre Platine (AJOUT : toujours cliquable, donc atteignable même sans boulon et dessin interactif coché — le trou relevé par le contrôle), panneau Cotes (qui liste GEOMETRIE_DESSIN ∩ champ_visible, plus seulement les cotes émises) ; CLE_PAR_COTE reçoit « dh ». xN / yN descendent dans Paramètres avancés › APPLICATION et restent dans la fenêtre Efforts. La légende « Repère : origine… » devient le « ? » de x (mm) (en-tête ≥ 1351 px, libellé de rangée en dessous).

## 5. Paramètres avancés (expander, même grille)
Poutre-poutre : GRUGEAGE | Rayon grugeage (mm) ? | Déversement empêché ▾ ? | · TROUS | d0 imposé (mm) ? | Filetage cisaillé ▾ | · COEFFICIENTS ? | γM0 | [γM1 (ANB) ? — plat] | γM2 boulons | γM2 nettes | [γMu (tying) ? — platine, si tying] ; ↳ glissement (cat ≠ A) | γM3 (ELU) | γM3,ser (ELS) | · MODÈLE | [Excentricité gA ▾ ? | k sur Fv,Rd (P) ? — cornières] | βLf joints longs ▾ ? | Intempéries ▾ ? | · ACIER PERSO | fy (MPa) | fu (MPa) | βw | · PRÉDIM ? | e1 = k·d0 | p1 = k·d0 | e2 = k·d0 |.
Pied : COEFFICIENTS ? | γM0 | γM2 | γc béton | γMc (EN 1992-4) (complet) | ↳ | αcc ? | · APPUI | Cf,d frottement ? | Mortier fg (MPa) | · RIGIDITÉ (complet) | Hauteur Lc (mm) ? | Contreventé ▾ | · APPLICATION (manuel) | xN (mm) ? | yN (mm) | · ACIER PERSO. Les captions ANB (163 et 190 caractères) disparaissent : le « ? » du bloc COEFFICIENTS les dit une fois. L'expander s'ouvre (🔴) quand l'alerte sélectionnée y pointe (inchangé). La case « Dessin interactif » n'y est plus (§7).

## 6. Cartouche (note et plan) — ex-« Identification (note et export) »
| Repère | Rédacteur | Visserie par boulon (« par tige » en vérification complète) ? | puis ↳ hors projet | Projet | Date | — affichés seulement si st.session_state.nom_projet / date sont vides (la condition même de interface._infos : dans un projet ces cases sont écrasées). Visserie : VALEUR par défaut « 1 rondelle + 1 écrou » conservée (pas un placeholder : le cartouche l'imprime et test_assemblages_interface l.139-140 la lit en session). id_red de la platine d'about est enfin imprimé au cartouche (note.py, texte.py) — défaut trouvé, figé par un chk neuf. Plus de « ? » sur Repère.

## 7. Tête de la colonne dessin et captions d'usage (textes exacts)
`st.container(horizontal=True, wrap=True)` : radio « Vue simple | Édition | Cotations complètes » (<p>ui_niveau) + case « Dessin interactif » ? (<p>ui_composant, déplacée des avancés). Captions sous les vues, une par module : cornières « Touche une cote pour la modifier, touche une pièce (poutre, cornière, boulon, cordon, VEd) pour ses paramètres ; glisse la poutre portée (gh, Δz). » (test l.144-145 : « touche une pièce », « glisse la poutre portée » conservés) ; plat « … (poutre, plat, boulon, cordon, VEd) … ; glisse la poutre portée (gh, Δz). » ; platine d'about « Touche une cote pour la modifier, touche une pièce (poutre, platine, boulon, cordon, VEd) pour ses paramètres ; glisse la poutre portée (Δz). » ; pied complet « Vue en plan : aire efficace (bleu), zone comprimée (rouge), axe neutre, C et T des tiges. Touche une cote ou une pièce (poteau, platine, tiges, massif, mortier, efforts). » ; manuel « Vue en plan : aire d'appui (bleu), zone comprimée (rouge), C, axe neutre, T et bras m de chaque boulon, bande comprimée (mEd). Glisse un boulon, un élément ou un raidisseur ; touche une pièce ou une cote. » En prédim, plus de caption « ◆ pilotés… » (P9). Fenêtres d'objet : titres courts (Groupe S, Groupe P, Cordons A/B, Cornières, Poutre secondaire, Plat, Platine, Soudure, Boulons, Tiges, Massif, Mortier, Efforts), libellés et aides depuis LIBELLE / AIDES ; la fenêtre Poutre secondaire reçoit gh, Δz, dc,sup, dc,inf, c (cles_de['beamS'] étendu : test l.155 suit) ; a_w ne reste que dans Soudure ; αcc quitte Massif ; perim quitte Efforts.

## 8. Barre d'outils et titre (PC et téléphone)
Titre court « ## Doubles cornières d'âme » / « Plat d'âme soudé » / « Platine d'about partielle » / « Pied de poteau sur béton » + `st.caption("<ancien titre long> · version x.y")` (titre long mot pour mot : les quatre tests d'en-tête restent verts, zéro « ? » ajouté) ; le bouton ❔ (73 px au téléphone) et la cellule « Version » disparaissent ; leur texte va dans Méthode « Ce que l'outil vérifie ». `projet_ui.barre_outils(prefixe, …)` partagée : `st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="center", key=f"{p}barre")`, boutons AVEC libellés en `width="content"` : ◀ Projet · 🏠 Accueil · 💾 Enregistrer · 📄 PDF · [A1 · Doubles cornières ▾] (width=230, champ `court` du registre : « Doubles cornières », « Plat d'âme », « Platine d'about », « Pied de poteau » ; `titre` reste pour le sommaire et les signets) · 🔄 Réinitialiser en fin de rangée (loin de 💾 : pas d'infobulle au doigt) ; 📂 Ouvrir hors projet ; ⬇️ Télécharger sous la barre après génération ; clés inchangées. Cible : 1 rangée à 1440 et 1024, ≤ 3 rangées (≤ 130 px) à 390 au lieu de 7 boutons empilés (≈ 510 px) ; bandeau et dessin dans le premier écran. `bloc_statut_ligne` perd le nowrap du statut (453 px mesurés).

## 9. Bandeau
Complément « — η max · <synthese.court(gov)> » (le court de la note, les cinq synthese.py ont `court`) au lieu de R.gov.label (88 caractères non coupés en pied complet). Manuel : « — épaisseur requise 19,8 ≤ tp 25 mm · 79,3 % · tronçon en T du boulon » (critère gouvernant parmi tT, tS, tc, tb) sans répéter « Épaisseur requise de la platine », et quatre chips T / S / c / b à l'écran à côté de « Platine » (synthese.ELEMENTS de la note inchangé). La ligne d'information sous le bandeau (« En information, hors verdict : … ») reste telle quelle (test_pied_poteau_ecran l.104-108 la lit). Rouges classés « suit » : test_assemblages_interface l.114 (« Pression diamétrale – âme de la poutre secondaire » → court), test_pied_poteau_ecran l.104-108 (« Épaisseur requise de la platine » → « Épaisseur requise tp »), test_platine_manuelle_ecran l.87-88 (forme ET nombre : 15,6 → 19,8) et l.102-104 (R0.t_req 15,56 → 19,81).

## 10. Onglet Vérifications, onglets et exports (rien ne se perd)
- Onglets inchangés : Vérifications · Prédim · Note · Benchmark · Méthode (pied complet : Répartition ; manuel : Efforts des tiges), clés <p>ui_onglet et contenus inchangés : « Texte : toutes les vérifications » + « ⬇️ Exporter le texte (.txt) », « 📄 Rapport détaillé (3 pages) » (asm), « Charger ces données dans l'outil », « Solutions proposées » / « Appliquer » ; le test du kit les énumère (contrôle 1).
- Légende EC / COMP / INT : plus en tête ; après la dernière table une caption d'une ligne (badges + « EC formule de l'Eurocode · COMP modèle complémentaire reconnu · INT interprétation de l'outil », natures DU MODULE : le pied garde « EC + INT / INT modèle d'ingénieur ») puis un expander fermé « Légende et références » = notation.LEGENDE_REFERENCES (source unique avec le pied de la note) ; fonction ui_commun.legende_natures(natures). Badges dans les cellules conservés.
- « Non revérifiés ici » et « Sans objet dans cette configuration » : textes intégraux dans un expander fermé « Périmètre de la vérification — n non revérifiés · m sans objet » après les tables ; le manuel garde « Non vérifiés ici » (il ne vérifie pas du tout cordons, raidisseurs, poteau, béton) ; AppTest lit les expanders dans md(at) : les chk restent verts.
- Tableau 3.3 : ligne « Pinces et entraxes (Tab. 3.3) : conforme » + expander fermé (ouvert d'office dès une ligne NON OK, boutons Localiser inchangés) ; la caption des règles « e ≥ 1,2·d0 ; … » vit dans l'expander ; lignes nommées par x.lab du moteur (PINCES_COURT n'est pas repris à l'écran : il perdrait le sens de la pince).
- Prédim : une seule caption « Touche « Appliquer » pour recopier une solution dans les données et la géométrie ; elle devient modifiable comme une saisie. Seule la vérification détaillée fait foi. » (le mode actif est dit par le bandeau).
- Titres de tables : « Cornières — aile A / aile B », « Paramètres retenus » ; commentaires internes des colonnes Réf. (« même modèle que … », « ancre VAL-A ») en commentaires Python.
- Manuel : le bloc d'état et la table « Platine — épaisseur requise et ses quatre critères » restent en tête ; les quatre tables de détail dans un expander OUVERT « Le détail, comme à la main » (md(at) les contient : test l.91-93 vert).

## 11. Note PDF — HYPOTHÈSES
Règle : ≤ 3 lignes de ≤ 110 caractères, composées par `note_kit.empaqueter(segments, n_max=3, larg=110)` (segments ÉCRITS, jamais coupés ; formes courtes prises du dernier au premier ; ValueError sinon → le test rougit). Ligne 1 modèle et critère, ligne 2 cas particuliers et grandeurs du modèle absentes du reste de la page, ligne 3 coefficients. Textes exacts et variantes : section « hypotheses » (mesurés avec la vraie fonction wrap du kit PDF, Carlito 6,2 pt, 210 pt : 4 à 6 lignes imprimées au lieu de 5 à 13). Gardés parce qu'ils ne vivent nulle part ailleurs sur la page : z et « MS = VEd·z + |MEd| = 6,25 kNm » (test_assemblages_note l.132 : une fois), Aeff et ΣT (pied complet), kc / kb, cordon déduit 9,1 mm et leff ≤ L/2 + m (manuel). Retirés parce que déjà sur la page : Ip, FRd, Fv,Rd, Fb,Rd, 0,55·tw (tables, formules, Paramètres retenus), C et pmax (vue en plan, table Appui), niveau 2 (ligne fc2 des tables), seuil de rigidité (alerte « rig » imprimée sous le bandeau), phrases de méthode (table « LE CALCUL À LA MAIN », onglet Méthode). Perte assumée par écrit : √(Ac1/Ac0) = 1,80 n'est plus imprimé sur la note complète (il reste dans Paramètres retenus à l'écran et dans l'export texte). Aucune phrase interdite du 21/09/2026 (« répartition élastique », « interaction quadratique », « MSB Part 5 ») ne revient : vérifié sur le jeu. L'export texte et le rapport détaillé des cornières gardent leur propre lignes_hypotheses (texte.py l.48, rapport.py l.275, corrigés figés) : décision écrite, révision datée au lot « vocabulaire ».

**Textes finaux (relancés le 01/11/2026 sur le cas par défaut de chaque écran, longueurs en caractères).**
Point 10 de la consolidation : un § SANS indication est de l'EN 1993-1-8 (légende du pied de la note
« EC3 = EN 1993-1-8 sauf indication ») ; un § du guide porte donc TOUJOURS « MSB P5 » — « (§4.2.1.2) » et
« (§4.2.4) » nus, à côté de « (§4.5.1) » de l'EN, se lisaient comme l'EN. « MSB P5 » une fois par ligne
quand c'est possible (`note_kit.empaqueter(…, une_fois=MSB)`), jamais retiré : la variante tying à
hp < 1,36·p3 de la platine d'about le garde deux fois sur sa ligne 2, faute de place (écart écrit, contrôlé).

- Doubles cornières (78 / 104 / 107) : « Articulé : rotule à la face de l'âme porteuse (MSB P5 §4.2.1.1) ;
  z = 50,0 mm. » · « Groupe S : MS = VEd·z + |MEd| = 6,25 kNm ; groupe P : cisaillement centré, 0,80·Fv,Rd
  (MSB P5 §4.2.1.2). » · « Grugeage : flexion de la section réduite (MSB P5 §4.2.4) ; γM0 1,00 · γM2 1,25
  (boulons, soudures, nettes). »
- Plat d'âme (108 / 105 / 86) : « Articulé : rotule à la face de l'âme porteuse (MSB P5 §3.2) ; z = 50,0 mm ;
  le groupe reprend VEd·z + |MEd|. » · « Plat court (z ≤ tp/0,15 : pas de déversement) ; grugeage : flexion
  de la section réduite (MSB P5 §4.2.4). » · « Cordons pleine hauteur, un par face (§4.5.3.3) ; γM0 1,00 ·
  γM1 1,10 (ANB) · γM2 1,25. »
- Platine d'about (94 / 108 / 90) : « Articulé : rotule à la face de l'âme porteuse (MSB P5 §2.2) ; groupe
  centré, VEd/n par boulon. » · « Platine : hp ≥ 1,36·p3 → rupture de bloc, k = 1,0 ; grugeage : section
  réduite, bras tp + c (MSB P5 §4.2.4). » · « Cordons sur hp (§4.5.3.3), pleine résistance 0,55·tw = 4,7 mm ;
  γM0 1,00 (ANB) · γM2 1,25. »
- Pied de poteau, vérification complète (110 / 108 / 87) : « Périmètre EN 1993-1-8 : platine, tiges, poteau,
  cordons, appui fjd ; béton d'ancrage EN 1992-4 en information. » · « Aire efficace EC3 §6.2.5 : c = 59 mm,
  Aeff = 1294 cm², fjd = 20,4 MPa (βj 2/3) ; platine rigide, ΣT = 43 kN. » · « Verdict niveau 1 ; tiges coulées
  (ψM,N = 1) ; γM0 1,00 · γM2 1,25 · γc 1,50 · γMc 1,50. »
- Pied de poteau, calcul manuel (94 / 104 / 93) : « Calcul manuel — critère élastique (Von Mises ≤ fy/γM0) ;
  appui : bande c = 59 mm (EC3 §6.2.5). » · « Platine rigide, lit sans traction (kc 109 N/mm³, kb 454 kN/mm) ;
  bras m à l'appui, cordon 9,1 mm déduit. » · « Leff ≤ L/2 + m ; tiges, appui : information ; γM0 1,00 (ANB) ·
  γM2 1,25 · γc 1,50 (αcc 0,85). »

Le paquetage (`note_kit.empaqueter`) prend le PREMIER jeu de formes dans l'ordre lexicographique (les
segments du début gardent leur forme longue, les courtes sont prises du dernier au premier) ; l'ancien
compteur sans remise à zéro laissait les segments suivants à leur forme la plus courte et ne trouvait plus de
solution sur quatre cas figés dès que les renvois disaient « MSB P5 ». Le segment des coefficients a une source
(`note_kit.coefficients`, trois formes : « (boulons, soudures, sections nettes) », « (boulons, soudures,
nettes) », rien). Contrôlés : 31 cas des doubles cornières + 6 variantes, 12 cas du plat d'âme + 1, 13 cas de la
platine d'about + 3 — ≤ 3 lignes de ≤ 110 caractères, aucun segment perdu, aucun § du guide sans « MSB P5 ».


## 12. Décision : critère de la platine manuelle — Élastique par défaut
`entrees.defaults()["crit"] = CRIT_ELAS`, `CRITERES = [CRIT_ELAS, CRIT_EC3]` ; valeurs brutes et fichiers inchangés ; charger_json : un fichier qui porte « mode » mais pas « crit » reçoit CRIT_EC3 (le critère de l'époque), chk neuf. Mesuré ce jour sur le cas par défaut (HEA 300, platine 500×500×25 S355, 4 × M24 8.8, N 300 / Mx 50 / My 30 / Vy 40) : EN 1993-1-8 → t_req 15,56 mm (tT 12,47 · tS 15,56 · tc 12,19 · tb 4,0), η 62,25 % ; Élastique → t_req 19,81 mm (tT 19,81 · tS 18,66 · tc 12,19 · tb 4,0), η 79,25 %, platine de 25 mm toujours VÉRIFIÉE ; C, pmax, ΣT et c inchangés (338,9 kN, 12,22 MPa, 38,9 kN, 59,4 mm). Pourquoi : section au nu √1,5 ≈ 1,22 (t_el = √(6·M·γM0/(b·fy)) contre √(4·…), manuel/moteur.py l.471-472) ; tronçon en T jusqu'à √3 ≈ 1,73 (l.433-439) parce que l'effet de levier n'est crédité qu'en plastique : rapport 1,59 ici. Conservatif. Écrit dans le même commit : entrees.py (défaut daté), manuel/methode.py l.41 (« Élastique (défaut) »), manuel/note.py l.214-215 (CONCLUSION conditionnée : « selon Von Mises ≤ fy/γM0 (critère élastique) » / « selon EN 1993-1-8 §6.2.4 (tronçons en T) »), bandeau et ligne 1 des HYPOTHÈSES, manuel/benchmark.py (VAL-M1..M4 : `inp` reçoit crit=CRIT_EC3 — « le harnais a trouvé » : une fiche opposable dépendait d'un défaut), test_platine_manuelle.py l.270-272 (compute(crit=CRIT_EC3).t_req < compute().t_req), test_platine_manuelle_ecran.py l.87-88, l.102-104, l.143-146 (poser CRIT_EC3 puis retour au défaut), chk neufs « defaults()["crit"] == CRIT_ELAS » et « t_req(défaut) > t_req(EC3) », docs/assemblages/PLATINE_MANUELLE.md « Écarts assumés » et REFONTE_UX.md §13 datés. (DEV_Etudes_structure n'a ni DECISIONS.md ni CLAUDE.md : ces deux fichiers sont les boîtes de ce dépôt.)

## 13. Kit partagé à écrire d'abord
- `acier/assemblages/carte.py` : `Description(PREFIXE, CHAMPS, champ_visible, LIBELLE, LIBELLE_MODE, AIDES, AFFICHAGE, PAS, BLOCS, AVANCES, PILOTEES_PAR_PREDIM, mode_predim, VISSERIE_DEFAUT, GEOMETRIE_DESSIN)` ; `css_kit(prefixe)` ; `en_tete(segments)` (segmented_control required=True, format_func, help, on_change) ; `ligne_objet`, `ligne_suite`, `table(prefixe, bloc, entetes, rangees)` ; `widget` (selectbox / number_input / text_input, format_func AFFICHAGE, help AIDES seulement, disabled verrou ∪ prédim, « ◆ », « 🔴 ») ; `valeur_widget`, `contient_chaud`, `avances`, `cartouche`, `carte`. Les quatre ecran_saisie.py ne gardent que leurs données ; manuel/ecran.py importe `table`.
- `ui_commun.py` : `titre_bloc(txt, help=None)`, `bloc_statut_ligne` sans nowrap, `tableau_html(..., defilant=)`, `legende_natures`, `expander_perimetre`, `tableau_33_replie`, palette unique (_PASTILLE, hexas de _formules / _tableau_33 → C_TRAIT), un seul _esc, retrait du code mort (open_bloc_left_right, close_bloc, bloc_verification). `acier/projet_ui.py` : `en_tete(titre_court, sous_titre)`, `barre_outils(prefixe, …)` ; `acier/registre.py` : champ `court`. `acier/assemblages/note_kit.py` : `empaqueter`. `acier/assemblages/resultats_commun.py` (au lot 2) : `onglets(prefixe, ONGLETS, dispatch)`, `dessins_entete`, `panneau_cotes(prefixe, cles, …)`.
- Contrôles à écrire (tests/test_assemblages_kit.py, rouge d'abord) : (1) rien ne se perd — pour chaque module, CHAMPS ∪ GEOMETRIE_DESSIN ∪ {ui_niveau, ui_composant} ⊂ carte ∪ avancés ∪ cartouche ∪ cotes ∪ fenêtres, ONGLETS et clés d'export inchangés ; (2) aide(k) is None hors AIDES et toute aide ≤ 160 ; (3) libellés ≤ 22 partout, ≤ 16 dans une rangée de 4, ≤ 20 dans une rangée de 3 ou une continuation, pour LIBELLE et LIBELLE_MODE ; (4) options affichées ≤ 28 et AFFICHAGE ⊂ options brutes ; (5) css_kit contient les quatre media queries ; (6) aucune ligne d'objet > 4 widgets ; (7) étanchéité des préfixes ; (8) empaqueter : ≤ 3 lignes ≤ 110, aucun segment perdu, sur le défaut ET les cas figés (ref_plat_ame 12, ref_platine_about 13, 31 cornières, 6 + 4 pied) ; (9) « ? » visibles dans la carte au défaut ≤ 5 / 4 / 4 / 4 / 7 ; (10) defaults()["crit"] == CRIT_ELAS, t_req(défaut) > t_req(EC3), charger_json sans crit → CRIT_EC3 ; (11) zéro boulon + dessin interactif → hef (et dh si tête) dans _panneaux(R, u)['plate'] du manuel ; (12) le rédacteur saisi figure au cartouche de la platine d'about ; (13) GEOMETRIE_DESSIN ∩ widgets de la carte = ∅ pour le manuel aussi.

## 14. Plan de mise en œuvre
Voir la section « plan » : 0 mesure → 1 kit → 2 plat d'âme → 3 platine d'about → 4 doubles cornières → 5 pied complet → 6 pied manuel + défaut élastique → 7 docs → 8 E4 (captures trois formats relues, suites au vert, push). Un commit par étape, harnais dans le même commit, aucun module commencé avant que le précédent soit vert et relu.

## 15. Tests à faire suivre (rouges classés à l'avance)
- test_assemblages_interface : l.114 bandeau (suit) ; l.126-135 CARTE_ATTENDUE (orient hors du jeu par défaut, mode_calc en segmented — cles_widgets compte at.segmented_control —, id_projet / id_date absents en projet, visserie au cartouche : suit) ; l.155 cles_de['beamS'] étendu (suit) ; l.157-159 inchangés ; l.106, l.139-145 verts.
- test_assemblages_note : l.114-117 phrases exactes (suit) ; l.121 phrases interdites et l.132-134 z / MS (verts) ; test_assemblages_texte / _regression / _parite : identiques (export inchangé).
- test_plat_ame_ecran : l.81 et l.184-186 verts ; l.88-89 bandeau (suit : court) ; l.96-97 carte (vert : fpl_mode_calc devient segmented → compter at.segmented_control).
- test_platine_about_ecran : l.100-102 (pab_nat_N absent au défaut : suit), l.126 (poser N_Ed > 0 avant Tying : suit), l.219-224 verts, chk neuf rédacteur (a trouvé).
- test_pied_poteau_ecran : l.97 vert ; l.104-108 bandeau court (suit) ; l.109 et l.165 set_value sur ppb_perim (segmented : suit) ; l.248 vert ; « Non revérifiés ici » dans l'expander (vert).
- test_platine_manuelle_ecran : l.87-88, l.96-99 (ppb_hef / ppb_d_h hors carte, ppb_crit / ppb_appui / ppb_mode segmented, xN / yN toujours number_input), l.102-104, l.143-146 (suit) ; l.98-101, l.108, l.116, l.130 verts (mots « disponible » / « indisponible » conservés) ; chk neufs (11), (13).
- test_platine_manuelle : l.270-272 (suit) ; §8 benchmark VAL-M (a trouvé). test_projet_assemblages, test_rapport_projet, test_fumee_pages : verts attendus (clés inchangées, libellé du sélecteur recomposé par libelle_entree) ; suites moteur : « RÉGRESSION CALCUL : 0 différence ».

## 16. Risques
Le palier 1181–1350 (flex-basis 126) n'est pas encore mesuré : lot 0 à 1280 et 1181, repli = 2 par rangée dès 1350 (mesuré). :has() et l'override de flex-basis d'un stElementContainer dépendent du DOM de Streamlit 1.66.1 : captures.js se rejoue à chaque montée de version. L'en-tête du manuel (3 segmented) peut passer sur deux rangées à 1440 : accepté et mesuré. Les continuations de 5 dimensions font 3 + 2 à 1440 : accepté (Personnalisé seulement). La caption d'usage perd la liste complète des pièces touchables sauf dans les trois modules poutre-poutre où elle est conservée. Les captures « après » doivent suivre le protocole des captures « avant » (fenêtre agrandie à la hauteur du contenu).

## 17. Consolidation du 01/11/2026 (étape 7) — une source par brique

Le kit n'est plus gelé : les briques que les modules avaient écrites localement pendant la phase parallèle
(amendement 7) reviennent dans `acier/assemblages/carte.py` et `ui_commun.py`, et les écarts entre modules se
résorbent. Contrôles : `tests/test_assemblages_kit.py` §1 bis et §11, les cinq suites d'écran, et
`tools/captures_assemblages.js` (nouveaux seuils ci-dessous).

- **Sous-titres « ↳ » à « ? »** (« ↳ AILES B (groupe S) ? », « ↳ ETA ? ») : `carte.aide_suite` lit AIDES[« AILES B »] ;
  `ui_commun.sous_titre_bloc(txt, help=)` ne coupe qu'avant « ( » ; `carte.aides_visibles` les compte. Le bloc du
  kit porte un 🔴 allumé par une PIÈCE du dessin (`bloc(…, chaud=)`, « anchors », « col »).
- **Tables du calcul manuel** : `carte.table(prefixe, bloc, entetes, rangees, titre, aide, chaud)` — en-tête
  `<p>th_` en conteneur horizontal de la grille des rangées, texte qui compense la marge −16 px (l'ancien
  en-tête mesurait 0 px de haut), « ? » NATIF (`help=` : infobulle au toucher, là où un `span title` ne
  s'ouvrait qu'au survol), titre de table rendu une seconde fois sous 1351 px (`<p>tt_`) ; une rangée peut être
  suivie de sa continuation « ↳ É1 ». `carte.rangee`, `carte.suite`, `carte.ligne_titre`.
- **CSS en un élément** : `carte.injecter_css(prefixe, *extra)` concatène dans UN `<style>` (une troisième balise
  était lue comme du texte) ; la colonne du dessin figée, les rangées de table par deux avec le 🗑️ à droite,
  le titre `<p>tt_`, les rangées d'alerte, la tête du dessin, les tables serrées sont des règles du kit.
- **Tables serrées** : `tableau_html(serre=1)` (classe `tab-serre` : cellules 3px 4px, en-têtes repliés sous
  1181 px — la matrice des cornières : 430 → 309 px de large au plus serré, 699 → 519 px de haut à 1024) et
  `serre=2` (`tab-dense` : police réduite, tables de détail du calcul manuel). Toute table de ≥ 5 colonnes
  défile d'office (`defilant=None`).
- **En-têtes des tables** : écrits en capitales par `ui_commun.capitales` (mots ≥ 3 lettres latines, hors
  parenthèses), plus de `text-transform: uppercase` — « η » restait « Η » (un H), « σvm » devenait « ΣVM ».
- **Titres de 11 lettres** (RAIDISSEURS, APPLICATION : 77 px dans 76) : règle du kit `ui_commun.espacement` —
  tout titre dont un mot compte ≥ 11 lettres prend 0,05 em au lieu de 0,07 em (74 px mesurés). Retenue plutôt
  que LARG_TITRE = 84 : la case des titres plus large réduisait les cases des avancés à 1351 px sous 120 px.
- **Grille** : une case d'une ligne d'objet ≤ 187 px à ≥ 1351 (`LARG_CASE_MAX` = (660 − 76 − 3 × 8) / 3 :
  une ligne de 2 ne s'étire plus à 284 px) ; une continuation de 4-5 cases s'aligne sous sa première case
  (retrait de 84 px, case du sous-titre ramenée par une marge négative) ; sous 1351 px le titre et le sous-titre
  posés au-dessus de leur rangée lui appartiennent (12 px au-dessus, 4 px sous le texte, écart de rangée 8 px —
  mesuré sur l'encre : 4 px au-dessus pour 6 au-dessous auparavant).
- **Alertes** : UNE forme pour les cinq écrans, `ui_commun.alertes(prefixe, R, ligne, champs, texte)` — une rangée
  par alerte (ligne étirée, bouton « Localiser » / « Masquer » de 104 px aligné à droite, même rangée au
  téléphone), les alertes d'information regroupées dans un repli « ⚠️ n remarques — … » dès qu'il y en a deux
  (la forme née au pied de poteau) ; les pastilles des champs en cause : `carte.pastilles`.
- **Hors carte, même clé** : `carte.libelle_long` (LIBELLE_LONG : « Rangées n1 (groupe S) » dans les toasts et
  les pastilles ; une fenêtre d'objet, dont le titre situe la clé, garde le libellé de la carte),
  `carte.traiter_clic` (clic du dessin → source unique → toast), `carte.champ_fenetre`, `carte.panneau_cotes`
  (grille `<p>cotes` : 4 colonnes ≥ 1181 px, 3 jusqu'à 641, 2 au téléphone ; 4 × 90 px à 1024 auparavant).
- **Panneau Cotes** (dessin fixe) : les cotes que le dessin émet — celles de l'Édition en Vue simple (le panneau
  était vide) — puis la géométrie visible qu'il n'émet pas (`carte.cotes_du_panneau`) : le dessin fixe n'ouvre
  aucune fenêtre, le panneau est alors le seul lieu de la géométrie.
- **Dimensions vides** (profil ou cornière Personnalisé, tube RHS sans dimensions) : `Description.DIMENSIONS`,
  `carte.dimensions_vides`, `carte.provisoire` — la vérification est suspendue (bandeau « Données incomplètes »,
  message court SOUS la ligne qui porte les dimensions, onglets en attente), le dessin garde provisoirement les
  dimensions du profil par défaut. Auparavant : taux > 999 % et alertes absurdes (poutre-poutre), ZeroDivisionError
  et dessin disparu (pied de poteau).
- **Cartouche hors projet** : `carte.HORS_PROJET` — Projet ET Date dès que le nom du projet est vide (la date de
  session, toujours initialisée par la page projet, cachait Date dans les cinq écrans) ; `projet_ui.infos_cartouche`
  (une source pour les cinq écrans) imprime alors id_date.
- **Notations** : la ligne « Notations » du pied de chaque note vit dans `notation.py` (NOTATIONS, NOTATIONS_MANUEL)
  et l'onglet Méthode l'affiche mot pour mot (paragraphe « Notations ») : les cotes, réduites à leur symbole (zp,
  c, e1, Δz, p3, gh, dc…), y retrouvent leur définition. Elle tient sur la ligne de la note (≤ 798 pt à 5,6 pt) —
  celle du calcul manuel la dépassait (801 pt, coupée d'un « … ») : défaut trouvé.
- **Périmètre** : `expander_perimetre` dit le mot de son contenu (« n non vérifiés » en calcul manuel).
- **Légende EC / COMP / INT** : après la DERNIÈRE table, avant les replis, dans les cinq écrans (le plat d'âme la
  posait après la répartition par boulon).
- **Vocabulaire** : l'export texte de la platine d'about n'imprime la nature de NEd qu'en traction (« NEd = −20,0 kN
  (Tying (accidentel)) ») ; la référence du calcul manuel « … §6.7 — répartition élastique » (phrase interdite des
  notes depuis le 21/09/2026) devient « … §6.7 (répartition) » dans la VUE (`notation.ref_courte`), le moteur garde
  sa référence ; l'étiquette d'efforts du pied « NEd 300 · Mx 50 · My 30 · V » (lue comme un texte coupé, et
  rognée dès que le massif ne la débordait pas) devient « … + » comme « VEd 125 + » des poutre-poutre, et la boîte
  du dessin l'englobe.
- **Dessin du calcul manuel** (relu sur les captures) : « C = … kN » et « mEd » se posent au premier emplacement
  libre (ni sur un boulon, ni sur le trait de la bande, ni l'un sur l'autre — « mEd » couvrait le boulon 4,
  « C = 339 kN » le trait de la bande) ; « T = … kN » part du boulon vers l'intérieur (centrée, elle couvrait le
  numéro du boulon).
- **En-tête de table** : ses libellés s'étirent comme les cases des rangées, leur « ? » suit le libellé (au bout
  de la case, il se lisait comme celui de la colonne suivante), la place du 🗑️ est réservée (un conteneur vide
  n'en prenait aucune : les libellés glissaient de 25 px) ; un « ↳ » nu (sans texte) reste à gauche de sa rangée
  à toutes les largeurs (seul sur sa ligne sous 1351 px, il s'éloignait de ses cases) ; le texte d'un titre posé
  au-dessus de sa rangée garde sa marge de compensation (sans elle il chevauchait de 4 à 5 px la case suivante :
  « RAIDISSEURS » était rogné par le bouton ➕).
- **Tables** : dans une colonne de nombres (alignée à droite), la valeur ne se coupe plus de son unité
  (« 188,2 / kN » sur deux lignes à 390 px).
- **Code mort** : `entrees.COURT` (plat d'âme, platine d'about, doubles cornières) n'avait plus de lecteur — retiré ;
  LIBELLE de l'écran est la source.
- **Seuils ajoutés à tools/captures_assemblages.js** : texte visible non tronqué dans toute case des lignes, des
  continuations, des rangées de table et de l'en-tête (`input.scrollWidth ≤ clientWidth`) ; titres de bloc sans
  débordement ; case d'une ligne de ≤ 2 cases ≤ 187 px ; sous-titre « ↳ » plus proche de sa rangée ; tables
  serrées tenues dans leur colonne. Un 6e argument rejoue un état (complet, fixe, simple, perso, avances).
- **Relecture des captures finales** (1440 · 1280 · 1024 · 390, cinq écrans et états) — corrigé avant de rendre :
  - une continuation de trois cases (« ↳ AILES A » : Fixation · Rangées n1 · Files n2 ; « ↳ » des efforts du
    pied : Vx · Vy · TEd) passait sur 2 + 1 rangées entre 641 et 1180 px (3 × 150 + 2 × 8 = 466 px pour 440 de
    colonne à 1024) : ses cases se partagent désormais la rangée entre 120 et 150 px (l'écart que §18 assumait
    est levé) ;
  - colonne Réf. : un sigle ne se coupe plus (« MSB / P5 §3.2.3 », « Tab. / 3.4 » ; `ui_commun.lier_sigles`) et le
    séparateur « · » ferme sa ligne (lié à « Tab. 3.4 », il restait seul sur une ligne à 390 px) — la coupure
    avant « § » reste permise, sans quoi la colonne s'élargirait au téléphone (mesuré : aucune table de
    référence ne défile, ni à 390 ni à 1024) ;
  - note de la vérification complète : une valeur ne se sépare plus de son unité en fin de ligne
    (« tT = 8,6 / mm ») ;
  - une table « libellé | valeur » (calcul à la main) ne dessine plus de rangée d'en-tête vide ;
  - vérification complète : « Côté comprimé » et « Côté tendu » passent l'une sous l'autre sous 1351 px (à 1024,
    deux demi-colonnes de 210 px, et la seconde table sortait de 26 px de la page) ; une rangée est dite
    « y− (y = -200 mm) » et non par la clé du moteur « y-:-200 » — écran (Vérifications, Répartition) et note,
    une source (`notation.rangee_lisible`) ;
  - calcul manuel : le « ? » du repère sur la première rangée de boulons seulement (il se répétait sur chaque
    boulon sous 1351 px) ; le bouton « ➕ Raidisseur » garde sa largeur de contenu comme « ➕ Élément soudé » ;
    au cartouche du plan, la 1re ligne BOULONS mesurait 273 pt pour 168 et perdait, coupée d'un « … », l'ancrage
    et hef : ils y restent, « perçages cotés depuis les bords » passe en 2e ligne après la visserie ;
  - harnais : l'état « avances » n'ouvrait pas l'expander (le résumé porte d'abord le nom de son icône).

## 18. Écarts assumés (écrits ici parce qu'ils ne se voient pas dans le code)

- √(Ac1/Ac0) = 1,80 n'est plus imprimé sur la note de la vérification complète (Paramètres retenus à l'écran et
  export texte le gardent) — §11.
- « Fixation ailes A » / « Fixation ailes B » et non « Fixation » : la fenêtre Cornières et l'alerte « soudée des deux
  côtés » montrent les deux fixations ensemble ; une forme par clé (P5) interdit deux « Fixation ».
- hc et zc (hauteur et position des cornières) ne sont pas dans la fenêtre Cornières : ce sont des cotes du dessin
  (et du panneau Cotes en dessin fixe) ; la note de la fenêtre le dit.
- Chips tT · tS · tc · tb du bandeau du calcul manuel à côté de « Platine » : la note garde ses ÉLÉMENTS
  (synthese.ELEMENTS inchangé) ; les quatre chips sont propres à l'écran.
- Le repli « ⚠️ n remarques » regroupe les alertes d'INFORMATION dès qu'il y en a deux, dans les cinq écrans : un
  clic de plus pour les lire, le bandeau et le premier dessin restent dans le premier écran du téléphone.
- Types d'élément courts (« Profilé I/H », « Tube RHS », « Tube CHS », « Plat », « Rond », « Rectangle ») : un RHS
  carré (SHS) est un « Tube RHS » de h = b ; le nom complet de la section est dans la case Profilé.
- id_date : affiché hors projet avec Projet, et imprimé alors au cartouche ; dans un projet, la date du projet.
- « MSB P5 » deux fois sur la ligne 2 des hypothèses de la platine d'about en tying à hp < 1,36·p3 (faute de place,
  jamais retiré : un § nu se lirait comme l'EN 1993-1-8).
- Les briques de RÉSULTATS encore écrites dans chaque module (tables simples, formules, onglets Prédim, Note,
  Benchmark) restent au plan (`resultats_commun.py`, §13) : leur contenu diffère d'un module à l'autre.

## 19. Extension du 02/11/2026 : la famille POUTRE – POTEAU, sept écrans

Les trois attaches `plat_ame_poteau` (`fpc_`), `doubles_cornieres_poteau` (`dcc_`) et `platine_about_poteau`
(`pac_`) sont nées À LA CHARTE : elles n'ont ni amendé un paragraphe, ni ouvert d'exception. Ce que l'extension
ajoute, et rien d'autre :

- **P10 (même ordre dans tous les écrans)** reçoit une ligne d'EN-TÊTE de plus, le **cas d'appui** : « Mode » et
  « Appui » côte à côte, en `segmented_control`, options affichées « Aile » / « Âme ». C'est le choix qui commande
  tout le reste de l'écran (le groupe « Poteau porteur », les dessins, les limites), il est donc AVANT les
  profilés. Les trois modules le placent au même endroit.
- **P6 (la géométrie se règle sur le dessin)** gagne quatre cotes propres à la famille : la recoupe des semelles
  `rc`, le jeu latéral `j`, la largeur droite entre congés `ℓ` et l'excentricité `e_col`. Les trois modules les
  portent sur la VUE EN PLAN, qui est **la vue porteuse** de cette famille.
- **§11 (HYPOTHÈSES ≤ 3 lignes de ≤ 110 caractères)** : les six notes de la famille (trois modules × deux appuis)
  tiennent, `derniers_avertissements == []`. Le segment du CAS D'APPUI vient en **deuxième** position, parce que
  `note_kit.empaqueter` raccourcit du dernier au premier et qu'en queue il se perdait.
- **Captures (amendement 10)** : `tools/captures_assemblages.js` connaît les trois préfixes et trois états de plus
  — `ame` (appui dans l'âme), `ameok` (le cas (b) réglé de la platine d'about) et `tying` (NEd en force de
  chaînage). Seuils tenus aux quatre largeurs sur les **sept** écrans.
- **Une mesure corrigée, pas un seuil** : `rows()` comptait « 0 rangée » pour une ligne passée au-dessus du
  viewport après défilement (`last = -99`), et `debordants` comptait comme « sorti de l'écran » ce qui est logé
  dans une boîte qui DÉFILE (`tableau_html(defilant=)`, §10 — le choix de la charte est justement de faire défiler
  les tables serrées). Deux faux rouges ; la règle « mesurer ce que l'œil voit » n'a pas changé. Le script le DIT
  désormais : « boîtes qui DÉFILENT (contenu plus large que la boîte, lisible en balayant) : 384/358 px ».

**Ce qui reste à l'œil, et qui ne change pas en l'état (relecture du 02/11/2026)**

- À 390 px, en tying, la table « Poteau porteur » des cornières sur poteau fait 384 px pour une colonne de 358 :
  elle DÉFILE, et la colonne « Réf. » (`§6.2.4 · Tab. 6.2 · Tab. 6.4 · MSB P5 §2.3`) se lit en balayant. C'est le
  comportement CHARTÉ (§10, point 9 de la consolidation du 01/11/2026 : défiler plutôt que déborder) ; la rendre
  `serre=1` la ferait tenir (−8 px de remplissage par colonne, 5 colonnes), mais `_table_simple` est la MÊME
  fonction dans cinq modules : la resserrer changerait aussi les trois écrans poutre–poutre sous 1181 px. Décision
  du bureau, pas d'un commit de consolidation.
- Au CATALOGUE (page projet vide), les boutons « Créer » d'une rangée ne s'alignent pas quand les accroches n'ont
  pas le même nombre de lignes (≈ 20 px d'écart ; visible depuis les quatre premiers types, pas né avec la famille
  neuve). `modules/assemblages.py` n'a pas été touché — l'étude §10.4 l'exclut — et le défaut ne se voit qu'au
  format PC : au téléphone les cartes s'empilent.
