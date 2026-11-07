# Soudures sur le site « Stab Méthodes »

> **Relecture indépendante du 07/11/2026** — aucun défaut **non conservatif** trouvé : sur 25 cas choisis à la main
> (chaque configuration, chaque jonction, Eurocode directionnelle / simplifiée / ELS, cas à la limite η = 0,95 · 0,9499 ·
> 0,995 · 1,000 · 1,004, efforts nuls et énormes, cordons qui ne comptent pas, calculs impossibles), le site passé par
> `api_web/routeur.traiter` affiche les mêmes chiffres que l'écran Streamlit rejoué (η, quatre lectures, cordons,
> points, messages, efforts réduits, formules, dessins, encart). Corrigé à la relecture : saisies hors de portée
> (§ 4), notes abîmées refusées en français, indices et signe « − » des dessins et tableaux, arrondi des limites aligné
> sur les formules du moteur, correspondance des repères d'efforts (§ 3), défauts 11 à 15 ci-dessous.

Le module `soudures` du site (`api_web/soudures.py`, établi `soudures`) reprend l'écran Streamlit `modules/soudures.py`
(le secours, inchangé). Le **calcul** est celui du moteur `acier/soudures` (sans Streamlit, décrit dans
[SOUDURES.md](SOUDURES.md)) : les deux écrans appellent la même fonction, `acier.soudures.etat.verifier(cas)`. Il n'y
avait donc rien à extraire ; ce qu'il fallait prouver, c'est que le site donne au moteur **le même état** que l'écran
Streamlit, et qu'il **affiche les mêmes chiffres**.

| Fichier | Rôle |
|---|---|
| `api_web/soudures.py` | le module du site : état de la note (des cas), saisie, résultat, fenêtres de calcul, sélecteurs, vues |
| `site/web/css/modules/soudures.css` | ce que l'établi ajoute à la charte : tableaux de la toile, cartes des cordons libres |
| `tests/test_api_web_soudures.py` | la parité avec l'écran Streamlit (rejoué), ce que le site affiche, les refus, les mutations |
| `docs/acier/SOUDURES_SITE.md` | ce document |

Moteur importé, sans ses fichiers d'écran : `configurations`, `criteres`, `dessin` (encart de la gorge), `etat`,
`formules`, `methode`, `schemas` (dessins de dessinateur) ; `acier/profiles` (catalogue). Jamais `note.py` (reportlab).
Le relevé du paquet (`site/web/construire.py`) le vérifie : bibliothèque standard et numpy seulement.

## 1. Ce que le site reprend de l'écran Streamlit — et seulement cela

L'écran Streamlit porte quelques effets de saisie dans son code d'écran ; le site les reprend **à l'identique**, et le
test les rejoue contre l'écran réel :

- la case « par tronçons » : cochée, `n = max(1, n)` ; décochée, `n = 0` ;
- le tableau des cordons libres (C8) relu comme `elements_depuis_lignes` le relit à chaque affichage : cellule vide
  → défaut du type, côté ±1, retours d'angle « non », noms complétés ;
- « ＋ Cas » : une console C3 qui garde la méthode du cas ouvert ; « Copier » ; « Supprimer » (le cas d'avant devient
  le cas ouvert) ; « Prendre la valeur Eurocode » : `k = round(k équivalent, 1)` ; « Revenir à 100 » ;
- le changement de configuration (`etat.changer_config` : défauts de la configuration, nuances et méthode gardées) ;
- l'ouverture d'un fichier JSON de l'écran (tous ses cas, BOM accepté) ;
- les **minimums** de chaque champ (`min_value` de l'écran) : `MINIMUMS`, écrits en `data-min` et appliqués au refus.

Les textes et listes recopiés (méthodes, critères, aides, frontaux, semelles, pénétration, gravité) sont **nommés** en
tête du module ; le test les relit dans le source de l'écran.

## 2. La preuve de parité (`tests/test_api_web_soudures.py`)

L'écran Streamlit est **exécuté** (`modules.soudures.show()`) avec un faux « st » qui enregistre chaque widget (sa
clé, son chemin dans l'état, son minimum, ses options, son rappel), chaque texte affiché, et qui sait cliquer un bouton,
éditer le tableau C8 (un vrai DataFrame) et fournir un fichier. Résultats au 07/11/2026 (après relecture) : **206 contrôles, 0 échec**.

- **Calcul et affichage** — 200 cas tirés au hasard (25 par configuration, dont ~30 % de calculs impossibles : tronçons
  qui ne tiennent pas, cordons alignés, aucun cordon utile) : l'état relu par les deux écrans est **identique**, le
  moteur rend les mêmes nombres **au bit près** (taux, points, propriétés du groupe, comparaison, messages,
  résistances), et les deux écrans affichent les mêmes chiffres : η du verdict, quatre lectures, messages et leur
  gravité, tableau des cordons (a, l, l_eff, |f| max, nature, point), détail par point (|f|, σ⊥, τ⊥, τ∥, σ_c, quatre taux),
  efforts réduits au point de référence, formules du point gouvernant (chacune présente dans la fenêtre de calcul).
- **Champs** — pour chaque configuration et chacune de ses variantes (jonction angle / partielle / pleine, semelles et
  âme, tronçons, support plat ou poteau, C7 pleine ou partielle) : chaque champ de l'écran Streamlit existe sur le site
  et le site n'en invente aucun ; mêmes minimums ; mêmes options de menu. Le support (« plat » ou un poteau I/H) est un
  menu unique dans l'écran, deux menus sur le site (famille, puis taille) : ils atteignent exactement la même liste.
- **Saisie** — 1 328 saisies (chaque nombre, case, menu, côté soudé de C4, tronçons, configuration, identification,
  dans chaque variante), en exécutant le rappel de l'écran : même état des deux côtés. Les boutons (＋ Cas, copier, supprimer
  le 1er / 2e / 3e cas, choisir un cas, k Eurocode, revenir à 100, ouvrir un fichier) et dix éditions du tableau C8
  (gorge, coordonnées, nom, côté, tronçons, rayon, ligne neuve, ligne retirée) : même note.
- **Mutations** — chaque garantie rougit quand on casse ce qu'elle surveille : relecture C8 faussée, nature des cordons,
  gravité des messages, écriture des messages, relecture C8 supprimée, minimum de gorge, « ＋ Cas » qui perd la méthode,
  k Eurocode arrondi à l'unité, parenthèses des carrés.

- **Relecture (section F)** — saisie hors de portée refusée (et sa mutation : sans borne, le moteur déborde), note
  `.stab` à 1e300 refusée, n de tronçons borné, notes abîmées refusées en français, cordon C8 « -1 » et cas « 99 » non retirés en
  silence, sélecteur lu en entier
  (« a = 1e300 mm » n'est plus « a = 1 mm »), indices des dessins (et l'inverse exact : rien d'autre ne change dans le
  SVG), encart de la gorge, signe « − », limite F_adm écrite comme la formule (et sa mutation), indices des messages,
  correspondance des repères d'efforts lue au moteur, pastille du verdict.

Lancement : `python3.12 tests/test_api_web_soudures.py` (environ 20 s). Le contrat commun :
`python3.12 tests/test_api_web_modules.py soudures` (17 contrôles).

## 3. L'écran du site

- **Données** (sections repliables numérotées) : 1 Assemblage (nom, configuration C1…C8 en tête avec sa description, le
  switch Eurocode / ELS et la phrase qui dit ce qu'il change) · 2 Géométrie (ou « Cordons libres » en C8 : une carte par
  segment ou arc, ajouter / retirer, point de référence R) · 3 Soudure (jonction angle / partielle ± renfort / pleine
  avec la gorge de pleine résistance équivalente, retours, deux faces, tronçons) · Efforts (avec leur réduction au point
  de référence) · Aciers (nuances A et B, acier de référence) · Méthode (critère, γ_M2, k, g, γ_F, les deux boutons,
  identification de la note, ouverture d'un fichier JSON de l'écran Streamlit).
- **Résultat** : verdict (η, cordon et point gouvernants, « valeur ≤ limite ») ; une ligne par cordon (gorge, jauge,
  taux), qui ouvre sa fenêtre de calcul, et son **sélecteur de gorge** (ou de pénétration) : chaque gorge recalculée,
  triée, ✓ retenue, ★ la plus petite qui tient ; en C5/C6 la ligne du **profilé** et son sélecteur (triés par masse,
  ★ le plus léger qui tient, mêmes cordons) ; les **quatre lectures** de la même soudure (EC3 directionnelle,
  simplifiée, ELS bureau k, ELS k Eurocode) et leur fenêtre ; les messages du moteur ; le pied (méthode, γ_M2 ou τ_adm).
- **Repères des efforts** : le cadre Efforts nomme les efforts dans le repère de la pièce (C5 : V_z le long de l'âme,
  M_y d'axe fort), la vue du plan de joint, les efforts réduits et les formules dans celui du joint (V_y, M_x). Le site
  écrit la correspondance sous les efforts réduits (« V_z devient V_y, M_y devient M_x… ; C3 : V devient −V_y »), LUE
  au moteur effort par effort (`correspondance_axes`), jamais recopiée.
- **Écriture** : les symboles des dessins du moteur (h_p, V_y, b_pl, t_p, σ_c, f_u, β_w, γ_M2) prennent leur indice
  (`indices_svg` : seuls les textes changent, un nom saisi « plat_A » n'est pas touché) ; les messages du moteur aussi
  (M_x, f_vw,d, l_eff) ; signe moins « − » dans les tableaux et les efforts ; dans la fenêtre de calcul et le verdict,
  les valeurs et limites sont arrondies comme les formules du moteur les impriment (`_fr_moteur`, demi vers
  l'extérieur : F_adm = 98,1 × 5 = 490,5 N/mm s'écrit « 491 » partout, et non « 490 » à côté d'une formule à « 491 »).
- **Fenêtres de calcul** : hypothèses, tableau vérification / formule / valeur / limite / taux, puis les formules de la
  méthode choisie substituées (`acier/soudures/formules.py`) : efforts au point de référence, groupe de cordons, point le
  plus chargé, critère ; au pied, les trois autres lectures au même point.
- **Toile** : les vues du moteur (`schemas.vues`, deux ou trois par configuration : sections réelles du catalogue,
  hachures, bandes de cordon a·√2, symboles EN 22553, chaînes de cotes) ; l'encart de la gorge (fig. 4.5) à la taille
  de la feuille ; le tableau des cordons ; le détail par point ; la méthode et ses sources (textes de la note).

## 4. Ce qui est couvert, ce qui ne l'est pas

Couvert (liste de l'écran Streamlit cochée une à une) : barre d'outils (cas, ＋ Cas, copier, supprimer, nom, ouvrir un
JSON), switch et sa phrase, huit configurations et tous leurs champs, jonctions et gorge de pleine résistance, tronçons
partout où l'écran les offre, support plat ou poteau du catalogue, profilé / tube du catalogue, côtés soudés de C4,
tableau des cordons libres (segments et arcs), point de référence, nuances, efforts et leur réduction, méthode (critère,
γ_M2, k, g, γ_F, k Eurocode, revenir à 100), identification, méthode et sources ; résultat (η, phrase, ligne de méthode,
quatre lectures, messages), dessins, encart de la gorge, **carte des taux** (= les lignes du Résultat, une jauge par
cordon), onglet Cordons (propriétés, tableau, cordons exclus), onglet Détail (tableau par point, formules).

Pas couvert :

- la **note PDF** (hors périmètre du site : bouton grisé) ;
- l'**enregistrement au format JSON de l'écran Streamlit** : le site enregistre sa note en `.stab` (le même état,
  enveloppé) et **lit** le JSON de l'écran, mais n'en écrit pas (besoin commun : une brique « télécharger un fichier »
  propre au module) ;
- le **téléphone** : l'établi n'a pas encore la barre d'onglets Données / Dessin / Résultat des maquettes ; sous 700 px
  seul le cadre Données est visible (besoin commun, tous modules).

Écarts **voulus** (le test les vérifie) :

- C7 en pleine pénétration : « pénétration p » et « deux faces » ne sont montrés qu'en partielle (ni le calcul ni le
  dessin ne les lisent en pleine) ;
- un segment devenu arc prend le rayon et les angles par défaut d'un arc (l'écran lui laisse r = 0, refusé ensuite) ; un
  arc devenu segment prend le segment par défaut (l'écran en fait un segment de longueur nulle) ;
- deux cordons libres du même nom sont refusés (voir défaut 4) ;
- la couleur des taux est le code du bureau (`C.niveau` : rouge > 100 %, vert foncé 95–100 %, vert pâle < 95 %) ;
  l'écran Streamlit colore en orange dès 85 % et sa pastille dit « attention » dès 85 % (`etat.SEUIL_ATTENTION`) ;
- « Calcul impossible » : le verdict donne la **raison du refus** ; l'écran Streamlit montre le premier message, souvent
  une simple note d'espacement des tronçons ;
- les messages du moteur sont écrits à la française (98,1 N/mm²) et un nombre négatif au carré entre parenthèses
  ((−157,9)²) : écriture seulement, aucun nombre ne change ;
- **portée des nombres** : toute saisie au-delà de 1 000 000 en valeur absolue (mm, kN, kN·m) est refusée avec un
  message (`BORNE`), et une note qui en porte est refusée : l'écran Streamlit l'accepte puis s'arrête sur
  `decimal.InvalidOperation` (défaut 11) ;
- **tronçons** : au plus 100 par cordon (`N_TRONCONS_MAX`, écrit en `data-max`) — l'écran Streamlit n'a pas de
  maximum ; au-delà, une ligne de résultat et une fenêtre de calcul par tronçon rendent l'écran illisible (200
  tronçons : 400 lignes, 2,5 Mo de fenêtres).

## 5. Défauts trouvés dans l'écran Streamlit (reproduits ou refusés, jamais corrigés en silence)

1. **Valeur hors liste lue d'un fichier** : un menu dont la valeur n'est pas dans la liste affiche la **première
   option** alors que le calcul prend la valeur du fichier (g = 9,8 N/kg s'affiche « 9,81 » et calcule 9,8 ; une nuance
   inconnue s'affiche « S235 » et le moteur la remplace en silence par S235). Le site **refuse** une telle note et dit
   pourquoi (`controler_cas`).
2. **Côté d'un cordon libre** ≠ ±1 dans un fichier : ramené en silence à +1 par la relecture du tableau. Refusé.
3. **Valeur sous le minimum d'un champ** dans un fichier (gorge 0,5 mm) : l'écran s'arrête sur
   `StreamlitValueBelowMinError` (vérifié). Le site refuse la note avec un message ; à la saisie, une valeur sous le
   minimum est refusée (« Au moins 1 mm… »), la note reste intacte.
4. **Deux cordons libres du même nom** : leurs résultats se mêlent en une seule chaîne (deux cordons de 200 mm deviennent
   « cordon A » de 400 mm dans le tableau). Le site refuse le nom en double.
5. **Type d'un cordon libre changé dans le tableau** : segment → arc de rayon nul (refusé par le moteur), arc → segment
   de longueur nulle. Le site pose les valeurs par défaut du nouveau type.
6. **« Par tronçons » coché** : n = 1 ; sur un pourtour (C6) un seul tronçon → « Refus : cordons alignés » et calcul
   impossible. Reproduit (parité) ; le verdict du site dit la raison.
7. **Verdict d'un calcul impossible** : le premier message, pas le refus (voir § 4).
8. **Nombres des messages du moteur** formatés à l'anglaise (« 98.1 N/mm² », `:g`) dans l'écran et la note.
9. **Substitution « −157,9² »** dans les formules du moteur (écran et note) : se lit −(157,9²) ; le moteur calcule bien
   (−157,9)². Écriture corrigée sur le site seulement.
10. **Noms non échappés** : l'encart de la gorge (`dessin.encart_gorge`), le tableau des cordons, le détail par point et
    la phrase du résultat écrivent les noms de cordon (saisis en C8) tels quels dans le HTML de l'écran Streamlit. Le
    site les échappe (test d'échappement avec un nom piégé).

11. **Saisie hors de portée** : gorge, effort ou coordonnée de 1e300 → l'écran s'arrête sur `decimal.InvalidOperation`
    (vérifié, faux « st ») ; avant la relecture, le site rendait une erreur Python. Refusé (§ 4).
12. **Arrondis contradictoires** : les formules du moteur arrondissent « demi vers l'extérieur » (F_adm = 490,5 →
    « 491 N/mm »), les tableaux de l'écran (`f"{x:.0f}"`) « au pair » (« 490 »). Sur le site, le verdict et le tableau de
    la fenêtre suivent la formule ; les tableaux des cordons et des points gardent l'arrondi de l'écran (parité).
13. **Deux noms pour un effort** (C5, C6, et C1, C2, C3, C4, C7) : V_z saisi s'appelle V_y dans la vue de face et les
    efforts réduits, M_y devient M_x, T devient M_z ; en C3, V vers le bas devient −V_y. L'écran Streamlit ne le dit
    nulle part ; le site écrit la correspondance (lue au moteur). Les dessins ne sont pas renommés (moteur partagé).
14. **Symboles des dessins** écrits « h_p », « V_y », « b_pl », « fu/(βw·γM2) » (soulignés et lettres collées) dans
    l'écran Streamlit ; indices sur le site.
15. **Étiquettes superposées (C8)** : un arc dont le point le plus à gauche tombe au milieu d'un segment (cas par défaut
    + « Ajouter un arc ») écrit « cordon 1 · a 5 » et « arc 3 · a 5 » l'une sur l'autre — dessin du moteur
    (`schemas.py`), partagé avec l'écran Streamlit et la note : non corrigé ici.

## 6. Vérification dans le navigateur

`python3.12 site/web/construire.py --dossier /tmp/site_soudures --modules soudures`, serveur local, puis
`node site/web/verifier_site.js http://127.0.0.1:8816 /tmp/captures_soudures_revue module:soudures` : 19 contrôles, 0 échec
(moteur démarré dans Chromium, résultat, dessin à sa taille, rien de coupé ni de débordant, fenêtre de calcul,
sélecteur, saisie recalculée et gardée au rechargement, aucune erreur JavaScript). En plus, chaque configuration a été
parcourue toutes sections ouvertes à 1 440 px et 1 180 px (tablette) : aucun libellé coupé, aucun débordement.
