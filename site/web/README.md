# Site « Stab Méthodes » — première version

Le vrai site (pas une maquette) : l'**accueil** et l'**établi** de chaque module publié (garde-corps, poutres et dalles
béton armé, taux d'armature, tableau des armatures, choix de profilé, soudures), avec de vrais calculs, au PC, à la
tablette et au téléphone. Il est **statique** : aucun serveur de calcul, aucune base de données pour l'instant. Le moteur Python du
dépôt tourne **dans le navigateur** (Pyodide, Python 3.13 compilé en WebAssembly) : les chiffres sont exactement ceux
de Streamlit, parce que c'est le même code.

| Page | Ce qu'elle fait |
|---|---|
| `index.html` | l'accueil : reprendre une note, « Nouvelle note » (chantier, dossier, nom, module), les chantiers |
| `garde-corps.html` | l'établi du garde-corps : données, dessins du moteur, résultat, calcul détaillé, choix des profilés |
| `poutre.html` | l'établi des poutres béton armé : sections, armatures, coupes, résultat par section, calcul détaillé |
| `<adresse>.html` | l'établi de chaque autre module du registre (`api_web/registre.py`), bâti sur le même gabarit |

## Comment ça marche

```
navigateur ── js/etabli.js ──► js/travailleur.js (Pyodide) ──► api_web/routeur.py ──► moteur du dépôt
   (affiche)                     (Python dans le navigateur)       (met en forme)        (calcule)
```

- `js/` n'affiche que des fragments HTML que le moteur rend ; il ne calcule rien. `etabli.js` pose les fragments,
  `toile.js` porte le zoom et le plein écran du dessin, `travailleur.js` fait tourner Python, `notes.js` garde les
  notes, `commun.js` ce que toutes les pages partagent (thème, messages, fichiers à télécharger), `meteo.js` va chercher
  la météo d'un lieu chez Open-Meteo quand l'ingénieur le demande (ci-dessous). `sonde.js` date le démarrage quand
  l'adresse d'un établi porte `?sonde=1` — une mesure, fermée par défaut (sans elle, rien ne change : `verifier_site.js`,
  filtre `sonde`) ; `outils/mesurer_demarrage.js` et `outils/couverture_donnees.js` en tirent les mesures datées de
  `docs/site/mesures/`.
- `api_web/` (à la racine du dépôt) relie l'écran au moteur : les champs, les effets de bord de la saisie, le résultat,
  les fenêtres de calcul, les sélecteurs. Testé en Python ordinaire : `tests/test_api_web.py` (garde-corps) et
  `tests/test_api_web_poutre.py` (poutres).
- Le moteur des poutres (`beton/poutre/`) est extrait de l'écran Streamlit `modules/poutre.py`, dont il reproduit les
  résultats : `tests/test_poutre_moteur.py`, `docs/beton/POUTRE_MOTEUR.md`.
- `moteur/moteur.zip` : les fichiers Python que le site exécute (relevés automatiquement), avec leur empreinte dans
  `moteur/manifeste.json` ; le navigateur la vérifie avant d'exécuter le paquet.
- `vendor/pyodide-0.29.5/` : le cœur Python du navigateur, tiré du paquet npm officiel, et la roue numpy dont
  l'empreinte est celle du `pyodide-lock.json` officiel (`outils/vendre_pyodide.py`). Rien n'est chargé ailleurs.
- Les notes s'enregistrent **dans le navigateur** (localStorage) ; le bouton **.stab** les exporte en fichier, et
  l'accueil les réimporte. Les comptes du bureau, le partage et les droits viendront avec Supabase.

## Trois formats, une seule page

Les fonctions sont les mêmes partout ; seule la disposition change (`css/site.css`, après la charte des maquettes).

| Format | Disposition de l'établi |
|---|---|
| PC et petit portable, 1 200 px et plus | la toile sur toute la largeur, les données et le résultat en cadres flottants qu'on replie (la maquette) |
| tablette portrait, 701 à 1 199 px (820 × 1 180, 768 × 1 024, 1 024 × 1 366) | la toile en haut sur toute la largeur, les données et le résultat côte à côte dessous ; jusqu'à 1 100 px, la barre du haut garde ses icônes |
| tablette paysage, 701 à 1 199 px (1 024 × 768, 1 180 × 820) | la toile à droite sur toute la hauteur, les données au-dessus du résultat à gauche |
| téléphone, 700 px et moins | l'en-tête compact (retour, nom de la note, taux, menu des actions), les onglets **Données · Dessin · Résultat** toujours visibles, la barre du bas (.stab, note PDF) ; la fenêtre de calcul et le sélecteur montent du bas sur tout l'écran |

- **La toile** : chaque vue porte sa légende (elle passe à la ligne plutôt que d'être coupée) ; la phrase du module se
  lit sous le titre de la première vue. **Zoom** : − / + / « 100 % » (ajuster), Ctrl + molette, pincer au doigt ;
  le dessin agrandi se fait glisser, un double appui le remet à sa feuille ; le zoom de chaque vue survit à un
  calcul. **Plein écran** : la vue couvre la fenêtre (Échap ou ✕ pour revenir). Un tableau (vue HTML) défile dans
  sa feuille, aligné en haut à gauche. Un module qui ne rend aucune vue : « Pas de dessin pour cette note. »
- **Au téléphone** : champs en 16 px (sinon iOS agrandit la page au toucher), cibles de 36 px au moins, aucun
  défilement de côté ; l'accueil a son en-tête et les cinq entrées de la barre en bas.
- **Les lettres grecques** d'un texte en capitales (en-têtes de tableau, titres de groupe) gardent leur casse : η ne
  devient plus « H », ni σ « Σ » (`commun.js`, `garderGrec`).
- **Le taux de la note** (pastille de l'en-tête du téléphone, carte de l'accueil) s'écrit comme les pastilles du
  moteur : « > 999 % » au-delà, la valeur exacte au survol (`commun.js`, `texteTaux` ; le plafond est lié à
  `commun.PLAFOND_TAUX` par `tests/test_site_web.py`).
- **Un champ verrouillé se voit** (`desactive=True` des briques : segments grisés, curseur interdit, case et texte
  estompés), le choix retenu reste lisible ; un nom s'écrit sans chiffres tabulaires (« Garde-corps », pas
  « Garde - corps ») ; la pièce d'une ligne de résultat passe sous son nom plutôt que d'être coupée.

## Télécharger un fichier propre au module

Un module pose dans ses fragments `<button type="button" class="btn sm" data-telecharger="<quoi>">…</button>`
(brique `C.bouton_telecharger`). Au clic, l'établi envoie au routeur
`{"module", "action": "telecharger", "etat", "quoi"}` ; la réponse `{"ok": true, "fichier": {"nom", "type", "texte"}}`
devient un fichier que le navigateur enregistre (un Blob : rien ne s'exécute, la politique de sécurité n'a rien à
permettre). Le nom est contrôlé par le routeur (`commun.refus_fichier`) puis nettoyé par l'écran (`commun.js`,
`nomDeFichier` : le dernier morceau d'un chemin, ni « / » ni « .. », ni caractère réservé, ni caractère invisible —
l'inversion du sens d'écriture U+202E ferait lire un « .json » comme un « .exe »). Un refus (`{"ok": false, "erreur"}`)
s'affiche en message ; le clic d'enregistrement ne referme pas la fenêtre de calcul d'où vient le bouton.

Vérifié **de bout en bout** dans Chromium (`verifier_site.js`, filtre « telecharger ») : le paquet du moteur servi au
navigateur y est celui du site, où le module des poutres gagne une fonction `telecharger` et ses boutons (rien de ceci
n'est publié) ; le bouton rendu par la brique, le routeur, le Blob et le fichier enregistré sous le nom donné par le
module, au PC et au téléphone, sous la politique de sécurité ; les refus du routeur (nom qui remonte les dossiers, page
HTML, fichier inconnu, module sans fichier) arrivent en message, sans fichier. Chromium doit tourner en locale UTF-8
(le contrôle la lui donne) : sous la locale « C » d'un serveur, il enregistre un nom accentué sous « download ».

Toute réponse du routeur est du **JSON strict** : un nombre non fini (un taux infini) part comme `null` — « Infinity »
ne se lit pas en JavaScript et la requête restait sans réponse ; l'écran, de son côté, change une réponse illisible en
message (`moteur.js`), la file des calculs ne se bloque plus.

## La météo du lieu (Open-Meteo) : une importation de données, à la demande

Le module **Maturité du béton** offre trois sources pour la température de cure : *Constante*, *Relevé* (collé ou lu dans
un fichier) et **Météo du lieu** — ce que faisait le mode « Météo » de l'écran Streamlit. Le calcul, lui, reste
**possible hors ligne** : sans réseau, une température constante ou un relevé saisi suffisent ; et une météo récupérée
est GARDÉE dans la note, qui se recalcule ensuite sans aucun appel.

```
moteur (Python)                       écran (js/meteo.js)                         moteur (Python)
meteo.plan : quelles dates à    ──►   bouton data-meteo (C.bouton_meteo) ;  ──►   « modifier » : {lieu, plan, geocodage,
l'archive, aux prévisions,            au clic : géocodage, puis chaque            reponses} BRUTS → meteo.lire_* (au bit
quelles variables                     appel, vers Open-Meteo SEULEMENT            près de l'écran Streamlit) → relevé
                                                                                  modifiable + récupération GARDÉE
```

- **Le moteur fait le plan** (`beton/maturite/meteo.py`, la logique de l'écran Streamlit recopiée : archive jusqu'à J−7,
  prévisions de J−6 à J+16, jours au-delà prolongés) ; le bouton le porte. **Le navigateur fait les appels** : il
  construit lui-même chaque adresse — la base vient de `js/meteo.js`, jamais du plan ; seuls les paramètres permis
  partent, encodés — puis rend les réponses **brutes** au module, qui les lit, les contrôle et les garde.
- **La politique de sécurité** ne s'ouvre qu'à ces données : `connect-src 'self'` et **exactement** les trois adresses
  d'Open-Meteo (géocodage, archive, prévisions). Aucun script, aucune feuille, aucune image ne vient d'ailleurs ; rien
  de ce qui est reçu ne s'exécute (des nombres et des dates, relus par le moteur ; un nom de lieu piégé renvoyé par le
  service s'affiche comme du texte). Les trois adresses sont des copies liées par `tests/test_site_web.py` (moteur,
  `js/meteo.js`, `_headers`).
- **Les refus arrivent en message**, jamais en exception : pas de réseau, service muet (20 s), saturé (un second essai
  2 s après un « trop de demandes », comme l'écran), en panne (avec sa raison), réponse illisible, réponse de plus de
  5 Mo (lue morceau par morceau, jamais au-delà ; rien ne part au moteur) — dits par `js/meteo.js` ; lieu introuvable,
  réponse incomplète ou dont les jours ne sont pas ceux demandés (le jour en cause nommé : le jour 1 du relevé est
  toujours le coulage), lieu ou dates changés pendant la récupération, un autre coulage ouvert pendant la récupération
  (le plan du bouton porte le rang du coulage), dates hors de portée (le bouton est alors grisé et le message écrit
  dessous) — dits par le module. La note reste intacte.
- Pendant la récupération, le bouton et l'indicateur disent « Récupération de la météo… ». Ensuite : le lieu trouvé, la
  date et l'heure de la récupération, le badge de l'écran (jours mesurés, prévus, prolongés), les mesures, la vue
  « Météo du lieu » et le tableau jour par jour ; les températures se corrigent à la main (l'écran le dit) ; le détail
  horaire (T min h, T max h) se charge à part, à la demande, comme dans Streamlit.
- **Vérifié dans Chromium** (`verifier_site.js`, filtre « meteo ») : les appels vers Open-Meteo sont INTERCEPTÉS et
  reçoivent des réponses au format du service (aucun appel réel) — qu'ils arrivent à l'interception prouve que la
  politique de sécurité les laisse partir ; au PC, au téléphone et à la tablette ; la note rechargée se recalcule sans
  aucun appel ; lieu introuvable, réseau coupé, service en panne, archive décalée d'un jour, réponse énorme (8 Mo), un
  autre coulage ouvert pendant la récupération, dates hors de portée en message. Des mutations le prouvent : la
  politique sans Open-Meteo (plus aucun appel, « Pas de réseau »), l'indicateur retiré ; et, à la relecture, la lecture
  sans plafond de taille, le plan sans le rang du coulage, les jours servis non comparés au coulage — chacune fait
  rougir exactement son contrôle.

## Le démarrage

Le travailleur charge le cœur Python, le paquet du moteur (en même temps), puis numpy **seulement pour les modules qui
s'en servent** : le relevé du paquet (`construire.py`, un sous-processus par module) les écrit dans
`moteur/manifeste.json` (« avec_numpy » : aujourd'hui garde-corps, choix de profilé, soudures). Un import de numpy
imprévu est guetté en Python : numpy est alors chargé et la requête rejouée. Mesure du 07/11/2026 (médianes, cache
plein) : poutres 2,46 s → 2,14 s ; soudures (numpy) 3,1 s, inchangé ; Python lui-même en prend 1,8 s. Le détail :
`docs/site/CAHIER_DES_CHARGES.md` § 10 bis.

## Un fichier .stab vient d'ailleurs : rien de ce qu'il porte ne s'exécute

Un .stab peut arriver d'un collègue ou d'un courriel. La règle :

- **Python échappe tout ce qu'il écrit en HTML** (liste des éléments, données, résultat, fenêtres de calcul, dessins) ;
  un nom de poutre, de section ou de garde-corps est du texte, jamais du balisage.
- **Les champs texte de la réponse** (titres des vues, résumé de la note) sont des données : la page les échappe en
  les posant.
- **À l'import, on ne garde que des données** (noms, chantier, dossier, indice, état) ; le dessin et le résumé du
  fichier sont jetés et refaits par le moteur à l'ouverture. Un import qui gardait la vignette laissait un fichier
  piégé exécuter du code sur l'accueil : c'est ce que vérifient `verifier_site.js` (« .stab piégé », dans le vrai
  navigateur) et `tests/test_api_web_poutre.py` (un nom de section piégé, venu d'un fichier, dans chaque fragment).

## Reconstruire après une modification du moteur

```bash
python3.12 site/web/construire.py        # pages, charte, paquet du moteur
python3.12 tests/test_site_web.py        # rougit si le paquet ne suit pas le dépôt
python3.12 tests/test_api_web.py
python3.12 tests/test_api_web_poutre.py
python3 site/web/outils/serveur_local.py 8770 &          # le site avec les en-têtes de _headers, comme Cloudflare
node site/web/verifier_site.js http://127.0.0.1:8770 <dossier_des_captures>          # tout le parcours
node site/web/verifier_site.js http://127.0.0.1:8770 <dossier> garde|poutre|piege   # une partie seulement
node site/web/verifier_site.js http://127.0.0.1:8770 <dossier> formats              # téléphone et tablettes, tous les modules
node site/web/verifier_site.js http://127.0.0.1:8770 <dossier> formats:dalle        # … un seul module
FORMATS=tel node site/web/verifier_site.js http://127.0.0.1:8770 <dossier> formats:dalle   # … le téléphone seul
node site/web/verifier_site.js http://127.0.0.1:8770 <dossier> telecharger          # télécharger, de bout en bout
node site/web/verifier_site.js http://127.0.0.1:8770 <dossier> meteo                # la météo du lieu (Open-Meteo intercepté)
#   (FORMATS : accueil, tel, tablettes, communs, numpy — pour prouver une correction par mutation sans tout rejouer)
```

## Mettre le site en ligne sur stab-methodes.be (Cloudflare Pages, gratuit)

1. **Cloudflare** → *Workers & Pages* → *Create* → *Pages* → *Connect to Git* → autoriser GitHub → choisir le dépôt
   `DEV_Etudes_structure` et la branche du site (aujourd'hui `claude/dalle-module-creation-sjmitr`).
   - *Framework preset* : **None** ; *Build command* : **vide** ; *Build output directory* : **`site/web`**.
   - *Environment variables* : **`SKIP_DEPENDENCY_INSTALL` = `1`** (le site n'a rien à installer ; sans elle,
     Cloudflare tenterait d'installer le `requirements.txt` de Streamlit).
   - *Save and Deploy* : le site est en ligne sur une adresse `….pages.dev` ; chaque poussée le met à jour.
   - Les pages s'ouvrent sans « .html » (`/poutre`, `/garde-corps`) : c'est ainsi que Pages les sert, et les liens du
     site visent déjà ces adresses.
2. Dans le projet Pages → *Custom domains* → *Set up a custom domain* → **stab-methodes.be** (le domaine est déjà
   chez Cloudflare : la DNS se règle toute seule).
3. **Le garder privé** (en attendant les comptes) : *Zero Trust* → *Access* → *Applications* → *Add an application*
   → *Self-hosted* → domaine `stab-methodes.be` → règle *Allow* → *Emails* : les adresses des collègues. Chacun
   reçoit un code par e-mail pour entrer. Gratuit jusqu'à 50 personnes.

`_headers` règle le cache (le cœur Python ne change jamais à adresse égale ; le paquet du moteur est relu à chaque
ouverture) et la **politique de sécurité** : tout vient du site lui-même, aucun script écrit dans une page ne
s'exécute, seul le WebAssembly de Pyodide est compilé ; la seule ouverture est de DONNÉES — les pages peuvent
interroger les trois adresses d'Open-Meteo (la météo du lieu, ci-dessus), rien d'autre. `outils/serveur_local.py` applique ce fichier en local comme
Cloudflare, et `verifier_site.js` vérifie d'abord qu'elle est active : tout le parcours tourne sous elle.

## Ce qui reste à faire

- La **note PDF** (le moteur l'écrit avec reportlab, absent du navigateur) : la générer côté navigateur ou par une
  petite fonction en ligne.
- Les **comptes**, les **projets** partagés, l'**équipe** et les **paramètres du bureau** (Supabase).
- Les autres modules, un par un, sur le même modèle : leur moteur extrait de l'écran Streamlit, puis une page.
