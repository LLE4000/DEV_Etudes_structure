# Site « Stab Méthodes » — première version

Le vrai site (pas une maquette) : l'**accueil**, le **garde-corps** et les **poutres béton armé**, avec de vrais
calculs. Il est **statique** : aucun serveur de calcul, aucune base de données pour l'instant. Le moteur Python du
dépôt tourne **dans le navigateur** (Pyodide, Python 3.13 compilé en WebAssembly) : les chiffres sont exactement ceux
de Streamlit, parce que c'est le même code.

| Page | Ce qu'elle fait |
|---|---|
| `index.html` | l'accueil : reprendre une note, « Nouvelle note » (chantier, dossier, nom, module), les chantiers |
| `garde-corps.html` | l'établi du garde-corps : données, dessins du moteur, résultat, calcul détaillé, choix des profilés |
| `poutre.html` | l'établi des poutres béton armé : sections, armatures, coupes, résultat par section, calcul détaillé |

## Comment ça marche

```
navigateur ── js/etabli.js ──► js/travailleur.js (Pyodide) ──► api_web/routeur.py ──► moteur du dépôt
   (affiche)                     (Python dans le navigateur)       (met en forme)        (calcule)
```

- `js/` n'affiche que des fragments HTML que le moteur rend ; il ne calcule rien.
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
s'exécute, seul le WebAssembly de Pyodide est compilé. `outils/serveur_local.py` applique ce fichier en local comme
Cloudflare, et `verifier_site.js` vérifie d'abord qu'elle est active : tout le parcours tourne sous elle.

## Ce qui reste à faire

- La **note PDF** (le moteur l'écrit avec reportlab, absent du navigateur) : la générer côté navigateur ou par une
  petite fonction en ligne.
- Les **comptes**, les **projets** partagés, l'**équipe** et les **paramètres du bureau** (Supabase).
- Les autres modules, un par un, sur le même modèle : leur moteur extrait de l'écran Streamlit, puis une page.
