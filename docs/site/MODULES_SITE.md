# Ajouter un module au site « Stab Méthodes »

Ce guide est le contrat de tout module du site (`site/web`). Deux modules le tiennent déjà et servent de modèles :
**garde-corps** (`api_web/garde_corps.py`, moteur `acier/garde_corps`) et **poutres béton armé** (`api_web/poutre.py`,
moteur `beton/poutre`, extrait de l'écran Streamlit `modules/poutre.py` avec ses tests de parité
`tests/test_poutre_moteur.py` et sa doc `docs/beton/POUTRE_MOTEUR.md`). Les lire avant d'écrire.

## 1. Le principe

Le site est statique. Le moteur Python du dépôt tourne **dans le navigateur** (Pyodide 0.29.5, Python 3.13). L'écran
(`site/web/js/etabli.js`) ne calcule rien : il envoie des requêtes JSON au routeur (`api_web/routeur.py`), qui appelle
le module, et pose les **fragments HTML** que le module rend. Les chiffres sont ceux de Streamlit parce que c'est le
même calcul.

Bibliothèques permises sur le chemin du calcul : **la bibliothèque standard et numpy**. Refusées (absentes du
navigateur) : streamlit, pandas, matplotlib, scipy, reportlab, PIL, fitz. Le relevé du paquet
(`site/web/construire.py`) les refuse et échoue si un chemin les charge.

## 2. Les fichiers d'un module — et SEULEMENT ceux-là

| Fichier | Rôle |
|---|---|
| `api_web/<clé>.py` | le module du site : déclare `MODULE = "<clé>"` et `PAGE = "<adresse>"` (il entre alors au registre, `api_web/registre.py`, sans rien modifier d'autre) |
| le moteur | s'il existe déjà sans Streamlit (`acier/soudures`, `acier/assemblages/...`, `plancher/charges`, `acier/profiles`) : l'importer, **sans** ses fichiers d'écran (`interface.py`, `ecran_*.py`, `note.py` importent streamlit/reportlab). S'il vit dans l'écran Streamlit (`modules/<x>.py`) : l'**extraire** dans un paquet de calcul (`beton/<x>/`, `bois/<x>/`, `sol/`, `flambement/`, `acier/<x>/`), comme `beton/poutre` |
| `api_web/dessin_<clé>.py` (ou dans le paquet du moteur) | les dessins SVG, à l'échelle |
| `tests/test_<x>_moteur.py` | la **parité** du moteur extrait avec l'écran Streamlit (faux `st`, comme `tests/test_poutre_moteur.py`), des contrôles à la main, des mutations qui rougissent |
| `tests/test_api_web_<clé>.py` | ce que l'écran du site affiche = ce que le moteur calcule ; saisies, refus, effets de bord, sélecteurs, échappement |
| `docs/<domaine>/<X>_MOTEUR.md` | la méthode, d'où vient chaque grandeur, la preuve de parité, ce qui n'est pas couvert, les défauts trouvés dans le module Streamlit |
| `site/web/css/modules/<clé>.css` (facultatif) | une feuille de style propre au module (chargée par son établi seulement) |

**Ne pas modifier** les fichiers partagés : `api_web/commun.py`, `routeur.py`, `registre.py`, `parcours.py`,
`icones.py`, `site/web/js/*`, `site/web/css/site.css`, `charte.css`, `construire.py`, `verifier_site.js`,
`tests/test_api_web_modules.py`, `tests/test_site_web.py`, ni `modules/*.py` (l'écran Streamlit reste le secours,
inchangé). Un besoin commun se signale dans le rapport final (« Besoin commun »), il ne se bricole pas.

Ne pas lancer `python3 site/web/construire.py` **sans** `--dossier` (site/web est reconstruit à l'intégration), ne pas
faire de commit ni de push : l'intégrateur s'en charge.

## 3. L'interface d'un module (`api_web/<clé>.py`)

```python
MODULE = "<clé>"          # la clé des familles de l'accueil (site/maquettes/construire.py, FAMILLES)
PAGE = "<adresse>"        # « charges-plancher » → établi charges-plancher.html (servi sans « .html »)
PARCOURS = [...]          # facultatif : requêtes de plus pour le parcours (api_web/parcours.py)

def nouvel_etat() -> dict                         # la note neuve (les défauts de l'écran Streamlit)
def normaliser(etat) -> dict                      # relit un état reçu (fichier .stab, navigateur) ; ValueError claire si abîmé
def modifier(etat, chemin, valeur) -> str | None  # une saisie ; chemin = data-chemin coupé aux « / » ; message si refusée (l'état reste intact)
def gerer_elements(etat, action, index)           # « activer », « ajouter », « dupliquer », « supprimer » (jamais le dernier)
def choisir_section(etat, role, nom)              # la ligne choisie dans un sélecteur
def selecteur(etat, role, familles, froid) -> str # le HTML d'un sélecteur (C.selecteur)
def resultat(etat) -> dict                        # voir ci-dessous
```

`resultat(etat)` rend : `resultat` (HTML du cadre Résultat), `volets` (fenêtres de calcul `C.fenetre_calcul`),
`vues` (`[{cle, titre, court, svg, legende}]` — titre, court et légende sont du TEXTE, la page les échappe ; `svg` est
du HTML posé dans la toile : un dessin `<svg …>`, ou un bloc `<div …>` quand la vue est un tableau — catalogue,
tableau des armatures ; `echelle` facultatif : `False` pour un dessin hors échelle — une courbe, un diagramme — sinon « mm · à l'échelle » s'écrit sur tout dessin `<svg>` et jamais sur un tableau ; la vignette de l'accueil est le premier dessin `<svg>`), `phrase` (HTML, sous le titre de la première vue), `statut_html` (`C.statut(niveau)`), `resume`
(`{taux, niveau, statut, gouvernant}` — `taux` arrondi par `C.pct` ou None ; `niveau` parmi `p f r c x i` ; `i` =
« Consultation » pour un module sans vérification : tableau, catalogue, courbe), `donnees` (HTML du cadre Données),
`elements` (HTML de la liste des éléments de la note, comme garde-corps et poutre).

Une note contient **plusieurs éléments** (plusieurs dalles, plusieurs cornières…) : copier le schéma de
`api_web/poutre.py` (`{"version": 1, "<éléments>": [...], "actif": 0}`, `gerer_elements`, `elements`, résumé de la
note = son élément le plus chargé).

## 4. Les briques d'écran (`api_web/commun.py`) — toujours elles, jamais du HTML refait à la main

- **Cadre Données** : `C.section(n, titre, résumé, corps, ouvert, clé)` (sections numérotées, repliables) ;
  `C.ligne(libellé, champ, symbole, aide, full)` ; champs `C.num(chemin, valeur, unité, dec, entier, desactive, mini,
  maxi, auto)`, `C.texte(chemin, valeur, indice, desactive)`, `C.choix` (menu), `C.segments(chemin, valeur, options,
  desactive)` (2 à 4 options), `C.case(chemin, valeur, texte, desactive)`, `C.zone_texte(…, desactive)` (plusieurs
  lignes : un tableau collé), `C.fichier` (le moteur reçoit `{"nom", "texte"}`), `C.aide`, `C.message`.
  Ajouter/retirer : boutons `class="addrow"` / `class="moins"` avec `data-chemin=".../+"` ou `".../<i>/-"` (voir
  `api_web/poutre.py`).
- **Résultat** : un **verdict** (`C.chip(taux, niveau, xl=True)` + la vérification gouvernante et sa comparaison
  « valeur ≤ limite » ; pour une valeur qui n'est pas un taux : `C.verdict_valeur`, ci-dessous), puis une ligne par
  vérification `C.ligne_res(nom, pièce, taux, niveau, calcul=, pk=, gov=, leger=, sans_taux=)` (deux lignes : nom et
  pièce, puis jauge et taux), groupées si utile (`<div class="elgrp">`), les avertissements (`C.message`), le pied
  (`pfoot` : méthode + `C.legende()`).
- **Fenêtre de calcul** : `C.fenetre_calcul(id, titre, sous_titre, hypothèses, lignes, détail, pied, colonnes)` —
  hypothèses en pastilles, tableau vérification/formule/valeur/limite/taux, puis le **détail des formules**
  `C.lignes_fx` : « symbole = expression = valeurs substituées = résultat », comme une note faite à la main.
- **Sélecteur** : `C.selecteur(…, unite, etoile, pastille)` — trié par ordre de grandeur, le retenu (✓) et l'optimal
  (★, le plus léger qui tient) marqués.
- **Télécharger** un fichier propre au module : `C.bouton_telecharger(quoi, libellé)` et la fonction
  `telecharger(etat, quoi)` du module (§ 4 ter).
- **Importer la météo d'un lieu** (Open-Meteo), à la demande : `C.bouton_meteo(chemin, libellé, plan)` (§ 4 quater).
- **Couleur** : `C.niveau(taux)` (rouge > 100 %, vert foncé 95–100 %, vert pâle < 95 % — les seuils du moteur,
  `ME.statut`) ; rouge aussi quand le moteur déclare non conforme sous 100 %. `C.pct` arrondit **sans changer de
  bande**.
- **Nombres** : `C.nb` (exact, virgule, sans séparateur de milliers — `C.num` les pose dans un champ) pour les
  données ; `C.fr(x, nd)` pour les résultats (mm² entiers, cm au mm, contraintes au centième…) ; `C.qte(x, unité,
  nd)` pour un nombre suivi de son unité. Unités toujours écrites ; indices en `<sub>` (`C.symb_math`).
- **Icônes** : `C.icon(nom)` (Lucide, `api_web/icones.py`).

### 4 bis. Les règles des nombres, et les briques ajoutées le 07/11/2026 (un exemple chacune)

**L'arrondi est celui des formules des moteurs** (`acier.formats.F`, c'est-à-dire `acier.js.to_fixed`) : **demi vers
l'extérieur**, sur la valeur binaire exacte du nombre. `C.fr`, `C.nb`, `C.pct`, `C.qte` et `C.num(entier=True)` le
suivent tous ; la règle est appelée, pas recopiée. Un tableau et la formule voisine écrivent donc le même chiffre.
Jamais de « -0 ». Un nombre de numpy (`numpy.int64`, `numpy.float32`) s'écrit comme un nombre Python.

```python
C.fr(490.5, 0)        # « 491 »  (l'ancien arrondi au pair écrivait « 490 » à côté de « 491 » dans la formule)
C.fr(-490.5, 0)       # « -491 » (vers l'extérieur, pas vers +∞)
C.fr(2.675, 2)        # « 2,67 » : 2,675 vaut 2,67499… en binaire — le moteur écrit la même chose
C.fr(1234567.891, 2)  # « 1 234 567,89 » (espace fine insécable entre les milliers)
C.fr(-0.04, 1)        # « 0,0 », jamais « -0,0 » ; C.fr(None) = C.fr(nan) = « — »
C.nb(210000, 0)       # « 210000 » (seuls les zéros APRÈS la virgule s'ôtent ; c'était « 21 »)
C.nb(0.5, 0)          # « 1 » (plantait)
C.pct(0.635)          # 64 ; C.pct(-0.025) = -3 ; la bande de couleur est toujours gardée (94,98 % → 94)
```

**`C.qte(x, unité, nd=1)`** — la valeur et son unité liées par une espace insécable ; les espaces de l'unité deviennent
insécables aussi. Au téléphone, « 1,380 » et « t / m³ » ne se séparent plus. L'unité est du HTML du module (jamais un
texte saisi).

```python
C.qte(1380.5, "kg/m³")        # « 1 380,5 kg/m³ » (insécable)
C.qte(1.38, "t / m³", 3)      # « 1,380 t / m³ » — aucune espace sécable
C.qte(None, "kN")             # « — »
```

**Les taux** (`C.chip`, `C.ligne_res`) : la jauge est bornée à [0, 100] % (un taux NÉGATIF laisse la jauge vide ; une
largeur CSS négative l'affichait pleine) ; au-delà de 999 %, le taux s'écrit **« > 999 % »** et sa valeur exacte est
au survol (`title`), comme `acier.formats.pct` — un nombre de huit chiffres cassait la pastille. Sans taux,
`ligne_res` écrit « à compléter », ou le texte neutre du module :

```python
C.ligne_res("Ø12 · e 15", "1 131 mm²/m", None, "c", sans_taux="consultation")   # pas « à compléter »
C.chip(12345678, "r", xl=True)   # <span class="chip xl r" title="12 345 678 %"><small>η</small>&gt; 999 %</span>
C.chip(None, "c")                # « — »
```

**`C.symb_math`** : toutes les lettres grecques (minuscules, majuscules, ϕ ϑ ϵ), la barre combinante (λ̄), une lettre
grecque dans l'indice, ⊥ et ∥ : `C.symb_math("χ_LT · λ̄_LT · M_Rd,θ · σ_⊥")` → χ<sub>LT</sub> · λ̄<sub>LT</sub> ·
M<sub>Rd,θ</sub> · σ<sub>⊥</sub>.

**`C.verdict_valeur(x, unité, titre, comparaison="", niv="c", nd=1, symbole="", calcul="", classe="")`** — le verdict
d'une valeur qui n'est PAS un pourcentage (kg/m³, mm²…), du même dessin que le verdict d'un taux : la valeur en grand
et l'unité en petit dans la pastille, puis le titre (TEXTE, échappé) et la comparaison (HTML du module). `calcul` : la
fenêtre de calcul que le verdict ouvre. Sa pastille ne se comprime jamais sous une comparaison longue (mesuré dans
Chromium : 164 px tenus pour « 1 380,5 kg/m³ », 114 sans la règle).

```python
C.verdict_valeur(q["kg_m3_majore"], "kg/m³", "Taux d'armature",
                 f"{C.qte(q['masse_acier_majoree'], 'kg')} d'acier pour {C.qte(q['Vc'], 'm³', 3)} de béton", calcul="q")
C.verdict_valeur(x["As"], "mm²/m", f"{b['nom']} · {x['nom']}", f"≥ A<sub>s,req</sub> = {C.qte(b['As_req'], 'mm²/m', 0)}",
                 niv="p", nd=0)
```

**`C.fenetre_calcul(…, colonnes=("Vérification", "Formule", "Valeur", "Limite", "Taux"))`** — les cinq titres du
tableau, pour un module qui ne vérifie pas ; une ligne sans taux (`"taux": None`) porte « — » :

```python
C.fenetre_calcul("q", "Quantitatif", "P1", hyps, lignes, detail,
                 colonnes=("Armature", "Formule", "Masse", "Apport au taux", "Part"))
```

**`C.selecteur(…, etoile="le plus léger qui tient", pastille="taux")`** — `etoile=""` retire la légende ★ du pied,
`pastille=""` retire le « · taux » (ou le remplace : `pastille="part"`) ; une ligne sans taux (`"taux": None`) n'a pas
de pastille :

```python
C.selecteur(role, titre, sous, "", rows, debut, "kg/m³", etoile="", pastille="part")
```

**`C.num(…, auto="auto · 3,5")` — « vide = automatique »** : le champ accepte d'être VIDÉ et montre son texte indicatif
quand il l'est (`valeur` None). Il part comme un texte (l'écran refuse un nombre vide) ; le module le lit par
**`C.lire_auto(valeur)`** : `None` pour un champ vidé (le module prend la valeur automatique), le nombre sinon, et
`ValueError` (message en français, à rendre tel quel) pour tout autre texte. Le contrat commun l'éprouve.

```python
# cadre Données
C.num(base + ["dist"], c["dist"], "cm", auto=f"auto · {C.nb(e_auto)}")
# modifier(etat, chemin, valeur)
try:
    c["dist"] = C.lire_auto(valeur)        # None : l'axe redevient automatique
except ValueError as e:
    return str(e)
```

**État désactivé** de `C.texte`, `C.segments`, `C.case`, `C.zone_texte` (comme `C.num(desactive=True)` et
`C.choix(desactive=True)`) : le champ reste lisible, ne répond plus, et porte la classe `locked` (et le cadenas pour
les champs de texte) ; la feuille du site le montre (segments grisés, le retenu cerclé, curseur interdit, case
estompée — contrôlé dans le navigateur, `verifier_site.js`) :

```python
C.texte(["nom"], d["nom"], desactive=verrouille)
C.segments(["fyk"], d["fyk"], [(n, f"B{n}") for n in EN.NUANCES], desactive=verrouille)
```

### 4 ter. Télécharger un fichier propre au module (le JSON de l'écran Streamlit, un tableau CSV…)

Le bouton : **`C.bouton_telecharger(quoi, libellé, icone="download")`** →
`<button type="button" class="btn sm" data-telecharger="<quoi>">icône + libellé</button>` (tout échappé). L'écran
envoie `{"module", "action": "telecharger", "etat", "quoi"}` ; le routeur appelle la fonction du module

```python
def telecharger(etat, quoi) -> dict | None:       # etat : déjà normalisé (normaliser)
    if quoi == "json":
        return {"nom": C.nom_fichier(etat["elements"][etat["actif"]]["nom"], "json"),   # un nom SÛR tiré d'un nom saisi
                "type": "application/json",
                "texte": json.dumps(etat_streamlit(etat), ensure_ascii=False, indent=2)}
    return None                                    # « quoi » inconnu (ou lever ValueError("message en français"))
```

et répond `{"ok": true, "fichier": {"nom", "type", "texte"}}`, ou `{"ok": false, "erreur": "<message en français>"}`
— jamais une exception — quand le module n'a pas de `telecharger`, que `quoi` est inconnu, que le nom est absurde
(vide, plus de 120 caractères, « / », « \ », « .. », caractère de contrôle ou invisible — l'inversion du sens
d'écriture U+202E, une marque de sens, un séparateur de ligne), que le type n'est pas `application/json`,
`text/csv` ou `text/plain`, ou que le texte dépasse 5 Mo (UTF-8). Les règles sont `C.refus_fichier` ;
`C.nom_fichier(nom saisi, extension)` rend toujours un nom qui les passe (« Poutre P1/2 : RDC » → « Poutre P1-2 -
RDC.json »). Le contrat commun exige que **chaque bouton annoncé rende un fichier valide** (et un JSON qui se relit).

### 4 quater. Importer la météo d'un lieu (Open-Meteo), à la demande

Le bouton : **`C.bouton_meteo(chemin, libellé, plan, petit=False)`** → `<button class="btn meteo" data-meteo="<plan,
JSON échappé>" data-meteo-chemin="<chemin>">`. Le **plan** vient du moteur (`beton/maturite/meteo.plan(lieu, début, fin,
aujourd'hui, horaire=False, position=None)`) : le lieu à géocoder (ou sa position), les plages de dates de l'archive et
des prévisions, les variables. Un plan impossible (`{"erreur": "…"}` : pas de lieu, dates hors de portée…) grise le
bouton et écrit le message dessous.

Au clic, `js/meteo.js` fait les appels — vers les trois adresses d'Open-Meteo et elles seules, paramètres filtrés et
encodés — puis l'écran envoie `{"action": "modifier", "chemin": <chemin>, "valeur": {"lieu", "plan", "geocodage",
"reponses"}}` : les réponses BRUTES. Le module les relit avec le moteur (`meteo.lire_geocodage`, `lire_journalier`,
`lire_horaire` ; le plan reçu doit être celui qu'il recalcule, sinon « lieu ou dates modifiés pendant la récupération »)
et les **garde dans la note** (`meteo.enregistrement` ; relu strictement par `meteo.normaliser`) : la note recalcule
ensuite hors ligne. Un refus rend un message, la note intacte. Le modèle : `api_web/age.py` (`_importer_meteo`).

Une note à **plusieurs éléments** : le plan du bouton porte le **rang de l'élément** (`age.py` : `_marque`, clé
`coulage`), et le module refuse une récupération revenue alors qu'un autre élément est ouvert — l'écran envoie
« modifier » à l'élément OUVERT au retour, pas à celui du clic (une copie, même lieu et mêmes dates, y perdait ses
corrections à la main). `js/meteo.js` ne lit pas une réponse de plus de 5 Mo.

```python
p = MT.plan(c["temperature"]["lieu"], debut, fin, aujourdhui())        # ValueError → {"erreur": str(e)}
s += C.bouton_meteo(["temperature", "meteo"], "Récupérer la météo", p)
```

### 4 quinquies. La note PDF d'un module « à cas » (le générateur validé de l'écran Streamlit)

Décision de l'ingénieur du 09/11/2026 : les notes PDF de Streamlit passent sur le site. Pour un module dont la note est
faite de CAS et dont l'état du site EST l'état du moteur, `telecharger(etat, "note_pdf", infos)` rend
`note_cas.fichier(NOTE, etat, infos)` (`api_web/note_cas.py`) : le générateur validé n'est pas réécrit ; un cas seul,
c'est `generer_note` telle quelle ; plusieurs cas, ses pages (`Note`, `Methode`) posées l'une après l'autre dans UN
document du kit `ndc_pdf`, chaque cas dans la taille que le générateur choisit (deux copies nommées, liées par les tests :
texte de chaque page = celui du cas seul). Une note qui déborde encore est refusée, jamais tronquée — ce n'est pas un cas
d'école : mesuré le 09/11/2026, 86 des 240 états d'une grille du garde-corps débordent dans le générateur validé (toute
forme L ou U avec « Cumuler qh et qv » sauf fixation scellée, L ou U à montant soudé…), et n'ont donc pas de note sur le
site (docs/site/DEFAUTS_STREAMLIT.md, à trancher par le bureau). L'identification :
la saisie de l'écran d'abord, puis la note du site (projet = chantier, à défaut le nom ; partie = le nom s'il y a un
chantier, sinon le titre du module ; indice ; date). `NOTE.fichiers` : ce que la note charge en plus de l'écran (le paquet
de la note les embarque). Le bouton « Note PDF » de la page s'allume par `construire.NOTE_PDF`.

- **Garde-corps** (`acier/garde_corps/note.py`) : TOUTE la note, deux pages par garde-corps (calcul : hypothèses, quatre
  charges, montant, main courante, remplissage, ancrage avec formules développées, récapitulatif ; détails : platine,
  tableau des montants, section, soudure, méthode) — identification : la note du site seule (pas de champ à l'écran).
  Prouvé par `tests/test_note_garde_corps.py`.
- **Soudures** (`acier/soudures/note.py`) : TOUTE la note, deux pages par cas (dessins, gorge, cordons, efforts,
  vérification au point gouvernant avec les formules de la méthode choisie ; méthode et détail) — sans les trois
  lectures indicatives des autres méthodes que l'écran montre ; identification : les champs « Identification de la
  note » de l'écran. Prouvé par `tests/test_note_soudures.py`.
- **Charges sur plancher** (`plancher/charges/note.py`) : TOUTE la note, deux pages par cas (plan, coupe, éléments,
  charges, vérifications, conclusion ; méthode et détail) ; un cas que le moteur ne construit pas arrête la note avec le
  message de l'écran ; identification : les champs de CHAQUE cas. Prouvé par `tests/test_note_plancher.py`.

Écarts MESURÉS entre l'écran et la note (rapportés, non corrigés : la note est le générateur validé) — l'arrondi d'un
taux en bord de bande (écran 94 % / 101 %, note 95 % / 100 % pour 94,57 % / 100,009 %) ; entre 85 et 95 %, l'écran dit
« Conforme » (code du bureau) et les notes des soudures et du plancher « CONFORME · ATTENTION » (seuil 0,85 de leurs
moteurs, celui de Streamlit).

À trancher — l'INDICE a deux sources (R2) : pour les soudures et le plancher, le champ « Indice » de l'écran
(« Identification de la note ») vaut « 0 » par défaut et passe avant l'indice de la note du site ; le cartouche de la
page dit « Indice A » (ou l'indice de révision saisi, « C »…) et le PDF écrit « indice 0 » — sauf si l'on vide le
champ. La poutre et le garde-corps impriment l'indice de la note du site.

## 5. Les règles

1. **Parité fonctionnelle avec Streamlit** : chaque champ, option, résultat, avertissement et dessin de l'écran
   Streamlit existe sur le site (l'organisation peut et doit être meilleure). La note PDF : le générateur validé de
   Streamlit, porté tel quel (§ 4 quinquies, la poutre : `api_web/note_poutre.py`) ; tant qu'un module ne l'a pas, son
   bouton reste grisé. Ce qui manque est écrit dans le rapport et dans la doc du moteur.
2. **Le calcul est celui de Streamlit** : on extrait, on ne réécrit pas. Les défauts trouvés dans le module Streamlit
   sont **reproduits** (parité) et **écrits** (doc, rapport), surtout s'ils sont non conservatifs — on ne les corrige
   pas en silence. Exception : une saisie que Streamlit ramène en silence à une valeur fausse (négatif → 0…) est
   **refusée** sur le site avec un message.
3. **Rien de neuf sans le dire** : un module qui n'a pas de calcul dans Streamlit (flambement) porte un avertissement
   visible « nouveau calcul, à valider par le bureau », et ses tests reproduisent des exemples publiés.
4. **Sécurité** : tout texte venu de l'état est échappé (`C.esc`) dans le HTML ; aucun `<script>`, aucun `on…=`,
   aucune ressource externe (politique de sécurité du site). Seule ouverture, de DONNÉES et à la demande : la météo d'un
   lieu chez Open-Meteo (`C.bouton_meteo`, § 4 quater) ; ce qu'elle renvoie est relu par le moteur et échappé.
5. **Beau et lisible** : la charte des maquettes (`site/maquettes`) ; des dessins de dessinateur (traits, cotes,
   hachures, échelle) comme `acier/garde_corps/schemas.py` et `api_web/dessin_poutre.py` ; les nombres alignés ; rien
   de coupé ; le français du bureau (pas d'anglicisme, pas de jargon de développeur).
6. **Noms en français** (comme le dépôt), commentaires et docstrings en français.

## 6. Se vérifier avant de rendre

```bash
python3.12 tests/test_api_web_modules.py <clé>   # le CONTRAT commun : réponse complète, champs reliés, parcours sans erreur…
python3.12 tests/test_api_web_<clé>.py           # les tests propres
python3.12 tests/test_<x>_moteur.py              # la parité du moteur
# dans le vrai navigateur, sur une copie à soi (d'autres modules se construisent en même temps) :
python3.12 site/web/construire.py --dossier /tmp/site_<clé> --modules <clé>
python3.12 site/web/outils/serveur_local.py <port libre, 8800-8899> --dossier /tmp/site_<clé> &
node site/web/verifier_site.js http://127.0.0.1:<port> /tmp/captures_<clé> module:<clé>
```

Puis **regarder les captures** (`/tmp/captures_<clé>/site_<clé>*.png`) : le dessin est-il lisible et beau, le
résultat clair, rien de coupé ? Corriger tant que ce n'est pas le cas.

## 7. Le rapport final

Fichiers créés ; ce qui est couvert et ce qui ne l'est pas (par rapport à Streamlit) ; défauts trouvés dans le module
Streamlit ; résultats des tests (nombres OK/échecs) et du contrôle navigateur ; besoins communs éventuels.
