# -*- coding: utf-8 -*-
"""La TAILLE DU CODE du site, module par module (demande du propriétaire, 13/11/2026 : « un petit tableau qui dit combien
de lignes de code pour le site, combien par module […] et après chaque lot, on voit la réduction »).

CE QUI EST COMPTÉ
- une LIGNE DE CODE est une ligne qui porte une instruction : ni vide, ni commentaire, ni docstring (Python : relevé par
  ``tokenize`` et ``ast`` ; JavaScript et CSS : les lignes de commentaire // et /* … */ mises à part). La colonne
  « avec commentaires » compte toutes les lignes du fichier ;
- le SITE est ce que la page charge : les fichiers Python du paquet du moteur (``site/web/moteur/manifeste.json``, la
  liste que la construction écrit), les générateurs de la note PDF (``modules/export_pdf*.py``, ``ndc_pdf/``,
  ``site/note_pdf/``), le JavaScript et les feuilles de style de ``site/web``, et la construction des pages
  (``site/web/construire.py``, ``site/maquettes/construire.py``) ;
- HORS DU SITE, compté à part : l'ancien logiciel Streamlit (les écrans de ``modules/``, ``streamlit_app.py``, leurs
  morceaux d'interface restés dans ``acier/``) et les TESTS (``tests/``, le harnais du navigateur).

Chaque fichier du site tombe dans UN groupe — un module de l'accueil, les bibliothèques, ou le code commun ; un fichier
qu'aucune règle ne range fait échouer le comptage (``ValueError``) : le tableau ne peut pas oublier du code en silence.

    /usr/bin/python3.12 outils/taille_du_code.py                      # le tableau d'aujourd'hui (Markdown)
    /usr/bin/python3.12 outils/taille_du_code.py --enregistrer "texte" # ajoute la mesure, datée, à l'historique
    /usr/bin/python3.12 outils/taille_du_code.py --comparer            # aujourd'hui face à la première mesure

L'historique (``docs/site/taille_du_code.json``) est une suite de MESURES datées : on en ajoute, on n'en réécrit pas.
"""
import ast
import datetime
import io
import json
import os
import re
import subprocess
import sys
import tokenize
import warnings

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFESTE = os.path.join(RACINE, "site", "web", "moteur", "manifeste.json")
HISTORIQUE = os.path.join(RACINE, "docs", "site", "taille_du_code.json")

# (clé, libellé) dans l'ordre de l'accueil (site/maquettes/construire.FAMILLES), puis les groupes du site qui ne sont pas
# des modules ; la Prédalle est absorbée par les Dalles (une page, DALLES_FUSION.md § 2.4) : un seul groupe
GROUPES = (
    ("poutre", "Poutres béton armé"),
    ("dalle", "Dalles et prédalles"),
    ("plancher", "Charges sur plancher"),
    ("fondation_grue", "Fondations de grues"),
    ("taux", "Taux d'armature"),
    ("enrobage", "Enrobage"),
    ("age", "Maturité du béton"),
    ("armatures", "Tableau des armatures"),
    ("donnees_beton", "Données béton"),
    ("profil", "Choix de profilé"),
    ("catalogue", "Catalogue des profilés"),
    ("corniere", "Cornières"),
    ("flambement", "Flambement"),
    ("assemblages", "Assemblages métalliques"),
    ("soudures", "Soudures"),
    ("garde_corps", "Garde-corps"),
    ("bois", "Poutres bois"),
    ("sol", "Rigidité du sol"),
    ("bibliotheques", "Bibliothèques (béton, acier, autres)"),
    ("commun", "Code commun (écran, note PDF, construction)"),
)
HORS_SITE = (("streamlit", "Ancien logiciel Streamlit (hors site)"), ("tests", "Tests (hors site)"))

# les règles, dans l'ordre : la première qui reconnaît le chemin (relatif à la racine, « / ») range le fichier
_R = re.compile
REGLES = (
    ("poutre", _R(r"^(beton/poutre/|api_web/(poutre|note_poutre|dessin_poutre|coupe_ba|poutre_streamlit)\.py$|"
                  r"modules/export_pdf\.py$|site/web/css/modules/poutre\.css$)")),
    ("dalle", _R(r"^(beton/(dalle|predalle)/|beton/treillis\.py$|api_web/(dalle|dalle_nappes|dalle_positions|dessin_dalle|"
                 r"note_dalle|plaque_compact|dalle_predalle|dessin_predalle)\.py$|modules/export_pdf_dalle\.py$|"
                 r"site/web/css/modules/(dalle|predalle)\.css$)")),
    ("plancher", _R(r"^(plancher/|api_web/(plancher|dessin_plancher)\.py$|site/web/css/modules/plancher\.css$)")),
    ("fondation_grue", _R(r"^(beton/fondation_grue/|api_web/(fondation_grue|note_fondation_grue)\.py$|"
                          r"site/web/css/modules/fondation_grue\.css$|site/web/js/(plans|plans-travailleur)\.js$)")),
    ("taux", _R(r"^(beton/taux/|api_web/(taux|dessin_taux)\.py$|site/web/css/modules/taux\.css$)")),
    ("enrobage", _R(r"^(beton/enrobage/|api_web/(enrobage|dessin_enrobage)\.py$|site/web/css/modules/enrobage\.css$)")),
    ("age", _R(r"^(beton/maturite/|api_web/(age|dessin_age)\.py$|site/web/css/modules/age\.css$|site/web/js/meteo\.js$)")),
    ("armatures", _R(r"^(beton/armatures/|api_web/armatures\.py$|site/web/css/modules/armatures\.css$)")),
    ("donnees_beton", _R(r"^(beton/donnees/|api_web/donnees_beton\.py$|site/web/css/modules/donnees_beton\.css$|"
                         r"beton_classes\.json$|docs/validation/exemples_publies\.json$)")),
    ("profil", _R(r"^(acier/choix_profil/|acier/barre/|api_web/(profil\w*|dessin_profil|dessin_barre|note_profil)\.py$|site/web/css/modules/profil\.css$)")),
    ("catalogue", _R(r"^(acier/profiles/|api_web/(catalogue|dessin_catalogue)\.py$|site/web/css/modules/catalogue\.css$)")),
    ("corniere", _R(r"^(acier/corniere/|api_web/(corniere|dessin_corniere|note_corniere)\.py$|"
                    r"site/web/css/modules/corniere\.css$)")),
    ("flambement", _R(r"^(acier/flambement/|api_web/(flambement|dessin_flambement)\.py$|"
                      r"site/web/css/modules/flambement\.css$)")),
    ("assemblages", _R(r"^(acier/assemblages/|api_web/(assemblages|assemblages_pied|assemblages_poteau|assemblages_poutre|"
                       r"note_assemblages)\.py$|site/web/css/modules/assemblages\.css$)")),
    ("soudures", _R(r"^(acier/soudures/|api_web/soudures\.py$|site/web/css/modules/soudures\.css$)")),
    ("garde_corps", _R(r"^(acier/garde_corps/|api_web/garde_corps\.py$|site/web/css/modules/garde_corps\.css$)")),
    ("bois", _R(r"^(bois/|api_web/(bois|dessin_bois)\.py$|site/web/css/modules/bois\.css$)")),
    ("sol", _R(r"^(sol/|api_web/(sol|dessin_sol)\.py$|site/web/css/modules/sol\.css$)")),
    ("bibliotheques", _R(r"^(beton/(bibliotheque|hourdis)/|api_web/(biblio_\w+|dessin_biblio_\w+)\.py$|"
                         r"site/web/css/bibliotheque[\w-]*\.css$|site/web/js/(biblio|biblio_note|bibliotheque)\.js$)")),
    ("commun", _R(r"^(api_web/(__init__|commun|etabli_compact|routeur|registre|parcours|fiches|icones|note_cas|pdf_reunion|gabarit)\.py$|"
                  r"beton/__init__\.py$|beton/position\.py$|acier/|rdm/|ndc_pdf/|site/note_pdf/|site/web/js/|site/web/css/|"
                  r"site/web/construire\.py$|site/maquettes/construire\.py$)")),
)


# ------------------------------------------------------------------ compter
def _lignes_python(texte: str) -> int:
    """Les lignes qui portent une instruction : celles d'un jeton autre qu'un commentaire ou une fin de ligne, moins
    celles des docstrings (module, classe, fonction)."""
    porteuses = set()
    for t in tokenize.generate_tokens(io.StringIO(texte).readline):
        if t.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER):
            continue
        porteuses.update(range(t.start[0], t.end[0] + 1))
    docs = set()
    with warnings.catch_warnings():               # (un « \( » d'un fichier compté n'est pas l'affaire du comptage)
        warnings.simplefilter("ignore")
        arbre = ast.parse(texte)
    for n in ast.walk(arbre):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body:
            p = n.body[0]
            if isinstance(p, ast.Expr) and isinstance(p.value, ast.Constant) and isinstance(p.value.value, str):
                docs.update(range(p.lineno, p.end_lineno + 1))
    return len(porteuses - docs)


def _lignes_js_css(texte: str, js: bool) -> int:
    """Les lignes qui ne sont ni vides, ni des commentaires (// en tête de ligne pour le JavaScript, /* … */ partout)."""
    n, dans_bloc = 0, False
    for ligne in texte.splitlines():
        s = ligne.strip()
        if dans_bloc:
            if "*/" not in s:
                continue
            s, dans_bloc = s.split("*/", 1)[1].strip(), False
        while s.startswith("/*"):
            if "*/" in s:
                s = s.split("*/", 1)[1].strip()
            else:
                s, dans_bloc = "", True
        if not s or (js and s.startswith("//")):
            continue
        n += 1
    return n


def compter(chemin: str):
    """(lignes de code, lignes du fichier) d'un fichier du dépôt."""
    with open(os.path.join(RACINE, chemin), encoding="utf-8") as f:
        texte = f.read()
    total = len(texte.splitlines())
    if chemin.endswith(".py"):
        return _lignes_python(texte), total
    if chemin.endswith((".js", ".mjs")):
        return _lignes_js_css(texte, True), total
    if chemin.endswith(".css"):
        return _lignes_js_css(texte, False), total
    return 0, 0                                   # (les données JSON du paquet : rangées, pas comptées comme du code)


# ------------------------------------------------------------------ ce qui est le site, ce qui ne l'est pas
def _suivis() -> list:
    out = subprocess.run(["git", "ls-files"], cwd=RACINE, capture_output=True, text=True, check=True).stdout
    return [x for x in out.splitlines() if x]


def fichiers_du_site(suivis: list) -> list:
    with open(MANIFESTE, encoding="utf-8") as f:
        moteur = [x for x in json.load(f)["fichiers"]]
    note = [x for x in suivis if re.match(r"^(modules/export_pdf(_dalle)?\.py|ndc_pdf/[^/]+\.py|site/note_pdf/.+\.py)$", x)]
    ecran = [x for x in suivis if re.match(r"^site/web/(js/[^/]+\.js|css/.+\.css)$", x)]
    construction = [x for x in ("site/web/construire.py", "site/maquettes/construire.py") if x in suivis]
    return sorted(set(moteur) | set(note) | set(ecran) | set(construction))


def groupe_de(chemin: str) -> str:
    for cle, motif in REGLES:
        if motif.search(chemin):
            return cle
    raise ValueError(f"aucune règle ne range « {chemin} » : ajouter une règle dans outils/taille_du_code.py")


def mesurer() -> dict:
    """{groupe: [lignes de code, lignes avec commentaires, fichiers]} pour le site et, à part, hors du site."""
    suivis = _suivis()
    site = fichiers_du_site(suivis)
    out = {cle: [0, 0, 0] for cle, _ in GROUPES + HORS_SITE}
    for f in site:
        g = groupe_de(f)
        c, t = compter(f)
        out[g][0] += c
        out[g][1] += t
        out[g][2] += 1
    dans_le_site = set(site)
    for f in suivis:
        if f in dans_le_site:
            continue
        if re.match(r"^(tests/.+\.py|site/web/verifier_[\w-]+\.js|outils/oracle_[\w-]+\.js)$", f):
            g = "tests"
        elif re.match(r"^(modules/[^/]+\.py|streamlit_app\.py|acier/.+\.py)$", f):
            g = "streamlit"
        else:
            continue
        c, t = compter(f)
        out[g][0] += c
        out[g][1] += t
        out[g][2] += 1
    return out


# ------------------------------------------------------------------ dire
def _nb(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def tableau(m: dict, ref: dict = None, titre_ref: str = "") -> str:
    """Le tableau Markdown : chaque module de l'accueil, le code commun, le total du site ; puis, à part, hors du site.
    Avec ``ref`` (une mesure d'avant), une colonne « écart » en lignes de code."""
    total = [sum(m[cle][k] for cle, _ in GROUPES) for k in range(3)]
    m = dict(m, total=total)
    if ref is not None:
        ref = dict(ref, total=[sum(ref.get(cle, [0])[0] for cle, _ in GROUPES)])

    def ligne(cle, nom, gras=False):
        c, t, n = m[cle]
        ecart = ""
        if ref is not None:
            d = c - ref.get(cle, [0])[0]
            ecart = " | " + ("—" if d == 0 else f"{'+' if d > 0 else '−'}{_nb(abs(d))}")
        cases = [nom, _nb(c), _nb(t), str(n)] + ([ecart[3:]] if ecart else [])
        if gras:
            cases = [f"**{x}**" for x in cases]
        return "| " + " | ".join(cases) + " |"
    tete = "| Module | Lignes de code | Avec commentaires | Fichiers" + (f" | Écart depuis {titre_ref}" if ref else "") + " |"
    sep = "|---|---:|---:|---:" + ("|---:" if ref else "") + "|"
    return "\n".join([tete, sep] + [ligne(cle, nom) for cle, nom in GROUPES] + [ligne("total", "Total du site", True)]
                     + [ligne(cle, nom) for cle, nom in HORS_SITE])


def _historique() -> list:
    if not os.path.exists(HISTORIQUE):
        return []
    with open(HISTORIQUE, encoding="utf-8") as f:
        return json.load(f)["mesures"]


def enregistrer(etiquette: str) -> dict:
    m = mesurer()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=RACINE, capture_output=True, text=True).stdout.strip()
    h = _historique()
    h.append({"date": datetime.date.today().isoformat(), "commit": commit, "etiquette": etiquette, "groupes": m})
    with open(HISTORIQUE, "w", encoding="utf-8") as f:
        json.dump({"lire": "outils/taille_du_code.py : une MESURE datée par lot ; on en ajoute, on n'en réécrit pas",
                   "mesures": h}, f, ensure_ascii=False, indent=1)
        f.write("\n")
    return m


def main(argv) -> int:
    if "--enregistrer" in argv:
        i = argv.index("--enregistrer")
        etiquette = argv[i + 1] if i + 1 < len(argv) else ""
        if not etiquette:
            print("--enregistrer demande une étiquette (« avant l'unification », « lot 2 »…)")
            return 2
        print(tableau(enregistrer(etiquette)))
        return 0
    m = mesurer()
    if "--comparer" in argv:
        h = _historique()
        if not h:
            print("aucune mesure enregistrée : lancer d'abord --enregistrer")
            return 2
        print(tableau(m, h[0]["groupes"], f"le {h[0]['date']} ({h[0]['etiquette']})"))
        return 0
    print(tableau(m))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
