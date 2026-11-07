# -*- coding: utf-8 -*-
"""L'API du site pour les DONNÉES BÉTON (api_web/donnees_beton.py), vérifiée en Python ordinaire : chaque valeur
AFFICHÉE est celle de sa source, lue ici indépendamment du module (le fichier, la norme transcrite, l'écran Streamlit
de référence, l'image du bureau) — jamais à travers lui.

1. Une note neuve : une consultation, la rubrique « Classes de béton », le statut « Consultation ».
2. Classes de béton : les cases EN 1992-1-1 = le tableau 3.1 PUBLIÉ (transcrit dans tests/test_donnees_beton_moteur.py) ;
   les cases du bureau = beton_classes.json (α_b, μ) et τ_adm = k·f_ck,cube/1,05 ; la fiche, le verdict, la ligne
   surlignée ; « tau_lim » du fichier signalé comme lu par aucun moteur.
3. Aciers : f_yk, f_yd (bureau 1,5 ; EN 1992-1-1 1,15), E_s, ε_yd.
4. Tableau des armatures : les 12 × 15 cases et les masses = l'image du bureau (transcription de
   tests/test_armatures_moteur.py, relue dans ses pixels) ; la lecture « par mètre » ; le dessin des diamètres.
5. Enrobages : c_min,dur et la classe proposée = l'écran Streamlit modules/enrobage.py (lu dans son texte) ; c_nom
   recalculé ici ; la coupe ; le feu (FEU_* de l'écran), les cases à vérifier en rouge (poteau, voile R180 et R240).
6. Ancrages et recouvrements : vide → « Tableaux du bureau à venir », aucune valeur ; un fichier d'exemple (de forme)
   s'affiche tel quel, texte échappé ; un fichier abîmé est dit en rouge.
7. Chaque formule de la fenêtre « d'où viennent les valeurs », réévaluée avec ses valeurs écrites, redonne son résultat.
8. Saisies : un choix non proposé est refusé, la note intacte ; une note abîmée lève une erreur claire ; un nom
   piégé est échappé partout ; les consultations s'ajoutent, se copient, se suppriment.
9. Mutations : abîmer une source ou le module fait rougir le contrôle qui le garde.

    python3 tests/test_api_web_donnees_beton.py
"""
import ast
import copy
import html
import json
import math
import os
import re
import sys
import tempfile

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from api_web import commun as C  # noqa: E402
from api_web import donnees_beton as D  # noqa: E402
from api_web import routeur as R  # noqa: E402
from beton.armatures import tableau as T  # noqa: E402
from beton.donnees import ancrages as AN  # noqa: E402
from beton.donnees import consultation as K  # noqa: E402
from beton.donnees import ec2 as EC  # noqa: E402
from beton.enrobage import donnees as ED  # noqa: E402
from beton.poutre import entree as PE  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


def req(**k):
    return json.loads(R.traiter(json.dumps({"module": "donnees_beton", **k})))


def saisir(etat, chemin, valeur):
    return req(action="modifier", etat=copy.deepcopy(etat), chemin=chemin.split("/"), valeur=valeur)


def nombre(t: str) -> float:
    return float(t.replace(C.FINE, "").replace(C.NBSP, "").replace(" ", "").replace(",", "."))


def texte(h: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", h)).replace(C.FINE, " ").replace(C.NBSP, " ").strip()


def vue(r, cle):
    return next(v for v in r["vues"] if v["cle"] == cle)


RANGEE = re.compile(r'<tr( class="on")?><th class="nm">(.*?)</th>(.*?)</tr>', re.S)
CASE = re.compile(r"<td([^>]*)>(.*?)</td>", re.S)


def rangees(h: str) -> list:
    """Les rangées du corps d'un tableau de la toile : [(surlignée, nom, [(attributs, contenu HTML)])]."""
    out = []
    for on, nom, corps in RANGEE.findall(h):
        cases = CASE.findall(corps)
        if cases:
            out.append((bool(on), texte(nom), cases))
    return out


# ------------------------------------------------------------------ les sources, lues SANS le module
def _assignations(chemin: str, noms) -> dict:
    """Des constantes d'un fichier Python, lues dans son arbre syntaxique (sans l'importer)."""
    arbre = ast.parse(open(os.path.join(RACINE, chemin), encoding="utf-8").read())
    out = {}
    for n in arbre.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in noms:
            try:
                out[n.targets[0].id] = ast.literal_eval(n.value)
            except ValueError:
                pass
    return out


FICHIER = json.load(open(os.path.join(RACINE, "beton_classes.json"), encoding="utf-8"))
NORME = _assignations("tests/test_donnees_beton_moteur.py", {"TABLEAU_3_1"})["TABLEAU_3_1"]
_L = {li.split(";")[0]: [float(x.replace(",", ".")) for x in li.split(";")[1:]] for li in NORME.splitlines()}
PUBLIE = {f"C{int(a)}/{int(b)}": {k: v[i] for k, v in _L.items()} for i, (a, b) in enumerate(zip(_L["fck"], _L["fck_cube"]))}
IMAGE = [li.split(";") for li in _assignations("tests/test_armatures_moteur.py", {"TRANSCRIPTION"})["TRANSCRIPTION"].splitlines()]
ECRAN = _assignations("modules/enrobage.py", {"C_MIN_DUR", "CLASSE_STRUCTURALE_DEFAULT", "FEU_DALLE", "FEU_VOILE", "FEU_POTEAU",
                                              "FEU_POUTRE", "R_LEVELS"})
ECRAN["EXPO"] = list(_assignations("modules/enrobage.py", {"EXPOSITION_INFO"})["EXPOSITION_INFO"])


# ------------------------------------------------------------------ les contrôles (réutilisés par les mutations)
def ecarts_classes(r, groupe: str) -> list:
    """Les cases de la vue « Classes de béton » qui ne sont pas leur source : le tableau 3.1 publié (EN 1992-1-1), le
    fichier et τ_adm = k·f_ck,cube/1,05 (bureau)."""
    rg = rangees(vue(r, "tableau")["svg"])
    out = [("classes", [n for _, n, _ in rg], list(FICHIER))] if [n for _, n, _ in rg] != list(FICHIER) else []
    for _, nom, cases in rg:
        f, p = FICHIER.get(nom, {}), PUBLIE.get(nom, {})
        if groupe == "ec2":
            att = [p.get(k) for k in ("fck", "fck_cube", "fcm", "fctm", "fctk_005", "fctk_095", "Ecm")]
        else:
            att = [f.get("fck"), f.get("fck_cube"), round(f.get("alpha_b", 0), 2), f.get("mu_a500"), f.get("mu_a400")]
            att += [round(k * f.get("fck_cube", 0) / 1.05, 2) for k in (0.016, 0.032, 0.064)]
        lu = [nombre(texte(c)) for _, c in cases]
        if len(lu) != len(att) or any(abs(a - b) > 1e-9 for a, b in zip(lu, att)):
            out.append((nom, lu, att))
    return out


def ecarts_aciers(r) -> list:
    out = []
    for _, nom, cases in rangees(vue(r, "tableau")["svg"]):
        fyk = int(nom[1:])
        att = [fyk, round(fyk / 1.5), round(fyk / 1.15), 200, round(fyk / 1.15 / 200000 * 1000, 2)]
        lu = [nombre(texte(c)) for _, c in cases]
        if lu != att:
            out.append((nom, lu, att))
    return out


def ecarts_armatures(r) -> list:
    """Les rangées de la vue « Tableau des armatures » face à l'image du bureau : n = 1…10, la masse, e = 15 30 35 40."""
    rg = rangees(vue(r, "tableau")["svg"])
    out = [] if len(rg) == len(IMAGE) else [("rangées", len(rg), len(IMAGE))]
    for (_, nom, cases), li in zip(rg, IMAGE):
        lu = [texte(c) for _, c in cases]
        att = [li[0]] + [str(int(x)) for x in li[1:11]] + [li[11]] + [str(int(x)) for x in li[13:17]]
        if [nom] + [x.replace(" ", "") for x in lu] != att:
            out.append((nom, lu, att))
    return out


def lire_durabilite(r) -> dict:
    """{(classe structurale, exposition): (valeur, proposée, retenue)} lus dans la vue."""
    out = {}
    for _, s, cases in rangees(vue(r, "tableau")["svg"]):
        for x, (attrs, c) in zip(ECRAN["EXPO"], cases):
            cls = re.search(r'class="([^"]*)"', attrs).group(1).split()
            out[(s, x)] = (int(re.sub(r"<i.*?</i>", "", c)), "pro" in cls and "●" in c, "cur" in cls)
    return out


def ecarts_durabilite(r) -> list:
    lu = lire_durabilite(r)
    out = []
    for s, vals in ECRAN["C_MIN_DUR"].items():
        for x, v in zip(ECRAN["EXPO"], vals):
            if lu.get((s, x), (None, None, None))[:2] != (v, ECRAN["CLASSE_STRUCTURALE_DEFAULT"][x] == s):
                out.append((s, x, lu.get((s, x)), v))
    return out + ([("cases", len(lu))] if len(lu) != 42 else [])


CASE_FEU = re.compile(r'<td class="(dout)?"[^>]*>(?:<i class="mk">!</i>)?<b>(\d+)</b>(?:<small>b ≥ (\d+)</small>)?</td>|<td></td>')


def lire_feu(r) -> dict:
    out = {}
    for _, nom, corps in RANGEE.findall(vue(r, "tableau")["svg"]):
        cs = [(bool(d), int(a), int(b) if b else None) for d, a, b in CASE_FEU.findall(corps) if a]
        out[texte(nom)] = cs
    return out


def ecarts_feu(r) -> list:
    out = []
    lu = lire_feu(r)
    for x in ECRAN["R_LEVELS"]:
        att = [(False, ECRAN["FEU_DALLE"][x], None), (x in ("R180", "R240"), ECRAN["FEU_VOILE"][x], None)]
        att += [(False, a, b) for b, a in ECRAN["FEU_POUTRE"][x]] + [(True, a, b) for b, a in ECRAN["FEU_POTEAU"][x]]
        if lu.get(x) != att:
            out.append((x, lu.get(x), att))
    return out


FX = re.compile(r'<div class="fx">(.*?)</div>', re.S)


def evaluer(expr: str) -> float:
    """Une expression substituée telle qu'écrite (« 0,30 × 30^(2/3) », « max(20 ; 20 ; 10) », « π × 12² / 4 »)."""
    e = re.sub(r"(?<=\d) (?=\d{3}(?!\d))", "", expr.replace(C.FINE, " "))           # « 200 000 » : les milliers
    e = e.replace(C.NBSP, "").replace("10⁻⁶", "1e-6").replace("²", "**2")
    e = e.replace(",", ".").replace(";", ",").replace("×", "*").replace("^", "**").replace("π", "pi").replace("−", "-")
    return eval(e, {"__builtins__": {}}, {"max": max, "ln": math.log, "pi": math.pi})  # noqa: S307 — nos propres textes


def ecarts_formules(volets: str) -> list:
    """Chaque ligne « symbole = formule = valeurs = résultat » de la fenêtre : les valeurs réévaluées redonnent le
    résultat écrit, à la précision de son écriture (un ‰ est un millième)."""
    out = []
    n = 0
    for fx in FX.findall(volets):
        parts = [texte(p) for p in re.findall(r"<span>(.*?)</span>", fx, re.S)]
        res = texte(re.search(r'<span class="r">(.*?)</span>', fx, re.S).group(1))
        if len(parts) < 2:
            continue
        n += 1
        m = re.match(r"([\d  ,]+)", res)
        nd = len(m.group(1).split(",")[1]) if "," in m.group(1) else 0
        x = evaluer(parts[-1]) * (1000 if res.endswith("‰") else 1)
        if abs(x - nombre(m.group(1))) > 0.5 * 10 ** -nd + 6e-4 * abs(x):
            out.append((parts[-1], x, res))
    return out if n else [("aucune formule", volets[:80])]


# ================================================================================================ 1. note neuve
print("=== 1 · Une note neuve")
r0 = req(action="nouveau")
chk("nouveau : la requête aboutit", r0["ok"], r0.get("erreur"))
E0 = r0["etat"]
chk("l'état : une consultation « Consultation 1 », rubrique « classes », C30/37 et B500 (défauts des modules béton)",
    len(E0["consultations"]) == 1 and E0["consultations"][0]["nom"] == "Consultation 1" and E0["consultations"][0]["rubrique"] == "classes"
    and E0["consultations"][0]["classes"]["classe"] == PE.BETON_DEFAUT == "C30/37" and E0["consultations"][0]["aciers"]["nuance"] == 500)
chk("le résumé : une CONSULTATION (niveau i, sans taux), la ligne ouverte nommée",
    r0["resume"] == {"taux": None, "niveau": "i", "statut": "Consultation", "gouvernant": "Consultation 1 · Classes de béton · C30/37"}, r0["resume"])
chk("la pastille d'état dit « Consultation »", "Consultation" in r0["statut_html"])
chk("une vue : le tableau de la rubrique (un bloc HTML, jamais « à l'échelle »)",
    [(v["cle"], v["svg"][:4]) for v in r0["vues"]] == [("tableau", "<div")] and "db-tab" in r0["vues"][0]["svg"])
chk("le cadre Données : le nom, la rubrique (cinq), les colonnes, la classe, les sources",
    re.findall(r'data-chemin="([^"]+)"', r0["donnees"]) == ["nom", "rubrique", "classes/groupe", "classes/groupe", "classes/classe"]
    and [o for o in re.findall(r'<option value="([^"]*)"', r0["donnees"].split('data-chemin="classes/groupe"')[0])]
    == ["classes", "aciers", "armatures", "enrobages", "ancrages"])

# ================================================================================================ 2. classes
print("\n=== 2 · Classes de béton")
e = ecarts_classes(r0, "ec2")
chk(f"EN 1992-1-1 : les {len(FICHIER) * 7} cases sont celles du tableau 3.1 PUBLIÉ (f_ck, f_ck,cube, f_cm, f_ctm, f_ctk,0,05, f_ctk,0,95, E_cm)",
    not e, e[:2])
rb = saisir(E0, "classes/groupe", "bureau")
e = ecarts_classes(rb, "bureau")
chk(f"méthode du bureau : les {len(FICHIER) * 8} cases sont le fichier (α_b, μ) et τ_adm = k·f_ck,cube / 1,05", not e, e[:2])
th = re.findall(r'<th class="n" title="([^"]*)"><span class="s">(.*?)</span><span class="u">(.*?)</span>', vue(r0, "tableau")["svg"])
chk("les en-têtes portent le symbole (indices en <sub>), l'unité et le nom au survol",
    [(s, u) for _, s, u in th] == [("f<sub>ck</sub>", "MPa"), ("f<sub>ck,cube</sub>", "MPa"), ("f<sub>cm</sub>", "MPa"), ("f<sub>ctm</sub>", "MPa"),
                                   ("f<sub>ctk,0,05</sub>", "MPa"), ("f<sub>ctk,0,95</sub>", "MPa"), ("E<sub>cm</sub>", "GPa")]
    and th[3][0] == "résistance moyenne en traction", th)
chk("la classe ouverte (C30/37) est la seule rangée surlignée", [n for on, n, _ in rangees(vue(r0, "tableau")["svg"]) if on] == ["C30/37"])
v = re.search(r'<span class="chip xl c"[^>]*>(.*?)</span><div class="gov"><b>(.*?)</b>(.*?)</div>', r0["resultat"], re.S)
chk("le verdict : f_ck = 30 MPa en grand, « Béton C30/37 », f_ck,cube, E_cm, f_ctm du tableau 3.1",
    v and texte(v.group(1)) == "fck30MPa" and v.group(2) == "Béton C30/37" and "37 MPa" in texte(v.group(3))
    and "33 GPa" in texte(v.group(3)) and "2,9 MPa" in texte(v.group(3)), v and v.groups())
fiche = {texte(s): texte(n) for s, n in re.findall(r'<td class="s"[^>]*>(.*?)</td><td class="n">(.*?)</td>', r0["resultat"])}
f = FICHIER["C30/37"]
chk("la fiche : toutes les valeurs de C30/37, chacune à sa source",
    fiche == {"fck": "30", "fck,cube": "37", "fcm": "38", "fctm": "2,9", "fctk,0,05": "2,0", "fctk,0,95": "3,8", "Ecm": "33",
              "αb": f"{f['alpha_b']:.2f}".replace(".", ","), "μB500": "0,1709", "μB400": "0,1904",
              "τadm,I": "0,56", "τadm,II": "1,13", "τadm,IV": "2,26"}, fiche)
chk("« tau_lim » du fichier est dit lu par aucun moteur (sa valeur, 0,70 N/mm², affichée comme telle)",
    "« tau_lim »" in r0["resultat"] and "0,70" in r0["resultat"] and "n'est lu par aucun module de calcul" in r0["resultat"])  # (le harnais suit : « moteur » est un mot de développeur)
r2 = saisir(E0, "classes/classe", "C40/50")
chk("choisir C40/50 (Données) : la rangée surlignée et le verdict suivent",
    [n for on, n, _ in rangees(vue(r2, "tableau")["svg"]) if on] == ["C40/50"] and "Béton C40/50" in r2["resultat"])
s = req(action="selecteur", etat=E0, role="ligne")
lignes = re.findall(r'class="pk-row( cur)?[^"]*" data-choix="([^"]+)"', s["html"])
chk("le sélecteur de la ligne : les classes du fichier, la ouverte cochée", [n for _, n in lignes] == list(FICHIER)
    and [n for c, n in lignes if c] == ["C30/37"], lignes)
r3 = req(action="choisir", etat=E0, role="ligne", nom="C20/25")
chk("un clic dans le sélecteur ouvre la classe", r3["etat"]["consultations"][0]["classes"]["classe"] == "C20/25" and "Béton C20/25" in r3["resultat"])

# ================================================================================================ 3. aciers
print("\n=== 3 · Aciers d'armature")
ra = saisir(E0, "rubrique", "aciers")
e = ecarts_aciers(ra)
chk("B400 et B500 : f_yk ; f_yd = f_yk/1,5 (bureau) et f_yk/1,15 (EN 1992-1-1) ; E_s = 200 GPa ; ε_yd = f_yd/E_s", not e and len(rangees(vue(ra, "tableau")["svg"])) == 2, e)
chk("les deux γ_s sont nommés dans l'en-tête : bureau 1,5, EN 1992-1-1 1,15",
    "Bureau · γs = 1,5" in texte(vue(ra, "tableau")["svg"]) and "EN 1992-1-1 · γs = 1,15" in texte(vue(ra, "tableau")["svg"]))
chk("le verdict : f_yk = 500 MPa ; f_yd 333 (bureau) et 435 (EN 1992-1-1)", "Acier B500" in ra["resultat"]
    # (le harnais suit : « EN 1992-1-1 : f_yd = 435 MPa », la source d'abord — relecture adverse du 07/11/2026)
    and re.search(r"bureau\s?: fyd\s?=\s?333\s?MPa", texte(ra["resultat"])) and re.search(r"EN 1992-1-1\s?: fyd\s?=\s?435\s?MPa", texte(ra["resultat"])))
ra4 = saisir(ra["etat"], "aciers/nuance", "400")
chk("la nuance B400 (segment) : la rangée et la fiche suivent", ra4["etat"]["consultations"][0]["aciers"]["nuance"] == 400
    and [n for on, n, _ in rangees(vue(ra4, "tableau")["svg"]) if on] == ["B400"] and "Acier B400" in ra4["resultat"])
chk("le γ_s du bureau est dit « à confirmer » pour l'annexe nationale belge (EN 1992-1-1)", "annexe nationale belge est à confirmer" in ra["resultat"])

# ================================================================================================ 4. armatures
print("\n=== 4 · Tableau des armatures")
rm = saisir(E0, "rubrique", "armatures")
e = ecarts_armatures(rm)
chk(f"les {len(IMAGE)} rangées, leurs 14 sections et leur masse sont celles de l'image du bureau (relue dans ses pixels)", not e, e[:2])
chk("la rangée du Ø ouvert (Ø12, le défaut du tableau des armatures) est surlignée",
    [n for on, n, _ in rangees(vue(rm, "tableau")["svg"]) if on] == ["12"])
chk("deux vues : le tableau, puis les diamètres à l'échelle (le Ø ouvert encadré)",
    [v["cle"] for v in rm["vues"]] == ["tableau", "diametres"] and vue(rm, "diametres")["svg"].startswith("<svg"))
rn = saisir(rm["etat"], "armatures/lecture", "nappe")
chk("lecture « par mètre » : l'en-tête dit e = 100/n (cm) sous chaque n, le groupe des espacements est mis en avant",
    "Barres par mètre · e en cm" in vue(rn, "tableau")["svg"] and all(f"<small>{x}</small>" in vue(rn, "tableau")["svg"] for x in ("33,3", "16,7", "10,0"))
    and not ecarts_armatures(rn))
r16 = saisir(rm["etat"], "armatures/d", "16")
v = re.search(r'<span class="chip xl c"[^>]*>(.*?)</span>', r16["resultat"])
chk("Ø16 : a = π·16²/4 = 201,1 mm² en grand, m = 1,580 kg/m (l'image)", texte(v.group(1)) == "a201,1mm²" and "m = 1,580" in texte(r16["resultat"]),
    v and v.group(1))

# ================================================================================================ 5. enrobages
print("\n=== 5 · Enrobages")
re_ = saisir(E0, "rubrique", "enrobages")
e = ecarts_durabilite(re_)
chk("durabilité : les 42 cases c_min,dur et les classes proposées (●) sont celles de l'écran Streamlit (modules/enrobage.py)", not e, e[:3])
lu = lire_durabilite(re_)
chk("la case ouverte (S4 · XC1) est la seule encadrée", [k for k, v in lu.items() if v[2]] == [("S4", "XC1")])


def c_nom(s, x, d, dd):
    return max(d, ECRAN["C_MIN_DUR"][s][ECRAN["EXPO"].index(x)], 10) + dd


for s, x, d, dd in (("S4", "XC1", 20, 10), ("S1", "X0", 12, 5), ("S6", "XD3/XS3", 25, 0), ("S3", "XC4", 32, 10)):
    rr = re_
    for ch, val in (("enrobages/classe_struct", s), ("enrobages/classe_expo", x), ("enrobages/diam_barre", str(d)), ("enrobages/delta_dev", str(dd))):
        rr = saisir(rr["etat"], ch, val)
    v = re.search(r'<span class="chip xl c"[^>]*>(.*?)</span>', rr["resultat"])
    chk(f"{s} · {x} · Ø{d} · Δc_dev {dd} : c_nom = max(Ø ; c_min,dur ; 10) + Δc_dev = {c_nom(s, x, d, dd)} mm (recalculé ici)",
        v and texte(v.group(1)) == f"cnom{c_nom(s, x, d, dd)}mm" and "Coupe d" in json.dumps([w["titre"] for w in rr["vues"]], ensure_ascii=False),
        v and v.group(1))
    if s == "S1":
        chk("S1 (sous S4) : l'avertissement du moteur de l'enrobage sur la classe réduite", "sous S4" in rr["resultat"])
rf = saisir(re_["etat"], "enrobages/tableau", "feu")
e = ecarts_feu(rf)
chk("feu : chaque distance à l'axe (et sa largeur b_min) est celle de l'écran Streamlit ; à vérifier en rouge : poteau (toutes) et voile R180, R240",
    not e, e[:2])
chk("feu : les avertissements rouges du moteur (poteau, μ_fi écrit avec son indice)",
    "Poteau : les distances a" in rf["resultat"] and 'class="msg grave"' in rf["resultat"] and "μ<sub>fi</sub>" in rf["resultat"])
rf2 = saisir(rf["etat"], "enrobages/r", "R240")
chk("R240 : la rangée surlignée, l'avertissement du voile (5 mm de plus à vérifier)",
    [n for on, n, _ in rangees(vue(rf2, "tableau")["svg"]) if on] == ["R240"] and "Voile R240" in rf2["resultat"])

# ================================================================================================ 6. ancrages
print("\n=== 6 · Ancrages et recouvrements")
rk = saisir(E0, "rubrique", "ancrages")
t = vue(rk, "tableau")["svg"]
chk("le fichier vide : « Tableaux du bureau à venir » dans la toile, le résultat et le cadre Données",
    "Tableaux du bureau à venir" in t and "Tableaux du bureau à venir" in rk["resultat"] and "Tableaux du bureau à venir" in rk["donnees"])
chk("… et AUCUNE valeur : ni case de tableau, ni nombre dans la toile, ni sélecteur de ligne",
    "<td" not in t and not re.search(r"\d", texte(t).replace("ancrages_recouvrements", "")) and "data-pk" not in rk["resultat"], texte(t)[:120])
chk("la fenêtre dit le fichier à remplir et son format", 'id="ck-source"' in rk["volets"] and "ancrages_recouvrements.json" in rk["volets"])
PIEGE = '<img src=x onerror=alert(1)>'
EXEMPLE = {"version": 1, "titre": "Exemple", "etat": "—", "aide": "",
           "tableaux": [{"cle": "forme_a", "titre": f"Forme A {PIEGE}", "grandeur": "l_x", "unite": "Ø", "source": f"essai {PIEGE}",
                         "conditions": ["condition de forme"], "colonnes": {"titre": "Ø", "unite": "mm", "valeurs": [1, 2, 3]},
                         "lignes": [{"libelle": "ligne un", "valeurs": [111, None, 333]}, {"libelle": f"ligne {PIEGE}", "valeurs": [444, 555, 666]}],
                         "notes": []},
                        {"cle": "forme_b", "titre": "Forme B", "grandeur": "l_y", "unite": "cm", "source": "essai",
                         "colonnes": {"titre": "Ø", "unite": "mm", "valeurs": [7]}, "lignes": [{"libelle": "seule", "valeurs": [777]}]}]}
garde = AN.CHEMIN
with tempfile.TemporaryDirectory() as tmp:
    AN.CHEMIN = os.path.join(tmp, "a.json")
    try:
        json.dump(EXEMPLE, open(AN.CHEMIN, "w", encoding="utf-8"), ensure_ascii=False)
        rx = saisir(E0, "rubrique", "ancrages")
        rg = rangees(vue(rx, "tableau")["svg"])
        chk("un fichier rempli : le premier tableau s'affiche tel quel (cases, case vide « — »), la première ligne ouverte",
            [(on, n, [texte(c) for _, c in cs]) for on, n, cs in rg] == [(True, "ligne un", ["111", "—", "333"]),
                                                                        (False, f"ligne {PIEGE}", ["444", "555", "666"])], rg)
        chk("… son titre, sa source et ses libellés sont ÉCHAPPÉS (texte, jamais balisage)",
            all("<img" not in h for h in (vue(rx, "tableau")["svg"], rx["resultat"], rx["volets"], rx["donnees"])))
        rx2 = saisir(rx["etat"], "ancrages/ligne", "1")
        chk("choisir la ligne 2 : la rangée surlignée et la fiche suivent", [n for on, n, _ in rangees(vue(rx2, "tableau")["svg"]) if on] == [f"ligne {PIEGE}"]
            and "555" in rx2["resultat"])
        rx3 = saisir(rx["etat"], "ancrages/tableau", "forme_b")
        chk("choisir le second tableau : il s'affiche, la ligne revient à la première",
            [n for on, n, _ in rangees(vue(rx3, "tableau")["svg"]) if on] == ["seule"] and rx3["etat"]["consultations"][0]["ancrages"] == {"tableau": "forme_b", "ligne": 0})
        s = req(action="selecteur", etat=rx["etat"], role="ligne")
        chk("le sélecteur propose les lignes du tableau ouvert", re.findall(r'data-choix="([^"]+)"', s["html"]) == ["ligne un", html.escape(f"ligne {PIEGE}")])
        refus = saisir(rx["etat"], "ancrages/tableau", "inconnu")
        chk("un tableau inconnu est refusé, la note intacte", refus["message"] and refus["etat"] == rx["etat"])
        # une note qui vise un tableau retiré du fichier revient au premier (le fichier évolue sans les notes)
        n = D.normaliser({"consultations": [{"rubrique": "ancrages", "ancrages": {"tableau": "retire", "ligne": 9}}]})
        rr = req(action="calculer", etat=n)
        chk("une note qui vise un tableau retiré du fichier montre le premier, sa première ligne", rr["ok"]
            and [x for on, x, _ in rangees(vue(rr, "tableau")["svg"]) if on] == ["ligne un"])
        json.dump({**EXEMPLE, "tableaux": [{**EXEMPLE["tableaux"][0], "lignes": [{"libelle": "x", "valeurs": [1]}]}]},
                  open(AN.CHEMIN, "w", encoding="utf-8"))
        rz = saisir(E0, "rubrique", "ancrages")
        chk("un fichier abîmé : l'erreur dite en rouge, aucun tableau tronqué", rz["ok"] and 'class="msg grave"' in rz["resultat"]
            and "valeur(s) attendue(s)" in rz["resultat"] and "<td" not in vue(rz, "tableau")["svg"])
    finally:
        AN.CHEMIN = garde
chk("le vrai fichier est relu après l'essai (vide)", AN.lire()["tableaux"] == [])

# ================================================================================================ 7. formules
print("\n=== 7 · La fenêtre « d'où viennent les valeurs » : chaque formule redonne son résultat")
for nom, r in (("classes C30/37", r0), ("classes C40/50", r2), ("aciers B500", ra), ("aciers B400", ra4), ("armatures Ø12", rm),
               ("armatures Ø16", r16), ("enrobages S4 · XC1", re_)):
    e = ecarts_formules(r["volets"])
    chk(f"{nom} : les formules réévaluées avec leurs valeurs écrites redonnent leur résultat", not e, e[:2])
# (le harnais suit : la source se nomme dans les mots du bureau — « le calcul du module Poutre » — et non plus par le chemin
# du code, relecture adverse du 07/11/2026)
chk("la fenêtre a ses sources en tête (fichier, norme, calcul du module Poutre)", "beton_classes.json" in r0["volets"]
    and "EN 1992-1-1:2004, tableau 3.1" in texte(r0["volets"]) and "le calcul du module Poutre" in r0["volets"])

# ================================================================================================ 8. saisies
print("\n=== 8 · Saisies, refus, échappement, consultations")
for ch, val, mot in (("rubrique", "acier", "Rubrique inconnue"), ("classes/classe", "C45/55", "n'est pas proposé"),
                     ("aciers/nuance", "450", "n'est pas proposé"), ("armatures/d", "18", "n'est pas proposé"),
                     ("enrobages/delta_dev", "3", "n'est pas proposé"), ("enrobages/r", "R45", "n'est pas proposé"),
                     ("classes/inconnu", "x", "Champ inconnu"), ("ailleurs/x", "1", "Champ inconnu")):
    out = saisir(E0, ch, val)
    chk(f"« {ch} = {val} » est refusé (« {mot} »), la note intacte", out["ok"] and mot in (out["message"] or "") and out["etat"] == E0, out.get("message"))
for abime, mot in (({"consultations": [{"rubrique": "metal"}]}, "rubrique inconnue"),
                   ({"consultations": [{"classes": {"classe": "C99/115"}}]}, "n'est pas proposé"),
                   ({"consultations": "x"}, "doit être une liste"), ({"consultations": [3]}, "illisible")):
    try:
        D.normaliser(abime)
        chk(f"une note abîmée ({mot}) lève une erreur claire", False, "acceptée")
    except ValueError as err:
        chk(f"une note abîmée ({mot}) lève une erreur claire", mot in str(err), str(err))
n = saisir(E0, "nom", PIEGE)
chk("un nom piégé est échappé dans chaque fragment", n["ok"] and all("<img" not in n[k] for k in ("resultat", "volets", "donnees", "elements", "phrase"))
    and all("<img" not in v["svg"] for v in n["vues"]))
a = req(action="ajouter", etat=E0)
chk("« ajouter » : une seconde consultation neuve, ouverte", a["etat"]["actif"] == 1 and a["etat"]["consultations"][1]["nom"] == "Consultation 2"
    and a["elements"].count('data-activer="') == 2)
dup = req(action="dupliquer", etat=saisir(E0, "rubrique", "enrobages")["etat"])
chk("« copier » garde la rubrique et les choix", dup["etat"]["consultations"][1]["rubrique"] == "enrobages"
    and dup["etat"]["consultations"][1]["nom"] == "Consultation 1 (copie)")
sup = req(action="supprimer", etat=a["etat"], index=0)
chk("« supprimer » : la seconde reste, la dernière jamais", len(sup["etat"]["consultations"]) == 1
    and len(req(action="supprimer", etat=sup["etat"], index=0)["etat"]["consultations"]) == 1)
chk("la liste de la note nomme la rubrique en court et la ligne ouverte (« Béton · C30/37 »)", "· Béton · C30/37" in texte(r0["elements"]))

# ================================================================================================ 8 bis. lisibilité
print("\n=== 8 bis · Ce que les captures ont fait corriger (figé ici)")


def normes_coupees(r) -> list:
    """Les références « EN 1992-1-x » des textes du module (aide, légende, pied, verdict) qui ne sont pas gardées sur
    une ligne : coupées à leurs traits d'union, elles se lisaient « EN 1992-1- » puis « 1. » (capture à 1 440 px)."""
    h = r["donnees"] + vue(r, "tableau")["svg"] + re.sub(r'title="[^"]*"', "", r["resultat"])
    h = re.sub(r'<span class="db-nw">EN 1992-1-[12](?::2004)?</span>', "", re.sub(r"<(select|option)[^>]*>.*?</\1>", "", h, flags=re.S))
    return re.findall(r"EN 1992-1-[12]", re.sub(r"<[^>]+>", "", h))


RUBS = {k: saisir(E0, "rubrique", k) for k in ("classes", "aciers", "armatures", "enrobages")}
for k, r in RUBS.items():
    chk(f"{k} : chaque « EN 1992-1-x » de l'aide, de la légende et du verdict se lit sur une ligne", not normes_coupees(r), normes_coupees(r)[:3])
chk("la phrase sous le titre ne répète pas la légende de la vue (elle dit la ligne ouverte)",
    all(texte(r["phrase"]) != vue(r, "tableau")["legende"] for r in list(RUBS.values()) + [rf, rk]))
chk("un nom de fichier n'entre pas dans une formule (« beton_classes » y devenait beton<sub>classes</sub>)",
    "beton<sub>" not in "".join(r["volets"] for r in RUBS.values()) and "beton<sub>classes</sub>" in C.lignes_fx([{"parts": ["x", "beton_classes.json"], "res": "1", "niveau": ""}]))
chk("une légende de vue (texte, échappée par la page) n'écrit pas de symbole à souligné (« c_min,dur »)",
    all("_" not in v["legende"].replace("ancrages_recouvrements", "") for r in list(RUBS.values()) + [rf, rk] for v in r["vues"]),
    [v["legende"] for r in RUBS.values() for v in r["vues"]])

# ================================================================================================ 9. mutations
print("\n=== 9 · Mutations : chaque contrôle rougit quand on abîme ce qu'il garde")


def avec(objet, nom, valeur, faire):
    """Rejoue ``faire`` avec ``objet.nom`` remplacé, puis le rétablit (les caches du moteur vidés autour)."""
    garde_ = getattr(objet, nom)
    setattr(objet, nom, valeur)
    K._classes.cache_clear(), K._aciers.cache_clear()
    try:
        return faire()
    finally:
        setattr(objet, nom, garde_)
        K._classes.cache_clear(), K._aciers.cache_clear()


chk("mutation « f_ctm × 1,05 » : les cases EN 1992-1-1 ne sont plus le tableau publié (le contrôle rougit)",
    avec(EC, "fctm", lambda fck: 1.05 * 0.30 * fck ** (2 / 3), lambda: ecarts_classes(req(action="nouveau"), "ec2")))
lu_ = PE._classes_lues()
sauve = copy.deepcopy(lu_)
try:
    lu_["C25/30"]["mu_a400"] = 0.2
    K._classes.cache_clear()
    rouge = ecarts_classes(saisir(E0, "classes/groupe", "bureau"), "bureau")
finally:
    lu_.clear(), lu_.update(sauve), K._classes.cache_clear()
chk("mutation « μ_B400 de C25/30 changé dans le fichier lu par le site » : le contrôle des cases du bureau rougit", rouge)
chk("mutation « τ_adm avec γ = 1,0 » : le contrôle des cases du bureau rougit",
    avec(__import__("beton.poutre.moteur", fromlist=["x"]), "GAMMA_TAU", 1.0, lambda: ecarts_classes(saisir(E0, "classes/groupe", "bureau"), "bureau")))
chk("mutation « f_yd du bureau avec γ_s = 1,4 » : le contrôle des aciers rougit",
    avec(K, "GAMMA_S_BUREAU", 1.4, lambda: ecarts_aciers(saisir(E0, "rubrique", "aciers"))))
chk("mutation « arrondi vers le bas » du tableau des armatures : le contrôle face à l'image rougit",
    avec(T, "arrondi", lambda x: int(math.floor(x)), lambda: ecarts_armatures(saisir(E0, "rubrique", "armatures"))))
chk("mutation « ρ = 7 850 » : les masses ne sont plus celles de l'image (le contrôle rougit)",
    avec(T, "MASSE_VOLUMIQUE", 7850, lambda: ecarts_armatures(saisir(E0, "rubrique", "armatures"))))
cmd = copy.deepcopy(ED.C_MIN_DUR)
chk("mutation « c_min,dur S4 · XC1 = 15 » dans les tables du site : le contrôle face à l'écran Streamlit rougit",
    avec(ED, "C_MIN_DUR", {**cmd, "S4": [15, 15] + cmd["S4"][2:]}, lambda: ecarts_durabilite(saisir(E0, "rubrique", "enrobages"))))
chk("mutation « classe proposée de XC4 = S3 » : le contrôle des ● rougit",
    avec(ED, "CLASSE_STRUCTURALE_DEFAULT", {**ED.CLASSE_STRUCTURALE_DEFAULT, "XC4": "S3"},
         lambda: ecarts_durabilite(saisir(E0, "rubrique", "enrobages"))))
feu_etat = rf["etat"]
chk("mutation « poteau R60 : a = 46 » : le contrôle du feu rougit",
    avec(ED, "FEU_POTEAU", {**ED.FEU_POTEAU, "R60": [(250, 46), (350, 40)]}, lambda: ecarts_feu(req(action="calculer", etat=feu_etat))))
import beton.enrobage.moteur as EM  # noqa: E402

chk("mutation « avertissements du moteur de l'enrobage tus » : les cases à vérifier ne sont plus rouges (le contrôle rougit)",
    avec(EM, "avertissements", lambda res: [], lambda: ecarts_feu(req(action="calculer", etat=feu_etat))))
chk("mutation « f_ctm écrit à la place de 0,7 × f_ctm » dans la fenêtre : le contrôle des formules rougit",
    ecarts_formules(r0["volets"].replace("0,7 × 2,896", "0,7 × 3,896")))
chk("mutation « références de norme laissées sécables » : le contrôle de lisibilité rougit",
    avec(D, "_normes", lambda h: h, lambda: normes_coupees(saisir(E0, "rubrique", "aciers"))))
chk("mutation « le nom échappé n'est plus échappé » : le contrôle d'échappement rougit",
    avec(C, "esc", lambda x: str(x), lambda: any("<img" in h for h in (lambda q: (q["resultat"], q["donnees"], q["elements"]))(saisir(E0, "nom", PIEGE)))))

# ================================================================================================ 10. relecture adverse
print("\n=== 10 · Relecture adverse du 07/11/2026 : ce qu'elle a trouvé, figé ici")
CHOIX = [("classes", "groupe"), ("classes", "classe"), ("aciers", "nuance"), ("armatures", "lecture"), ("armatures", "d"),
         ("enrobages", "tableau"), ("enrobages", "classe_expo"), ("enrobages", "classe_struct"), ("enrobages", "diam_barre"),
         ("enrobages", "delta_dev"), ("enrobages", "r")]


def booleens_admis() -> list:
    """Les choix qu'un booléen (true / false d'un fichier .stab abîmé) remplit encore : False valait 0 et True 1 —
    « delta_dev »: false donnait Δc_dev = 0 mm (c_nom de 30 à 20 mm), sans un mot ; NON CONSERVATIF."""
    out = []
    for rub, k in CHOIX:
        for b in (True, False):
            try:
                n = D.normaliser({"consultations": [{rub: {k: b}}]})
                out.append(("relu", rub, k, b, n["consultations"][0][rub][k]))
            except ValueError:
                pass
            e = D.nouvel_etat()
            if D.modifier(e, [rub, k], b) is None:
                out.append(("saisi", rub, k, b, e["consultations"][0][rub][k]))
    return out


e = booleens_admis()
chk("un booléen n'est jamais un choix (« delta_dev »: false n'est pas Δc_dev = 0 mm), ni relu ni saisi", not e, e[:3])

NOM_MAX = getattr(D, "NOM_MAX", 80)


def noms_admis() -> list:
    """Les noms absurdes qu'une consultation accepte : trop longs (un nom d'un million de caractères rendait une
    réponse de 4 Mo), pas un texte (une liste devenait « [1, 2] »), ou porteurs d'un caractère invisible ou de
    commande (U+202E retourne l'affichage de la liste de la note)."""
    out = []
    for nom in ("x" * (NOM_MAX + 1), [1, 2], {"a": 1}, 12, "Dalle‮ R+1", "a\nb", "a‏b", "a b"):
        e = D.nouvel_etat()
        if D.modifier(e, ["nom"], nom) is None:
            out.append(("saisi", repr(nom)[:24]))
        try:
            D.normaliser({"consultations": [{"nom": nom}]})
            out.append(("relu", repr(nom)[:24]))
        except ValueError:
            pass
    return out


e = noms_admis()
chk(f"un nom trop long (> {NOM_MAX}), qui n'est pas un texte ou qui porte un caractère invisible est refusé (saisi ou relu)",
    not e, e[:4])
ok_noms = [n for n in ("x" * NOM_MAX, "Dalle R+1 · enrobage", "", None) if saisir(E0, "nom", n).get("message") is not None]
chk(f"… un nom ordinaire, vide, ou de {NOM_MAX} caractères passe", not ok_noms, ok_noms)
long_ = saisir(E0, "nom", "x" * 1_000_000)
chk("un nom d'un million de caractères : un refus lisible, une réponse de taille ordinaire (elle pesait 4 Mo)",
    long_["ok"] and long_["message"] and long_["etat"] == E0 and len(json.dumps(long_)) < 100_000, len(json.dumps(long_)))


def saisies_mal_dites() -> list:
    """Les requêtes abîmées qui rendent une exception Python (anglais, nom de classe) au lieu d'un refus en français,
    ou qui ouvrent ou SUPPRIMENT une consultation qu'aucun index ne désignait (« supprimer -5 » supprimait la
    première : la borne ramenait l'index dans la liste)."""
    out = []
    e2 = req(action="ajouter", etat=E0)["etat"]
    for q in (dict(action="modifier", etat=E0, chemin=["rubrique"], valeur=["classes"]),
              dict(action="modifier", etat=E0, chemin=["rubrique"], valeur={"a": 1})):
        r = req(**q)
        if not (r.get("ok") and r.get("message") and r["etat"] == E0):
            out.append((q["valeur"], r.get("erreur") or r.get("message")))
    for action, index in (("activer", "abc"), ("activer", 7), ("activer", -1), ("activer", True), ("supprimer", -5),
                          ("supprimer", 9), ("supprimer", "x"), ("supprimer", 1.5)):
        r = req(action=action, etat=e2, index=index)
        err = r.get("erreur") or ""
        if r.get("ok") or "consultation" not in err.lower() or re.search(r"invalid|unhashable|cannot|Overflow|TypeError", err):
            out.append((action, index, err[:60] or "acceptée"))
    r = json.loads(R.traiter('{"module": "donnees_beton", "action": "activer", "index": 1e400, "etat": ' + json.dumps(e2) + "}"))
    if r.get("ok") or "consultation" not in (r.get("erreur") or "").lower():
        out.append(("activer", "1e400", (r.get("erreur") or "acceptée")[:60]))
    return out


e = saisies_mal_dites()
chk("une saisie ou un index abîmés : un refus en français, aucune consultation ouverte ni supprimée au hasard", not e, e[:4])
chk("… un index valable ouvre et supprime toujours", req(action="activer", etat=req(action="ajouter", etat=E0)["etat"], index=0)["etat"]["actif"] == 0
    and len(req(action="supprimer", etat=req(action="ajouter", etat=E0)["etat"], index=1)["etat"]["consultations"]) == 1)


def notes_relues_en_silence() -> list:
    """Les notes abîmées que normaliser relit EN SILENCE (une note neuve, une clé jetée, un tableau devenu le premier)
    au lieu de lever l'erreur claire que le contrat demande."""
    out = []
    for abime in ([1, 2], "note", {"consultations": {}}, {"consultations": 0}, {"consultations": [{"pirate": 1}]},
                  {"consultations": [{"classes": {"classe": "C30/37", "pirate": 1}}]}, {"consultations": [{"ancrages": [1]}]},
                  {"consultations": [{"ancrages": {"tableau": 5}}]}, {"consultations": [{"ancrages": {"ligne": -1}}]},
                  {"consultations": [{"ancrages": {"ligne": True}}]}, {"consultations": [{"ancrages": {"x": 1}}]}):
        try:
            D.normaliser(abime)
            out.append(json.dumps(abime, ensure_ascii=False)[:60])
        except ValueError:
            pass
    return out


e = notes_relues_en_silence()
chk("une note abîmée (pas un objet, une clé inconnue, un tableau d'ancrage qui n'est pas un texte…) lève une erreur claire", not e, e[:4])
chk("… une note vide ou sans consultation est une note neuve", all(D.normaliser(x) == D.nouvel_etat() for x in
                                                                   ({}, {"consultations": []}, {"consultations": None})))


def feu_sans_reserve(r) -> list:
    """Les valeurs du feu que le moteur de l'enrobage dit « à vérifier dans la norme » (poteau : toutes ; voile : R180 et
    R240) et que la FICHE ou le VERDICT du résultat écrivent comme des valeurs sûres. La toile les montrait en rouge,
    le résultat non — et au téléphone, c'est le résultat qu'on lit."""
    duree = r["etat"]["consultations"][r["etat"]["actif"]]["enrobages"]["r"]
    att = {"Dalle": False, "Voile": duree in ("R180", "R240"), "Poutre": False, "Poteau": True}
    out = []
    groupes = re.findall(r'<tr class="g"><td colspan="3">(.*?)</td></tr>(.*?)(?=<tr class="g">|</table>)', r["resultat"], re.S)
    vus = set()
    for titre, corps in groupes:
        typ = texte(titre).split(" ")[0]
        if typ in att:
            vus.add(typ)
            marques = ["db-dout" in v for v in re.findall(r'<td class="n">(.*?)</td>', corps)]
            if not marques or any(m != att[typ] for m in marques):
                out.append(("fiche", duree, typ, marques))
    gov = re.search(r'<div class="gov">(.*?)</div>', r["resultat"], re.S).group(1)
    l1, _, l2 = gov.partition("<br>")
    avant, _, voile = l1.partition("voile")
    if "db-dout" in avant or ("db-dout" in voile) != att["Voile"] or "db-dout" not in l2:
        out.append(("verdict", duree, texte(gov)))
    return out + ([("groupes", sorted(vus))] if vus != set(att) else [])


e = [x for d in ED.R_LEVELS for x in feu_sans_reserve(saisir(rf["etat"], "enrobages/r", d))]
chk("feu, chaque durée : les valeurs « à vérifier » (poteau, voile R180 et R240) le sont aussi dans la fiche et le verdict, "
    "les autres non", not e, e[:3])
chk("mutation « avertissements du moteur tus » : la fiche n'a plus de réserve (le contrôle rougit)",
    avec(EM, "avertissements", lambda res: [], lambda: feu_sans_reserve(saisir(rf["etat"], "enrobages/r", "R240"))))


def egal_secable(r) -> list:
    """Les « = » du verdict et de la phrase qu'une espace ordinaire laisse couper : « Δc_dev = » en fin de ligne,
    « 10 mm » à la suivante (captures PC du 07/11/2026 : verdicts de l'enrobage et du feu)."""
    gov = "".join(re.findall(r'<div class="gov">(.*?)</div>', r["resultat"], re.S))
    return re.findall(r".{0,10}(?: =|= ).{0,10}", html.unescape(re.sub(r"<[^>]+>", "", gov + "\n" + r["phrase"])))


TOUS = {"classes": r0, "bureau": rb, "aciers": ra, "armatures": rm, "nappe": rn, "durabilité": re_, "feu": rf2, "ancrages": rk}
e = {k: egal_secable(r) for k, r in TOUS.items() if egal_secable(r)}
chk("verdict et phrase : « symbole = valeur » ne se coupe jamais au « = » (espaces insécables)", not e, list(e.items())[:2])


def legendes_secables(r) -> list:
    """Les légendes des vues (du TEXTE, posé et échappé par la page : la protection par balise n'y entre pas) où
    « EN 1992-1-x » se coupe — au téléphone, « … bureau et EN 1992- » puis « 1-1 » (capture du 07/11/2026, Aciers).
    Gardée : « EN », espace insécable, « 1992 », chaque trait d'union suivi d'un gluon (U+2060, sans largeur)."""
    gardees = ("EN 1992-⁠1-⁠1", "EN 1992-⁠1-⁠2")
    return [m.group(0) for v in r["vues"] for m in re.finditer(r"EN\W{0,2}1992\W{0,2}1\W{0,2}[12]", v["legende"]) if m.group(0) not in gardees]


e = {k: legendes_secables(r) for k, r in TOUS.items() if legendes_secables(r)}
chk("chaque « EN 1992-1-x » d'une légende de vue se lit sur une ligne", not e, list(e.items())[:2])
chk("… et la légende garde ses mots (la page les écrit tels quels)", "bureau et EN" in vue(ra, "tableau")["legende"]
    and "1992" in vue(ra, "tableau")["legende"])

JARGON = re.compile(r"\.py\b|\btests/|\bmoteurs?\b|\b(?!DONNEES_BETON\.md)[A-Z][A-Z0-9]*_[A-Z0-9_…]+|\b(?:moteur|entree|tableau|dessins)\.[a-z_]+"
                    r"|\bbeton/(?!donnees/ancrages_recouvrements\.json)[a-z_/]+")
SOULIGNE = re.compile(r"(?<![\w/])(?!beton_classes|ancrages_recouvrements|tau_lim)[A-Za-zΑ-Ωα-ω]+_[a-z]+")


def jargon(r, sel: str = "") -> list:
    """Les noms de CODE écrits à l'écran (fichier Python, fonction, constante, test, chemin d'un paquet) et les
    symboles laissés à souligné (« f_yd » au lieu de f<sub>yd</sub>) : le bureau lit ses sources en français — le
    fichier des classes, l'écran Enrobage, la norme. Restent nommés les FICHIERS que le bureau ouvre lui-même
    (beton_classes.json, le fichier des ancrages à remplir) et la page de documentation de son format."""
    tout = (r["resultat"] + r["volets"] + r["donnees"] + r["phrase"] + r["elements"] + sel
            + "".join(v["svg"] for v in r["vues"] if v["svg"].startswith("<div")))
    visible = html.unescape(re.sub(r"<[^>]+>", " ", tout)) + " " + " ".join(v["legende"] + " " + v["titre"] for v in r["vues"])
    titres = " ".join(html.unescape(x) for x in re.findall(r'title="([^"]*)"', tout))
    return sorted({m.group(0) for m in JARGON.finditer(visible + " " + titres)} | {m.group(0) for m in SOULIGNE.finditer(visible)})


e = {k: jargon(r, req(action="selecteur", etat=r["etat"], role="ligne").get("html", "")) for k, r in TOUS.items()}
e = {k: v for k, v in e.items() if v}
chk("aucun nom de code à l'écran (ni fichier .py, ni fonction, ni constante, ni test), aucun symbole à souligné", not e, list(e.items())[:3])
chk("mutation « une source écrite comme un chemin de code » : le contrôle du jargon rougit",
    jargon({**r0, "phrase": r0["phrase"] + " (beton/poutre/moteur.materiaux)"}))

with tempfile.TemporaryDirectory() as tmp:
    AN.CHEMIN = os.path.join(tmp, "a.json")
    try:
        json.dump({**EXEMPLE, "tableaux": [{**EXEMPLE["tableaux"][0], "lignes": [{"libelle": "x", "valeurs": [1]}]}]},
                  open(AN.CHEMIN, "w", encoding="utf-8"))
        rz = saisir(E0, "rubrique", "ancrages")


        def illisible_tu(rz) -> list:
            return [nom for nom, h in (("toile", vue(rz, "tableau")["svg"]), ("données", rz["donnees"]),
                                       ("verdict", rz["resultat"].split('<div class="els">')[0]), ("phrase", rz["phrase"]),
                                       ("liste de la note", rz["elements"]))
                    if "illisible" not in h or "à venir" in h]
        tu = illisible_tu(rz)
        chk("un fichier des ancrages abîmé : la toile, le cadre Données, le verdict et la phrase disent « illisible » (et non "
            "« à venir… aucune valeur proposée en attendant », comme si rien n'avait été versé)", not tu, tu)
        chk("… la toile dit l'endroit fautif", "valeur(s) attendue(s)" in vue(rz, "tableau")["svg"] and "<td" not in vue(rz, "tableau")["svg"])
        lire_ = D._ancrages
        chk("mutation « l'erreur du fichier tue » : le contrôle du fichier illisible rougit",
            avec(D, "_ancrages", lambda: {**lire_(), "erreur": None}, lambda: illisible_tu(saisir(E0, "rubrique", "ancrages"))))
        json.dump(EXEMPLE, open(AN.CHEMIN, "w", encoding="utf-8"), ensure_ascii=False)
        rx = saisir(E0, "rubrique", "ancrages")
        chk("un tableau d'ancrage : « ligne » = true n'ouvre pas la ligne 2 (un booléen n'est pas un index)",
            saisir(rx["etat"], "ancrages/ligne", True)["message"] and saisir(rx["etat"], "ancrages/ligne", "1")["message"] is None)
        e = jargon(rx, req(action="selecteur", etat=rx["etat"], role="ligne").get("html", ""))
        chk("un tableau d'ancrage affiché : aucun nom de code", not e, e)
        chk("la liste de la note nomme le tableau et la ligne ouverts (elle disait « à venir » sous un tableau affiché)",
            "Forme A" in texte(rx["elements"]) and "ligne un" in texte(rx["elements"]) and "à venir" not in rx["elements"], texte(rx["elements"]))
        chk("mutation « la liste dit toujours à venir » : le contrôle rougit",
            avec(D, "_resume_ancrages", lambda c: "à venir", lambda: "à venir" in saisir(E0, "rubrique", "ancrages")["elements"]))
    finally:
        AN.CHEMIN = garde
chk("le vrai fichier est relu après l'essai (vide)", AN.lire()["tableaux"] == [])


def _ancien_lire_choix(valeur, permis):
    """La lecture d'un choix AVANT la relecture adverse : ``valeur == p`` laissait passer True == 1, False == 0."""
    for p in permis:
        if valeur == p or (not isinstance(valeur, bool) and str(valeur) == str(p)):
            return p
    return None


for nom, objet, attr, valeur, controle in (
        ("un booléen lu comme un choix (l'ancienne lecture)", D, "_lire_choix", _ancien_lire_choix, booleens_admis),
        ("nom jamais refusé", D, "_refus_nom", lambda n: None, noms_admis),
        ("index ramené dans la liste (l'ancienne borne)", D, "_index", lambda index, n: min(max(0, int(index or 0)), n - 1), saisies_mal_dites),
        ("valeur douteuse écrite sans réserve", D, "_douteux", lambda h: h,
         lambda: feu_sans_reserve(saisir(rf["etat"], "enrobages/r", "R180"))),
        ("« = » laissé sécable (et normes non gardées)", D, "_normes", lambda h: h, lambda: egal_secable(saisir(E0, "rubrique", "enrobages"))),
        ("légende laissée sécable", D, "_legende", lambda t: t, lambda: legendes_secables(saisir(E0, "rubrique", "aciers")))):
    try:
        rouge = avec(objet, attr, valeur, controle)
    except Exception as err:  # noqa: BLE001 — une mutation qui fait planter le contrôle le fait rougir aussi
        rouge = f"exception : {err}"
    chk(f"mutation « {nom} » : le contrôle rougit", rouge)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)
