# -*- coding: utf-8 -*-
"""Le catalogue des hourdis Ergon (beton/donnees/hourdis/ergon_2018.json) et son lecteur (beton/hourdis/catalogue.py).

Le fichier est transcrit hors production par tools/transcrire_hourdis_ergon.py, qui le confronte au PDF du fabricant
(relecture du texte, relevé des coupes ; mesure docs/validation/hourdis_ergon_transcription_2026-11-11.md). Le PDF
n'est pas dans le dépôt : ce test garde ce qui peut l'être sans lui.

1. Le dossier : chaque fichier est nommé par CATALOGUES, et réciproquement.
2. La COHÉRENCE de chaque profil (``defauts``) : chaînes de cotes = b et h, nombre d'alvéoles, poids en kN/m² = kN/m
   ÷ largeur nominale, M_Rd croissant dans l'ordre du tableau et avec l'aire des torons (EN 10138-3), M_Rd avec chape
   > M_Rd, désignations lisibles et uniques, provenance (pages) ; le DESSIN relevé redonne A, I_x, W publiés à 2 %
   près, sa largeur 1196 mm, sa hauteur h, ses alvéoles au droit des cotes.
3. L'EMPREINTE des valeurs de calcul (caractéristiques et tableaux) : changer une valeur du catalogue exige de
   relancer l'outil sur le PDF et de reporter ici la nouvelle empreinte — une valeur ne se corrige pas à la main.
4. Des valeurs relues dans le guide, écrites ici une seconde fois (copie nommée, liée au fichier par ce test).
5. Le lecteur : fiche du module Charges sur plancher (unités, E_cm du tableau 3.1, nature Rd, chape C30/37 ≥ 50 mm,
   V_Rd avec chape absent), torons, noms de type, recherche, bibliothèque standard seulement.
6. Mutations : chaque défaut injecté dans une copie du catalogue fait rougir le contrôle qui le garde.

    python3 tests/test_hourdis_catalogue.py
"""
import copy
import hashlib
import json
import os
import re
import subprocess
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from beton.donnees import ec2 as EC  # noqa: E402
from beton.hourdis import catalogue as K  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


# EN 10138-3, torons à 7 fils : aire nominale (mm²) selon le diamètre (mm) — pour l'ordre des M_Rd seulement
AIRE_TORON = {9.3: 52.0, 12.5: 93.0, 15.2: 140.0}
EMPREINTE_ERGON = "6005d087c266aa9f"


def _integrales(pts):
    A = S = I = 0.0
    for i in range(len(pts)):
        (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % len(pts)]
        c = x0 * y1 - x1 * y0
        A += c
        S += (y0 + y1) * c
        I += (y0 * y0 + y0 * y1 + y1 * y1) * c
    s = 1.0 if A > 0 else -1.0
    return s * A / 2.0, s * S / 6.0, s * I / 12.0


def section_dessin(p):
    """A, I_x, W_sup, W_inf du dessin relevé (contour moins alvéoles), intégrés ici sans le moteur."""
    A, S, I = _integrales(p["dessin"]["contour"])
    for v in p["dessin"]["alveoles"]:
        a, s, i = _integrales(v)
        A, S, I = A - a, S - s, I - i
    h = max(y for _, y in p["dessin"]["contour"])
    yg = S / A
    Ig = I - A * yg * yg
    return {"A": A, "Ix": Ig, "W_sup": Ig / (h - yg), "W_inf": Ig / yg}


def empreinte(cat):
    """L'empreinte des valeurs de CALCUL et de mise en œuvre (pas du dessin, pas des textes)."""
    vals = [[p["nom"], p["page"], p["h"], p["b"], p["poids_kN_m"], p["poids_kN_m2"], p["A"], p["Ix"], p["W_sup"], p["W_inf"],
             p["transport"], p["joints"], p["cotes"]["chaine_horizontale"], p["cotes"]["chaine_verticale"],
             p["cotes"]["alveoles"], p["appui"], p["retrecis"], p["reservations"], p["ancrage_alveole"],
             [[v["nom"], v["M_Rd"], v["V_Rd"], v["M_Rd_chape"]] for v in p["variantes"]]] for p in cat["profils"]]
    m = cat["meta"]
    vals.append([m["largeur_nominale"]["valeur"], m["beton"]["classe"], m["beton"]["fck"], m["valeurs"]["nature"],
                 m["chape"]["epaisseur_min"], m["chape"]["classe"], m["chape"]["fck"], m["notation"]["torons"]])
    return hashlib.sha256(json.dumps(vals, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def defauts(cat):
    """Les défauts de cohérence d'un catalogue (liste vide : cohérent)."""
    d = []
    m = cat["meta"]
    b_nom = m["largeur_nominale"]["valeur"]
    for cle_ in ("fabricant", "edition", "document", "transcription"):
        if not m.get(cle_):
            d.append(f"meta.{cle_} absent")
    if m["valeurs"]["nature"] not in ("Rd", "service"):
        d.append("nature inconnue")
    noms = set()
    for p in cat["profils"]:
        n, c = p["nom"], p["cotes"]
        if not isinstance(p.get("page"), int) or not p["page"] > 0:
            d.append(f"{n} : page absente")
        if abs(sum(c["chaine_horizontale"]) - p["b"]) > 1e-6:
            d.append(f"{n} : chaîne horizontale {sum(c['chaine_horizontale'])} ≠ b {p['b']}")
        if abs(sum(c["chaine_verticale"]) - p["h"]) > 1e-6:
            d.append(f"{n} : chaîne verticale {sum(c['chaine_verticale'])} ≠ h {p['h']}")
        a = c["alveoles"]
        if a["nombre"] != len(c["chaine_horizontale"]) - 1:
            d.append(f"{n} : {a['nombre']} alvéoles pour {len(c['chaine_horizontale'])} cotes")
        if a["forme"] == "oblongue" and (len(c["chaine_verticale"]) != 3 or c["chaine_verticale"][1] != a["hauteur"]):
            d.append(f"{n} : hauteur d'alvéole hors de la chaîne verticale")
        if a["forme"] == "circulaire" and not a["diametre"] < p["h"]:
            d.append(f"{n} : diamètre ≥ h")
        if a["forme"] == "profilée" and len(a["largeurs"]) != a["nombre"]:
            d.append(f"{n} : largeurs d'alvéoles")
        if abs(p["poids_kN_m"] / (b_nom / 1000.0) - p["poids_kN_m2"]) > 0.0051:
            d.append(f"{n} : {p['poids_kN_m']} kN/m ÷ {b_nom / 1000.0} ≠ {p['poids_kN_m2']} kN/m²")
        if not (p["W_sup"] < p["Ix"] and p["W_inf"] < p["Ix"]):
            d.append(f"{n} : modules")
        # le dessin relevé
        s = section_dessin(p)
        for k in ("A", "Ix", "W_sup", "W_inf"):
            if abs(s[k] / p[k] - 1.0) > 0.02:
                d.append(f"{n} : {k} du dessin {s[k]:.4g} à {100 * (s[k] / p[k] - 1):+.1f} % du publié {p[k]:.4g}")
        xs = [x for x, _ in p["dessin"]["contour"]]
        ys = [y for _, y in p["dessin"]["contour"]]
        if abs(max(xs) - min(xs) - p["b"]) > 1.0 or abs(max(ys) - min(ys) - p["h"]) > 1.0:
            d.append(f"{n} : dessin {max(xs) - min(xs):.1f} × {max(ys) - min(ys):.1f} pour {p['b']} × {p['h']}")
        if len(p["dessin"]["alveoles"]) != a["nombre"]:
            d.append(f"{n} : {len(p['dessin']['alveoles'])} alvéoles dessinées pour {a['nombre']}")
        else:
            centres_cotes, x = [], 0.0
            for e in c["chaine_horizontale"][:-1]:
                x += e
                centres_cotes.append(x)
            centres = sorted((min(q[0] for q in v) + max(q[0] for q in v)) / 2.0 for v in p["dessin"]["alveoles"])
            pire = max(abs(u - w) for u, w in zip(centres, centres_cotes))
            if pire > 5.0:
                d.append(f"{n} : alvéole dessinée à {pire:.1f} mm de sa cote")
        # les variantes
        prec_M, prec_aire = -1.0, -1.0
        for v in p["variantes"]:
            if not v["nom"].startswith(n + "-"):
                d.append(f"{v['nom']} : hors du profil {n}")
                continue
            if v["nom"] in noms:
                d.append(f"{v['nom']} : en double")
            noms.add(v["nom"])
            if not v["M_Rd"] > prec_M:
                d.append(f"{v['nom']} : M_Rd {v['M_Rd']} ≤ la ligne précédente {prec_M}")
            prec_M = v["M_Rd"]
            if not (v["V_Rd"] > 0 and v["M_Rd"] > 0):
                d.append(f"{v['nom']} : valeur nulle")
            if v.get("M_Rd_chape") is not None and not v["M_Rd_chape"] > v["M_Rd"]:
                d.append(f"{v['nom']} : M_Rd avec chape ≤ M_Rd")
            try:
                aire = sum(k * AIRE_TORON[o] for k, o in torons_de(cat, p, v))
            except (ValueError, KeyError) as e:
                d.append(f"{v['nom']} : désignation illisible ({e})")
                continue
            if not aire > prec_aire:
                d.append(f"{v['nom']} : aire des torons {aire} ≤ la ligne précédente {prec_aire} alors que M_Rd croît")
            prec_aire = aire
    return d


def torons_de(cat, p, v):
    """Les torons d'une variante, par le lecteur (un catalogue copié n'est pas dans le cache du lecteur : on passe
    par une copie de la notation, comme le lecteur la lit)."""
    notation = {k.upper(): float(x) for k, x in cat["meta"]["notation"]["torons"].items()}
    out = []
    for morceau in v["nom"][len(p["nom"]) + 1:].split("-"):
        mm = re.fullmatch(r"(\d+)([A-Za-z]?)", morceau)
        if not mm:
            raise ValueError(morceau)
        out.append((int(mm.group(1)), notation[mm.group(2).upper()]))
    return out


# ================================================================================================ 1 · le dossier
fichiers = sorted(f for f in os.listdir(K.DOSSIER) if not f.startswith("."))
chk("dossier : chaque fichier est nommé par CATALOGUES, et réciproquement", fichiers == sorted(K.CATALOGUES), fichiers)
chk("dossier : un catalogue Ergon 2018", [k for k, _ in K.catalogues()] == ["ergon_2018"])
cat = K.catalogue("ergon_2018")
chk("lecture : titre « Ergon 2018 »", K.titre(cat) == "Ergon 2018")

# ================================================================================================ 2 · cohérence
d0 = defauts(cat)
chk("cohérence : aucun défaut sur le catalogue livré", d0 == [], d0)
chk("cohérence : 7 profils, 53 variantes", len(cat["profils"]) == 7 and sum(len(p["variantes"]) for p in cat["profils"]) == 53)
chk("cohérence : profils dans l'ordre du guide",
    [p["nom"] for p in cat["profils"]] == ["SP 150", "SP 200", "SPN 200", "SP 265", "SP 320", "SP 400", "SPZ 500"])
chk("cohérence : une page par profil, 29 à 35", [p["page"] for p in cat["profils"]] == list(range(29, 36)))
ecarts = cat["meta"]["ecarts"]
chk("écarts du guide nommés : SP 265 (chaîne), SP 400 (p. 20), SPZ 500 (joints)",
    [e["profil"] for e in ecarts] == ["SP 265", "SP 400", "SPZ 500"] and all(e["texte"] and e["page"] for e in ecarts))
p265 = K.profil(cat, "SP 265")
chk("écart SP 265 : le bord livré vaut 154 (imprimé 222)", p265["cotes"]["chaine_horizontale"] == [154, 222, 222, 222, 222, 154])
chk("écart SP 400 : alvéoles de 216 mm (p. 34), pas 218 (p. 20)", K.profil(cat, "SP 400")["cotes"]["alveoles"]["largeur"] == 216)
pz = K.profil(cat, "SPZ 500")
chk("écart SPZ 500 : joints 18,2 (p. 35) et 16,6 (p. 20) livrés tous deux", pz["joints"] == 18.2 and pz["joints_p20"] == 16.6)
chk("non publié : V_Rd,c avec couche collaborante nommé", any("V_Rd,c avec couche" in t for t in cat["meta"]["non_publie"]))
chk("aucune variante ne porte de V_Rd avec chape (non publié)", all("V_Rd_chape" not in v for p in cat["profils"] for v in p["variantes"]))

# ================================================================================================ 3 · empreinte
e = empreinte(cat)
chk(f"empreinte des valeurs de calcul = {EMPREINTE_ERGON} (relancer l'outil sur le PDF pour en changer)", e == EMPREINTE_ERGON, e)

# ================================================================================================ 4 · valeurs relues
RELUES = {  # (profil, désignation) : (M_Rd kN·m, V_Rd,c kN, M_Rd avec 50 mm kN·m), relues dans le guide
    "SP 150-5X": (44, 95, 61), "SP 150-9X": (77, 101, 103), "SP 200-4-2X": (113, 94, 141), "SP 200-3-2X": (92, 92, 116),
    "SPN 200-6x": (76, 101, 97), "SPN 200-8-2x": (191, 115, 233), "SP 265-2-4X": (133, 138, 159),
    "SP 265-8-2X": (279, 145, 319), "SP 320-5": (195, 170, 223), "SP 320-11": (418, 185, 464),
    "SP 400-9": (447, 229, 487), "SP 400-13": (635, 234, 687), "SPZ 500-8": (497, 342, 544),
    "SPZ 500-14-2Y": (1006, 394, 1069),
}
for nom, (m_, v_, mc_) in RELUES.items():
    _, var = K.variante(cat, nom)
    chk(f"relue : {nom} = {m_} / {v_} / {mc_}", (var["M_Rd"], var["V_Rd"], var["M_Rd_chape"]) == (m_, v_, mc_), var)
CARAC = {  # p. 20 : h, b, kN/m, kN/m², A (10³ mm²), I_x (10⁶ mm⁴), W_sup, W_inf (10⁶ mm³)
    "SP 150": (150, 1196, 2.94, 2.45, 114, 291, 3.84, 3.93), "SP 200": (200, 1196, 3.33, 2.78, 127, 630, 6.22, 6.38),
    "SPN 200": (200, 1196, 3.51, 2.92, 134, 648, 6.24, 6.74), "SP 265": (265, 1196, 4.59, 3.83, 174, 1467, 10.64, 11.54),
    "SP 320": (320, 1196, 4.94, 4.12, 187, 2421, 15.03, 15.23), "SP 400": (400, 1196, 5.93, 4.94, 222, 4541, 22.44, 22.97),
    "SPZ 500": (500, 1196, 7.56, 6.30, 276, 8550, 33.78, 34.63),
}
for nom, t in CARAC.items():
    p = K.profil(cat, nom)
    lu = (p["h"], p["b"], p["poids_kN_m"], p["poids_kN_m2"], p["A"] / 1e3, p["Ix"] / 1e6, p["W_sup"] / 1e6, p["W_inf"] / 1e6)
    chk(f"relue p. 20 : {nom}", all(abs(a - b) < 1e-9 for a, b in zip(lu, t)), lu)
m = cat["meta"]
chk("relue : béton C55/67 (p. 13, 20), chape C30/37 ≥ 50 mm (p. 23), valeurs de calcul ELU (nature Rd)",
    m["beton"]["classe"] == "C55/67" and m["beton"]["fck"] == 55 and m["chape"]["epaisseur_min"] == 50
    and m["chape"]["fck"] == 30 and m["chape"]["page"] == 23 and m["valeurs"]["nature"] == "Rd")
chk("relue : appuis SP 200 80 / 120 mm, SPZ 500 130 / 200 mm (p. 25)",
    K.profil(cat, "SP 200")["appui"] == {"beton_acier": 80, "maconnerie": 120}
    and pz["appui"] == {"beton_acier": 130, "maconnerie": 200})
chk("relue : ancrage en alvéole SP 400 6,0 kN, rien pour SPN 200 et SPZ 500 (p. 28)",
    K.profil(cat, "SP 400")["ancrage_alveole"] == 6.0 and K.profil(cat, "SPN 200")["ancrage_alveole"] is None
    and pz["ancrage_alveole"] is None)

# ================================================================================================ 5 · le lecteur
f = K.fiche_plancher("ergon_2018", "SP 200-4-2X")
chk("fiche plancher : nom « SP 200-4-2X · Ergon 2018 », coupé au « · » = la désignation",
    f["nom"] == "SP 200-4-2X · Ergon 2018" and f["nom"].split(" · ")[0] == "SP 200-4-2X")
chk("fiche plancher : b = 1,2 m (largeur nominale), h = 0,2 m, g = 2,78 kN/m² (avec joints)",
    f["b_element"] == 1.2 and f["h"] == 0.2 and f["g"] == 2.78)
chk("fiche plancher : I = 630·10⁶ mm⁴ = 6,30·10⁻⁴ m⁴", abs(f["I"] - 6.30e-4) < 1e-15)
chk("fiche plancher : E = E_cm(C55/67) du tableau 3.1 = 38 214 MPa",
    f["E"] == round(EC.Ecm(55.0) * 1000.0) == 38214.0)
chk("fiche plancher : M_Rd 113, V_Rd 94, nature Rd, M_Rd avec chape 141, V_Rd avec chape absent",
    (f["M_Rd"], f["V_Rd"], f["nature"], f["M_Rd_chape"], f["V_Rd_chape"]) == (113.0, 94.0, "Rd", 141.0, None))
chk("fiche plancher : conditions de la chape — 0,05 m, f_ck 30 MPa", f["e_chape"] == 0.05 and f["fck_exigee"] == 30.0)
chk("fiche plancher : M_cr et pression locale non publiés (None)", f["M_cr"] is None and f["p_loc_adm"] is None)
chk("fiche plancher : la source cite les pages 30 et 20 et le béton", "p. 30" in f["source"] and "p. 20" in f["source"]
    and "C55/67" in f["source"] and "Ergon" in f["source"])
f2 = K.fiche_plancher("ergon_2018", "SP 200-4-2X")
f2["M_Rd"] = 0.0
chk("fiche plancher : une nouvelle fiche à chaque appel (le catalogue n'est pas touché)",
    K.fiche_plancher("ergon_2018", "SP 200-4-2X")["M_Rd"] == 113.0)
toutes = K.fiches_plancher()
chk("fiches plancher : 53, noms uniques", len(toutes) == 53 and len({x["nom"] for x in toutes}) == 53)
chk("fiches plancher : toutes Rd, M_Rd et V_Rd > 0", all(x["nature"] == "Rd" and x["M_Rd"] > 0 and x["V_Rd"] > 0 for x in toutes))
gr = K.groupes_plancher()
chk("groupes : un par profil, dans l'ordre, 53 noms", len(gr) == 7 and gr[0][0] == "Ergon 2018 · SP 150"
    and sum(len(g[1]) for g in gr) == 53)
chk("recherche : nom de type → (clé, profil, variante) ; inconnu → None",
    K.chercher_type("SPZ 500-14-2Y · Ergon 2018")[1]["nom"] == "SPZ 500" and K.chercher_type("SP 999 · Ergon") is None)
chk("torons : SP 200-4-2X → 4 Ø 12,5 + 2 Ø 9,3", K.torons(cat, "SP 200-4-2X") == [(4, 12.5), (2, 9.3)])
chk("torons : SPN 200-6x (indice minuscule) → 6 Ø 9,3", K.torons(cat, "SPN 200-6x") == [(6, 9.3)])
chk("torons : SPZ 500-14-2Y → 14 Ø 12,5 + 2 Ø 15,2 ; SP 320-5 → 5 Ø 12,5",
    K.torons(cat, "SPZ 500-14-2Y") == [(14, 12.5), (2, 15.2)] and K.torons(cat, "SP 320-5") == [(5, 12.5)])
for mauvais in ("SP 200-4-2Q",):
    try:
        K.variante(cat, mauvais)
        chk("variante inconnue refusée", False)
    except KeyError:
        chk("variante inconnue refusée (KeyError)", True)
try:
    K.catalogue("rector_2020")
    chk("catalogue inconnu refusé", False)
except KeyError:
    chk("catalogue inconnu refusé (KeyError)", True)
# bibliothèque standard seulement (et le paquet beton)
sonde = ("import sys\nclass R:\n    def find_spec(self, n, p=None, t=None):\n"
         "        if n.split('.')[0] in ('numpy','scipy','fitz','streamlit','matplotlib','reportlab','pandas'):\n"
         "            raise ImportError(n)\nsys.meta_path.insert(0, R())\n"
         f"sys.path.insert(0, {RACINE!r})\nfrom beton.hourdis import catalogue as K\nprint(len(K.fiches_plancher()))")
r = subprocess.run([sys.executable, "-c", sonde], capture_output=True, text=True)
chk("lecteur : charge et résout sans numpy, scipy, fitz, streamlit", r.returncode == 0 and r.stdout.strip() == "53", r.stderr[-300:])

# ================================================================================================ 6 · mutations
def muter(nom, action, attendu):
    c = copy.deepcopy(cat)
    action(c)
    dd = defauts(c)
    chk(f"mutation « {nom} » : rougit ({attendu})", any(attendu in x for x in dd), dd[:3])


def _prof(c, nom):
    return next(p for p in c["profils"] if p["nom"] == nom)


muter("chaîne SP 265 imprimée (222 au bord)", lambda c: _prof(c, "SP 265")["cotes"].__setitem__(
    "chaine_horizontale", [154, 222, 222, 222, 222, 222]), "chaîne horizontale")
muter("chaîne verticale fausse", lambda c: _prof(c, "SP 320")["cotes"]["chaine_verticale"].__setitem__(1, 250), "chaîne verticale")
muter("une alvéole oubliée", lambda c: _prof(c, "SP 200")["cotes"]["alveoles"].__setitem__("nombre", 5), "alvéoles pour")
muter("poids en kN/m² faux", lambda c: _prof(c, "SP 200").__setitem__("poids_kN_m2", 2.88), "kN/m ÷")
muter("deux M_Rd échangés", lambda c: (lambda vs: (vs[0].__setitem__("M_Rd", 63), vs[1].__setitem__("M_Rd", 51)))(
    _prof(c, "SP 200")["variantes"]), "≤ la ligne précédente")
muter("M_Rd avec chape sous M_Rd", lambda c: _prof(c, "SP 150")["variantes"][0].__setitem__("M_Rd_chape", 40), "avec chape ≤")
muter("I_x faux de 10 %", lambda c: _prof(c, "SP 400").__setitem__("Ix", 4995e6), "Ix du dessin")
muter("aire fausse de 5 %", lambda c: _prof(c, "SP 150").__setitem__("A", 120e3), "A du dessin")
muter("dessin à la mauvaise échelle", lambda c: _prof(c, "SPZ 500")["dessin"].__setitem__(
    "contour", [[x * 1.01, y * 1.01] for x, y in _prof(c, "SPZ 500")["dessin"]["contour"]]), "dessin")
muter("une alvéole dessinée en moins", lambda c: _prof(c, "SP 150")["dessin"]["alveoles"].pop(), "alvéoles dessinées")
muter("désignation hors du profil", lambda c: _prof(c, "SP 200")["variantes"][0].__setitem__("nom", "SP 265-4X"), "hors du profil")
muter("désignation illisible", lambda c: _prof(c, "SP 200")["variantes"][0].__setitem__("nom", "SP 200-4Q"), "illisible")
muter("variante en double", lambda c: _prof(c, "SP 200")["variantes"].append(dict(_prof(c, "SP 200")["variantes"][-1],
                                                                                    M_Rd=999)), "en double")
muter("page perdue", lambda c: _prof(c, "SP 320").__setitem__("page", None), "page absente")
muter("transcription anonyme", lambda c: c["meta"].__setitem__("transcription", ""), "meta.transcription")
c = copy.deepcopy(cat)
_prof(c, "SP 200")["variantes"][4]["V_Rd"] = 95
chk("mutation « un V_Rd changé » : l'empreinte rougit", empreinte(c) != EMPREINTE_ERGON)
c = copy.deepcopy(cat)
c["meta"]["chape"]["epaisseur_min"] = 40
chk("mutation « chape de 40 mm » : l'empreinte rougit", empreinte(c) != EMPREINTE_ERGON)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
sys.exit(1 if KO else 0)
