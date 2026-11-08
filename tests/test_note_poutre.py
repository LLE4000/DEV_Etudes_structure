# -*- coding: utf-8 -*-
"""La NOTE PDF du module « Poutres béton armé » sur le site (api_web/note_poutre.py), vérifiée en Python ordinaire.

- PARITÉ : pour chaque cas, la planche que le générateur Streamlit calcule (modules/export_pdf._compute_section, sur les
  valeurs que le site traduit) rend les mêmes grandeurs que le moteur du site — h_min, A_s,req, A_s,min, A_s,max, A_s
  posé, τ, τ_adm, s_th, s_max, s_lim, A_st, état : la note dit ce que l'écran affiche ;
- les positions imposées (« pos »), les centres de gravité imposés, les cadres partiels et les épingles passent par les
  clés de l'écran Streamlit (surcharges, from/to) — retirer une traduction fait rougir la parité ;
- le site complète chaque planche : abscisses et étiquettes « L1 : 2Ø16 » des lits, latérales, dispositions
  constructives, état (jamais « Vérifié » sans effort) ;
- SANS ces clés, la note de Streamlit est inchangée (étiquettes « Lit 1 : 3 Ø16 », pas de cinquième vérification) ;
- la note sort en PDF (garde + une planche par section, toutes poutres) ; le routeur la rend en base 64 ; un fichier
  binaire mal formé est refusé ;
- la note se fait SANS fontTools ni Pillow (comme dans le navigateur), avec les seules polices du paquet.

    python3 tests/test_note_poutre.py
"""
import base64
import copy
import json
import os
import subprocess
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from api_web import commun as C  # noqa: E402
from api_web import note_poutre as NP  # noqa: E402
from api_web import poutre as P  # noqa: E402
from api_web import routeur as R  # noqa: E402
from beton.poutre import entree as EN  # noqa: E402
from beton.poutre import moteur as MO  # noqa: E402
from modules import export_pdf as EP  # noqa: E402
from ndc_pdf import data as ND  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:260]}"))


def proche(a, b, tol=1e-9):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def cas(*saisies, base=None):
    e = copy.deepcopy(base) if base else P.nouvel_etat()
    for chemin, valeur in saisies:
        msg = P.modifier(e, chemin, valeur)
        assert msg is None, (chemin, valeur, msg)
    return e


S0 = ["sections", 0]
CAS = {
    "poutre neuve (sans effort)": cas(),
    "travée 30 × 60, 3Ø20, M = 120, V = 40": cas((["b"], 30), (["h"], 60), (S0 + ["M_inf"], 120), (S0 + ["V"], 40),
                                               (S0 + ["inf", 0, "n"], 3), (S0 + ["inf", 0, "d"], 20), (S0 + ["pas"], 20)),
    "appui M_sup = 90, deux lits en haut": cas((S0 + ["M_sup"], 90), (S0 + ["V"], 60), (S0 + ["sup", "+"], 1), (S0 + ["pas"], 15)),
    "trois lits inférieurs de Ø différents": cas((["h"], 70), (S0 + ["M_inf"], 260), (S0 + ["inf", 0, "n"], 4), (S0 + ["inf", 0, "d"], 25),
                                                 (S0 + ["inf", "+"], 1), (S0 + ["inf", 1, "d"], 20), (S0 + ["inf", "+"], 1), (S0 + ["inf", 2, "d"], 16)),
    "positions imposées (lit 1 et entraxe du lit 2)": cas((S0 + ["M_inf"], 80), (S0 + ["inf", "+"], 1), (S0 + ["inf", 0, "pos"], "5,5"),
                                                         (S0 + ["inf", 1, "pos"], "4")),
    "centre de gravité imposé": cas((S0 + ["M_inf"], 70), (S0 + ["yG_inf"], "6,5")),
    "deux cadres (intérieur) et une épingle": cas((["b"], 35), (S0 + ["V"], 160), (S0 + ["M_inf"], 150), (S0 + ["inf", 0, "n"], 5),
                                                  (S0 + ["etriers", "+"], "etrier"), (S0 + ["etriers", "+"], "epingle"), (S0 + ["pas"], 12)),
    "trois cadres côte à côte": cas((["b"], 50), (["h"], 80), (S0 + ["V"], 300), (S0 + ["inf", 0, "n"], 6), (S0 + ["etriers", "+"], "etrier"),
                                    (S0 + ["etriers", "+"], "etrier"), (S0 + ["etriers", 1, "de"], 1), (S0 + ["etriers", 1, "a"], 3),
                                    (S0 + ["etriers", 2, "de"], 4), (S0 + ["etriers", 2, "a"], 6)),
    "barres inclinées (τ en bande IV)": cas((S0 + ["V"], 250), (S0 + ["etriers", 0, "d"], 12), (S0 + ["pas"], 8)),
    "surchargée (M = 400)": cas((S0 + ["M_inf"], 400), (S0 + ["V"], 400)),
    "B400, γs 1,15, C20/25": cas((["fyk"], "400"), (["gamma_s"], "1,15"), (["beton"], "C20/25"), (S0 + ["M_inf"], 60), (S0 + ["V"], 30)),
    "enrobage nul (lu 3 cm) et jeux": cas((["enrobage"], 0), (["jeu_premier_lit"], "0,5"), (["jeu_entre_lits"], 2), (S0 + ["M_inf"], 90),
                                          (S0 + ["inf", "+"], 1)),
    "deux sections, entraxe horizontal imposé": cas((S0 + ["M_inf"], 110), (S0 + ["inf", 0, "rep"], "entraxe"), (S0 + ["inf", 0, "sh"], 8),
                                                   (["sections", "+"], 1), (["sections", 1, "M_sup"], 75), (["sections", 1, "V"], 50)),
    "latérales imposées (h = 90)": cas((["h"], 90), (S0 + ["M_inf"], 200), (S0 + ["laterales", "n"], 3), (S0 + ["laterales", "d"], 12)),
}
CAS["deux poutres (γs différents)"] = cas((["gamma_s"], "1,15"), base=cas((S0 + ["M_inf"], 50), base=CAS["travée 30 × 60, 3Ø20, M = 120, V = 40"]))
P.gerer_elements(CAS["deux poutres (γs différents)"], "ajouter")
P.modifier(CAS["deux poutres (γs différents)"], S0 + ["M_inf"], 95)

BETON = NP._beton()

# 1. la parité : le calcul de la planche Streamlit = le moteur du site
ecarts = []
n_sections = 0
for nom, e in CAS.items():
    for bi, p in enumerate(e["poutres"]):
        p = EN.normaliser(p)
        calc = MO.verifier(p)
        beams, vals = NP.valeurs(p, bi + 1)
        for sj, rs in enumerate(calc["sections"]):
            n_sections += 1
            Rp = EP._compute_section(vals, BETON, bi + 1, sj + 1)
            g = rs["grandeurs"]
            paires = [("h_min", Rp["h_min_poutre"], g["h_min"]), ("hu_min", Rp["hmin"], g["hu_min"]),
                      ("d_inf", Rp["di"], g["d_inf"]), ("d_sup", Rp["ds"], g["d_sup"]), ("d_cis", Rp["dsh"], g["d_cisaillement"]),
                      ("As_req_inf", Rp["As_req_inf"], g["As_req_inf"]), ("As_req_sup", Rp["As_req_sup"], g["As_req_sup"]),
                      ("As_min_inf", Rp["As_min_inf"], g["As_min_inf"]), ("As_min_sup", Rp["As_min_sup"], g["As_min_sup"]),
                      ("As_max", Rp["As_max"], g["As_max"]), ("As_inf", Rp["As_inf"], g["As_inf"]), ("As_sup", Rp["As_sup"], g["As_sup"]),
                      ("fyd", Rp["fyd"], g["fyd"])]
            Sh = Rp["shear"]
            if g["V"] > 0:
                paires += [("tau", Sh["tau"], g["tau"]), ("tau_lim", Sh["tau_lim"], g["tau_lim"]), ("Ast", Sh["Ast"], g["Ast"]),
                           ("s_th", Sh["pas_th"], g["s_th"]), ("s_max", Sh["s_max"], g["s_max"]), ("s_lim", Sh["pas_lim"], g["s_lim"])]
            elif Sh is not None:
                ecarts.append((nom, rs["nom"], "shear sans V"))
            for cle, a, b in paires:
                if not proche(a, b):
                    ecarts.append((nom, bi, rs["nom"], cle, a, b))
            if Rp["etat_global"] != rs["statut"]:
                ecarts.append((nom, bi, rs["nom"], "etat", Rp["etat_global"], rs["statut"]))
chk(f"parité : {len(CAS)} cas, {n_sections} sections — la planche Streamlit rend les grandeurs et l'état du moteur du site",
    not ecarts and n_sections >= 17, ecarts[:4])

# 2. les traductions sont nécessaires (une traduction retirée rougit la parité)
e = CAS["positions imposées (lit 1 et entraxe du lit 2)"]
p = EN.normaliser(e["poutres"][0])
_, vals = NP.valeurs(p)
sans = {k: v for k, v in vals.items() if "override" not in k}
d_avec = EP._compute_section(vals, BETON, 1, 1)["di"]
d_sans = EP._compute_section(sans, BETON, 1, 1)["di"]
chk("les positions imposées passent par les surcharges de l'écran (sans elles, d change)",
    proche(d_avec, MO.verifier(p)["sections"][0]["grandeurs"]["d_inf"]) and not proche(d_avec, d_sans), (d_avec, d_sans))
e = CAS["deux cadres (intérieur) et une épingle"]
p = EN.normaliser(e["poutres"][0])
_, vals = NP.valeurs(p)
lignes = EP._shear_lines(vals, 1, 1)[2]
chk("le cadre intérieur et l'épingle gardent leurs barres (from/to du module)",
    [(g["brins"], g["from"], g["to"]) for g in lignes] == [(2, None, None), (2, 2, 4), (1, 3, 3)], [(g["brins"], g["from"], g["to"]) for g in lignes])

# 3. ce que le site ajoute à chaque planche
pl = NP.resultats(CAS["trois lits inférieurs de Ø différents"])
lits = pl[0]["R"]["geo_inf"]["lits"]
chk("les lits portent « L1 : 4Ø25 », « L2 : 4Ø20 », « L3 : 4Ø16 » et leurs abscisses (cm)",
    [l_["label"] for l_ in lits] == ["L1 : 4Ø25", "L2 : 4Ø20", "L3 : 4Ø16"] and all(len(l_["x"]) == l_["n"] for l_ in lits), [l_.get("label") for l_ in lits])
sec = ND.construire_sections(pl)[0]
chk("… la coupe de la planche reprend étiquettes et abscisses (mm)",
    [lab for lab, _ in sec["coupe"]["labs_inf"]] == ["L1 : 4Ø25", "L2 : 4Ø20", "L3 : 4Ø16"]
    and sec["coupe"]["lits_inf"][0]["xs"] == [x * 10.0 for x in lits[0]["x"]], sec["coupe"]["labs_inf"])
chk("… et une cinquième vérification : les dispositions constructives", [v["titre"] for v in sec["verifs"]][-1] == "Dispositions constructives",
    [v["titre"] for v in sec["verifs"]])
pl = NP.resultats(CAS["poutre neuve (sans effort)"])
chk("une section sans effort n'est jamais « Vérifié » : « Données incomplètes »", pl[0]["etat_label"] == NP.ETAT_INCOMPLET
    and ND.construire_sections(pl)[0]["etat"] == NP.ETAT_INCOMPLET, pl[0].get("etat_label"))
pl = NP.resultats(cas((S0 + ["M_inf"], 40), (S0 + ["V"], 20), (S0 + ["pas"], 20), (S0 + ["inf", "+"], 1)))
chk("deux lits Ø16 au jeu du bureau : la planche est « À surveiller » (14 mm libres < 25 mm)", pl[0]["etat_label"] == "À surveiller"
    and any(v["etat"] == "att" and "14 mm" in v["texte"] for v in ND.construire_sections(pl)[0]["verifs"][-1]["verdicts"]), pl[0]["etat_label"])
pl = NP.resultats(CAS["latérales imposées (h = 90)"])
chk("les latérales imposées (3 Ø12 par face) remplacent la règle automatique", pl[0]["peau"]["n"] == 3 and pl[0]["peau"]["d"] == 12
    and "2×3 Ø12" in ND.construire_sections(pl)[0]["coupe"]["peau"]["label"], pl[0]["peau"])
pl = NP.resultats(CAS["deux poutres (γs différents)"])
chk("deux poutres : une planche par section de chaque poutre, chacune avec SON γs",
    [x["poutre"] for x in pl] == ["V1", "V2"] and proche(pl[0]["R"]["gamma_s"], 1.15) and proche(pl[1]["R"]["gamma_s"], 1.5), [x["poutre"] for x in pl])

# 4. sans les clés du site, la note de Streamlit est inchangée
beams = [{"id": 1, "nom": "Poutre 1", "sections": [{"id": 1, "nom": "A"}]}]
vals = {"b1_b": 30, "b1_h": 60, "b1_sec1_M_inf": 120.0, "b1_sec1_V": 80.0, "b1_sec1_n_as_inf": 3, "b1_sec1_ø_as_inf": 16}
res_st = EP._collecter_resultats(beams, vals, BETON)
st = ND.construire_sections(res_st)[0]
chk("sans les clés du site : « Lit 1 : 3 Ø16 », pas d'abscisses imposées, quatre vérifications, l'état du module",
    st["coupe"]["labs_inf"][0][0] == "Lit 1 : 3 Ø16" and "xs" not in st["coupe"]["lits_inf"][0] and len(st["verifs"]) == 4
    and st["etat"] == ND.ETAT_LABELS[res_st[0]["R"]["etat_global"]], (st["coupe"]["labs_inf"], len(st["verifs"]), st["etat"]))

# 5. le PDF : garde + une planche par section ; le routeur le rend en base 64
e = CAS["deux sections, entraxe horizontal imposé"]
octets = NP.pdf(e, {"nom_projet": "Essai"})
chk("la note est un PDF : la garde et une planche par section (3 pages)", octets[:5] == b"%PDF-" and octets.count(b"/Type /Page\n") + octets.count(b"/Type /Page ") >= 3,
    octets[:8])
r = json.loads(R.traiter(json.dumps({"module": "poutre", "action": "telecharger", "etat": e, "quoi": "note_pdf",
                                     "infos": {"nom": "Note <b>", "chantier": "Chantier", "indice": "B"}})))
f = r.get("fichier") or {}
chk("le routeur rend la note en base 64, nommée d'après la note (nom de fichier sûr)",
    r["ok"] and f.get("type") == "application/pdf" and base64.b64decode(f["base64"])[:5] == b"%PDF-" and f["nom"] == "Note de calcul - Note -b.pdf",
    (r.get("erreur"), f.get("nom")))
r = json.loads(R.traiter(json.dumps({"module": "poutre", "action": "telecharger", "etat": e, "quoi": "autre"})))
chk("un autre fichier que la note est refusé avec un message", not r["ok"] and "inconnu" in r["erreur"], r)
chk("refus_fichier : un PDF au base 64 illisible est refusé", C.refus_fichier({"nom": "a.pdf", "type": "application/pdf", "base64": "@@@"}) is not None)
chk("refus_fichier : un PDF sans contenu est refusé", C.refus_fichier({"nom": "a.pdf", "type": "application/pdf"}) is not None)
chk("refus_fichier : un PDF valide passe", C.refus_fichier({"nom": "a.pdf", "type": "application/pdf", "base64": base64.b64encode(b"%PDF-1.4").decode()}) is None)
r = json.loads(R.traiter(json.dumps({"module": "garde_corps", "action": "telecharger", "etat": None, "quoi": "x", "infos": {"nom": "y"}})))
chk("un module qui n'attend pas « infos » ne les reçoit pas (pas d'erreur de signature)", "infos" not in str(r.get("erreur", "")), r)

# 6. comme dans le navigateur : sans fontTools ni Pillow, avec les seules polices du paquet
SONDE = r"""
import sys, json, base64, os
sys.path.insert(0, {stub!r}); sys.path.insert(1, {racine!r})
class Refus:
    def find_spec(self, nom, chemin=None, cible=None):
        if nom.split(".")[0] in ("fontTools", "_rl_accel"):
            raise ImportError("refusé : " + nom)
sys.meta_path.insert(0, Refus())
from api_web import poutre as P, note_poutre as NP
e = P.nouvel_etat(); P.modifier(e, ["sections", 0, "M_inf"], 80)
o = NP.pdf(e, {{}})
import PIL
print(json.dumps({{"pdf": o[:5].decode(), "pil": PIL.__file__, "fonttools": "fontTools" in sys.modules}}))
"""
polices = os.path.join(RACINE, "ndc_pdf", "fonts")
sys.path.insert(0, os.path.join(RACINE, "site", "web"))
import construire as K  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
tmp = tempfile.mkdtemp()
for f in K.NOTE_POLICES:
    shutil.copy(os.path.join(polices, f), tmp)
p_ = subprocess.run([sys.executable, "-c", SONDE.format(stub=os.path.join(RACINE, "site", "note_pdf"), racine=RACINE)],
                    capture_output=True, text=True, env={**os.environ, "NDC_FONT_DIR": tmp})
shutil.rmtree(tmp, ignore_errors=True)
try:
    out = json.loads(p_.stdout.strip().splitlines()[-1])
except Exception:  # noqa: BLE001
    out = {"erreur": p_.stderr[-500:]}
chk("sans fontTools, avec le Pillow factice et les seules polices du paquet : la note se fait",
    out.get("pdf") == "%PDF-" and out.get("pil", "").startswith(os.path.join(RACINE, "site", "note_pdf")) and out.get("fonttools") is False, out)

# 7. un enregistrement de l'écran Streamlit (poutre_ba.json) s'ouvre sur le site : l'aller-retour garde le calcul
from api_web import poutre_streamlit as PS  # noqa: E402
ecarts = []
for nom, e in CAS.items():
    if len(e["poutres"]) > 1:
        continue
    p = EN.normaliser(e["poutres"][0])
    beams, vals = PS.valeurs(p)
    vals = {**vals, **{k + "_raw": str(v) for k, v in vals.items() if isinstance(v, float)}, "version_app": "1.2.41"}
    e2 = PS.depuis_streamlit({"version": "2.39", "beams": beams, "values": vals})
    a, b = MO.verifier(p), MO.verifier(e2["poutres"][0])
    for sa, sb in zip(a["sections"], b["sections"]):
        for k, x in sa["grandeurs"].items():
            y = sb["grandeurs"][k]
            if isinstance(x, float) and not proche(x, y) or (not isinstance(x, float) and x != y):
                ecarts.append((nom, sa["nom"], k, x, y))
        if sa["statut"] != sb["statut"] or len(a["sections"]) != len(b["sections"]):
            ecarts.append((nom, sa["nom"], "statut"))
chk("aller-retour par un fichier Streamlit (poutre_ba.json) : chaque grandeur du calcul est retrouvée", not ecarts, ecarts[:3])
r = json.loads(R.traiter(json.dumps({"module": "poutre", "action": "calculer",
                                     "etat": {"streamlit": {"beams": [{"id": 1, "nom": "P1", "sections": [{"id": 1, "nom": "A"}, {"id": 2, "nom": "B"}]}],
                                                            "values": {"b1_b": 25, "b1_h": 50, "b1_sec1_M_inf": 90, "b1_sec2_M_sup": 60,
                                                                       "b1_sec1_shear_n_lines": 1, "b1_sec1_shear_line0_type": "Étriers (3 brins)",
                                                                       "b1_sec1_shear_line0_d": 8}}}})))
pt = r["etat"]["poutres"][0] if r.get("ok") else {}
chk("le routeur traduit un fichier Streamlit au premier calcul (deux sections ; un cadre à 3 brins = étrier + épingle)",
    r.get("ok") and [s_["nom"] for s_ in pt["sections"]] == ["A", "B"] and pt["b"] == 25.0
    and [x["type"] for x in pt["sections"][0]["etriers"]] == ["etrier", "epingle"], r.get("erreur") or pt)
for mauvais in ({"beams": "x", "values": {}}, {"beams": [], "values": {"autre": 1}}, {"beams": [{"id": "z"}], "values": {"b1_b": 20}}):
    try:
        PS.depuis_streamlit(mauvais)
        chk(f"un fichier qui n'en est pas un est refusé ({str(mauvais)[:40]})", False)
    except ValueError as err:
        chk(f"un fichier qui n'en est pas un est refusé avec un message ({str(mauvais)[:40]})", "Streamlit" in str(err) or "aucune poutre" in str(err), err)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)
