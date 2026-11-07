# -*- coding: utf-8 -*-
"""Le module « Taux d'armature » du site (api_web/taux.py, api_web/dessin_taux.py) : ce que l'écran affiche est ce que
le moteur beton/taux calcule — et ce que l'écran Streamlit écrirait (mêmes précisions) ; les saisies, les refus, les
effets de bord de l'écran Streamlit, le sélecteur de diamètre, les dessins à l'échelle, l'échappement, les éléments de
la note. Chaque contrôle rougit si l'on retire ce qu'il protège.

    python3.12 tests/test_api_web_taux.py
"""
import ast
import copy
import json
import math
import os
import re
import subprocess
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from api_web import commun as C  # noqa: E402
from api_web import dessin_taux as DT  # noqa: E402
from api_web import routeur as R  # noqa: E402
from api_web import taux as T  # noqa: E402
from beton.taux import entree as EN  # noqa: E402
from beton.taux import moteur as MO  # noqa: E402
from beton.taux import saisie as SA  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:400]}]" if info and not cond else ""))


def req(**k):
    return json.loads(R.traiter(json.dumps(k)))


def etat_de(*poutres, actif=0):
    return T.normaliser({"poutres": [copy.deepcopy(p) for p in poutres], "actif": actif})


def st(x, nd):
    """Un nombre écrit comme l'écran Streamlit l'écrit (f"{x:.nd f}"), virgule décimale."""
    return f"{x:.{nd}f}".replace(".", ",")


def sans_espaces(s):
    return s.replace(" ", "").replace("\xa0", "").replace(" ", "")


EX = copy.deepcopy(T._EXEMPLE)
P2 = {"nom": "P2", "forme": "rectangulaire", "L": 7.35, "b": 35.0, "h": 70.0, "enrobage": 2.5, "majoration": 7.5, "acier": "B500C",
      "familles": [{"role": "principal", "position": "ame_inf", "n": 4, "d": 25}, {"role": "principal", "position": "ame_inf", "n": 3, "d": 20},
                   {"role": "renfort", "position": "ame_sup", "n": 3, "d": 16, "de": 0, "a": 25},
                   {"role": "lateral", "position": "lat_d", "n": 1, "d": 12, "retours": 1, "h_retour": 12.5}],
      "types_etriers": [{"nom": "E1", "d": 8.0}, {"nom": "E2", "d": 7.5}],
      "zones": [{"nom": "Z1", "de": 0, "a": 20, "s": 10, "type": "E2"}, {"nom": "Z2", "de": 20, "a": 80, "s": 20, "type": "E1"},
                {"nom": "Z3", "de": 60, "a": 100, "s": 12.5, "type": "E1"}]}
TRONC = {"nom": "TR", "L": 1.0, "familles": [{"role": "principal"}], "types_etriers": [{"nom": "E1", "d": 8.0}],
         "zones": [{"nom": "Z1", "de": 0, "a": 30, "s": 10, "type": "E1"}]}
DEUX = {"nom": "DT", "forme": "deux_talons", "talon_gauche": {"b": 20.0, "h": 25.0}, "talon_droit": {"b": 30.0, "h": 35.0},
        "familles": [{"role": "principal", "n": 3}], "types_etriers": [{"nom": "E1", "d": 8.0}], "zones": [{"nom": "Z1", "de": 0, "a": 100, "s": 20, "type": "E1"}]}
VIDE = EN.nouvelle_poutre()
CAS = [("exemple (talon gauche, latérales à retours, deux zones)", EX), ("P2 (deux types, Ø7,5, zones qui se recouvrent, B500C, 7,5 %)", P2),
       ("troncature des étriers", TRONC), ("deux talons (ignorés)", DEUX), ("poutre neuve, sans armature", VIDE)]

# ------------------------------------------------------------------ 1. l'écran = le moteur = ce que Streamlit écrirait
print("=== 1. Le résultat, les fenêtres et le tableau : les chiffres du moteur, écrits comme l'écran Streamlit ===")
for nom, p in CAS:
    etat = etat_de(p)
    out = T.resultat(etat)
    r = MO.calculer(etat["poutres"][0])
    q = r["quantites"]
    res = out["resultat"]
    chk(f"{nom} : le verdict est le taux majoré du moteur ({st(q['kg_m3_majore'], 1)} kg/m³, comme la métrique de l'écran)",
        f'{C.fr(q["kg_m3_majore"], 1)}<small>kg/m³</small>' in res and C.fr(q["kg_m3_majore"], 1) == st(q["kg_m3_majore"], 1).replace(",", ",")
        and sans_espaces(C.fr(q["kg_m3_majore"], 1)) == st(q["kg_m3_majore"], 1))
    synth = re.search(r'<div class="tx-synth">(.*?)</div></div>', res, re.S).group(1)
    attendus = [st(q["Vc"], 3) + " m³", st(q["masse_acier_majoree"], 1) + " kg", st(q["masse_beton"], 0) + " kg", st(q["kg_m3"], 1) + " kg/m³"]
    chk(f"{nom} : la synthèse porte Vc, la masse majorée, le poids du béton (précisions de l'écran) et le taux sans majoration",
        all(a in sans_espaces(synth).replace("m³", " m³").replace("kg", " kg").replace(" kg/ m³", " kg/m³") or sans_espaces(a) in sans_espaces(synth) for a in attendus),
        (attendus, synth[:300]))
    lignes = re.findall(r'<div class="el"[^>]*data-calcul="([^"]+)"[^>]*><div class="l1"><b>([^<]*)</b>.*?<span>(.*?)</span>.*?<span class="pct c">([^<]*)</span>', res, re.S)
    att = [(f"f{f['index']}", C.esc(f"{f['code']} · {T.ROLES_COURTS[f['role']]}"), f"{C.esc(f['libelle'])} · {C.fr(f['masse_majoree'], 1)} kg",
            f"{T._part(f['part'])}{C.NBSP}%") for f in r["familles"]]
    att += [(f"z{e['index']}", C.esc(f"{e['zone']} · étriers {e['type']}"), f"{e['n']} × Ø{C.nb(e['d'])} · {C.fr(e['masse_majoree'], 1)} kg",
             f"{T._part(e['part'])}{C.NBSP}%") for e in r["etriers"]]
    chk(f"{nom} : une ligne par famille et par zone calculée (nom, barres, masse majorée, part de l'acier), dans l'ordre",
        [tuple(x) for x in lignes] == att, (lignes[:2], att[:2]))
    lues = [int(x[3].replace(C.NBSP, "").rstrip("%")) for x in lignes]
    vraies = [f["part"] * 100 for f in r["familles"]] + [e["part"] * 100 for e in r["etriers"]]
    chk(f"{nom} : chaque part affichée est la part du moteur arrondie au % (à 0,5 près ; 99 % pour une part presque entière)",
        all(abs(a - b) <= 0.5 + 1e-9 or (a == 99 and b < 100) for a, b in zip(lues, vraies)), list(zip(lues, vraies)))
    parts = [f["part"] for f in r["familles"]] + [e["part"] for e in r["etriers"]]
    chk(f"{nom} : les parts du moteur somment à 1 (ou à 0 sans acier)", abs(sum(parts) - (1.0 if parts else 0.0)) < 1e-12 or not parts)
    # le tableau récapitulatif : les colonnes de l'écran Streamlit, au même format
    tab = next(v["svg"] for v in out["vues"] if v["cle"] == "tableau")
    rows = re.findall(r"<tr><td class=\"co\">.*?</td><th>([^<]*)</th><td class=\"po\">([^<]*)</td><td>([^<]*)</td><td>([^<]*)</td>"
                      r"<td>([^<]*)</td><td>([^<]*)</td><td>([^<]*)</td><td class=\"kg\">([^<]*)</td></tr>", tab)
    ecran = [(f["code"], f["libelle_position"], str(f["n_eff"]), str(int(f["d"])), st(f["L_active"], 2), st(f["L_tot"], 2), st(f["m_lin"], 3),
              st(f["masse_majoree"], 2)) for f in r["familles"]]
    ecran += [(e["type"], f"Zone {e['zone']}", str(e["n"]), C.nb(e["d"]), st(e["L_un"], 2), st(e["L_tot"], 2), st(e["m_lin"], 3), st(e["masse_majoree"], 2))
              for e in r["etriers"]]
    chk(f"{nom} : le tableau récapitulatif = celui de l'écran Streamlit, ligne par ligne ({len(ecran)} lignes)",
        [tuple(sans_espaces(c) for c in x) for x in rows] == [tuple(sans_espaces(c) for c in x) for x in ecran], (rows[:2], ecran[:2]))
    if not ecran:
        chk(f"{nom} : sans armature, le tableau le dit (comme « Ajoute des armatures… »)", "Aucune armature" in tab)
    # les avertissements du moteur, tous montrés
    chk(f"{nom} : chaque avertissement du moteur est affiché ({len(r['avertissements'])})", all(C.esc(a["message"]) in res for a in r["avertissements"]))
    # les fenêtres de calcul : une pour la poutre, une par famille, une par zone ; les résultats des formules
    vol = out["volets"]
    ids = re.findall(r'id="ck-([^"]+)"', vol)
    chk(f"{nom} : une fenêtre par ligne (q, f…, z…)", ids == ["q"] + [f"f{f['index']}" for f in r["familles"]] + [f"z{e['index']}" for e in r["etriers"]], ids)
    fen = {i: vol[vol.index(f'id="ck-{i}"'):] for i in ids}
    fautes = []
    for f in r["familles"]:
        w = fen[f"f{f['index']}"].split('class="flyout calcul"')[0] if False else fen[f"f{f['index']}"]
        for t in (f"{C.fr(f['L_tot'], 3)} m", f"{C.fr(f['m_lin'], 3)} kg/m", f"{C.fr(f['masse_majoree'], 2)} kg", f"{C.fr(f['apport'], 1)} kg/m³"):
            if t not in w:
                fautes.append((f["code"], t))
    for e in r["etriers"]:
        w = fen[f"z{e['index']}"]
        for t in (f"{e['n']} étriers", f"{C.fr(e['L_un'], 3)} m", f"{C.fr(e['L_tot'], 3)} m", f"{C.fr(e['masse_majoree'], 2)} kg"):
            if t not in w:
                fautes.append((e["zone"], t))
    w = fen["q"]
    for t in (f"{C.fr(q['Vc'], 3)} m³", f"{C.fr(q['masse_acier_majoree'], 2)} kg", f"{C.fr(q['kg_m3_majore'], 1)} kg/m³", f"{C.fr(q['kg_m3'], 1)} kg/m³"):
        if t not in w:
            fautes.append(("q", t))
    chk(f"{nom} : les fenêtres de calcul portent les résultats du moteur (longueurs, kg/m, masses, apports, taux)", not fautes, fautes[:4])
    chk(f"{nom} : les en-têtes de la fenêtre disent un quantitatif (armature, masse, apport au taux, part)",
        all(x in w for x in ("<th>Armature</th>", ">Masse</th>", ">Apport au taux</th>", ">Part</th>")) and "<th>Vérification</th>" not in vol)
    # « apport » : masse / Vc, et leur somme est le taux
    if r["familles"] or r["etriers"]:
        s_ap = sum(x["apport"] for x in r["familles"] + r["etriers"])
        chk(f"{nom} : la somme des apports (masse / Vc) est le taux majoré", abs(s_ap - q["kg_m3_majore"]) < 1e-9 * max(1, q["kg_m3_majore"]))

# ------------------------------------------------------------------ 2. les formules substituées : recalculées à partir de leurs valeurs
print("\n=== 2. Les formules « symbole = expression = valeurs = résultat » se recalculent ===")
etat = etat_de(P2)
r = MO.calculer(etat["poutres"][0])
f = r["familles"][3]
fx = {x["parts"][0]: x for x in T._fx_famille(f, 7.5, r["quantites"]["Vc"], 7.35)}
v = fx["L_tot"]["parts"][2]
nums = [float(x.replace(",", ".").replace(" ", "").replace(" ", "")) for x in re.findall(r"[\d  ]*\d(?:,\d+)?", v)]
chk("L_tot d'une latérale à un retour : n_eff × L_active + r × h_r / 100 × n_eff, recalculée sur les valeurs écrites",
    len(nums) == 6 and abs(nums[0] * nums[1] + nums[2] * nums[3] / nums[4] * nums[5] - f["L_tot"]) < 1e-9, (v, f["L_tot"]))
z = r["etriers"][0]
nz = T._fx_zone(z, r["geometrie"], 7.5, r["quantites"]["Vc"], 7.35)
chk("n d'une zone : ⌊L_zone / s⌋ + 1 écrit avec son quotient", f"⌊{T._v(z['L_zone'])} / {T._v(z['s_m'])}⌋ + 1" in nz[1]["parts"][2] and nz[1]["res"] == f"{z['n']} étriers")
tr = MO.calculer(etat_de(TRONC)["poutres"][0])
chk("troncature : le quotient 2,9999999999999996 est écrit en entier (il « tombe juste » à l'écran, pas en machine)",
    "2,9999999999999996" in T._fx_zone(tr["etriers"][0], tr["geometrie"], 5.0, tr["quantites"]["Vc"], 1.0)[1]["parts"][2]
    and tr["etriers"][0]["n"] == 3)
chk("troncature : la fenêtre de la zone le dit", "un étrier de moins" in T.resultat(etat_de(TRONC))["volets"])
fq = {x["parts"][0]: x for x in T._fx_poutre(r)}
chk("V_c = A_c · L / 10 000 écrit avec ses milliers séparés (« 2 450 × 7,35 / 10 000 »)",
    fq["V_c"]["parts"][2] == f"2{C.NBSP}450 × 7,35 / 10 000" and fq["V_c"]["res"] == f"{C.fr(r['quantites']['Vc'], 3)} m³", fq["V_c"])

# ------------------------------------------------------------------ 3. les dessins, à l'échelle
print("\n=== 3. Les dessins : coupe et élévation à l'échelle, positions du moteur ===")
for nom, p in CAS[:4]:
    e = etat_de(p)
    r = MO.calculer(e["poutres"][0])
    svg = DT.coupe("t", r)
    H = r["geometrie"]["h"] * 10
    cercles = re.findall(r'<circle cx="([-\d.]+)" cy="([-\d.]+)" r="([\d.]+)" fill="(#[0-9a-f]{6})" stroke="(#[0-9A-Fa-f]{6})" stroke-width="[\d.]+"/>', svg)
    barres = [(DT._f(x * 10), DT._f(H - y * 10), DT._f(f["d"] / 2), f["couleur"]) for f in r["familles"] for x, y in f["barres"]]
    chk(f"{nom} : chaque barre de la coupe est au centre que l'écran Streamlit dessine (×10 : mm), à son Ø et dans sa couleur ({len(barres)})",
        [c[:4] for c in cercles[:len(barres)]] == barres, (cercles[:2], barres[:2]))
    poly = re.search(r'<polygon points="([^"]+)"', svg).group(1)
    chk(f"{nom} : le contour du béton est le polygone du moteur, à l'échelle", poly == " ".join(f"{DT._f(x * 10)},{DT._f(H - y * 10)}" for x, y in r["geometrie"]["polygone"]))
    el = DT.elevation("t", r)
    n_et = len(re.findall(rf'stroke="{DT.ETRIER}"', el))
    chk(f"{nom} : l'élévation dessine autant d'étriers que le module en compte ({sum(x['n'] for x in r['etriers'])})", n_et == sum(x["n"] for x in r["etriers"]), n_et)
    for f in r["familles"]:
        if f["L_active"] > 0:
            x1, x2 = DT._f(r["geometrie"]["L"] * 1000 * f["de"] / 100), DT._f(r["geometrie"]["L"] * 1000 * f["a"] / 100)
            if f'x1="{x1}"' not in el or f'x2="{x2}"' not in el:
                chk(f"{nom} : {f['code']} de « De » à « À » dans l'élévation", False, (x1, x2))
r = MO.calculer(etat_de(DEUX)["poutres"][0])
chk("deux talons : la coupe montre les talons non comptés, en fantôme, et le dit", "talons non comptés" in DT.coupe("t", r) and DT.FANTOME in DT.coupe("t", r))
chk("élancement : hauteurs ×1 jusqu'à L/h = 6, puis amplifiées à un élancement dessiné ≤ 6 (et écrit sur le dessin)",
    DT.amplification(3.6, 60.0) == 1.0 and DT.amplification(6.0, 60.0) == 2 and DT.amplification(30.0, 60.0) == 10
    and "hauteurs × 10" in DT.elevation("t", MO.calculer(dict(EN.nouvelle_poutre(), L=30.0)))
    and "hauteurs × 2" in T.resultat(etat_de(EN.nouvelle_poutre()))["vues"][1]["legende"])


def chevauchements(svg):
    """Les textes horizontaux d'un dessin dont les boîtes (estimées : 0,55 × taille par caractère, de −0,8 à +0,2 taille
    autour de la ligne de base) se recouvrent : deux écritures l'une sur l'autre."""
    boites = []
    for m in re.finditer(r'<text x="([-\d.]+)" y="([-\d.]+)" text-anchor="(\w+)" font-size="([\d.]+)"[^>]*>([^<]*)</text>', svg):
        if "rotate(" in m.group(0):
            continue
        x, y, ancre, t, s = float(m.group(1)), float(m.group(2)), m.group(3), float(m.group(4)), m.group(5)
        w = 0.55 * t * len(re.sub(r"&\w+;", "x", s))
        x0 = x - w / 2 if ancre == "middle" else (x - w if ancre == "end" else x)
        boites.append((x0, y - 0.8 * t, x0 + w, y + 0.2 * t, s))
    return [(a[4], b[4]) for i, a in enumerate(boites) for b in boites[i + 1:]
            if min(a[2], b[2]) - max(a[0], b[0]) > 0 and min(a[3], b[3]) - max(a[1], b[1]) > 0]


for nom, p in CAS + [("deux talons, cotes et note du fantôme", dict(DEUX, familles=[{"role": "principal", "n": 3}, {"role": "lateral", "position": "lat_2"}]))]:
    r = MO.calculer(etat_de(p)["poutres"][0])
    for vue, svg in (("coupe", DT.coupe("t", r)), ("élévation", DT.elevation("t", r))):
        ch = chevauchements(svg)
        chk(f"{nom} : {vue} — aucune écriture sur une autre (la note des talons fantômes recouvrait la cote de l'âme)", not ch, ch[:3])
out = T.resultat(etat_de(EX))
chk("trois vues pour une poutre (coupe, élévation, tableau) ; la première est un dessin SVG", [v["cle"] for v in out["vues"]] == ["coupe", "elevation", "tableau"]
    and out["vues"][0]["svg"].startswith("<svg") and out["vues"][2]["svg"].startswith("<div"))

# ------------------------------------------------------------------ 4. la saisie : refus, effets de bord de l'écran Streamlit
print("\n=== 4. La saisie : refus avec un message, effets de bord de l'écran Streamlit ===")


def mod(e, chemin, valeur):
    e2 = copy.deepcopy(e)
    m = T.modifier(e2, chemin, valeur)
    return e2, m


e0 = etat_de(EX)
REFUS = [(["b"], "4"), (["h"], "301"), (["enrobage"], "0,5"), (["L"], "61"), (["L"], "abc"), (["majoration"], "51"), (["talon_gauche", "b"], "0"),
         (["familles", "0", "n"], "21"), (["familles", "0", "n"], "2,5"), (["familles", "0", "d"], "18"), (["familles", "0", "retours"], "3"),
         (["familles", "0", "de"], "-5"), (["familles", "0", "a"], "101"), (["familles", "0", "position"], "lat_g"), (["familles", "9", "n"], "2"),
         (["types_etriers", "0", "d"], "18"), (["zones", "0", "s"], "1"), (["zones", "0", "type"], "E9"), (["enrobage"], "15"),
         (["b"], "6"), (["familles", "+"], "chapeau"), (["inconnu"], "1"), (["forme"], "en T")]
mauvais = []
for chemin, valeur in REFUS:
    e2, m = mod(e0, chemin, valeur)
    if not (m and m.startswith("Saisie refusée") and e2 == e0):
        mauvais.append((chemin, valeur, m))
chk(f"{len(REFUS)} saisies hors du domaine de l'écran : refusées avec un message, la note intacte (jamais ramenées en silence)", not mauvais, mauvais)
e2, m = mod(e0, ["talon_gauche", "b"], "0")
chk("talon de 0 cm (l'écran Streamlit le lirait 10 cm) : refusé, le message dit pourquoi", "hors du domaine" in m, m)
e2, m = mod(e0, ["L"], "8")
chk("une saisie valable s'écrit (L = 8 m) sans message", m is None and e2["poutres"][0]["L"] == 8.0)
e1 = etat_de(dict(EX, familles=[{"role": "principal", "de": 100 / 3}]))
e2, m = mod(e1, ["familles", "0", "de"], "33,333")
chk("réécrire ce que le champ affiche (33,333) garde la valeur exacte (100/3) : rien ne change", m is None and e2 == e1)
e2, m = mod(e1, ["familles", "0", "de"], "33,3")
chk("une autre valeur (33,3) s'écrit", m is None and e2["poutres"][0]["familles"][0]["de"] == 33.3)
# effets de bord : ceux de beton/taux/saisie.py (prouvés contre l'écran dans tests/test_taux_moteur.py)
p = copy.deepcopy(e0["poutres"][0])
SA.ajouter_famille(p, "principal")
e2, m = mod(e0, ["familles", "+"], "principal")
chk("« + Lit principal » : la famille de l'écran (2Ø16, âme en bas, 0 à 100 %)", m is None and e2["poutres"][0] == EN.normaliser(p))
e2, m = mod(e0, ["familles", "+"], "talon")
chk("« + Renfort talon » : talon gauche inférieur (forme à talon gauche)", e2["poutres"][0]["familles"][-1]["position"] == "talon_g_inf")
e2, m = mod(etat_de(DEUX), ["familles", "+"], "talon")
chk("deux talons : « + Renfort talon » refusé avec la raison (l'écran Streamlit s'arrêterait sur « Talon gauche non défini. »)",
    m and "Talon gauche non défini" in m and e2 == etat_de(DEUX), m)
chk("deux talons : le bouton « Renfort de talon » n'est pas proposé, ni les positions de talon",
    'data-valeur="talon"' not in T.formulaire(etat_de(DEUX)) and 'value="talon_g_inf"' not in T.formulaire(etat_de(DEUX)))
e2, m = mod(e0, ["forme"], "deux_talons")
chk("vers deux talons avec une famille sur un talon : refusé, la famille nommée", m and "INF2" in m and e2 == e0, m)
e2, m = mod(e0, ["forme"], "rectangulaire")
chk("forme sans talon : la famille du talon passe sur « Armatures inférieures (âme) », et c'est dit",
    e2["poutres"][0]["familles"][2]["position"] == "ame_inf" and m and "INF2" in m and "son talon" in m, m)
e2, m = mod(e0, ["familles", "3", "retours"], "0")
chk("« Retours » = 0 remet la hauteur de retour à 0 (edit_family)", e2["poutres"][0]["familles"][3]["h_retour"] == 0.0 and m is None)
e2, m = mod(e0, ["familles", "0", "-"], "1")
chk("retirer une famille", len(e2["poutres"][0]["familles"]) == 3 and m is None)
e2, m = mod(e0, ["zones", "+"], "1")
chk("« + Zone d'étriers » : Z3 de 0 à 33 % à 15 cm sur le premier type (add_stirrup_zone)",
    e2["poutres"][0]["zones"][-1] == {"nom": "Z3", "de": 0.0, "a": 33.0, "s": 15.0, "type": "E1"})
e2, m = mod(etat_de(VIDE), ["zones", "+"], "1")
chk("« + Zone d'étriers » sans type : un type E1 Ø8 est créé d'abord (le bouton « + Étrier »)",
    e2["poutres"][0]["types_etriers"] == [{"nom": "E1", "d": 8.0}] and e2["poutres"][0]["zones"][0]["type"] == "E1")
e2, m = mod(e0, ["types_etriers", "0", "nom"], "Cadre")
chk("un type renommé : ses zones le suivent", [z["type"] for z in e2["poutres"][0]["zones"]] == ["Cadre", "Cadre"])
e3 = etat_de(P2)
e2, m = mod(e3, ["types_etriers", "1", "-"], "1")
chk("un type supprimé : ses zones passent au premier type restant (edit_stirrup_zone), et c'est dit",
    [z["type"] for z in e2["poutres"][0]["zones"]] == ["E1", "E1", "E1"] and m and "Z1" in m, m)
e2, m = mod(e0, ["types_etriers", "0", "-"], "1")
chk("le dernier type supprimé : les zones ne comptent plus, c'est dit (et l'écran le montre en rouge)",
    m and "ne comptent plus" in m and "non comptée : sans type" in T.formulaire(e2), m)
e2, m = mod(e3, ["types_etriers", "1", "d"], "7.5")
chk("un Ø d'étrier hors du menu (7,5, admis par l'écran) reste proposé et retenu", e2 == e3 and m is None and 'value="7.5" selected' in T.formulaire(e3))
# les mètres
em = etat_de(dict(EX, longueurs="metre"))
e2, m = mod(em, ["familles", "1", "a_m"], "2,5")
chk("en mètres : « À » = 2,5 m s'écrit 100 × 2,5 / 6 % (edit_family)", e2["poutres"][0]["familles"][1]["a"] == 100.0 * 2.5 / 6.0 and m is None)
e2, m = mod(em, ["familles", "1", "a_m"], "7")
chk("en mètres : au-delà de la portée, refusé", m and "hors de la portée" in m and e2 == em, m)
e2, m = mod(em, ["L"], "4")
p = copy.deepcopy(em["poutres"][0])
SA.changer_portee(p, 4.0)
chk("en mètres, portée 6 → 4 m : les mètres qui tiennent sont gardés, les autres gardent leur % (D8), et c'est dit",
    e2["poutres"][0] == EN.normaliser(p) and m and "INF1" in m and "SUP1" not in m, m)
chk("en mètres, les champs « de », « à » des familles montrent des mètres", 'data-chemin="familles/1/a_m"' in T.formulaire(em) and "familles/1/a\"" not in T.formulaire(em))

# ------------------------------------------------------------------ 5. le sélecteur de diamètre
print("\n=== 5. Le sélecteur de diamètre ===")
h = T.selecteur(e0, "d-0")
rows = re.findall(r'<button class="pk-row([^"]*)" data-choix="([^"]+)" data-prop="[^"]*"><span class="mk">.*?</span><span class="nm">[^<]*</span><span class="kg">([^<]*)</span>', h)
cands = MO.candidats_diametre(e0["poutres"][0], 0)
chk("les dix Ø de l'écran, triés, chacun avec le taux de la poutre recalculé par le moteur",
    [x[1] for x in rows] == [c["nom"] for c in cands] and [x[2] for x in rows] == [C.fr(c["kg_m3_majore"], 1) for c in cands]
    and [c["d"] for c in cands] == list(EN.DIAMETRES), rows[:3])
chk("la disposition retenue est cochée (✓), une seule ; aucune « optimale » (un quantitatif ne vérifie rien), et c'est écrit",
    [x[1] for x in rows if "cur" in x[0]] == ["3Ø20"] and not any("opt" in x[0] for x in rows) and "pas d'optimal" in h)
chk("le pied du sélecteur dit ce que montrent les colonnes (taux de la poutre, part de la famille), pas « taux » seul",
    "taux de la poutre (kg/m³) · part de la famille dans l'acier</span>" in h and "· taux</span>" not in h)
e2 = copy.deepcopy(e0)
T.choisir_section(e2, "d-0", "3Ø25")
chk("choisir 3Ø25 : la famille passe en Ø25 et le résultat suit", e2["poutres"][0]["familles"][0]["d"] == 25
    and C.fr(MO.calculer(e2["poutres"][0])["quantites"]["kg_m3_majore"], 1) in T.resultat(e2)["resultat"])
r = req(module="taux", action="selecteur", etat=e0, role="d-9")
chk("un sélecteur inconnu rend une erreur lisible", not r["ok"] and "sélecteur inconnu" in r["erreur"])
chk("chaque famille annonce son sélecteur (data-pk) dans le résultat, les zones non",
    re.findall(r'data-pk="([^"]+)"', T.resultat(e0)["resultat"]) == ["d-0", "d-1", "d-2", "d-3"])

# ------------------------------------------------------------------ 6. l'échappement
print("\n=== 6. Un texte saisi est échappé partout ===")
PIEGE = '<img src=x onerror=alert(1)>'
e = etat_de(dict(P2, nom=PIEGE, types_etriers=[{"nom": PIEGE, "d": 8.0}], zones=[{"nom": PIEGE + "z", "de": 0, "a": 50, "s": 15, "type": PIEGE}]),
            dict(EX, nom=PIEGE + "2"))
out = T.resultat(e)
frag = {"résultat": out["resultat"], "fenêtres": out["volets"], "données": out["donnees"], "éléments": out["elements"], "phrase": out["phrase"],
        "dessins": "".join(v["svg"] for v in out["vues"]), "sélecteur": T.selecteur(e, "d-0")}
fuites = [k for k, v in frag.items() if "<img" in v]
chk("noms de poutre, de type et de zone piégés : aucun « <img » brut dans les fragments", not fuites, fuites)
chk("… et ils s'affichent comme du texte (échappés)", "&lt;img src=x onerror=alert(1)&gt;" in out["donnees"] and "&lt;img" in frag["dessins"])
chk("titres et légendes des vues : du texte (la page les échappe)", all("<" not in v["titre"] + v["court"] + v["legende"] for v in out["vues"]))

# ------------------------------------------------------------------ 7. la note : plusieurs poutres
print("\n=== 7. La note : plusieurs poutres, résumé, vue « Note » ===")
e = etat_de(EX, P2, TRONC)
out = T.resultat(e)
res = [MO.calculer(p) for p in e["poutres"]]
s = MO.synthese_note(res)
chk("le résumé est une consultation (pas de taux de travail) qui nomme le taux de la note et la poutre la plus armée",
    out["resume"]["taux"] is None and out["resume"]["niveau"] == "i" and C.fr(s["kg_m3_majore"], 1) in out["resume"]["gouvernant"]
    and "la plus armée : P2" in out["resume"]["gouvernant"], out["resume"])
chk("la note entière : Σ masses majorées / Σ volumes", abs(s["kg_m3_majore"] - sum(r["quantites"]["masse_acier_majoree"] for r in res)
                                                       / sum(r["quantites"]["Vc"] for r in res)) < 1e-12)
note = next((v for v in out["vues"] if v["cle"] == "note"), None)
chk("la vue « Note » liste chaque poutre et le total", note is not None and all(C.fr(r["quantites"]["kg_m3_majore"], 1) in note["svg"] for r in res)
    and C.fr(s["kg_m3_majore"], 1) in note["svg"])
chk("le pied du résultat donne le taux de la note", f'{C.fr(s["kg_m3_majore"], 1)}{C.NBSP}kg/m³' in out["resultat"])
chk("la liste des poutres : chacune avec son taux en kg/m³", all(f"{C.fr(r['quantites']['kg_m3_majore'], 1)}{C.NBSP}kg/m³" in out["elements"] for r in res))
e2 = copy.deepcopy(e)
T.gerer_elements(e2, "dupliquer")
chk("copier la poutre ouverte : « (copie) », ouverte", e2["poutres"][-1]["nom"] == "P1 (copie)" and e2["actif"] == 3)
T.gerer_elements(e2, "supprimer", 0)
chk("supprimer la première : l'ouverte reste la même", e2["actif"] == 2 and e2["poutres"][2]["nom"] == "P1 (copie)")
e3 = etat_de(EX)
T.gerer_elements(e3, "supprimer", 0)
chk("la dernière poutre ne se supprime pas", len(e3["poutres"]) == 1)
T.gerer_elements(e3, "ajouter")
chk("ajouter : P2, la poutre neuve de l'écran", e3["poutres"][1] == dict(EN.nouvelle_poutre(), nom="P2") and e3["actif"] == 1)

# ------------------------------------------------------------------ 8. la relecture d'une note
print("\n=== 8. La relecture d'un fichier .stab ===")
for abime, attendu in (({"poutres": [{"b": "30"}]}, "nombre attendu"), ({"poutres": [{"forme": "deux_talons", "familles": [{"role": "renfort", "position": "talon_g_inf"}]}]},
                                                                         "Talon gauche non défini"), ({"poutres": {"a": 1}}, "liste")):
    try:
        T.normaliser(abime)
        chk(f"note abîmée ({attendu}) : refusée", False)
    except ValueError as ex:
        chk(f"note abîmée : refusée avec une erreur lisible (« {attendu} »)", attendu in str(ex), str(ex))
chk("une note sans poutre : la note neuve", T.normaliser({}) == T.nouvel_etat() and T.normaliser({"poutres": []}) == T.nouvel_etat())
chk("la note neuve est celle de l'écran Streamlit au premier affichage (aucune armature, 1,080 m³)",
    T.nouvel_etat()["poutres"][0] == dict(EN.nouvelle_poutre(), nom="P1") and f"1,080{C.NBSP}m³" in T.resultat(T.nouvel_etat())["resultat"])
for actif in ("abc", [1], {"a": 1}, True):
    try:
        T.normaliser({"poutres": [EX], "actif": actif})
        chk(f"actif = {actif!r} : refusé", False)
    except ValueError as ex:
        chk(f"actif = {actif!r} : refusé en français (« rang de la poutre ouverte »), pas par une erreur Python", "rang de la poutre ouverte" in str(ex), str(ex))
chk("actif écrit en texte (« 1 ») : relu", T.normaliser({"poutres": [EX, P2], "actif": "1"})["actif"] == 1)

# ------------------------------------------------------------------ 8 bis. la relecture du 07/11/2026 : ce qu'elle a corrigé
print("\n=== 8 bis. Relecture : les défauts trouvés et corrigés ===")
for nom, p in CAS:
    v = re.search(r'<div class="verdict.*?</div></div>', T.resultat(etat_de(p))["resultat"], re.S).group(0)
    coupures = re.findall(r"\d[ ,.]?\d* (?:kg|m³|%)", re.sub(r"<[^>]+>", "", v))
    chk(f"{nom} : le verdict ne sépare jamais un nombre de son unité (espace insécable ; au téléphone « 1,380 » et « m³ » "
        "étaient sur deux lignes)", not coupures and v.count('class="tx-l"') == 2, coupures)
h = T.selecteur(etat_de(EX), "d-0")
chk("sélecteur : le « Ø » du sous-titre reste avec son mot (il était seul sur sa ligne)", f"retient le{C.NBSP}Ø" in h)
talon_z = MO.calculer(dict(EX))
codes_tz = [a["code"] for a in talon_z["avertissements"]]
sans_z = [a["code"] for a in MO.calculer(dict(EX, zones=[]))["avertissements"]]
rect_z = [a["code"] for a in MO.calculer(dict(EX, forme="rectangulaire", familles=[]))["avertissements"]]
chk("D11 dit : une forme à talon avec des étriers avertit que les cadres du talon ne sont pas comptés ; sans étrier ou sans "
    "talon, rien", "etriers_talon" in codes_tz and "etriers_talon" not in sans_z and "etriers_talon" not in rect_z, (codes_tz, sans_z, rect_z))
chk("… et l'avertissement est affiché dans le résultat", "cadres ou épingles du talon" in T.resultat(etat_de(EX))["resultat"])
chk("D11 : deux talons (talons ignorés, D2) n'ajoute pas cet avertissement",
    "etriers_talon" not in [a["code"] for a in MO.calculer(dict(DEUX))["avertissements"]])

# ------------------------------------------------------------------ 9. le chemin du calcul : ce qui se charge dans le navigateur
print("\n=== 9. Le chemin du navigateur : ni streamlit, ni pandas, ni matplotlib ===")
for fic in ("api_web/taux.py", "api_web/dessin_taux.py"):
    arbre = ast.parse(open(fic, encoding="utf-8").read())
    noms = [a.name for n in ast.walk(arbre) if isinstance(n, ast.Import) for a in n.names] + \
           [n.module for n in ast.walk(arbre) if isinstance(n, ast.ImportFrom) and n.level == 0]
    tiers = [x for x in noms if x.split(".")[0] not in sys.stdlib_module_names | {"beton"}]
    chk(f"{fic} n'importe que la bibliothèque standard, le moteur et les briques du site", not tiers, tiers)
SOUS = r'''
import json, sys
INTERDITS = ("streamlit", "pandas", "matplotlib", "scipy", "reportlab", "PIL", "fitz")
class Refus:
    def find_spec(self, nom, chemin=None, cible=None):
        if nom.split(".")[0] in INTERDITS:
            raise ImportError("refusé : " + nom)
sys.meta_path.insert(0, Refus())
sys.path.insert(0, sys.argv[1])
from api_web import routeur as R
etat = json.loads(sys.stdin.read())
r = json.loads(R.traiter(json.dumps({"module": "taux", "action": "calculer", "etat": etat})))
print(json.dumps({"ok": r["ok"], "erreur": r.get("erreur"), "interdits": sorted(m for m in sys.modules if m.split(".")[0] in INTERDITS)}))
'''
pr = subprocess.run([sys.executable, "-I", "-c", SOUS, RACINE], input=json.dumps(etat_de(EX, P2)), capture_output=True, text=True, check=False)
o = json.loads(pr.stdout or "{}") if pr.returncode == 0 else {"ok": False, "erreur": pr.stderr[-300:]}
chk("le module calcule par le routeur avec streamlit, pandas, matplotlib… REFUSÉS (le navigateur ne les a pas)", o.get("ok") and not o.get("interdits"), o)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)
