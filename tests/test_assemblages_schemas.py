# -*- coding: utf-8 -*-
"""Parité des SCHÉMAS avec les dessins du HTML de référence.

Le générateur Python (schemas.py) est comparé au 2ᵉ script du HTML exécuté
sous Node (outils/oracle_dessins.js) : pour chaque cas, chaque niveau de
cotation et chaque vue, mêmes éléments dans le même ordre, mêmes classes,
mêmes coordonnées (± 1e-6), mêmes textes de cote, mêmes clés modifiables,
même boîte de vue, même cartouche. Sont rejoués : les 31 cas du corrigé aux
trois niveaux, les alertes mises en évidence (chaque alerte du corrigé), le
mode prédimensionnement (cotes verrouillées) et 60 cas aléatoires.

Garanties propres au module (sans Node) : les 31 cotes existent, 18 portent
une clé, 13 sont calculées ; une cote hors niveau apparaît quand une alerte
la désigne ; le SVG est un document XML valide ; la notation affichée est
celle de l'Eurocode (hc, c, dc,sup, dc,inf, Δz) alors que la parité avec le
HTML se rejoue en notation du moteur ; en mode édition toutes les cotes
modifiables sont posées ; les poignées + / − et l'étiquette des groupes de
boulons existent à l'écran, pas sur la note.

Lancement : python3 tests/test_assemblages_schemas.py
"""
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)
sys.path.insert(0, os.path.join(RACINE, "tests"))

from acier.assemblages.poutre_poutre.doubles_cornieres import moteur, schemas  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom
          + (f"   [{info}]" if info and not cond else ""))


with open("acier/reference/reference_double_corniere.json", encoding="utf-8") as fh:
    REF = json.load(fh)
NS = "{http://www.w3.org/2000/svg}"
NUM = re.compile(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?")


def _hl(R, ident):
    if not ident:
        return None
    for a in R.alerts:
        if a.id == ident:
            return dict(dims=set(a.dims), elems=set(a.elems))
    return None


def _norm(svg):
    """Séquence normalisée des éléments d'un SVG : (balise, classes, nombres,
    texte, clé) — indépendante de l'espace de noms et du style embarqué."""
    root = ET.fromstring(svg)
    out = [("svg", "", [float(x) for x in NUM.findall(root.get("viewBox"))], "", "")]

    def rec(el):
        for e in el:
            tag = e.tag.replace(NS, "")
            if tag == "style":
                continue
            cls = " ".join(sorted(e.get("class", "").split()))
            nums = []
            for a in ("x", "y", "width", "height", "rx", "cx", "cy", "r", "points", "d",
                      "transform", "font-size"):
                v = e.get(a)
                if v is not None:
                    nums.extend(float(x) for x in NUM.findall(v))
            txt = (e.text or "").strip() if tag == "text" else ""
            out.append((tag, cls, nums, txt, e.get("data-key") or ""))
            rec(e)
    rec(root)
    return out


def _compare(a, b, tol=1e-6):
    if len(a) != len(b):
        return [f"{len(a)} éléments attendus, {len(b)} obtenus"]
    ec = []
    for i, (x, y) in enumerate(zip(a, b)):
        if x[0] != y[0] or x[1] != y[1] or x[3] != y[3] or x[4] != y[4]:
            ec.append(f"élément {i} : {x[:2]} {x[3]!r} {x[4]!r} ≠ {y[:2]} {y[3]!r} {y[4]!r}")
        elif len(x[2]) != len(y[2]) or any(abs(p - q) > tol * max(1, abs(p)) for p, q in zip(x[2], y[2])):
            ec.append(f"élément {i} ({x[0]} {x[1]}) : coordonnées {x[2][:6]} ≠ {y[2][:6]}")
        if len(ec) >= 3:
            break
    return ec


# ================================================================
print("=== 1. Garanties propres au module ===")
R0 = moteur.compute({})
e0 = schemas.elevation(R0, schemas.Options(lvl=2, interactive=True))
p0 = schemas.plan(R0, schemas.Options(lvl=2, interactive=True))
ids = set()
keys = set()


def rec(prims):
    for p in prims:
        if p["t"] == "g":
            if p.get("id"):
                ids.add(p["id"])
            if p.get("key"):
                keys.add(p["key"])
            rec(p["enfants"])


rec(e0.cotes); rec(p0.cotes)
Rg = moteur.compute(dict(d_nb=40, n2S_u=2, n2P_u=2))
eg = schemas.elevation(Rg, schemas.Options(lvl=2, interactive=True))
pg = schemas.plan(Rg, schemas.Options(lvl=2, interactive=True))
rec(eg.cotes); rec(pg.cotes)
Rw = moteur.compute(dict(fix_P="Soudée", fix_S="Soudée", lh_S=40))
rec(schemas.elevation(Rw, schemas.Options(lvl=2, interactive=True)).cotes)
rec(schemas.plan(Rw, schemas.Options(lvl=2, interactive=True)).cotes)
ATTENDUES = set("Lc zc e1S p1S e1botS e1b he e2b gh ln dnt dnb dtop z e1P p1P e1botP ztP gA p3 bA bB tc twS e2A e2B p2S p2P aS aP lhS".split())
chk("les 31 cotes identifiées existent toutes", ATTENDUES <= ids, str(sorted(ATTENDUES - ids)))
chk("18 cotes modifiables (clé d'entrée) hors mode édition, 13 calculées",
    keys == set(schemas.CLE_PAR_COTE.values()) - {"lh_P"} and len(schemas.COTES_CALCULEES) == 13, str(sorted(keys)))
rec(schemas.plan(Rw, schemas.Options(lvl=2, interactive=True, editables=True)).cotes)
chk("mode édition, cornières soudées : l'étiquette des retours ℓh des ailes A s'ajoute (19 clés)",
    keys == set(schemas.CLE_PAR_COTE.values()) and len(keys) == 19, str(sorted(keys)))
chk("le SVG est un document XML valide (élévation et plan)",
    ET.fromstring(e0.svg()) is not None and ET.fromstring(p0.svg()) is not None)
svg1 = schemas.elevation(R0, schemas.Options(lvl=1, interactive=True)).svg()
chk("niveau 1 : gh (niveau 2) absente ; dnb = 0 : étiquette absente",
    'data-dim="gh"' not in svg1 and 'data-dim="dnb"' not in svg1)
Rb = moteur.compute(dict(LC_u=260))
hl = _hl(Rb, "h_dispo")
svgb = schemas.elevation(Rb, schemas.Options(lvl=0, interactive=True, hl=hl)).svg()
chk("niveau 0 + alerte h_dispo : les cotes Lc et zc apparaissent quand même, en rouge",
    'data-dim="Lc"' in svgb and 'data-dim="zc"' in svgb and 'class="dl ed hot"' in svgb)
Rp = moteur.compute(dict(mode_calc="PRÉDIMENSIONNEMENT"))
svgp = schemas.elevation(Rp, schemas.options_ecran(Rp, 2)).svg()
chk("prédimensionnement : Lc verrouillée (pas de data-key), gh reste modifiable",
    'data-key="LC_u"' not in svgp and 'data-key="g_h"' in svgp)
chk("cotes_modifiables() liste les cotes cliquables dans l'ordre de pose",
    [k for k, _, _ in schemas.cotes_modifiables(e0)][:3] == ["e2b_u", "g_h", "l_n"])
# --- notation Eurocode, mode édition, poignées
svg_ec = schemas.elevation(R0, schemas.options_ecran(R0, 1)).svg()
chk("notation Eurocode par défaut : hc = 190, c = 150, dc,sup 50, zc 50 ; plus de « Lc », « ln », « dnt »",
    all(x in svg_ec for x in (">hc = 190<", ">c = 150<", ">dc,sup 50<", ">zc 50<", ">Δz 0<"))
    and not any(x in svg_ec for x in (">Lc", ">ln ", ">dnt", "déc.")), svg_ec[:0])
chk("plus de cartouche texte à l'écran (colonne figée courte ; la carte et les panneaux portent ces données)",
    "hc 190 mm" not in svg_ec and "Cornières : 2 ×" not in svg_ec)
chk("notation du moteur sur demande (parité) : Lc = 190",
    ">Lc = 190<" in schemas.elevation(R0, schemas.Options(lvl=1, interactive=True, notation={})).svg())
chk("mode édition : gh (niveau 2), Δz et dc,inf (étiquettes de valeur nulle) sont posées au niveau 1",
    all(f'data-dim="{x}"' in svg_ec for x in ("gh", "dtop", "dnb")) and 'data-dim="gh"' not in svg1)
chk("poignées des deux groupes : + / − (data-action) et étiquette cliquable (data-group)",
    all(x in svg_ec for x in ('data-action="n1S_u:+1"', 'data-action="n1S_u:-1"', 'data-action="n1P_u:+1"',
                              'data-group="S"', 'data-group="P"', ">3 × 1<")))
svg_r = schemas.elevation(R0, schemas.options_rapport()).svg()
chk("note : ni poignées ni cotes cliquables, notation Eurocode",
    "data-action" not in svg_r and "data-group" not in svg_r and "data-key" not in svg_r and ">hc = 190<" in svg_r)
chk("prédimensionnement : poignées présentes mais inertes (n1 piloté)",
    "data-action" not in schemas.elevation(Rp, schemas.options_ecran(Rp, 1)).svg()
    and 'class="gp"' in schemas.elevation(Rp, schemas.options_ecran(Rp, 1)).svg())
chk("le SVG avec poignées reste un document XML valide",
    ET.fromstring(svg_ec) is not None and ET.fromstring(schemas.plan(R0, schemas.options_ecran(R0, 1)).svg()) is not None)
# --- rendu réaliste : congés réels, hachures, cordons à leur taille
sec = schemas._section_I(300, 390, 11, 19, 27)
chk("section en I : quatre congés de rayon r (HEA 400 : r = 27), contour fermé",
    len(sec) == 8 + 4 * (schemas.SEGMENTS_ARC + 1) and min(abs(x) for x, y in sec if 19 < y < 371) == 5.5
    and any(abs(x - (5.5 + 27 * (1 - math.cos(math.pi / 4)))) < 1e-9 and abs(y - (19 + 27 * (1 - math.sin(math.pi / 4)))) < 1e-9 for x, y in sec))
chk("semelles rectangulaires : les quatre coins sous/sur semelle existent (le biseau du 21/09 ne revient pas)",
    all(any(abs(x - cx) < 1e-9 and abs(y - cy) < 1e-9 for x, y in sec)
        for cx, cy in ((150, 19), (-150, 19), (150, 371), (-150, 371))))
corn = schemas._corniere_plan(5.5, 105.5, 10, 4.25, 14.25, 104.25, 1, 12)
chk("cornière en plan : congé de racine r = 12 (centre (27,5 ; 26,25)) et bouts arrondis r/2",
    any(abs(x - (27.5 - 12 * math.cos(math.pi / 4))) < 1e-9 and abs(y - (26.25 - 12 * math.sin(math.pi / 4))) < 1e-9 for x, y in corn)
    and any(abs(x - (99.5 + 6 * math.cos(math.pi / 4))) < 1e-9 for x, y in corn))
h = schemas._hachures([(0, 0), (10, 0), (10, 10), (0, 10)], 5)
chk("hachures à 45° d'un carré de 10 (pas 5) : trois segments dans le carré, extrémités sur le bord",
    len(h) == 3 and all(0 - 1e-9 <= v <= 10 + 1e-9 for seg in h for p in seg for v in p)
    and all(abs((b[1] - a[1]) - (b[0] - a[0])) < 1e-9 for a, b in h))
svg_r = schemas.elevation(R0, schemas.options_ecran(R0, 1)).svg()
chk("élévation réaliste : section en I hachurée (classe ht), rondelles (bw), rayon du grugeage r_n = 10 (arc : plus de 6 sommets)",
    'class="ht"' in svg_r and svg_r.count('class="bw"') == 3 and 'class="ps"' in svg_r
    and svg_r.split('class="ps"')[1].split("/>")[0].count(",") > 6)
Rw = moteur.compute(dict(fix_P="Soudée", fix_S="Soudée", lh_S=40, a_S=5, a_P=6))
svg_w = schemas.plan(Rw, schemas.options_ecran(Rw, 1)).svg()
chk("plan réaliste : cordons en triangles a·√2 (4 : deux cornières × deux côtés), cornières hachurées",
    svg_w.count('class="wb"') == 4 and svg_w.count('class="ht htc"') == 2)
svg_fid = schemas.plan(Rw, schemas.Options(lvl=1, interactive=True, realiste=False)).svg()
chk("mode fidèle au HTML (parité) : ni hachures, ni congés, cordons en disques (wd)",
    'class="ht' not in svg_fid and 'class="wb"' not in svg_fid and 'class="wd"' in svg_fid)
chk("rapport : rendu réaliste aussi", 'class="ht"' in schemas.elevation(R0, schemas.options_rapport()).svg())
# --- pièces sélectionnables, drag, cartouche
chk("écran : chaque pièce est un groupe cliquable (beamP, beamS, cleat, bolts, efforts) et la poutre "
    "portée porte data-drag=g_h",
    all(f'data-group="{g}"' in svg_ec for g in ("beamP", "beamS", "cleat", "bolts", "efforts"))
    and 'data-drag="g_h"' in svg_ec and 'data-dsym="gh"' in svg_ec)
chk("étiquette des efforts : « VEd 125 » cliquable (panneau efforts)",
    ">VEd 125<" in svg_ec and 'class="ef ed"' in svg_ec)
svg_pl = schemas.plan(R0, schemas.options_ecran(R0, 1)).svg()
chk("plan : pièces cliquables aussi (deux cornières, âmes, boulons) et drag de la portée",
    svg_pl.count('data-group="cleat"') == 2 and 'data-group="beamS"' in svg_pl and 'data-drag="g_h"' in svg_pl)
svg_note = schemas.elevation(R0, schemas.options_rapport()).svg()
chk("parité et note : aucun groupe de pièce, pas d'étiquette d'efforts",
    "data-group" not in schemas.elevation(R0, schemas.Options(lvl=2, interactive=True)).svg()
    and "data-group" not in svg_note and "VEd" not in svg_note)
chk("lignes de congé tf + r de la poutre portée (classe flr, écran et note)",
    'class="flr"' in svg_ec and 'class="flr"' in svg_note)
chk("le cartouche reste rejoué par la parité (notation moteur) et absent de la note",
    "Cornières : 2 ×" in schemas.elevation(R0, schemas.Options(lvl=1, interactive=True, notation={})).svg()
    and "Cornières : 2 ×" not in svg_note)
chk("étiquettes de cote discrètes : halo blanc, plus de fond jaune permanent",
    'fill="#FFFFFF" fill-opacity="0.85"' in svg_ec and "#FFF7CF" not in svg_ec.split("</style>")[1])
# --- traits d'axe des boulons (rendu réaliste)
chk("boulons S : traits d'axe mixtes (rangées + files) à la place des croix ; boulons P : axes fins, plus de rouge",
    'class="ax"' in svg_ec and 'class="cm"' not in svg_ec and 'class="bp"' not in svg_ec
    and 'stroke-dasharray="8 2.5 2 2.5"' in svg_ec)
chk("plan réaliste : axes aussi (files S, boulons P)", 'class="ax"' in svg_pl and 'class="bp"' not in svg_pl)
chk("alerte sur les boulons P : l'axe passe en rouge épais",
    'class="ax hot"' in schemas.elevation(R0, schemas.options_ecran(R0, 1, dict(dims=set(), elems={"boltsP"}))).svg())
chk("parité : croix et tirets du HTML conservés (cm, bp)",
    'class="cm"' in schemas.elevation(R0, schemas.Options(lvl=2, interactive=True, realiste=False)).svg())
# --- symboles de soudure, glissement 2 axes, arêtes cachées
chk("glissement 2 axes : la poutre portée porte gh (horizontal) ET Δz (vertical) en élévation, gh seul en plan",
    'data-drag="g_h"' in svg_ec and 'data-drag2="d_top"' in svg_ec and 'data-dsym2="Δz"' in svg_ec
    and "data-drag2" not in svg_pl and 'data-drag="g_h"' in svg_pl)
svg_ws = schemas.elevation(Rw, schemas.options_ecran(Rw, 1)).svg()
svg_wp = schemas.plan(Rw, schemas.options_ecran(Rw, 1)).svg()
chk("cordons B soudés : symbole EN 22553 en élévation (flèche wsy + triangle wst) avec la désignation « a 5 » éditable",
    'class="wsy"' in svg_ws and 'class="wst"' in svg_ws and 'data-key="a_S"' in svg_ws and ">a 5<" in svg_ws)
chk("cordons A soudés : symbole en plan (un seul, cornière du haut) avec « a 6 » éditable",
    svg_wp.count('class="wst"') == 1 and 'data-key="a_P"' in svg_wp and ">a 6<" in svg_wp)
chk("le symbole appartient à la pièce cordon (clic → panneau)", 'data-group="weldS"' in svg_ws)
chk("arêtes cachées des semelles en plan (trait interrompu hd) : porteuse + portée ; absentes en parité",
    svg_pl.count('class="hd"') == 2 and 'stroke-dasharray="4 2.5"' in svg_pl
    and 'class="hd"' not in schemas.plan(R0, schemas.Options(lvl=1, interactive=True, realiste=False)).svg())
chk("note : symboles et arêtes cachées aussi (rendu réaliste)",
    'class="hd"' in schemas.plan(R0, schemas.options_rapport()).svg()
    and 'class="wst"' in schemas.elevation(Rw, schemas.options_rapport()).svg())

# --- lignes de rupture : le bord d'un profil coupé est un zigzag, pas un
# --- bord franc (élévation : portée à droite ; plan : porteuse haut et bas,
# --- portée à droite ; vue de droite : porteuse des deux côtés) — et la
# --- parité (realiste=False) garde la géométrie du HTML
opt_par = schemas.Options(lvl=1, interactive=True, notation={}, realiste=False)
e_r = schemas.elevation(R0, schemas.Options(lvl=1))
e_p = schemas.elevation(R0, opt_par)
poly_e = next(p for p in e_r.corps if p["t"] == "poly" and "ps" in p["cls"])
poly_ep = next(p for p in e_p.corps if p["t"] == "poly" and "ps" in p["cls"])
xs = sorted(pt[0] for pt in poly_e["pts"])
chk("élévation : bord droit de la portée en ligne de rupture (pointe unique hors du bord coupé) ; "
    "bord franc en parité",
    xs[-1] > xs[-2] + 1.5 and abs(max(pt[0] for pt in poly_ep["pts"]) - xs[-2]) < 1e-6)
pl_r = schemas.plan(R0, schemas.Options(lvl=1))
pl_p = schemas.plan(R0, opt_par)
poly_pp = next(p for p in pl_r.corps if p["t"] == "poly" and "pp" in p["cls"])
poly_ps = next(p for p in pl_r.corps if p["t"] == "poly" and "ps" in p["cls"])
ys = sorted(pt[1] for pt in poly_pp["pts"]); xs2 = sorted(pt[0] for pt in poly_ps["pts"])
chk("plan : âme porteuse rompue en haut ET en bas, âme portée rompue à droite ; rectangles en parité",
    ys[0] < ys[1] - 1.5 and ys[-1] > ys[-2] + 1.5 and xs2[-1] > xs2[-2] + 1.5
    and any(p["t"] == "rect" and "pp" in p["cls"] for p in pl_p.corps)
    and any(p["t"] == "rect" and "ps" in p["cls"] for p in pl_p.corps))
vd = schemas.vue_droite(R0, schemas.options_fabrication())
face = next(p for p in vd.corps if p["t"] == "poly" and "pp" in p["cls"])
xs3 = sorted(pt[0] for pt in face["pts"])
chk("vue de droite : la face de l'âme porteuse est rompue des deux côtés (une pointe par bord)",
    xs3[0] < xs3[1] - 1.5 and xs3[-1] > xs3[-2] + 1.5)

# --- plan de principe (fabrication) : renvois de perçage, rayon du
# --- grugeage, cote zt, répartition des cotes entre vues, police imposée
# --- (fs 12 = TEXTE_MM × 5, les conditions réelles de la page à 1:5)
svg_fe = schemas.elevation(R0, schemas.options_fabrication(schemas.EXCLURE_ELEVATION, 12)).svg()
svg_fd = schemas.vue_droite(R0, schemas.options_fabrication(fs_force=12)).svg()
svg_ecran = schemas.elevation(R0, schemas.options_ecran(R0)).svg()
chk("fabrication : renvois 3×Ø22 (élévation) et 6×Ø22 (vue de droite), rayon du grugeage « r 10 »",
    ">3×Ø22<" in svg_fe and ">r 10<" in svg_fe and ">6×Ø22<" in svg_fd)
chk("fabrication : zt = 85 (perçage P depuis le dessus de la porteuse) sur la vue de droite seulement",
    "zt = 85" in svg_fd and "zt" not in svg_fe)
chk("les renvois de perçage n'existent pas à l'écran (les panneaux portent déjà Ø et n)",
    "3×Ø22" not in svg_ecran and "r 10<" not in svg_ecran)
chk("répartition sans doublon : hc sur la vue de droite, pas sur l'élévation du plan de principe",
    "hc = 190" in svg_fd and "hc = 190" not in svg_fe)
f12 = [schemas.elevation(R0, schemas.options_fabrication(schemas.EXCLURE_ELEVATION, 12)),
       schemas.plan(R0, schemas.options_fabrication(schemas.EXCLURE_PLAN, 12)),
       schemas.vue_droite(R0, schemas.options_fabrication(fs_force=12))]
chk("police imposée (fs_force) : les trois vues du plan de principe partagent la même taille de texte",
    all(v.fs == 12 for v in f12))

# ================================================================
print("\n=== 2. Parité avec les dessins du HTML (oracle Node) ===")
NODE = shutil.which("node")
if not NODE:
    print("  IGNORÉ  node introuvable : parité des dessins non rejouée")
else:
    cas = []
    for c in REF["cas"]:
        cas.append(dict(id=c["id"], inputs=c["inputs"], niveaux=[0, 1, 2], hl=None))
        for a in c["attendu"]["alertes"]:
            cas.append(dict(id=c["id"] + "+" + a["id"], inputs=c["inputs"], niveaux=[1], hl=a["id"]))
    cas.append(dict(id="predim", inputs=dict(mode_calc="PRÉDIMENSIONNEMENT"), niveaux=[2], hl=None))
    rng = random.Random(3)
    from alea_double_corniere import tirage  # noqa: E402
    for i in range(60):
        cas.append(dict(id=f"alea_{i}", inputs=tirage(rng), niveaux=[rng.choice([0, 1, 2])], hl=None))
    fichier = os.path.join(os.environ.get("TMPDIR", "/tmp"), "cas_dessins.json")
    with open(fichier, "w", encoding="utf-8") as fh:
        json.dump(cas, fh, ensure_ascii=False)
    proc = subprocess.run([NODE, "outils/oracle_dessins.js",
                           "acier/reference/assemblage_double_corniere_EC3.html", fichier],
                          capture_output=True, text=True, timeout=600)
    chk("l'oracle des dessins s'exécute", proc.returncode == 0, proc.stderr[:300])
    attendu = json.loads(proc.stdout) if proc.returncode == 0 else []
    n_vues = 0
    div = []
    for c, exp in zip(cas, attendu):
        R = moteur.compute(c["inputs"])
        hl = _hl(R, c["hl"])
        for l in c["niveaux"]:
            # notation du moteur (le HTML écrit Lc, ln, dnt…) et géométrie du
            # HTML (rectangles sans congés ni hachures) : la parité prouve la
            # feuille de cotation ; le rendu réaliste est testé au §1
            opt = schemas.Options(lvl=l, interactive=True, hl=hl, notation={}, realiste=False,
                                  locked=set(schemas.PILOTEES_PAR_PREDIM) if R.pred else None)
            for vue, fn in (("elev", schemas.elevation), ("plan", schemas.plan)):
                n_vues += 1
                try:
                    ec = _compare(_norm(exp["vues"][str(l)][vue]), _norm(fn(R, opt).svg()))
                except Exception as e:  # noqa: BLE001
                    ec = [f"exception {type(e).__name__}: {e}"]
                if ec:
                    div.append(f"{c['id']} niveau {l} {vue} : " + " | ".join(ec))
    chk(f"{n_vues} vues comparées ({len(cas)} cas × niveaux × 2 vues), {len(div)} divergente(s)",
        n_vues >= 200 and not div, " || ".join(div[:3]))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:600])
sys.exit(1 if KO else 0)
