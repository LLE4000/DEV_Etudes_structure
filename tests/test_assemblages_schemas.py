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
la désigne ; le SVG est un document XML valide.

Lancement : python3 tests/test_assemblages_schemas.py
"""
import json
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
chk("18 cotes modifiables (clé d'entrée), 13 calculées",
    keys == set(schemas.CLE_PAR_COTE.values()) and len(schemas.COTES_CALCULEES) == 13, str(sorted(keys)))
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
            opt = schemas.Options(lvl=l, interactive=True, hl=hl,
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
