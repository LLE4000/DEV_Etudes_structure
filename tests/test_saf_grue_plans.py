# -*- coding: utf-8 -*-
"""Générateur SAF — fondation de grue : le PLAN FERRAILLÉ, les COUPES, le BORDEREAU et la NOTE (lot 4d, § 6 du cahier des
charges du 15/11/2026) : les tests D1 à D3, et les onglets de l'écran.

  D1  l'onglet Ferraillage : toutes les barres du bordereau sont sur le plan, avec leur repère ; les treillis (texte et
      carré de grille), les renforts de bandes (zone hachurée, étiquette), les renforts locaux, les U, les cages, les
      repères des coupes A-A et B-B, les cotes ; le fond « taux de travail du treillis » avec sa ligne noire à 100 % ;
  D2  l'onglet Coupes : A-A et B-B, cotées (h, hauteur des plots, longueur, enrobage), les mêmes repères que le
      bordereau ;
  D3  la note PDF : efforts (diagrammes M(s), V(s)), ferraillage, plan, deux coupes, bordereau, quantités, « Non vérifiés » ;
  et l'écran : les trois onglets, la barre de chaque onglet, le bordereau dépliable ; les MUTATIONS.

    /usr/bin/python3.12 tests/test_saf_grue_plans.py
    /usr/bin/python3.12 tests/test_saf_grue_plans.py --rapide [--racine dossier]
"""
import base64
import html as H
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(ICI)
ARGS = sys.argv[1:]
RACINE_MOTEUR = ARGS[ARGS.index("--racine") + 1] if "--racine" in ARGS else RACINE
RAPIDE = "--rapide" in ARGS
os.chdir(RACINE)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, RACINE_MOTEUR)
sys.path.insert(1, ICI)

from api_web import commun as C  # noqa: E402
from api_web import dessin_ferraillage as DF  # noqa: E402
from api_web import routeur  # noqa: E402
from api_web import saf as M  # noqa: E402
from saf.fondation_grue import projet as PR  # noqa: E402

OK, KO = [], []
INFOS = {"nom": "Grue G1", "chantier": "Chantier A", "indice": "B", "date": "15/11/2026"}


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:500]}]" if info != "" and not cond else ""))


def q(**k):
    return json.loads(routeur.traiter(json.dumps(dict(module="saf", **k))))


def mod(etat, chemin, valeur):
    r = q(action="modifier", etat=etat, chemin=[str(c) for c in chemin], valeur=valeur)
    assert r["ok"] and not r.get("message"), (chemin, valeur, r.get("message"))
    return r


def txt(h):
    return " ".join(H.unescape(re.sub(r"<[^>]+>", " ", h)).replace(C.NBSP, " ").replace(C.FINE, " ").split())


def txt_svg(svg):
    """Les textes d'un dessin, chacun une fois (le liseré blanc d'un texte — son double « fill=none » — écarté)."""
    return txt(re.sub(r'<text[^>]*fill="none"[^>]*>.*?</text>', "", svg, flags=re.S))


def pdf_texte(octets):
    return " ".join(subprocess.run(["pdftotext", "-", "-"], input=octets, capture_output=True, check=True).stdout.decode("utf-8")
                    .replace(C.NBSP, " ").replace(C.FINE, " ").split())


def calcul(etat):
    el = PR.lire(etat["fondations"][etat["actif"]])
    calc = M.calculer(el)
    return el, calc, M.ferraillage(el, calc), M.efforts(el, calc)


def rapides():
    type_ = M._etat_type()
    el, calc, fer, ef = calcul(type_)
    reps_bd = [ln["rep"] for ln in fer["bordereau"]["lignes"]]

    print("== l'écran : les trois onglets")
    r = q(action="calculer", etat=type_)
    ent = r["vues"][0]["entete"]
    chk("la barre du centre : trois onglets — Plan (choisi), Ferraillage, Coupes",
        re.findall(r'data-chemin="vue/onglet" data-valeur="([a-z]+)"', ent) == ["plan", "ferraillage", "coupes"]
        and re.search(r'class="on" data-agir="1" data-chemin="vue/onglet" data-valeur="plan"', ent) is not None, ent[:300])
    rf = mod(type_, ["vue", "onglet"], "ferraillage")
    ent_f, svg_f = rf["vues"][0]["entete"], rf["vues"][0]["svg"]
    print("== D1 — le plan ferraillé (onglet Ferraillage)")
    reps_plan = DF.reperes(svg_f)
    chk(f"D1 toutes les barres du bordereau sont sur le plan, avec leur repère ({len(set(reps_bd))} repères : "
        f"{', '.join(sorted(set(reps_bd), key=lambda x: (x[0], int(x[1:]))))})",
        set(reps_bd) <= set(reps_plan) and 'aria-label="Plan ferraillé de la fondation"' in svg_f, sorted(set(reps_bd) - set(reps_plan)))
    t_f = txt_svg(svg_f)
    fi, fs = fer["faces"]["inf"], fer["faces"]["sup"]
    u = fer["U"]
    chk("D1 le plan dit les treillis (« Treillis inf. Ø.. e = .. cm (x et y) – T1, T2 », « Treillis sup. … – T3, T4 », un carré de grille), les "
        "U (« U1 : Ø.. e = .. cm – .. U par bord »), les renforts de bandes en zone hachurée (une hachure pour le bas, une pour le haut) avec "
        "leur étiquette « R1 : n Ø.. inf. – L = .. m », les renforts locaux (rectangle et repère), les repères des coupes A et B, les cotes",
        txt(f"Treillis inf. Ø{fi['d']} e = {C.fr(fi['e'], 0)} cm (x et y) – T1, T2") in t_f
        and txt(f"Treillis sup. Ø{fs['d']} e = {C.fr(fs['e'], 0)} cm (x et y) – T3, T4") in t_f
        and txt(f"U1 : Ø{u['d']} e = {C.fr(u['e'], 0)} cm – {u['bords'][0]['n']} U par bord") in t_f
        and 'id="fer_hinf"' in svg_f and 'id="fer_hsup"' in svg_f and svg_f.count('class="df-r"') == sum(
            1 for c in ("inf", "sup") for b in fer["faces"][c]["bandes"].values() if b["renfort"])
        and re.search(r"R1 : \d+ Ø\d+ (inf|sup)\. – L = \d+,\d\d m", t_f) is not None
        and svg_f.count('class="df-l"') == len({ln["rep"] for ln in fer["bordereau"]["lignes"] if ln["famille"] == "L"})
        and {ln["rep"] for ln in fer["bordereau"]["lignes"] if ln["famille"] == "L"} <= set(re.findall(r"\bL\d+\b", t_f))
        and svg_f.count('data-coupe="A"') == 2 and svg_f.count('data-coupe="B"') == 2 and "8,00" in t_f, t_f[:400])
    chk("D1 la barre de l'onglet Ferraillage : les onglets et le choix du fond (sans fond, ou le taux de travail du treillis seul d'une face et "
        "d'une direction) ; plus de carte, de cas ni de k",
        re.findall(r'data-chemin="(vue/[a-z]+)"', ent_f) == ["vue/onglet"] * 3 + ["vue/fond"]
        and re.findall(r'<option value="([a-z_]*)"', ent_f) == ["", "inf_x", "inf_y", "sup_x", "sup_y"], re.findall(r'data-chemin="(vue/[a-z]+)"', ent_f))
    rfd = mod(rf["etat"], ["vue", "fond"], "sup_x")
    svg_d = rfd["vues"][0]["svg"]
    cv = fs["couverture"]["x"]
    rouges = int(((cv["taux_treillis"] > 1.0 + 1e-9) & cv["controle"]).sum())
    chk(f"D1 le fond « taux treillis sup. x » : le taux de travail du treillis SEUL (besoin ÷ treillis) peint par élément hors des zones de "
        f"renfort, la ligne noire à 100 % autour des {rouges} éléments où il ne suffit pas, la légende avec son trait noir à 100 %",
        rouges > 0 and 'class="df-100"' in svg_d and "taux treillis sup. x" in txt_svg(svg_d) and "100 % limite" in txt_svg(svg_d)
        and "data:image/png;base64," in svg_d and 'class="sf-lim"' in svg_d, (rouges, txt(svg_d)[:200]))
    print("== D2 — les coupes (onglet Coupes)")
    rc = mod(type_, ["vue", "onglet"], "coupes")
    svg_c = rc["vues"][0]["svg"]
    t_c = txt_svg(svg_c)
    reps_c = DF.reperes(svg_c)
    chk("D2 deux coupes, « Coupe A-A (le long de BX1) » et « Coupe B-B (le long de BY1) », cotées (longueur 8,00 m, h = 0,50 m, c = 5,0 cm) ; "
        "les barres des deux treillis (T1 à T4), les U aux deux bouts, les renforts des bandes coupées ; les mêmes repères que le bordereau",
        "Coupe A-A (le long de BX1)" in t_c and "Coupe B-B (le long de BY1)" in t_c and t_c.count("8,00 m") == 2
        and t_c.count("h = 0,50 m · c = 5,0 cm") == 2 and {"T1", "T2", "T3", "T4", "U1"} <= set(reps_c) and set(reps_c) <= set(reps_bd)
        and all(re.search(r'<line [^>]*data-rep="' + t_ + '"', grp) for grp, ts in ((svg_c.split('class="df-coupe-B"')[0], ("T1", "T3")),
                                                                              (svg_c.split('class="df-coupe-B"')[1], ("T2", "T4"))) for t_ in ts)
        and re.findall(r'data-chemin="(vue/[a-z]+)"', rc["vues"][0]["entete"]) == ["vue/onglet"] * 3, (reps_c, t_c[:300]))
    ec = mod(type_, ["grue", "plots", "type"], "coule")
    el8, _, fer8, _ = calcul(ec["etat"])
    r8 = mod(ec["etat"], ["vue", "onglet"], "coupes")
    t8, reps8 = txt_svg(r8["vues"][0]["svg"]), DF.reperes(r8["vues"][0]["svg"])
    chk("D2 plots coulés : la coupe montre le plot (« plot 0,35 m ») et les C de sa cage (repères C du bordereau)",
        t8.count("plot 0,35 m") == 2 and any(x.startswith("C") for x in reps8)
        and set(reps8) <= {ln["rep"] for ln in fer8["bordereau"]["lignes"]}, reps8)
    print("== le bordereau à l'écran")
    res = r["resultat"]
    t_r = txt(res)
    bd = fer["bordereau"]
    chk(f"le bordereau, dépliable dans la carte Dalle : une ligne par repère du bordereau ({len(bd['lignes'])}), les totaux par Ø, le total "
        f"({C.fr(bd['total'], 0)} kg), le béton ({C.fr(bd['beton'], 2)} m³) et le ratio",
        '<details class="sf-tkd sf-bord"' in res and res.count("<tr><td><b>") >= len(bd["lignes"])
        and txt(f"total {C.fr(bd['total'], 0)} kg · béton {C.fr(bd['beton'], 2)} m³") in t_r, t_r[t_r.find("Bordereau"):][:200])


def complets():
    print("== D3 — la note PDF")
    type_ = M._etat_type()
    el, calc, fer, ef = calcul(type_)
    note = q(action="telecharger", etat=type_, quoi="note_pdf", infos=INFOS)
    octets = base64.b64decode(note["fichier"]["base64"])
    t = pdf_texte(octets)
    bd = fer["bordereau"]
    chk("D3 la note PDF : à la suite du plan et de la vérification rapide, les planches « Efforts » (diagrammes M(s) et V(s) des 4 bandes), "
        "« Ferraillage » (le plan ferraillé, le tableau des treillis, renforts, U et contrôles du moteur poutre), « Coupes » (A-A et B-B), "
        "« Bordereau » (le bordereau et les quantités), et « Non vérifiés »",
        note["ok"] and all(x in t for x in ("F1 — Efforts", "DIAGRAMMES M(s) ET V(s)", "BX1 — M(s)", "BY2 — |V|(s)", "F1 — Ferraillage",
                                            "PLAN FERRAILLÉ", "F1 — Coupes", "Coupe A-A (le long de BX1)", "Coupe B-B (le long de BY1)",
                                            "F1 — Bordereau", "Bordereau des aciers", "Quantités", "Non vérifiés : le poinçonnement"))
        and fr(bd["total"], 1) in t and fr(bd["beton"], 2) in t, t[-800:])
    rb = ef["bandes"][0]
    chk("D3 la note dit les chiffres de l'écran : BX1 M⁺ = 71,6 et M⁻ = -63,3 kNm, l'acier total, les repères du bordereau",
        "71,6" in t and "-63,3" in t and all(r_ in t for r_ in ("T1", "T2", "T3", "T4", "U1")), None)


def fr(x, nd=1):
    return C.fr(x, nd).replace(C.NBSP, " ").replace(C.FINE, " ")


MUTATIONS = [
    ("dessin_ferraillage.py", '''            out.append(DS._texte(v.X(p[0]), v.Y(p[1]), txt, t * 0.72, COULEUR[cote], "middle", gras=True, angle=ang, attrs=f' data-rep="{rep}"'))''',
     '''            pass''', "les renforts de bandes sans étiquette sur le plan"),
    ("dessin_ferraillage.py", '''        out.append(DS._texte(v.X((x0 + x1) / 2), v.Y((y0 + y1) / 2) + t * 0.25, ln["rep"], t * 0.62, col, "middle", gras=True,''',
     '''        out.append(DS._texte(v.X((x0 + x1) / 2), v.Y((y0 + y1) / 2) + t * 0.25, "", t * 0.62, col, "middle", gras=True,''', "les renforts locaux sans repère"),
    ("dessin_ferraillage.py", '''        out.append(_ligne_100(v, ef, champ["valeurs"], ident))''', '''        pass''', "le fond sans la ligne noire à 100 %"),
    ("dessin_ferraillage.py", '''    out.append(barre(c, L - c, zz[sens], d, col, rep_l))''', '''    pass''', "les coupes sans les barres dans le sens de la coupe"),
    ("dessin_ferraillage.py", '''    for nom, lettre in (("BX1", "A"), ("BY1", "B")):
        if nom not in noms:
            continue
        b = noms[nom]''', '''    for nom, lettre in ():
        if nom not in noms:
            continue
        b = noms[nom]''', "le plan sans les repères des coupes"),
    ("saf.py", '''        svg = (DF.plan(el, calc["modele"], fer, ef, el["vue"].get("fond") or None) if onglet == "ferraillage"''',
     '''        svg = (DF.plan(el, calc["modele"], fer, ef, None) if onglet == "ferraillage"''', "le fond choisi ignoré"),
    ("note_saf.py", '''            out += NF.planches(el, calc, s, style, fer, M.efforts(el, calc), _planche, SR.non_verifies(el, calc, fer["non_verifies"]))''',
     '''            pass''', "la note sans les planches du ferraillage"),
]
TEMOIN = ("dessin_ferraillage.py", "ECH = DS.ECH", "ECH = DS.ECH")


def muter(fichier, ancien, nouveau):
    with tempfile.TemporaryDirectory() as tmp:
        for nom in os.listdir(RACINE):
            if nom in (".git", ".temoin_tests", "__pycache__", "node_modules"):
                continue
            if nom == "api_web":
                shutil.copytree(os.path.join(RACINE, nom), os.path.join(tmp, nom), ignore=shutil.ignore_patterns("__pycache__"))
            else:
                os.symlink(os.path.join(RACINE, nom), os.path.join(tmp, nom))
        cible = os.path.join(tmp, "api_web", fichier)
        src = open(cible, encoding="utf-8").read()
        if src.count(ancien) != 1:
            return False, None
        open(cible, "w", encoding="utf-8").write(src.replace(ancien, nouveau))
        args = [sys.executable, os.path.abspath(__file__), "--racine", tmp] + (["--rapide"] if fichier != "note_saf.py" else ["--sans-mutations"])
        p = subprocess.run(args, capture_output=True, text=True, cwd=RACINE)
        return True, p.returncode


def main():
    rapides()
    if RAPIDE:
        print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
        return 1 if KO else 0
    complets()
    if "--sans-mutations" not in ARGS:
        print("== mutations")
        trouve, code = muter(*TEMOIN)
        chk("témoin : l'écran recopié sans rien changer passe les contrôles", trouve and code == 0, code)
        for fichier, ancien, nouveau, nom in MUTATIONS:
            trouve, code = muter(fichier, ancien, nouveau)
            chk(f"mutation « {nom} » ({fichier}) : un contrôle rougit", trouve and code == 1,
                "texte absent ou répété" if not trouve else f"code {code} : AUCUN contrôle ne rougit")
    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
    for nom, info in KO:
        print("   -", nom[:160], "|", str(info)[:300])
    return 1 if KO else 0


if __name__ == "__main__":
    sys.exit(main())
