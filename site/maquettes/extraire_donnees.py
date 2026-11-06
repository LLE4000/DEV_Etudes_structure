# -*- coding: utf-8 -*-
"""Maquettes du site « Stab Méthodes » : extrait du MOTEUR les vrais dessins (SVG) et les vrais
chiffres que les maquettes affichent — rien n'y est inventé : le garde-corps par défaut, ses six
vues, ses éléments, le tableau des profilés du montant ; une vue de soudure et un plan de
charges sur plancher pour les vignettes de l'accueil.

Lancement : python3 site/maquettes/extraire_donnees.py (depuis la racine) → site/maquettes/donnees.json
"""
import base64
import re
import copy
import json
import os
import subprocess
import sys
import tempfile

RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RACINE)
os.chdir(RACINE)

from acier.garde_corps import bibliotheque as BI  # noqa: E402
from acier.garde_corps import charges as CH  # noqa: E402
from acier.garde_corps import etat as ET  # noqa: E402
from acier.garde_corps import formules as FO  # noqa: E402
from acier.garde_corps import membres as ME  # noqa: E402
from acier.garde_corps import note as NO  # noqa: E402
from acier.garde_corps import schemas as SC  # noqa: E402
from acier.soudures import etat as SET  # noqa: E402
from acier.soudures import schemas as SSC  # noqa: E402
from plancher.charges import dessin as PDS  # noqa: E402
from plancher.charges import etat as PET  # noqa: E402
from plancher.charges import verification as PV  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dessin_poutre as DP  # noqa: E402

SORTIE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "donnees.json")


def niveau(t):
    """Le code couleur du bureau : r (> 100 %), f (95 à 100 %), p (en dessous) — par le statut du moteur."""
    return {ME.NOK: "r", ME.ATTENTION: "f", ME.OK: "p"}.get(ME.statut(t), "p")


def pct(t: float, nd: int = 0):
    """Un taux à afficher, arrondi au plus proche SANS quitter la bande de sa couleur (le statut du moteur décide) :
    94,98 % s'affiche 94 % (95 % se lirait vert foncé) ; 100,2 % s'affiche 101 % (100 % se lirait admis) ; 63,7 % reste 64 %."""
    import math
    x, n, k = t * 100, niveau(t), 10 ** nd
    v = math.floor(x * k + 0.5) / k
    if n == "r":
        v = max(v, 100 + 1 / k)
    elif n == "f":
        v = min(max(v, 95.0), 100.0)
    else:
        v = min(v, 95 - 1 / k)
    return int(v) if nd == 0 else round(v, nd)


def elements(res):
    out = []
    for e in res["elements"]:
        parts = e["libelle"].split(" · ")
        d = e["detail"] if isinstance(e.get("detail"), dict) else {}
        cas_c = CH.CAS_LIBELLES.get(d.get("cas_charge"), "") if d.get("cas_charge") else ""
        dim = e["cle"] in ET.ROLES
        out.append({"cle": e["cle"], "nom": parts[0], "section": parts[1] if dim and len(parts) > 1 else "",
                    "desc": " · ".join(parts[2:] if dim else parts[1:]),
                    "critere": ME.critere_texte(e["critere"]) if dim else e["critere"], "cas": cas_c,
                    "taux": pct(e["taux"]), "niveau": niveau(e["taux"])})
    return out


def selecteur(cas, role, groupes, fams):
    """Le sélecteur de profilé d'un élément : les candidats du moteur TRIÉS PAR MASSE (l'ordre de grandeur), le retenu et
    l'optimal (le plus léger qui tient) marqués, chaque taux avec son niveau ; une fenêtre qui commence un peu avant
    l'optimal et va bien au-delà du retenu, pour la molette."""
    cands = sorted((c for c in ET.candidats(cas, role, groupes, 15.0) if c["section"]["famille"].replace("-F", "") in fams),
                   key=lambda c: c["masse"])
    best = ET.plus_leger(cands)
    cur = ET.spec_du_role(cas, role).get("nom")
    noms = [c["section"]["nom"] for c in cands]
    i_o = noms.index(best["section"]["nom"]) if best else 0
    i_c = noms.index(cur) if cur in noms else i_o
    d, f = max(0, min(i_o, i_c) - 6), min(len(cands), max(i_o, i_c) + 16)
    def prop(sec):
        fam = sec["famille"]
        norme = ("EN 10219 · formé à froid" if fam.endswith("-F") else "EN 10210 · fini à chaud") if fam.replace("-F", "") in ("RHS", "SHS", "CHS") else "barre pleine"
        vals = [f"A = {fr(sec['A'], 2)} cm²"] + ([f"I<sub>y</sub> = {fr(sec['Iy'], 2)} cm⁴"] if sec.get("Iy") else []) \
            + ([f"W<sub>el,y</sub> = {fr(sec['Wel_y'], 2)} cm³"] if sec.get("Wel_y") else [])
        return " · ".join([norme] + vals)
    return [{"nom": c["section"]["nom"], "kgm": round(c["masse"], 2), "eta": pct(min(c["taux"], 9.99)), "prop": prop(c["section"]),
             "niveau": niveau(c["taux"]), "retenue": c["section"]["nom"] == cur,
             "optimal": best is not None and c["section"]["nom"] == best["section"]["nom"]} for c in cands[d:f]]


def blocs_gc(res):
    """Les blocs de formules du moteur, par élément ; la soudure (groupe, point, critère) est mise au même format."""
    sd = FO.soudure(res)
    soud = {"cas": "", "autres": "", "groupes": [(t, sd.get(k, [])) for k, t in (("groupe", "Groupe de cordons"), ("point", "Point le plus sollicité"),
                                                                                    ("critere", "Critère"))]}
    soud["subs"] = [x for _, xs in soud["groupes"] for x in xs]
    return {"montant": FO.montant(res), "lisse": FO.lisse(res), "barreau": FO.barre(res), "platine": FO.platine(res), "soudure": soud}


def calculs(res):
    """« Voir le calcul » : les formules du moteur, groupe par groupe, telles que la note les imprime (FO.morceaux) ;
    chaque taux porte son niveau pour la couleur. Rien n'est recalculé ici."""
    out = {}
    for cle, b in blocs_gc(res).items():
        if not b["groupes"]:
            continue
        groupes = []
        for lib, subs in b["groupes"]:
            lignes = []
            for s in subs:
                parts, r = FO.morceaux(s)
                lignes.append({"parts": parts, "res": r, "niveau": niveau(s.res) if s.nom.startswith("η") else ""})
            groupes.append({"titre": lib, "lignes": lignes})
        out[cle] = {"cas": b["cas"], "autres": b["autres"], "groupes": groupes}
    return out


def apercu_note(cas):
    """La première page de la VRAIE note (moteur → PDF → image) : l'aperçu d'un indice périmé, sous son filigrane."""
    pdf = NO.generer_note(ET.nouvel_etat(), cas, {"nom_projet": "Réalex", "partie": "Garde-corps terrasse R+1", "indice": "A"})
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "note.pdf")
        with open(src, "wb") as f:
            f.write(pdf)
        subprocess.run(["pdftoppm", "-png", "-r", "36", "-f", "1", "-l", "1", "-singlefile", src, os.path.join(tmp, "p1")], check=True)
        with open(os.path.join(tmp, "p1.png"), "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


def garde_corps(cas, prefixe):
    res = ET.verifier(cas)
    vues = [{"cle": k, "titre": t, "svg": SC.svg(d, f"{prefixe}_{k}"), "mini": SC.svg(d, f"{prefixe}_m_{k}")} for k, t, d in SC.vues(res)]
    g = res["gouvernant"]
    # le tableau des profilés du montant, tubes formés à froid compris : les plus légers ne tiennent pas (rouge), puis la bande
    # 95–100 % (vert foncé), puis ceux qui tiennent largement (vert pâle) — une fenêtre de quatorze lignes autour de l'optimal
    cands = [c for c in ET.candidats(cas, "montant", ["creux", "froid"], 15.0) if c["section"]["famille"].replace("-F", "") == "RHS"]
    best = ET.plus_leger(cands)
    cur = cas["montant"]["section"]["nom"]
    i0 = max(0, (cands.index(best) if best in cands else 0) - 5)
    table = [{"nom": c["section"]["nom"], "dims": BI.dimensions_texte(c["section"]), "kgm": round(c["masse"], 2),
              "eq": pct(min(c["taux_eq"], 9.99)), "fl": pct(min(c["taux_fleche"], 9.99)),
              "eta": pct(min(c["taux"], 9.99)), "retenue": c["section"]["nom"] == cur,
              # la couleur de chaque cellule vient du statut du moteur : l'écran ne recopie aucun seuil
              "n_eq": niveau(c["taux_eq"]), "n_fl": niveau(c["taux_fleche"]), "n_eta": niveau(c["taux"]),
              "optimal": best is not None and c["section"]["nom"] == best["section"]["nom"]} for c in cands[i0:i0 + 14]]
    # l'optimal proposé par le bouton « ★ » de chaque élément, avec les familles cochées (comme l'écran Streamlit) ; pour le
    # montant, les réglages de la feuille des profilés ci-dessus (« Formés à froid » coché) : la ligne et la feuille disent le même optimal
    defaut = {"montant": (["creux", "froid"], ("RHS",)), "lisse": (["creux"], ("RHS",)), "barreau": (["plats", "ronds"], ("PLAT", "ROND"))}
    optimaux = {}
    for role, (groupes, fams) in defaut.items():
        cc = [c for c in ET.candidats(cas, role, groupes, 15.0) if c["section"]["famille"].replace("-F", "") in fams]
        b = ET.plus_leger(cc)
        if b is not None:
            optimaux[role] = {"nom": b["section"]["nom"], "taux": pct(b["taux"]), "niveau": niveau(b["taux"]),
                              "retenu": b["section"]["nom"] == ET.spec_du_role(cas, role).get("nom")}
    selecteurs = {role: selecteur(cas, role, groupes, fams) for role, (groupes, fams) in defaut.items()}
    return {"eta": pct(res["taux_max"]), "niveau": niveau(res["taux_max"]), "statut": res["statut"], "optimaux": optimaux,
            "selecteurs": selecteurs, "calculs": calculs(res),
            "gouvernant": g["libelle"], "critere": ME.critere_texte(g["critere"]), "phrase": res["geometrie"]["phrase"],
            "type": res["type_libelle"], "masse": round(res["masse"]), "elements": elements(res), "vues": vues,
            "table_montant": table, "optimal_montant": {"nom": best["section"]["nom"], "taux": pct(best["taux"])} if best else None}


def fr(x: float, nd: int = 1) -> str:
    """Un nombre à la française (virgule, espace fine des milliers)."""
    s = f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")
    return s


def tableau_gc(res):
    """La grande fenêtre de calcul du garde-corps : une ligne par vérification (valeur, limite, taux), lue dans les
    formules du moteur — chaque taux η = valeur / limite y porte ses deux termes (Sub.vals)."""
    noms = {"η_eq": "Contrainte équivalente", "η_τ": "Cisaillement", "η_f": "Flèche", "η_t": "Ancrages · traction",
            "η_inter": "Ancrages · traction et cisaillement", "η_p": "Platine · flexion"}
    unites = {"σ": "N/mm²", "τ": "N/mm²", "f": "mm", "F": "kN", "N": "kN", "V": "kN", "|": "N/mm"}
    out = {}
    for cle, b in blocs_gc(res).items():
        subs = b["subs"]
        defs = {s.nom: s for s in subs}
        lignes = []
        for s in subs:
            nom = noms.get(s.nom) or ("Cordons de soudure" if cle == "soudure" and s.nom == "η" else None)
            if not nom or not s.vals:
                continue
            (sv, vv), (sl, vl) = list(s.vals.values())[:2]
            d = defs.get(sv)
            u = (d.unit if d is not None and d.unit else unites.get(sv[0], ""))
            simple = re.fullmatch(r"\w+/\w+", s.template or "") is not None
            formule = (FO.morceaux(d)[0][1] if d is not None and len(FO.morceaux(d)[0]) > 1 else "") if simple else FO.morceaux(s)[0][1]
            lignes.append({"verif": nom, "sym": sv if simple else s.nom, "formule": formule,
                           "valeur": f"{fr(vv, 2 if abs(vv) < 10 else 1)} {u}".strip() if simple else fr(s.res, 2),
                           "sym_lim": sl if simple else "", "limite": f"{fr(vl, 1)} {u}".strip() if simple else "1,00",
                           "taux": pct(s.res, 1), "niveau": niveau(s.res)})
        out[cle] = sorted(lignes, key=lambda li: -li["taux"])
    return out


def poutre_ba():
    """Le module « Poutres béton armé » de l'écran Streamlit (modules/poutre.py, v2.40), piloté SANS écran : son état
    est rempli en mémoire et ses fonctions de calcul rendent les résultats. Les valeurs d'entrée sont un exemple (une
    poutre 30 × 50 : une section en travée, une sur appui) ; les résultats sont ceux du module.

    Deux grandeurs que le module calcule dans son code d'ÉCRAN (τ et le pas théorique s_th) sont recalculées ici avec
    ses formules (render_dimensionnement_section) ; l'état du module (etat_tau, etat_pas) les contre-vérifie ci-dessous.
    C'est la séparation que le site devra faire d'abord : un moteur poutre, avec ses formules nommées et testées."""
    import logging
    import math
    logging.disable(logging.WARNING)
    import streamlit as st
    import modules.poutre as P
    with open(os.path.join(RACINE, "beton_classes.json"), encoding="utf-8") as f:
        P.BETON_DATA = json.load(f)
    KB, KS = P.KB, P.KS

    def charger(poutre):
        st.session_state.clear()
        P._ensure_global_defaults()
        P._init_beams_if_needed()
        st.session_state.beams[0]["sections"] = [{"id": i + 1, "nom": s["nom"]} for i, s in enumerate(poutre["sections"])]
        P._ensure_defaults_for_beam(1)
        if "jeu_entre_lits_cm" in poutre:
            st.session_state["jeu_entre_lits_cm"] = poutre["jeu_entre_lits_cm"]
        for k in ("b", "h", "beton", "fyk", "enrobage_beton"):
            st.session_state[KB(k, 1)] = poutre[k]
        for i, s in enumerate(poutre["sections"]):
            sid = i + 1
            for k in ("M_inf", "M_sup", "V"):
                st.session_state[KS(k, 1, sid)] = float(s[k])
            for face in ("inf", "sup"):
                lits = s[face]
                st.session_state[KS(f"nlits_{face}", 1, sid)] = len(lits)
                for j, (n, d) in enumerate(lits, start=1):
                    suf = "" if j == 1 else f"_l{j}"
                    st.session_state[KS(f"n_as_{face}{suf}", 1, sid)] = n
                    st.session_state[KS(f"ø_as_{face}{suf}", 1, sid)] = d
            st.session_state[KS("shear_line0_d", 1, sid)] = s["etrier"][0]
            st.session_state[KS("shear_pas", 1, sid)] = float(s["etrier"][1])

    def verifier(poutre, sid):
        """Les cinq vérifications d'une section, comme l'écran Streamlit les affiche (mêmes taux)."""
        r = P._dimensionnement_compute_states(1, sid, P.BETON_DATA)
        s = poutre["sections"][sid - 1]
        b, h = r["b"], r["h"]
        out = []
        # 1 · hauteur : h_min = h_u,min + y_G de la face la plus exigeante ; taux h_min / h
        if r["M_max"] > 0:
            f1 = (f"h<sub>u,min</sub> = √(M / (α<sub>b</sub>·b·μ)) = √({fr(r['M_max'], 0)} × 10⁶ / ({fr(r['alpha_b'], 2)} × {fr(b * 10, 0)} × "
                  f"{fr(r['mu_val'], 4)})) = {fr(r['hmin_calc'], 1)} cm")
            f2 = f"h<sub>min</sub> = h<sub>u,min</sub> + y<sub>G</sub> = {fr(r['hmin_calc'], 1)} + {fr(r['e_cdg_gov'], 1)} = {fr(r['h_min_poutre'], 1)} cm"
            out.append({"cle": "hauteur", "nom": "Hauteur", "spec": f"{fr(b, 0)} × {fr(h, 0)} cm", "formules": [f1, f2],
                        "lit": "h<sub>min</sub> = h<sub>u,min</sub> + y<sub>G</sub> ≤ h",
                        "valeur": f"h<sub>min</sub> = {fr(r['h_min_poutre'], 1)} cm", "limite": f"h = {fr(h, 0)} cm",
                        "t": r["h_min_poutre"] / h, "etat": r["etat_h"]})
        # 2-3 · armatures : besoin = max(A_s,req ; A_s,min) ; taux besoin / A_s
        for face, nom, opp in (("inf", "Armatures inférieures", "sup"), ("sup", "Armatures supérieures", "inf")):
            M = r[f"M_{face}_val"]
            d = r[f"d_utile_{face}"]
            req, mn, tot = r[f"As_req_{face}_final"], r[f"As_min_{face}_eff"], r[f"As_{face}_total"]
            fl = []
            if M > 0:
                fl.append(f"A<sub>s,req</sub> = M / (f<sub>yd</sub>·0,9·d) = {fr(M, 0)} × 10⁶ / ({fr(r['fyd'], 1)} × 0,9 × {fr(d * 10, 0)}) = {fr(req, 0)} mm²")
            fl.append(f"A<sub>s,min</sub> = max(0,26·f<sub>ctm</sub>/f<sub>yk</sub>·b·h ; 0,0013·b·h ; 0,25·A<sub>s,req,{opp}</sub>) = "
                      f"max({fr(r['As_min_ec'], 0)} ; {fr(r['As_min_plancher'], 0)} ; {fr(0.25 * r[f'As_formule_{opp}'], 0)}) = {fr(mn, 0)} mm²")
            besoin = max(req, mn)
            out.append({"cle": face, "nom": nom, "spec": f"{r[f'{face}_detail']} · {fr(tot, 0)} mm²", "formules": fl,
                        "lit": "max(A<sub>s,req</sub> ; A<sub>s,min</sub>) ≤ A<sub>s</sub>",
                        "valeur": f"{fr(besoin, 0)} mm²", "limite": f"A<sub>s</sub> = {fr(tot, 0)} mm²", "t": besoin / tot,
                        "etat": r[f"etat_{face}"]})
        # 4-5 · effort tranchant et étriers (formules du module, contre-vérifiées par ses états)
        V = r["V_val"]
        if V > 0:
            tau = V * 1e3 / (0.75 * b * h * 100)
            bandes = [(r["tau_1"], "τ<sub>adm,I</sub>", "0,016", "pas besoin d'étriers"), (r["tau_2"], "τ<sub>adm,II</sub>", "0,032", "besoin d'étriers"),
                      (r["tau_4"], "τ<sub>adm,IV</sub>", "0,064", "barres inclinées et étriers")]
            lim, nom_lim, k, besoin_txt = next((bd for bd in bandes if tau <= bd[0]), (r["tau_4"], "τ<sub>adm,IV</sub>", "0,064", "pas acceptable"))
            assert (tau <= r["tau_2"]) == (r["etat_tau"] == "ok"), "τ recalculé ≠ état du module"
            fck_cube = P.BETON_DATA[r["beton"]]["fck_cube"]
            out.append({"cle": "tau", "nom": "Effort tranchant", "spec": f"V = {fr(V, 0)} kN · {besoin_txt}", "lit": "τ = V / (0,75·b·h) ≤ τ<sub>adm</sub>",
                        "formules": [f"τ = V / (0,75·b·h) = {fr(V, 0)} × 10³ / (0,75 × {fr(b * 10, 0)} × {fr(h * 10, 0)}) = {fr(tau, 2)} N/mm²",
                                     f"{nom_lim} = {k}·f<sub>ck,cube</sub> / 1,05 = {k} × {fr(fck_cube, 0)} / 1,05 = {fr(lim, 2)} N/mm²"],
                        "valeur": f"τ = {fr(tau, 2)} N/mm²", "limite": f"{nom_lim} = {fr(lim, 2)} N/mm²", "t": tau / lim, "etat": r["etat_tau"]})
            ast = P._shear_lines_total_Ast_mm2(1, sid)
            d_sh = r["d_utile_shear"]
            pas = float(s["etrier"][1])
            pas_th = ast * r["fyd"] * (d_sh * 10.0) / (V * 1e3) / 10.0
            s_max = min(0.75 * d_sh, 30.0)
            pas_lim = min(pas_th, s_max)
            assert (pas <= pas_lim) == (r["etat_pas"] == "ok"), "pas recalculé ≠ état du module"
            out.append({"cle": "pas", "nom": "Étriers", "spec": f"{P._shear_lines_summary(1, sid)} · e = {fr(pas, 0)} cm",
                        "lit": "e ≤ min(s<sub>th</sub> ; s<sub>max</sub>)",
                        "formules": [f"s<sub>th</sub> = A<sub>st</sub>·f<sub>yd</sub>·d / V = {fr(ast, 1)} × {fr(r['fyd'], 1)} × {fr(d_sh * 10, 0)} / ({fr(V, 0)} × 10³) = "
                                     f"{fr(math.floor(pas_th * 10) / 10, 1)} cm",
                                     f"s<sub>max</sub> = min(0,75·d ; 30 cm) = {fr(math.floor(s_max * 10) / 10, 1)} cm"],
                        "valeur": f"e = {fr(pas, 0)} cm", "limite": f"s<sub>lim</sub> = {fr(math.floor(pas_lim * 10) / 10, 1)} cm", "t": pas / pas_lim,
                        "etat": r["etat_pas"]})
        for v in out:
            v["taux"] = pct(v["t"])
            v["niveau"] = niveau(v["t"])
        lits = {face: [(n, d, P._get_dist_lit(1, sid, face, j)) for j, (n, d) in enumerate(s[face], start=1)] for face in ("inf", "sup")}
        return r, out, lits

    v1 = {"nom": "V1", "b": 30, "h": 50, "beton": "C30/37", "fyk": 500, "enrobage_beton": 3.0, "jeu_entre_lits_cm": 1.5, "sections": [
        {"nom": "A", "role": "travée", "M_inf": 122, "M_sup": 0, "V": 60, "inf": [(3, 20)], "sup": [(2, 14)], "etrier": (8, 20)},
        {"nom": "B", "role": "appui", "M_inf": 0, "M_sup": 95, "V": 120, "inf": [(2, 20)], "sup": [(4, 16)], "etrier": (8, 10)}]}
    charger(v1)
    sections, coupes = [], []
    for i, s in enumerate(v1["sections"]):
        r, verifs, lits = verifier(v1, i + 1)
        t = max(v["t"] for v in verifs)
        sections.append({"nom": s["nom"], "role": s["role"], "M_inf": s["M_inf"], "M_sup": s["M_sup"], "V": s["V"],
                         "inf": r["inf_detail"], "sup": r["sup_detail"], "etrier": f"{P._shear_lines_summary(1, i + 1)} · e = {s['etrier'][1]} cm",
                         "verifs": verifs, "eta": pct(t), "niveau": niveau(t)})
        M_txt = " · ".join(x for x in (f"M<tspan font-size='70%' dy='4'>inf</tspan><tspan dy='-4'> = {fr(s['M_inf'], 0)} kN·m</tspan>" if s["M_inf"] else "",
                                       f"M<tspan font-size='70%' dy='4'>sup</tspan><tspan dy='-4'> = {fr(s['M_sup'], 0)} kN·m</tspan>" if s["M_sup"] else "",
                                       f"V = {fr(s['V'], 0)} kN") if x)
        coupes.append({"nom": f"Section {s['nom']} · {s['role']}", "b": v1["b"], "h": v1["h"], "c": v1["enrobage_beton"], "d_et": s["etrier"][0],
                       "inf": lits["inf"], "sup": lits["sup"], "etrier": f"étrier Ø{s['etrier'][0]} · e = {s['etrier'][1]} cm", "efforts": M_txt,
                       # la face tendue : sur appui, le moment négatif tend le haut, et d se mesure depuis le bas
                       "tendu": "sup" if s["M_sup"] > s["M_inf"] else "inf"})
    r0 = P._dimensionnement_compute_states(1, 1, P.BETON_DATA)
    hyp = {"beton": r0["beton"], "fck": P.BETON_DATA[r0["beton"]]["fck"], "fck_cube": P.BETON_DATA[r0["beton"]]["fck_cube"], "fyk": r0["fyk"],
           "gamma_s": r0["gamma_s"], "fyd": round(r0["fyd"], 1), "alpha_b": r0["alpha_b"], "mu": r0["mu_val"], "fctm": round(r0["fctm"], 2),
           "b": v1["b"], "h": v1["h"], "c": v1["enrobage_beton"]}
    # le sélecteur d'armatures de la section A, face inférieure : chaque disposition passe par le module
    cands = []
    for d in (12, 14, 16, 20, 25):
        for n in (2, 3, 4, 5):
            essai = copy.deepcopy(v1)
            essai["sections"][0]["inf"] = [(n, d)]
            charger(essai)
            r = P._dimensionnement_compute_states(1, 1, P.BETON_DATA)
            t = max(r["As_req_inf_final"], r["As_min_inf_eff"]) / r["As_inf_total"]
            cands.append({"nom": f"{n}Ø{d}", "aire": round(r["As_inf_total"]), "eta": pct(min(t, 9.99)), "niveau": niveau(t),
                          "ok": r["etat_inf"] == "ok", "retenue": (n, d) == (3, 20)})
    cands.sort(key=lambda c: (c["aire"], c["nom"]))
    best = next((c for c in cands if c["ok"] and c["eta"] <= 100), None)
    for c in cands:
        c["optimal"] = c is best
    # V2 : une poutre qui ne tient pas (même module) ; V3 : sollicitations pas encore saisies
    v2 = {"nom": "V2", "b": 25, "h": 45, "beton": "C25/30", "fyk": 500, "enrobage_beton": 3.0, "sections": [
        {"nom": "A", "role": "travée", "M_inf": 150, "M_sup": 0, "V": 100, "inf": [(4, 20)], "sup": [(2, 12)], "etrier": (8, 15)}]}
    charger(v2)
    _, verifs2, _ = verifier(v2, 1)
    t2 = max(v["t"] for v in verifs2)
    t1 = max(v["t"] for s_ in sections for v in s_["verifs"])
    gov = max(((s_["nom"], v) for s_ in sections for v in s_["verifs"]), key=lambda x: x[1]["t"])
    # les valeurs par défaut du module (la page Paramètres les montre comme valeurs du bureau)
    st.session_state.clear()
    P._ensure_global_defaults()
    P._init_beams_if_needed()
    P._ensure_defaults_for_beam(1)
    defauts = {"gamma_s": st.session_state["gamma_s"], "enrobage": st.session_state[KB("enrobage_beton", 1)], "beton": st.session_state[KB("beton", 1)],
               "fyk": st.session_state[KB("fyk", 1)], "jeu_lits": st.session_state["jeu_entre_lits_cm"], "jeu_1": st.session_state["jeu_enrobage_cm"],
               "techno": f"Ø{st.session_state['techno_d_mm']} · {fr(st.session_state['techno_s_max_cm'], 0)} cm", "unites": f"{st.session_state['units_len']} · {st.session_state['units_as']}"}
    return {"hyp": hyp, "sections": sections, "eta": pct(t1), "niveau": niveau(t1), "gouvernant": f"Section {gov[0]} · {gov[1]['nom'].lower()}",
            "svg": {"A": DP.svg("pba_A", coupes[:1]), "B": DP.svg("pba_B", coupes[1:]), "AB": DP.svg("pba_AB", coupes)},
            "selecteur": cands, "v2": {"eta": pct(min(t2, 9.99)), "niveau": niveau(t2)}, "defauts": defauts, "version": P.APP_VERSION}


def main():
    etat = ET.nouvel_etat()
    cas = ET.cas_actif(etat)
    # les seuils du code couleur, lus dans le moteur pour la légende (aucune copie à l'écran)
    donnees = {"seuils": {"attention": round(ME.SEUIL_ATTENTION * 100), "nok": round(ME.SEUIL_NOK * 100)},
               "gc": garde_corps(cas, "gc")}
    donnees["note_page"] = apercu_note(cas)
    donnees["gc"]["tableau"] = tableau_gc(ET.verifier(cas))
    donnees["gc"]["fy"] = ET.verifier(cas)["acier"]["fy"]
    donnees["gc_defauts"] = {k: cas["methode"][k] for k in ("mode", "gamma_adm", "fleche_val", "nuance", "E", "gamma_M0", "gamma_M2")}
    donnees["poutre"] = poutre_ba()
    # une variante plus chargée pour les vignettes : catégorie C5 (non conforme) et une hauteur qui met le montant entre 95 et 100 %
    c5 = copy.deepcopy(cas)
    c5["charges"]["categorie"] = "C5"
    c5["charges"].update(CH.normaliser(c5["charges"]))
    donnees["gc_c5"] = {"eta": pct(ET.verifier(c5)["taux_max"]), "niveau": niveau(ET.verifier(c5)["taux_max"])}
    haut = copy.deepcopy(cas)
    for H in range(1150, 1400, 5):
        haut["geometrie"]["H"] = float(H)
        r = ET.verifier(haut)
        if niveau(r["taux_max"]) == "f":
            donnees["gc_haut"] = {"H": H, "eta": pct(r["taux_max"]), "niveau": niveau(r["taux_max"])}
            break
    # des variantes réelles pour les vignettes et la liste des rapports : rampant, lisses en L, platine latérale, flèche L/200
    def variante(nom, modifier, vue):
        c = copy.deepcopy(cas)
        modifier(c)
        r = ET.verifier(c)
        vv = {k: d for k, _, d in SC.vues(r)}
        donnees[nom] = {"eta": pct(r["taux_max"]), "niveau": niveau(r["taux_max"]),
                        "svg": SC.svg(vv[vue], f"{nom}_{vue}") if vue in vv else ""}
    variante("gc_rampant", lambda c: c["geometrie"].update({"pente": 32.0}), "elevation")
    def _lisses_L(c):
        from acier.garde_corps import types as TY
        c["type"] = "lisses"
        TY.appliquer_type(c, "lisses")
        c["geometrie"]["forme"] = "L"
    variante("gc_L", _lisses_L, "plan")
    variante("gc_lat", lambda c: c["fixation"].update({"type": "laterale"}), "platine")
    variante("gc_l200", lambda c: c["methode"].update({"fleche_val": 200.0}), "elevation")
    # soudures : la vue principale du cas par défaut
    se = SET.nouvel_etat()
    sres = SET.verifier(SET.cas_actif(se))
    sv = SSC.vues(sres)
    donnees["soudure"] = {"svg": SSC.svg(sv[0][2], "sd_m0") if sv else "", "eta": pct(sres.get("taux_max") or 0),
                          "niveau": niveau(sres.get("taux_max") or 0)}
    # charges sur plancher : le plan du cas par défaut
    pe = PET.nouvel_etat()
    pc = PET.cas_actif(pe)
    plancher, charges, reps, params, index = PET.construire(pe, pc)
    pres = PV.verifier(plancher, charges, reps, params)
    ctx = PET.contexte_dessin(pe, pc, pres, plancher, index)
    psvg, _, _ = PDS.plan(ctx)
    t = pres.get("taux_max") if isinstance(pres, dict) else None
    donnees["plancher"] = {"svg": psvg, "eta": pct(t or 0), "niveau": niveau(t or 0)}
    with open(SORTIE, "w", encoding="utf-8") as f:
        json.dump(donnees, f, ensure_ascii=False)
    print(f"{SORTIE} : {os.path.getsize(SORTIE) / 1024:.0f} ko · garde-corps η {donnees['gc']['eta']} % · {len(donnees['gc']['vues'])} vues · "
          f"{len(donnees['gc']['table_montant'])} lignes · C5 {donnees['gc_c5']['eta']} % · haut {donnees.get('gc_haut')} · "
          f"soudure {donnees['soudure']['eta']} % · plancher {donnees['plancher']['eta']} %")


if __name__ == "__main__":
    main()
