# -*- coding: utf-8 -*-
"""Maquettes du site « Stab Méthodes » : extrait du MOTEUR les vrais dessins (SVG) et les vrais
chiffres que les maquettes affichent — rien n'y est inventé : le garde-corps par défaut, ses six
vues, ses éléments, le tableau des profilés du montant ; une vue de soudure et un plan de
charges sur plancher pour les vignettes de l'accueil.

Lancement : python3 site/maquettes/extraire_donnees.py (depuis la racine) → site/maquettes/donnees.json
"""
import copy
import json
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RACINE)
os.chdir(RACINE)

from acier.garde_corps import bibliotheque as BI  # noqa: E402
from acier.garde_corps import charges as CH  # noqa: E402
from acier.garde_corps import etat as ET  # noqa: E402
from acier.garde_corps import membres as ME  # noqa: E402
from acier.garde_corps import schemas as SC  # noqa: E402
from acier.soudures import etat as SET  # noqa: E402
from acier.soudures import schemas as SSC  # noqa: E402
from plancher.charges import dessin as PDS  # noqa: E402
from plancher.charges import etat as PET  # noqa: E402
from plancher.charges import verification as PV  # noqa: E402

SORTIE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "donnees.json")


def niveau(t):
    """Le code couleur du bureau : r (> 100 %), f (95 à 100 %), p (en dessous) — par le statut du moteur."""
    return {ME.NOK: "r", ME.ATTENTION: "f", ME.OK: "p"}.get(ME.statut(t), "p")


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
                    "taux": round(e["taux"] * 100), "niveau": niveau(e["taux"])})
    return out


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
              "eq": round(min(c["taux_eq"], 9.99) * 100), "fl": round(min(c["taux_fleche"], 9.99) * 100),
              "eta": round(min(c["taux"], 9.99) * 100), "retenue": c["section"]["nom"] == cur,
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
            optimaux[role] = {"nom": b["section"]["nom"], "taux": round(b["taux"] * 100), "niveau": niveau(b["taux"]),
                              "retenu": b["section"]["nom"] == ET.spec_du_role(cas, role).get("nom")}
    return {"eta": round(res["taux_max"] * 100), "niveau": niveau(res["taux_max"]), "statut": res["statut"], "optimaux": optimaux,
            "gouvernant": g["libelle"], "critere": ME.critere_texte(g["critere"]), "phrase": res["geometrie"]["phrase"],
            "type": res["type_libelle"], "masse": round(res["masse"]), "elements": elements(res), "vues": vues,
            "table_montant": table, "optimal_montant": {"nom": best["section"]["nom"], "taux": round(best["taux"] * 100)} if best else None}


def main():
    etat = ET.nouvel_etat()
    cas = ET.cas_actif(etat)
    # les seuils du code couleur, lus dans le moteur pour la légende (aucune copie à l'écran)
    donnees = {"seuils": {"attention": round(ME.SEUIL_ATTENTION * 100), "nok": round(ME.SEUIL_NOK * 100)},
               "gc": garde_corps(cas, "gc")}
    # une variante plus chargée pour les vignettes : catégorie C5 (non conforme) et une hauteur qui met le montant entre 95 et 100 %
    c5 = copy.deepcopy(cas)
    c5["charges"]["categorie"] = "C5"
    c5["charges"].update(CH.normaliser(c5["charges"]))
    donnees["gc_c5"] = {"eta": round(ET.verifier(c5)["taux_max"] * 100), "niveau": niveau(ET.verifier(c5)["taux_max"])}
    haut = copy.deepcopy(cas)
    for H in range(1150, 1400, 5):
        haut["geometrie"]["H"] = float(H)
        r = ET.verifier(haut)
        if niveau(r["taux_max"]) == "f":
            donnees["gc_haut"] = {"H": H, "eta": round(r["taux_max"] * 100), "niveau": niveau(r["taux_max"])}
            break
    # des variantes réelles pour les vignettes et la liste des rapports : rampant, lisses en L, platine latérale, flèche L/200
    def variante(nom, modifier, vue):
        c = copy.deepcopy(cas)
        modifier(c)
        r = ET.verifier(c)
        vv = {k: d for k, _, d in SC.vues(r)}
        donnees[nom] = {"eta": round(r["taux_max"] * 100), "niveau": niveau(r["taux_max"]),
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
    donnees["soudure"] = {"svg": SSC.svg(sv[0][2], "sd_m0") if sv else "", "eta": round((sres.get("taux_max") or 0) * 100),
                          "niveau": niveau(sres.get("taux_max") or 0)}
    # charges sur plancher : le plan du cas par défaut
    pe = PET.nouvel_etat()
    pc = PET.cas_actif(pe)
    plancher, charges, reps, params, index = PET.construire(pe, pc)
    pres = PV.verifier(plancher, charges, reps, params)
    ctx = PET.contexte_dessin(pe, pc, pres, plancher, index)
    psvg, _, _ = PDS.plan(ctx)
    t = pres.get("taux_max") if isinstance(pres, dict) else None
    donnees["plancher"] = {"svg": psvg, "eta": round((t or 0) * 100), "niveau": niveau(t or 0)}
    with open(SORTIE, "w", encoding="utf-8") as f:
        json.dump(donnees, f, ensure_ascii=False)
    print(f"{SORTIE} : {os.path.getsize(SORTIE) / 1024:.0f} ko · garde-corps η {donnees['gc']['eta']} % · {len(donnees['gc']['vues'])} vues · "
          f"{len(donnees['gc']['table_montant'])} lignes · C5 {donnees['gc_c5']['eta']} % · haut {donnees.get('gc_haut')} · "
          f"soudure {donnees['soudure']['eta']} % · plancher {donnees['plancher']['eta']} %")


if __name__ == "__main__":
    main()
