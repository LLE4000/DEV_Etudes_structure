# -*- coding: utf-8 -*-
"""La NOTE DE CALCUL PDF du module « Poutres béton armé » sur le site.

Elle n'est pas réécrite : c'est le générateur VALIDÉ de l'écran Streamlit (``modules/export_pdf.py`` v3.0, mise en page
``ndc_pdf``) qui la produit, dans le navigateur (Pyodide), à partir des valeurs que l'écran Streamlit aurait en mémoire
pour la même poutre. Ce fichier TRADUIT l'état du site (``beton/poutre``, normalisé) en ces valeurs (clés ``b<i>_…``,
``b<i>_sec<j>_…`` de ``st.session_state``), appelle ``export_pdf._collecter_resultats`` — qui refait le calcul avec
``_compute_section``, le calcul du PDF Streamlit — puis complète chaque planche de ce que le site sait en plus :

- la RÉPARTITION des barres de chaque lit (abscisses « x » du moteur) et l'étiquette « L1 : 2Ø16 » ;
- les armatures LATÉRALES du site (imposées ou automatiques : la même règle que le PDF) ;
- les DISPOSITIONS CONSTRUCTIVES (espacements EC2 8.2(2), enrobage, barres dans l'étrier, angles des cadres) ;
- l'ÉTAT de la planche : jamais « Vérifié » sans effort (« Données incomplètes »), « À surveiller » quand une
  disposition l'est.

Le calcul de la planche est donc celui de Streamlit ; ``tests/test_note_poutre.py`` prouve, cas par cas, qu'il rend
les mêmes grandeurs que le moteur du site (h_min, A_s,req, A_s,min, τ, s_lim…) : la note dit ce que l'écran affiche.

Plusieurs poutres : une note, une planche par section, dans l'ordre des poutres. Chaque poutre est traduite à part (les
réglages « globaux » de Streamlit — γ_s, jeux, armatures de peau — sont propres à chaque poutre du site), puis les
planches sont réunies dans un seul document.
"""
import base64
import copy
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from beton.poutre import entree as EN
from beton.poutre import moteur as MO

from .poutre_streamlit import valeurs  # noqa: F401 — la traduction vers l'écran Streamlit (un seul endroit)

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ETAT_INCOMPLET = "Données incomplètes"


def _beton() -> Dict:
    with open(os.path.join(RACINE, "beton_classes.json"), encoding="utf-8") as f:
        return json.load(f)


def resultats(etat: Dict) -> List[Dict]:
    """Les planches de la note (les payloads de ``ndc_pdf.data.construire_sections``) : celles du PDF Streamlit, une
    par section de chaque poutre, complétées des clés du site (abscisses et étiquettes des lits, latérales,
    dispositions, état)."""
    from modules import export_pdf as EP
    beton = _beton()
    out = []
    for bi, p in enumerate(etat["poutres"]):
        p = EN.normaliser(p)
        calc = MO.verifier(p)
        beams, vals = valeurs(p, bi + 1)
        planches = EP._collecter_resultats(beams, vals, beton)
        for pl, rs in zip(planches, calc["sections"]):
            R = pl["R"]
            for face in ("inf", "sup"):
                for j, (lit, geo) in enumerate(zip(R[f"geo_{face}"]["lits"], rs["geometrie"][face]["lits"])):
                    lit["x"] = list(geo["x"])
                    lit["label"] = f"L{j + 1} : {geo['n']}Ø{geo['d']}"
            lat = rs["geometrie"]["laterales"]
            # les axes des latérales : au contact du nu intérieur de l'étrier (le dessin du site, dispositions.py)
            xg = rs["grandeurs"]["enrobage_calcul"] + rs["grandeurs"]["d_etrier"] / 10.0 + float(lat["d"]) / 20.0
            pl["peau"] = ({"d": lat["d"], "n": lat["n"], "ys": list(lat["ys"]), "xs": [xg, p["b"] - xg]}
                          if lat["n"] else None)
            # l'emprise exacte de chaque ligne d'effort tranchant (mm depuis la face gauche) : le dessin la pose telle quelle
            cads = rs["geometrie"]["etriers"]["cadres"]
            pl["cadres_geo"] = [{"x0": cd["x0"] * 10.0, "x1": cd["x1"] * 10.0} for cd in cads]
            if len(pl.get("stirrups") or []) != len(cads):
                # sans effort tranchant, la planche Streamlit ne garde que la première ligne : le dessin les veut toutes
                pl["stirrups"] = [{"type": cd["type"], "d": cd["d"], "brins": cd["brins"],
                                   "from": None if cd["complet"] else cd["de"], "to": None if cd["complet"] else cd["a"]}
                                  for cd in cads]
            pl["dispositions"] = rs["dispositions"]
            if not rs["sollicitee"]:
                pl["etat_label"] = ETAT_INCOMPLET
            else:
                pl["etat_label"] = {MO.NOK: "Non vérifié", MO.WARN: "À surveiller"}.get(rs["statut_complet"], "Vérifié")
            out.append(pl)
    return out


def document(etat: Dict, infos: Optional[Dict], chemin: str):
    """Le document construit (pas encore écrit) : la garde et les planches, comme ``generer_rapport_pdf`` (même
    construction, même repli en trois colonnes si une planche déborde). ``d.warnings`` : ce qui déborde encore."""
    from modules import export_pdf as EP
    from ndc_pdf import data as ndc_data
    infos = infos or {}
    planches = resultats(etat)
    doc_meta = ndc_data.construire_doc(infos, date_defaut=datetime.today().strftime("%d/%m/%Y"))
    sections = ndc_data.construire_sections(planches)
    d = EP._style_ndc(2).build(chemin, sections=sections, doc=doc_meta)
    if d.warnings:
        d = EP._style_ndc(3).build(chemin, sections=sections, doc=doc_meta)
    return d


def pdf(etat: Dict, infos: Optional[Dict] = None) -> bytes:
    """La note PDF (octets) : garde A4 portrait + une planche A4 paysage par section. Une planche qui déborde encore en
    trois colonnes est REFUSÉE avec un message : ce qui sort de la page serait perdu sans un mot (une observation des
    dispositions, un verdict) — jamais une note tronquée."""
    import tempfile
    fd, chemin = tempfile.mkstemp(suffix=".pdf", prefix="note_poutre_")
    os.close(fd)
    try:
        d = document(etat, infos, chemin)
        if d.warnings:
            raise ValueError("La note ne tient pas sur sa planche (" + " ; ".join(d.warnings) + ") : elle ne sort pas "
                             "tronquée. Réduire le nombre d'observations de la section (dispositions constructives).")
        d.save()
        with open(chemin, "rb") as f:
            return f.read()
    finally:
        try:
            os.remove(chemin)
        except OSError:
            pass


def fichier(etat: Dict, infos: Optional[Dict] = None) -> Dict:
    """Le fichier à télécharger : {"nom", "type": "application/pdf", "base64"} (le routeur le contrôle)."""
    infos = dict(infos or {})
    nom = str(infos.get("nom") or "").strip() or (etat["poutres"][0].get("nom") or "Poutres")
    doc = {"nom_projet": str(infos.get("chantier") or "").strip() or nom,
           "partie": nom if str(infos.get("chantier") or "").strip() else "Poutres en béton armé",
           "indice": str(infos.get("indice") or "A"), "date": str(infos.get("date") or "")}
    octets = pdf(copy.deepcopy(etat), doc)
    from . import commun as C
    return {"nom": C.nom_fichier(f"Note de calcul - {nom}", "pdf"), "type": "application/pdf",
            "base64": base64.b64encode(octets).decode("ascii")}
