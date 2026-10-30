# -*- coding: utf-8 -*-
"""Rapport d'un PROJET d'assemblages : page(s) de garde — bandeau, cartouche,
sommaire avec pastille et numéro de page — puis la note de chaque assemblage,
telle que son module la produit, le tout assemblé en un seul PDF avec un
signet par assemblage. Sans Streamlit : ``generer_rapport(projet, infos)``
rend les octets.

La garde est celle des notes Poutre / Dalle (``ndc_pdf.styles.Style.garde_lignes``) :
même bandeau, même cartouche, même sommaire ; les numéros de page sont
calculés sur les notes réellement produites (deux pages par assemblage
aujourd'hui, mais c'est le PDF qui fait foi), et le sommaire continue sur une
deuxième page de garde au-delà de onze assemblages.

Une entrée dont le calcul ou la note échoue ne fait pas échouer le rapport :
elle y figure par une page d'erreur et une pastille « calcul impossible ».
"""
import importlib
import os
import tempfile
from datetime import datetime

import pymupdf
from reportlab.lib.pagesizes import A4, landscape

from acier import registre
from acier import projet as modele
from ndc_pdf import data as ndc_data
from ndc_pdf.kit import Doc
from ndc_pdf.styles import Encre

A4P, A4L = A4, landscape(A4)
TITRE = "Assemblages métalliques"
BUREAU = "Bureau d'Études Valens · Note de calcul"
LEGENDE = ("Vert : vérifié   ·   Ocre : vérifié sous réserve des alertes   ·   "
           "Rouge : non vérifié ou calcul impossible")
ETATS = {"ok": "Vérifié", "att": "Sous réserve", "ko": "Non vérifié", "err": "Calcul impossible"}


def _module(type_id):
    t = registre.trouver(type_id)
    if not t or not t[1].disponible:
        raise ValueError(f"Type d'assemblage inconnu : {type_id}")
    return registre.charger(t[1])


def _note_de(m):
    return importlib.import_module(m.__name__.rsplit(".", 1)[0] + ".note")


def titre_de(type_id):
    t = registre.trouver(type_id)
    return t[1].titre if t else str(type_id)


def infos_note(infos, e):
    """Les informations du cartouche d'UNE note dans le rapport : celles du
    projet, la partie complétée du repère pour que chaque note se reconnaisse."""
    u = e.get("values") or {}
    infos = infos or {}
    rep = modele.repere(e)
    partie = " · ".join(x for x in (str(infos.get("partie") or ""), rep) if x)
    return {"nom_projet": str(infos.get("nom_projet") or u.get("id_projet") or ""),
            "partie": partie or str(u.get("id_rep") or ""),
            "date": str(infos.get("date") or u.get("id_date") or datetime.today().strftime("%d/%m/%Y")),
            "indice": str(infos.get("indice") or "0"),
            "visserie": str(e.get("visserie") or "")}


def etat_de(R):
    if R is None:
        return "err"
    if R.verified:
        return "att" if (R.reserve or R.alerts) else "ok"
    return "ko"


def _page_erreur(e, message):
    """Une page A4 paysage qui dit pourquoi la note manque."""
    fd, chemin = tempfile.mkstemp(suffix=".pdf", prefix="erreur_assemblage_")
    os.close(fd)
    S = Encre()
    d = Doc(chemin, A4L, title="Note de calcul")
    d.new_page(A4L)
    d.t(S.margin, d.H - 48, f"{modele.repere(e) or '—'} — {titre_de(e['type'])}", S.f_h, 14, S.ink)
    d.t(S.margin, d.H - 72, "CALCUL IMPOSSIBLE", S.f_v, 8, S.ko, track=1.6)
    d.t(S.margin, d.H - 92, d.fit(message, S.f_b, 9, d.W - 2 * S.margin), S.f_b, 9, S.mut)
    d.t(S.margin, d.H - 110, "Ouvrir l'assemblage dans l'application pour corriger ses données.", S.f_b, 9, S.mut)
    d.save()
    with open(chemin, "rb") as fh:
        data = fh.read()
    os.remove(chemin)
    return data


def notes(projet, infos):
    """``[(entrée, état, octets, nb_pages), …]`` dans l'ordre du projet."""
    out = []
    for e in modele.entrees(projet):
        try:
            m = _module(e["type"])
            u = {k: m.ecran_saisie.valeur_widget(k, (e.get("values") or {}).get(k)) for k in m.CLES}
            R = m.compute(u)
            pdf = _note_de(m).generer_note(R, infos_note(infos, e))
            etat = etat_de(R)
        except Exception as exc:  # noqa: BLE001 — le rapport doit sortir, l'erreur s'y lit
            pdf, etat = _page_erreur(e, f"{type(exc).__name__}: {exc}"), "err"
        n = pymupdf.open(stream=pdf, filetype="pdf").page_count
        out.append((e, etat, pdf, n))
    return out


def lignes_sommaire(elems, style):
    couleurs = {"ok": style.ok, "att": style.att, "ko": style.ko, "err": style.mut}
    return [dict(titre=f"{modele.repere(e) or '—'} — {titre_de(e['type'])}",
                 sous=modele.description(e.get("values")), etat=ETATS[etat],
                 couleur=couleurs[etat], pages=n)
            for (e, etat, _pdf, n) in elems]


def generer_rapport(projet, infos):
    """Le rapport (bytes) : garde(s) + notes, signets par assemblage."""
    infos = infos or {}
    elems = notes(projet, infos)
    S = Encre()
    lignes = lignes_sommaire(elems, S)
    doc_meta = ndc_data.construire_doc(infos, date_defaut=datetime.today().strftime("%d/%m/%Y"))
    doc_meta["bureau"], doc_meta["titre"] = BUREAU, TITRE

    fd, chemin = tempfile.mkstemp(suffix=".pdf", prefix="garde_assemblages_")
    os.close(fd)
    d = Doc(chemin, A4P, title=f"Note de calcul — {TITRE}")
    d.new_page(A4P)
    n_garde = S.garde_lignes(d, doc_meta, lignes, legende=LEGENDE)
    d.save()
    with open(chemin, "rb") as fh:
        garde = fh.read()
    os.remove(chemin)

    out = pymupdf.open()
    out.insert_pdf(pymupdf.open(stream=garde, filetype="pdf"))
    toc, page = [], n_garde + 1
    for (e, etat, pdf, n), lg in zip(elems, lignes):
        out.insert_pdf(pymupdf.open(stream=pdf, filetype="pdf"))
        toc.append([1, lg["titre"], page])
        page += n
    if toc:
        out.set_toc(toc)
    out.set_metadata({"title": f"Note de calcul — {TITRE}", "author": "Bureau d'Études Valens"})
    data = out.tobytes(garbage=3, deflate=True)
    out.close()
    return data
