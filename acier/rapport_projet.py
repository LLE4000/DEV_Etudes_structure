# -*- coding: utf-8 -*-
"""Rapport d'un PROJET d'assemblages : page(s) de garde — bandeau, cartouche,
sommaire avec pastille et numéro de page —, un RÉCAPITULATIF (une ligne par
assemblage : profilés, efforts, taux maximal, vérification dimensionnante,
état), puis la note de chaque assemblage telle que son module la produit, le
tout assemblé en un seul PDF avec un signet par partie. Sans Streamlit :
``generer_rapport(projet, infos)`` rend les octets.

La garde est celle des notes Poutre / Dalle (``ndc_pdf.styles.Style.garde_lignes``) :
même bandeau, même cartouche, même sommaire ; les numéros de page sont
calculés sur les notes réellement produites (deux pages par assemblage
aujourd'hui, mais c'est le PDF qui fait foi), et le sommaire continue sur une
deuxième page de garde au-delà de onze lignes.

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
from ndc_pdf.kit import Doc, chip, mix
from ndc_pdf.styles import Encre

A4P, A4L = A4, landscape(A4)
TITRE = "Assemblages métalliques"
BUREAU = "Bureau d'Études Valens · Note de calcul"
LEGENDE = ("Vert : vérifié   ·   Ocre : vérifié sous réserve des alertes   ·   "
           "Rouge : non vérifié ou calcul impossible")
ETATS = {"ok": "Vérifié", "att": "Sous réserve", "ko": "Non vérifié", "err": "Calcul impossible"}
CAP_RECAP = 30                     # lignes du récapitulatif par page (pas de 20 pt)
TITRE_RECAP = "Récapitulatif des assemblages"


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
    """``[(entrée, info, octets, nb_pages), …]`` dans l'ordre du projet ;
    ``info`` = ``{"etat", "eta", "gov"}`` (taux maximal et vérification
    dimensionnante, None / message en cas d'échec)."""
    out = []
    for e in modele.entrees(projet):
        try:
            m = _module(e["type"])
            u = {k: m.ecran_saisie.valeur_widget(k, (e.get("values") or {}).get(k)) for k in m.CLES}
            R = m.compute(u)
            pdf = _note_de(m).generer_note(R, infos_note(infos, e))
            info = {"etat": etat_de(R), "eta": float(R.eta_max or 0.0), "gov": R.gov.label if R.gov else ""}
        except Exception as exc:  # noqa: BLE001 — le rapport doit sortir, l'erreur s'y lit
            message = f"{type(exc).__name__}: {exc}"
            pdf, info = _page_erreur(e, message), {"etat": "err", "eta": None, "gov": message}
        n = pymupdf.open(stream=pdf, filetype="pdf").page_count
        out.append((e, info, pdf, n))
    return out


def _couleur(style, etat):
    return {"ok": style.ok, "att": style.att, "ko": style.ko}.get(etat, style.mut)


def lignes_sommaire(elems, style):
    return [dict(titre=f"{modele.repere(e) or '—'} — {titre_de(e['type'])}",
                 sous=modele.description(e.get("values")), etat=ETATS[info["etat"]],
                 couleur=_couleur(style, info["etat"]), pages=n)
            for (e, info, _pdf, n) in elems]


def pages_recap(n_elems):
    return max(1, -(-n_elems // CAP_RECAP))


def _recap(S, d, doc, elems):
    """Le récapitulatif : un tableau, CAP_RECAP lignes par page (portrait,
    même bandeau que la garde). Retourne le nombre de pages dessinées."""
    W, H = d.W, d.H
    m = 52
    # en-têtes déjà en capitales : « η » passé en majuscule deviendrait « Η »
    cols = [("N°", 22), ("REPÈRE", 44), ("TYPE", 118), ("PROFILÉS · EFFORTS", 112),
            ("η MAX", 40), ("DIMENSIONNANTE", 80), ("ÉTAT", 0)]
    n_pages = 0
    for debut in range(0, max(1, len(elems)), CAP_RECAP):
        d.new_page((W, H))
        n_pages += 1
        d.box(0, H - 96, W, 96, fill=S.panel)
        d.t(m, H - 44, doc.get("bureau", ""), S.f_v, 8.6, S.acc, track=1.6)
        d.t(m, H - 72, TITRE_RECAP + (" (suite)" if debut else ""), S.f_h, 18, S.ink)
        d.line(m, H - 96, W - m, H - 96, S.acc, 1.2)
        y = H - 122
        x = m
        for lab, w in cols:
            d.t(x, y, lab, S.f_v, 6.6, S.mut, track=1.1)
            x += w
        y -= 6
        d.line(m, y, W - m, y, S.acc, 0.8)
        for i, (e, info, _pdf, _n) in enumerate(elems[debut:debut + CAP_RECAP], start=debut):
            y -= 20
            etat = info["etat"]
            u = e.get("values") or {}
            eta = ("—" if info.get("eta") is None
                   else "> 999 %" if info["eta"] > 9.99 else f"{info['eta'] * 100:.0f} %")
            cellules = [(f"{i + 1:02d}", S.f_h, 8.0, mix(S.acc, "#FFFFFF", 0.55)),
                        (modele.repere(e) or "—", S.f_h, 8.4, S.ink),
                        (titre_de(e["type"]), S.f_b, 8.0, S.ink),
                        (modele.description(u), S.f_b, 7.6, S.mut),
                        (eta, S.f_v, 8.0, _couleur(S, etat)),
                        (info.get("gov") or "", S.f_b, 7.4, S.mut)]
            x = m
            for (txt, f, size, col), (_lab, w) in zip(cellules, cols):
                d.t(x, y, d.fit(str(txt), f, size, w - 6), f, size, col)
                x += w
            chip(d, W - m, y - 2, ETATS[etat].upper(), S.f_v, 6.4, "#FFFFFF", _couleur(S, etat), align="right")
            d.line(m, y - 6, W - m, y - 6, mix(S.rule, S.ink, 0.06), 0.5)
        d.t(m, 46, LEGENDE, S.f_b, 7.4, S.mut)
    return n_pages


def generer_rapport(projet, infos):
    """Le rapport (bytes) : garde(s), récapitulatif, notes ; signets."""
    infos = infos or {}
    elems = notes(projet, infos)
    S = Encre()
    lignes = lignes_sommaire(elems, S)
    n_recap = pages_recap(len(elems))
    avant = [dict(titre=TITRE_RECAP, sous="Taux de travail et vérification dimensionnante de chaque assemblage",
                  pages=n_recap)]
    doc_meta = ndc_data.construire_doc(infos, date_defaut=datetime.today().strftime("%d/%m/%Y"))
    doc_meta["bureau"], doc_meta["titre"] = BUREAU, TITRE

    fd, chemin = tempfile.mkstemp(suffix=".pdf", prefix="garde_assemblages_")
    os.close(fd)
    d = Doc(chemin, A4P, title=f"Note de calcul — {TITRE}")
    d.new_page(A4P)
    n_garde = S.garde_lignes(d, doc_meta, lignes, legende=LEGENDE, avant=avant)
    n_recap_dessine = _recap(S, d, doc_meta, elems)
    d.save()
    with open(chemin, "rb") as fh:
        tete = fh.read()
    os.remove(chemin)
    assert n_recap_dessine == n_recap, "le sommaire et le récapitulatif ne comptent pas les mêmes pages"

    out = pymupdf.open()
    out.insert_pdf(pymupdf.open(stream=tete, filetype="pdf"))
    toc = [[1, "Récapitulatif", n_garde + 1]]
    page = n_garde + n_recap + 1
    for (e, info, pdf, n), lg in zip(elems, lignes):
        out.insert_pdf(pymupdf.open(stream=pdf, filetype="pdf"))
        toc.append([1, lg["titre"], page])
        page += n
    out.set_toc(toc)
    out.set_metadata({"title": f"Note de calcul — {TITRE}", "author": "Bureau d'Études Valens"})
    data = out.tobytes(garbage=3, deflate=True)
    out.close()
    return data
