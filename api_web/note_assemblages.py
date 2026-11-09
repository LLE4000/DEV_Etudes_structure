# -*- coding: utf-8 -*-
"""La NOTE DE CALCUL PDF du module « Assemblages métalliques » sur le site : le RAPPORT DE PROJET de l'écran Streamlit
(``acier/rapport_projet.py``, bouton « Générer PDF » de la page projet) — garde avec sommaire, récapitulatif, puis la
note de CHAQUE assemblage de la note — fait dans le navigateur (Pyodide).

Rien n'est réécrit :

- la note de chaque assemblage est celle de son générateur VALIDÉ (``acier/assemblages/<famille>/<configuration>/
  note.py``, ``generer_note(R, infos)`` ; celle du pied de poteau aiguille elle-même vers ``manuel/note.py`` quand
  l'assemblage est calculé « à la main »), sur le résultat ``R`` que l'écran du site affiche (``assemblages._bilan`` :
  le moteur de l'écran Streamlit, parité prouvée par tests/test_api_web_assemblages*.py) ;
- la garde, le sommaire et le récapitulatif sont dessinés par les fonctions de ``rapport_projet`` (``lignes_sommaire``,
  ``pages_recap``, ``_recap``, ``infos_note``, ``etat_de``) et le kit ``ndc_pdf`` (``Style.garde_lignes``) ;
- seule la RÉUNION des PDF change : Streamlit les met bout à bout avec pymupdf, absent du navigateur ; le site les
  réunit en Python pur (``api_web/pdf_reunion.py``), pages recopiées octet pour octet.

COPIE NOMMÉE : ``document`` reprend le déroulé de ``rapport_projet.generer_rapport`` (garde, récapitulatif, notes,
signets, titre, auteur), et ``module_note`` la règle de ``rapport_projet._note_de`` (le fichier ``note.py`` voisin de
l'écran de la configuration) sans charger cet écran, qui importe streamlit. ``tests/test_note_assemblages.py`` les
lie : pour une même note, le PDF du site et celui de Streamlit ont le même texte page à page, les mêmes signets, le
même titre et le même auteur.

Ce que le site change, pour que la note dise les chiffres de l'écran (décision de l'ingénieur du 09/11/2026, S4) :

- le TAUX du récapitulatif est celui que l'écran affiche (``assemblages.eta_R`` arrondi par ``commun.pct``) : le η
  max du moteur, borné à 100 % quand sa vérification est satisfaite, porté au rapport de la pire pince NON conforme
  (Tableau 3.3) — la vérification dimensionnante nomme alors la GÉOMÉTRIE, comme la tête du résultat de l'écran
  (``assemblages._cause_geometrie``). Streamlit y écrivait le η max du moteur seul (une tige à 30 mm du bord :
  « 0 % » à côté de « Non vérifié »). Un taux hors échelle (η infini : aucune épaisseur ne suffit) s'écrit
  « > 999 % », comme dans la note de l'assemblage (l'écran le plafonne à « 999 % ») ;
- un assemblage « à compléter » (dimension vide), « à corriger » (une saisie que l'écran refuse) ou dont le calcul
  est impossible ne part pas dans une note : la note entière est REFUSÉE, avec un message qui le nomme (Streamlit
  calculait les deux premiers en silence, et posait une page « calcul impossible » pour le troisième). Le bouton
  « Note PDF » de l'écran Streamlit d'un assemblage est d'ailleurs grisé dans ces cas-là.

Les informations de la page de garde : celles de la note du site (``infos`` : nom, chantier, indice, date) deviennent
celles du projet Streamlit (``projet_ui.infos``) — ``nom_projet`` = le chantier, ou à défaut le nom de la note ;
``partie`` = le nom de la note quand un chantier est donné ; ``indice`` ; ``date`` (la date du jour à défaut) ; puis
chaque note reçoit ``rapport_projet.infos_note`` (la partie suivie du repère de l'assemblage, sa visserie).
"""
import base64
import html
import importlib
import math
import os
import tempfile
from datetime import datetime
from typing import Dict, List, Optional

from . import assemblages as A
from . import commun as C
from . import pdf_reunion as PR

# le sous-titre de la ligne « Récapitulatif » du sommaire (rapport_projet.generer_rapport : la même phrase)
SOUS_TITRE_RECAP = "Taux de travail et vérification dimensionnante de chaque assemblage"
# l'auteur du document (rapport_projet.generer_rapport : set_metadata)
AUTEUR = "Bureau d'Études Valens"


def module_note(cle: str):
    """Le générateur de la note d'une configuration : le ``note.py`` voisin de son écran Streamlit (la règle de
    ``rapport_projet._note_de``), importé SANS l'écran."""
    from acier import registre as RG
    a = RG.trouver(cle)
    if not a:
        raise ValueError(f"configuration inconnue : {cle}")
    return importlib.import_module(a[1].module.rsplit(".", 1)[0] + ".note")


def infos_projet(infos: Optional[Dict]) -> Dict:
    """Les informations du projet Streamlit (``projet_ui.infos``) depuis celles de la note du site."""
    infos = infos or {}
    nom = str(infos.get("nom") or "").strip()
    chantier = str(infos.get("chantier") or "").strip()
    return {"nom_projet": chantier or nom, "partie": nom if chantier else "",
            "date": str(infos.get("date") or "").strip() or datetime.today().strftime("%d/%m/%Y"),
            "indice": str(infos.get("indice") or "").strip() or "0"}


def _motif(cfg, b: Dict) -> Optional[str]:
    """Pourquoi un assemblage ne part pas dans la note (None : il part) — les mots de l'écran du site."""
    if b["erreur"]:
        return f"calcul impossible ({b['erreur']})"
    if b["vides"]:
        return "à compléter (dimensions à saisir : " + ", ".join(A._nom_champ(cfg, k) for k in b["vides"]) + ")"
    if b["refus"]:
        return "à corriger (" + " ; ".join(msg for _k, msg in b["refus"]) + ")"
    return None


def info_recap(R) -> Dict:
    """La ligne de l'assemblage dans le récapitulatif : l'état de ``rapport_projet.etat_de``, le TAUX AFFICHÉ par
    l'écran et la vérification dimensionnante — la géométrie quand c'est elle qui gouverne la tête du résultat."""
    from acier import rapport_projet as RP
    eta = A.eta_R(R)
    # le taux de la tête du résultat (commun.pct, sans décimale) ; hors échelle (aucune épaisseur ne suffit : η infini),
    # « > 999 % » comme la note de l'assemblage — l'écran l'écrit « 999 % », plafonné
    if isinstance(eta, (int, float)) and math.isfinite(eta) and eta <= 9.99:
        eta = A.taux(eta) / 100.0
    else:
        eta = math.inf
    cause = A._cause_geometrie(R)
    if cause is not None:
        gov = html.unescape(cause[0])
    else:
        gov = R.gov.label if R.gov else ""
    return {"etat": RP.etat_de(R), "eta": eta, "gov": gov}


def elements(etat: Dict, infos: Dict) -> List:
    """``[(entrée, info, octets de sa note, nombre de pages)]`` dans l'ordre de la note (la forme de
    ``rapport_projet.notes``) ; ValueError si un assemblage ne peut pas partir (``_motif``)."""
    from acier import rapport_projet as RP
    calc, refus = [], []
    for i, el in enumerate(etat["assemblages"]):
        cfg = A._cfg(el)
        b = A._bilan(cfg, el)
        m = _motif(cfg, b)
        if m:
            refus.append(f"{A._repere(el, i)} ({cfg.court}) : {m}")
        calc.append((el, cfg, b))
    if refus:
        raise ValueError("La note PDF n'est pas préparée : " + " ; ".join(refus)
                         + ". Corriger ces données dans le cadre Données de l'assemblage.")
    out = []
    for el, cfg, b in calc:
        e = {"type": el["type"], "values": el["values"], "visserie": el.get("visserie", "")}
        pdf = module_note(cfg.cle).generer_note(b["R"], RP.infos_note(infos, e))
        out.append((e, info_recap(b["R"]), pdf, PR.nombre_de_pages(pdf)))
    return out


def document(etat: Dict, infos: Dict) -> bytes:
    """Le PDF de la note (``infos`` : celles du projet Streamlit, ``infos_projet``) : garde(s) et sommaire,
    récapitulatif, la note de chaque assemblage ; un signet par partie. COPIE NOMMÉE du déroulé de
    ``rapport_projet.generer_rapport`` (voir l'en-tête)."""
    from acier import rapport_projet as RP
    from ndc_pdf import data as ndc_data
    from ndc_pdf.kit import Doc
    from ndc_pdf.styles import Encre
    infos = infos or {}
    elems = elements(etat, infos)
    S = Encre()
    lignes = RP.lignes_sommaire(elems, S)
    n_recap = RP.pages_recap(len(elems))
    avant = [dict(titre=RP.TITRE_RECAP, sous=SOUS_TITRE_RECAP, pages=n_recap)]
    doc_meta = ndc_data.construire_doc(infos, date_defaut=datetime.today().strftime("%d/%m/%Y"))
    doc_meta["bureau"], doc_meta["titre"] = RP.BUREAU, RP.TITRE
    fd, chemin = tempfile.mkstemp(suffix=".pdf", prefix="garde_assemblages_")
    os.close(fd)
    try:
        d = Doc(chemin, RP.A4P, title=f"Note de calcul — {RP.TITRE}")
        d.new_page(RP.A4P)
        n_garde = S.garde_lignes(d, doc_meta, lignes, legende=RP.LEGENDE, avant=avant)
        n_recap_dessine = RP._recap(S, d, doc_meta, elems)
        d.save()
        with open(chemin, "rb") as fh:
            tete = fh.read()
    finally:
        try:
            os.remove(chemin)
        except OSError:
            pass
    if n_recap_dessine != n_recap:
        raise ValueError("le sommaire et le récapitulatif ne comptent pas les mêmes pages")
    signets = [("Récapitulatif", n_garde + 1)]
    page = n_garde + n_recap + 1
    for (_e, _info, _pdf, n), lg in zip(elems, lignes):
        signets.append((lg["titre"], page))
        page += n
    return PR.reunir([tete] + [pdf for _e, _i, pdf, _n in elems], signets=signets,
                     titre=f"Note de calcul — {RP.TITRE}", auteur=AUTEUR)


def etat_releve() -> Dict:
    """Une note qui passe par TOUS les générateurs : les sept configurations, le pied de poteau dans ses deux modes
    (calcul à la main, vérification complète). La construction du site la rejoue pour relever ce que la note charge
    (``site/web/construire.py``, le paquet de la note) ; ``tests/test_note_assemblages.py`` la joue aux conditions du
    navigateur."""
    lst = [A.nouvel_assemblage(cle, f"A{i + 1}") for i, cle in enumerate(A.ordre())]
    etat = {"version": 1, "assemblages": lst, "actif": len(lst) - 1}
    A.gerer_elements(etat, "ajouter")                         # un pied de poteau de plus (la configuration ouverte)…
    A.modifier(etat, ["exemple"], "VAL-A")                    # … en vérification complète
    return etat


def fichier(etat: Dict, infos: Optional[Dict] = None) -> Dict:
    """Le fichier à télécharger : {"nom", "type": "application/pdf", "base64"} (le routeur le contrôle). Le nom dit
    la note entière : « Note de calcul - <nom de la note>.pdf »."""
    infos = dict(infos or {})
    nom = str(infos.get("nom") or "").strip() or "Assemblages"
    octets = document(etat, infos_projet(infos))
    return {"nom": C.nom_fichier(f"Note de calcul - {nom}", "pdf"), "type": "application/pdf",
            "base64": base64.b64encode(octets).decode("ascii")}
