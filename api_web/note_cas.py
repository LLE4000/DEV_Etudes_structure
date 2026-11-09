# -*- coding: utf-8 -*-
"""La NOTE DE CALCUL PDF, sur le site, des modules dont la note est faite de CAS : « Garde-corps », « Soudures »,
« Charges sur plancher ».

Elle n'est pas réécrite : ce sont les générateurs VALIDÉS de l'écran Streamlit (``acier/garde_corps/note.py``,
``acier/soudures/note.py``, ``plancher/charges/note.py``, mise en page ``ndc_pdf``) qui la font, dans le navigateur
(Pyodide), sur l'état de la note du site — qui EST l'état du moteur. Le générateur refait le calcul de chaque cas avec
le moteur, par la même fonction que l'écran du site (``etat.verifier``, ``verification.verifier``) : les tests
tests/test_note_garde_corps.py, test_note_soudures.py, test_note_plancher.py prouvent que la note dit les chiffres que
l'écran affiche (taux, verdicts, efforts, profils).

Le générateur fait la note d'UN cas : deux pages A4 paysage (le calcul ; les détails et la méthode) — trois pour un
garde-corps dont une colonne ne tient pas (sa page de suite, acier/garde_corps/note.py). Le site fait celle de TOUTE la
note — les pages de chaque cas, dans l'ordre de la liste des cas, en UN seul document — sans rien réécrire :

- UN SEUL CAS : c'est ``generer_note`` du générateur, telle quelle ;
- PLUSIEURS CAS : pour chaque cas, la mise en page est choisie comme ``generer_note`` la choisit, sur un brouillon
  (``disposition``) : la ``mise_en_page`` du générateur quand il en a une (le garde-corps : la taille, puis les blocs
  qui passent en page de suite), sinon ``variante`` — une COPIE de la boucle de ``generer_note`` (taille normale, puis
  serrée si une colonne déborde : la première qui tient, la dernière à défaut). Puis les pages du cas sont posées dans
  le document commun (``pages``) : par le générateur quand il sait les poser (``poser`` du garde-corps), sinon par ses
  classes de pages, ``Note(calc, doc, k).construire`` puis ``Methode(calc, doc, k, …).construire`` — une COPIE du corps
  de ``_essai``, qui n'ouvre ni n'enregistre de document. Ces deux copies (soudures, plancher) sont NOMMÉES ici et
  LIÉES par les tests : la variante retenue est celle du générateur pour le cas seul, et le texte de chaque page
  (pdftotext) est celui de la note de ce cas faite seule par le générateur.

Une note qui DÉBORDE encore est REFUSÉE avec un message (la règle de la note de la poutre, api_web/note_poutre.py) : ce
qui sort d'une colonne serait perdu sans un mot — jamais une note tronquée. Le message dit la vraie cause : la mise en
page du générateur pour la configuration du cas, pas un texte saisi. Mesuré le 09/11/2026 : plus aucun garde-corps
(grille de 240 états : tests/test_note_garde_corps_grille.py) ; soudures, 2 états sur 56 d'une grille (C6 en quatre
tronçons, Eurocode). L'écran Streamlit rend telles quelles les notes qui débordent (``derniers_avertissements``,
qu'il ne montre pas).

L'IDENTIFICATION de l'en-tête et du cartouche (projet, partie, indice, date) est TOUJOURS celle de la NOTE du site
(R2 : une seule source ; son nom, son chantier, son indice, la date du jour que la page envoie), avec la
correspondance de la note de la poutre : projet = le chantier, à défaut le nom de la note ; partie = le nom de la note
quand il y a un chantier, sinon le titre du module ; indice = celui de la note, « A » à défaut ; date = celle de la
page, à défaut celle du jour (le générateur). L'identification que porte l'état du moteur (``identification`` des
soudures, de chaque cas du plancher, d'un fichier Streamlit du garde-corps) ne s'imprime PAS sur le site, et l'écran
du site ne la montre plus (décision du 09/11/2026 : avant, le champ « Indice » des soudures et du plancher, « 0 » par
défaut, passait devant l'indice de la note — le PDF disait « indice 0 » sous une note B). Elle reste dans l'état : un
fichier de l'écran Streamlit la garde, aller et retour.
"""
import base64
import copy
import importlib
import os
import tempfile
from datetime import datetime
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple


# ce que l'écran d'un module dit à la place des anciens champs « Identification de la note » (soudures, plancher)
AIDE_IDENTIFICATION = ("L'en-tête et le cartouche de la note PDF portent le nom, le chantier, l'indice et la date de "
                       "cette note du site — ceux du cartouche des dessins —, comme pour tous les modules : une seule source.")


class Generateur(NamedTuple):
    """Ce que le site sait du générateur de la note d'un module."""
    module: str                         # le module Python du générateur (« acier.garde_corps.note »)
    titre: str                          # le titre du module : la partie sans chantier, le nom du fichier sans nom de note
    # page 1 → les arguments de ``Methode`` après (calc, doc, k) : ``(n1.echelle,)``… ; None quand le générateur pose
    # lui-même les pages d'un cas (``poser`` : le garde-corps)
    suite: Optional[Callable]
    # les fichiers du dépôt que la NOTE charge en plus de l'écran du module et du paquet de la note de la poutre
    # (ndc_pdf, le Pillow factice) : le paquet de la note (site/web/construire.py, note.zip) les embarque. La liste est
    # PROUVÉE complète par la sonde « comme dans le navigateur » de tests/test_note_<module>.py
    fichiers: Tuple[str, ...] = ()


def _texte(x) -> str:
    return str(x or "").strip()


def nom_note(gen: Generateur, etat: Dict, infos: Optional[Dict]) -> str:
    """Le nom de la note : celui que la note du site porte ; à défaut, le nom du cas quand il est seul, le titre du
    module sinon."""
    nom = _texte((infos or {}).get("nom"))
    if nom:
        return nom
    if len(etat["cas"]) == 1:
        return _texte(etat["cas"][0].get("nom")) or gen.titre
    return gen.titre


def infos_du_cas(gen: Generateur, etat: Dict, cas: Dict, infos: Optional[Dict]) -> Dict:
    """Les « infos » que le générateur attend pour un cas ({"nom_projet", "partie", "indice", "date"}) : TOUJOURS
    celles de la note du site (voir l'en-tête du fichier) ; l'identification que porte l'état du cas n'y entre pas."""
    infos = infos or {}
    nom, chantier = nom_note(gen, etat, infos), _texte(infos.get("chantier"))
    out = {"nom_projet": chantier or nom, "partie": nom if chantier else gen.titre,
           "indice": _texte(infos.get("indice")) or "A"}
    if _texte(infos.get("date")):
        out["date"] = _texte(infos.get("date"))
    return out


def variante(NT, calc, doc_meta: Dict, brouillon: str):
    """(k, débordements) : la taille que ``generer_note`` retient pour ce cas — la première variante où rien ne déborde,
    la dernière à défaut. COPIE de la boucle de ``generer_note`` (liée par les tests à ``derniere_variante``), pour un
    générateur qui n'a pas sa propre ``mise_en_page`` (soudures, plancher)."""
    k, warn = NT.VARIANTES[-1][0], []
    for k, _nom in NT.VARIANTES:
        warn = NT._essai(calc, doc_meta, k, brouillon)
        if not warn:
            break
    return k, list(warn)


def disposition(NT, calc, doc_meta: Dict, brouillon: str) -> Tuple[float, Tuple, List[str]]:
    """(k, reports, débordements) du cas : la ``mise_en_page`` du générateur quand il en a une (le garde-corps : la
    taille, puis les blocs passés en page de suite), sinon ``variante`` (rien en page de suite)."""
    if hasattr(NT, "mise_en_page"):
        k, _nom, reports, warn = NT.mise_en_page(calc, doc_meta, brouillon)
        return k, tuple(reports), list(warn)
    k, warn = variante(NT, calc, doc_meta, brouillon)
    return k, (), warn


def pages(NT, gen: Generateur, d, calc, doc_meta: Dict, k: float, reports: Tuple = ()) -> List[str]:
    """Les pages du cas posées dans le document ``d``. Un générateur qui sait les poser lui-même (``poser`` : le
    garde-corps, page de suite comprise) les pose ; sinon les deux pages de ``_essai`` (COPIE de son corps, sans ouvrir
    ni enregistrer de document ; liée par les tests au texte de la note du cas seul). Rend les débordements."""
    if hasattr(NT, "poser"):
        return list(NT.poser(d, calc, doc_meta, k, reports))
    d.new_page(NT.A4L)
    n1 = NT.Note(calc, doc_meta, k)
    w = list(n1.construire(d))
    d.new_page(NT.A4L)
    w += NT.Methode(calc, doc_meta, k, *gen.suite(n1)).construire(d)
    return w


def _refus_debordement(cas: Dict, warn: List[str]) -> ValueError:
    return ValueError(f"La note du cas « {_texte(cas.get('nom')) or 'sans nom'} » ne tient pas dans sa mise en page ("
                      + " ; ".join(warn) + ") : elle ne sort pas tronquée. Ce n'est pas un texte saisi trop long : la "
                      "mise en page du générateur de la note de ce module ne prévoit pas assez de place pour le contenu "
                      "de ce cas (sa configuration). À signaler au bureau ; l'écran garde tout le calcul.")


def pdf(gen: Generateur, etat: Dict, infos: Optional[Dict] = None) -> bytes:
    """La note PDF (octets) de toute la note : deux pages par cas, dans l'ordre des cas."""
    NT = importlib.import_module(gen.module)
    from ndc_pdf import data as ndc_data
    from ndc_pdf.kit import Doc
    liste = etat["cas"]
    # deux fichiers temporaires, effacés à la fin (le générateur, appelé sans chemin, laisse le sien : dans le navigateur
    # il resterait en mémoire jusqu'au rechargement de la page)
    fichiers = []
    try:
        for _ in range(2):
            fd, ch = tempfile.mkstemp(suffix=".pdf", prefix="note_cas_")
            os.close(fd)
            fichiers.append(ch)
        brouillon, chemin = fichiers
        if len(liste) == 1:
            octets = NT.generer_note(etat, liste[0], infos_du_cas(gen, etat, liste[0], infos), chemin=chemin)
            if NT.derniers_avertissements:
                raise _refus_debordement(liste[0], NT.derniers_avertissements)
            return octets
        titre = "Note de calcul — " + nom_note(gen, etat, infos)
        d = Doc(chemin, NT.A4L, title=titre)
        for cas in liste:
            doc_meta = ndc_data.construire_doc(infos_du_cas(gen, etat, cas, infos), date_defaut=datetime.today().strftime("%d/%m/%Y"))
            doc_meta["titre"] = titre
            calc = NT.Calcul(etat, cas)
            k, reports, warn = disposition(NT, calc, doc_meta, brouillon)
            if warn:
                raise _refus_debordement(cas, warn)
            warn = pages(NT, gen, d, calc, doc_meta, k, reports)
            if warn:
                raise _refus_debordement(cas, warn)
        d.save()
        with open(chemin, "rb") as f:
            return f.read()
    finally:
        for ch in fichiers:
            try:
                os.remove(ch)
            except OSError:
                pass


def fichier(gen: Generateur, etat: Dict, infos: Optional[Dict] = None) -> Dict:
    """Le fichier à télécharger : {"nom", "type": "application/pdf", "base64"} (le routeur le contrôle). Le nom est
    celui de la note du site, comme pour la poutre : « Note de calcul - <nom>.pdf »."""
    from . import commun as C
    octets = pdf(gen, copy.deepcopy(etat), dict(infos or {}))
    return {"nom": C.nom_fichier(f"Note de calcul - {nom_note(gen, etat, infos)}", "pdf"), "type": "application/pdf",
            "base64": base64.b64encode(octets).decode("ascii")}
