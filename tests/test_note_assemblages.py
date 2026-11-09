# -*- coding: utf-8 -*-
"""La NOTE PDF du module « Assemblages métalliques » sur le site (api_web/note_assemblages.py, api_web/pdf_reunion.py),
vérifiée en Python ordinaire. Chaque contrôle rougit si l'on retire ce qu'il protège.

  1. La RÉUNION des PDF (pdf_reunion) : les pages recopiées telles quelles — texte (pdftotext) et rendu (pymupdf, au
     pixel près) de chaque note seule = ceux de ses pages dans le document réuni ; signets, titre, auteur ; même
     entrée → mêmes octets ; une référence dans une chaîne n'est pas renumérotée ; un PDF illisible est refusé.
  2. Les SEPT configurations (et le pied de poteau dans ses deux modes), deux états au moins chacune, dont un non
     conforme : la note se fait par le routeur (« telecharger », « note_pdf ») ; ses pages sont celles du générateur
     validé seul (texte identique) ; ses CHIFFRES CLÉS sont ceux de l'écran du site (taux de tête, verdict, taux de
     chaque pièce, efforts, profilés, boulons ; le récapitulatif : le taux de la tête et l'état).
  3. La géométrie qui gouverne (pince non conforme) : le récapitulatif dit le taux et la cause de l'écran.
  4. PARITÉ avec Streamlit : pour une même note, le PDF du site et le rapport de projet de l'écran Streamlit
     (acier/rapport_projet.generer_rapport, pymupdf) ont le même texte page à page, les mêmes signets, le même titre,
     le même auteur — la copie nommée du déroulé est liée.
  5. Les REFUS : un assemblage à compléter, à corriger ou dont le calcul est impossible ne part pas dans une note ; le
     message nomme l'assemblage. Un autre fichier que la note est refusé.
  6. La page de garde : les informations de la note du site (nom, chantier, indice, date) → celles du projet.
  7. Les conditions du NAVIGATEUR : un sous-processus où fontTools, les accélérateurs C, streamlit, numpy,
     matplotlib, pymupdf/fitz, pandas, scipy sont refusés, avec le Pillow factice et les seules polices du paquet de
     la note, fait la note de ``etat_releve`` (tous les générateurs) ; il dit ce qu'elle charge (fichiers du dépôt,
     polices) ; rapport_projet n'importe plus pymupdf en tête.
  8. MUTATIONS : sans le masque des chaînes, sans la renumérotation, sans le taux de l'écran au récapitulatif, sans
     les refus — les contrôles qui les gardent rougissent.

    /usr/bin/python3.12 tests/test_note_assemblages.py        (pdftotext, pymupdf et streamlit requis)
"""
import ast
import base64
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

import pymupdf  # noqa: E402

from acier import rapport_projet as RP  # noqa: E402
from acier.bibliotheques import PERSO  # noqa: E402
from api_web import registre  # noqa: E402
from api_web import routeur as RT  # noqa: E402

registre.module("assemblages")
from api_web import assemblages as A  # noqa: E402
from api_web import commun as C  # noqa: E402
from api_web import note_assemblages as NA  # noqa: E402
from api_web import pdf_reunion as PR  # noqa: E402

OK, KO = [], []
TMP = tempfile.mkdtemp(prefix="note_assemblages_")
INFOS = {"nom": "Halle A", "chantier": "", "indice": "B", "date": "09/11/2026"}


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))
    return bool(cond)


def texte(pdf: bytes, f=None, l=None, layout=False) -> str:
    """Le texte pdftotext de ``pdf`` (pages f à l)."""
    fd, p = tempfile.mkstemp(suffix=".pdf", dir=TMP)
    os.close(fd)
    with open(p, "wb") as fh:
        fh.write(pdf)
    a = ["pdftotext"] + (["-layout"] if layout else []) + (["-f", str(f), "-l", str(l)] if f else []) + [p, "-"]
    r = subprocess.run(a, capture_output=True, text=True)
    os.remove(p)
    return r.stdout + (("\n[stderr] " + r.stderr) if r.stderr.strip() else "")


def asm(cle, rep, *saisies):
    """Un assemblage de la configuration ``cle``, saisi comme à l'écran (``modifier``)."""
    el = A.nouvel_assemblage(cle, rep)
    e = {"version": 1, "assemblages": [el], "actif": 0}
    for ch, v in saisies:
        msg = A.modifier(e, ch, v)
        assert msg is None, (cle, ch, v, msg)
    return e["assemblages"][0]


def note(*els, infos=None):
    """La note par le routeur (comme le navigateur) : (réponse, octets ou None)."""
    etat = {"version": 1, "assemblages": [copy.deepcopy(e) for e in els], "actif": 0}
    r = json.loads(RT.traiter(json.dumps({"module": "assemblages", "action": "telecharger", "etat": etat,
                                          "quoi": "note_pdf", "infos": infos if infos is not None else INFOS})))
    return r, (base64.b64decode(r["fichier"]["base64"]) if r.get("ok") else None)


def ecran(el):
    """Ce que l'écran du site affiche pour cet assemblage (le routeur, action « calculer »)."""
    etat = {"version": 1, "assemblages": [copy.deepcopy(el)], "actif": 0}
    return json.loads(RT.traiter(json.dumps({"module": "assemblages", "action": "calculer", "etat": etat})))


# ===================================================================== 1. la réunion des PDF
print("— 1. la réunion des PDF")
el1, el2 = asm("plat_ame", "R1"), asm("platine_beton", "R2")
n1 = NA.module_note("plat_ame").generer_note(A._bilan(A._cfg(el1), el1)["R"], {"nom_projet": "Essai", "date": "09/11/2026"})
n2 = NA.module_note("platine_beton").generer_note(A._bilan(A._cfg(el2), el2)["R"], {"nom_projet": "Essai", "date": "09/11/2026"})
reuni = PR.reunir([n1, n2], signets=[("Plat d'âme — A1", 1), ("Pied de poteau — « A2 »", 3)], titre="Note de calcul — essai",
                  auteur="Bureau d'Études Valens")


def controle_reunion(reuni, n1, n2):
    """Les pages réunies = les notes seules : texte et rendu, page à page."""
    d = pymupdf.open(stream=reuni, filetype="pdf")
    ecarts = []
    for (src, debut) in ((n1, 1), (n2, 3)):
        s = pymupdf.open(stream=src, filetype="pdf")
        if texte(src) != texte(reuni, debut, debut + s.page_count - 1):
            ecarts.append(("texte", debut))
        for j in range(s.page_count):
            if s[j].get_pixmap(dpi=36).samples != d[debut - 1 + j].get_pixmap(dpi=36).samples:
                ecarts.append(("rendu", debut + j))
    return d, ecarts


d, ecarts = controle_reunion(reuni, n1, n2)
chk("réunion : 4 pages, texte (pdftotext) et rendu (pixels) de chaque note seule identiques à ses pages réunies",
    d.page_count == 4 and not ecarts and not d.is_repaired, (d.page_count, ecarts, d.is_repaired))
chk("réunion : les signets (titres accentués, guillemets) pointent sur la première page de chaque note",
    d.get_toc() == [[1, "Plat d'âme — A1", 1], [1, "Pied de poteau — « A2 »", 3]], d.get_toc())
chk("réunion : le titre et l'auteur du document", d.metadata["title"] == "Note de calcul — essai" and d.metadata["author"] == "Bureau d'Études Valens",
    d.metadata)
chk("réunion : pdfinfo lit le document sans erreur",
    "Pages:           4" in subprocess.run(["pdfinfo", "-"], input=reuni, capture_output=True).stdout.decode()
    and not subprocess.run(["pdfinfo", "-"], input=reuni, capture_output=True).stderr.strip())
chk("réunion : reproductible (mêmes notes, mêmes signets → mêmes octets)",
    PR.reunir([n1, n2], signets=[("Plat d'âme — A1", 1), ("Pied de poteau — « A2 »", 3)], titre="Note de calcul — essai",
              auteur="Bureau d'Études Valens") == reuni)
chk("réunion : nombre_de_pages lit chaque note (2 pages)", PR.nombre_de_pages(n1) == 2 and PR.nombre_de_pages(n2) == 2)


def controle_chaines():
    return PR._renumeroter(b"<< /T (a 3 0 R \\) 4 0 R) /U <3320302052> /P 3 0 R /K [ 4 0 R ] >>", {3: 9, 4: 10}) \
        == b"<< /T (a 3 0 R \\) 4 0 R) /U <3320302052> /P 9 0 R /K [ 10 0 R ] >>"


chk("réunion : une référence écrite dans une chaîne (« a 3 0 R ») n'est pas renumérotée, les vraies le sont", controle_chaines())
chk("réunion : « /F2 8 0 R » : la référence est 8 (le chiffre d'un nom n'en fait pas partie)",
    PR._renumeroter(b"<< /F2 8 0 R /F3+0 12 0 R >>", {8: 1, 12: 2}) == b"<< /F2 1 0 R /F3+0 2 0 R >>")
refus = []
for nom, octets in (("pas un PDF", b"bonjour"),
                    ("renvois compressés", b"%PDF-1.5\n1 0 obj\n<< /Type /XRef >>\nendobj\nstartxref\n9\n%%EOF\n"),
                    ("tronqué", n1[: len(n1) // 2])):
    try:
        PR.reunir([octets])
        refus.append((nom, "accepté"))
    except PR.PdfIllisible:
        pass
chk("réunion : un PDF illisible (pas un PDF, renvois compressés, tronqué) est refusé avec PdfIllisible", not refus, refus)

# ===================================================================== 2. les sept configurations, l'écran et la note
print("— 2. les configurations : la note = le générateur validé, ses chiffres = ceux de l'écran")
CAS = [
    ("doubles cornières · défaut", asm("doubles_cornieres", "A1")),
    ("doubles cornières · V = 300 kN (non conforme)", asm("doubles_cornieres", "A1", (["v", "V_Ed"], 300))),
    ("plat d'âme · défaut", asm("plat_ame", "A2")),
    ("plat d'âme · V = 260 kN, N = 20 kN (non conforme)", asm("plat_ame", "A2", (["v", "V_Ed"], 260), (["v", "N_Ed"], 20))),
    ("platine d'about · défaut", asm("platine_about", "A3")),
    ("platine d'about · V = 400 kN (non conforme)", asm("platine_about", "A3", (["v", "V_Ed"], 400))),
    ("plat d'âme sur poteau · défaut", asm("plat_ame_poteau", "A4")),
    ("plat d'âme sur poteau · V = 600 kN (non conforme)", asm("plat_ame_poteau", "A4", (["v", "V_Ed"], 600))),
    ("doubles cornières sur poteau · défaut", asm("doubles_cornieres_poteau", "A5")),
    ("doubles cornières sur poteau · V = 600 kN (non conforme)", asm("doubles_cornieres_poteau", "A5", (["v", "V_Ed"], 600))),
    ("platine d'about sur poteau · défaut", asm("platine_about_poteau", "A6")),
    ("platine d'about sur poteau · V = 600 kN (non conforme)", asm("platine_about_poteau", "A6", (["v", "V_Ed"], 600))),
    ("pied de poteau à la main · défaut", asm("platine_beton", "A7")),
    ("pied de poteau à la main · N = 2 500 kN (non conforme)", asm("platine_beton", "A7", (["v", "N_Ed"], 2500))),
    ("pied de poteau complet · VAL-A", asm("platine_beton", "A8", (["exemple"], "VAL-A"))),
    ("pied de poteau complet · N = 3 000 kN (aucune épaisseur ne suffit)", asm("platine_beton", "A8", (["exemple"], "VAL-A"), (["v", "N_Ed"], 3000))),
]
EFFORTS = (("V_Ed", "VEd", "kN"), ("N_Ed", "NEd", "kN"), ("M_Ed", "MEd", "kNm"), ("Mx_Ed", "Mx,Ed", "kNm"),
           ("My_Ed", "My,Ed", "kNm"), ("Vx_Ed", "Vx,Ed", "kN"), ("Vy_Ed", "Vy,Ed", "kN"))
_ETA_PDF = re.compile(r"η max (> 999|[\d  ]+,\d) %")
_PIECES_PDF = re.compile(r"([A-ZÉÈ][\wéèêàç']+) (> 999|\d+) %")


def chiffres_ecran(r):
    """Les chiffres de l'écran : le taux de la tête (pastille du verdict), son niveau, les taux des pièces."""
    tete = r["resultat"].split('class="verdict"')[1]
    m = re.search(r'class="chip xl (\w)"[^>]*><small>η</small>([^<]+)</span>', tete)
    pieces = re.findall(r'<div class="el[^"]*"[^>]*><div class="l1"><b>([^<]+)</b>.*?<span class="pct (\w)"[^>]*>([^<]+)</span>',
                        r["resultat"])
    return {"niveau": m.group(1), "taux": m.group(2).replace(C.NBSP, " ").replace("&gt;", ">"),
            "pieces": [(n, t.replace(C.NBSP, " ").replace("&gt;", ">")) for n, _niv, t in pieces
                       if n not in ("Géométrie", "Proposition")]}


def pourcent(t):
    """« 74 % » → 74.0 ; hors échelle (« > 999 % » de la note, « 999 % » plafonné de l'écran) → inf."""
    t = t.replace(" ", " ").strip().rstrip("%").strip()
    x = float("inf") if t.startswith(">") else float(t.replace(" ", "").replace(",", "."))
    return float("inf") if x >= 999 else x


def controle_cas(liste):
    """Pour chaque (nom, assemblage) : la note par le routeur, comparée au générateur seul et à l'écran. [écarts], n"""
    ecarts, n_cas = [], 0
    for nom, el in liste:
        r = ecran(el)
        rep, pdf = note(el)
        if not rep.get("ok"):
            ecarts.append((nom, "note refusée", rep.get("erreur")))
            continue
        n_cas += 1
        b = A._bilan(A._cfg(el), el)
        R, u = b["R"], el["values"]
        seule = NA.module_note(el["type"]).generer_note(R, RP.infos_note(NA.infos_projet(INFOS), el))
        p1 = texte(pdf, 3, 3)
        e = chiffres_ecran(r)
        # 2a. les pages de l'assemblage sont celles de son générateur seul
        if PR.nombre_de_pages(pdf) != 4 or texte(pdf, 3, 4) != texte(seule):
            ecarts.append((nom, "pages ≠ note seule"))
        # 2b. le taux de tête et le verdict
        m = _ETA_PDF.search(p1)
        eta_pdf = pourcent(m.group(1)) if m else None
        eta_ecran = pourcent(e["taux"])
        if eta_pdf is None or not (abs(eta_pdf - eta_ecran) < 1.0 or (eta_ecran >= 999 and eta_pdf >= 999)):
            ecarts.append((nom, "η", eta_pdf, e["taux"]))
        verdict_pdf = re.search(r"^(NON VÉRIFIÉ|VÉRIFIÉ)$", p1, re.M)
        if not verdict_pdf or (verdict_pdf.group(1) == "NON VÉRIFIÉ") != (e["niveau"] == "r"):
            ecarts.append((nom, "verdict", verdict_pdf and verdict_pdf.group(1), e["niveau"]))
        # 2c. les taux des pièces de la note (sa ligne « Boulons 71 % Plat 55 % … ») sont tous affichés par l'écran
        # (multiensemble des pourcentages entiers ; l'écran en montre davantage : les critères de la platine du pied de
        # poteau, les lignes « information » — égalité stricte pour les attaches poutre–poutre et poutre–poteau)
        pdf_pieces = sorted(pourcent(t + " %") for _n, t in _PIECES_PDF.findall(p1.split("η max", 1)[1].split("\n\n", 2)[1]))
        ecr_pieces = sorted(pourcent(t) for _n, t in e["pieces"])
        reste = list(ecr_pieces)
        manque = [x for x in pdf_pieces if not (x in reste and not reste.remove(x))]
        if not pdf_pieces or manque or (el["type"] != "platine_beton" and pdf_pieces != ecr_pieces):
            ecarts.append((nom, "pièces", pdf_pieces, e["pieces"]))
        # 2d. les efforts saisis (les champs de l'écran), les profilés et les boulons
        for k, lab, unite in EFFORTS:
            m = re.search(re.escape(lab) + r" (-?[\d  ]+,\d+) " + unite + r"\b", p1)
            if k in u and f"{lab} " in p1 and (not m or abs(float(m.group(1).replace(" ", "").replace(" ", "").replace(",", "."))
                                                            - float(u[k] or 0)) > 1e-9):
                ecarts.append((nom, "effort", k, u[k], m and m.group(0)))
        profils = [u[k] for k in ("prof_S", "prof_P", "prof_c") if u.get(k) and u[k] != PERSO]
        profils += [x.get("prof") for x in (u.get("elems") or []) if isinstance(x, dict) and x.get("prof")]
        for p in profils:
            if p not in p1 or p not in r["phrase"]:
                ecarts.append((nom, "profilé", p, p in p1, p in r["phrase"]))
        boulon = u.get("boulon_u") or u.get("tige_u") or ""
        if boulon and (boulon not in p1 or boulon not in r["resultat"]):
            ecarts.append((nom, "boulon", boulon))
        # 2e. le récapitulatif : la ligne de l'assemblage porte le taux de la tête de l'écran et l'état du moteur
        recap = [ln for ln in texte(pdf, 2, 2, layout=True).splitlines() if re.match(r"\s*01\s", ln)]
        attendu = "> 999 %" if eta_ecran >= 999 else e["taux"]          # hors échelle : comme la note de l'assemblage
        if not recap or attendu not in recap[0].replace(" ", " ") or RP.ETATS[RP.etat_de(R)].upper() not in recap[0]:
            ecarts.append((nom, "récapitulatif", recap[:1], attendu))
    return ecarts, n_cas


ecarts, n_cas = controle_cas(CAS)
chk(f"{n_cas} cas (sept configurations, pied de poteau dans ses deux modes, la moitié non conformes) : la note se fait, "
    "ses pages sont celles du générateur seul, et ses chiffres clés (η de tête, verdict, taux des pièces, efforts, profilés, "
    "boulons, récapitulatif) sont ceux de l'écran", not ecarts and n_cas == len(CAS), ecarts[:4])
chk("au moins deux états par configuration, dont un non conforme",
    all(sum(1 for _n, el in CAS if el["type"] == cle) >= 2 for cle in A.ordre())
    and all(any(ecran(el)["statut_html"].count("Non conforme") for _n, el in CAS if el["type"] == cle) for cle in A.ordre()))

# ===================================================================== 3. la géométrie qui gouverne
print("— 3. la géométrie qui gouverne la tête")
GEO = None
for k, v in (("e2b_u", 22), ("e2b_u", 24), ("e1_u", 24), ("e1_u", 25)):
    el = asm("plat_ame", "G1", (["v", k], v))
    R = A._bilan(A._cfg(el), el)["R"]
    if A._cause_geometrie(R) is not None:
        GEO = el
        break


def controle_geometrie(el):
    R = A._bilan(A._cfg(el), el)["R"]
    r = ecran(el)
    e = chiffres_ecran(r)
    _rep, pdf = note(el)
    ligne = [ln for ln in texte(pdf, 2, 2, layout=True).splitlines() if re.match(r"\s*01\s", ln)]
    suite = texte(pdf, 2, 2)
    return (pdf is not None and ligne and e["taux"] in ligne[0].replace(" ", " ") and "Géométrie" in suite
            and e["niveau"] == "r" and "NON VÉRIFIÉ" in ligne[0]), (e["taux"], ligne[:1], R.eta_max)


if chk("un état où la géométrie gouverne la tête de l'écran (pince non conforme au-delà du η du moteur)", GEO is not None):
    ok_geo, info_geo = controle_geometrie(GEO)
    chk("géométrie : le récapitulatif dit le taux de la tête de l'écran (pas le η du moteur seul) et nomme la géométrie",
        ok_geo, info_geo)
    # (relecture du 09/11/2026) la PAGE de l'assemblage reste celle du générateur validé : quand la géométrie gouverne,
    # elle écrit le η max du MOTEUR et « NON VÉRIFIÉ », pas le taux de pince de l'écran. L'écart est MESURÉ ici et ÉCRIT
    # (MODULES_SITE.md § 4 quinquies, qui disait que chaque page d'assemblage porte le η de l'écran) : ce contrôle lie
    # l'écrit à la mesure
    p_geo = texte(note(GEO)[1], 3, 3)
    m_geo = _ETA_PDF.search(p_geo)
    eta_page, eta_tete = (pourcent(m_geo.group(1)) if m_geo else None), pourcent(chiffres_ecran(ecran(GEO))["taux"])
    with open(os.path.join(RACINE, "docs", "site", "MODULES_SITE.md"), encoding="utf-8") as fh:
        doc_site = fh.read()
    chk("géométrie : la page de l'assemblage garde le η max du moteur (pas le taux de l'écran) avec « NON VÉRIFIÉ » — "
        "écart écrit dans MODULES_SITE.md",
        eta_page is not None and abs(eta_page - eta_tete) >= 1.0 and "NON VÉRIFIÉ" in p_geo
        and "sauf quand la GÉOMÉTRIE gouverne" in doc_site, (eta_page, eta_tete))

# ===================================================================== 4. la parité avec le rapport de projet Streamlit
print("— 4. parité avec le rapport de projet de l'écran Streamlit")
els = [el for _n, el in CAS]
_rep, pdf_site = note(*els)
projet = {"assemblages": [{"id": i + 1, "type": el["type"], "values": copy.deepcopy(el["values"]), "visserie": el["visserie"]}
                          for i, el in enumerate(els)], "suivant": len(els) + 1}
pdf_st = RP.generer_rapport(projet, NA.infos_projet(INFOS))
ds, dt = pymupdf.open(stream=pdf_site, filetype="pdf"), pymupdf.open(stream=pdf_st, filetype="pdf")
ecarts = [k + 1 for k in range(max(ds.page_count, dt.page_count))
          if k >= min(ds.page_count, dt.page_count) or texte(pdf_site, k + 1, k + 1, True) != texte(pdf_st, k + 1, k + 1, True)]
chk(f"le PDF du site et le rapport de projet Streamlit ({dt.page_count} pages, {len(els)} assemblages) : même texte page à page",
    ds.page_count == dt.page_count and not ecarts, (ds.page_count, dt.page_count, ecarts[:5]))
chk("… mêmes signets", ds.get_toc() == dt.get_toc(), (ds.get_toc()[:3], dt.get_toc()[:3]))
chk("… même titre et même auteur", (ds.metadata["title"], ds.metadata["author"]) == (dt.metadata["title"], dt.metadata["author"]),
    (ds.metadata, dt.metadata))
if GEO is not None:
    pr = {"assemblages": [{"id": 1, "type": GEO["type"], "values": GEO["values"], "visserie": GEO["visserie"]}], "suivant": 2}
    st_ = texte(RP.generer_rapport(pr, NA.infos_projet(INFOS)), 2, 2)
    si_ = texte(note(GEO)[1], 2, 2)
    chk("géométrie : seule la ligne du récapitulatif diffère de Streamlit (le taux et la cause de l'écran)",
        st_ != si_ and "Géométrie" in si_ and "Géométrie" not in st_)

# ===================================================================== 5. les refus
print("— 5. les refus")
vide = asm("plat_ame", "V1", (["v", "prof_S"], PERSO))
vide["values"]["hS_u"] = None                               # une dimension du profilé personnalisé restée vide
corriger = asm("platine_about", "V2")
corriger["values"]["V_Ed"] = None                           # V_Ed vide (une note relue) : l'écran le refuse
impossible = asm("plat_ame_poteau", "V3")
impossible["values"]["n1_u"] = 80.0                         # au-delà de COMPTES_MAXI : le calcul n'est pas lancé


def controle_refus():
    out = []
    for nom, el, mot in (("à compléter", vide, "à compléter"), ("à corriger", corriger, "à corriger"),
                         ("calcul impossible", impossible, "calcul impossible")):
        r, pdf = note(asm("doubles_cornieres", "OK"), el)
        rep = el["values"]["id_rep"]
        if r.get("ok") or pdf is not None or rep not in r.get("erreur", "") or mot not in r.get("erreur", ""):
            out.append((nom, r.get("ok"), r.get("erreur")))
    return out


refuses = controle_refus()
chk("un assemblage à compléter, à corriger ou dont le calcul est impossible : la note est refusée, le message le nomme",
    not refuses, refuses)
r = json.loads(RT.traiter(json.dumps({"module": "assemblages", "action": "telecharger", "etat": None, "quoi": "autre"})))
chk("un autre fichier que la note est refusé avec un message", not r["ok"] and "inconnu" in r["erreur"], r)

# ===================================================================== 6. la garde
print("— 6. la page de garde")
r, pdf = note(asm("plat_ame", "A1"), infos={"nom": "Note <b>/RDC", "chantier": "Halle A", "indice": "C", "date": "01/12/2026"})
g = texte(pdf, 1, 1, layout=True)
chk("garde : chantier → PROJET, nom de la note → PARTIE, indice, date ; le nom du fichier dit la note entière",
    re.search(r"PROJET\s+Halle A", g) and re.search(r"PARTIE\s+Note <b>/RDC", g) and re.search(r"DATE\s+01/12/2026", g)
    and re.search(r"INDICE\s+C", g) and r["fichier"]["nom"] == C.nom_fichier("Note de calcul - Note <b>/RDC", "pdf")
    == "Note de calcul - Note -b-RDC.pdf"
    and r["fichier"]["type"] == "application/pdf", (g[:400], r["fichier"]["nom"]))
r, pdf = note(asm("plat_ame", "A1"), infos={"nom": "Halle B"})
g = texte(pdf, 1, 1, layout=True)
chk("garde : sans chantier, le nom de la note est le PROJET ; l'assemblage porte son repère (« A1 » en tête de sa note)",
    re.search(r"PROJET\s+Halle B", g) and "Halle B · A1" not in g and "A1" in texte(pdf, 3, 3), g[:300])
chk("infos_projet : la correspondance site → projet Streamlit",
    NA.infos_projet({"nom": "N", "chantier": "C", "indice": "B", "date": "02/02/2026"})
    == {"nom_projet": "C", "partie": "N", "date": "02/02/2026", "indice": "B"}
    and NA.infos_projet({"nom": "N"})["nom_projet"] == "N" and NA.infos_projet({})["indice"] == "0")

# ===================================================================== 7. les conditions du navigateur
print("— 7. les conditions du navigateur")
SONDE = r"""
import builtins, json, os, sys
sys.path.insert(0, {stub!r}); sys.path.insert(1, {racine!r}); os.chdir({racine!r})
class Refus:
    def find_spec(self, nom, chemin=None, cible=None):
        r = nom.split(".")[0]
        if r in ("fontTools", "_rl_accel", "rl_accel", "streamlit", "numpy", "matplotlib", "pymupdf", "fitz", "pandas", "scipy") \
                or nom == "reportlab.lib._rl_accel":
            raise ImportError("refusé dans le navigateur : " + nom)
sys.meta_path.insert(0, Refus())
lus = set(); _open = builtins.open
def espion(f, *a, **k):
    try: lus.add(os.path.abspath(f))
    except Exception: pass
    return _open(f, *a, **k)
builtins.open = espion
from api_web import registre
registre.module("assemblages")
avant = {{os.path.abspath(m.__file__) for m in list(sys.modules.values()) if getattr(m, "__file__", None)}}
lus.clear()
from api_web import note_assemblages as NA, pdf_reunion as PR
import base64
f = NA.fichier(NA.etat_releve(), {{"nom": "relevé"}})
o = base64.b64decode(f["base64"])
mods = {{os.path.abspath(m.__file__) for m in list(sys.modules.values()) if getattr(m, "__file__", None)}}
racine = {racine!r} + os.sep
depot = sorted(os.path.relpath(x, {racine!r}) for x in (mods - avant) | lus if x.startswith(racine) and not x.endswith(".pyc"))
import PIL
print(json.dumps({{"pdf": o[:5].decode(), "pages": PR.nombre_de_pages(o), "depot": depot,
                  "polices": sorted(os.path.basename(x) for x in lus if x.endswith(".ttf")),
                  "pil": PIL.__file__, "numpy": "numpy" in sys.modules, "pymupdf": "pymupdf" in sys.modules,
                  "fonttools": "fontTools" in sys.modules}}))
"""
sys.path.insert(0, os.path.join(RACINE, "site", "web"))
import construire as K  # noqa: E402
polices = os.path.join(TMP, "polices")
os.makedirs(polices)
for f_ in K.NOTE_POLICES:
    shutil.copy(os.path.join(RACINE, "ndc_pdf", "fonts", f_), polices)
stub = os.path.join(RACINE, "site", "note_pdf")
p_ = subprocess.run([sys.executable, "-c", SONDE.format(stub=stub, racine=RACINE)], capture_output=True, text=True,
                    env={**os.environ, "NDC_FONT_DIR": polices})
try:
    S_ = json.loads(p_.stdout.strip().splitlines()[-1])
except Exception:  # noqa: BLE001
    S_ = {"erreur": p_.stderr[-800:]}
n_attendu = 1 + 1 + 2 * len(NA.etat_releve()["assemblages"])
chk(f"navigateur : sans fontTools, streamlit, numpy, pymupdf…, avec le Pillow factice et les six polices du paquet, la note de "
    f"etat_releve (huit assemblages, tous les générateurs) se fait ({n_attendu} pages)",
    S_.get("pdf") == "%PDF-" and S_.get("pages") == n_attendu and S_.get("pil", "").startswith(stub)
    and S_.get("numpy") is False and S_.get("pymupdf") is False and S_.get("fonttools") is False, S_)
chk("navigateur : les polices lues sont celles du paquet de la note (construire.NOTE_POLICES)",
    S_.get("polices") and set(S_["polices"]) <= set(K.NOTE_POLICES), S_.get("polices"))
NOTES = sorted(os.path.relpath(os.path.join(d_, f_), RACINE) for d_, _s, fs in os.walk(os.path.join(RACINE, "acier", "assemblages"))
               for f_ in fs if f_ == "note.py")
chk(f"navigateur : la note charge les {len(NOTES)} générateurs, leur kit, le rapport de projet et la réunion (à mettre dans le paquet de la note)",
    set(NOTES + ["acier/assemblages/note_kit.py", "acier/rapport_projet.py", "acier/assemblages/poutre_poutre/doubles_cornieres/rapport.py",
                 "api_web/note_assemblages.py", "api_web/pdf_reunion.py"]) <= set(S_.get("depot") or []),
    sorted(set(NOTES) - set(S_.get("depot") or [])))
print("     fichiers du dépôt chargés par la note (hors site/note_pdf) :",
      ", ".join(x for x in (S_.get("depot") or []) if not x.startswith("site/")))
arbre = ast.parse(open(os.path.join(RACINE, "acier", "rapport_projet.py"), encoding="utf-8").read())
en_tete = [n for n in arbre.body if isinstance(n, (ast.Import, ast.ImportFrom))
           and any((a.name or "").split(".")[0] in ("pymupdf", "fitz") for a in n.names)]
chk("rapport_projet n'importe pas pymupdf en tête de module (le navigateur charge sa garde et son récapitulatif)", not en_tete)

# ===================================================================== 8. mutations
print("— 8. mutations")
garde_hc = PR._hors_chaines
PR._hors_chaines = lambda corps: corps
try:
    chk("mutation : sans le masque des chaînes, la référence écrite dans un titre est renumérotée (le contrôle rougit)",
        not controle_chaines())
finally:
    PR._hors_chaines = garde_hc
garde_rn = PR._renumeroter
PR._renumeroter = lambda corps, table: corps
try:
    try:
        mut = PR.reunir([n1, n2])
        _d, ec = controle_reunion(mut, n1, n2)
        rougit = bool(ec)
    except Exception:  # noqa: BLE001 — un document illisible rougit aussi
        rougit = True
    chk("mutation : sans la renumérotation, les pages réunies ne sont plus les notes (le contrôle rougit)", rougit)
finally:
    PR._renumeroter = garde_rn
if GEO is not None:
    garde_ir = NA.info_recap
    NA.info_recap = lambda R: {"etat": RP.etat_de(R), "eta": float(R.eta_max or 0.0), "gov": R.gov.label if R.gov else ""}
    try:
        chk("mutation : le récapitulatif de Streamlit (η du moteur seul) au lieu du taux de l'écran rougit le contrôle de la géométrie",
            not controle_geometrie(GEO)[0])
    finally:
        NA.info_recap = garde_ir


class _AutreCalcul:
    """L'aiguillage du site, sauf que la note est calculée sur des efforts majorés de 25 % : une note qui ne dirait plus
    les chiffres de l'écran."""
    def __getattr__(self, nom):
        return getattr(A, nom)

    @staticmethod
    def _bilan(cfg, el):
        el = copy.deepcopy(el)
        for k in ("V_Ed", "N_Ed"):
            if isinstance(el["values"].get(k), (int, float)):
                el["values"][k] *= 1.25
        return A._bilan(cfg, el)


NA.A = _AutreCalcul()
try:
    ec, _n = controle_cas([c for c in CAS if c[1]["type"] in ("plat_ame", "platine_beton")])
    chk("mutation : une note calculée sur d'autres efforts que l'écran rougit la parité des chiffres clés (η, pièces, efforts)",
        len(ec) >= 4 and {x[1] for x in ec} >= {"η", "effort"}, ec[:3])
finally:
    NA.A = A
garde_m = NA._motif
NA._motif = lambda cfg, b: None
try:
    chk("mutation : sans les refus, une note part avec un assemblage à compléter ou à corriger (le contrôle rougit)",
        bool(controle_refus()))
finally:
    NA._motif = garde_m

shutil.rmtree(TMP, ignore_errors=True)
print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)
