# -*- coding: utf-8 -*-
"""Les outils communs des suites de la NOTE PDF des modules à cas du site (garde-corps, soudures, charges sur
plancher : api_web/note_cas.py) — tests/test_note_garde_corps.py, test_note_soudures.py, test_note_plancher.py.

- ``pages`` : le texte de chaque page d'un PDF (pdftotext -layout), pour comparer la note de toute la note à celle de
  chaque cas faite seule par le générateur ;
- ``polices`` : les polices que le PDF porte (pdffonts) ;
- ``ecran`` : le texte que l'écran du site montre (le cadre Résultat, sans balises ni dessins) ;
- ``sonde`` : la note faite COMME DANS LE NAVIGATEUR, dans un sous-processus calqué sur la sonde du paquet de la note
  (site/web/construire.py, ``_SONDE_NOTE``) : fontTools, les accélérateurs C de reportlab, streamlit et matplotlib
  refusés, le Pillow factice du site (site/note_pdf), les seules polices du paquet de la note (``NOTE_POLICES``) ;
  numpy est permis (le moteur de ces modules le charge déjà : ``avec_numpy`` du manifeste) et son chargement est dit.
  Elle rend les fichiers du dépôt que la NOTE charge en plus de ce que l'écran a chargé, les polices lues, et les PDF ;
- les contrôles communs : la note demandée au routeur (``note_et_tailles``), son identité page à page avec la note de
  chaque cas faite seule par le générateur (``controle_identite``), l'identification (``controle_infos``), le relevé
  des fichiers du paquet (``releve_paquet``, ``paquet_integre``).
"""
import base64
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUB = os.path.join(RACINE, "site", "note_pdf")
REFUS = ("fontTools", "_rl_accel", "rl_accel", "streamlit", "matplotlib")


def pages(octets: bytes, brut: bool = False):
    """Le texte de chaque page, dans l'ordre : mis en page (pdftotext -layout : la comparaison page à page), ou ``brut``
    (-raw : l'ordre du dessin, où une ligne coupée en deux reste d'un tenant, pour y lire une phrase)."""
    p = subprocess.run(["pdftotext", "-raw" if brut else "-layout", "-", "-"], input=octets, capture_output=True, check=True)
    return p.stdout.decode("utf-8").split("\f")[:-1]


def polices(octets: bytes):
    """Les polices que le PDF porte (noms de base, sans le préfixe du sous-ensemble « ABCDEF+ »)."""
    fd, ch = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    try:
        with open(ch, "wb") as f:
            f.write(octets)
        p = subprocess.run(["pdffonts", ch], capture_output=True, text=True, check=True)
    finally:
        os.remove(ch)
    noms = set()
    for ligne in p.stdout.splitlines()[2:]:
        if ligne.strip():
            noms.add(ligne.split()[0].split("+")[-1])
    return sorted(noms)


def ecran(fragment: str) -> str:
    """Le texte d'un fragment de l'écran : sans dessins ni balises, entités relues, espaces regroupées."""
    h = re.sub(r"<svg.*?</svg>", " ", fragment, flags=re.S)
    h = re.sub(r"<(sub|sup)>(.*?)</\1>", r"\2", h)
    h = re.sub(r"<[^>]+>", " | ", h)
    return re.sub(r"\s+", " ", re.sub(r"(\s*\|\s*)+", " | ", html.unescape(h))).replace(" ", " ")


def pct_note(t) -> str:
    """Le taux comme la NOTE l'écrit (les générateurs : ``f"{t * 100:.0f} %"``, arrondi ordinaire, sans tenir la bande
    de couleur) — à comparer à l'écran, qui l'arrondit DANS sa bande (commun.pct)."""
    return f"{t * 100:.0f} %"


SONDE = r"""
import builtins, json, os, sys
RACINE, STUB, REFUS = {racine!r}, {stub!r}, {refus!r}
sys.path.insert(0, STUB); sys.path.insert(1, RACINE); os.chdir(RACINE)
class Refus:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in REFUS or name == "reportlab.lib._rl_accel":
            raise ImportError("refusé dans le navigateur : " + name)
sys.meta_path.insert(0, Refus())
lus = set(); _open = builtins.open
def espion(f, *a, **k):
    try: lus.add(os.path.abspath(f))
    except Exception: pass
    return _open(f, *a, **k)
builtins.open = espion
def depot():
    mods = {{os.path.abspath(m.__file__) for m in list(sys.modules.values()) if getattr(m, "__file__", None)}}
    return {{os.path.relpath(f, RACINE) for f in mods | lus if f.startswith(RACINE + os.sep) and not f.endswith(".pyc")}}
import importlib
M = importlib.import_module("api_web." + {module!r})
with _open({etats!r}, encoding="utf-8") as f:
    etats = json.load(f)
for e in etats:
    e = M.normaliser(e)
    for i in range(len(e["cas"])):
        e["actif"] = i
        M.resultat(e)
ecran = depot()
numpy_ecran = "numpy" in sys.modules
sorties = [M.telecharger(M.normaliser(e), "note_pdf", {infos!r})["base64"] for e in etats]
note = depot()
import PIL
print(json.dumps({{"note": sorted(f for f in note - ecran if not f.endswith(".ttf")), "numpy_ecran": numpy_ecran,
                   "numpy": "numpy" in sys.modules, "polices": sorted(os.path.basename(f) for f in lus if f.endswith(".ttf")),
                   "pil": PIL.__file__, "fonttools": "fontTools" in sys.modules,
                   "rl_accel": any(m.split(".")[-1] == "_rl_accel" for m in sys.modules), "sorties": sorties}}))
"""


def sonde(module: str, etats, infos) -> dict:
    """La note de chaque état faite comme dans le navigateur (voir l'en-tête) ; {"erreur"} si elle échoue."""
    sys.path.insert(0, os.path.join(RACINE, "site", "web"))
    try:
        import construire as K
    finally:
        sys.path.pop(0)
    dossier = tempfile.mkdtemp()
    try:
        for f in K.NOTE_POLICES:
            shutil.copy(os.path.join(RACINE, "ndc_pdf", "fonts", f), dossier)
        ch = os.path.join(dossier, "etats.json")
        with open(ch, "w", encoding="utf-8") as f:
            json.dump(etats, f, ensure_ascii=False)
        code = SONDE.format(racine=RACINE, stub=STUB, refus=REFUS, module=module, etats=ch, infos=infos)
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=RACINE,
                           env={**os.environ, "NDC_FONT_DIR": dossier})
    finally:
        shutil.rmtree(dossier, ignore_errors=True)
    try:
        out = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return {"erreur": p.stderr[-1500:]}
    out["sorties"] = [base64.b64decode(s) for s in out["sorties"]]
    out["polices_paquet"] = list(K.NOTE_POLICES)
    return out


# ------------------------------------------------------------------ les contrôles communs aux trois modules
def norm(t: str) -> str:
    """Un texte comparable : espaces insécables et fines en espaces, blancs regroupés."""
    return re.sub(r"\s+", " ", t.replace(" ", " ").replace(" ", " ").replace(" ", " "))


def present(nombre: str, texte: str) -> bool:
    """``nombre`` (« 64 % », « 0,495 kN·m ») figure dans ``texte`` comme un nombre entier : pas « 8 % » dans « 1,8 % »."""
    return re.search(r"(?<![\d,.])" + re.escape(norm(nombre)), norm(texte)) is not None


def lignes_resultat(resultat: str):
    """Les lignes du cadre Résultat (commun.ligne_res) : [(nom, pièce, niveau, taux affiché)]."""
    out = []
    for m in re.finditer(r'<div class="el[^"]*"[^>]*><div class="l1"><b>([^<]+)</b>(.*?)</div><div class="l2">.*?'
                         r'<span class="pct (\w+)"[^>]*>([^<]+)</span></div></div>', resultat, re.S):
        out.append((html.unescape(m.group(1)), ecran(m.group(2)).strip(" |"), m.group(3), norm(html.unescape(m.group(4)))))
    return out


def verdict(resultat: str):
    """La grande pastille du verdict : (niveau, taux affiché) — None sans taux (calcul impossible)."""
    m = re.search(r'<span class="chip xl (\w+)"[^>]*><small>η</small>([^<]+)</span>', resultat)
    return (m.group(1), norm(html.unescape(m.group(2)))) if m else None


def resultats_fenetres(volets: str):
    """Le résultat de chaque ligne de formule des fenêtres de calcul (« = 0,495 kN·m », la pastille « 63,2 % »)."""
    out = []
    for m in re.finditer(r'<div class="fx">(.*?)</div>', volets, re.S):
        r = re.findall(r'<span class="(?:r|chip \w+)"[^>]*>(.*?)</span>', m.group(1))
        if r:
            out.append((ecran(m.group(1)), norm(ecran(r[-1]).strip(" |"))))
    return out


def note_et_tailles(mod, etat: dict, infos: dict):
    """(fichier, octets, [k posé pour chaque cas]) : la note demandée au ROUTEUR du site (comme la page), un espion sur
    ``note_cas.pages`` relève la taille de chaque cas (rien n'est relevé pour un cas seul : c'est ``generer_note``)."""
    import copy
    from api_web import note_cas as NC
    from api_web import routeur as R
    vrai, ks = NC.pages, []

    def espion(NT, gen, d, calc, doc_meta, k):
        ks.append(k)
        return vrai(NT, gen, d, calc, doc_meta, k)
    NC.pages = espion
    try:
        r = json.loads(R.traiter(json.dumps({"module": mod.MODULE, "action": "telecharger", "etat": copy.deepcopy(etat),
                                             "quoi": "note_pdf", "infos": infos})))
    finally:
        NC.pages = vrai
    if not r.get("ok"):
        raise AssertionError(r.get("erreur"))
    return r["fichier"], base64.b64decode(r["fichier"]["base64"]), ks


def controle_identite(mod, etat: dict, infos: dict, octets: bytes, ks=None) -> list:
    """Les écarts entre la note de TOUTE la note et la note de chaque cas faite SEULE par le générateur : deux pages
    par cas, le texte de chaque page, la taille retenue (``ks`` : relevée par ``note_et_tailles``)."""
    import importlib
    from api_web import note_cas as NC
    NT = importlib.import_module(mod.NOTE.module)
    tx = pages(octets)
    ecarts = []
    n = len(etat["cas"])
    if len(tx) != 2 * n:
        ecarts.append(f"{len(tx)} pages pour {n} cas")
    for i, c in enumerate(etat["cas"]):
        seul = pages(NT.generer_note(etat, c, NC.infos_du_cas(mod.NOTE, etat, c, infos)))
        k_seul = dict((nom, k) for k, nom in NT.VARIANTES)[NT.derniere_variante]
        for j in range(2):
            if 2 * i + j >= len(tx) or tx[2 * i + j] != seul[j]:
                ecarts.append(f"cas {i + 1} « {c.get('nom')} », page {j + 1} : le texte n'est pas celui du cas seul")
        if ks is not None and n > 1 and (i >= len(ks) or ks[i] != k_seul):
            ecarts.append(f"cas {i + 1} : taille {ks[i] if i < len(ks) else None} au lieu de {k_seul} (le générateur, cas seul)")
    return ecarts


def controle_infos(octets: bytes, etat: dict, attendu) -> list:
    """Les écarts d'identification : ``attendu(i, cas)`` → les textes que les deux pages du cas i doivent porter."""
    tx = pages(octets)
    ecarts = []
    for i, c in enumerate(etat["cas"]):
        t = norm(" ".join(tx[2 * i:2 * i + 2]))
        for x in attendu(i, c):
            if norm(x) not in t:
                ecarts.append(f"cas {i + 1} : « {x} » absent")
    return ecarts


def releve_paquet(mod, sortie: dict) -> dict:
    """Ce que la sonde dit des fichiers : {"hors": les fichiers que la note charge en plus de l'écran, hors du paquet
    du moteur (manifeste) et du paquet de la note de la poutre (ndc_pdf, Pillow factice, export_pdf) ; "declares" :
    ``mod.NOTE.fichiers`` ; "non_charges" : les déclarés que la note n'a pas chargés}."""
    with open(os.path.join(RACINE, "site", "web", "moteur", "manifeste.json"), encoding="utf-8") as f:
        moteur = set(json.load(f)["fichiers"])
    deja = {f for f in sortie["note"] if f.startswith(("ndc_pdf/", "site/note_pdf/")) or f == "modules/export_pdf.py"}
    hors = sorted(set(sortie["note"]) - moteur - deja)
    return {"hors": hors, "declares": list(mod.NOTE.fichiers),
            "non_charges": [f for f in mod.NOTE.fichiers if f not in sortie["note"]]}


def paquet_integre(mod):
    """Après l'intégration (le module dans ``construire.NOTE_PDF``) : les fichiers déclarés absents du paquet de la note
    (note.zip) ; None tant que le module n'y est pas."""
    import io
    import zipfile
    sys.path.insert(0, os.path.join(RACINE, "site", "web"))
    try:
        import construire as K
    finally:
        sys.path.pop(0)
    if mod.MODULE not in K.NOTE_PDF:
        return None
    octets, _ = K.paquet_note()
    noms = set(zipfile.ZipFile(io.BytesIO(octets)).namelist())
    return [f for f in mod.NOTE.fichiers if f not in noms]
