# -*- coding: utf-8 -*-
"""Les APERÇUS des établis et la VERSION du site (lot S4 : docs/site/REFONTE_ETABLI.md, S4-D3, S4-D8, C7, C12).

- chaque aperçu (site/web/moteur/apercus/<clé>.json) est, À L'OCTET, ce que rend
  ``routeur.traiter(json.dumps({"module": clé, "action": "nouveau"}))`` dans un processus NEUF (rien d'un autre module
  chargé ; un CPython ordinaire, sans les refus du navigateur que construire.py applique : deux chemins, une réponse) ;
- chacun est une réponse COMPLÈTE (``ok`` vrai, ``etat``, ``resultat``) — construire.py refuse d'écrire les autres ;
- la liste du manifeste (« apercus » : clé → empreinte) est celle des 17 établis, ni plus ni moins, chaque empreinte
  celle de son fichier, et aucun autre fichier dans moteur/apercus ;
- la réponse « nouveau » ne dépend pas du JOUR : rejouée sous quatre dates simulées (date.today, datetime.now), elle
  est la même, et égale à l'aperçu — age compris (S4-D8 l'avait mesuré ; on ne compte plus sur le jour où le test passe) ;
- la version du site : la même sur les 24 pages et dans le manifeste ; une construction dans un dossier temporaire la
  reproduit ; elle CHANGE si un octet d'un fichier publié change (js, css, moteur, aperçus, favicon, _headers) ou si un
  gabarit change, et NE CHANGE PAS pour un fichier non publié (configuration, outils, le manifeste lui-même) ;
- chaque contrôle a sa MUTATION : la même vérification, sur une entrée abîmée exprès, doit rougir (sinon le contrôle ne
  contrôle rien) — la dernière section les rejoue.

    python3 tests/test_apercus.py
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(RACINE, "site", "web")
sys.path.insert(0, WEB)
import construire as SW  # noqa: E402  (le générateur du site)

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


# ------------------------------------------------------------------ les vérifications (des fonctions : la mutation les rejoue)
def egal_octet(fichier: bytes, reponse: bytes) -> bool:
    """L'aperçu est la réponse du routeur, à l'octet."""
    return isinstance(fichier, bytes) and fichier == reponse


def complet(cle: str, texte: str) -> bool:
    """Une réponse complète au sens de construire.verifier_apercu (ok vrai, etat, resultat) — la porte de la construction."""
    try:
        SW.verifier_apercu(cle, texte)
        return True
    except SystemExit:
        return False


def liste_juste(manifeste_apercus: dict, dossier_apercus: str, etablis: list) -> bool:
    """La liste du manifeste : les 17 établis, chacun avec l'empreinte (16 hex) de SON fichier, et rien d'autre dans le
    dossier."""
    if not isinstance(manifeste_apercus, dict) or sorted(manifeste_apercus) != sorted(etablis) or len(etablis) != 17:
        return False
    fichiers = sorted(os.listdir(dossier_apercus)) if os.path.isdir(dossier_apercus) else []
    if fichiers != sorted(f"{k}.json" for k in etablis):
        return False
    for k, h in manifeste_apercus.items():
        with open(os.path.join(dossier_apercus, f"{k}.json"), "rb") as f:
            if h != hashlib.sha256(f.read()).hexdigest()[:16]:
                return False
    return True


def meme_reponse(reponses: list, attendue: bytes) -> bool:
    """Les réponses sous les quatre dates : quatre, toutes égales à l'aperçu."""
    return len(reponses) == len(JOURS) and all(r == attendue for r in reponses)


def version_partout(pages: dict, version: str) -> bool:
    """Une balise stab-version par page, la même valeur (16 hex) sur les 24 pages (22 + Projets et Équipe, 11/11/2026)."""
    return (len(pages) == 24 and re.fullmatch(r"[0-9a-f]{16}", version or "") is not None
            and all(re.findall(r'<meta name="stab-version" content="([^"]*)">', h) == [version] for h in pages.values()))


# ------------------------------------------------------------------ un processus NEUF par réponse
# Un CPython ordinaire : le dépôt sur le chemin, rien d'autre de chargé ; ``avant`` s'exécute AVANT tout import du
# moteur (la date simulée) ; la sortie ne porte que la réponse (tout autre écrit part sur l'erreur).
_TRAITER = """
import json, os, sys
sys.path.insert(0, {racine!r}); os.chdir({racine!r})
_sortie, sys.stdout = sys.stdout, sys.stderr
{avant}
from api_web import routeur as R
_sortie.buffer.write(R.traiter(json.dumps({{"module": {cle!r}, "action": "nouveau"}})).encode("utf-8"))
_sortie.flush()
"""
# la date simulée : date.today() et datetime.now()/today() rendent le jour donné, pour tout le moteur (posé avant le
# premier import, il vaut aussi pour ``from datetime import datetime``)
_DATE = """
import datetime as _dt
_J = _dt.date.fromisoformat({jour!r})
class _Date(_dt.date):
    @classmethod
    def today(cls):
        return cls(_J.year, _J.month, _J.day)
class _DateHeure(_dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(_J.year, _J.month, _J.day, 9, 41, 7, tzinfo=tz)
    @classmethod
    def today(cls):
        return cls(_J.year, _J.month, _J.day, 9, 41, 7)
_dt.date, _dt.datetime = _Date, _DateHeure
"""
# quatre jours : un hiver, un été, une fin février, une autre décennie
JOURS = ("2026-01-15", "2026-07-31", "2027-02-28", "2031-12-24")


def processus(code: str) -> bytes:
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, cwd=RACINE)
    if p.returncode != 0:
        return b"ECHEC " + p.stderr[-800:]
    return p.stdout


def traiter_neuf(cle: str, jour: str = "") -> bytes:
    return processus(_TRAITER.format(racine=RACINE, cle=cle, avant=_DATE.format(jour=jour) if jour else ""))


ETABLIS = SW.etablis()
DOSSIER_APERCUS = os.path.join(WEB, "moteur", SW.APERCUS)
MAN = json.load(open(os.path.join(WEB, "moteur", "manifeste.json"), encoding="utf-8"))

# 1. chaque aperçu = le routeur, dans un processus neuf, à l'octet ; une réponse complète
chk(f"les établis sont les {len(ETABLIS)} modules du site moins les redirections (predalle est servi par la dalle)",
    len(ETABLIS) == 17 and "predalle" not in ETABLIS and set(ETABLIS) == set(SW.ADRESSES) - set(SW.REDIRECTIONS), ETABLIS)
with ThreadPoolExecutor(max_workers=max(1, min(4, os.cpu_count() or 1))) as ex:
    NEUFS = dict(zip(ETABLIS, ex.map(traiter_neuf, ETABLIS)))
FICHIERS = {}
for k in ETABLIS:
    chemin = os.path.join(DOSSIER_APERCUS, f"{k}.json")
    FICHIERS[k] = open(chemin, "rb").read() if os.path.exists(chemin) else None
    chk(f"aperçu « {k} » : égal, à l'octet, à routeur.traiter(nouveau) dans un processus neuf ({len(NEUFS[k])} octets)",
        egal_octet(FICHIERS[k], NEUFS[k]), (None if FICHIERS[k] is None else len(FICHIERS[k]), NEUFS[k][:200]))
    chk(f"aperçu « {k} » : une réponse complète (ok vrai, etat, resultat)",
        FICHIERS[k] is not None and complet(k, FICHIERS[k].decode("utf-8")))

# 2. la construction REFUSE un aperçu incomplet : par le vrai chemin (un module inconnu : le routeur répond ok faux), et
# sur chaque manque
_base = json.loads(NEUFS["poutre"])
_ABIMES = {"ok faux": json.dumps({**_base, "ok": False}), "ok « true » (texte)": json.dumps({**_base, "ok": "true"}),
           "sans etat": json.dumps({k: v for k, v in _base.items() if k != "etat"}),
           "sans resultat": json.dumps({k: v for k, v in _base.items() if k != "resultat"}), "pas du JSON": "{pas du json"}


def refuse_incomplets() -> tuple:
    """La porte de la construction : le vrai chemin (un module inconnu, le routeur répond ok faux) refusé, chaque réponse
    abîmée refusée, la vraie acceptée. Rend (verdict, détail)."""
    try:
        SW.apercu("module_inexistant")
        refus = "accepté"
    except SystemExit as e:
        refus = str(e)
    acceptes = [n for n, t in _ABIMES.items() if complet("poutre", t)]
    return ("n'est pas une réponse complète" in refus and not acceptes and complet("poutre", NEUFS["poutre"].decode())), (refus[:200], acceptes)


_v_porte, _d_porte = refuse_incomplets()
chk("construire.apercu refuse d'écrire une réponse qui n'est pas complète (le vrai chemin : un module inconnu ; et ok faux, ok en texte, "
    "sans etat, sans resultat, pas du JSON)", _v_porte, _d_porte)

# 3. la liste du manifeste
chk("le manifeste liste les aperçus des 17 établis (« apercus » : clé → empreinte de 16 hex de son fichier), et moteur/apercus ne porte "
    "rien d'autre", liste_juste(MAN.get("apercus"), DOSSIER_APERCUS, ETABLIS), (MAN.get("apercus"), sorted(os.listdir(DOSSIER_APERCUS))))

# 4. la réponse « nouveau » ne dépend pas du jour : quatre dates simulées, chacune dans un processus neuf
_SONDE_JOUR = _DATE + """
import datetime, sys
sys.path.insert(0, {racine!r})
from api_web import age
sys.stdout.write(age.aujourdhui().isoformat() + " " + age.maintenant() + " " + datetime.date.today().isoformat())
"""
JOURS_VUS = [processus(_SONDE_JOUR.format(jour=j, racine=RACINE)).decode() for j in JOURS]
chk("la date simulée atteint le moteur : age.aujourdhui(), age.maintenant() et date.today() rendent chacun des quatre jours",
    JOURS_VUS == [f"{j} {j}T09:41 {j}" for j in JOURS], JOURS_VUS)
with ThreadPoolExecutor(max_workers=max(1, min(4, os.cpu_count() or 1))) as ex:
    _sous_date = list(ex.map(lambda kj: (kj, traiter_neuf(*kj)), [(k, j) for k in ETABLIS for j in JOURS]))
PAR_JOUR = {k: [r for (k_, _j), r in _sous_date if k_ == k] for k in ETABLIS}
for k in ETABLIS:
    chk(f"aperçu « {k} » : la même réponse sous les quatre dates ({', '.join(JOURS)}), égale à l'aperçu"
        + (" — age compris (S4-D8 : sa réponse « nouveau » ne dépend pas du jour)" if k == "age" else ""),
        meme_reponse(PAR_JOUR[k], FICHIERS[k]), [hashlib.sha256(r).hexdigest()[:12] for r in PAR_JOUR[k]])

# 5. la version du site
PAGES = {n: open(os.path.join(WEB, n), encoding="utf-8").read() for n in SW.pages_rendues()}
VERSION_LUE = (re.findall(r'<meta name="stab-version" content="([^"]*)">', PAGES.get("index.html", "")) or [""])[0]
chk("la version : la même balise stab-version sur les 24 pages, et la même valeur dans le manifeste",
    version_partout(PAGES, VERSION_LUE) and MAN.get("version") == VERSION_LUE, (VERSION_LUE, MAN.get("version")))
chk("la version des pages est celle que construire.version_site recalcule sur le site (fichiers publiés et pages rendues, version vide)",
    SW.version_site(WEB) == VERSION_LUE, (SW.version_site(WEB), VERSION_LUE))
_pub = SW.fichiers_publies(WEB)
chk("les fichiers de la version : js/**, css/**, moteur/** (aperçus compris, sans le manifeste), favicon.svg, _headers — aucune page, "
    "rien de vendor/ ni des outils", "moteur/manifeste.json" not in _pub and {"favicon.svg", "_headers", "js/modules.js", "css/site.css",
                                                                              "moteur/moteur.zip", "moteur/note.zip"} <= set(_pub)
    and all(f"moteur/apercus/{k}.json" in _pub for k in ETABLIS) and not any(p.endswith(".html") or p.startswith(("vendor/", "outils/")) for p in _pub)
    and all(p.startswith(("js/", "css/", "moteur/")) or p in ("favicon.svg", "_headers") for p in _pub), _pub[:8])


def octet_change(chemin: str):
    """Change UN octet du fichier (au milieu) ; rend de quoi le remettre."""
    with open(chemin, "rb") as f:
        avant = f.read()
    i = len(avant) // 2
    with open(chemin, "wb") as f:
        f.write(avant[:i] + bytes([avant[i] ^ 0x01]) + avant[i + 1:])
    return avant


def change_si(tmp: str, rel: str, publies=None) -> bool:
    """La version change quand UN octet de ``rel`` change (puis le fichier est remis)."""
    base = SW.version_site(tmp, publies=None if publies is None else publies(tmp))
    chemin = os.path.join(tmp, rel)
    avant = octet_change(chemin)
    try:
        apres = SW.version_site(tmp, publies=None if publies is None else publies(tmp))
    finally:
        with open(chemin, "wb") as f:
            f.write(avant)
    return apres != base


def inchange_si(tmp: str, rel: str, publies=None) -> bool:
    """La version NE change PAS quand ``rel`` (un fichier non publié) change ou apparaît (puis il est remis ou retiré)."""
    base = SW.version_site(tmp, publies=None if publies is None else publies(tmp))
    chemin = os.path.join(tmp, rel)
    existait = os.path.exists(chemin)
    if existait:
        avant = octet_change(chemin)
    else:
        os.makedirs(os.path.dirname(chemin), exist_ok=True)
        with open(chemin, "w", encoding="utf-8") as f:
            f.write("un fichier que Cloudflare ne publie pas\n")
    try:
        apres = SW.version_site(tmp, publies=None if publies is None else publies(tmp))
    finally:
        if existait:
            with open(chemin, "wb") as f:
                f.write(avant)
        else:
            os.remove(chemin)
    return apres == base


PUBLIES_CHANGENT = ("js/modules.js", "js/etabli.js", "js/travailleur.js", "css/site.css", "css/charte.css", "moteur/moteur.zip",
                    "moteur/note.zip", "moteur/apercus/age.json", "moteur/apercus/poutre.json", "favicon.svg", "_headers")
NON_PUBLIES = ("moteur/manifeste.json", "wrangler.jsonc", "README.md", "outils/nouveau.py", "construire.py")
TMP = tempfile.mkdtemp(prefix="site_version_")
try:
    # une VRAIE construction dans un dossier temporaire (le relevé, les paquets, les aperçus, la version, les pages)
    MAN_T = SW.construire(TMP)
    PAGES_T = {n: open(os.path.join(TMP, n), encoding="utf-8").read() for n in SW.pages_rendues()}
    chk("construit dans un dossier temporaire : la même version sur les 24 pages, dans le manifeste, recalculée — et la MÊME que celle du "
        "site (la construction est reproductible, aperçus compris)",
        version_partout(PAGES_T, MAN_T.get("version")) and SW.version_site(TMP) == MAN_T["version"] == VERSION_LUE
        and MAN_T.get("apercus") == MAN.get("apercus"), (MAN_T.get("version"), VERSION_LUE))
    _change = {rel: change_si(TMP, rel) for rel in PUBLIES_CHANGENT}
    chk(f"la version CHANGE si un octet d'un fichier publié change ({len(PUBLIES_CHANGENT)} fichiers : js, css, paquets, aperçus, favicon, _headers)",
        all(_change.values()), [r for r, v in _change.items() if not v])
    _js_neuf = os.path.join(TMP, "js", "nouveau.js")
    _v0 = SW.version_site(TMP)
    with open(_js_neuf, "w", encoding="utf-8") as f:
        f.write("export const X = 1;\n")
    _v_ajout = SW.version_site(TMP)
    os.remove(_js_neuf)
    _pages_g = SW.pages_rendues()
    _pages_g["dalle.html"] = _pages_g["dalle.html"].replace("</body>", " </body>", 1)
    chk("la version change aussi si un fichier publié APPARAÎT (js/nouveau.js) ou si un gabarit change (une page rendue d'un octet de plus)",
        _v_ajout != _v0 and SW.version_site(TMP, pages=_pages_g) != _v0 and SW.version_site(TMP) == _v0, (_v0, _v_ajout))
    _inchange = {rel: inchange_si(TMP, rel) for rel in NON_PUBLIES}
    chk(f"la version NE change PAS pour un fichier non publié ({', '.join(NON_PUBLIES)}) — le manifeste porte la version, il n'en fait pas partie",
        all(_inchange.values()), [r for r, v in _inchange.items() if not v])

    # 6. les MUTATIONS : chaque contrôle, rejoué sur une entrée abîmée, doit rougir
    MUT = {}
    _k = "poutre"
    MUT["à l'octet : un octet de l'aperçu changé"] = egal_octet(FICHIERS[_k][:100] + bytes([FICHIERS[_k][100] ^ 1]) + FICHIERS[_k][101:], NEUFS[_k])
    MUT["à l'octet : la réponse d'un AUTRE module"] = egal_octet(FICHIERS["dalle"], NEUFS[_k])
    MUT["complète : l'aperçu sans etat"] = complet(_k, json.dumps({k: v for k, v in json.loads(FICHIERS[_k]).items() if k != "etat"}))
    MUT["complète : l'aperçu sans resultat"] = complet(_k, json.dumps({k: v for k, v in json.loads(FICHIERS[_k]).items() if k != "resultat"}))
    MUT["liste : un établi retiré du manifeste"] = liste_juste({k: v for k, v in MAN["apercus"].items() if k != "age"}, DOSSIER_APERCUS, ETABLIS)
    MUT["liste : une clé de trop (predalle)"] = liste_juste({**MAN["apercus"], "predalle": "0" * 16}, DOSSIER_APERCUS, ETABLIS + ["predalle"])
    MUT["liste : une empreinte fausse"] = liste_juste({**MAN["apercus"], "age": "0" * 16}, DOSSIER_APERCUS, ETABLIS)
    _intrus = os.path.join(TMP, "moteur", SW.APERCUS, "intrus.json")
    shutil.copyfile(os.path.join(TMP, "moteur", SW.APERCUS, "age.json"), _intrus)
    MUT["liste : un fichier de trop dans moteur/apercus"] = liste_juste(MAN_T["apercus"], os.path.join(TMP, "moteur", SW.APERCUS), ETABLIS)
    os.remove(_intrus)
    # la date : une réponse qui DÉPEND du jour (la sonde ci-dessus, qui écrit le jour) — le contrôle doit la voir
    MUT["date : une réponse qui écrit le jour"] = meme_reponse([j.encode() for j in JOURS_VUS], JOURS_VUS[0].encode())
    MUT["date : trois dates au lieu de quatre"] = meme_reponse(PAR_JOUR["age"][:3], FICHIERS["age"])
    _p_mut = dict(PAGES_T)
    _p_mut["poutre.html"] = _p_mut["poutre.html"].replace(MAN_T["version"], "0" * 16, 1)
    MUT["version partout : une page d'une autre version"] = version_partout(_p_mut, MAN_T["version"])
    MUT["version partout : une page sans balise"] = version_partout({**PAGES_T, "taux-armature.html": re.sub(r'<meta name="stab-version"[^>]*>\n', "", PAGES_T["taux-armature.html"])}, MAN_T["version"])
    MUT["version partout : 23 pages au lieu de 24"] = version_partout({n: h for n, h in PAGES_T.items() if n != "predalle.html"}, MAN_T["version"])
    # la version sensible : une empreinte qui OUBLIE js/ (ou css/, ou moteur/) ne voit pas le changement
    for rel, racine_ in (("js/etabli.js", "js/"), ("css/site.css", "css/"), ("moteur/apercus/age.json", "moteur/"), ("_headers", "_headers")):
        MUT[f"version sensible : l'empreinte sans {racine_}"] = change_si(TMP, rel, publies=lambda d, r_=racine_: [p for p in SW.fichiers_publies(d) if not p.startswith(r_)])
    # … et une empreinte qui prend TOUT le dossier voit les fichiers non publiés
    def _tout(d):
        return sorted(os.path.relpath(os.path.join(a, f), d).replace(os.sep, "/") for a, ds, fs in os.walk(d) if "vendor" not in a.split(os.sep) for f in fs)
    MUT["version insensible : l'empreinte prend le manifeste"] = inchange_si(TMP, "moteur/manifeste.json", publies=_tout)
    MUT["version insensible : l'empreinte prend la configuration"] = inchange_si(TMP, "wrangler.jsonc", publies=_tout)
    # la porte de la construction : un aperçu incomplet ÉCRIT malgré tout serait vu par le contrôle de complétude
    _porte = SW.verifier_apercu
    SW.verifier_apercu = lambda cle, texte: None      # la porte retirée
    try:
        MUT["porte : construire.verifier_apercu retiré"] = refuse_incomplets()[0]
    finally:
        SW.verifier_apercu = _porte
    for n, v in MUT.items():
        chk(f"mutation — {n} : le contrôle rougit", v is False, v)
    chk(f"chaque famille de contrôles a ses mutations ({len(MUT)} : à l'octet, complétude, liste, date, version partout, sensible, "
        "insensible, porte de la construction)", len(MUT) >= 20
        and all(any(n.startswith(f) for n in MUT) for f in ("à l'octet", "complète", "liste", "date", "version partout", "version sensible",
                                                             "version insensible", "porte")), sorted(MUT))
finally:
    shutil.rmtree(TMP, ignore_errors=True)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)
