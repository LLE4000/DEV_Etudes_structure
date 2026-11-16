# -*- coding: utf-8 -*-
"""La GRILLE de 240 états du garde-corps et sa note PDF (acier/garde_corps/note.py, le générateur validé, celui de
l'écran Streamlit ET du site).

La grille : 5 types × 3 formes × « Cumuler qh et qv » non/oui × 4 fixations × 2 méthodes, le reste par défaut (les
saisies de l'écran du site, api_web/garde_corps.modifier). C'est celle de la relecture du 09/11/2026 : 86 de ses 240
états débordaient de leurs deux pages même en taille serrée (docs/site/DEFAUTS_STREAMLIT.md).

LA RÉFÉRENCE FIGÉE (``tests/ref_note_garde_corps_grille.json``) : pour chaque état, la note du générateur tel qu'il
était le 09/11/2026 (commit e4c173c), son nombre de pages, l'empreinte SHA-256 du texte de chaque page (pdftotext
-layout) et les débordements qu'il relevait. Un texte changé VOLONTAIREMENT depuis est un ÉCART NOMMÉ (``ECARTS_NOMMES``,
avec sa raison) : (a) refait la note avec l'ancien morceau remis et retrouve l'empreinte d'avant. Elle s'écrit UNE fois,
avec le générateur d'avant :

    python3 tests/test_note_garde_corps_grille.py --geler

Ce que la suite prouve :

- (a) les états qui TENAIENT dans leurs deux pages donnent EXACTEMENT la même note qu'avant : même nombre de pages, le
  texte de chaque page identique (empreinte) ;
- (b) CHACUN des 240 états donne une note COMPLÈTE, sans refus : aucune colonne ne déborde plus (le relevé du
  générateur, ``derniers_avertissements``, est vide — c'est lui qui fait refuser la note par le site) ; une note qui
  débordait a une page 3 de suite (et une seule), les autres en restent à deux pages ; et RIEN n'est perdu en route :
  pour chaque état qui reporte des blocs, un espion relève le CONTENU que le générateur pose sur les pages du document
  (titres de blocs et leurs taux, libellés de groupes, lignes d'hypothèses, paragraphes, chaque formule et son
  résultat, chaque tableau, chaque dessin) — la note avec sa page de suite pose EXACTEMENT le même contenu que l'ancienne
  mise en page en deux pages (qui le posait tout entier, mais en partie sous le cadre, sur la conclusion, sous le
  cartouche ou hors de la page) : rien de perdu, rien de doublé ; et le texte du PDF (pdftotext) porte chaque titre de
  bloc et chaque résultat de formule ;
- MUTATIONS : un générateur qui oublie de poser un bloc reporté en page 3, ou qui ne reporte rien (l'ancienne
  conduite), fait rougir (b).

    python3 tests/test_note_garde_corps_grille.py
"""
import collections
import contextlib
import copy
import hashlib
import itertools
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from acier.garde_corps import methode as MT  # noqa: E402
from acier.garde_corps import note as NT  # noqa: E402
from api_web import garde_corps as G  # noqa: E402
from ndc_pdf import data as ndc_data  # noqa: E402
from ndc_pdf import kit as KIT  # noqa: E402
from ndc_pdf import riche as RI  # noqa: E402

REF = os.path.join(RACINE, "tests", "ref_note_garde_corps_grille.json")
TYPES = ("barreaudage", "lisses", "panneau", "main_courante", "murale")
FORMES = ("droite", "L", "U")
CUMUL = (False, True)
FIXATIONS = ("platine", "laterale", "soudee", "scellee")
METHODES = ("els", "eurocode")
# une identification fixe (la date du jour changerait le texte de la note chaque jour)
INFOS = {"nom_projet": "Grille", "partie": "Garde-corps", "indice": "A", "date": "09/11/2026"}

OK, KO = [], []

# ÉCARTS NOMMÉS : les textes changés VOLONTAIREMENT depuis la référence du 09/11/2026, chacun avec sa raison — (titre de la
# section de methode.py, morceau d'aujourd'hui, morceau d'avant, raison). La référence ne se réécrit pas (elle vient du
# générateur d'avant la page de suite, le seul qui relevait les débordements) : pour (a), la note d'un état qui tenait et
# dont l'empreinte a bougé est refaite avec l'ANCIEN morceau remis le temps de la comparaison — l'empreinte d'avant doit
# revenir au bit près, donc rien d'autre n'a changé. Le texte d'aujourd'hui a son propre contrôle.
ECARTS_NOMMES = [
    ("Vérification à l'ELS",
     "dérivées de la nuance et de l'épaisseur de chaque pièce (tableau 3.3 ANB ; 3.8 et 3.10 ANB pour les tubes) : "
     "σ_adm = f_y/1,5 (156,7 N/mm² pour le S235 jusqu'à 16 mm)",
     "dérivées de la nuance : σ_adm = f_y/1,5 (156,7 N/mm² pour le S235)",
     "lot A, 2e passe (S6) : σ_adm dépend de la nuance ET de l'épaisseur de chaque pièce — tests/test_garde_corps_note.py"),
]


@contextlib.contextmanager
def anciens_textes(ecarts=None):
    """Les sections de méthode avec l'ANCIEN morceau de chaque écart nommé (le temps d'une comparaison à la référence)."""
    vrai = MT.SECTIONS
    sections = copy.deepcopy(vrai)
    for titre, neuf, ancien, _ in (ECARTS_NOMMES if ecarts is None else ecarts):
        sec = next(x for x in sections if x["titre"].startswith(titre))
        assert any(neuf in p for p in sec["paragraphes"]), f"écart nommé introuvable : {titre}"
        sec["paragraphes"] = [p.replace(neuf, ancien) for p in sec["paragraphes"]]
    MT.SECTIONS = sections
    try:
        yield
    finally:
        MT.SECTIONS = vrai


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:400]}"))


def grille():
    """Les 240 états : (clé lisible, état de la note du site)."""
    for t, f, cu, fx, m in itertools.product(TYPES, FORMES, CUMUL, FIXATIONS, METHODES):
        e = G.nouvel_etat()
        # la référence a été figée quand un garde-corps neuf s'appelait « Garde-corps 1 » (il s'appelle G1 depuis le
        # 11/11/2026, api_web/garde_corps.NOM_DEFAUT) : la note imprime ce nom, la grille le garde tel qu'il était
        e["cas"][0]["nom"] = "Garde-corps 1"
        for ch, v in ((["type"], t), (["geometrie", "forme"], f), (["charges", "cumul"], cu), (["fixation", "type"], fx),
                      (["methode", "mode"], m)):
            msg = G.modifier(e, ch, v)
            assert msg is None, (t, f, cu, fx, m, ch, msg)
        yield f"{t} · {f} · {'cumul' if cu else 'séparées'} · {fx} · {m}", G.normaliser(e)


def textes(octets: bytes):
    """Le texte de chaque page (pdftotext -layout)."""
    p = subprocess.run(["pdftotext", "-layout", "-", "-"], input=octets, capture_output=True, check=True)
    return p.stdout.decode("utf-8").split("\f")[:-1]


def empreinte(octets: bytes) -> dict:
    tx = textes(octets)
    return {"pages": len(tx), "sha": [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in tx]}


# ------------------------------------------------------------------ l'espion du contenu posé
class Espion:
    """Relève le CONTENU que le générateur pose sur les pages d'un document ``cible`` (pas sur un brouillon) : un
    élément par appel des briques de contenu, sans ce qui dépend de la place (largeurs, positions)."""

    def __init__(self):
        self.cible, self.journal, self.vrais = None, [], []

    def _sur(self, mod, nom, cle, methode=False):
        vrai = getattr(mod, nom)
        self.vrais.append((mod, nom, vrai))
        esp = self

        if methode:
            def f(obj, d, *a, **k):
                if d is esp.cible:
                    esp.journal.append((nom,) + cle(*a, **k))
                return vrai(obj, d, *a, **k)
        else:
            def f(d, *a, **k):
                if d is esp.cible:
                    esp.journal.append((nom,) + cle(*a, **k))
                return vrai(d, *a, **k)
        setattr(mod, nom, f)

    def __enter__(self):
        self._sur(NT.Note, "titre", lambda fr, txt, taux=None: (txt, None if taux is None else round(taux, 9)), methode=True)
        self._sur(NT.Note, "etiquette", lambda fr, txt: (txt,), methode=True)
        self._sur(NT.Note, "ligne_hyp", lambda fr, label, val, col=None: (label, val), methode=True)
        self._sur(RI, "paragraphe", lambda fr, s, *a, **k: (s,))
        self._sur(RI, "formule_calee", lambda fr, parts, res, *a, **k: (tuple(parts), res))
        self._sur(KIT, "table", lambda fr, cols, lignes, *a, **k: (tuple(c[0] for c in cols), repr(lignes), repr(k.get("note"))))
        self._sur(NT, "peindre_scene", lambda scene, *a, **k: (id(scene),))
        return self

    def __exit__(self, *exc):
        for mod, nom, vrai in reversed(self.vrais):
            setattr(mod, nom, vrai)

    def poser(self, calc, doc_meta, k, reports):
        """Le contenu posé par ``poser`` (les pages d'un cas) dans un document neuf, et les débordements."""
        fd, ch = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        try:
            d = KIT.Doc(ch, NT.A4L)
            self.cible, self.journal = d, []
            w = NT.poser(d, calc, doc_meta, k, reports)
            self.cible = None
            return collections.Counter(self.journal), w, d.n
        finally:
            os.remove(ch)


def plat(t: str) -> str:
    """Un texte sans blancs (les titres espacés lettre à lettre, les fines insécables) : pour y chercher un libellé."""
    return re.sub(r"\s+", "", t)


def resultat_texte(res: str) -> str:
    """Le résultat d'une formule (« \\res{0,45 \\t{kN}} ») tel que le PDF l'écrit, sans blancs."""
    return plat(re.sub(r"\\[a-zA-Z]+\{|[{}$\\]", "", res))


def complet(calc, doc_meta, k, reports, octets_note) -> list:
    """Les écarts de (b) pour un état qui reporte des blocs : le contenu de la note avec sa page de suite contre celui
    de l'ancienne mise en page (deux pages, rien de reporté), puis les titres et résultats dans le texte du PDF."""
    with Espion() as esp:
        neuf, w_neuf, n_neuf = esp.poser(calc, doc_meta, k, reports)
        ancien, _w, n_ancien = esp.poser(calc, doc_meta, k, ())
    out = []
    if w_neuf:
        out.append(("débordements", w_neuf))
    if n_ancien != 2 or n_neuf != 3:
        out.append(("pages", n_ancien, n_neuf))
    if neuf != ancien:
        out.append(("contenu", "perdu", list((ancien - neuf).items())[:3], "doublé", list((neuf - ancien).items())[:3]))
    texte = plat("".join(textes(octets_note)))
    for (nom, *args), n in neuf.items():
        if nom == "titre":
            cherche = plat(args[0].replace("…", ""))[:18]
        elif nom == "formule_calee" and args[1]:
            cherche = resultat_texte(args[1])
        else:
            continue
        if cherche and texte.count(cherche) < (1 if nom == "titre" else n):
            out.append(("absent du PDF", nom, cherche))
    return out


def geler():
    if os.path.exists(REF):
        print(f"{os.path.relpath(REF, RACINE)} existe : la référence ne se réécrit pas (la retirer d'abord, en le disant).")
        return 1
    out = {}
    for cle, e in grille():
        octets = NT.generer_note(e, e["cas"][0], INFOS)
        out[cle] = {**empreinte(octets), "variante": NT.derniere_variante, "debordements": list(NT.derniers_avertissements)}
    with open(REF, "w", encoding="utf-8") as f:
        json.dump({"_": "Note du garde-corps, générateur du 09/11/2026 (commit e4c173c) : pages, empreinte SHA-256 du texte "
                        "de chaque page (pdftotext -layout), variante retenue, débordements relevés — tests/test_note_garde_corps_grille.py",
                   "infos": INFOS, "etats": out}, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.write("\n")
    print(f"{len(out)} états figés, {sum(1 for v in out.values() if v['debordements'])} débordaient")
    return 0


def main():
    with open(REF, encoding="utf-8") as f:
        ref = json.load(f)
    chk("la référence figée porte les 240 états de la grille, avec l'identification fixe de la suite",
        len(ref["etats"]) == 240 and ref["infos"] == INFOS, (len(ref["etats"]), ref["infos"]))
    tenaient = {k for k, v in ref["etats"].items() if not v["debordements"]}
    print(f"--   référence : {len(tenaient)} états tenaient dans leurs deux pages, {240 - len(tenaient)} débordaient")
    t0 = time.time()
    ecarts_a, ecarts_b, suites, n_b, nommes = [], [], collections.Counter(), 0, 0
    for cle, e in grille():
        octets = NT.generer_note(e, e["cas"][0], INFOS)
        warn, reports, variante = list(NT.derniers_avertissements), NT.derniers_reports, NT.derniere_variante
        emp = empreinte(octets)
        r = ref["etats"].get(cle)
        if r is None:
            ecarts_a.append((cle, "absent de la référence"))
            continue
        if cle in tenaient and (emp["pages"] != r["pages"] or emp["sha"] != r["sha"]):
            with anciens_textes():                      # les écarts NOMMÉS remis : l'empreinte d'avant doit revenir
                emp = empreinte(NT.generer_note(e, e["cas"][0], INFOS))
            NT.derniere_variante = variante
            if emp["pages"] != r["pages"] or emp["sha"] != r["sha"]:
                ecarts_a.append((cle, emp["pages"], r["pages"], [i + 1 for i, (a, b) in enumerate(zip(emp["sha"], r["sha"])) if a != b]))
            else:
                nommes += 1
        # (b) aucune colonne ne déborde ; une page de suite pour (et seulement pour) ce qui débordait
        if warn:
            ecarts_b.append((cle, "déborde encore", warn))
        if (cle in tenaient) == bool(reports) or emp["pages"] != (3 if reports else 2):
            ecarts_b.append((cle, "page de suite", emp["pages"], reports))
        if reports:
            n_b += 1
            suites[tuple(sorted(f"page {c[0]}, colonne {c[1] + 1} : {b}" for c, b in reports))] += 1
            doc_meta = ndc_data.construire_doc(INFOS, date_defaut=datetime.today().strftime("%d/%m/%Y"))
            doc_meta["titre"] = "Note de calcul — garde-corps"
            calc = NT.Calcul(e, e["cas"][0])
            k = dict((nom, kk) for kk, nom in NT.VARIANTES)[NT.derniere_variante]
            ecarts_b += [(cle,) + x for x in complet(calc, doc_meta, k, reports, octets)]
    print(f"--   {240} notes faites en {time.time() - t0:.0f} s ; {nommes} états qui tenaient ne diffèrent de la référence que par les "
          f"écarts nommés ({' ; '.join(x[3] for x in ECARTS_NOMMES)})")
    chk(f"(a) les {len(tenaient)} états qui tenaient donnent EXACTEMENT la note d'avant (pages, texte de chaque page), aux seuls écarts "
        "NOMMÉS près (l'ancien texte remis, l'empreinte d'avant revient)",
        not ecarts_a and len(tenaient) == 154, ecarts_a[:4])
    for s, n in sorted(suites.items(), key=lambda x: -x[1]):
        print(f"--   page de suite ({n} états) : {' ; '.join(s)}")
    chk(f"(b) les 240 états ont une note complète, sans refus : aucune colonne ne déborde ; les {n_b} qui débordaient ont une "
        "page 3 de suite, qui pose EXACTEMENT le contenu de l'ancienne mise en page (rien de perdu, rien de doublé), titres "
        "et résultats de formules dans le texte du PDF",
        not ecarts_b and n_b == 240 - len(tenaient), ecarts_b[:4])

    # MUTATIONS : (a) reste strict malgré les écarts nommés — un écart NON nommé (une autre phrase de méthode changée) fait
    # rougir la comparaison d'un état ELS qui tenait, même avec les écarts nommés remis
    cle_els = "barreaudage · droite · séparées · platine · els"
    e_els = dict(grille())[cle_els]
    r_els = ref["etats"][cle_els]
    with anciens_textes():
        sha_nomme = empreinte(NT.generer_note(e_els, e_els["cas"][0], INFOS))["sha"]
    autre = ("Modèle", "Montant principal : console encastrée en pied", "Montant principal : console ENCASTRÉE en pied", "mutation")
    with anciens_textes(ECARTS_NOMMES + [autre]):
        sha_mutant = empreinte(NT.generer_note(e_els, e_els["cas"][0], INFOS))["sha"]
    sha_sans = empreinte(NT.generer_note(e_els, e_els["cas"][0], INFOS))["sha"]
    chk("mutation « une autre phrase de méthode changée, non nommée » : (a) rougit, même avec les écarts nommés remis ; sans eux "
        "aussi (le texte d'aujourd'hui n'est pas celui de la référence)",
        sha_nomme == r_els["sha"] and sha_mutant != r_els["sha"] and sha_sans != r_els["sha"], (cle_els,))

    # MUTATIONS : (b) doit rougir
    e = dict(grille())["barreaudage · U · cumul · platine · eurocode"]
    doc_meta = ndc_data.construire_doc(INFOS, date_defaut="")
    calc = NT.Calcul(e, e["cas"][0])
    fd, ch = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    try:
        k, _nom, reports, _w = NT.mise_en_page(calc, doc_meta, ch)
        with open(ch, "rb") as f:
            octets = f.read()
        vrai = NT.Suite.construire

        def oublie(self, d, echelle):                # la page 3 posée sans son dernier bloc
            self.blocs = self.blocs[:-1]
            return vrai(self, d, echelle)
        NT.Suite.construire = oublie
        try:
            rouge = complet(calc, doc_meta, k, reports, octets)
        finally:
            NT.Suite.construire = vrai
        chk("mutation « un bloc reporté oublié en page 3 » : (b) rougit (contenu perdu)", any(x[0] == "contenu" for x in rouge), rouge)
        vrai_m = NT.mise_en_page

        def ancienne(calc_, doc_, chemin):           # la boucle d'avant : les deux tailles, rien en page de suite
            for k_, nom_ in NT.VARIANTES:
                w_ = NT._essai(calc_, doc_, k_, chemin)
                if not w_:
                    break
            return k_, nom_, (), w_
        NT.mise_en_page = ancienne
        try:
            NT.generer_note(e, e["cas"][0], INFOS)
            rouge = list(NT.derniers_avertissements)
        finally:
            NT.mise_en_page = vrai_m
        chk("mutation « rien ne part en page de suite » (l'ancienne conduite) : (b) rougit (débordements relevés)", bool(rouge), rouge)
    finally:
        os.remove(ch)

    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
    for nom, info in KO:
        print("   -", nom, "|", str(info)[:300])
    return 1 if KO else 0


if __name__ == "__main__":
    sys.exit(geler() if "--geler" in sys.argv else main())
