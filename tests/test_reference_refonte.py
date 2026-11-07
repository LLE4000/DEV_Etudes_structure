# -*- coding: utf-8 -*-
"""Les CALCULS et les VALEURS AFFICHÉES de chaque module du site, figés AVANT la refonte de l'écran.

La refonte (ouvrir un calcul sans passer par « Nouvelle note », liste à gauche, dessin au centre, propriétés et
résultats à droite, panneaux repliables) change la MISE EN PAGE ; elle ne doit changer AUCUN calcul ni aucune valeur
affichée. Ce test le prouve, module par module : chaque cas est une suite de requêtes du routeur
(``api_web/routeur.traiter`` : nouveau, modifier chemin/valeur, ajouter, choisir…), rejouée sur l'API JSON, et sa
PHOTO est comparée à celle gelée dans ``tests/reference_refonte/<clé>.json``.

    /usr/bin/python3.12 tests/test_reference_refonte.py                 # compare tous les modules, nomme chaque écart
    /usr/bin/python3.12 tests/test_reference_refonte.py poutre dalle    # ces modules-là
    /usr/bin/python3.12 tests/test_reference_refonte.py --geler [clés]  # RÉGÉNÈRE les photos (après un changement VOULU)

Par module : au moins trois cas de RÉFÉRENCE (la note neuve, un exemple publié ou réaliste, un cas chargé près de
100 %) et au moins trois cas LIMITES (valeur minimale ou nulle admise, valeur hors domaine refusée avec son message,
élément ajouté ou supprimé, géométrie non conforme, champ vide « à compléter », note abîmée…). Assemblages : ses sept
configurations et les deux modes du pied de poteau. Chaque cas peut déclarer ce qu'il ATTEND (un message, une erreur,
un taux dans une bande, un niveau, un nombre d'éléments) : c'est contrôlé au gel comme à la comparaison, pour qu'un cas
qui ne montre plus ce qu'il annonce ne passe pas en silence.

CE QUI EST FIGÉ (la photo d'un cas, indépendante de la mise en page HTML)
- ``messages`` : le message rendu par chaque requête (refus d'une saisie), ``erreur`` : l'erreur du routeur (sans la
  trace) quand la requête échoue ;
- ``resume`` : taux, niveau, statut, gouvernant ; ``statut`` et ``phrase`` en TEXTE ;
- ``resultat`` : le cadre Résultat découpé en GROUPES (libellé, puis ses quantités « valeur unité », la quantité en
  « % » étant le taux) — voir l'extracteur ci-dessous ; ``calculs`` : de même pour les fenêtres de calcul (les
  formules détaillées : c'est là qu'un écart de 0,1 % dans un moteur se voit) ;
- ``avis`` : le texte de chaque avertissement rendu (``C.message``), où qu'il soit posé (données, résultat, éléments) ;
- ``champs`` : chaque champ de saisie, chemin → valeur AFFICHÉE (tous fragments confondus : un champ qui change de
  panneau ne compte pas pour un écart) ; ``verrous`` : les champs verrouillés ; ``menus`` : le nombre et l'empreinte
  des options de chaque menu et de chaque groupe de segments ;
- ``actions`` : ce qu'on peut faire d'autre que saisir — ajouter/retirer (``data-chemin`` sans ``data-sorte``),
  sélecteurs (``data-pk``), fenêtres de calcul (``data-calcul``), téléchargements, météo ;
- ``vues`` : clé, titre, court, légende de chaque vue, et l'EMPREINTE de ses nombres (la liste triée des textes qui
  portent un chiffre — un dessin à l'échelle change de cotes si le calcul change) ;
- ``elements`` : quand la note en a plusieurs, chaque élément ouvert à son tour (requête « activer ») : sa phrase et
  son résultat en groupes (le ``resume`` est celui de la NOTE : son élément le plus chargé) ;
- ``selecteurs`` (cas choisis) : le contenu de chaque sélecteur annoncé, découpé en groupes ; ``fichiers`` (cas
  choisis) : nom, type, taille et empreinte de chaque fichier à télécharger ;
- ``etat`` : l'état de la note à la fin du cas (ce que le navigateur enregistre), à l'identique.

CE QUI N'EST PAS FIGÉ : le balisage (balises, classes CSS, attributs de style, ordre des blocs), les identifiants
internes des dessins (motifs de hachures numérotés), les traits et positions des dessins (seuls leurs nombres le
sont), les textes du cadre Données hors champs (libellés, titres et résumés de sections : la refonte les réorganise),
la liste des éléments en HTML (remplacée par la phrase et le résultat de chacun), le texte de la légende des couleurs, les infobulles
(``title``), les boutons de la liste des éléments (``data-ajouter``, ``data-dupliquer`` : la refonte les déplace) et
le moteur dans le navigateur (Pyodide : ``site/web/verifier_site.js`` le contrôle).

L'EXTRACTEUR ne dépend PAS des classes CSS. Il dépend :
1. du TEXTE visible et de son ORDRE dans chaque fragment : chaque frontière de balise sépare deux mots (« f<sub>ck</sub> »
   se lit « f ck », quelle que soit la balise), les blancs sont réduits ; sont ignorés le contenu de ``<style>``,
   ``<script>``, ``<title>`` (infobulle d'un dessin), des ``<option>`` et des ``<textarea>`` (ce sont des champs) ;
2. de l'écriture des nombres du site (``C.fr`` : virgule décimale, espace fine insécable U+202F entre les milliers) ;
3. des attributs FONCTIONNELS que lit ``site/web/js/etabli.js`` — pas de style : ``data-chemin`` + ``data-sorte``
   (un champ), ``value``/``checked``/``selected``/``disabled``, ``data-valeur`` d'un segment, le segment retenu marqué
   par la classe ``on`` OU ``aria-pressed="true"`` OU ``aria-checked="true"`` (la seule classe lue, et elle a deux
   remplaçants admis), ``data-pk``, ``data-calcul``, ``data-telecharger``, ``data-meteo-chemin``, ``data-activer`` ;
4. de ``C.legende()`` (ôtée du résultat, c'est une légende fixe) et de ``C.message`` (espionnée pour les avis) ;
5. de ``api_web.age.aujourdhui`` / ``maintenant``, figés au 07/11/2026 (le plan de la météo dépend du jour).

Un GROUPE : on lit les mots dans l'ordre ; un nombre suivi d'une unité (« 402 mm² », « 30 % ») est une quantité ; un
mot qui porte une lettre ouvre un nouveau libellé dès qu'une quantité a été lue ; les symboles (« × », « ≤ », « — »)
suivent la quantité ou le libellé qu'ils touchent. Un groupe se lit donc (libellé, valeur unité…, taux %) :
["Armatures inférieures 2Ø16 ·", "402 mm²", "30 %"]. Les groupes sont GELÉS tels quels (lisibles) mais COMPARÉS par
leurs atomes (multi-ensembles) : chaque mot des libellés, et chaque quantité avec son contexte (les deux derniers mots
du libellé, son rang dans le groupe). Un bloc déplacé dans la page — même s'il colle autrement deux textes voisins —
n'est donc pas un écart ; un nombre, une unité, un taux, un mot changé en est un, et il est nommé (« A s = » : 402 mm²
→ 403 mm² ; texte disparu « … »).

PREUVES (07/11/2026, rejouables : ``tests/reference_refonte/mutations.py``) : une formule d'un moteur changée de 0,1 % (f_yd du moteur poutre ;
σ_c de la méthode directionnelle des soudures), un avertissement retiré, un message de refus retiré, une valeur de champ
affichée changée : ROUGE, et l'écart est nommé. Un changement de balisage SEUL (ligne de résultat, grande pastille,
sections en <details>, champs en <span>, segment retenu par aria-pressed, messages en <p>, blocs du résultat de la
poutre réordonnés) : VERT sur les 18 modules. ATTENTION, trouvé par la même épreuve : des modules RELISENT le balisage
de ``api_web/commun.py`` — le tableau ``<table class="vtab">`` de ``C.fenetre_calcul`` (catalogue, données béton : ôté
par expression régulière ; sol : vérifié, RuntimeError s'il manque), la ligne ``<div class="l2">…</span></div>`` de
``C.ligne_res`` (catalogue, données béton), la petite pastille ``<span class="chip c">0 %</span>`` de ``C.chip`` et le
texte « >à compléter< » (plancher). Changer ce balisage-là change ce que l'œil voit : le test le dit (et c'est juste).
"""
import collections
import copy
import hashlib
import html
import json
import math
import os
import re
import sys
import time
from html.parser import HTMLParser

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

import datetime as _dt  # noqa: E402

from api_web import age as AGE  # noqa: E402
from api_web import commun as C  # noqa: E402
from api_web import registre as REG  # noqa: E402
from api_web import routeur as R  # noqa: E402

DOSSIER = os.path.join(RACINE, "tests", "reference_refonte")
OK, KO = [], []

# le jour de l'écran, figé : le plan du bouton « Récupérer la météo » (et ses messages) en dépend
AGE.aujourdhui = lambda: _dt.date(2026, 11, 7)
AGE.maintenant = lambda: "2026-11-07T08:30"


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else "\n        | " + str(info)[:4000].replace("\n", "\n        | ")))


# ====================================================================== l'extracteur (aucune classe CSS, sauf « on »)
_IGNORES = {"style", "script", "title", "template"}
_MILLIERS = " "                                   # l'espace fine insécable de C.fr, entre les milliers
_BLANCS = re.compile(r"[^\S ]+")                 # tout blanc SAUF l'espace fine (elle est DANS un nombre)
_NOMBRE = r"[-−+]?(?:\d{1,3}(?: \d{3})+|\d+)(?:,\d+)?"
_JETON_NOMBRE = re.compile(r"([(\[{≈~]*)(" + _NOMBRE + r")(%?)([)\]},;:.]*)")
_UNITE = re.compile(r"(?:%|‰|°C?|[A-Za-zµΩ]{1,4}[²³⁴]?(?:[·./][A-Za-zµΩ]{1,4}[²³⁴]?){0,2})[,;:.)]*")
_MOTS_COURTS = {"a", "à", "au", "aux", "avec", "ce", "dans", "de", "des", "du", "en", "est", "et", "la", "le", "les", "ne",
                "ni", "ou", "par", "pas", "pour", "que", "qui", "sa", "se", "si", "son", "sous", "sur", "un", "une", "x",
                "dont", "soit", "sans", "vers", "fois", "plus", "max", "min", "mais", "car", "il", "on", "via", "pas"}


class _Lecteur(HTMLParser):
    """Un passage sur un fragment HTML : le texte visible (en morceaux, une frontière de balise = une séparation), les
    champs (``data-chemin`` + ``data-sorte``), les verrous, les options des menus et des segments, les actions."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.morceaux, self._ignore = [], 0
        self.champs, self.verrous, self.menus, self.actions = {}, set(), {}, set()
        self._select, self._options, self._choisie = None, [], None
        self._zone, self._zone_txt = None, []

    def _poser(self, chemin, valeur):
        self.champs.setdefault(chemin, valeur)       # un champ affiché deux fois : la première valeur

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        self.morceaux.append(" ")
        if tag in _IGNORES:
            self._ignore += 1
        if tag == "option" and self._select is not None:
            v = a.get("value", "")
            self._options.append(v)
            if "selected" in a and self._choisie is None:
                self._choisie = v
        chemin, sorte = a.get("data-chemin"), a.get("data-sorte")
        for cle, quoi in (("data-pk", "pk"), ("data-calcul", "calcul"), ("data-telecharger", "telecharger"),
                          ("data-meteo-chemin", "meteo")):
            if a.get(cle):
                self.actions.add(f"{quoi}:{a[cle]}")
        if chemin is None:
            return
        if sorte is None:                             # ajouter, retirer, lire un fichier : une ACTION, pas un champ
            self.actions.add(f"chemin:{chemin}")
            return
        if "disabled" in a:
            self.verrous.add(chemin)
        if tag == "select":
            self._select, self._options, self._choisie = chemin, [], None
        elif tag == "textarea":
            self._zone, self._zone_txt = chemin, []
        elif tag == "input" and a.get("type") == "checkbox":
            self._poser(chemin, "checked" in a)
        elif tag == "input":
            self._poser(chemin, a.get("value") or "")
        elif "data-valeur" in a:                      # un segment : ses options, et le retenu
            self.menus.setdefault(chemin, []).append(a["data-valeur"])
            classes = (a.get("class") or "").split()
            if "on" in classes or a.get("aria-pressed") == "true" or a.get("aria-checked") == "true":
                self._poser(chemin, a["data-valeur"])

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag in _IGNORES:
            self._ignore -= 1

    def handle_endtag(self, tag):
        self.morceaux.append(" ")
        if tag in _IGNORES and self._ignore:
            self._ignore -= 1
        if tag == "select" and self._select is not None:
            choisie = self._choisie if self._choisie is not None else (self._options[0] if self._options else "")
            self._poser(self._select, choisie)
            self.menus.setdefault(self._select, []).extend(self._options)
            self._select = None
        if tag == "textarea" and self._zone is not None:
            self._poser(self._zone, "".join(self._zone_txt))
            self._zone = None

    def handle_data(self, d):
        if self._zone is not None:
            self._zone_txt.append(d)
            return
        if not self._ignore and self._select is None:
            self.morceaux.append(d)


def lire(fragment: str) -> _Lecteur:
    lec = _Lecteur()
    lec.feed(fragment or "")
    lec.close()
    return lec


def jetons(fragment: str) -> list:
    """Les mots visibles d'un fragment, dans l'ordre (l'espace fine des milliers reste DANS le nombre)."""
    texte = "".join(lire(fragment).morceaux)
    return [j for j in _BLANCS.split(texte) if j and j != _MILLIERS]


def texte(fragment: str) -> str:
    return " ".join(jetons(fragment))


def _a_lettre(j: str) -> bool:
    return any(ch.isalpha() for ch in j)


def groupes(fragment: str) -> list:
    """Le fragment découpé en GROUPES [libellé, quantité, …] : « Armatures inférieures 2Ø16 · » puis « 402 mm² »,
    « 30 % ». Une quantité est un nombre et son unité (la mesure en « % » est un taux) ; un symbole (« × », « ≤ »,
    « — ») est gardé à sa place. Un groupe sans quantité (un avertissement, un titre) est gardé aussi."""
    out, libelle, valeurs = [], [], []
    js = jetons(fragment)
    i = 0
    while i < len(js):
        j = js[i]
        m = _JETON_NOMBRE.fullmatch(j)
        if m:
            q = (m.group(1) + m.group(2)).replace(_MILLIERS, " ")
            unite = m.group(3)
            if not unite and i + 1 < len(js) and _UNITE.fullmatch(js[i + 1]) and js[i + 1].lower() not in _MOTS_COURTS:
                unite = js[i + 1]
                i += 1
            valeurs.append(f"{q} {unite}".strip() + m.group(4))
        elif _a_lettre(j):
            if valeurs:
                out.append([" ".join(libelle)] + valeurs)
                libelle, valeurs = [], []
            libelle.append(j)
        elif valeurs:
            valeurs.append(j)
        else:
            libelle.append(j)
        i += 1
    if libelle or valeurs:
        out.append([" ".join(libelle)] + valeurs)
    return out


def nombres(fragment: str) -> str:
    """L'EMPREINTE des nombres d'une vue : les textes qui portent un chiffre, triés, joints par une espace."""
    return " ".join(sorted(j.replace(_MILLIERS, " ") for j in jetons(fragment) if re.search(r"\d", j)))


def _empreinte(x) -> str:
    return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]


# l'espion des avertissements : chaque C.message rendu pendant la requête finale d'un cas
_AVIS = []
_MESSAGE = C.message


def _message_espion(texte_, grave=False):
    _AVIS.append(("⚠ " if grave else "") + texte(texte_))
    return _MESSAGE(texte_, grave)


C.message = _message_espion


# ====================================================================== les cas : tests/reference_refonte/cas.py
def _charger_cas() -> dict:
    """Les cas, module par module (``CAS = {clé: [{nom, sorte, pas, attendu, selecteurs, fichiers}]}``) : ils vivent
    à côté de leurs photos ; ce fichier-ci est l'outil (rejouer, extraire, comparer, nommer)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("reference_refonte_cas", os.path.join(DOSSIER, "cas.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.CAS


CAS = _charger_cas()


# ====================================================================== la photo d'un cas
def _req(cle: str, q: dict) -> dict:
    return json.loads(R.traiter(json.dumps(dict(q, module=cle))))


def _sans_trace(erreur) -> str:
    return str(erreur or "").split("\nTraceback")[0].strip()


def photographier(cle: str, case: dict) -> dict:
    """Rejoue les requêtes du cas et rend sa photo (voir l'en-tête : ce qui est figé)."""
    etat, messages, rep = None, [], None
    for i, q in enumerate(case["pas"]):
        q = copy.deepcopy(q)
        if "etat" not in q and q.get("action") != "nouveau" and etat is not None:
            q["etat"] = etat
        if i == len(case["pas"]) - 1:
            del _AVIS[:]
        rep = _req(cle, q)
        if not rep.get("ok"):
            return {"messages": messages, "erreur": _sans_trace(rep.get("erreur"))}
        messages.append(rep.get("message"))
        etat = rep.get("etat", etat)
    if "resume" not in rep:                           # une requête finale qui n'est pas un calcul (sélecteur)
        return {"messages": messages, "html": groupes(rep.get("html", ""))}
    avis = list(_AVIS)
    lus = {k: lire(rep.get(k, "")) for k in ("donnees", "elements", "resultat", "volets", "phrase")}
    champs, verrous, menus, actions = {}, set(), {}, set()
    for lec in lus.values():
        for k, v in lec.champs.items():
            champs.setdefault(k, v)
        verrous |= lec.verrous
        actions |= lec.actions
        for k, v in lec.menus.items():
            menus.setdefault(k, v)
    resultat = (rep.get("resultat") or "").replace(C.legende(), "")
    photo = {
        "messages": messages,
        "resume": rep["resume"],
        "statut": texte(rep.get("statut_html", "")),
        "phrase": texte(rep.get("phrase", "")),
        "resultat": groupes(resultat),
        "calculs": groupes(rep.get("volets", "")),
        "avis": avis,
        "champs": {k: (v.replace(_MILLIERS, " ") if isinstance(v, str) else v) for k, v in sorted(champs.items())},
        "verrous": sorted(verrous),
        "menus": {k: f"{len(v)} · {_empreinte(v)}" for k, v in sorted(menus.items())},
        "actions": sorted(actions),
        "vues": [{"cle": v.get("cle"), "titre": v.get("titre"), "court": v.get("court"), "legende": v.get("legende"),
                  "nombres": nombres(v.get("svg", ""))} for v in rep.get("vues", [])],
        "etat": etat,
    }
    n = (rep.get("elements") or "").count("data-activer=")
    if n > 1:                                         # chaque élément ouvert à son tour : sa phrase et son résultat
        photo["elements"] = []                        # (le résumé, lui, est celui de la NOTE : son élément le plus chargé)
        for k in range(n):
            a = _req(cle, {"action": "activer", "etat": copy.deepcopy(etat), "index": k})
            photo["elements"].append({"phrase": texte(a.get("phrase", "")),
                                      "resultat": groupes((a.get("resultat") or "").replace(C.legende(), ""))}
                                     if a.get("ok") else {"erreur": _sans_trace(a.get("erreur"))})
    if case.get("selecteurs"):
        roles = list(dict.fromkeys(re.findall(r'data-pk="([^"]+)"', rep.get("resultat", "") + rep.get("donnees", ""))))
        photo["selecteurs"] = {}
        for role in roles:
            s = _req(cle, {"action": "selecteur", "etat": copy.deepcopy(etat), "role": html.unescape(role)})
            photo["selecteurs"][html.unescape(role)] = groupes(s["html"]) if s.get("ok") else {"erreur": _sans_trace(s.get("erreur"))}
    if case.get("fichiers"):
        tout = "".join(rep.get(k, "") for k in ("donnees", "elements", "resultat", "volets"))
        photo["fichiers"] = {}
        for quoi in dict.fromkeys(re.findall(r'data-telecharger="([^"]*)"', tout)):
            f = _req(cle, {"action": "telecharger", "etat": copy.deepcopy(etat), "quoi": html.unescape(quoi)})
            photo["fichiers"][html.unescape(quoi)] = (
                {"nom": f["fichier"]["nom"], "type": f["fichier"]["type"], "octets": len(f["fichier"]["texte"].encode("utf-8")),
                 "empreinte": _empreinte(f["fichier"]["texte"])} if f.get("ok") else {"erreur": f.get("erreur")})
    return photo


# ====================================================================== ce qu'un cas annonce, contrôlé
def manquements(photo: dict, attendu: dict) -> list:
    """Ce que le cas annonce (``attendu``) et que sa photo ne montre pas."""
    out = []
    res = photo.get("resume") or {}
    if attendu.get("erreur"):
        if "erreur" not in photo:
            out.append("une erreur du routeur était attendue")
    elif "erreur" in photo:
        out.append(f"erreur inattendue : {photo['erreur'][:200]}")
    if attendu.get("message") and not (photo.get("messages") and photo["messages"][-1]):
        out.append("un message de refus était attendu à la dernière requête")
    if attendu.get("sans_message") and any(photo.get("messages") or []):
        out.append(f"aucun message attendu, reçu : {[m for m in photo['messages'] if m]}")
    if "taux" in attendu:
        lo, hi = attendu["taux"]
        t = res.get("taux")
        if t is None or not lo <= t <= hi:
            out.append(f"taux attendu dans [{lo}, {hi}] %, obtenu {t}")
    if "niveau" in attendu and res.get("niveau") not in attendu["niveau"]:
        out.append(f"niveau attendu {attendu['niveau']!r}, obtenu {res.get('niveau')!r}")
    if "elements" in attendu and len(photo.get("elements") or [res]) != attendu["elements"]:
        out.append(f"{attendu['elements']} élément(s) attendu(s), {len(photo.get('elements') or [res])} obtenu(s)")
    if "avis" in attendu and not any(attendu["avis"] in a for a in photo.get("avis", [])):
        out.append(f"avertissement attendu contenant « {attendu['avis']} »")
    if "texte" in attendu:
        tout = json.dumps([photo.get("resultat"), photo.get("avis"), photo.get("messages")], ensure_ascii=False)
        if attendu["texte"] not in tout:
            out.append(f"texte attendu dans le résultat : « {attendu['texte']} »")
    return out


# ====================================================================== nommer les écarts
def _court(x, n=160) -> str:
    s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)
    return s if len(s) <= n else s[:n - 1] + "…"


def _ecarts_json(a, b, chemin="", out=None, n=12) -> list:
    out = [] if out is None else out
    if len(out) >= n:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in b:
                out.append(f"{chemin}/{k} : disparu (était {_court(a[k], 80)})")
            elif k not in a:
                out.append(f"{chemin}/{k} : apparu ({_court(b[k], 80)})")
            else:
                _ecarts_json(a[k], b[k], f"{chemin}/{k}", out, n)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            _ecarts_json(x, y, f"{chemin}/{i}", out, n)
    elif a != b:
        out.append(f"{chemin or '/'} : {_court(a, 80)} → {_court(b, 80)}")
    return out


def _contexte(mots: list) -> str:
    return " ".join([m for m in mots if _a_lettre(m)][-2:])


def _libelles(gs: list) -> dict:
    """(contexte, rang, valeur) → le libellé complet du groupe (pour nommer un écart lisiblement)."""
    out = {}
    for g in gs:
        ctx = _contexte(g[0].split(" ") if g[0] else [])
        for i, v in enumerate(g[1:]):
            out.setdefault((ctx, i, v), g[0])
            out.setdefault((ctx, i), g[0])
    return out


def _atomes(gs: list) -> collections.Counter:
    """Ce que la comparaison COMPTE dans une liste de groupes : chaque mot des libellés (« mot ») et chaque quantité avec
    son contexte — les deux derniers mots à lettre de son libellé et son rang dans le groupe (« val »). Deux blocs de
    texte voisins qui fusionnent autrement (un bloc déplacé dans la page) donnent les mêmes atomes : ce n'est pas un écart."""
    c = collections.Counter()
    for g in gs:
        mots = g[0].split(" ") if g[0] else []
        c.update(("mot", m) for m in mots)
        ctx = _contexte(mots)
        c.update(("val", ctx, i, v) for i, v in enumerate(g[1:]))
    return c


def _mots_ecartes(gs: list, diff: collections.Counter) -> list:
    """Les mots de ``diff`` retrouvés dans l'ordre des groupes ``gs``, réunis en passages contigus."""
    reste, passages, courant = collections.Counter(diff), [], []
    for g in gs:
        for m in (g[0].split(" ") if g[0] else []):
            if reste[("mot", m)] > 0:
                reste[("mot", m)] -= 1
                courant.append(m)
            elif courant:
                passages.append(" ".join(courant))
                courant = []
        if courant and g[1:]:
            passages.append(" ".join(courant))
            courant = []
    if courant:
        passages.append(" ".join(courant))
    return passages


def _ecarts_groupes(quoi: str, gele: list, vu: list, n=10) -> list:
    """Deux listes de groupes comparées par leurs ATOMES (multi-ensembles) : une quantité changée est nommée par son
    contexte (« A s = » : 402 mm² → 403 mm²), un texte disparu ou apparu est rendu dans l'ordre de lecture."""
    a, b = _atomes(gele), _atomes(vu)
    if a == b:
        return []
    manque, trop = a - b, b - a
    vm = [k for k in manque.elements() if k[0] == "val"]
    vt = [k for k in trop.elements() if k[0] == "val"]
    lg, lv = _libelles(gele), _libelles(vu)
    out = []
    for k in vm:
        lib = _court(lg.get(k[1:], k[1]), 70)
        p = next((t for t in vt if t[1] == k[1] and t[2] == k[2]), None)
        if p is not None:
            vt.remove(p)
            out.append(f"{quoi} « {lib} » : {k[3]} → {p[3]}")
        else:
            out.append(f"{quoi} « {lib} » : {k[3]} disparu")
    out += [f"{quoi} « {_court(lv.get(t[1:], t[1]), 70)} » : {t[3]} apparu" for t in vt]
    out += [f"{quoi} : texte disparu « {_court(x, 150)} »" for x in _mots_ecartes(gele, manque)]
    out += [f"{quoi} : texte apparu « {_court(x, 150)} »" for x in _mots_ecartes(vu, trop)]
    return out[:n] + ([f"{quoi} : … et {len(out) - n} autre(s) écart(s)"] if len(out) > n else [])


def _ecarts_nombres(quoi: str, gele: str, vu: str) -> list:
    a, b = collections.Counter(gele.split(" ")), collections.Counter(vu.split(" "))
    if a == b:
        return []
    return [f"{quoi} : nombres disparus {_court(sorted((a - b).elements()), 120)} ; apparus {_court(sorted((b - a).elements()), 120)}"]


def ecarts(gele: dict, vu: dict) -> list:
    """Chaque différence entre la photo gelée et la photo du jour, nommée."""
    out = []
    for k in sorted(set(gele) | set(vu)):
        if k not in vu:
            out.append(f"{k} : disparu de la photo (était {_court(gele[k], 200)})")
            continue
        if k not in gele:
            out.append(f"{k} : apparu dans la photo ({_court(vu[k], 200)})")
            continue
        a, b = gele[k], vu[k]
        if a == b:
            continue
        if k in ("resultat", "calculs", "html"):
            out += _ecarts_groupes({"resultat": "résultat", "calculs": "calcul", "html": "sélecteur"}[k], a, b)
        elif k == "vues":
            va, vb = {v["cle"]: v for v in a}, {v["cle"]: v for v in b}
            if [v["cle"] for v in a] != [v["cle"] for v in b]:
                out.append(f"vues : {[v['cle'] for v in a]} → {[v['cle'] for v in b]}")
            for c in va:
                if c in vb:
                    for champ in ("titre", "court", "legende"):
                        if va[c].get(champ) != vb[c].get(champ):
                            out.append(f"vue {c} · {champ} : « {va[c].get(champ)} » → « {vb[c].get(champ)} »")
                    out += _ecarts_nombres(f"vue {c}", va[c]["nombres"], vb[c]["nombres"])
        elif k == "selecteurs" and isinstance(a, dict) and isinstance(b, dict):
            for role in sorted(set(a) | set(b)):
                if role not in a or role not in b:
                    out.append(f"sélecteur {role} : {'apparu' if role not in a else 'disparu'}")
                elif a[role] != b[role]:
                    out += (_ecarts_groupes(f"sélecteur {role}", a[role], b[role]) if isinstance(a[role], list) and isinstance(b[role], list)
                            else [f"sélecteur {role} : {_court(a[role])} → {_court(b[role])}"])
        elif k == "elements" and isinstance(a, list) and isinstance(b, list) and len(a) == len(b) \
                and all(isinstance(x, dict) and "resultat" in x for x in a + b):
            for i, (x, y) in enumerate(zip(a, b)):
                if x["phrase"] != y["phrase"]:
                    out.append(f"élément {i + 1} · phrase : « {x['phrase']} » → « {y['phrase']} »")
                out += _ecarts_groupes(f"élément {i + 1} · résultat", x["resultat"], y["resultat"])
        elif k in ("verrous", "actions"):
            out.append(f"{k} : disparus {sorted(set(a) - set(b))[:8]} ; apparus {sorted(set(b) - set(a))[:8]}")
        elif k == "avis":
            ca, cb = collections.Counter(a), collections.Counter(b)
            out += [f"avis disparu : « {_court(x, 150)} »" for x in (ca - cb).elements()]
            out += [f"avis apparu : « {_court(x, 150)} »" for x in (cb - ca).elements()]
            if ca == cb:
                out.append("avis : même contenu, autre ordre")
        elif k == "champs":
            out += [f"champ {e}" for e in _ecarts_json(a, b, "", None, 12)]
        else:
            out += [f"{k}{e}" if e.startswith("/") else f"{k} {e}" for e in _ecarts_json(a, b, "", None, 12)]
    return out


# ====================================================================== écrire et relire les photos
def _dump(x, retrait=0) -> str:
    """Un JSON lisible et stable pour git : les dictionnaires dépliés, une liste de nombres ou de textes sur UNE ligne,
    un groupe par ligne, l'état de la note sur une ligne."""
    pad, pad2 = " " * retrait, " " * (retrait + 1)
    if isinstance(x, dict):
        if not x:
            return "{}"
        lignes = []
        for k, v in x.items():
            val = json.dumps(v, ensure_ascii=False) if k in ("etat", "pas") else _dump(v, retrait + 1)
            lignes.append(f"{pad2}{json.dumps(k, ensure_ascii=False)}: {val}")
        return "{\n" + ",\n".join(lignes) + f"\n{pad}}}"
    if isinstance(x, list):
        if not x:
            return "[]"
        if all(not isinstance(v, (dict, list)) for v in x):
            return json.dumps(x, ensure_ascii=False)
        return "[\n" + ",\n".join(pad2 + _dump(v, retrait + 1) for v in x) + f"\n{pad}]"
    return json.dumps(x, ensure_ascii=False)


def _fichier(cle: str) -> str:
    return os.path.join(DOSSIER, f"{cle}.json")


def geler(cle: str) -> None:
    sortie = {"module": cle,
              "lire": "tests/test_reference_refonte.py : la photo de chaque cas (voir son en-tête) ; --geler la régénère",
              "cas": []}
    for case in CAS[cle]:
        t0 = time.time()
        photo = photographier(cle, case)
        manque = manquements(photo, case["attendu"])
        chk(f"{cle} · {case['nom']} : gelé, et montre ce qu'il annonce ({time.time() - t0:.2f} s)", not manque, manque)
        sortie["cas"].append({"nom": case["nom"], "sorte": case["sorte"], "attendu": case["attendu"], "pas": case["pas"],
                              "photo": photo})
    os.makedirs(DOSSIER, exist_ok=True)
    with open(_fichier(cle), "w", encoding="utf-8") as f:
        f.write(_dump(sortie) + "\n")


def comparer(cle: str) -> None:
    chemin = _fichier(cle)
    if not os.path.exists(chemin):
        chk(f"{cle} : la photo gelée existe ({os.path.relpath(chemin, RACINE)})", False, "lancer --geler")
        return
    with open(chemin, encoding="utf-8") as f:
        gele = json.load(f)
    noms = [c["nom"] for c in CAS[cle]]
    chk(f"{cle} : les cas gelés sont ceux du test ({len(noms)})", [c["nom"] for c in gele["cas"]] == noms,
        f"gelés {[c['nom'] for c in gele['cas']]} ≠ test {noms} : regeler (--geler {cle})")
    definis = {c["nom"]: c for c in CAS[cle]}
    for g in gele["cas"]:
        t0 = time.time()
        d = definis.get(g["nom"])
        # les requêtes rejouées sont celles du GEL : une entrée qui aurait changé (un exemple de module modifié) se voit
        if d is not None and json.loads(json.dumps(d["pas"])) != g["pas"]:
            chk(f"{cle} · {g['nom']} : ses requêtes sont celles du gel", False,
                "les requêtes du cas ont changé depuis le gel : " + "; ".join(_ecarts_json(g["pas"], json.loads(json.dumps(d["pas"])))[:4]))
        vu = photographier(cle, {**g, "selecteurs": "selecteurs" in g["photo"], "fichiers": "fichiers" in g["photo"]})
        e = ecarts(g["photo"], vu) + [f"attendu : {m}" for m in manquements(vu, g.get("attendu", {}))]
        chk(f"{cle} · {g['nom']} [{g['sorte']}] : calcul et affichage identiques ({time.time() - t0:.2f} s)", not e, "\n".join(e))


def main(argv) -> int:
    t0 = time.time()
    geler_ = "--geler" in argv
    choisis = [a for a in argv if not a.startswith("--")] or REG.cles()
    inconnus = [c for c in choisis if c not in REG.cles()]
    chk("les modules demandés sont au registre", not inconnus, inconnus)
    if not [a for a in argv if not a.startswith("--")]:
        chk(f"chaque module du registre a ses cas ({len(REG.cles())})", set(CAS) >= set(REG.cles()), sorted(set(REG.cles()) - set(CAS)))
    for cle in choisis:
        if cle not in CAS:
            continue
        refs = sum(c["sorte"] == "référence" for c in CAS[cle])
        lims = sum(c["sorte"] == "limite" for c in CAS[cle])
        chk(f"{cle} : au moins 3 cas de référence et 3 cas limites ({refs} + {lims})", refs >= 3 and lims >= 3)
        (geler if geler_ else comparer)(cle)
    print(f"\nDurée : {time.time() - t0:.1f} s")
    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
    for nom, info in KO:
        print("   -", nom, "|", str(info).splitlines()[0][:200] if str(info) else "")
    return 1 if KO else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
