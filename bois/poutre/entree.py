# -*- coding: utf-8 -*-
"""Poutre bois — l'ENTRÉE du moteur : la poutre neuve et la relecture STRICTE d'une poutre reçue (l'écran du site, un
fichier .stab).

La poutre (mm, m, kN/m, kN/m²) ::

    {"nom": "P1",
     "classe": "C24",                                  # EN 338 (C14 … C50) ou EN 14080 (GL24h … GL32c) : fixe le TYPE
     "standard": true, "section": "75x225",            # une section standard du type du bois …
     "b": 75.0, "h": 225.0,                            # … ou sur mesure (standard : false) ; toujours la section effective
     "L": 4.0,                                         # la portée entre axes des appuis (m)
     "charges": {"mode": "lineaire",                   # « lineaire » : G, Q en kN/m ; « surfacique » : g, q en kN/m² × entraxe (m)
                 "G": 0.6, "Q": 0.8, "g": 1.5, "q": 2.0, "entraxe": 0.4,
                 "poids_propre": true,                 # le poids propre ρ_mean·b·h·9,81 est ajouté à G
                 "nature": "A",                        # la nature de la charge variable : fixe sa durée et ψ₂ …
                 "duree": "moyen", "psi2": 0.3},       # … que l'on ne saisit que pour la nature « autre »
     "appui": {"verifier": true, "a": 100.0, "debord": 0.0},     # la longueur d'appui a et le débord au-delà (mm)
     "maintien": {"type": "continu", "entraxe": 1.0},  # « continu », « appuis », « entretoises » (à l'entraxe donné, m)
     "service": 1, "gamma_G": 1.35, "gamma_Q": 1.5, "systeme": false,
     "lim_inst": 300, "lim_fin": 250,                  # w_inst ≤ L/300, w_net,fin ≤ L/250
     "contre_fleche": 0.0}                             # mm

Règle de ``normaliser`` — RIEN n'est ramené en silence :
- une clé ABSENTE (ou ``null``) prend son défaut (``donnees.DEFAUTS``) ;
- une clé INCONNUE, un type faux (un texte pour un nombre, 1.0 pour une classe de service, 0 pour une case), une
  valeur hors de ses bornes (``donnees.BORNES``) ou hors de son menu lèvent ``ValueError``, avec un message en
  français qui nomme le champ ;
- une valeur DÉDUITE d'une autre — b et h d'une section standard ; la durée et ψ₂ d'une nature autre que « autre » —
  est remplie quand elle manque, et REFUSÉE quand elle contredit ce qu'elle doit valoir (une note qui dirait
  « habitation, ψ₂ = 0,7 » n'est pas corrigée en douce : elle est refusée) ;
- trois contrôles croisés : la longueur d'appui reste plus courte que la portée (a < L), l'entraxe des entretoises
  ne dépasse pas la portée, la section standard est du type du bois.

``normaliser(normaliser(p)) == normaliser(p)``.
"""
import copy
import math
from typing import Dict, List, Optional, Tuple

from . import donnees as D

CLES = ("nom", "classe", "standard", "section", "b", "h", "L", "charges", "appui", "maintien", "service", "gamma_G", "gamma_Q",
        "systeme", "lim_inst", "lim_fin", "contre_fleche")
CLES_GROUPES = {"charges": ("mode", "G", "Q", "g", "q", "entraxe", "poids_propre", "nature", "duree", "psi2"),
                "appui": ("verifier", "a", "debord"),
                "maintien": ("type", "entraxe")}
LIBELLES = {"nom": "le nom", "classe": "la classe de résistance", "standard": "le choix « section standard »",
            "section": "la section standard", "charges": "les charges", "appui": "l'appui", "maintien": "le maintien latéral",
            "service": "la classe de service", "systeme": "l'effet système", "lim_inst": "la limite de flèche instantanée",
            "lim_fin": "la limite de flèche finale", "charges/mode": "le mode de saisie des charges",
            "charges/poids_propre": "la case « poids propre compté »", "charges/nature": "la nature de la charge variable",
            "charges/duree": "la durée de la charge variable", "appui/verifier": "la case « vérifier l'appui »",
            "maintien/type": "le type de maintien latéral"}
LIBELLES.update({k: v[3] for k, v in D.BORNES.items()})


def _txt(x: float) -> str:
    """Un nombre dans un message : virgule décimale, sans zéros inutiles (0,5 ; 30 ; 1,35)."""
    return f"{x:.6g}".replace(".", ",")


def _dire(x) -> str:
    """Une valeur reçue, redite dans un message : un nombre à la virgule, le reste tel quel."""
    if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x):
        return _txt(x)
    return repr(x)


def _maj(s: str) -> str:
    """La capitale d'un début de phrase — sauf sur un symbole (« ψ₂ », « γ_G » ne deviennent pas « Ψ₂ », « Γ_G »)."""
    return s[:1].upper() + s[1:] if s[:1].isascii() else s


# ------------------------------------------------------------------ les sections standard
def tag(b: float, h: float) -> str:
    """« 75x225 » : la clé d'une section (mm)."""
    return f"{b:g}x{h:g}"


def designation(nom: str) -> str:
    """« 75x225 » → « 75 × 225 » : la section telle qu'on l'écrit."""
    return str(nom).replace("x", " × ")


def lire_tag(texte) -> str:
    """Une section écrite « 75 × 225 », « 75x225 » ou « 75 X 225 » → sa clé « 75x225 »."""
    return str(texte).replace("×", "x").replace("X", "x").replace(" ", "").replace(" ", "").replace(" ", "")


def type_classe(classe: str) -> str:
    """Le type du bois d'une classe : « massif » (C…) ou « lamelle » (GL…)."""
    if classe not in D.CLASSES:
        raise ValueError(f"La classe de résistance « {classe} » est inconnue ({', '.join(D.CLASSES)})")
    return D.CLASSES[classe]["type"]


def sections(typ: str) -> List[Dict]:
    """Les sections standard d'un type, dans l'ordre de la liste : [{"nom": "75x225", "b": 75.0, "h": 225.0}, …]."""
    return [{"nom": tag(b, h), "b": float(b), "h": float(h)} for b, h in D.SECTIONS[typ]]


def section_standard(typ: str, nom: str) -> Tuple[float, float]:
    """(b, h) en mm de la section standard ``nom`` du type ``typ`` ; ValueError si elle n'est pas dans la liste."""
    for b, h in D.SECTIONS[typ]:
        if tag(b, h) == nom:
            return float(b), float(h)
    raise ValueError(f"La section standard « {designation(nom)} » n'existe pas en {D.COURTS_TYPE[typ]} "
                     f"({len(D.SECTIONS[typ])} sections, de {designation(tag(*D.SECTIONS[typ][0]))} à {designation(tag(*D.SECTIONS[typ][-1]))})")


def section_proche(typ: str, b: float, h: float) -> str:
    """La section standard du type la plus PROCHE de b × h : la même si elle existe, sinon celle dont l'aire est la
    plus voisine (à égalité, la hauteur la plus voisine). Sert quand on revient de « sur mesure » à « standard »."""
    return min(sections(typ), key=lambda s: (abs(s["b"] * s["h"] - b * h), abs(s["h"] - h), abs(s["b"] - b)))["nom"]


# ------------------------------------------------------------------ la poutre neuve
def nouvelle_poutre(nom: str = D.DEFAUTS["nom"]) -> Dict:
    """La poutre neuve (conception § 1.5) : C24, 75 × 225, L = 4,00 m, G = 0,6 et Q = 0,8 kN/m (habitation), poids
    propre compté, classe de service 1, appui de 100 mm vérifié, maintien continu, L/300 et L/250."""
    p = copy.deepcopy(D.DEFAUTS)
    p["nom"] = nom
    return p


# ------------------------------------------------------------------ la relecture stricte
def _nombre(x, chemin: str) -> float:
    lib = LIBELLES[chemin]
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{_maj(lib)} : un nombre est attendu (reçu : {x!r})")
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"{_maj(lib)} : un nombre fini est attendu (reçu : {x!r})")
    mini, maxi, unite, _ = D.BORNES[chemin]
    if v < mini or v > maxi:
        u = f" {unite}" if unite else ""
        raise ValueError(f"{_maj(lib)} doit rester entre {_txt(mini)} et {_txt(maxi)}{u} (reçu : {_txt(v)})")
    return v + 0.0                      # jamais « -0.0 »


def _parmi(x, chemin: str, permis, dire=str):
    """``x`` est l'une des valeurs permises, DU MÊME TYPE (1.0 ou True ne sont pas la classe de service 1)."""
    if isinstance(x, bool) or not any(type(x) is type(q) and x == q for q in permis):
        raise ValueError(f"{_maj(LIBELLES[chemin])} : {x!r} n'est pas proposé ({', '.join(dire(q) for q in permis)})")
    return x


def _case(x, chemin: str) -> bool:
    if not isinstance(x, bool):
        raise ValueError(f"{_maj(LIBELLES[chemin])} : vrai ou faux est attendu (reçu : {x!r})")
    return x


def _groupe(poutre: Dict, nom: str) -> Dict:
    """Un groupe de la poutre (charges, appui, maintien) : un dictionnaire sans clé inconnue ; absent : vide."""
    g = poutre.get(nom)
    if g is None:
        return {}
    if not isinstance(g, dict):
        raise ValueError(f"{_maj(LIBELLES[nom])} : un groupe de champs est attendu (reçu : {type(g).__name__})")
    inconnues = sorted(str(k) for k in g if k not in CLES_GROUPES[nom])
    if inconnues:
        raise ValueError(f"{_maj(LIBELLES[nom])} : champ(s) inconnu(s) {', '.join(inconnues)}")
    return g


def normaliser(poutre: Dict) -> Dict:
    """Une copie COMPLÈTE et CONTRÔLÉE de la poutre (la règle est en tête du fichier)."""
    if not isinstance(poutre, dict):
        raise ValueError(f"Poutre illisible : un groupe de champs est attendu (reçu : {type(poutre).__name__})")
    inconnues = sorted(str(k) for k in poutre if k not in CLES)
    if inconnues:
        raise ValueError(f"Poutre : champ(s) inconnu(s) {', '.join(inconnues)}")

    def val(source: Dict, cle: str, defauts: Dict):
        v = source.get(cle)
        return defauts[cle] if v is None else v

    def lire(cle: str):
        return val(poutre, cle, D.DEFAUTS)

    nom = lire("nom")
    if not isinstance(nom, str):
        raise ValueError(f"Le nom : un texte est attendu (reçu : {nom!r})")
    if len(nom) > D.NOM_MAX:
        raise ValueError(f"Le nom est trop long : {len(nom)} caractères (au plus {D.NOM_MAX})")
    classe = lire("classe")
    if not isinstance(classe, str) or classe not in D.CLASSES:
        raise ValueError(f"La classe de résistance : {classe!r} n'est pas proposée ({', '.join(D.CLASSES)})")
    typ = D.CLASSES[classe]["type"]
    out = {"nom": nom, "classe": classe, "standard": _case(lire("standard"), "standard")}

    # la section : standard (du type du bois) ou sur mesure ; b et h sont toujours la section effective
    section = poutre.get("section")
    if section is None:
        section = D.SECTION_DEFAUT[typ]
    if not isinstance(section, str):
        raise ValueError(f"La section standard : un texte « b x h » est attendu (reçu : {section!r})")
    b_std, h_std = section_standard(typ, section)
    out["section"] = section
    if out["standard"]:
        for cle, std in (("b", b_std), ("h", h_std)):
            x = poutre.get(cle)
            if x is not None and (isinstance(x, bool) or not isinstance(x, (int, float)) or float(x) != std):
                raise ValueError(f"{_maj(LIBELLES[cle])} ({_dire(x)}) contredit la section standard {designation(section)} : "
                                 f"choisir « sur mesure » pour la saisir")
            out[cle] = std
    else:
        out["b"] = _nombre(b_std if poutre.get("b") is None else poutre["b"], "b")
        out["h"] = _nombre(h_std if poutre.get("h") is None else poutre["h"], "h")
    out["L"] = _nombre(lire("L"), "L")

    # les charges
    ch, dch = _groupe(poutre, "charges"), D.DEFAUTS["charges"]
    c = {"mode": _parmi(val(ch, "mode", dch), "charges/mode", D.MODES_CHARGE)}
    for cle in ("G", "Q", "g", "q", "entraxe"):
        c[cle] = _nombre(val(ch, cle, dch), f"charges/{cle}")
    c["poids_propre"] = _case(val(ch, "poids_propre", dch), "charges/poids_propre")
    c["nature"] = _parmi(val(ch, "nature", dch), "charges/nature", D.CLES_NATURE)
    if c["nature"] == D.NATURE_LIBRE:
        c["duree"] = _parmi(val(ch, "duree", dch), "charges/duree", D.DUREES)
        c["psi2"] = _nombre(val(ch, "psi2", dch), "charges/psi2")
    else:
        duree, psi2 = D.DUREE_NATURE[c["nature"]], D.PSI2_NATURE[c["nature"]]
        lib = D.LIBELLES_NATURE[c["nature"]]
        if ch.get("duree") is not None and ch["duree"] != duree:
            dite = D.LIBELLES_DUREE.get(ch["duree"], _dire(ch["duree"])) if isinstance(ch["duree"], str) else _dire(ch["duree"])
            raise ValueError(f"La durée de la charge variable (« {dite} ») contredit sa nature « {lib} » "
                             f"({D.LIBELLES_DUREE[duree]}) : choisir la nature « autre » pour la saisir")
        x = ch.get("psi2")
        if x is not None and (isinstance(x, bool) or not isinstance(x, (int, float)) or float(x) != psi2):
            raise ValueError(f"ψ₂ ({_dire(x)}) contredit la nature « {lib} » de la charge variable (ψ₂ = {_txt(psi2)}) : "
                             f"choisir la nature « autre » pour le saisir")
        c["duree"], c["psi2"] = duree, psi2
    out["charges"] = c

    # l'appui, le maintien latéral
    ap, dap = _groupe(poutre, "appui"), D.DEFAUTS["appui"]
    out["appui"] = {"verifier": _case(val(ap, "verifier", dap), "appui/verifier"),
                    "a": _nombre(val(ap, "a", dap), "appui/a"), "debord": _nombre(val(ap, "debord", dap), "appui/debord")}
    ma, dma = _groupe(poutre, "maintien"), D.DEFAUTS["maintien"]
    out["maintien"] = {"type": _parmi(val(ma, "type", dma), "maintien/type", D.MAINTIENS),
                       "entraxe": _nombre(val(ma, "entraxe", dma), "maintien/entraxe")}

    # les hypothèses
    out["service"] = _parmi(lire("service"), "service", D.CLASSES_SERVICE)
    out["gamma_G"] = _nombre(lire("gamma_G"), "gamma_G")
    out["gamma_Q"] = _nombre(lire("gamma_Q"), "gamma_Q")
    out["systeme"] = _case(lire("systeme"), "systeme")
    out["lim_inst"] = _parmi(lire("lim_inst"), "lim_inst", D.LIMITES_INST, lambda n: f"L/{n}")
    out["lim_fin"] = _parmi(lire("lim_fin"), "lim_fin", D.LIMITES_FIN, lambda n: f"L/{n}")
    out["contre_fleche"] = _nombre(lire("contre_fleche"), "contre_fleche")

    # les contrôles croisés
    L_mm = out["L"] * 1000.0
    if out["appui"]["a"] >= L_mm:
        raise ValueError(f"La longueur d'appui a ({_txt(out['appui']['a'])} mm) doit rester plus courte que la portée L "
                         f"({_txt(L_mm)} mm entre axes des appuis)")
    if out["maintien"]["type"] == "entretoises" and out["maintien"]["entraxe"] > out["L"]:
        raise ValueError(f"L'entraxe des entretoises ({_txt(out['maintien']['entraxe'])} m) ne peut pas dépasser la portée L "
                         f"({_txt(out['L'])} m)")
    return out


def poutre_avec_section(poutre: Dict, nom: Optional[str]) -> Dict:
    """La même poutre avec la section standard ``nom`` (b et h suivent) ; ``nom`` None : la poutre telle quelle."""
    if nom is None:
        return poutre
    p = copy.deepcopy(poutre)
    p.update(standard=True, section=nom, b=None, h=None)
    return p
