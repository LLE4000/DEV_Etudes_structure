# -*- coding: utf-8 -*-
"""Le module « Prix des grues » du site (catégorie « Autres ») : le prix d'une grue (tour, montage rapide, mobile) pour un
chantier, poste par poste, avec la source et l'écusson de chaque prix ; la base des prix des loueurs, tenue à jour ; le
prix réel des fondations de grue de nos chantiers.

Trois pages, chacune en trois colonnes (entrées · tableaux · résultats), comme les autres modules de l'établi :
« Calcul » (``prix_grues_calcul``), « Fondations » (``prix_grues_fondations``), « Base des prix » (``prix_grues_base``).

UNE NOTE = UN PROJET : ses grues (GT1, GT2…), chacune avec son calcul. LA BASE DES PRIX n'est pas dans la note : c'est le
CARNET du module (api_web/routeur.py) — l'écran la garde dans ce navigateur (js/carnet.js) et la joint à chaque requête ;
elle s'importe une fois depuis le fichier JSON de départ (confidentiel : jamais dans le dépôt, jamais dans le site). Aucun
prix n'est dans le code. Le jour de la mise en ligne d'un stockage partagé, le carnet deviendra celui du bureau.

Rien n'est calculé ici : le moteur est ``grues/prix`` (le calcul, les fondations, les écussons, les dossiers, les
modifications de la base). Ce fichier tient l'état de la note, écrit les saisies et assemble l'écran."""
import copy
import datetime as dt
import json
from typing import Dict, List, Optional, Sequence

from grues.prix import base as B
from grues.prix import calcul as K
from grues.prix import dossiers as D
from grues.prix import fondations as F
from grues.prix import periode as P

from . import commun as C
from . import etabli_compact as E
from . import prix_grues_calcul as PG
from . import prix_grues_commun as PC

MODULE = "prix_grues"
PAGE = "prix-grues"
CARNET = True                    # la base des prix voyage avec la requête (voir l'en-tête du fichier)
TITRE = "Prix des grues"
GENRE = E.Genre("Grues", "la grue", "cette grue", "une grue")
PAGES = ("calcul", "fondations", "base")
GRUES_MAX = 30
LIBRES_MAX = 30
REPERE_MAX = 20
TEXTE_MAX = 200
BASE_MAX = 4 * 1024 * 1024       # octets du fichier de la base (le fichier de départ fait ~120 ko)
TYPE_JSON = "application/json"
# le jour des écussons : aujourd'hui (un test le fixe : le 14/11/2026 du cahier)
AUJOURDHUI: Optional[dt.date] = None


def aujourdhui() -> dt.date:
    return AUJOURDHUI or dt.date.today()


# ------------------------------------------------------------------ l'état
def _vue_base() -> Dict:
    return {"loueur": None, "tableau": None, "filtre": "", "recherche": "", "case": None, "tout": False}


def _vue_fond() -> Dict:
    return {"famille": "tour", "demolition": False, "toutes": False, "case": None}


def nouvel_etat() -> Dict:
    return {"version": 1, "page": "calcul", "grues": [K.nouvelle_grue("GT1")], "actif": 0, "vb": _vue_base(), "vf": _vue_fond(),
            "initiales": ""}


def _lire_grue(x: Dict, i: int) -> Dict:
    g = K.nouvelle_grue(f"GT{i + 1}")
    if not isinstance(x, dict):
        return g
    for k, v in x.items():
        if k in g:
            if isinstance(g[k], dict) and isinstance(v, dict):
                g[k] = {**g[k], **copy.deepcopy(v)}
            elif isinstance(g[k], list) and isinstance(v, list):
                g[k] = copy.deepcopy(v)[:LIBRES_MAX]
            elif not isinstance(g[k], (dict, list)):
                g[k] = copy.deepcopy(v)
    g["repere"] = " ".join(str(g.get("repere") or f"GT{i + 1}").split())[:REPERE_MAX] or f"GT{i + 1}"
    return g


def normaliser(etat: Dict) -> Dict:
    """L'état reçu de l'écran (ou d'un fichier .stab) relu : chaque grue complétée de ses valeurs par défaut, la page et la
    grue ouverte valides. La base (le carnet) n'est jamais dans la note."""
    e = etat if isinstance(etat, dict) else {}
    gs = [_lire_grue(x, i) for i, x in enumerate((e.get("grues") or [])[:GRUES_MAX])] or [K.nouvelle_grue("GT1")]
    try:
        actif = int(e.get("actif") or 0)
    except (TypeError, ValueError):
        actif = 0
    vb = {**_vue_base(), **{k: v for k, v in (e.get("vb") or {}).items() if k in _vue_base()}}
    vf = {**_vue_fond(), **{k: v for k, v in (e.get("vf") or {}).items() if k in _vue_fond()}}
    return {"version": 1, "page": e.get("page") if e.get("page") in PAGES else "calcul", "grues": gs,
            "actif": min(max(0, actif), len(gs) - 1), "vb": vb, "vf": vf, "initiales": str(e.get("initiales") or "")[:6]}


def lire_carnet(carnet) -> List[Dict]:
    """Le carnet reçu de l'écran : [la base des prix], si elle se relit ; sinon vide."""
    if isinstance(carnet, list) and carnet and isinstance(carnet[0], dict):
        try:
            B.verifier(carnet[0])
            return [carnet[0]]
        except B.BaseIllisible:
            return []
    return []


def base_de(etat: Dict) -> Optional[Dict]:
    c = etat.get("carnet")
    return c[0] if isinstance(c, list) and c and isinstance(c[0], dict) and lire_carnet(c) else None


def _g(etat: Dict) -> Dict:
    return etat["grues"][etat["actif"]]


def _repere_libre(gs: List[Dict]) -> str:
    pris = {g.get("repere") for g in gs}
    k = len(gs) + 1
    while f"GT{k}" in pris:
        k += 1
    return f"GT{k}"


def gerer_elements(etat: Dict, action: str, index: Optional[int] = None) -> None:
    """Les grues du projet : en ouvrir une, en ajouter (du même catalogue, sur la même période), en copier une, en
    supprimer une (jamais la dernière)."""
    gs = etat["grues"]

    def rang(x):
        try:
            return min(max(0, int(x)), len(gs) - 1)
        except (TypeError, ValueError):
            return etat["actif"]
    if action == "activer":
        etat["actif"] = rang(index)
    elif action == "ajouter" and len(gs) < GRUES_MAX:
        a = _g(etat)
        n = K.nouvelle_grue(_repere_libre(gs), a.get("cas"))
        n["periode"] = copy.deepcopy(a.get("periode"))
        gs.append(n)
        etat["actif"] = len(gs) - 1
    elif action == "dupliquer" and len(gs) < GRUES_MAX:
        i = rang(index if index is not None else etat["actif"])
        c = copy.deepcopy(gs[i])
        c["repere"] = _repere_libre(gs)
        gs.insert(i + 1, c)
        etat["actif"] = i + 1
    elif action == "supprimer" and len(gs) > 1:
        i = rang(index if index is not None else etat["actif"])
        del gs[i]
        etat["actif"] = min(etat["actif"], len(gs) - 1)


# ------------------------------------------------------------------ les saisies
def _nombre(v, mini: float = 0.0, maxi: float = 1e9, vide_ok: bool = True) -> Optional[float]:
    x = C.lire_auto(v) if vide_ok else C.lire_nombre(v)
    if x is None:
        if vide_ok:
            return None
        raise ValueError("un nombre")
    if not mini <= x <= maxi:
        raise ValueError(f"entre {PC.nombre(mini)} et {PC.nombre(maxi)}")
    return x


def _bool(v) -> bool:
    b = C.lire_bool(v)
    if b is None:
        raise ValueError("oui ou non")
    return b


def _date_iso(v) -> Optional[str]:
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    d = P.lire_date(v)
    if not d:
        raise ValueError(f"« {str(v)[:20]} » n'est pas une date (AAAA-MM-JJ ou JJ/MM/AAAA)")
    return d.isoformat()


def _code(c: str) -> str:
    c = str(c)
    if not c or len(c) > 40 or not all(ch.isalnum() or ch == "_" for ch in c):
        raise ValueError("poste inconnu")
    return c


def _ecrire_grue(base: Optional[Dict], s: Dict, ch: List[str], valeur) -> Optional[str]:
    k = ch[0] if ch else ""
    if k == "cas":
        if valeur and (not base or not B.catalogue(base, valeur)):
            raise ValueError("catalogue inconnu")
        if valeur != s.get("cas"):
            s.update(cas=valeur or None, grue=None, choix={}, quantites={}, forces={})
        return None
    if k == "grue":
        if valeur and (not base or not B.grue(base, valeur) or B.grue(base, valeur).get("cas") != s.get("cas")):
            raise ValueError("cette grue n'est pas dans le catalogue")
        s["grue"] = valeur or None
        return None
    if k in ("hsc_m", "voie_m"):
        s[k] = _nombre(valeur, 0, 1000)
        return None
    if k == "butees":
        x = _nombre(valeur, 0, 100)
        s[k] = int(x) if x is not None else None
        return None
    if k == "heures_jour":
        x = _nombre(valeur, 0, 24)
        s[k] = 8 if x is None else x
        return None
    if k == "periode" and len(ch) == 2:
        p = s.setdefault("periode", {})
        if ch[1] == "mode":
            if valeur not in ("dates", "mois", "jours"):
                raise ValueError("période inconnue")
            p["mode"] = valeur
        elif ch[1] in ("debut", "fin"):
            p[ch[1]] = _date_iso(valeur)
            if p.get("debut") and p.get("fin") and p["fin"] <= p["debut"]:
                return "La date de fin est avant la date de début"
        elif ch[1] in ("mois", "jours"):
            p[ch[1]] = _nombre(valeur, 0, 3650)
        else:
            raise ValueError("champ inconnu")
        return None
    if k == "chassis":
        if valeur not in K.CHASSIS:
            raise ValueError("châssis inconnu")
        s["chassis"] = valeur
        return None
    if k == "conduite":
        if valeur not in K.CONDUITES:
            raise ValueError("conduite inconnue")
        s["conduite"] = valeur
        return None
    if k == "fondation" and len(ch) == 2:
        fd = s.setdefault("fondation", {})
        if ch[1] == "type":
            t = F.type_fondation(valeur)
            if not t:
                raise ValueError("type de fondation inconnu")
            fd["type"] = t
        elif ch[1] == "demolition":
            fd["demolition"] = _bool(valeur)
        else:
            raise ValueError("champ inconnu")
        return None
    if k == "electricite" and len(ch) == 2:
        el = s.setdefault("electricite", {})
        if ch[1] == "prix_kwh":
            el["prix_kwh"] = _nombre(valeur, 0, 10)
        elif ch[1] == "inf_32a":
            el["inf_32a"] = _bool(valeur)
        else:
            raise ValueError("champ inconnu")
        return None
    if k == "telescopage" and len(ch) == 2 and ch[1] in ("nombre", "jours"):
        x = _nombre(valeur, 0, 100)
        s.setdefault("telescopage", {})[ch[1]] = 0 if x is None else (int(x) if ch[1] == "nombre" else x)
        return None
    if k == "operateur":
        s["operateur"] = _bool(valeur)
        return None
    if k == "ma" and len(ch) >= 2:
        ma = s.setdefault("ma", {})
        if ch[1] in ("ballast_sam", "ballast_dim") and len(ch) == 2:
            ma[ch[1]] = _bool(valeur)
        elif ch[1] == "heures_personnel" and len(ch) == 3:
            hp = ma.setdefault("heures_personnel", {})
            x = _nombre(valeur, 0, 24)
            if x is None:
                hp.pop(_code(ch[2]), None)
            else:
                hp[_code(ch[2])] = x
        else:
            raise ValueError("champ inconnu")
        return None
    if k == "choix" and len(ch) == 2:
        v = str(valeur).lower()
        if v not in K.CHOIX:
            raise ValueError("Auto, Oui ou Non")
        if v == "auto":
            s.setdefault("choix", {}).pop(_code(ch[1]), None)
        else:
            s.setdefault("choix", {})[_code(ch[1])] = v
        return None
    if k == "quantites" and len(ch) == 2:
        x = _nombre(valeur, 0, 1e6)
        if x is None:
            s.setdefault("quantites", {}).pop(_code(ch[1]), None)
        else:
            s.setdefault("quantites", {})[_code(ch[1])] = x
        return None
    if k == "forces" and len(ch) == 3:
        fs = s.setdefault("forces", {})
        code = _code(ch[1])
        if ch[2] == "-":
            fs.pop(code, None)
            return "Prix de la base repris"
        if ch[2] == "prix":
            x = _nombre(valeur, 0, 1e8)
            if x is None:
                fs.pop(code, None)
                return None
            fs[code] = {"prix": x, "justification": (fs.get(code) or {}).get("justification", "")}
            return None   # (sa justification est demandée sous la ligne, en orange ; « Points d'attention » le redit)
        if ch[2] == "justification":
            if code in fs:
                fs[code]["justification"] = " ".join(str(valeur or "").split())[:TEXTE_MAX]
            return None
        raise ValueError("champ inconnu")
    if k == "libres":
        ls = s.setdefault("libres", [])
        if ch[1:] == ["+"]:
            if len(ls) >= LIBRES_MAX:
                raise ValueError(f"{LIBRES_MAX} lignes libres au plus")
            ls.append({"libelle": "", "unite": "FF", "quantite": 1, "prix": None, "mo": False})
            return None
        try:
            i = int(ch[1])
            l = ls[i]
        except (ValueError, IndexError):
            raise ValueError("ligne libre inconnue")
        champ = ch[2] if len(ch) > 2 else ""
        if champ == "-":
            del ls[i]
        elif champ == "libelle":
            l["libelle"] = " ".join(str(valeur or "").split())[:TEXTE_MAX]
        elif champ == "unite":
            if valeur not in ("FF", "€/J", "€/S", "€/M"):
                raise ValueError("unité inconnue")
            l["unite"] = valeur
        elif champ in ("quantite", "prix"):
            l[champ] = _nombre(valeur, 0, 1e8)
        elif champ == "mo":
            l["mo"] = _bool(valeur)
        else:
            raise ValueError("champ inconnu")
        return None
    raise ValueError("champ inconnu")


def _importer(etat: Dict, valeur) -> str:
    """La base de départ (ou une sauvegarde) lue depuis le fichier JSON choisi sur l'appareil."""
    if not isinstance(valeur, dict) or not isinstance(valeur.get("texte"), str):
        raise ValueError("fichier illisible")
    if len(valeur["texte"].encode("utf-8")) > BASE_MAX:
        raise ValueError("fichier trop gros pour une base des prix")
    try:
        b = json.loads(valeur["texte"])
    except ValueError:
        raise ValueError("ce fichier n'est pas un fichier JSON")
    B.verifier(b)
    etat["carnet"] = [b]
    nb_g, nb_p = len(B.grues(b)), len(B.postes(b))
    return f"Base des prix importée : {len(B.loueurs(b))} loueurs, {nb_g} grues, {nb_p} postes"


def modifier(etat: Dict, chemin: Sequence, valeur) -> Optional[str]:
    ch = [str(c) for c in chemin]
    base = base_de(etat)
    try:
        if ch == ["page"]:
            if valeur not in PAGES:
                raise ValueError("page inconnue")
            etat["page"] = valeur
            return None
        if ch == ["nom"]:
            r = " ".join(str(valeur or "").split())[:REPERE_MAX]
            if not r:
                raise ValueError("un repère ne peut pas être vide")
            _g(etat)["repere"] = r
            return None
        if ch == ["initiales"]:
            etat["initiales"] = " ".join(str(valeur or "").split())[:6]
            return None
        if ch == ["base", "importer"]:
            return _importer(etat, valeur)
        if ch and ch[0] == PG.CH:
            return _ecrire_grue(base, _g(etat), ch[1:], valeur)
        if ch and ch[0] == "aller" and len(ch) == 2 and ch[1] in PAGES:
            return _aller(etat, ch[1], valeur)
        if ch and ch[0] in ("vb", "vf", "b"):
            if base is None:
                raise ValueError("aucune base des prix")
            from . import prix_grues_base as PB
            from . import prix_grues_fondations as PF
            if ch[0] == "vb" or (ch[0] == "b" and ch[1:2] != ["fond"]):
                return PB.modifier(etat, base, ch, valeur, aujourdhui())
            return PF.modifier(etat, base, ch, valeur, aujourdhui())
    except B.BaseIllisible as e:
        raise ValueError(str(e))
    raise ValueError("champ inconnu")


def _aller(etat: Dict, page: str, valeur) -> Optional[str]:
    """Aller à une page en y ouvrant une case : « Corriger dans la base » (une case de prix), « Voir les chantiers » (une
    case de la grille des fondations)."""
    etat["page"] = page
    v = str(valeur or "")
    if page == "base" and v.count("|") == 2:
        sorte, ident, cle = v.split("|")
        etat["vb"]["case"] = [sorte, ident, cle]
        if sorte == "grue":
            etat["vb"]["tableau"] = None
        base = base_de(etat)
        if base:
            if sorte == "grue" and B.grue(base, ident):
                cas = B.grue(base, ident).get("cas")
                etat["vb"].update(loueur=(B.catalogue(base, cas) or {}).get("loueur"), tableau=cas)
            elif sorte == "poste":
                etat["vb"].update(loueur=(B.catalogue(base, cle) or {}).get("loueur"), tableau="postes")
    elif page == "fondations" and v.count("|") == 2:
        modele, t, dem = v.split("|")
        etat["vf"].update(case=[modele, t], demolition=dem == "avec", toutes=True)
    return None


# ------------------------------------------------------------------ l'écran
def _badges(base: Dict, auj: dt.date) -> Dict:
    att = sum(1 for r in F.retours(base) if r["statut"] == F.A_VERIFIER)
    pas = sum(1 for l in B.loueurs(base) if D.dossier(base, l.get("nom"), auj)["ecusson"] == "pas_a_jour")
    out = {}
    if att:
        out["fondations"] = (str(att), f"{att} retour(s) de chantier à valider")
    if pas:
        out["base"] = ("●", f"{pas} dossier(s) de loueur pas à jour")
    return out


def _sorties() -> str:
    return E.exporter([E.ligne_export('data-telecharger="base_json"', "hard-drive", "Base des prix (JSON)",
                                      "la sauvegarde complète de la base de cet appareil")])


def _noms(etat: Dict, base: Optional[Dict]) -> List:
    out = []
    for g in etat["grues"]:
        gr = B.grue(base, g.get("grue")) if base and g.get("grue") else None
        out.append((g.get("repere") or "GT", f"· {C.esc(PG._court(gr.get('modele')))}" if gr else "· à choisir", False))
    return out


def resultat(etat: Dict) -> Dict:
    base = base_de(etat)
    auj = aujourdhui()
    s = _g(etat)
    page = etat["page"]
    volets = ""
    resume = {"taux": None, "niveau": "i", "statut": "Base des prix à importer", "gouvernant": ""}
    if base is None:
        corps = PC.onglets(page) + C.section(1, "Base des prix", "à importer",
                                             C.ligne("Fichier", PC.import_base()) + C.aide("Le fichier JSON de la base, une seule fois."), True, "base")
        vues = [{"cle": "vide", "titre": "Prix des grues", "court": "Base", "echelle": False, "svg": PC.ecran_vide()}]
        res = ('<div class="pbody"><div class="pg-res"><div class="pg-total nul">— € <small>HTVA</small></div>'
               '<p class="pg-info">Importer la base des prix pour commencer.</p></div></div>')
    elif page == "base":
        from . import prix_grues_base as PB
        corps, vues, res, volets = PB.page(etat, base, auj)
        corps = PC.onglets(page, _badges(base, auj)) + corps
    elif page == "fondations":
        from . import prix_grues_fondations as PF
        corps, vues, res, volets = PF.page(etat, base, auj)
        corps = PC.onglets(page, _badges(base, auj)) + corps
    else:
        r = K.calculer(base, s, auj)
        corps = PC.onglets(page, _badges(base, auj)) + PG.rubriques(base, s, r, auj)
        vues = [PG.vue(r, s)]
        res = PG.colonne(r, s, _sorties())
        volets = PG.volets(r)
    if base is not None:
        tot = []
        for g in etat["grues"]:
            rg = K.calculer(base, g, auj) if g is not s or page != "calcul" else r
            if rg["pret"]:
                tot.append(rg)
        if tot:
            n_manq = sum(len(x["manquants"]) for x in tot)
            somme = sum(x["total_arrondi"] for x in tot)
            resume = {"taux": None, "niveau": "i", "statut": f"{PC.euros(somme)} HTVA" + (" · incomplet" if n_manq else ""),
                      "gouvernant": f"{len(tot)} grue{'s' if len(tot) > 1 else ''}"}
        else:
            resume = {"taux": None, "niveau": "i", "statut": "Grue à choisir", "gouvernant": ""}
    donnees = E.arbre(GENRE, _noms(etat, base), etat["actif"], corps, "grue", compact=True)
    return {"resultat": res, "volets": volets, "vues": vues, "phrase": "", "statut_html": "", "resume": resume,
            "donnees": donnees, "elements": ""}


# ------------------------------------------------------------------ les fichiers
def telecharger(etat: Dict, quoi: str, infos: Optional[Dict] = None, carnet=None) -> Optional[Dict]:
    c = lire_carnet(carnet)
    base = c[0] if c else None
    if quoi == "base_json":
        if base is None:
            raise ValueError("aucune base des prix sur cet appareil")
        return {"nom": f"base_prix_grues_{aujourdhui().isoformat()}.json", "type": TYPE_JSON,
                "texte": json.dumps(base, ensure_ascii=False, indent=1)}
    return None
