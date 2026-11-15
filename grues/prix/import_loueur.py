# -*- coding: utf-8 -*-
"""APPLIQUER le fichier relu d'un loueur (cahier § 7.2), ligne par ligne selon les cases cochées à l'écran de contrôle, et
« Annuler le dernier import ».

- un prix appliqué prend l'ANNÉE et la PÉRIODE de validité du fichier, et la source « Fichier {loueur} {année} reçu le … » ;
- « les prix non modifiés restent valables » = Oui : les prix CONFIRMÉS prennent aussi l'année du fichier (sinon ils
  gardent la leur) ;
- une valeur CORRIGÉE par nous n'est pas écrasée : la valeur du loueur est mise à jour dessous, la correction est marquée
  « à revoir » ;
- une case VIDÉE ne s'applique que si on la coche ; une valeur NON LUE s'applique si on la saisit à la main ;
- une nouvelle grue prend le numéro suivant (les identifiants ne se renumérotent jamais) ; une grue « plus proposée » est
  retirée des listes (sans être effacée) ; une autre prestation devient un POSTE si on le demande ;
- les réponses aux questions et les remarques vont au dossier du loueur (« marquer comme traité ») ;
- tout va dans l'historique ; l'état d'avant de chaque objet touché est gardé : « Annuler le dernier import » le remet.

Une ligne de la lecture a une CLÉ stable (``cle_item``) : les cases cochées s'y rattachent, même si la lecture est refaite."""
import copy
import datetime as dt
import re
from typing import Dict, List, Optional, Tuple

from . import base as B
from . import edition as ED
from . import valeurs as V
from .fichier_loueur import LOUEUR_DU_CATALOGUE, TYPE_COURT

SORTES = ("changements", "cases_videes", "valeurs_non_lues", "nouvelles_grues", "grues_retirees", "autres_prestations")
PAR_DEFAUT = {"changements": True, "cases_videes": False, "valeurs_non_lues": False, "nouvelles_grues": True,
              "grues_retirees": True, "autres_prestations": False}


def cle_item(sorte: str, it: Dict) -> str:
    """La clé stable d'une ligne de la lecture (pour les cases cochées et les saisies de l'écran)."""
    if sorte in ("changements", "cases_videes", "valeurs_non_lues"):
        return "|".join(str(x) for x in (sorte[:2], it.get("type"), it.get("id"), it.get("cas") or "", it.get("cle")))
    if sorte == "nouvelles_grues":
        return f"ng|{it.get('ligne')}"
    if sorte == "grues_retirees":
        return f"gr|{it.get('id')}"
    if sorte == "autres_prestations":
        return f"ap|{it.get('id')}"
    return f"{sorte}|{it.get('id')}"


def coche(choix: Dict, sorte: str, it: Dict) -> bool:
    v = (choix or {}).get(cle_item(sorte, it))
    return PAR_DEFAUT[sorte] if v is None else bool(v)


def a_surligner(ch: Dict, seuil: float) -> bool:
    """Un écart au-delà du seuil (15 % par défaut), ou une BAISSE : la ligne est surlignée."""
    av, ap = ch.get("avant"), ch.get("apres")
    if V.est_nombre(av) and V.est_nombre(ap):
        return ap < av or (av != 0 and abs(100.0 * (ap - av) / av) > seuil)
    return False


def source_du_fichier(lecture: Dict, recu_le: dt.date) -> str:
    return f"Fichier {lecture['loueur']} {lecture['annee']} reçu le {recu_le.strftime('%d/%m/%Y')}"


def deja_importe(base: Dict, lecture: Dict) -> Optional[Dict]:
    return next((i for i in base.get("imports") or [] if i.get("empreinte") == lecture.get("empreinte") and not i.get("annule")), None)


def _validite(lecture: Dict) -> Optional[Dict]:
    i = lecture.get("infos") or {}
    du, au = i.get("valable_du"), i.get("valable_au")
    return {"du": du, "au": au} if (du or au) else None


def _gid_suivant(base: Dict) -> str:
    n = 0
    for g in B.grues(base):
        m = re.match(r"^G(\d+)$", str(g.get("id") or ""))
        if m:
            n = max(n, int(m.group(1)))
    return f"G{n + 1:03d}"


def _code_suivant(base: Dict, loueur_id) -> str:
    racine = f"AUTRE_{re.sub(r'[^A-Za-z0-9]+', '', str(loueur_id or 'X')).upper()}_"
    pris = {str(p.get("code")) for p in B.postes(base)}
    n = 1
    while f"{racine}{n:02d}" in pris:
        n += 1
    return f"{racine}{n:02d}"


class _Instantane:
    """L'état d'avant des objets touchés par l'import : « Annuler le dernier import » le remet."""

    def __init__(self, base: Dict, nom: str):
        self.base, self.nom = base, nom
        self.grues: Dict[str, Dict] = {}
        self.postes: Dict[str, Dict] = {}
        self.ppl = copy.deepcopy(base.get("postes_par_logique") or {})
        self.loueur = copy.deepcopy(B.loueur(base, nom))
        self.dossier = copy.deepcopy((base.get("dossiers") or {}).get(nom))
        self.neuves: List[str] = []
        self.nouveaux_postes: List[str] = []

    def grue(self, g: Dict):
        self.grues.setdefault(g["id"], copy.deepcopy(g))

    def poste(self, p: Dict):
        self.postes.setdefault(p["code"], copy.deepcopy(p))

    def vers_dict(self) -> Dict:
        return {"grues": self.grues, "postes": self.postes, "postes_par_logique": self.ppl, "loueur": self.loueur,
                "dossier": self.dossier, "neuves": self.neuves, "nouveaux_postes": self.nouveaux_postes}


def _ecrire_grue(base: Dict, g: Dict, cle: str, valeur, annee: int, source: str, validite: Optional[Dict], a_revoir: List):
    if cle in (g.get("corrections") or {}):
        g["corrections"][cle]["a_revoir"] = True
        a_revoir.append(("grue", g["id"], cle))
    g[cle] = valeur
    g.setdefault("annee_par_champ", {})[cle] = annee
    g.setdefault("source_par_champ", {})[cle] = source
    if validite:
        g.setdefault("validite_par_champ", {})[cle] = dict(validite)
    else:
        (g.get("validite_par_champ") or {}).pop(cle, None)


def _ecrire_poste(p: Dict, cas: str, valeur, annee: int, source: str, validite: Optional[Dict], a_revoir: List):
    if cas in (p.get("corrections") or {}):
        p["corrections"][cas]["a_revoir"] = True
        a_revoir.append(("poste", p["code"], cas))
    p.setdefault("prix_par_cas", {})[cas] = valeur
    p.setdefault("annee_par_cas", {})[cas] = annee
    p.setdefault("source_par_cas", {})[cas] = source
    if validite:
        p.setdefault("validite_par_cas", {})[cas] = dict(validite)
    else:
        (p.get("validite_par_cas") or {}).pop(cas, None)


def _dater(o: Dict, sorte: str, cle: str, annee: int, source: str, validite: Optional[Dict]):
    """Un prix CONFIRMÉ (inchangé) prend l'année et la période du fichier."""
    if sorte == "grue":
        o.setdefault("annee_par_champ", {})[cle] = annee
        o.setdefault("source_par_champ", {})[cle] = source
        if validite:
            o.setdefault("validite_par_champ", {})[cle] = dict(validite)
    else:
        o.setdefault("annee_par_cas", {})[cle] = annee
        o.setdefault("source_par_cas", {})[cle] = source
        if validite:
            o.setdefault("validite_par_cas", {})[cle] = dict(validite)


def appliquer(base: Dict, lecture: Dict, choix: Optional[Dict] = None, saisies: Optional[Dict] = None,
              recu_le: Optional[dt.date] = None, initiales: str = "", aujourdhui: Optional[dt.date] = None) -> Dict:
    """Applique la lecture (``lecture_loueur.lire``) selon ``choix`` ({cle_item: bool}) et ``saisies`` ({cle_item: texte}
    pour les valeurs non lues). → {appliques, confirmes_dates, neuves, retirees, postes, reponses, a_revoir, message}."""
    auj = aujourdhui or dt.date.today()
    recu = recu_le or auj
    nom, annee = lecture["loueur"], int(lecture["annee"])
    if not B.loueur(base, nom):
        raise ValueError("loueur inconnu")
    d_ = deja_importe(base, lecture)
    if d_:
        raise ValueError(f"Ce fichier a déjà été importé le {d_.get('date')}")
    source = source_du_fichier(lecture, recu)
    validite = _validite(lecture)
    origine = f"fichier {nom} {annee}"
    ini = initiales
    inst = _Instantane(base, nom)
    a_revoir: List[Tuple] = []
    n = {"appliques": 0, "confirmes_dates": 0, "neuves": [], "retirees": 0, "postes": [], "reponses": 0}

    def changer(it: Dict, apres):
        if it.get("type") == "grue":
            g = B.grue(base, it["id"])
            if not g:
                return
            inst.grue(g)
            avant = g.get(it["cle"])
            _ecrire_grue(base, g, it["cle"], apres, annee, source, validite, a_revoir)
            if apres is not None:
                g["a_verifier"] = [x for x in g.get("a_verifier") or [] if x != it["cle"]]
            ED.historiser(base, f"{g.get('modele')} ({g.get('cas')})", it["cle"], avant, apres, origine, "", ini, auj)
        else:
            p = B.poste(base, it["id"][2:])
            if not p:
                return
            inst.poste(p)
            if it["cle"] == "unite":
                avant = p.get("unite")
                p["unite"] = apres
                ED.historiser(base, f"{p.get('poste')}", "unite", avant, apres, origine, "", ini, auj)
            else:
                cas = it.get("cas")
                avant = (p.get("prix_par_cas") or {}).get(cas)
                _ecrire_poste(p, cas, apres, annee, source, validite, a_revoir)
                if apres is not None:
                    p["a_verifier"] = [x for x in p.get("a_verifier") or [] if x != cas]
                ED.historiser(base, f"{p.get('poste')} ({cas})", cas, avant, apres, origine, "", ini, auj)
        n["appliques"] += 1

    for it in lecture.get("changements") or []:
        if coche(choix, "changements", it):
            changer(it, it.get("apres"))
    for it in lecture.get("cases_videes") or []:
        if coche(choix, "cases_videes", it):
            changer(it, None)
    for it in lecture.get("valeurs_non_lues") or []:
        t = (saisies or {}).get(cle_item("valeurs_non_lues", it))
        if t is None or not str(t).strip():
            continue
        v = ED.lire_valeur(t)
        if it.get("type") == "grue":
            changer(it, v)
        else:
            p = B.poste(base, it["id"][2:])
            cas_l = [it["cas"]] if it.get("cas") else list((p or {}).get("prix_par_cas") or {})
            for c_ in cas_l:
                changer({**it, "cle": "prix", "cas": c_}, v)
    # les prix confirmés prennent l'année du fichier si le loueur dit qu'ils restent valables
    if (lecture.get("infos") or {}).get("inchanges_valables") is True:
        for it in lecture.get("prix_confirmes") or []:
            if it.get("type") == "grue":
                g = B.grue(base, it["id"])
                if g:
                    inst.grue(g)
                    _dater(g, "grue", it["cle"], annee, source, validite)
                    n["confirmes_dates"] += 1
            else:
                p = B.poste(base, it["id"][2:])
                if p and it.get("cas"):
                    inst.poste(p)
                    _dater(p, "poste", it["cas"], annee, source, validite)
                    n["confirmes_dates"] += 1
        if n["confirmes_dates"]:
            ED.historiser(base, f"dossier {nom}", "prix confirmés", None, n["confirmes_dates"], origine,
                          f"année {annee} (prix non modifiés valables)", ini, auj)
    for it in lecture.get("nouvelles_grues") or []:
        if not coche(choix, "nouvelles_grues", it):
            continue
        vals = dict(it.get("valeurs") or {})
        if not str(vals.get("modele") or "").strip():
            continue
        gid = _gid_suivant(base)
        g = {"id": gid, "cas": it["cas"], **{k: v for k, v in vals.items() if k != "retiree"},
             "annee_tarif": annee, "source_tarif": source}
        if validite:
            g["validite"] = dict(validite)
        base.setdefault("grues", []).append(g)
        inst.neuves.append(gid)
        n["neuves"].append(gid)
        ED.historiser(base, f"{g.get('modele')} ({it['cas']})", "nouvelle grue", None, gid, origine, "", ini, auj)
    for it in lecture.get("grues_retirees") or []:
        if not coche(choix, "grues_retirees", it):
            continue
        g = B.grue(base, it["id"])
        if g and not B.retiree(g):
            inst.grue(g)
            g["retiree"] = "Oui"
            n["retirees"] += 1
            ED.historiser(base, f"{g.get('modele')} ({g.get('cas')})", "retiree", None, "Oui", origine, "plus proposée par le loueur", ini, auj)
    lo = B.loueur(base, nom)
    for it in lecture.get("autres_prestations") or []:
        if not coche(choix, "autres_prestations", it) or not it.get("libelle"):
            continue
        cats = [c for c in lo.get("catalogues") or [] if B.catalogue(base, c)]
        t = (it.get("type") or "").strip()
        cibles = [c for c in cats if TYPE_COURT.get((B.catalogue(base, c) or {}).get("type")) == t] if t and t != "Toutes" else []
        cibles = cibles or cats
        code = _code_suivant(base, lo.get("id"))
        p = {"code": code, "poste": it["libelle"], "unite": it.get("unite") or "FF",
             "prix_par_cas": {c: it.get("prix") for c in cibles}, "annee_par_cas": {c: annee for c in cibles},
             "source_par_cas": {c: source for c in cibles}, "source": source, "fourni_par": LOUEUR_DU_CATALOGUE,
             "auto_par_defaut": "Non"}
        if it.get("remarque"):
            p["remarque_loueur"] = it["remarque"]
        base.setdefault("postes", []).append(p)
        ppl = base.setdefault("postes_par_logique", {})
        for c in cibles:
            lg = B.logique(base, c)
            if lg and code not in ppl.setdefault(lg, []):
                ppl[lg].append(code)
        inst.nouveaux_postes.append(code)
        n["postes"].append(code)
        ED.historiser(base, f"{it['libelle']}", "nouveau poste", None, code, origine, "autre prestation du loueur", ini, auj)
    # réponses et remarques : au dossier du loueur
    dos = base.setdefault("dossiers", {}).setdefault(nom, {})
    qs = {q.get("id"): q for q in lo.get("questions") or [] if isinstance(q, dict)}
    for it in lecture.get("reponses") or []:
        q = qs.get(it.get("id"))
        if q is not None:
            q["reponse"] = it.get("reponse")
            q["traitee"] = False
        else:
            dos.setdefault("messages", []).append({"date": recu.isoformat(), "id": it.get("id"), "texte": it.get("reponse"), "traitee": False})
        n["reponses"] += 1
    for it in lecture.get("remarques") or []:
        objet = it.get("modele") or it.get("id")
        dos.setdefault("messages", []).append({"date": recu.isoformat(), "id": it.get("id"), "objet": objet, "texte": it.get("texte"),
                                               "traitee": False})
    if dos.get("demande"):
        dos["demande"]["reponse_importee"] = True
    resume = {"fichier": lecture.get("fichier"), "recu_le": recu.isoformat(), "annee": annee, "changements": n["appliques"]}
    dos["dernier_fichier"] = resume
    dos.setdefault("echanges", []).append({"sorte": "reçu", "date": recu.isoformat(), "annee": annee, "fichier": lecture.get("fichier"),
                                           "changements": n["appliques"]})
    imports = base.setdefault("imports", [])
    iid = f"I{len(imports) + 1:03d}"
    imports.append({"id": iid, "date": auj.isoformat(), "loueur": nom, "annee": annee, "fichier": lecture.get("fichier"),
                    "empreinte": lecture.get("empreinte"), "source": source, "appliques": n["appliques"],
                    "confirmes_dates": n["confirmes_dates"], "neuves": list(n["neuves"]), "retirees": n["retirees"],
                    "postes": list(n["postes"]), "initiales": (ini or "")[:6]})
    base["import_annulable"] = {"id": iid, "loueur": nom, "avant": inst.vers_dict()}
    n["a_revoir"] = a_revoir
    n["id"] = iid
    parts = [f"{n['appliques']} prix appliqué{'s' if n['appliques'] > 1 else ''}"]
    if n["confirmes_dates"]:
        parts.append(f"{n['confirmes_dates']} confirmé{'s' if n['confirmes_dates'] > 1 else ''} pour {annee}")
    if n["neuves"]:
        parts.append(f"{len(n['neuves'])} grue{'s' if len(n['neuves']) > 1 else ''} ajoutée{'s' if len(n['neuves']) > 1 else ''}")
    if n["retirees"]:
        parts.append(f"{n['retirees']} retirée{'s' if n['retirees'] > 1 else ''}")
    if a_revoir:
        parts.append(f"{len(a_revoir)} corrigé{'s' if len(a_revoir) > 1 else ''} par nous à revoir")
    n["message"] = f"Import {nom} {annee} : " + ", ".join(parts)
    return n


def annuler_dernier(base: Dict, initiales: str = "", aujourdhui: Optional[dt.date] = None) -> str:
    """Remet chaque objet touché par le dernier import dans son état d'avant (les grues et postes ajoutés sont retirés)."""
    a = base.get("import_annulable")
    if not a:
        raise ValueError("aucun import à annuler")
    av = a["avant"]
    gs = base.get("grues") or []
    base["grues"] = [copy.deepcopy(av["grues"][g.get("id")]) if g.get("id") in av["grues"] else g
                     for g in gs if g.get("id") not in av["neuves"]]
    ps = base.get("postes") or []
    base["postes"] = [copy.deepcopy(av["postes"][p.get("code")]) if p.get("code") in av["postes"] else p
                      for p in ps if p.get("code") not in av["nouveaux_postes"]]
    base["postes_par_logique"] = copy.deepcopy(av["postes_par_logique"])
    nom = a["loueur"]
    if av.get("loueur") is not None:
        base["loueurs"] = [copy.deepcopy(av["loueur"]) if l.get("nom") == nom else l for l in base.get("loueurs") or []]
    dos = base.setdefault("dossiers", {})
    if av.get("dossier") is None:
        dos.pop(nom, None)
    else:
        dos[nom] = copy.deepcopy(av["dossier"])
    for i in base.get("imports") or []:
        if i.get("id") == a["id"]:
            i["annule"] = (aujourdhui or dt.date.today()).isoformat()
    ED.historiser(base, f"dossier {nom}", "import annulé", a["id"], None, "annulation", "", initiales, aujourdhui)
    base.pop("import_annulable", None)
    return f"Import {a['id']} annulé : la base est revenue à l'état d'avant"


def marquer_demande(base: Dict, nom: str, annee: int, demande_par: str = "", retour_avant: Optional[str] = None,
                    initiales: str = "", aujourdhui: Optional[dt.date] = None) -> str:
    """Le fichier est parti : le dossier passe en « Demande envoyée » jusqu'à l'import de la réponse."""
    if not B.loueur(base, nom):
        raise ValueError("loueur inconnu")
    auj = aujourdhui or dt.date.today()
    dos = base.setdefault("dossiers", {}).setdefault(nom, {})
    dos["demande"] = {"annee": int(annee), "envoyee_le": auj.isoformat(), "demande_par": (demande_par or "")[:60],
                      "retour_avant": retour_avant, "reponse_importee": False}
    dos.setdefault("echanges", []).append({"sorte": "envoyé", "date": auj.isoformat(), "annee": int(annee)})
    ED.historiser(base, f"dossier {nom}", "demande", None, f"fichier {annee} envoyé", "échange", "", initiales, auj)
    return f"{nom} : demande {annee} envoyée"


def annuler_demande(base: Dict, nom: str, initiales: str = "", aujourdhui: Optional[dt.date] = None) -> str:
    dos = (base.get("dossiers") or {}).get(nom) or {}
    if not dos.get("demande"):
        return "Aucune demande en cours"
    dos.pop("demande", None)
    ED.historiser(base, f"dossier {nom}", "demande", None, "annulée", "échange", "", initiales, aujourdhui)
    return f"{nom} : demande retirée"
