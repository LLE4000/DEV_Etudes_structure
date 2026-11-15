# -*- coding: utf-8 -*-
"""Le PRIX D'UNE GRUE, poste par poste (cahier § 6, § 6.1).

Une grue du projet (``nouvelle_grue``) : son catalogue (le « cas » : un type de grue chez un loueur, avec sa LOGIQUE — T
grue tour, R montage rapide, MS grue mobile grutier à part, MA grue mobile grutier compris), sa grue, sa période, les
heures du grutier par jour, ses options (châssis, conduite, fondation, électricité, télescopage, opérateur, services du
loueur), et, poste par poste, le CHOIX (Auto, Oui, Non), la quantité, le prix forcé (avec sa justification) et les lignes
libres.

``calculer`` rend chaque ligne (quantité, unité, prix et son origine, quantité facturée, total, écusson), les sous-totaux
par groupe, le matériel et la main-d'œuvre, le total arrondi à l'euro, les PRIX MANQUANTS (une ligne retenue sans prix :
jamais 0 €) et les POINTS D'ATTENTION.

Les codes des postes sont ceux des tests. Les prix propres à un modèle viennent de la grue (``LOC``, ``MONT``…), les prix
communs à un catalogue viennent des postes (``ANTICOL_LOC``, ``GRUTIER``…) ; ``FOND`` vient des fondations, ``ELEC`` de la
consommation et du prix du kWh."""
import datetime as dt
import math
from typing import Dict, List, Optional, Tuple

from . import base as B
from . import ecussons as E
from . import fondations as F
from . import valeurs as V
from .periode import periode as calculer_periode

# ------------------------------------------------------------------ constantes
CHASSIS = ("standard", "pied", "roulant", "hors_standard")
NOMS_CHASSIS = {"standard": "standard (fixe lesté)", "pied": "pied d'ancrage", "roulant": "roulant", "hors_standard": "hors standard"}
CHAMP_CHASSIS = {"standard": "location_jour_chassis_standard", "hors_standard": "location_jour_chassis_standard",
                 "pied": "location_jour_pied_ancrage", "roulant": "location_jour_chassis_roulant"}
CONDUITES = ("auto", "cabine", "telecommande", "telecommande_externe")
NOMS_CONDUITE = {"cabine": "cabine", "telecommande": "télécommande du loueur", "telecommande_externe": "télécommande louée à l'extérieur"}
CHOIX = ("auto", "oui", "non")

# la main-d'œuvre (§ 6 « Total ») : le grutier, ses heures sup., le télescopage, le déplacement du personnel et les
# suppléments horaires du loueur (MA)
MO = {"GRUTIER", "HS", "TELESC_GRUT", "TELESC_HS", "DEPL_GRUTIER"}
PREFIXE_MO = ("SUP_",)
# les postes de montage/démontage d'un accessoire suivent sa ligne de location (§ 6 « Montage et démontage des accessoires »)
SUIT = {"ANTICOL_MONT": "ANTICOL_LOC", "ANTICOL_DEM": "ANTICOL_LOC", "CAM_MONT": "CAM_LOC", "CAM_DEM": "CAM_LOC",
        "TELEC_MONT": "TEL", "TELEC_DEM": "TEL", "BALISAGE_MONT": "BALISAGE_LOC", "BALISAGE_DEM": "BALISAGE_LOC",
        "ECLAIRAGE_MONT": "ECLAIRAGE", "NACELLE_TR": "NACELLE", "PLAQUES_GRUE_TR": "PLAQUES_GRUE", "GITTERBOX_TR": "GITTERBOX"}
ROULANT = {"RAIL", "BUTEE", "RAIL_POSE", "RAIL_TRANSPORT", "RAIL_EVAC", "RAIL_DEPOSE"}
SANS_PRIX_ROULANT = {"RIJWERK", "ANTICOL_RIJWERK"}
OUI_PAR_DEFAUT = {"T": {"ANTICOL_LOC", "CAM_LOC", "TW_CH", "TW_PED_CH", "VINCOTTE", "PUB"}, "R": {"VINCOTTE", "PUB"},
                  "MS": set(), "MA": set()}
HORAIRES_MA = {"SUP_NUIT": "nuit en semaine", "SUP_SAM": "samedi", "SUP_DIM": "dimanche ou nuit le week-end", "SUP_FERIE": "jour férié"}
PERSONNEL_MA = ("RIGGER", "SUPERVISEUR")
POSTE_CARBURANT = "CARBURANT"
POSTE_FONDATION = "FONDATION"
PREFIXE_CHASSIS_HS = "CHASSIS_HS"
PAS_POSE_RAILS_M = 30.0      # pose et démontage des rails : 30 m par jour (liste du loueur ; démontage supposé pareil, § 14.2)
PAS_TRANSPORT_RAILS_M = 36.0  # transport des rails : 36 m par transport
QUANTITE_DEFAUT = {"TW_CH": 5}

GROUPES = (("location", "Location"), ("montage", "Montage et démontage"), ("conduite", "Conduite"),
           ("accessoires", "Accessoires"), ("energie", "Électricité"), ("fondation", "Fondation"),
           ("mo", "Main-d'œuvre"), ("loueur", "Loueur"), ("libres", "Lignes libres"))
GROUPE_DE = {"LOC": "location", "CHROUL": "location", "MONT": "montage", "DEMONT": "montage", "POIDS": "montage",
             "MONT_WE": "montage", "DEM_WE": "montage", "MONT_16": "montage", "TLS": "montage", "TEL": "conduite",
             "TELV": "conduite", "CAB": "conduite", "TELEC_MONT": "conduite", "TELEC_DEM": "conduite", "ELEC": "energie",
             "CARB": "energie", "FOND": "fondation", "GRUTIER": "mo", "HS": "mo", "TELESC_GRUT": "mo", "TELESC_HS": "mo"}


def est_mo(code: str) -> bool:
    return code in MO or code.startswith(PREFIXE_MO)


def nouvelle_grue(repere: str = "GT1", cas: Optional[str] = None, gid: Optional[str] = None) -> Dict:
    """Une grue du projet, vide : ce que l'écran saisit."""
    return {"repere": repere, "cas": cas, "grue": gid, "hsc_m": None,
            "periode": {"mode": "dates", "debut": None, "fin": None, "mois": None, "jours": None},
            "heures_jour": 8, "chassis": "standard", "voie_m": None, "butees": None, "conduite": "auto",
            "fondation": {"type": "Dalle sur sol", "demolition": True},
            "electricite": {"prix_kwh": None, "inf_32a": False}, "telescopage": {"nombre": 0, "jours": 0},
            "operateur": False, "ma": {"ballast_sam": False, "ballast_dim": False, "heures_personnel": {}},
            "choix": {}, "quantites": {}, "forces": {}, "libres": []}


def _num(v, defaut=None) -> Optional[float]:
    if V.est_nombre(v):
        return float(v)
    if isinstance(v, str):
        x = V.lire_nombre(v)
        return x if x is not None else defaut
    return defaut


def _fr(x, nd: int = 2) -> str:
    """Un nombre à la française, sans décimales inutiles (« 1 300 », « 0,48 », « 8,0 » pour des mois)."""
    if x is None:
        return "—"
    s = f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")
    if nd and "," in s:
        s = s.rstrip("0").rstrip(",")
    return s


# ------------------------------------------------------------------ le contexte d'un calcul
class _Ctx:
    def __init__(self, base: Dict, saisie: Dict, aujourdhui: dt.date):
        self.base, self.s, self.auj = base, saisie, aujourdhui
        self.cas = saisie.get("cas")
        cat = B.catalogue(base, self.cas) or {}
        self.lg = cat.get("code_calcul")
        self.nom_loueur = cat.get("loueur")
        self.g = B.grue(base, saisie.get("grue")) or {}
        self.reg = B.reglages(base)
        self.hn = _num(self.reg.get("heures_normales"), 8.0) or 8.0
        self.h = max(0.0, _num(saisie.get("heures_jour"), self.hn))
        self.per = calculer_periode(saisie.get("periode") or {}, self.reg, self.lg or "T", B.jours_par_mois(base, self.nom_loueur))
        self.chassis = saisie.get("chassis") if saisie.get("chassis") in CHASSIS else "standard"
        if self.lg != "T":
            self.chassis = "standard"
        self.tel = saisie.get("telescopage") or {}
        self.nb_tel = max(0.0, _num(self.tel.get("nombre"), 0.0) or 0.0) if self.lg == "T" else 0.0
        self.jours_tel = max(0.0, _num(self.tel.get("jours"), 0.0) or 0.0)
        self.operateur = bool(saisie.get("operateur")) and self.lg == "MS"
        el = saisie.get("electricite") or {}
        self.inf32 = bool(el.get("inf_32a")) and self.lg in ("T", "R")
        self.kwh_chantier = _num(el.get("prix_kwh"))
        self.voie = max(0.0, _num(saisie.get("voie_m"), 0.0) or 0.0)
        self.butees = max(0.0, _num(saisie.get("butees"), 0.0) or 0.0)
        self.ma = saisie.get("ma") or {}
        self.choix = {k: str(v).lower() for k, v in (saisie.get("choix") or {}).items() if str(v).lower() in CHOIX}
        self.quantites = saisie.get("quantites") or {}
        self.forces = saisie.get("forces") or {}
        self.conduite = self._conduite()

    def _conduite(self) -> str:
        c = self.s.get("conduite")
        if c in ("cabine", "telecommande") and self.lg in ("T", "R"):
            return c
        if c == "telecommande_externe" and self.lg == "T":
            return c
        if self.lg == "R":
            return "telecommande"
        if self.lg == "T":
            return "cabine" if V.nature(B.prix_grue(self.g, "cabine_jour")["valeur"]) == V.COMPRIS else "telecommande"
        return ""

    def q(self, code: str, defaut: float) -> float:
        v = _num(self.quantites.get(code))
        return max(0.0, v) if v is not None else defaut

    @property
    def jours(self) -> float:
        return self.per.get("jours") or 0.0

    @property
    def jg(self) -> float:
        return self.per.get("jours_grutier") or 0.0


# ------------------------------------------------------------------ les lignes
def _ligne(code: str, libelle: str, groupe: str, unite: str, origine: Optional[Dict], quantite: float,
           facturee: float, facturee_unite: str, quantite_editable: bool = False, quantite_unite: str = "",
           detail: str = "") -> Dict:
    return {"code": code, "libelle": libelle, "groupe": groupe, "unite": unite, "origine": origine, "quantite": quantite,
            "quantite_editable": quantite_editable, "quantite_unite": quantite_unite, "facturee": facturee,
            "facturee_unite": facturee_unite, "detail": detail, "mo": est_mo(code), "fondation": None}


def _selon_unite(unite: str, c: _Ctx) -> Tuple[float, str]:
    """La quantité facturée d'un prix selon son unité : €/J × jours, €/S × semaines, €/M × mois, forfait × 1."""
    u = (unite or "").upper().replace(" ", "")
    if u in ("€/J", "€/JOUR"):
        return c.jours, "j"
    if u in ("€/S", "€/SEM", "€/SEMAINE"):
        return c.per.get("semaines") or 0.0, "sem."
    if u in ("€/M", "€/MOIS"):
        return c.per.get("mois") or 0.0, "mois"
    return 1.0, ""


def _lignes_grue(c: _Ctx) -> List[Dict]:
    """Les lignes dont le prix est propre au modèle (la grue)."""
    g, lg, out = c.g, c.lg, []
    pg = lambda k: B.prix_grue(g, k)  # noqa: E731
    if lg in ("T", "R", "MS"):
        if lg == "T":
            cle = CHAMP_CHASSIS[c.chassis]
            lib = "Location" + ("" if c.chassis in ("standard", "hors_standard") else f" (châssis {NOMS_CHASSIS[c.chassis]})")
        elif lg == "MS" and c.operateur:
            cle, lib = "location_jour_avec_operateur", "Location avec opérateur"
        else:
            cle, lib = "location_jour_chassis_standard", "Location" + (" sans opérateur" if lg == "MS" else "")
        out.append(_ligne("LOC", lib, "location", "€/J", pg(cle), 1, c.jours, "j"))
        if lg == "T":
            out.append(_ligne("CHROUL", "Châssis roulant (le châssis)", "location", "€/J", pg("chassis_roulant_jour"), 1, c.jours, "j"))
        out.append(_ligne("MONT", "Transport + montage", "montage", "FF", pg("transport_montage_ff"), 1, 1, ""))
        out.append(_ligne("DEMONT", "Démontage + transport", "montage", "FF", pg("demontage_transport_ff"), 1, 1, ""))
        if lg == "R":
            out.append(_ligne("POIDS", "Transport des poids de test", "montage", "FF", pg("transport_poids_test_ff"), 1, 1, ""))
        if lg == "T":
            out.append(_ligne("TLS", "Intervention du loueur pour télescopage", "montage", "FF", pg("telescopage_intervention_ff"),
                              c.nb_tel, 1, "", quantite_unite="télescopage(s)"))
        if lg in ("T", "R"):
            u = g.get("telecommande_loueur_unite") or "€/J"
            f, fu = _selon_unite(u, c)
            out.append(_ligne("TEL", "Télécommande du loueur", "conduite", u, pg("telecommande_loueur_prix"), 1, f, fu))
            if lg == "T":
                out.append(_ligne("TELV", "Télécommande louée à l'extérieur", "conduite", "€/S", pg("telecommande_externe_prix"), 1,
                                  c.per.get("semaines") or 0.0, "sem."))
            out.append(_ligne("CAB", "Cabine", "conduite", "€/J", pg("cabine_jour"), 1, c.jours, "j"))
            out.append(_ligne("AI", "Anti-intrusion", "accessoires", "€/J", pg("anti_intrusion_jour"), 1, c.jours, "j"))
            out.append(_ligne_elec(c))
    elif lg == "MA":
        hf, minimum = heures_facturees(c)
        out.append(_ligne("LOC", "Location, grutier compris", "location", "€/h", pg("location_heure_grutier_compris"), hf, c.jours, "j",
                          quantite_unite="h/j", detail=f"{_fr(hf)} h/j" + (f" (minimum {_fr(minimum)} h)" if minimum and minimum > c.h else "")))
        out.append(_ligne("DEPL", "Déplacement de la grue", "loueur", "€/h", pg("deplacement_grue_heure"), c.q("DEPL", 0.0), 1, "",
                          quantite_editable=True, quantite_unite="h"))
        out.append(_ligne("BAL", "Transport du ballast", "loueur", "FF", pg("transport_ballast_ff"), 1, 1, ""))
    return out


def heures_facturees(c: _Ctx) -> Tuple[float, Optional[float]]:
    """Logique MA : les heures facturées par jour = max(h/j ; minimum) — le minimum de la grue (5 h jusqu'à 80 t, 8 h
    au-delà) pour un jour, les heures normales (8 h) par jour dès que la location dure plusieurs jours."""
    if c.jours > 1:
        m = c.hn
    else:
        m = _num(c.g.get("minimum_heures"))
    return max(c.h, m or 0.0), m


def _ligne_elec(c: _Ctx) -> Dict:
    """L'électricité (§ 6) : la consommation connue × le prix du kWh (celui du chantier, sinon le dernier des réglages) ×
    les mois ; sinon le forfait par jour × les jours facturés ; sinon un prix manquant."""
    conso = V.montant(B.prix_grue(c.g, "consommation_kwh_mois")["valeur"])
    mois = c.per.get("mois") or 0.0
    if conso:
        if c.kwh_chantier is not None:
            o = {"valeur": c.kwh_chantier, "annee": c.auj.year, "source": "prix du kWh du chantier", "validite": None,
                 "corrige": False, "a_verifier": False, "ref": ("chantier", "prix_kwh", None)}
        else:
            o = B.prix_kwh(c.base) or {"valeur": None, "annee": None, "source": "réglages", "ref": ("reglage", "prix_kwh", None)}
        kwh = V.montant(o.get("valeur"))
        an = f" ({o.get('annee')})" if o.get("annee") else ""
        return _ligne("ELEC", "Électricité", "energie", "€/kWh", o, conso, mois, "mois", quantite_unite="kWh/mois",
                      detail=f"{_fr(conso, 0)} kWh/mois × {_fr(kwh, 3) if kwh is not None else '—'} €/kWh{an} × {_fr(mois, 1)} mois")
    o = B.prix_grue(c.g, "electricite_jour")
    return _ligne("ELEC", "Électricité (forfait)", "energie", "€/J", o, 1, c.jours, "j", detail="forfait par jour × jours facturés")


def _ligne_fondation(c: _Ctx) -> Dict:
    fd = c.s.get("fondation") or {}
    t = F.type_fondation(fd.get("type")) or "Dalle sur sol"
    dem = bool(fd.get("demolition"))
    k = F.case(c.base, c.g.get("modele"), t, dem, c.auj)
    o = {"valeur": k["valeur"], "annee": None, "source": k["source"] or "aucun prix", "validite": None, "corrige": k["origine"] == "correction",
         "a_verifier": False, "ref": ("fondation", F.cle_case(c.g.get("modele") or "", t, dem), None)}
    ln = _ligne("FOND", f"Fondation ({t.lower()}, {'avec' if dem else 'sans'} démolition)", "fondation", "FF", o, 1, 1, "")
    ln["fondation"] = k
    return ln


def _lignes_postes(c: _Ctx) -> List[Dict]:
    """Les lignes dont le prix est commun au catalogue (les postes de la logique)."""
    out = []
    for p in B.postes_de(c.base, c.lg):
        code = p.get("code")
        if code == POSTE_FONDATION:
            continue
        o = B.prix_poste(p, c.cas)
        u = p.get("unite") or ""
        lib = p.get("poste") or code
        grp = GROUPE_DE.get(code) or ("loueur" if c.lg == "MA" and code not in MO else ("mo" if est_mo(code) else "accessoires"))
        if code == POSTE_CARBURANT:
            f, fu = _selon_unite(u, c)
            out.append(_ligne("CARB", lib, "energie", u, o, 1, f, fu))
            continue
        if code.startswith(PREFIXE_CHASSIS_HS):
            out.append(_ligne(code, lib, "location", u, o, 1, c.per.get("semaines") or 0.0, "sem."))
            continue
        if code in ("GRUTIER", "TELESC_GRUT"):
            hs = min(c.h, c.hn)
            f = c.jg if code == "GRUTIER" else c.nb_tel * c.jours_tel
            out.append(_ligne(code, lib, "mo", u, o, hs, f, "j", quantite_unite="h/j",
                              detail=f"{_fr(hs)} h/j × {_fr(f, 1)} j" + (" (calendrier réel)" if code == "GRUTIER" and c.per.get("grutier_reel") else "")))
            continue
        if code in ("HS", "TELESC_HS", "SUP_HS"):
            hs = max(0.0, c.h - c.hn)
            f = c.jg if code == "HS" else (c.nb_tel * c.jours_tel if code == "TELESC_HS" else c.jours)
            out.append(_ligne(code, lib, grp, u, o, hs, f, "j", quantite_unite="h/j"))
            continue
        if code in HORAIRES_MA:
            out.append(_ligne(code, lib, grp, u, o, c.q(code, 0.0), 1, "", quantite_editable=True, quantite_unite="h"))
            continue
        if code in PERSONNEL_MA:
            hp = _num((c.ma.get("heures_personnel") or {}).get(code), c.h) or 0.0
            f = max(hp, c.hn) * c.jours
            out.append(_ligne(code, lib, grp, u, o, c.q(code, 0.0), f, "h", quantite_editable=True, quantite_unite="pers.",
                              detail=f"max({_fr(hp)} ; {_fr(c.hn)}) h × {_fr(c.jours, 1)} j"))
            continue
        if code == "DEPL_GRUTIER":
            out.append(_ligne(code, lib, grp, u, o, c.q(code, 1.0), c.jours, "j", quantite_editable=True, quantite_unite="trajet/j"))
            continue
        if u == "%":
            out.append(_ligne(code, lib, grp, u, o, 1, 0.0, "€"))   # la base se calcule à la fin (les autres lignes du loueur)
            continue
        if u == "coef":
            out.append(_ligne(code, lib, grp, u, o, 1, 0.0, "€"))   # × le transport du ballast (à la fin)
            continue
        if code == "RAIL":
            out.append(_ligne(code, lib, grp, u, o, c.q(code, c.voie), c.jours, "j", quantite_editable=True, quantite_unite="m"))
            continue
        if code == "BUTEE":
            out.append(_ligne(code, lib, grp, u, o, c.q(code, c.butees), c.jours, "j", quantite_editable=True, quantite_unite="butée(s)"))
            continue
        if code in ("RAIL_POSE", "RAIL_DEPOSE"):
            n = math.ceil(c.voie / PAS_POSE_RAILS_M) if c.voie else 0
            out.append(_ligne(code, lib, grp, u, o, c.q(code, n), 1, "", quantite_editable=True, quantite_unite="j",
                              detail=f"{_fr(c.voie, 1)} m / {_fr(PAS_POSE_RAILS_M, 0)} m par jour, arrondi au-dessus"))
            continue
        if code in ("RAIL_TRANSPORT", "RAIL_EVAC"):
            n = math.ceil(c.voie / PAS_TRANSPORT_RAILS_M) if c.voie else 0
            out.append(_ligne(code, lib, grp, u, o, c.q(code, n), 1, "", quantite_editable=True, quantite_unite="transport(s)",
                              detail=f"{_fr(c.voie, 1)} m / {_fr(PAS_TRANSPORT_RAILS_M, 0)} m par transport, arrondi au-dessus"))
            continue
        f, fu = _selon_unite(u, c)
        defaut = QUANTITE_DEFAUT.get(code, 0.0 if c.lg == "MA" else 1.0)
        out.append(_ligne(code, lib, grp, u, o, c.q(code, defaut), f, fu, quantite_editable=True))
    return out


def _libres(c: _Ctx) -> List[Dict]:
    out = []
    for i, l in enumerate(c.s.get("libres") or []):
        if not isinstance(l, dict):
            continue
        prix = _num(l.get("prix"))
        o = {"valeur": prix, "annee": c.auj.year, "source": "ligne libre", "validite": None, "corrige": False,
             "a_verifier": False, "ref": ("libre", i, None)}
        u = l.get("unite") or "FF"
        f, fu = _selon_unite(u, c)
        ln = _ligne(f"LIBRE{i + 1}", l.get("libelle") or f"Ligne libre {i + 1}", "libres", u, o, _num(l.get("quantite"), 1.0) or 0.0,
                    f, fu, quantite_editable=True)
        ln["mo"] = bool(l.get("mo"))
        ln["libre"] = i
        out.append(ln)
    return out


# ------------------------------------------------------------------ le choix Auto
def _auto(code: str, ln: Dict, c: _Ctx, retenus: Dict[str, bool]) -> Tuple[bool, str]:
    """Le choix « Auto » d'une ligne (§ 6.1) et sa raison."""
    lg = c.lg
    if code == "LOC":
        return True, "location"
    if code in ("MONT", "DEMONT", "POIDS"):
        return True, "montage et démontage"
    if code == "FOND":
        return (lg != "MS"), ("pas de fondation pour une grue mobile louée sur une période" if lg == "MS" else "fondation")
    if code == "ELEC":
        return True, "électricité"
    if code == "CARB":
        return True, "carburant"
    if code == "CHROUL" or code in ROULANT:
        return c.chassis == "roulant", "châssis roulant" if c.chassis == "roulant" else "seulement avec un châssis roulant"
    if code in SANS_PRIX_ROULANT:
        return False, "sans prix : à demander"
    if code.startswith(PREFIXE_CHASSIS_HS):
        return c.chassis == "hors_standard", "châssis hors standard" if c.chassis == "hors_standard" else "seulement avec un châssis hors standard"
    if code == "TEL":
        return c.conduite == "telecommande", "conduite à la télécommande" if c.conduite == "telecommande" else "conduite " + NOMS_CONDUITE.get(c.conduite, "")
    if code == "TELV":
        return c.conduite == "telecommande_externe", "télécommande louée à l'extérieur" if c.conduite == "telecommande_externe" else "seulement si on la choisit"
    if code == "CAB":
        return c.conduite == "cabine", "conduite en cabine" if c.conduite == "cabine" else "conduite " + NOMS_CONDUITE.get(c.conduite, "")
    if code == "AI":
        return True, "anti-intrusion"
    if code in ("TLS", "TELESC_GRUT"):
        return c.nb_tel > 0, "télescopage" if c.nb_tel > 0 else "pas de télescopage"
    if code == "TELESC_HS":
        return c.nb_tel > 0 and c.h > c.hn, "heures sup. pendant le télescopage"
    if code == "GRUTIER":
        return (not c.operateur), ("grutier compris dans la location avec opérateur" if c.operateur else "grutier")
    if code in ("HS", "SUP_HS"):
        return c.h > c.hn, f"plus de {_fr(c.hn)} h par jour" if c.h > c.hn else f"{_fr(c.h)} h par jour"
    if code == "GB80" and c.inf32:
        return True, "raccordement inférieur à 32 A"
    if code == "DEPL":
        return True, "déplacement de la grue"
    if code == "BAL":
        return True, "transport du ballast"
    if code == "BALLAST_SAM":
        return bool(c.ma.get("ballast_sam")), "ballast le samedi ou la nuit"
    if code == "BALLAST_DIM":
        return bool(c.ma.get("ballast_dim")), "ballast le dimanche ou la nuit le week-end"
    if code == "DEPL_GRUTIER":
        return c.jours > 1, "location de plus d'un jour" if c.jours > 1 else "location d'un jour"
    if code in HORAIRES_MA or code in PERSONNEL_MA:
        return ln["quantite"] > 0, "selon l'horaire" if code in HORAIRES_MA else "selon le personnel"
    if code.startswith("LIBRE"):
        return True, "ligne libre"
    if code == "CONTRIB_ENV":
        return True, "contribution environnementale"
    if code in SUIT:
        lu = SUIT[code]
        r = retenus.get(lu, False)
        if code in ("TELEC_MONT", "TELEC_DEM"):
            return r and c.tel_payante, "suit la télécommande payante"
        return r, f"suit {lu}"
    if code in OUI_PAR_DEFAUT.get(lg, ()):
        return True, "retenu par défaut"
    if lg == "MA":
        return ln["quantite"] > 0, "selon la quantité"
    d = str(c.base and (B.poste(c.base, code) or {}).get("auto_par_defaut") or "Non").strip().lower()
    return d == "oui", "pas retenu par défaut" if d != "oui" else "retenu par défaut"


# ------------------------------------------------------------------ le calcul
def _evaluer(ln: Dict, c: _Ctx, retenus: Dict[str, bool]) -> None:
    code = ln["code"]
    auto, raison = _auto(code, ln, c, retenus)
    ch = c.choix.get(code, "auto")
    ln["choix"] = ch
    ln["auto"] = auto
    ln["auto_raison"] = raison
    ln["retenu"] = auto if ch == "auto" else ch == "oui"
    retenus[code] = ln["retenu"]
    f = c.forces.get(code)
    if isinstance(f, dict) and _num(f.get("prix")) is not None:
        ln["force"] = {"prix": _num(f.get("prix")), "justification": str(f.get("justification") or "").strip()}
        ln["prix"] = ln["force"]["prix"]
        ln["ecusson"] = None
    else:
        ln["force"] = None
        o = ln["origine"] or {}
        ln["prix"] = V.montant(o.get("valeur"))
        ln["ecusson"] = (ln["fondation"] or {}).get("ecusson") if code == "FOND" else E.ecusson(o, c.auj, besoin=ln["retenu"])
    n = V.nature((ln["origine"] or {}).get("valeur")) if not ln["force"] else V.NOMBRE
    ln["nature"] = n
    ln["manquant"] = ln["retenu"] and ln["prix"] is None
    ln["manquant_raison"] = {V.NON_DISPONIBLE: "non disponible pour ce modèle", V.SUR_DEMANDE: "sur demande au loueur",
                             V.TEXTE: "pas un prix"}.get(n, "prix inconnu") if ln["manquant"] else ""
    if ln["manquant"]:
        ln["ecusson"] = E.MANQUANT
    ln["montant"] = None if ln["prix"] is None else ln["prix"] * ln["quantite"] * ln["facturee"]
    ln["total"] = ln["montant"] if ln["retenu"] and ln["montant"] is not None else (None if ln["manquant"] else 0.0)


def calculer(base: Dict, saisie: Dict, aujourdhui: Optional[dt.date] = None) -> Dict:
    """Le prix d'une grue du projet. → {logique, catalogue, loueur, grue, periode, lignes, groupes, materiel, main_oeuvre,
    total, total_arrondi, manquants, attention, forces, ecusson, ecusson_raison, hypotheses, complet}."""
    aujourdhui = aujourdhui or dt.date.today()
    c = _Ctx(base, saisie, aujourdhui)
    out = {"logique": c.lg, "catalogue": c.cas, "loueur": c.nom_loueur, "grue": c.g.get("id"), "modele": c.g.get("modele"),
           "periode": c.per, "lignes": [], "groupes": [], "materiel": 0.0, "main_oeuvre": 0.0, "total": 0.0,
           "total_arrondi": 0, "manquants": [], "attention": [], "ecusson": E.MANQUANT, "ecusson_raison": "",
           "hypotheses": [], "complet": False, "pret": False}
    if not c.lg:
        out["ecusson_raison"] = "choisir un catalogue"
        return out
    if not c.g:
        out["ecusson_raison"] = "choisir une grue"
        return out
    out["pret"] = True
    lignes = _lignes_grue(c)
    if c.lg in ("T", "R", "MS", "MA"):
        lignes.append(_ligne_fondation(c))
    lignes += _lignes_postes(c) + _libres(c)
    # TEL est « payante » quand son prix est connu et non nul : ses montage et démontage la suivent alors
    tel = next((ln for ln in lignes if ln["code"] == "TEL"), None)
    c.tel_payante = bool(tel and (V.montant((tel["origine"] or {}).get("valeur")) or 0) > 0)
    retenus: Dict[str, bool] = {}
    for ln in lignes:
        if ln["code"] not in SUIT:
            _evaluer(ln, c, retenus)
    for ln in lignes:
        if ln["code"] in SUIT:
            _evaluer(ln, c, retenus)
    # les coefficients du ballast (× le transport du ballast) et la contribution environnementale (% des lignes du loueur)
    bal = next((ln for ln in lignes if ln["code"] == "BAL"), None)
    for ln in lignes:
        if ln["unite"] == "coef":
            base_bal = bal["prix"] if bal and bal["prix"] is not None else None
            ln["facturee"] = base_bal or 0.0
            ln["facturee_unite"] = "€"
            ln["detail"] = f"× transport du ballast ({_fr(base_bal)} €)" if base_bal is not None else "transport du ballast inconnu"
            ln["montant"] = None if ln["prix"] is None or base_bal is None else ln["prix"] * base_bal
            if ln["retenu"] and base_bal is None and not ln["manquant"]:
                ln["manquant"], ln["manquant_raison"], ln["ecusson"] = True, "transport du ballast inconnu", E.MANQUANT
            ln["total"] = ln["montant"] if ln["retenu"] and ln["montant"] is not None else (None if ln["manquant"] else 0.0)
    for ln in lignes:
        if ln["unite"] == "%":
            assiette = sum(x["total"] or 0.0 for x in lignes if x["retenu"] and x is not ln and x["code"] != "FOND"
                           and not x["code"].startswith("LIBRE") and x["unite"] != "%")
            ln["facturee"] = assiette
            ln["facturee_unite"] = "€"
            ln["detail"] = f"{_fr(ln['prix'])} % × {_fr(assiette)} € (lignes du loueur, sans la fondation)" if ln["prix"] is not None else ""
            ln["montant"] = None if ln["prix"] is None else ln["prix"] / 100.0 * assiette
            ln["quantite"] = 1
            ln["total"] = ln["montant"] if ln["retenu"] and ln["montant"] is not None else (None if ln["manquant"] else 0.0)
    out["lignes"] = lignes
    # les groupes et les totaux
    noms = dict(GROUPES)
    if c.lg == "MA":
        noms["loueur"] = c.nom_loueur or "Loueur"
    if c.lg == "MS":
        noms["energie"] = "Carburant"
    for k, _ in GROUPES:
        ls = [ln for ln in lignes if ln["groupe"] == k]
        if ls:
            out["groupes"].append({"cle": k, "libelle": noms[k], "total": sum(ln["total"] or 0.0 for ln in ls if ln["retenu"]),
                                   "retenues": sum(1 for ln in ls if ln["retenu"]), "lignes": [ln["code"] for ln in ls]})
    out["materiel"] = sum(ln["total"] or 0.0 for ln in lignes if ln["retenu"] and not ln["mo"])
    out["main_oeuvre"] = sum(ln["total"] or 0.0 for ln in lignes if ln["retenu"] and ln["mo"])
    out["total"] = out["materiel"] + out["main_oeuvre"]
    out["total_arrondi"] = int(math.floor(out["total"] + 0.5))
    out["manquants"] = [{"code": ln["code"], "libelle": ln["libelle"], "raison": ln["manquant_raison"]} for ln in lignes if ln["manquant"]]
    out["forces"] = [{"code": ln["code"], "libelle": ln["libelle"], "prix": ln["force"]["prix"], "justification": ln["force"]["justification"]}
                     for ln in lignes if ln["force"] and ln["retenu"]]
    out["complet"] = not out["manquants"]
    out["attention"] = attention(c, out)
    ecs = [E.pour_ensemble(ln["ecusson"], ln["origine"], aujourdhui) for ln in lignes if ln["retenu"] and not ln["force"] and ln["ecusson"]]
    if out["manquants"]:
        n = len(out["manquants"])
        out["ecusson"], out["ecusson_raison"] = E.MANQUANT, f"{n} prix manquant{'s' if n > 1 else ''}"
    else:
        ag = E.agreger(ecs, vide=E.A_VERIFIER)
        cpt = E.compter(ecs)
        out["ecusson"] = ag
        if ag == E.PAS_A_JOUR:
            n = cpt[E.PAS_A_JOUR]
            out["ecusson_raison"] = f"{n} prix pas à jour"
        elif ag == E.A_VERIFIER:
            n = cpt[E.A_VERIFIER]
            out["ecusson_raison"] = f"{n} prix à vérifier"
        else:
            out["ecusson_raison"] = "tous les prix sont à jour"
    out["hypotheses"] = hypotheses(c, out)
    return out


# ------------------------------------------------------------------ hypothèses et points d'attention
def hypotheses(c: _Ctx, r: Dict) -> List[Tuple[str, str]]:
    p = c.per
    out = []
    if c.lg == "MA":
        hf, _ = heures_facturees(c)
        out += [("Jours", _fr(p.get("jours"), 1)), ("Heures par jour", f"{_fr(c.h)} h (facturées : {_fr(hf)} h)")]
    else:
        out += [("Mois", _fr(p.get("mois"), 1)), ("Jours facturés", f"{_fr(p.get('jours'), 1)} (mois × {_fr(p.get('jours_par_mois'))})"),
                ("Jours du grutier", _fr(p.get("jours_grutier"), 1) + (" (calendrier réel)" if p.get("grutier_reel") else "")),
                ("Heures par jour", f"{_fr(c.h)} h")]
    if c.lg == "T":
        out.append(("Châssis", NOMS_CHASSIS[c.chassis]))
    if c.lg in ("T", "R"):
        out.append(("Conduite", NOMS_CONDUITE.get(c.conduite, "—")))
    fd = next((ln for ln in r["lignes"] if ln["code"] == "FOND"), None)
    if fd:
        k = fd["fondation"] or {}
        out.append(("Fondation", f"{(k.get('type') or '').lower()}, {'avec' if k.get('demolition') else 'sans'} démolition"))
    el = next((ln for ln in r["lignes"] if ln["code"] == "ELEC"), None)
    if el and el["unite"] == "€/kWh":
        o = el["origine"] or {}
        out.append(("Prix du kWh", f"{_fr(V.montant(o.get('valeur')), 3)} €/kWh" + (f" ({o.get('annee')})" if o.get("annee") else "")))
    if c.lg == "MS":
        out.append(("Opérateur", "avec opérateur" if c.operateur else "sans opérateur (grutier à part)"))
    return out


def _att(texte: str, cible: Optional[str] = None, niveau: str = "attention") -> Dict:
    return {"texte": texte, "cible": cible, "niveau": niveau}


def variante_hsc(base: Dict, g: Dict, hsc) -> Optional[Dict]:
    """La HSC saisie et la variante de la grue (§ 6 « HSC et variante ») : la variante « > 39 m » quand la HSC tient dans
    « max. 39 m » (et l'inverse). → {texte, autre (id de l'autre variante)} ou ``None``."""
    h = _num(hsc)
    if h is None:
        return None
    vmin, vmax = _num(g.get("hsc_variante_min_m")), _num(g.get("hsc_variante_max_m"))
    autres = [x for x in B.grues_du(base, g.get("cas")) if x.get("id") != g.get("id")]
    if vmin is not None and h <= vmin:
        autre = next((x for x in autres if _num(x.get("hsc_variante_max_m")) is not None and h <= _num(x.get("hsc_variante_max_m"))), None)
        return {"texte": f"HSC {_fr(h)} m ≤ {_fr(vmin)} m : la variante max. {_fr(vmin)} m suffit", "autre": autre.get("id") if autre else None}
    if vmax is not None and h > vmax:
        autre = next((x for x in autres if _num(x.get("hsc_variante_min_m")) is not None and h > _num(x.get("hsc_variante_min_m"))), None)
        return {"texte": f"HSC {_fr(h)} m > {_fr(vmax)} m : prendre la variante > {_fr(vmax)} m", "autre": autre.get("id") if autre else None}
    return None


def attention(c: _Ctx, r: Dict) -> List[Dict]:
    out = []
    lignes = {ln["code"]: ln for ln in r["lignes"]}
    v = variante_hsc(c.base, c.g, c.s.get("hsc_m")) if c.lg == "T" else None
    if v:
        a = _att(v["texte"], "hsc_m")
        a["autre"] = v["autre"]
        out.append(a)
    if c.lg == "T" and c.chassis == "roulant":
        out.append(_att("Châssis roulant : vérifier la longueur de voie et le nombre de butées ; prix du train de roulement et du "
                        "capteur à demander ; combinaison des prix à confirmer avec le loueur", "chassis"))
        if not c.voie:
            out.append(_att("Châssis roulant : longueur de voie à saisir", "voie_m"))
    if c.lg == "MS":
        out.append(_att("Grue mobile : le grutier et le carburant sont-ils compris ?", "operateur"))
    if c.lg == "MA":
        hf, m = heures_facturees(c)
        if c.jours <= 1 and m is None:
            out.append(_att("Minimum facturé de la grue inconnu : les heures saisies sont facturées telles quelles", "heures_jour"))
        if lignes.get("DEPL") and lignes["DEPL"]["retenu"] and not lignes["DEPL"]["quantite"]:
            out.append(_att("Heures de déplacement de la grue à saisir", "DEPL"))
    vieux = [ln for ln in r["lignes"] if ln["retenu"] and not ln["force"] and ln["ecusson"] == E.PAS_A_JOUR]
    if vieux:
        ans = sorted({str((ln["origine"] or {}).get("annee")) for ln in vieux if (ln["origine"] or {}).get("annee")})
        out.append(_att(f"Tarif pas à jour : {len(vieux)} prix d'une année passée" + (f" ({', '.join(ans)})" if ans else "")))
    if len(c.per.get("annees") or []) > 1:
        a = c.per["annees"]
        out.append(_att(f"Période sur {a[0]} et {a[-1]} : vérifier l'indexation des prix", "periode"))
    nd = c.g.get("nb_disponibles")
    if V.est_nombre(nd) and nd == 0:
        out.append(_att("Parc : le loueur indique 0 grue disponible pour l'année", "grue"))
    if c.inf32:
        out.append(_att("Raccordement inférieur à 32 A : Greenbox nécessaire", "GB80"))
    if lignes.get("GB140") and lignes["GB140"]["retenu"]:
        out.append(_att("Greenbox 140 kVA : elle sert à deux grues, ne la compter que sur une des deux", "GB140"))
    if lignes.get("TEL") and lignes.get("CAB") and lignes["TEL"]["retenu"] and lignes["CAB"]["retenu"]:
        out.append(_att("Cabine et télécommande retenues toutes les deux : une grue se conduit en cabine ou à la télécommande", "conduite"))
    if lignes.get("TELV") and lignes.get("TEL") and lignes["TELV"]["retenu"] and lignes["TEL"]["retenu"]:
        out.append(_att("Deux télécommandes retenues (celle du loueur et une louée à l'extérieur)", "conduite"))
    for ln in r["lignes"]:
        if ln["force"] and ln["retenu"] and not ln["force"]["justification"]:
            out.append(_att(f"Prix forcé sans justification : {ln['libelle']}", ln["code"]))
    nf = len(r["forces"])
    if nf:
        out.append(_att(f"{nf} prix forcé{'s' if nf > 1 else ''}", None, "info"))
    fd = lignes.get("FOND")
    if fd and fd["retenu"] and not fd["force"] and (fd["fondation"] or {}).get("a_verifier"):
        n = fd["fondation"]["a_verifier"]
        out.append(_att(f"Fondation : {n} retour{'s' if n > 1 else ''} de chantier à valider", "FOND", "info"))
    for cd in B.conditions(c.base, c.nom_loueur):
        if "attention" in str(cd.get("effet") or ""):
            out.append(_att(f"Conditions du loueur : {cd.get('texte')}", None, "info"))
    return out
