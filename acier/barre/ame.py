# -*- coding: utf-8 -*-
"""L'ÂME d'une barre vérifiée selon l'EN 1993-1-5 — le troisième bloc de résultats, à côté des deux méthodes, et
TOUJOURS calculé : la méthode historique ne traite ni le voilement de l'âme, ni la charge concentrée, ni les
raidisseurs (docs/acier/BARRES_ACIER.md § 5.9 ; brief § 6.5 et § 8.3).

    le voilement par CISAILLEMENT d'une âme élancée (§ 5, et § 7.1 avec la flexion), dans chaque section à vérifier ;
    l'âme sous une FORCE CONCENTRÉE (§ 6), à CHAQUE appui posé et sous CHAQUE charge ponctuelle qui appuie sur une
    semelle — seule (η₂), puis avec la flexion (§ 7.2) ;
    LA DÉCISION, lieu par lieu : « pas de raidisseur », avec les taux ; ou « raidisseur » — une paire de plats
    dimensionnée et vérifiée (raidisseurs.py, § 9) ;
    le voilement de l'âme induit par la semelle comprimée (§ 8).

Les efforts sont ceux de CALCUL : les combinaisons ELU (6.10) de la barre chargée, quelle que soit la méthode cochée.

OÙ UNE FORCE APPUIE SUR L'ÂME (les « lieux ») :
- un APPUI simple, quand la barre y est POSÉE (``appuis.mode`` « poses ») et que la réaction pousse la semelle basse ;
  un appui « par l'âme » (platine, cornières, gousset) n'appuie sur aucune semelle ; un encastrement est un assemblage,
  hors du module ; une réaction de soulèvement demande un ancrage, hors du module — signalée ;
- une CHARGE PONCTUELLE qui pousse la semelle où elle est posée : vers le bas sur la semelle haute, vers le haut sous
  la semelle basse (``niveau``) ; appliquée au centre de cisaillement, elle entre par l'âme.
Les charges ponctuelles de plusieurs cas à la même abscisse sont UNE force (la plus courte de leurs longueurs d'appui) ;
une charge au droit d'un appui partage son lieu : chaque semelle chargée y est vérifiée, et la force qui passe d'une
semelle à l'autre — la plus petite des deux — l'est en type (b).

LE TYPE (figure 6.1) se déduit de la géométrie : (c) près d'un bout de la barre, tant que 2 + 6·(s_s + c)/h_w < 6 —
c étant la distance du bout de la barre au bord de la longueur chargée, avec le débord saisi pour un appui de rive — ;
(a) ailleurs.

UN RAIDISSEUR EST NÉCESSAIRE dès qu'une combinaison dépasse η₂ = 1 ou (η₂ + 0,8·η₁)/1,4 = 1 ; et à chaque appui
posé d'une âme élancée au cisaillement (§ 5.1(2)). Il est dimensionné pour la plus grande force du lieu.

Ce que ce bloc ne fait pas (v1), et dit : la charge concentrée sur un profil creux (EN 1993-1-8) ; les raidisseurs
intermédiaires d'une âme élancée (elle est vérifiée sans eux) ; les soudures des raidisseurs (EN 1993-1-8) ; DEUX FORCES
RAPPROCHÉES sur la même semelle — leurs longueurs chargées l_y se recouvrent —, que le § 6.3(2) demande de vérifier aussi
ENSEMBLE (leur somme, sur leur entraxe) : la vérification « A-GR » est alors rendue « non couvert », jamais « ok ».
"""
from typing import Dict, List, Optional, Sequence, Tuple

from acier.flambement import normes as NO

from . import combinaisons as CO
from . import ec3_ame as AM
from . import ec3_section as ES
from . import parametres as PA
from . import raidisseurs as RA
from . import resultat as RE
from . import statique as ST
from .enveloppe import Enveloppe

METHODE, NORME = AM.METHODE, AM.NORME
EPS = 1e-9
MEME_ABSCISSE = 1e-6             # m : deux forces à la même abscisse
MODES_APPUI = ("poses", "ame")
COTES = {"haut": "semelle haute", "bas": "semelle basse"}


def _fy(b: Dict, s: Dict) -> Tuple[float, float, Optional[str]]:
    """(f_y de calcul de la barre, f_y de son âme selon t_w, d'où vient la première) — la table unique, lue par
    ec3_section.lire ; Refus si la nuance ou l'épaisseur en sortent."""
    d = ES.lire({"section": s, "nuance": b["nuance"], "parametres": b["parametres"]})
    return d["fy"], d["fy_parois"]["ame"], d["fy_source"]


def _fy_semelle(b: Dict, s: Dict) -> float:
    """f_y,f — la f_y des semelles selon t_f (table unique) : celle du voilement induit par la semelle (§ 8)."""
    d = ES.lire({"section": s, "nuance": b["nuance"], "parametres": b["parametres"]})
    return d["fy_parois"].get("semelle", d["fy"])


def refusee(message: str) -> Dict:
    """Le bloc d'une barre dont l'âme ne peut pas être vérifiée (une épaisseur au-delà de 80 mm (domaine du module)…) : « non couvert »."""
    J = RE.Journal(METHODE, NORME)
    J.verif("A-AME", "Vérification impossible", NORME, "", None, None, "", [], statut=RE.NON_COUVERT,
            messages=[J.alerte("refus", "attention", message)])
    r = J.rendre()
    r.update({"combinaison": None, "combinaisons": [], "a_confirmer": [], "lieux": []})
    return r


def _communes(b: Dict, s: Dict, fy: float, P: Dict, fyw: Optional[float] = None, source: Optional[str] = None,
              fyf: Optional[float] = None) -> Tuple[List[str], Optional[Dict], Optional[Dict]]:
    """Les hypothèses que les deux modes disent, l'élancement de l'âme (None : sans objet) et le voilement induit par
    la semelle (None hors des I). ``fyw`` : la f_y de l'âme selon t_w ; ``source`` : d'où vient la f_y de calcul ;
    ``fyf`` : celle des semelles (§ 8)."""
    fyw = fy if fyw is None else fyw
    ame_fy = f" (âme : f_y,w = {RE.fr(fyw)} N/mm², son ε pour 72·ε/η, λ̄_w et λ̄_F)" if fyw != fy else ""
    hyp = [f"Efforts de calcul (ELU) ; γ_M0 = {RE.fr(P['gamma_M0'], 2)}, γ_M1 = {RE.fr(P['gamma_M1'], 2)}.",
           f"Acier {b['nuance']} : f_y = {RE.fr(fy)} N/mm²" + (f" ({source})" if source else "") + ame_fy
           + f" ; η = {RE.fr(AM.eta_de(s, P), 2)}."]
    el = AM.elancement(s, fy, P, fy_ame=fyw)
    if el is not None:
        if el["elancee"]:
            hyp.append(f"Âme élancée (h_w/t_w = {RE.fr(el['rapport'], 1)} > 72·ε/η = {RE.fr(el['limite'], 1)}) : voilement par cisaillement "
                       "vérifié sans raidisseur intermédiaire, montant d'extrémité non rigide, sans la contribution des semelles ; "
                       "interaction avec la flexion vérifiée dans toutes les sections.")
        else:
            hyp.append(f"Âme : h_w/t_w = {RE.fr(el['rapport'], 1)} ≤ 72·ε/η = {RE.fr(el['limite'], 1)} — pas de voilement par cisaillement "
                       "à vérifier (§ 5.1(2)).")
    sm = AM.semelle(s, fy, P, fy_semelle=fyf)
    if sm is not None and sm["ok"]:
        hyp.append(f"Voilement de l'âme induit par la semelle (§ 8) : h_w/t_w = {RE.fr(sm['rapport'], 1)} ≤ {RE.fr(sm['limite'])} — "
                   "sans objet.")
    return hyp, el, sm


# ------------------------------------------------------------------ efforts imposés
def efforts(b: Dict, s: Dict) -> Dict:
    """Le bloc de l'âme d'une barre aux EFFORTS IMPOSÉS : le voilement par cisaillement sous les efforts de calcul. Les
    appuis et les charges ne sont pas connus : aucune charge concentrée n'est vérifiée, et le bloc le dit."""
    try:
        P = PA.jeu(b["parametres"])
        fy, fyw, src = _fy(b, s)
    except RE.Refus as e:
        return refusee(str(e))
    e = b["efforts"]["elu"]
    J = RE.Journal(METHODE, NORME)
    hyp, _, sm = _communes(b, s, fy, P, fyw, src, _fy_semelle(b, s))
    AM.cisaillement(J, s, fy, P, abs(e["Vz"]) * 1e3, abs(e["My"]) * 1e6, e["N"] * 1e3, fy_ame=fyw)
    AM.cisaillement(J, s, fy, P, abs(e["Vy"]) * 1e3, abs(e["Mz"]) * 1e6, e["N"] * 1e3, direction="y", fy_ame=fyw)
    if sm is not None and not sm["ok"]:
        AM.ecrire_semelle(J, s, fy, P, sm)
    hyp.append("Efforts imposés : les appuis et les charges ne sont pas décrits — l'âme sous une force concentrée n'est pas vérifiée.")
    r = J.rendre()
    r.update({"hypotheses": hyp, "combinaison": None, "combinaisons": [], "a_confirmer": [], "lieux": []})
    return r


# ------------------------------------------------------------------ barre chargée : les lieux
def lieux(b: Dict, cas: Sequence[Dict], sols: Sequence) -> List[Dict]:
    """Les endroits où une force appuie sur l'âme, de gauche à droite : {"base" (préfixe de ses vérifications), "nom",
    "genre" (« appui » | « charge »), "x" (m), "appui" (son rang, ou None), "charges" [(rang du cas, valeur kN)],
    "ss" {"haut", "bas"} (mm ; None : semelle non chargée), "dispo" (mm d'âme jusqu'à chaque bout de la barre)}."""
    Lt = sum(b["travees"])
    ap = b["appuis"]
    appuis = list(sols[0].appuis) if sols else []
    poses = ap["mode"] == "poses"

    def depassement(k: int) -> float:
        """L'âme au-delà de l'axe d'un appui de rive posé : la moitié de sa longueur d'appui, et son débord."""
        x_bout = 0.0 if k == 0 else Lt
        a = appuis[k] if appuis else None
        if a is not None and poses and a["type"] == "appui" and abs(a["x"] - x_bout) < MEME_ABSCISSE:
            return ap["ss"] / 2.0 + ap["c"]
        return 0.0

    out: List[Dict] = []
    if poses:
        for i, a in enumerate(appuis):
            if a["type"] == "appui":
                out.append({"x": a["x"], "appui": i, "nom": f"Appui {a['nom']}", "genre": "appui", "charges": [],
                            "ss": {"haut": None, "bas": ap["ss"]}, "contact": {"haut": None, "bas": ap.get("contact", "inconnu")}})
    if b["niveau"] in ("haut", "bas"):
        groupes: Dict[float, Dict] = {}
        for ci, c in enumerate(cas):
            for ch in c.get("charges") or []:
                if ch["type"] == "P" and ch["v"]:
                    g = groupes.setdefault(round(ch["a"], 6), {"x": ch["a"], "parts": [], "ss": [], "contact": []})
                    g["parts"].append((ci, ch["v"]))
                    g["ss"].append(ch.get("ss", 0.0))
                    g["contact"].append(ch.get("contact", "inconnu"))
        for g in sorted(groupes.values(), key=lambda g: g["x"]):
            hote = next((L for L in out if L["appui"] is not None and abs(L["x"] - g["x"]) < MEME_ABSCISSE), None)
            if hote is None:
                hote = {"x": g["x"], "appui": None, "nom": f"Charge à {RE.fr(g['x'], 2)} m", "genre": "charge", "charges": [],
                        "ss": {"haut": None, "bas": None}, "contact": {"haut": None, "bas": None}}
                out.append(hote)
            hote["charges"] = g["parts"]
            if hote["ss"][b["niveau"]] is None:
                hote["ss"][b["niveau"]] = min(g["ss"])
                # le contact de plusieurs charges : établi s'il l'est pour toutes ; « non » dès qu'une le refuse
                hote["contact"][b["niveau"]] = ("oui" if all(k == "oui" for k in g["contact"]) else
                                                "non" if "non" in g["contact"] else "inconnu")
    out.sort(key=lambda L: L["x"])
    e_g, e_d = depassement(0), depassement(-1)
    for k, L in enumerate(out):
        L["base"] = f"A-L{k:02d}"
        L["dispo"] = (L["x"] * 1e3 + e_g, (Lt - L["x"]) * 1e3 + e_d)
    return out


def forces(L: Dict, niveau: str, sols: Sequence, facteurs: Sequence[float]) -> Dict[str, float]:
    """Les forces qui APPUIENT sur chaque semelle du lieu sous une combinaison (kN, ≥ 0) : {"haut", "bas"} ; et
    "soulevement" (kN > 0) quand la réaction de l'appui tire la barre vers le bas."""
    P = sum(facteurs[ci] * v for ci, v in L["charges"])
    R = 0.0
    if L["appui"] is not None:
        R = sum(f * sol.appuis[L["appui"]]["R"] for f, sol in zip(facteurs, sols) if f)
    return {"haut": P if (niveau == "haut" and P > EPS) else 0.0,
            "bas": (R if R > EPS else 0.0) + (-P if (niveau == "bas" and P < -EPS) else 0.0),
            "soulevement": -R if R < -EPS else 0.0}


def resistances(L: Dict, s: Dict, fy: float, P: Dict, fyw: Optional[float] = None) -> Dict[str, Dict]:
    """La résistance de l'âme SEULE au lieu ``L``, par semelle chargée (« haut », « bas ») et, quand les deux le sont,
    pour la force transmise de l'une à l'autre (« b ») — acier/barre/ec3_ame.resistance, contact d'extrémité non établi
    (le verdict du type (c) ; la variante avec contact y est donnée pour information) ; ``fyw`` : la f_y de l'âme.
    Chaque résistance garde l'âme qui l'a donnée (« ame » : b_f et t_f de la semelle chargée, brief § 8.2) ; la force
    transmise d'un PRS à semelles inégales prend la semelle qui donne la plus petite F_Rd (choix prudent déclaré)."""
    d = min(L["dispo"])
    out = {}
    for cote, ss in L["ss"].items():
        if ss is None:
            continue
        a = AM.ame_de(s, cote)
        hw = a["hw"]
        c = max(d - ss / 2.0, 0.0)
        type_ = "c" if 2.0 + 6.0 * (min(ss, hw) + c) / hw < AM.KF_C_MAX else "a"
        contact = (L.get("contact") or {}).get(cote) or "inconnu"
        out[cote] = dict(AM.resistance(a, fy, P, ss, type_, c, contact=contact, fy_ame=fyw), ame=a)
    if len(out) == 2:
        essais = [dict(AM.resistance(AM.ame_de(s, k), fy, P, min(L["ss"]["haut"], L["ss"]["bas"]), "b", fy_ame=fyw), ame=AM.ame_de(s, k))
                  for k in ("haut", "bas")]
        out["b"] = min(essais, key=lambda r: r["FRd"])
    return out


def _titre(L: Dict, cote: str) -> str:
    if cote == "b":
        return f"{L['nom']} : force transmise d'une semelle à l'autre"
    if L["genre"] == "charge":
        return f"{L['nom']} : âme sous la charge"
    return f"{L['nom']} : âme sous la réaction" if cote == "bas" else f"{L['nom']} : âme sous la charge appliquée"


def _saisie(L: Dict, cote: str) -> float:
    """La longueur d'appui SAISIE de la force vérifiée (mm) : celle de sa semelle ; la plus courte des deux pour la
    force transmise."""
    return min(L["ss"].values()) if cote == "b" else L["ss"][cote]


def _mots_force(r: Dict, saisie: float) -> List[str]:
    """Ce qu'on dit de la force vérifiée : sa longueur d'appui quand elle n'a pas été saisie, ou qu'elle est bornée ;
    puis le type de la force et les choix du calcul (ec3_ame.mots_resistance)."""
    mots = []
    if saisie <= 0:
        mots.append("Longueur d'appui s_s non saisie : 0, le cas le plus défavorable.")
    elif saisie > r["ss"]:
        mots.append(f"s_s bornée à h_w = {RE.fr(r['ss'], 1)} mm.")
    return mots + AM.mots_resistance(r)


# ------------------------------------------------------------------ barre chargée
def chargee(b: Dict, s: Dict, cas: Sequence[Dict], sols: Sequence, N_cas: Sequence[float], pts: Sequence[float]) -> Dict:
    """Le bloc de l'âme d'une barre CHARGÉE : la forme de acier/barre/resultat.py (chaque vérification dit sa combinaison
    et son abscisse), et "lieux" — la décision de chaque appui et de chaque charge ponctuelle :

        {"base", "nom", "genre", "x", "type", "ss" (mm), "F_Ed" (kN, la plus grande force du lieu), "sans": {"F_Rd" (kN),
         "eta2", "interaction" (None sans flexion)} (l'âme seule, au plus défavorable), "decision" (« aucun » |
         « raidisseur »), "raison", "raidisseur" (None | {"bs", "ts", "chanfrein", "designation"}), "verifications"
         (leurs identifiants), "gouvernant", "taux", "statut"}."""
    try:
        P = PA.jeu(b["parametres"])
        fy, fyw, src = _fy(b, s)
    except RE.Refus as e:
        return refusee(str(e))
    gM0 = P["gamma_M0"]
    env = Enveloppe(METHODE, NORME)
    hyp, el, sm = _communes(b, s, fy, P, fyw, src, _fy_semelle(b, s))
    alertes: Dict[str, Dict] = {}
    combs = [ST.Combinaison(co["nom"], sols, co["facteurs"], N_cas) for co in CO.elu(cas, P)]

    # ---- le voilement par cisaillement, dans chaque section à vérifier
    if el is not None and el["elancee"]:
        for C in combs:
            for x in ST.sections(b, C, pts):
                for M, V in ST.cotes(C.lire(x)):
                    if abs(V) < EPS:
                        continue
                    J = RE.Journal(METHODE, NORME)
                    AM.cisaillement(J, s, fy, P, abs(V) * 1e3, abs(M) * 1e6, C.N * 1e3, fy_ame=fyw)
                    env.verser(J.rendre(), {"combinaison": C.nom, "x": x})
    if sm is not None and not sm["ok"]:
        J = RE.Journal(METHODE, NORME)
        AM.ecrire_semelle(J, s, fy, P, sm)
        env.verser(J.rendre(), {"combinaison": combs[0].nom})

    # ---- les forces concentrées : où, combien, sous quelle combinaison
    endroits = lieux(b, cas, sols)
    a = AM.ame_de(s)
    ap = b["appuis"]
    genres = {x["type"] for x in (sols[0].appuis if sols else [])}
    if genres - {"appui"}:
        hyp.append("Encastrement : un assemblage, hors du module — aucune force concentrée n'y est vérifiée.")
    if "appui" in genres and ap["mode"] == "poses":
        hyp.append(f"Barre posée sur ses appuis : longueur d'appui s_s = {RE.fr(ap['ss'])} mm"
                   + (" (non saisie : 0, le cas le plus défavorable)" if ap["ss"] <= 0 else "")
                   + f", débord en rive c = {RE.fr(ap['c'])} mm.")
    elif "appui" in genres:
        hyp.append("Appuis par l'âme (platine, cornières, gousset) : aucune force concentrée sur l'âme aux appuis.")
    if any(ch["type"] == "P" for c in cas for ch in c.get("charges") or []):
        hyp.append({"haut": "Charges ponctuelles posées sur la semelle haute.",
                    "bas": "Charges ponctuelles suspendues à la semelle basse : seule une charge vers le haut appuie sur l'âme.",
                    "centre": "Charges ponctuelles appliquées au centre de cisaillement : elles entrent par l'âme, sans force concentrée "
                              "sur une semelle."}[b["niveau"]])
    occurrences: Dict[str, List[Dict]] = {}
    for L in endroits:
        occ = []
        for C in combs:
            f = forces(L, b["niveau"], sols, C.facteurs)
            if f["soulevement"] > 0:
                alertes.setdefault(f"soulevement_{L['base']}", {"code": f"soulevement_{L['base']}", "niveau": "info",
                                                                "message": f"{L['nom']} : réaction de soulèvement ({C.nom}) — ancrage hors du module."})
            if f["haut"] > 0 or f["bas"] > 0:
                e = C.lire(L["x"])
                Ms = e["M_g"] if abs(e["M_g"]) >= abs(e["M_d"]) else e["M_d"]
                occ.append({"C": C, "haut": f["haut"], "bas": f["bas"], "N": C.N, "M": max(abs(e["M_g"]), abs(e["M_d"])), "M_s": Ms,
                            "V": max(abs(e["V_g"]), abs(e["V_d"]))})
        occurrences[L["base"]] = occ
    charges = [L for L in endroits if occurrences[L["base"]]]
    if a is None:
        if charges and s["forme"] in ("tube_carre", "tube_rect", "tube_rond"):
            alertes["tube_charge"] = {"code": "tube_charge", "niveau": "info",
                                      "message": "Profil creux : l'âme sous une force concentrée (appuis, charges ponctuelles) n'est pas "
                                                 "vérifiée ici — EN 1993-1-8."}
        charges = []
    else:
        hyp.append("Force concentrée : sans raidisseur transversal voisin (a → ∞) ; η₁ élastique (§ 7.2) ; semelle chargée tendue : "
                   "critère 6.2.1(5) de l'EN 1993-1-1 à la jonction (§ 7.2(2)) ; semelle comprimée supposée tenue latéralement (§ 6.1(1)).")
        if any(L["genre"] == "appui" for L in endroits):
            hyp.append("Le déversement suppose des appuis en fourche : un raidisseur d'appui peut être nécessaire pour l'assurer.")

    # ---- la classe de la section : elle ne change A et W (η₁) qu'en classe 4
    classe4, classe = False, None
    if charges:
        g0 = ES.verifier({"section": s, "nuance": b["nuance"], "NEd": 1.0, "parametres": b["parametres"], "voilement": False})["grandeurs"]
        classe = g0.get("classe")
        classe4 = classe == 4

    def section_elastique(N: float, M: float) -> Tuple[float, float, str, str]:
        """(A, W, leurs noms) du η₁ élastique sous (N kN, M kN·m) : A et W_el ; A_eff et W_eff en classe 4."""
        if classe4:
            g = ES.verifier({"section": s, "nuance": b["nuance"], "NEd": N * 1e3, "MyEd": M * 1e6, "parametres": b["parametres"],
                             "voilement": False})["grandeurs"]
            ef = g.get("efficace")
            if g.get("classe") == 4 and ef is not None:
                return (ef["A_eff"] if N > 0 else s["A"], ef["Weff_y"] or s["Wel_y"], "A_eff" if N > 0 else "A", "W_eff,y")
        return s["A"], s["Wel_y"], "A", "W_el,y"

    # ---- la semelle chargée est-elle comprimée ou tendue (brief § 8.2) ? σ_x au plan moyen ; le critère 6.2.1(5) à la jonction
    def elastique(o: Dict, k: str, F: float, r: Dict) -> Optional[Dict]:
        """Le contrôle de la semelle chargée ``k`` sous la force F (kN) de l'occurrence ``o`` : {"etat", "sx", "sz", "tau",
        "crit", …} — "crit" None quand la semelle est comprimée partout (l'interaction (7.2) seule)."""
        A, I = s["A"], s["Iy"]
        if classe4:                                       # classe 4 : A_eff, et I_eff tiré de W_eff (À VÉRIFIER : approché)
            A_, W_, _, _ = section_elastique(o["N"], o["M"])
            A, I = A_, s["Iy"] * W_ / s["Wel_y"]
        nv = AM.niveaux_semelle(s, k)
        etat = AM.etat_semelle([AM.sigma_x(o["N"] * 1e3, o["M_s"] * 1e6, A, I, nv["z_moyen"])])
        out = {"etat": etat, "cote": k, "F": F, "A": A, "I": I, "nv": nv, "crit": None}
        if etat == AM.COMPRIMEE:
            return out
        sx = AM.sigma_x(o["N"] * 1e3, o["M_s"] * 1e6, A, I, nv["z_jonction"])
        sz = AM.sigma_z(F * 1e3, r["ss"], nv["tf"], s["tw"])
        tau = o["V"] * 1e3 * nv["S"] / (I * s["tw"])
        out.update(sx=sx, sz=sz["sigma_z"], se=sz["se"], tau=tau, crit=AM.critere_elastique(sx, sz["sigma_z"], tau, fy, gM0))
        return out

    # ---- première passe : l'âme seule tient-elle, sous chaque combinaison ?
    etudes: Dict[str, Dict] = {}
    for L in charges:
        res = resistances(L, s, fy, P, fyw)
        pire = {"eta2": 0.0, "inter": None, "F": 0.0, "cote": None, "C": None, "crit": None, "V": 0.0}
        for o in occurrences[L["base"]]:
            essais = [(k, o[k]) for k in ("haut", "bas") if o[k] > 0 and k in res]
            if len(essais) == 2:
                essais.append(("b", min(o["haut"], o["bas"])))
            o["cote"], o["F"] = max(essais, key=lambda t: t[1] * 1e3 / res[t[0]]["FRd"])
            o["eta2"] = o["F"] * 1e3 / res[o["cote"]]["FRd"]
            o["b"] = dict(essais).get("b")
            o["Fmax"] = max(o["haut"], o["bas"])
            # § 7.2(2) : semelle tendue → pas d'interaction, le critère 6.2.1(5) ; « les deux » → les deux contrôles
            ctrl = [elastique(o, k, o[k], res[k]) for k in ("haut", "bas") if o[k] > 0 and k in res]
            o["el"] = max((x for x in ctrl if x["crit"] is not None), key=lambda x: x["crit"], default=None)
            o["etats"] = {x["cote"]: x["etat"] for x in ctrl}
            comprimee = any(x["etat"] != AM.TENDUE for x in ctrl)
            o["eta1"] = o["inter"] = o["eta1_pl"] = o["inter_pl"] = None
            if comprimee and (o["M"] > EPS or abs(o["N"]) > EPS):
                A, W, nomA, nomW = section_elastique(o["N"], o["M"])
                o.update({"A": A, "W": W, "nomA": nomA, "nomW": nomW})
                o["eta1"] = AM.eta1(o["N"] * 1e3, o["M"] * 1e6, A, W, fy, gM0)
                o["inter"] = AM.interaction(o["eta2"], o["eta1"])
                # DÉCISION BUREAU n° 1 (brief § 13.2) : la variante η₁ plastique, classes 1 et 2 (commentaire JRC, p. 98), pour information
                Wp = s.get("Wpl_y") if classe in (1, 2) and s.get("Wpl_y") else W
                o["eta1_pl"] = AM.eta1(o["N"] * 1e3, o["M"] * 1e6, A, Wp, fy, gM0)
                o["inter_pl"] = AM.interaction(o["eta2"], o["eta1_pl"])
                FRd_k = res[o["cote"]]["FRd"]
                o["seuil"] = (AM.seuil_local(FRd_k, o["eta1"]) / 1e3, AM.seuil_local(FRd_k, o["eta1_pl"]) / 1e3)
            if o["eta2"] > pire["eta2"]:
                pire.update({"eta2": o["eta2"], "cote": o["cote"], "C": o["C"], "FRd": res[o["cote"]]["FRd"], "Fo": o["F"]})
            if o["inter"] is not None and (pire["inter"] is None or o["inter"] > pire["inter"]):
                pire.update({"inter": o["inter"], "C_inter": o["C"], "somme": o["eta2"] + AM.INTERACTION_72[0] * o["eta1"],
                             "inter_pl": o["inter_pl"], "seuil": o["seuil"], "M_inter": o["M"]})
            if o["el"] is not None and (pire["crit"] is None or o["el"]["crit"] > pire["crit"]):
                pire.update({"crit": o["el"]["crit"], "C_crit": o["C"], "el": o["el"]})
            pire["F"] = max(pire["F"], o["Fmax"])
            pire["V"] = max(pire["V"], o["V"])
        raison = None
        if pire["eta2"] > 1.0:
            raison = (f"Sans raidisseur : F_Ed = {RE.fr(pire['Fo'], 1)} kN > F_Rd = {RE.fr(pire['FRd'] / 1e3, 1)} kN "
                      f"({pire['C'].nom}).")
        elif pire["inter"] is not None and pire["inter"] > 1.0:
            raison = (f"Sans raidisseur : η₂ + 0,8·η₁ = {RE.fr(pire['somme'], 2)} > 1,4 ({pire['C_inter'].nom}).")
        elif pire["crit"] is not None and pire["crit"] > 1.0:
            raison = (f"Sans raidisseur : critère 6.2.1(5) à la jonction = {RE.fr(pire['crit'], 2)} > 1, "
                      f"{AM.ETATS_SEMELLE[pire['el']['etat']]} ({pire['C_crit'].nom}).")
        elif el is not None and el["elancee"] and L["genre"] == "appui":
            raison = "Âme élancée au cisaillement : raidisseurs aux appuis (§ 5.1(2))."
        etudes[L["base"]] = {"res": res, "pire": pire, "raison": raison, "raidisseur": None}

    # ---- deux forces rapprochées sur la même semelle (§ 6.3(2)) : à vérifier aussi ensemble — ce bloc ne le fait pas, et le dit
    proches = []
    for L1, L2 in zip(charges, charges[1:]):
        r1, r2 = etudes[L1["base"]]["res"], etudes[L2["base"]]["res"]
        ecart = (L2["x"] - L1["x"]) * 1e3
        if any(ecart < (r1[k]["ly"] + r2[k]["ly"]) / 2.0 for k in ("haut", "bas") if k in r1 and k in r2):
            proches.append(f"{L1['nom'][:1].lower() + L1['nom'][1:]} et {L2['nom'][:1].lower() + L2['nom'][1:]}")
    if proches:
        J = RE.Journal(METHODE, NORME)
        J.verif("A-GR", "Forces concentrées rapprochées", f"{AM.REF} § 6.3(2)", "", None, None, "", [], statut=RE.NON_COUVERT,
                messages=[J.alerte("forces_rapprochees", "attention", f"Forces rapprochées ({' ; '.join(proches)}) : à vérifier aussi ENSEMBLE — "
                                                                         "leur somme, sur leur entraxe (§ 6.3(2)). Non vérifié ici.")])
        env.verser(J.rendre(), {"combinaison": combs[0].nom})

    # ---- les raidisseurs : chacun avec l'âme disponible jusqu'au bout de la barre, ou jusqu'à mi-chemin du voisin
    raidis = [L for L in charges if etudes[L["base"]]["raison"] and RA.gabarit(s) is not None]
    for i, L in enumerate(raidis):
        g, d = L["dispo"]
        if i > 0:
            g = min(g, (L["x"] - raidis[i - 1]["x"]) * 1e3 / 2.0)
        if i < len(raidis) - 1:
            d = min(d, (raidis[i + 1]["x"] - L["x"]) * 1e3 / 2.0)
        etude = etudes[L["base"]]
        rigide = el is not None and el["elancee"] and L["genre"] == "appui"
        # sous une charge, dans une âme élancée, le raidisseur sert aussi au cisaillement : N = F_Ed + N_st (§ 9.3.3(3) ; brief § 8.4)
        Nst, cis = 0.0, None
        if el is not None and el["elancee"] and L["genre"] == "charge":
            lw = AM.resistance_cisaillement(el, fy, P)["lambda_w"]
            cis = {"V": etude["pire"]["V"] * 1e3, "fyw": fy, "lambda_w": lw}
            Nst = RA.effort_cisaillement(cis["V"], el["hw"], el["tw"], fy, lw, P["gamma_M1"])
        etude["raidisseur"] = RA.dimensionner(s, fy, P, etude["pire"]["F"] * 1e3, (g, d), appui_rigide=rigide, nuance=b["nuance"], fy_ame=fyw,
                                              Nst=Nst)
        if etude["raidisseur"] is not None:
            etude["raidisseur"]["cisaillement"] = cis

    # ---- seconde passe : ce qui est vérifié, combinaison par combinaison
    for L in charges:
        etude = etudes[L["base"]]
        res, p = etude["res"], etude["raidisseur"]
        for o in occurrences[L["base"]]:
            J = RE.Journal(METHODE, NORME)
            r = res[o["cote"]]
            if p is not None:
                cote0 = etude["pire"]["cote"]
                avant = AM.ecrire_resistance(J, res[cote0]["ame"], fy, P, res[cote0])
                RA.ecrire(J, L["base"], L["nom"], s, fy, P, dict(p, F=o["Fmax"] * 1e3), avant,
                          [etude["raison"]] + _mots_force(res[cote0], _saisie(L, cote0)))
            else:
                cotes = [k for k in ("haut", "bas") if o[k] > 0 and k in res]
                k = max(cotes, key=lambda k: o[k] * 1e3 / res[k]["FRd"])
                mots = _mots_force(res[k], _saisie(L, k))
                impossible = bool(etude["raison"])               # il faut un raidisseur, et le profilé n'en reçoit pas
                if impossible:
                    mots.append(J.alerte(f"sans_raidisseur_{L['base']}", "attention", f"{L['nom']} : raidisseur nécessaire, mais le profilé "
                                                                                          "est trop étroit pour en recevoir un."))
                J.verif(f"{L['base']}-F", _titre(L, k), f"{AM.REF} § 6.2", "F_Ed ≤ F_Rd", o[k], res[k]["FRd"] / 1e3, "kN",
                        AM.ecrire_resistance(J, res[k]["ame"], fy, P, res[k]), messages=mots, statut=RE.NOK if impossible else None,
                        taux=o[k] * 1e3 / res[k]["FRd"] if impossible else None)
                if o["b"] is not None:
                    J.verif(f"{L['base']}-B", _titre(L, "b"), f"{AM.REF} § 6.2", "F_Ed ≤ F_Rd", o["b"], res["b"]["FRd"] / 1e3, "kN",
                            AM.ecrire_resistance(J, res["b"]["ame"], fy, P, res["b"], prefixe="b_"),
                            messages=_mots_force(res["b"], _saisie(L, "b")))
                if o["eta1"] is not None:
                    cles = [J.f("eta2", "η₂", [RE.terme("F_Ed", o["F"], "kN"), RE.terme("F_Rd", r["FRd"] / 1e3, "kN")], o["eta2"], "", "a_eta2"),
                            AM.ecrire_eta1(J, o["N"] * 1e3, o["M"] * 1e6, o["A"], o["W"], fy, gM0, o["nomA"], o["nomW"])]
                    cles.append(J.f("i72", "η", [RE.terme("η₂", o["eta2"]), RE.terme("η₁", o["eta1"])], o["inter"], "", "a_72"))
                    mots72 = [f"η₁ plastique (classes 1 et 2, variante) : {RE.fr(o['eta1_pl'], 3)}, taux {RE.fr(100.0 * o['inter_pl'], 1)} %."
                              if o["eta1_pl"] != o["eta1"] else "η₁ élastique (classe 3 ou 4 : pas de variante plastique).",
                              f"Seuil de résistance locale à M_Ed = {RE.fr(o['M'], 2)} kN·m imposé : {RE.fr(o['seuil'][0], 1)} kN "
                              f"(η₁ plastique : {RE.fr(o['seuil'][1], 1)} kN) — ce n'est pas la capacité de la barre."]
                    J.verif(f"{L['base']}-FM", f"{L['nom']} : force concentrée et flexion", f"{AM.REF} § 7.2", "(η₂ + 0,8·η₁)/1,4 ≤ 1",
                            o["inter"], 1.0, "", cles, taux=o["inter"], messages=mots72)
                if o["el"] is not None:
                    x = o["el"]
                    nv = x["nv"]
                    cles = [J.f("se", "s_e", [RE.terme("s_s", res[x["cote"]]["ss"], "mm"), RE.terme("t_f", nv["tf"], "mm")], x["se"], "mm", "a_se"),
                            J.f("sigma_z", "σ_z,Ed", [RE.terme("F_Ed", x["F"], "kN"), RE.terme("s_e", x["se"], "mm"), RE.terme("t_w", s["tw"], "mm")],
                                x["sz"], "N/mm²", "a_sigma_z"),
                            J.f("sigma_x", "σ_x", [RE.terme("N_Ed", o["N"], "kN"), RE.terme("A", x["A"], "mm²"), RE.terme("M_Ed", o["M_s"], "kN·m"),
                                                   RE.terme("z", nv["z_jonction"], "mm"), RE.terme("I_y", x["I"] / 1e4, "cm⁴")], x["sx"], "N/mm²", "a_sigma_x"),
                            J.f("tau_j", "τ_Ed", [RE.terme("V_Ed", o["V"], "kN"), RE.terme("S_f", nv["S"] / 1e3, "cm³"), RE.terme("I_y", x["I"] / 1e4, "cm⁴"),
                                                  RE.terme("t_w", s["tw"], "mm")], x["tau"], "N/mm²", "a_tau_j"),
                            J.f("c615", "critère", [RE.terme("σ_x", x["sx"], "N/mm²"), RE.terme("σ_z", x["sz"], "N/mm²"), RE.terme("τ", x["tau"], "N/mm²"),
                                                    RE.terme("f_y", fy, "N/mm²"), RE.terme("γ_M0", gM0)], x["crit"], "", "a_615")]
                    J.verif(f"{L['base']}-EL", f"{L['nom']} : semelle chargée tendue, critère élastique", "EN 1993-1-1 § 6.2.1(5) ; "
                            f"{AM.REF} § 7.2(2)", "(σ_x/f)² + (σ_z/f)² − (σ_x/f)·(σ_z/f) + 3·(τ/f)² ≤ 1", x["crit"], 1.0, "", cles, taux=x["crit"],
                            messages=[f"{AM.ETATS_SEMELLE[x['etat']].capitalize()} (σ_x au plan moyen) : à la jonction âme-semelle, σ_z,Ed de "
                                      "l'EN 1993-1-5 § 3.2.3 (compression positive)."
                                      + (" Les deux contrôles, le plus défavorable retenu." if x["etat"] == AM.LES_DEUX else "")])
            env.verser(J.rendre(), {"combinaison": o["C"].nom, "x": L["x"]})

    r = env.rendre()
    par_id = {v["id"]: v for v in r["verifications"]}
    rendus = []
    for L in charges:
        etude = etudes[L["base"]]
        pire, p = etude["pire"], etude["raidisseur"]
        vs = [v for i, v in par_id.items() if i.startswith(L["base"] + "-")]
        gouv = max(vs, key=lambda v: (v["statut"] == RE.NOK, v["taux"] or 0.0))
        r0 = etude["res"][pire["cote"]]
        x_el = pire.get("el")
        sans = {"F_Rd": r0["FRd"] / 1e3, "eta2": pire["eta2"], "interaction": pire["inter"], "interaction_plastique": pire.get("inter_pl"),
                "critere": pire["crit"], "seuil": None if pire.get("seuil") is None else
                {"M_Ed": pire["M_inter"], "elastique": pire["seuil"][0], "plastique": pire["seuil"][1]},
                # la compression locale de l'âme sous F_Ed, à la jonction (§ 3.2.3 ; brief § 15.4) : la semelle de la plus grande force
                "sigma_z": AM.sigma_z(pire["F"] * 1e3, r0["ss"], r0["ame"]["tf"], s["tw"])["sigma_z"],
                "semelle": sorted({e_ for o in occurrences[L["base"]] for e_ in o["etats"].values()}),
                "contact": r0.get("contact_etat") if r0["type"] == "c" else None}
        avec = None if p is None else RA.vue(p, pire["F"] * 1e3, p.get("Nst") or 0.0)
        rendus.append({"base": L["base"], "nom": L["nom"], "genre": L["genre"], "x": L["x"], "type": r0["type"], "ss": r0["ss"],
                       "F_Ed": pire["F"], "sans": sans, "avec": avec,
                       "rappel": RA.RAPPEL_TAU if p is not None else None,
                       "decision": "raidisseur" if p is not None else "aucun", "raison": etude["raison"],
                       "raidisseur": None if p is None else {"bs": p["bs"], "ts": p["ts"], "chanfrein": p["chanfrein"],
                                                             "designation": RA.designation(p)},
                       "verifications": [v["id"] for v in vs], "gouvernant": gouv["id"], "taux": gouv["taux"],
                       "statut": RE.NOK if any(v["statut"] == RE.NOK for v in vs) else RE.OK})
    if any(L["decision"] == "raidisseur" for L in rendus):
        hyp.append("Raidisseurs : de la nuance de la barre, f_y de chaque plat selon son épaisseur (table unique, option a de l'ANB) ; f_y de "
                   "calcul de la section du raidisseur = min(plats ; âme) ; leurs soudures sont hors du module (EN 1993-1-8).")
    connues = {x["code"] for x in r["alertes"]}
    r["alertes"] += [x for k, x in alertes.items() if k not in connues]
    r.update({"hypotheses": hyp, "lieux": rendus})
    return r
