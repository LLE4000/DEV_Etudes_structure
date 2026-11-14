# -*- coding: utf-8 -*-
"""Barres en acier (module « Choix de profilé ») — les RÉSULTATS : la colonne de droite et les fenêtres de calcul.

À droite : la tête (la barre, son profilé, son acier) ; un VERDICT par méthode (le taux le plus fort et ce qu'il
compare) ; le profilé retenu (il ouvre le sélecteur) ; un bloc par méthode — chaque vérification dit ce qu'elle
compare, où elle a été trouvée, et ouvre SA fenêtre de calcul — ; la comparaison des deux méthodes, sujet par sujet ;
les avertissements du moteur ; la masse et la flèche de service.

Les fenêtres de calcul : une par vérification (« h-H-SIGMA », « e-E-Mb »), « tout le calcul » de chaque méthode
(« s1 », « s2 » : le numéro de son bloc), la comparaison (« cmp ») et les paramètres à confirmer (« ac »). Chaque
formule y est écrite par ``acier/barre/ecriture`` — la même ligne que dans la note PDF.

Ce fichier MET EN FORME ce que le moteur a calculé et nommé : il ne calcule rien.
"""
from typing import Dict, List, Optional, Tuple

from acier.barre import ecriture as W
from acier.barre import resultat as RE

from . import commun as C
from . import etabli_compact as E
from . import profil_mots as M

ROLE = "profil"
N = C.NBSP
NOTE_COMPARAISON = ("La méthode historique compare des contraintes de service à σ_adm = f_y / s ; l'Eurocode 3 compare des efforts "
                    "pondérés à des résistances. Les deux taux se lisent côte à côte, ils ne se déduisent pas l'un de l'autre.")
NOTE_A_CONFIRMER = ("Ces valeurs changent un résultat et leur source n'est pas le texte de l'annexe nationale belge : elles sont à lire "
                    "sur le texte avant de s'y fier.")


def fenetre(methode: str, id_: str) -> str:
    """Le nom de la fenêtre de calcul d'une vérification."""
    return f"{M.PREFIXE[methode]}-{id_}"


def methodes(r: Dict) -> List[Tuple[int, str, Dict]]:
    """Les méthodes rendues, dans l'ordre de l'écran : (numéro de leur bloc, clé, résultat)."""
    return [(i + 1, k, x) for i, (k, x) in enumerate(r["methodes"].items())]


def comparable(r: Dict) -> bool:
    """Les deux méthodes sont-elles rendues, avec au moins un taux à mettre en regard ?"""
    return any(li["historique"] is not None or li["ec3"] is not None for li in r["comparaison"])


def gouvernante(x: Dict) -> Optional[Dict]:
    """La vérification qui gouverne une méthode ; sans gouvernant (rien n'est jugé), la première non couverte."""
    if x.get("gouvernant"):
        return next(v for v in x["verifications"] if v["id"] == x["gouvernant"])
    return next((v for v in x["verifications"] if v["statut"] == RE.NON_COUVERT), None)


def niveau_methode(x: Dict) -> str:
    """La couleur d'une méthode : rouge si elle refuse, celle de son taux sinon ; « v » (à vérifier) si une vérification
    n'est pas couverte ; « n » si elle n'a rien eu à vérifier."""
    if x["statut"] == RE.NOK:
        return "r"
    if x["statut"] == RE.NON_COUVERT:
        return "v"
    return C.niveau(x["taux_max"]) if x.get("taux_max") is not None else "n"


def mot_niveau(niv: str) -> str:
    """Ce qu'on dit d'un niveau : « Conforme », « Conforme, attention » (95 à 100 %), « Non conforme », « À vérifier »
    (une vérification non couverte), « Sans effort » — l'écran et la note PDF le disent avec les mêmes mots."""
    return {"r": "Non conforme", "v": "À vérifier", "n": "Sans effort"}.get(niv, C.STATUTS[niv][1] if niv in C.STATUTS else "Conforme")


def etat_methode(x: Dict) -> str:
    """« Conforme · 92 % », « Non conforme · 179 % », « À vérifier » (du texte, pour l'en-tête d'un bloc)."""
    niv = niveau_methode(x)
    mot = mot_niveau(niv)
    if x.get("taux_max") is None:
        return mot
    p = C.pct(x["taux_max"])
    return f"{mot} · " + (f">{N}{C.PLAFOND_TAUX}{N}%" if p > C.PLAFOND_TAUX else f"{C.pc(p)}{N}%")


# ------------------------------------------------------------------ la tête et le verdict
def dims(el: Dict) -> str:
    """« IPE 300 · S235 · 6,00 m » : ce qu'on lit d'une barre sous son nom (HTML)."""
    s = f"{C.esc(el['profil'])} · {C.esc(el['nuance'])}"
    if el["mode"] == "chargee":
        s += f" · {' + '.join(M.m(x) for x in el['travees'])}{N}m"
    return s


def verdicts(r: Dict) -> str:
    """Un verdict par méthode : la pastille de son taux le plus fort, la vérification qui gouverne et ce qu'elle compare ;
    il ouvre la fenêtre de calcul de cette vérification."""
    out = ""
    ms = methodes(r)
    for _, k, x in ms:
        g = gouvernante(x)
        if g is None:
            out += (f'<div class="verdict"><div class="gov"><b>{M.METHODES[k]}</b>aucun effort à vérifier</div></div>')
            continue
        if x.get("taux_max") is not None:
            pastille = C.chip(C.pct(x["taux_max"]), "r" if x["statut"] == RE.NOK else C.niveau(x["taux_max"]), xl=True)
        else:                                    # rien n'a de taux : un refus, ou une vérification non couverte
            pastille = f'<span class="chip xl {"r" if x["statut"] == RE.NOK else "c"} pf-mot">{M.MOT_STATUT[x["statut"]]}</span>'
        ou = M.lieu(g)
        suite = f'<br><span class="pf-ou">{C.esc(g["combinaison"])}{" · " + C.esc(ou) if ou else ""}</span>' if g.get("combinaison") else ""
        out += (f'<div class="verdict" data-calcul="{fenetre(k, g["id"])}" title="Voir le calcul">{pastille}<div class="gov">'
                f'<b>{M.METHODES[k]} · {C.esc(g["titre"][:1].lower() + g["titre"][1:])}</b>{M.comparaison(g)}{suite}</div></div>')
    return f'<div class="pf-verdicts n{len(ms)}">{out}</div>'


# ------------------------------------------------------------------ les blocs
def _ligne_profil(r: Dict) -> str:
    """La ligne « Profilé » : le bouton qui ouvre le sélecteur, et sa masse. Le plus léger de la famille qui tient se
    lit DANS le sélecteur (★) : la ligne le disait quand le sélecteur avait déjà vérifié la famille — le résultat d'une
    note dépendait alors de ce qui avait été ouvert avant, ce que le moteur partagé du site interdit (un écran rend ce
    que rend un moteur neuf ; trouvé par tests/test_moteur_partage.py, P1, le 14/11/2026)."""
    quoi = f"{M.n(r['masse'], 1)}{N}kg/m" if r.get("masse") else ""
    taux = [x["taux_max"] for x in r["methodes"].values() if x.get("taux_max") is not None]
    t = max(taux) if taux else None
    niv = "r" if r["statut"] == RE.NOK else (C.niveau(t) if t is not None else "c")
    return C.ligne_verif("Profilé", quoi, C.pct(t) if t is not None else None, niv, pk=ROLE, piece=C.esc(r["profil"]), etat_txt="—")


def _ligne(k: str, v: Dict, gov: bool) -> str:
    """Une vérification : son titre, ce qu'elle compare et où, son taux — ou son état en mots quand elle n'en a pas."""
    ou = M.lieu(v)
    cmp_ = M.comparaison(v) + (f' <span class="pf-ou">· {C.esc(ou)}</span>' if ou else "")
    taux = C.pct(v["taux"]) if v["taux"] is not None else None
    return C.ligne_verif(v["titre"], cmp_, taux, M.niveau(v), calcul=fenetre(k, v["id"]), gov=gov,
                         etat_txt=M.MOT_STATUT[v["statut"]].lower() if taux is None else "")


def _efforts_bloc(el: Dict, k: str, x: Dict) -> str:
    """La ligne sous l'en-tête d'un bloc : la norme, puis les charges que la méthode prend."""
    quoi = "charges de service" if k == "historique" else "combinaisons ELU (6.10)"
    if el["mode"] == "efforts":
        quoi = "efforts de service" if k == "historique" else "efforts de calcul"
    n_c = len(x.get("combinaisons") or [])
    return f"{C.esc(x['norme'])} · {quoi}" + (f" · {n_c} combinaison{'s' if n_c > 1 else ''}" if n_c else "")


def _comparaison(r: Dict) -> str:
    """Le tableau de la comparaison : un sujet par ligne, le taux de chaque méthode en pastille."""
    rows = '<div class="pf-cmp pf-cmp-h"><span></span><span>Historique</span><span>Eurocode 3</span></div>'
    for li in r["comparaison"]:
        cases = ""
        for k in ("historique", "ec3"):
            t = li[k]
            cases += (C.chip(C.pct(t), C.niveau(t)) if t is not None else '<span class="chip c">—</span>')
        fort = ' pf-cmp-t' if li["sujet"] == "Le plus défavorable" else ""
        rows += f'<div class="pf-cmp{fort}"><span>{C.esc(li["sujet"])}</span>{cases}</div>'
    return rows


def blocs(el: Dict, r: Dict) -> str:
    """Le profilé, puis un bloc par méthode et, quand les deux sont rendues, leur comparaison."""
    out = f'<div class="els pf-profil">{_ligne_profil(r)}</div>'
    ms = methodes(r)
    for i, k, x in ms:
        if not x["verifications"]:
            continue                                       # aucun effort : le verdict le dit, un bloc vide n'ajoute rien
        g = x.get("gouvernant")
        rows = "".join(_ligne(k, v, v["id"] == g) for v in x["verifications"])
        pied = "".join(C.message(M.sub(a["message"]), a.get("niveau") != "info") for a in x["alertes"])
        niv = niveau_methode(x)
        # (le titre COURT, celui du bouton de la rubrique « Méthodes » : « Méthode historique » se lisait « MÉTHODE HIS… »
        #  devant « Conforme, attention · 97 % » ou « Non conforme · 179 % » — trouvé par le harnais des barres, 14/11/2026)
        out += E.bloc(i, M.METHODES_COURT[k], M.CLASSE_BLOC.get(niv, "inc"), etat_methode(x), _efforts_bloc(el, k, x), rows,
                      f'<div class="pf-msgs">{pied}</div>' if pied else "", theme=True)
    if comparable(r):
        ecart = next((li for li in r["comparaison"] if li["sujet"] == "Le plus défavorable"), None)
        etat = ""
        if ecart and ecart["historique"] is not None and ecart["ec3"] is not None:
            etat = f"{C.pc(C.pct(ecart['historique']))}{N}% · {C.pc(C.pct(ecart['ec3']))}{N}%"
        out += E.bloc(len(ms) + 1, "Comparaison", "inc", etat, "le taux le plus fort de chaque méthode, sujet par sujet",
                      _comparaison(r), theme=True, calcul="cmp")
    return out


def pied(r: Dict) -> str:
    """Sous les blocs : la masse du profilé, la flèche de service (une information : elle n'est pas vérifiée)."""
    s = ""
    if r.get("masse"):
        s += f'<div class="mass"><span>Masse du profilé</span><b>{M.n(r["masse"], 1)}{N}kg/m</b></div>'
    fl = (r.get("statique") or {}).get("fleche")
    if fl and fl.get("valeur"):
        rapport = f" · L/{C.fr(fl['rapport'], 0)}" if fl.get("rapport") else ""
        s += (f'<div class="mass pf-fleche" title="Flèche de service la plus grande ({C.esc(fl["combinaison"])}, à {M.m(fl["x"])} m) : une '
              f'information, elle n\'est pas vérifiée"><span>Flèche de service</span><b>{C.fr(abs(fl["valeur"]), 1)}{N}mm{rapport}</b></div>')
    return f'<div class="pfoot">{s}<div class="legend">{C.legende()}</div></div>'


# ------------------------------------------------------------------ les fenêtres de calcul
def _fx(bloc: Dict, symboles=None) -> str:
    """Une formule : « symbole = formule = valeurs = résultat » (acier/barre/ecriture)."""
    parts, res = W.morceaux(bloc, symboles)
    h = f'<span class="s">{M.sub(parts[0])}</span>' + "".join(f'<span class="eq">=</span><span>{M.sub(p)}</span>' for p in parts[1:])
    return f'<div class="fx">{h}<span class="eq">=</span><span class="r">{C.esc(res)}</span></div>'


def interaction(v: Dict) -> bool:
    """La vérification compare-t-elle une somme sans unité à 1 (une interaction) ? Son taux EST alors sa dernière formule."""
    return not v["unite"] and v.get("limite") is not None and abs(v["limite"] - 1.0) < 1e-12


def formules_de(x: Dict, v: Dict, taux: bool = True) -> List[Tuple[Dict, Optional[List[str]]]]:
    """Les formules d'une vérification, dans l'ordre du moteur : [(bloc, symboles ou None)]. Son taux s'écrit avec les
    deux membres de son critère (« η = σ/σ_adm ») ; une interaction n'en a pas (sa dernière formule le donne) ; ``taux``
    faux : sans cette ligne (la note PDF, dont le verdict dit déjà les deux membres et le taux)."""
    par = {f["cle"]: f for f in x["calcul"]}
    out: List[Tuple[Dict, Optional[List[str]]]] = []
    for cle in v["formules"]:
        f = par[cle]
        if f["gabarit"] == "taux":
            g, _, d = (v.get("critere") or "").partition(" ≤ ")
            if taux and d and not interaction(v):
                out.append((f, [g, d]))
        else:
            out.append((f, None))
    return out


def _groupe(x: Dict, v: Dict) -> str:
    """Les formules d'une vérification (``formules_de``), ses messages, sous son titre."""
    lignes = "".join(_fx(f, syms) for f, syms in formules_de(x, v))
    msgs = "".join(f'<div class="pf-msg">{M.sub(t)}</div>' for t in v["messages"])
    if not lignes and not msgs:
        return ""
    ou = M.lieu(v)
    sous = " · ".join(t for t in (v.get("combinaison") or "", ou) if t)
    st = f'<span class="ck-comb">{C.esc(sous)}</span>' if sous else ""
    return f'<div class="ck-g"><div class="ck-t">{C.esc(v["titre"])}{st}</div>{lignes}{msgs}</div>'


def _ligne_tableau(v: Dict, gov: bool) -> Dict:
    crit = v.get("critere") or ""
    g, _, d = crit.partition(" ≤ ")
    if v["valeur"] is None:
        valeur, limite = "—", "—"
    elif interaction(v):
        valeur, limite = C.fr(v["valeur"], 3), "1"
    else:
        valeur = f"{M.sub(g)} = {M.r1(v['valeur'])}{N}{C.esc(v['unite'])}"
        limite = f"{M.sub(d)} = {M.r1(v['limite'])}{N}{C.esc(v['unite'])}"
    ligne = {"nom": C.esc(v["titre"]), "sous": C.esc(v["reference"]), "formule": M.sub(crit), "valeur": valeur, "limite": limite,
             "taux": C.pct(v["taux"]) if v["taux"] is not None else None, "niveau": M.niveau(v), "gov": gov}
    if v["taux"] is None:
        ligne["etat_txt"] = M.MOT_STATUT[v["statut"]]
    return ligne


def _hyps(el: Dict, r: Dict, x: Dict) -> List[str]:
    """Les hypothèses d'une méthode, en pastilles : celles que le moteur écrit."""
    return [f"<b>{C.esc(r['profil'])}</b> · {C.esc(el['nuance'])}"] + [M.sub(h) for h in x["hypotheses"]]


def _combinaisons(x: Dict) -> str:
    """Le bilan des combinaisons d'une méthode : chacune, la vérification qui y gouverne et son taux."""
    if len(x.get("combinaisons") or []) < 2:
        return ""
    titres = {v["id"]: v["titre"] for v in x["verifications"]}
    lignes = ""
    for c in x["combinaisons"]:
        niv = "r" if c["statut"] == RE.NOK else (C.niveau(c["taux"]) if c["taux"] is not None else "c")
        quoi = titres.get(c["gouvernant"], c["gouvernant"] or "—")
        lignes += (f'<div class="fx pf-comb"><span class="s">{C.esc(c["nom"])}</span><span>{C.esc(quoi[:1].lower() + quoi[1:])}</span>'
                   f'<span class="chip {niv}">{M.taux_txt(c["taux"])}</span></div>')
    return f'<div class="ck-g"><div class="ck-t">Combinaisons</div>{lignes}</div>'


def _fenetre_comparaison(el: Dict, r: Dict, nom: str) -> str:
    lignes = []
    for li in r["comparaison"]:
        h, e = li["historique"], li["ec3"]
        pire = max((t for t in (h, e) if t is not None), default=None)
        lignes.append({"nom": C.esc(li["sujet"]), "sous": "", "formule": "", "valeur": M.taux_txt(h), "limite": M.taux_txt(e),
                       "taux": C.pct(pire), "niveau": C.niveau(pire), "gov": li["sujet"] == "Le plus défavorable"})
    detail = f'<div class="ck-g"><div class="ck-t">Ce que les deux méthodes comparent</div><div class="pf-msg">{C.esc(NOTE_COMPARAISON)}</div></div>'
    return C.fenetre_calcul("cmp", f"{nom} · comparaison des deux méthodes", dims(el), [], lignes, detail,
                            colonnes=("Sujet", "", "Historique", "Eurocode 3", "Le plus fort"))


def _fenetre_a_confirmer(r: Dict, nom: str) -> str:
    """Les paramètres de l'annexe nationale dont la source reste à confirmer : leur valeur, d'où elle vient."""
    ac = (r["methodes"].get("ec3") or {}).get("a_confirmer") or []
    if not ac:
        return ""
    lignes = [{"nom": M.sub(p["nom"]), "sous": C.esc(p["source"]), "formule": "", "valeur": C.esc(valeur_param(p["valeur"])), "limite": "—",
               "niveau": "c", "etat_txt": "à confirmer"} for p in ac]
    detail = f'<div class="ck-g"><div class="ck-t">Pourquoi « à confirmer »</div><div class="pf-msg">{C.esc(NOTE_A_CONFIRMER)}</div></div>'
    return C.fenetre_calcul("ac", f"{nom} · paramètres à confirmer", "annexe nationale belge de l'Eurocode 3", [], lignes, detail,
                            colonnes=("Paramètre", "", "Valeur prise", "", "État"))


def valeur_param(v) -> str:
    """La valeur d'un paramètre de l'annexe nationale, écrite : un nombre, une liste, un nom."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return M.n(v, 3)
    if isinstance(v, (list, tuple)):
        return " ; ".join(M.n(x, 2) for x in v)
    return str(v).replace("_", " ")


def volets(el: Dict, r: Dict, nom: str) -> str:
    """Les fenêtres de calcul : une par vérification, « tout le calcul » de chaque méthode, la comparaison, les
    paramètres à confirmer."""
    out = ""
    sous = dims(el)
    for i, k, x in methodes(r):
        hyps = _hyps(el, r, x)
        g = x.get("gouvernant")
        tout = ""
        for v in x["verifications"]:
            grp = _groupe(x, v)
            tout += grp
            ou = " · ".join(t for t in (v.get("combinaison") or "", M.lieu(v)) if t)
            out += C.fenetre_calcul(fenetre(k, v["id"]), f"{nom} · {C.esc(v['titre'][:1].lower() + v['titre'][1:])}",
                                    f"{M.METHODES[k]} · {sous}" + (f" · {C.esc(ou)}" if ou else ""), hyps, [_ligne_tableau(v, True)], grp)
        lignes = [_ligne_tableau(v, v["id"] == g) for v in x["verifications"]]
        out += C.fenetre_calcul(f"s{i}", f"{nom} · {M.METHODES[k].lower()}", f"{C.esc(x['norme'])} · {sous} · tout le calcul", hyps, lignes,
                                tout + _combinaisons(x))
    if comparable(r):
        out += _fenetre_comparaison(el, r, nom)
    out += _fenetre_a_confirmer(r, nom)
    return out
