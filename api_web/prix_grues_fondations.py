# -*- coding: utf-8 -*-
"""La page « FONDATIONS » du module « Prix des grues » (cahier § 3) : une grille, un panneau de détail, rien de plus.

- à GAUCHE : les filtres (famille de grues : tour et montage rapide par défaut, mobiles, toutes ; démolition sans ou
  avec ; « toutes les grues », sinon seulement celles qui ont un prix) ;
- au CENTRE : la grille — une ligne par grue, une colonne par type de fondation (dalle sur sol, poutres sur sol, sur
  pieux) ; chaque case : le prix retenu en gros, dessous d'où il vient (« 3 chantiers · 2024 à 2026 », « ancien
  tableau », « corrigé ») et son écusson ; « — » sans prix. Le choix sans / avec démolition change toute la grille ;
- à DROITE : sans case choisie, les compteurs et les retours à vérifier ; une case choisie : son prix, sa source, son
  écusson, une petite bande du minimum au maximum (un point par chantier), les chantiers (une carte chacun) et les
  actions — valider ou écarter un retour (avec un motif), corriger le prix de la case, revenir au calcul automatique.

Rien n'est calculé ici : ``grues.prix.fondations`` rend chaque case, ``grues.prix.edition`` valide, écarte, corrige."""
import datetime as dt
from typing import Dict, List, Optional, Tuple

from grues.prix import base as B
from grues.prix import ecussons as E
from grues.prix import edition as ED
from grues.prix import fondations as F

from . import commun as C
from . import prix_grues_commun as PC

FAMILLES = (("tour", "Tour", "Grues tour et à montage rapide"), ("mobiles", "Mobiles", "Grues mobiles"), ("toutes", "Toutes", "Toutes les grues"))
LOGIQUES_FAMILLE = {"tour": ("T", "R"), "mobiles": ("MS", "MA"), "toutes": ("T", "R", "MS", "MA")}
TYPES_COURTS = {"Dalle sur sol": "Dalle sur sol", "Poutres sur sol": "Poutres", "Sur pieux": "Pieux"}


# ------------------------------------------------------------------ les saisies
def modifier(etat: Dict, base: Dict, ch: List[str], valeur, auj: dt.date) -> Optional[str]:
    vf = etat["vf"]
    ini = etat.get("initiales") or ""
    if ch[0] == "vf":
        k = ch[1] if len(ch) > 1 else ""
        if k == "famille":
            if valeur not in {f[0] for f in FAMILLES}:
                raise ValueError("famille inconnue")
            vf["famille"] = valeur
        elif k == "demolition":
            vf["demolition"] = valeur in (True, "avec", "true", "1")
        elif k == "toutes":
            vf["toutes"] = bool(C.lire_bool(valeur))
        elif k == "case":
            v = str(valeur or "")
            if v.count("|") >= 1 and F.type_fondation(v.split("|")[1]):
                vf["case"] = [v.split("|")[0], F.type_fondation(v.split("|")[1])]
            else:
                vf["case"] = None
        else:
            raise ValueError("champ inconnu")
        return None
    # « b/fond/… »
    k = ch[2] if len(ch) > 2 else ""
    if k == "retour" and len(ch) == 4:
        if valeur not in (F.VALIDE, F.ECARTE, F.A_VERIFIER):
            raise ValueError("statut inconnu")
        motif = ""
        r = next((x for x in F.retours(base) if x["id"] == ch[3]), None)
        if r and valeur == F.ECARTE:
            motif = (r["brut"].get("motif") or "").strip()
        return ED.statut_retour(base, ch[3], valeur, motif, ini, auj)
    if k == "motif" and len(ch) == 4:
        r = next((x for x in F.retours(base) if x["id"] == ch[3]), None)
        if not r:
            raise ValueError("retour inconnu")
        r["brut"]["motif"] = " ".join(str(valeur or "").split())[:200]
        return None
    if k == "corriger" and len(ch) == 3:
        if not vf.get("case"):
            raise ValueError("choisir une case de la grille")
        modele, t = vf["case"]
        return ED.corriger_fondation(base, modele, t, bool(vf.get("demolition")), valeur, "", ini, auj)
    if k == "motif_case" and len(ch) == 3:
        if not vf.get("case"):
            raise ValueError("choisir une case de la grille")
        modele, t = vf["case"]
        c = (base.get("fondations_corrections") or {}).get(F.cle_case(modele, t, bool(vf.get("demolition"))))
        if not c:
            raise ValueError("cette case n'est pas corrigée")
        c["motif"] = " ".join(str(valeur or "").split())[:200]
        return None
    raise ValueError("champ inconnu")


# ------------------------------------------------------------------ la grille
def modeles(base: Dict, famille: str, toutes: bool) -> List[str]:
    lgs = LOGIQUES_FAMILLE.get(famille, LOGIQUES_FAMILLE["toutes"])
    vus = []
    for g in B.grues(base):
        if B.logique(base, g.get("cas")) in lgs and not B.retiree(g) and g.get("modele") not in vus:
            vus.append(g.get("modele"))
    if not toutes:
        avec = set(F.grues_avec_prix(base))
        vus = [m for m in vus if m in avec]
    return vus


def _cellule(k: Dict, choisie: bool) -> str:
    v = f'<b>{C.esc(PC.euros(k["valeur"]))}</b>' if k["valeur"] is not None else "<b>—</b>"
    src = k["source"] or ("aucun prix" if k["valeur"] is None else "")
    att = f' · {k["a_verifier"]} à vérifier' if k["a_verifier"] and k["origine"] != "retours" else ""
    ec = PC.point(k["ecusson"]) if k["valeur"] is not None else ""
    return (f'<td class="pg-fc{" sel" if choisie else ""}{" nul" if k["valeur"] is None else ""}"><button type="button" data-agir="1" '
            f'data-chemin="vf/case" data-valeur="{C.esc(k["modele"] + "|" + k["type"])}" title="Voir les chantiers">{v}'
            f'<small>{ec}{C.esc(src + att)}</small></button></td>')


def grille(base: Dict, vf: Dict, auj: dt.date) -> str:
    ms = modeles(base, vf.get("famille") or "tour", bool(vf.get("toutes")))
    dem = bool(vf.get("demolition"))
    sel = vf.get("case")
    entete = "".join(f'<th class="n">{C.esc(TYPES_COURTS[t])}</th>' for t in F.TYPES)
    rows = ""
    for m in ms:
        cases = "".join(_cellule(F.case(base, m, t, dem, auj), sel == [m, t]) for t in F.TYPES)
        rows += f'<tr><th class="pg-gn"><span>{C.esc(m)}</span></th>{cases}</tr>'
    vide = "" if ms else '<tr><td colspan="4" class="pg-info">Aucune grue avec un prix de fondation dans cette famille.</td></tr>'
    titre = f"Prix des fondations · {'avec' if dem else 'sans'} démolition"
    return (f'<div class="pg-tab-w"><table class="pg-tab pg-grille" aria-label="{C.esc(titre)}"><thead><tr><th>Grue</th>{entete}</tr></thead>'
            f'<tbody>{rows}{vide}</tbody></table></div>')


# ------------------------------------------------------------------ à droite
def _kv(k: str, v: str) -> str:
    return f'<div class="pg-kv"><span>{C.esc(k)}</span><b>{v}</b></div>'


def bande(k: Dict) -> str:
    """La petite bande du minimum au maximum, un point par chantier validé (et en creux, ceux à vérifier)."""
    pts = [(c["cout_actualise"], c["statut"]) for c in k["chantiers"] if c["cout_actualise"] is not None]
    if not pts:
        return ""
    lo, hi = min(p for p, _ in pts), max(p for p, _ in pts)
    larg = 260.0

    def x(v):
        return 10 + (larg - 20) * ((v - lo) / (hi - lo) if hi > lo else 0.5)
    cercles = "".join(f'<circle cx="{x(v):.1f}" cy="14" r="4.5" class="{"v" if s == F.VALIDE else "a"}"/>' for v, s in pts)
    moy = f'<path d="M{x(k["valeur"]):.1f} 5v18" class="m"/>' if k.get("valeur") is not None and lo <= k["valeur"] <= hi else ""
    return (f'<div class="pg-bande"><svg viewBox="0 0 {larg:.0f} 28" role="img" aria-label="Du minimum au maximum">'
            f'<path d="M10 14H{larg - 10:.0f}" class="l"/>{moy}{cercles}</svg>'
            f'<div class="pg-bande-t"><span>min {C.esc(PC.euros(lo))}</span><span>max {C.esc(PC.euros(hi))}</span></div></div>')


def _carte(r: Dict, cote: str, auj: dt.date) -> str:
    st = r["statut"]
    cls = {F.VALIDE: "v", F.A_VERIFIER: "a", F.ECARTE: "e"}.get(st, "")
    boutons = ""
    if st != F.VALIDE:
        boutons += (f'<button type="button" class="btn sm" data-agir="1" data-chemin="b/fond/retour/{C.esc(r["id"])}" data-valeur="{F.VALIDE}">'
                    f'{C.icon("check")}Valider</button>')
    if st != F.ECARTE:
        boutons += (f'<button type="button" class="btn sm ghost" data-agir="1" data-chemin="b/fond/retour/{C.esc(r["id"])}" data-valeur="{F.ECARTE}">'
                    f'{C.icon("x")}Écarter</button>')
    if st != F.A_VERIFIER:
        boutons += (f'<button type="button" class="btn sm ghost" data-agir="1" data-chemin="b/fond/retour/{C.esc(r["id"])}" data-valeur="{F.A_VERIFIER}">'
                    f'{C.icon("undo-2")}À vérifier</button>')
    motif = (f'<input class="pg-in t" data-agir="1" data-chemin="b/fond/motif/{C.esc(r["id"])}" value="{C.esc(r["brut"].get("motif") or "")}" '
             f'placeholder="motif (écarter)" aria-label="Motif">')
    return (f'<div class="pg-carte {cls}"><div class="pg-carte-h"><b>{C.esc(r["chantier"] or r["id"])}</b>'
            f'<span class="pg-st {cls}">{C.esc(F.LIBELLES_STATUT.get(st, st))}</span></div>'
            + _kv("Année", C.esc(r["annee"] or "inconnue")) + _kv("Coût", C.esc(PC.euros(r[cote])))
            + (_kv("Repère", C.esc(r["repere"])) if r.get("repere") else "")
            + (_kv("Main-d'œuvre", "non chiffrée") if r.get("mo_non_chiffree") else "")
            + _kv("Origine", C.esc("ancien retour" if r["origine"] == "historique" else "retour du chantier"))
            + f'<div class="pg-fl-a">{boutons}</div>{motif}</div>')


def droite(base: Dict, vf: Dict, auj: dt.date) -> str:
    tous = F.retours(base)
    if not vf.get("case"):
        valides = sum(1 for r in tous if r["statut"] == F.VALIDE)
        att = [r for r in tous if r["statut"] == F.A_VERIFIER]
        corps = (_kv("Retours validés", str(valides)) + _kv("Retours à vérifier", str(len(att)))
                 + _kv("Demandes en cours", "0") + _kv("À relancer", "0"))
        liste = "".join(f'<li><button type="button" class="pg-lien" data-agir="1" data-chemin="vf/case" '
                        f'data-valeur="{C.esc((r["grue"] or "") + "|" + (r["type"] or ""))}">{C.esc(r["grue"])} · '
                        f'{C.esc(TYPES_COURTS.get(r["type"], r["type"] or ""))}</button> <small>{C.esc(r["chantier"] or r["id"])}</small></li>'
                        for r in att[:40])
        return (f'<div class="pbody"><div class="pg-res"><div class="pg-bloc"><div class="pg-bloc-h">Retours des chantiers</div>{corps}</div>'
                + (f'<div class="pg-liste att"><b>À vérifier ({len(att)})</b><ul>{liste}</ul></div>' if att else "")
                + '</div></div>')
    modele, t = vf["case"]
    dem = bool(vf.get("demolition"))
    k = F.case(base, modele, t, dem, auj)
    cote = "avec" if dem else "sans"
    lies = [r for r in tous if r["grue"] == modele and r["type"] == t and r[cote] is not None]
    lies.sort(key=lambda r: {F.VALIDE: 0, F.A_VERIFIER: 1, F.ECARTE: 2}.get(r["statut"], 3))
    corr = k.get("correction") or {}
    v_corr = PC.nombre(corr.get("valeur"), 2) if corr else ""
    correction = (C.ligne("Corriger", f'<div class="field"><input class="pg-in" data-agir="1" data-chemin="b/fond/corriger" inputmode="decimal" '
                                      f'value="{C.esc(v_corr)}" placeholder="montant"><span class="u">€</span></div>',
                          aide="Notre prix pour cette case : il passe avant les chantiers et le tableau ; vide : retour au calcul automatique"))
    if corr:
        correction += C.ligne("Motif", f'<div class="field text"><input class="pg-in t" data-agir="1" data-chemin="b/fond/motif_case" '
                                       f'value="{C.esc(corr.get("motif") or "")}" placeholder="pourquoi ?"></div>')
        correction += (f'<button type="button" class="btn sm ghost" data-agir="1" data-chemin="b/fond/corriger" data-valeur="">'
                       f'{C.icon("undo-2")}Revenir au calcul automatique</button>')
    stats = ""
    if k["n"]:
        stats = (_kv("Chantiers", str(k["n"])) + _kv("Moyenne", C.esc(PC.euros(k["moyenne"])))
                 + _kv("Minimum", C.esc(PC.euros(k["min"]))) + _kv("Maximum", C.esc(PC.euros(k["max"]))))
    tab = _kv("Ancien tableau", C.esc(PC.euros(k["tableau"])) if k["tableau"] is not None else "—")
    entete = (f'<div class="pg-bloc sel"><div class="pg-bloc-h">{C.esc(modele)}<small>{C.esc(t)} · {"avec" if dem else "sans"} démolition</small></div>'
              f'<div class="pg-total sm">{C.esc(PC.euros(k["valeur"]))}</div>'
              f'<div class="pg-ec-l">{PC.ecusson(k["ecusson"])}<span>{C.esc(k["source"] or "aucun prix")}'
              f'{" · actualisé" if k["actualise"] else ""}{" · année inconnue" if k["annee_inconnue"] else ""}</span></div>'
              f'{bande(k)}{stats}{tab}{correction}</div>')
    cartes = "".join(_carte(r, cote, auj) for r in lies) or '<p class="pg-info">Aucun retour de chantier pour cette case.</p>'
    fermer = (f'<button type="button" class="btn sm ghost" data-agir="1" data-chemin="vf/case" data-valeur="">{C.icon("x")}Fermer</button>')
    return f'<div class="pbody"><div class="pg-res">{entete}<div class="pg-sous">Chantiers</div>{cartes}<div class="pg-fl-a">{fermer}</div></div></div>'


def gauche(base: Dict, vf: Dict) -> str:
    corps = (C.ligne("Grues", C.segments(["vf", "famille"], vf.get("famille") or "tour", FAMILLES))
             + C.ligne("Démolition", C.segments(["vf", "demolition"], "avec" if vf.get("demolition") else "sans", (("sans", "Sans"), ("avec", "Avec"))))
             + C.ligne("", C.case(["vf", "toutes"], bool(vf.get("toutes")), "toutes les grues (même sans prix)")))
    fam = {f[0]: f[2] for f in FAMILLES}[vf.get("famille") or "tour"]
    out = C.section(1, "Filtres", C.esc(f"{fam.lower()} · {'avec' if vf.get('demolition') else 'sans'} démolition"), corps, True, "filtres")
    dem = ('<div class="pg-liste-bt">'
           '<button type="button" class="pg-bt" disabled title="Bientôt : le fichier Excel envoyé au chantier"><span class="nm">Nouvelle demande</span><span class="pc">bientôt</span></button>'
           '<button type="button" class="pg-bt" disabled title="Bientôt : le fichier rempli par le chantier"><span class="nm">Importer un fichier reçu</span><span class="pc">bientôt</span></button>'
           '</div>')
    out += C.section(2, "Demandes au chantier", "bientôt", dem, False, "demandes")
    return out


def page(etat: Dict, base: Dict, auj: dt.date):
    vf = etat["vf"]
    dem = bool(vf.get("demolition"))
    vues = [{"cle": "grille", "titre": f"Prix des fondations · {'avec' if dem else 'sans'} démolition", "court": "Grille", "echelle": False,
             "svg": f'<div class="pg-detail">{grille(base, vf, auj)}</div>'}]
    return gauche(base, vf), vues, droite(base, vf, auj), ""
