# -*- coding: utf-8 -*-
"""L'ÉCHANGE ANNUEL avec un loueur, à l'écran (cahier § 7.1, § 7.2) :

- sur le dossier du loueur (page « Base des prix », à droite) : « Préparer le fichier {année} » (l'année, « Demandé
  par », « Retour souhaité avant le »), « Écrire le mail » (un mail prêt ; le fichier se joint à la main), « Le fichier
  est parti » (le dossier passe en « Demande envoyée »), « Importer le fichier reçu », « Annuler le dernier import », les
  remarques reçues (« marquer comme traité ») ;
- la page de CONTRÔLE d'un fichier reçu (page « import ») : à gauche le fichier, ses informations et les filtres par type
  de changement ; au centre le tableau des changements (objet, champ, avant, après, écart, la case « appliquer » ; un
  écart au-delà du seuil ou une baisse surligné ; les cases vidées décochées ; les valeurs non lues à saisir ; les
  nouvelles grues, les grues retirées, les autres prestations) ; à droite les compteurs, les avertissements,
  « Appliquer » et « Abandonner ».

Le fichier reçu reste dans la NOTE (``etat["imp"]``) tant que le contrôle dure : la lecture est refaite contre la base à
chaque écran (``grues.prix.lecture_loueur``), les cases cochées s'y rattachent par une clé stable. Rien n'est calculé ni
lu ici : le moteur est ``grues.prix`` (fichier_loueur, lecture_loueur, import_loueur)."""
import base64
import binascii
import datetime as dt
import hashlib
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

from grues.prix import base as B
from grues.prix import fichier_loueur as FL
from grues.prix import import_loueur as IL
from grues.prix import lecture_loueur as LL
from grues.prix import valeurs as V

from . import commun as C
from . import prix_grues_commun as PC

MIME_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
FICHIER_MAX = 2 * 1024 * 1024
ID_IMPORT = "pg-import-loueur"
FILTRES = (("tout", "Tout"), ("changements", "Changements"), ("cases_videes", "Cases vidées"), ("valeurs_non_lues", "Non lues"),
           ("grues", "Grues"), ("autres_prestations", "Autres prestations"), ("messages", "Réponses"), ("refus", "Refus"))
NOMS = {"changements": "Changements", "cases_videes": "Cases vidées", "valeurs_non_lues": "Valeurs non lues",
        "nouvelles_grues": "Nouvelles grues", "grues_retirees": "Grues plus proposées", "autres_prestations": "Autres prestations",
        "reponses": "Réponses aux questions", "remarques": "Remarques", "refus": "Refusé"}


def jeton(sorte: str, it: Dict) -> str:
    """La clé d'une ligne dans un chemin de l'écran (sans « / ») : l'empreinte de sa clé stable."""
    return hashlib.sha1(IL.cle_item(sorte, it).encode("utf-8")).hexdigest()[:12]


def _date_fr(iso) -> str:
    from grues.prix.periode import lire_date
    d = lire_date(iso) if iso else None
    return d.strftime("%d/%m/%Y") if d else "—"


# ------------------------------------------------------------------ l'état
def prep_de(etat: Dict, auj: dt.date) -> Dict:
    """« Préparer le fichier » : l'année (par défaut l'année suivante à partir de novembre), « Demandé par », « Retour
    souhaité avant le » — ce que la note a retenu, complété des valeurs par défaut (sans rien écrire dans la note)."""
    p = {"annee": FL.annee_par_defaut(auj), "par": "", "retour": None}
    q = etat["vb"].get("prep")
    if isinstance(q, dict):
        p.update({k: q[k] for k in p if q.get(k) is not None})
    return p


def lire_prep(q) -> Optional[Dict]:
    """Le « Préparer le fichier » d'une note relue (ou None)."""
    if not isinstance(q, dict):
        return None
    out = {}
    try:
        if q.get("annee") is not None and 2000 <= int(q["annee"]) <= 2100:
            out["annee"] = int(q["annee"])
    except (TypeError, ValueError):
        pass
    out["par"] = " ".join(str(q.get("par") or "").split())[:60]
    r = q.get("retour")
    out["retour"] = r if isinstance(r, str) and len(r) == 10 else None
    return out


def lire_imp(imp) -> Optional[Dict]:
    """Le fichier en cours de contrôle d'une note relue (ou None)."""
    if not (isinstance(imp, dict) and isinstance(imp.get("base64"), str) and isinstance(imp.get("nom"), str)):
        return None
    ch, sa = imp.get("choix"), imp.get("saisies")
    return {"nom": imp["nom"][:120], "base64": imp["base64"],
            "choix": {str(k): bool(v) for k, v in ch.items()} if isinstance(ch, dict) else {},
            "saisies": {str(k): str(v)[:40] for k, v in sa.items()} if isinstance(sa, dict) else {},
            "filtre": imp.get("filtre") if imp.get("filtre") in dict(FILTRES) else "tout",
            "page_avant": imp.get("page_avant") if imp.get("page_avant") in ("calcul", "fondations", "base") else "base"}


def lecture(etat: Dict, base: Dict) -> Optional[Dict]:
    imp = etat.get("imp")
    if not imp:
        return None
    return LL.lire(base, base64.b64decode(imp["base64"]), imp["nom"])


# ------------------------------------------------------------------ les saisies
def _lire_fichier(valeur) -> Tuple[str, bytes]:
    if not isinstance(valeur, dict) or not isinstance(valeur.get("base64"), str):
        raise ValueError("fichier illisible")
    nom = str(valeur.get("nom") or "fichier.xlsx")[:120]
    try:
        o = base64.b64decode(valeur["base64"], validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("fichier illisible")
    if len(o) > FICHIER_MAX:
        raise ValueError("fichier trop gros (2 Mo au plus)")
    return nom, o


def modifier(etat: Dict, base: Dict, ch: List[str], valeur, auj: dt.date) -> Optional[str]:
    ini = etat.get("initiales") or ""
    vb = etat["vb"]
    if ch[:2] == ["vb", "prep"] and len(ch) == 3:
        p = prep_de(etat, auj)
        vb["prep"] = p
        if ch[2] == "annee":
            x = V.lire_nombre(str(valeur)) if valeur is not None else None
            if x is None or not 2000 <= x <= 2100:
                raise ValueError("une année (2027…)")
            p["annee"] = int(x)
        elif ch[2] == "par":
            p["par"] = " ".join(str(valeur or "").split())[:60]
        elif ch[2] == "retour":
            from grues.prix.periode import lire_date
            t = str(valeur or "").strip()
            d = lire_date(t) if t else None
            if t and not d:
                raise ValueError("une date (jj/mm/aaaa)")
            p["retour"] = d.isoformat() if d else None
        else:
            raise ValueError("champ inconnu")
        return None
    if ch[:2] == ["b", "demande"] and len(ch) == 3:
        nom = vb.get("loueur")
        if not nom:
            raise ValueError("choisir un loueur")
        if ch[2] == "envoyee":
            p = prep_de(etat, auj)
            return IL.marquer_demande(base, nom, p["annee"], p.get("par") or "", p.get("retour"), ini, auj)
        if ch[2] == "retirer":
            return IL.annuler_demande(base, nom, ini, auj)
        raise ValueError("champ inconnu")
    if ch == ["b", "import", "annuler"]:
        return IL.annuler_dernier(base, ini, auj)
    if ch[:2] == ["b", "message"] and len(ch) == 4 and ch[3] == "traite":
        nom = vb.get("loueur")
        msgs = (((base.get("dossiers") or {}).get(nom) or {}).get("messages")) or []
        try:
            m = msgs[int(ch[2])]
        except (ValueError, IndexError):
            raise ValueError("message inconnu")
        m["traitee"] = bool(C.lire_bool(valeur))
        return None
    if ch[0] != "imp":
        raise ValueError("champ inconnu")
    k = ch[1] if len(ch) > 1 else ""
    if k == "fichier":
        nom, o = _lire_fichier(valeur)
        try:
            lu = LL.lire(base, o, nom)
        except LL.FichierRefuse as e:
            raise ValueError(str(e))
        etat["imp"] = {"nom": nom, "base64": base64.b64encode(o).decode("ascii"), "choix": {}, "saisies": {}, "filtre": "tout",
                       "page_avant": etat.get("page") if etat.get("page") != "import" else "base"}
        etat["page"] = "import"
        vb["loueur"] = lu["loueur"]
        r = lu["resume"]
        return (f"Fichier {lu['loueur']} {lu['annee']} lu : {r['changements']} changement(s), {r['prix_confirmes']} prix confirmé(s)"
                + (f", {r['refus']} refusé(s)" if r["refus"] else "") + " — rien n'est appliqué avant « Appliquer »")
    imp = etat.get("imp")
    if not imp:
        raise ValueError("aucun fichier en cours de contrôle")
    if k == "filtre":
        if valeur not in dict(FILTRES):
            raise ValueError("filtre inconnu")
        imp["filtre"] = valeur
        return None
    if k == "choix" and len(ch) == 3:
        imp["choix"][ch[2]] = bool(C.lire_bool(valeur))
        return None
    if k == "saisie" and len(ch) == 3:
        t = " ".join(str(valeur or "").split())[:40]
        if t:
            from grues.prix import edition as ED
            ED.lire_valeur(t)       # refusée ici si ce n'est pas un prix
            imp["saisies"][ch[2]] = t
        else:
            imp["saisies"].pop(ch[2], None)
        return None
    if k == "abandonner":
        etat["page"] = imp.get("page_avant") or "base"
        etat["imp"] = None
        return "Contrôle abandonné : rien n'a été appliqué"
    if k == "appliquer":
        lu = lecture(etat, base)
        choix, saisies = {}, {}
        for sorte in IL.SORTES:
            for it in lu.get(sorte) or []:
                j = jeton(sorte, it)
                if j in imp["choix"]:
                    choix[IL.cle_item(sorte, it)] = imp["choix"][j]
                if sorte == "valeurs_non_lues" and j in imp["saisies"]:
                    saisies[IL.cle_item(sorte, it)] = imp["saisies"][j]
        n = IL.appliquer(base, lu, choix, saisies, auj, ini, auj)
        etat["page"] = "base"
        etat["imp"] = None
        vb.update(loueur=lu["loueur"], case=None)
        return n["message"]
    raise ValueError("champ inconnu")


# ------------------------------------------------------------------ le dossier du loueur (page « Base des prix »)
def _kv(k: str, v: str) -> str:
    return f'<div class="pg-kv"><span>{C.esc(k)}</span><b>{v}</b></div>'


def _champ(chemin: str, valeur, indice: str = "", mode: str = "text") -> str:
    return (f'<div class="field text"><input class="pg-in t" data-agir="1" data-chemin="{C.esc(chemin)}" value="{C.esc(valeur or "")}" '
            f'placeholder="{C.esc(indice)}" inputmode="{mode}"></div>')


def mail(base: Dict, nom: str, annee: int, retour: Optional[str]) -> str:
    """Le lien « mailto: » du mail prêt (§ 7.1) : l'objet et le texte du cahier ; le fichier se joint à la main."""
    lo = B.loueur(base, nom) or {}
    a = ((lo.get("contact") or {}).get("mail") or "") if isinstance(lo.get("contact"), dict) else ""
    quand = _date_fr(retour) if retour else "dès que possible"
    objet = f"Mise à jour de vos prix {annee}"
    corps = (f"Bonjour, vous trouverez en pièce jointe notre fichier de prix {annee}. Il reprend vos grues et vos prestations avec "
             f"les prix que nous avons aujourd'hui. Pouvez-vous corriger ce qui a changé, compléter les cases orange et nous le "
             f"renvoyer avant le {quand} ? Merci d'avance.")
    return f"mailto:{quote(a)}?subject={quote(objet)}&body={quote(corps)}"


def bloc_dossier(etat: Dict, base: Dict, nom: str, auj: dt.date) -> str:
    """L'échange avec le loueur, sous son dossier : où en est la demande, préparer, écrire, importer, annuler."""
    p = prep_de(etat, auj)
    dos = (base.get("dossiers") or {}).get(nom) or {}
    dem, der = dos.get("demande"), dos.get("dernier_fichier")
    etat_ = ""
    if dem and not dem.get("reponse_importee"):
        etat_ += _kv("Demande envoyée", C.esc(f"le {_date_fr(dem.get('envoyee_le'))} · prix {dem.get('annee')}")
                     + (f"<small> · retour avant le {C.esc(_date_fr(dem.get('retour_avant')))}</small>" if dem.get("retour_avant") else ""))
    etat_ += _kv("Dernier fichier reçu", C.esc(f"{_date_fr(der.get('recu_le'))} · prix {der.get('annee')} · {der.get('changements')} changement(s)")
                 if der else "—")
    an = p["annee"]
    prep = (C.ligne("Année", _champ("vb/prep/annee", str(an), "2027", "numeric"))
            + C.ligne("Demandé par", _champ("vb/prep/par", p.get("par"), "nom ou initiales"))
            + C.ligne("Retour avant le", _champ("vb/prep/retour", _date_fr(p.get("retour")) if p.get("retour") else "", "jj/mm/aaaa")))
    act = (f'<button type="button" class="btn sm primary" data-telecharger="fichier_loueur">{C.icon("download")}'
           f'Préparer le fichier {C.esc(an)}</button>'
           f'<a class="btn sm" href="{C.esc(mail(base, nom, an, p.get("retour")))}">{C.icon("mail")}Écrire le mail</a>')
    if dem and not dem.get("reponse_importee"):
        act += (f'<button type="button" class="btn sm ghost" data-agir="1" data-chemin="b/demande/retirer" data-valeur="1">'
                f'{C.icon("undo-2")}Retirer la demande</button>')
    else:
        act += (f'<button type="button" class="btn sm ghost" data-agir="1" data-chemin="b/demande/envoyee" data-valeur="1">'
                f'{C.icon("check")}Le fichier est parti</button>')
    imp = (f'<label class="btn sm fichier pg-import">{C.icon("upload")}Importer le fichier reçu'
           f'<input type="file" id="{ID_IMPORT}" data-agir="1" data-binaire="1" data-chemin="imp/fichier" '
           f'accept=".xlsx,{MIME_XLSX}" hidden></label>')
    if etat.get("imp"):
        imp = (f'<button type="button" class="btn sm primary" data-agir="1" data-chemin="page" data-valeur="import">'
               f'{C.icon("list-filter")}Reprendre le contrôle du fichier</button>') + imp
    ann = base.get("import_annulable")
    if ann and ann.get("loueur") == nom:
        imp += (f'<button type="button" class="btn sm ghost" data-agir="1" data-chemin="b/import/annuler" data-valeur="1">'
                f'{C.icon("undo-2")}Annuler le dernier import</button>')
    msgs = ""
    for i, m in enumerate(dos.get("messages") or []):
        if m.get("traitee"):
            continue
        msgs += (f'<div class="pg-msg"><div><small>{C.esc(_date_fr(m.get("date")))} · {C.esc(m.get("objet") or m.get("id") or "")}</small>'
                 f'<p>{C.esc(m.get("texte"))}</p></div><button type="button" class="btn xs ghost" data-agir="1" '
                 f'data-chemin="b/message/{i}/traite" data-valeur="1">{C.icon("check")}Traité</button></div>')
    msgs = f'<div class="pg-sous">Remarques reçues</div>{msgs}' if msgs else ""
    return (f'<div class="pg-bloc pg-echange"><div class="pg-bloc-h">Échange {C.esc(an)}</div>{etat_}'
            f'<div class="pg-sous">Préparer le fichier</div>{prep}<div class="pg-fl-a">{act}</div>'
            f'<div class="pg-sous">Réponse du loueur</div><div class="pg-fl-a">{imp}</div>{msgs}</div>')


def telecharger_fichier(etat: Dict, base: Dict, auj: dt.date) -> Dict:
    nom = etat["vb"].get("loueur")
    if not nom or not B.loueur(base, nom):
        raise ValueError("choisir un loueur")
    p = prep_de(etat, auj)
    n, o = FL.preparer(base, nom, p["annee"], p.get("par") or None, p.get("retour"), auj)
    return {"nom": n, "type": MIME_XLSX, "base64": base64.b64encode(o).decode("ascii")}


# ------------------------------------------------------------------ la page de contrôle
def _texte(v) -> str:
    if v is None:
        return "vide"
    if V.est_nombre(v):
        return "compris" if v == 0 else PC.nombre(v, 3)
    return str(v)


def _champ_lib(base: Dict, it: Dict) -> str:
    if it.get("type") == "grue":
        g = B.grue(base, it.get("id")) or {}
        ch = B.champ(base, B.logique(base, g.get("cas")) or "", it.get("cle")) or {}
        return (ch.get("libelle") or it.get("cle")) + (f" ({ch['unite']})" if ch.get("unite") and ch.get("unite") != "Oui" else "")
    if it.get("cle") == "unite":
        return "Unité"
    cat = B.catalogue(base, it.get("cas") or "") or {}
    return f"Prix · {FL.TYPE_COURT.get(cat.get('type'), cat.get('type') or 'tous')}"


def _objet(it: Dict) -> str:
    return it.get("modele") or it.get("poste") or it.get("id") or ""


def _corrige(base: Dict, it: Dict) -> bool:
    if it.get("type") == "grue":
        g = B.grue(base, it.get("id")) or {}
        return it.get("cle") in (g.get("corrections") or {})
    p = B.poste(base, str(it.get("id") or "")[2:]) or {}
    return (it.get("cas") or "") in (p.get("corrections") or {})


def _coche(imp: Dict, sorte: str, it: Dict) -> Tuple[str, bool]:
    j = jeton(sorte, it)
    v = imp["choix"].get(j)
    return j, IL.PAR_DEFAUT[sorte] if v is None else v


def _case_appliquer(j: str, on: bool, texte: str) -> str:
    return (f'<label class="case pg-ap"><input type="checkbox" data-agir="1" data-chemin="imp/choix/{j}"{" checked" if on else ""}>'
            f'<span>{C.esc(texte)}</span></label>')


def _tete(titre: str, n: int) -> str:
    return f'<tr class="pg-grp"><td colspan="6">{C.esc(titre)} <small>{n}</small></td></tr>'


def centre(base: Dict, lu: Dict, imp: Dict, seuil: float) -> str:
    f = imp.get("filtre") or "tout"
    voir = (lambda *s: f == "tout" or f in s)
    rows = ""
    if voir("changements") and lu["changements"]:
        rows += _tete(NOMS["changements"], len(lu["changements"]))
        for it in lu["changements"]:
            j, on = _coche(imp, "changements", it)
            fort = IL.a_surligner(it, seuil)
            ec = it.get("ecart_pct")
            note = '<small class="pg-av">corrigé par nous : la valeur du loueur passe dessous, à revoir</small>' if _corrige(base, it) else ""
            rows += (f'<tr class="{"fort" if fort else ""}"><td class="ob">{C.esc(_objet(it))}{note}</td><td>{C.esc(_champ_lib(base, it))}</td>'
                     f'<td class="n">{C.esc(_texte(it.get("avant")))}</td><td class="n ap">{C.esc(_texte(it.get("apres")))}</td>'
                     f'<td class="n">{C.esc((("+" if ec > 0 else "") + PC.nombre(ec, 1) + " %") if ec is not None else ("nouveau" if it.get("avant") is None else "—"))}</td>'
                     f'<td>{_case_appliquer(j, on, "appliquer")}</td></tr>')
    if voir("cases_videes") and lu["cases_videes"]:
        rows += _tete(NOMS["cases_videes"], len(lu["cases_videes"]))
        for it in lu["cases_videes"]:
            j, on = _coche(imp, "cases_videes", it)
            rows += (f'<tr class="vide-l"><td class="ob">{C.esc(_objet(it))}</td><td>{C.esc(_champ_lib(base, it))}</td>'
                     f'<td class="n">{C.esc(_texte(it.get("avant")))}</td><td class="n ap">vidée</td><td class="n">—</td>'
                     f'<td>{_case_appliquer(j, on, "vider")}</td></tr>')
    if voir("valeurs_non_lues") and lu["valeurs_non_lues"]:
        rows += _tete(NOMS["valeurs_non_lues"], len(lu["valeurs_non_lues"]))
        for it in lu["valeurs_non_lues"]:
            j = jeton("valeurs_non_lues", it)
            g = B.grue(base, it.get("id")) if it.get("type") == "grue" else None
            avant = g.get(it.get("cle")) if g else None
            rows += (f'<tr class="nonlu"><td class="ob">{C.esc(_objet(it))}</td><td>{C.esc(_champ_lib(base, it))}</td>'
                     f'<td class="n">{C.esc(_texte(avant)) if g else "—"}</td><td class="n ap">« {C.esc(it.get("texte"))} »</td>'
                     f'<td colspan="2"><div class="field text"><input class="pg-in" data-agir="1" data-chemin="imp/saisie/{j}" '
                     f'value="{C.esc(imp["saisies"].get(j) or "")}" placeholder="valeur à saisir" inputmode="decimal" '
                     f'aria-label="Valeur à saisir"></div></td></tr>')
    if voir("grues") and (lu["nouvelles_grues"] or lu["grues_retirees"]):
        rows += _tete("Grues", len(lu["nouvelles_grues"]) + len(lu["grues_retirees"]))
        for it in lu["nouvelles_grues"]:
            j, on = _coche(imp, "nouvelles_grues", it)
            vals = " · ".join(f"{k} {_texte(v)}" for k, v in (it.get("valeurs") or {}).items() if k != "modele")
            rows += (f'<tr class="neuf"><td class="ob">{C.esc((it.get("valeurs") or {}).get("modele") or "sans nom")}<small>{C.esc(vals)}</small></td>'
                     f'<td>Nouvelle grue · {C.esc(FL.TYPE_COURT.get((B.catalogue(base, it["cas"]) or {}).get("type"), it["cas"]))}</td>'
                     f'<td class="n">—</td><td class="n ap">nouvelle</td><td class="n">—</td><td>{_case_appliquer(j, on, "ajouter au catalogue")}</td></tr>')
        for it in lu["grues_retirees"]:
            j, on = _coche(imp, "grues_retirees", it)
            rows += (f'<tr><td class="ob">{C.esc(it.get("modele"))}</td><td>Plus proposée</td><td class="n">proposée</td>'
                     f'<td class="n ap">retirée</td><td class="n">—</td><td>{_case_appliquer(j, on, "retirer")}</td></tr>')
    if voir("autres_prestations") and lu["autres_prestations"]:
        rows += _tete(NOMS["autres_prestations"], len(lu["autres_prestations"]))
        for it in lu["autres_prestations"]:
            j, on = _coche(imp, "autres_prestations", it)
            prix = _texte(it.get("prix")) if it.get("prix") is not None else (f"« {it.get('prix_non_lu')} »" if it.get("prix_non_lu") else "vide")
            rows += (f'<tr><td class="ob">{C.esc(it.get("libelle") or "sans libellé")}<small>{C.esc(it.get("remarque") or "")}</small></td>'
                     f'<td>{C.esc((it.get("type") or "toutes") + " · " + (it.get("unite") or "unité ?"))}</td><td class="n">—</td>'
                     f'<td class="n ap">{C.esc(prix)}</td><td class="n">—</td><td>{_case_appliquer(j, on, "créer un poste")}</td></tr>')
    if voir("messages") and (lu["reponses"] or lu["remarques"]):
        rows += _tete("Réponses et remarques (au dossier du loueur)", len(lu["reponses"]) + len(lu["remarques"]))
        qs = {q.get("id"): q for q in (B.loueur(base, lu["loueur"]) or {}).get("questions") or [] if isinstance(q, dict)}
        for it in lu["reponses"]:
            q = qs.get(it.get("id")) or {}
            rows += (f'<tr><td class="ob">{C.esc(q.get("question") or ("Autre remarque" if it.get("id") == "REMARQUE" else it.get("id")))}</td>'
                     f'<td>Réponse</td><td colspan="4" class="txt">{C.esc(it.get("reponse"))}</td></tr>')
        for it in lu["remarques"]:
            rows += (f'<tr><td class="ob">{C.esc(it.get("modele") or it.get("id"))}</td><td>Remarque</td>'
                     f'<td colspan="4" class="txt">{C.esc(it.get("texte"))}</td></tr>')
    if voir("refus") and lu["refus"]:
        rows += _tete(NOMS["refus"], len(lu["refus"]))
        for it in lu["refus"]:
            rows += (f'<tr class="refus"><td class="ob">{C.esc(it.get("id"))}</td><td>{C.esc(it.get("cle") or "")}</td>'
                     f'<td colspan="4" class="txt">{C.esc(it.get("raison"))}</td></tr>')
    if not rows:
        rows = '<tr><td colspan="6" class="pg-info">Rien à montrer pour ce filtre.</td></tr>'
    return (f'<div class="pg-tab-w"><table class="pg-tab pg-ctl"><thead><tr><th>Objet</th><th>Champ</th><th class="n">Avant</th>'
            f'<th class="n">Après</th><th class="n">Écart</th><th>Appliquer</th></tr></thead><tbody>{rows}</tbody></table></div>')


def _compteurs(lu: Dict) -> Dict[str, int]:
    r = dict(lu["resume"])
    r["completes"] = sum(1 for c in lu["changements"] if c.get("avant") is None)
    r["changements_prix"] = len(lu["changements"]) - r["completes"]
    return r


def gauche(base: Dict, lu: Dict, imp: Dict) -> str:
    i = lu.get("infos") or {}
    oui = {True: "Oui", False: "Non"}.get(i.get("inchanges_valables"), "pas dit")
    info = (C.ligne("Loueur", f"<b>{C.esc(lu['loueur'])}</b>") + C.ligne("Année", f"<b>{C.esc(lu['annee'])}</b>")
            + C.ligne("Fichier", f'<span class="pg-fic">{C.esc(imp.get("nom"))}</span>') + C.ligne("Rempli par", C.esc(i.get("rempli_par") or "—"))
            + (C.ligne("Mail", C.esc(i.get("mail"))) if i.get("mail") else "") + (C.ligne("Téléphone", C.esc(i.get("telephone"))) if i.get("telephone") else "")
            + C.ligne("Date", C.esc(_date_fr(i.get("date"))))
            + C.ligne("Validité", C.esc(f"{_date_fr(i.get('valable_du'))} → {_date_fr(i.get('valable_au'))}"))
            + C.ligne("Prix non modifiés valables", C.esc(oui)))
    cpt = _compteurs(lu)
    n = {"tout": None, "changements": cpt["changements"], "cases_videes": cpt["cases_videes"], "valeurs_non_lues": cpt["valeurs_non_lues"],
         "grues": cpt["nouvelles_grues"] + cpt["grues_retirees"], "autres_prestations": cpt["autres_prestations"],
         "messages": cpt["reponses"] + cpt["remarques"], "refus": cpt["refus"]}
    bt = ""
    for k, t in FILTRES:
        c = n[k]
        if c == 0 and k != imp.get("filtre"):
            continue
        bt += (f'<button type="button" class="pg-bt{" on" if k == (imp.get("filtre") or "tout") else ""}" data-chemin="imp/filtre" data-sorte="choix" '
               f'data-valeur="{k}"><span class="nm">{C.esc(t)}</span>{f"<span class={chr(34)}pc{chr(34)}>{c}</span>" if c is not None else ""}</button>')
    return (C.section(1, "Fichier reçu", C.esc(f"{lu['loueur']} {lu['annee']}"), info, True, "fichier")
            + C.section(2, "Filtre", C.esc(dict(FILTRES).get(imp.get("filtre") or "tout")), f'<div class="pg-liste-bt">{bt}</div>', True, "filtre-imp"))


def droite(base: Dict, lu: Dict, imp: Dict, seuil: float) -> str:
    cpt = _compteurs(lu)
    lignes = (("Prix changés", cpt["changements_prix"]), ("Prix complétés", cpt["completes"]), ("Prix confirmés", cpt["prix_confirmes"]),
              ("Cases vidées", cpt["cases_videes"]), ("Valeurs non lues", cpt["valeurs_non_lues"]), ("Nouvelles grues", cpt["nouvelles_grues"]),
              ("Grues plus proposées", cpt["grues_retirees"]), ("Autres prestations", cpt["autres_prestations"]),
              ("Réponses et remarques", cpt["reponses"] + cpt["remarques"]), ("Refusé", cpt["refus"]))
    corps = "".join(_kv(k, str(v)) for k, v in lignes)
    att = []
    d_ = IL.deja_importe(base, lu)
    if d_:
        att.append(f"Ce fichier a déjà été importé le {_date_fr(d_.get('date'))} : il ne peut pas l'être deux fois.")
    forts = sum(1 for c in lu["changements"] if IL.a_surligner(c, seuil))
    if forts:
        att.append(f"{forts} changement{'s' if forts > 1 else ''} au-delà de {PC.nombre(seuil, 1)} % ou en baisse (surlignés) : à regarder.")
    if cpt["cases_videes"]:
        att.append("Cases vidées par le loueur : elles ne s'appliquent que si on les coche.")
    if cpt["valeurs_non_lues"]:
        att.append("Valeurs non lues : taper la valeur à la main pour l'appliquer.")
    if cpt["refus"]:
        att.append(f"{cpt['refus']} case{'s' if cpt['refus'] > 1 else ''} refusée{'s' if cpt['refus'] > 1 else ''} : ce loueur ne fournit pas ces prix.")
    if (lu.get("infos") or {}).get("inchanges_valables") is not True:
        att.append("Le loueur ne dit pas que les prix non modifiés restent valables : ils gardent leur ancienne année.")
    ac = sum(1 for c in lu["changements"] if _corrige(base, c))
    if ac:
        att.append(f"{ac} prix corrigé{'s' if ac > 1 else ''} par nous : notre correction reste, la valeur du loueur passe dessous (à revoir).")
    avert = "".join(f'<li>{C.esc(a)}</li>' for a in att)
    avert = f'<ul class="pg-att">{avert}</ul>' if avert else ""
    act = (f'<button type="button" class="btn primary" data-agir="1" data-chemin="imp/appliquer" data-valeur="1"{" disabled" if d_ else ""}>'
           f'{C.icon("check")}Appliquer</button>'
           f'<button type="button" class="btn ghost" data-agir="1" data-chemin="imp/abandonner" data-valeur="1">{C.icon("x")}Abandonner</button>')
    return (f'<div class="pbody"><div class="pg-res"><div class="verdict pg-verdict"><div class="pg-total sm">{C.esc(lu["loueur"])} '
            f'{C.esc(lu["annee"])}<small>contrôle avant import</small></div></div>'
            f'<div class="pg-bloc"><div class="pg-bloc-h">Compteurs</div>{corps}</div>{avert}<div class="pg-fl-a pg-imp-a">{act}</div>'
            f'<p class="pg-info">Rien n\'est appliqué avant « Appliquer ». Ensuite, « Annuler le dernier import » (dossier du loueur) remet la base.</p></div></div>')


def page(etat: Dict, base: Dict, auj: dt.date):
    """(corps, vues, résultat, volets) de la page de contrôle ; une note sans fichier revient à la base."""
    imp = etat.get("imp")
    try:
        lu = lecture(etat, base)
    except LL.FichierRefuse as e:
        etat["imp"] = None
        etat["page"] = "base"
        raise ValueError(str(e))
    seuil = float(B.reglages(base).get("seuil_ecart") or 15)
    vues = [{"cle": "import", "titre": f"Contrôle du fichier {lu['loueur']} {lu['annee']}", "court": "Contrôle", "echelle": False,
             "svg": f'<div class="pg-detail">{centre(base, lu, imp, seuil)}</div>'}]
    return gauche(base, lu, imp), vues, droite(base, lu, imp, seuil), ""
