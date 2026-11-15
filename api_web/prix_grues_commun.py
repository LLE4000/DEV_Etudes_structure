# -*- coding: utf-8 -*-
"""Les briques d'écran COMMUNES aux trois pages du module « Prix des grues » (api_web/prix_grues.py) : l'ÉCUSSON (un
bouclier en SVG, son pictogramme et son mot — jamais la couleur seule, lisible imprimé en noir et blanc : cahier § 9), le
point d'écusson d'une case de tableau, les montants (« 14 455 € », « 0,79 € »), les onglets du module et l'écran vide.

Ce fichier ne calcule rien : les écussons viennent du moteur (grues/prix/ecussons.py)."""
from typing import Optional

from grues.prix import ecussons as E

from . import commun as C

BOUCLIER = "M12 2.2 19.6 5v6.1c0 4.9-3.2 8.6-7.6 10.7C7.6 19.7 4.4 16 4.4 11.1V5z"
PICTOS = {
    E.A_JOUR: '<path d="M8.6 12.1l2.4 2.4 4.6-4.9"/>',
    E.PAS_A_JOUR: '<circle cx="12" cy="12" r="4.1"/><path d="M12 9.6V12l1.7 1.1"/>',
    E.A_VERIFIER: '<path d="M12 8.2v4.6"/><path d="M12 15.6h.01" stroke-width="2.6"/>',
    E.MANQUANT: '<path d="M10.1 10a2 2 0 1 1 2.9 1.8c-.6.3-1 .8-1 1.4v.3"/><path d="M12 15.9h.01" stroke-width="2.6"/>',
    E.CORRIGE: '<path d="M9 15.3l.6-2.6 4.3-4.3 2 2-4.3 4.3z"/><path d="M8.4 16.2h3.3"/>',
    E.DEMANDE: '<rect x="8" y="9.2" width="8" height="5.8" rx=".8"/><path d="M8.3 9.6 12 12.4l3.7-2.8"/>',
}
ONGLETS = (("calcul", "Calcul"), ("fondations", "Fondations"), ("base", "Base des prix"))


def _svg(cle: str) -> str:
    return (f'<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path class="b" d="{BOUCLIER}"/>'
            f'<g class="p">{PICTOS.get(cle, "")}</g></svg>')


def ecusson(cle: Optional[str], raison: str = "", petit: bool = False) -> str:
    """L'écusson et son mot (``petit`` : la taille d'une ligne de tableau) ; ``raison`` au survol."""
    if not cle:
        return ""
    mot = E.LIBELLES.get(cle, cle)
    t = f"{mot} · {raison}" if raison else mot
    return f'<span class="pg-ec pg-ec-{cle}{" sm" if petit else ""}" title="{C.esc(t)}">{_svg(cle)}<span>{C.esc(mot)}</span></span>'


def point(cle: Optional[str], raison: str = "") -> str:
    """Le point d'écusson d'une case (un petit bouclier, son pictogramme, le mot au survol et pour les lecteurs d'écran)."""
    if not cle:
        return '<span class="pg-pt pg-pt-0" aria-hidden="true"></span>'
    mot = E.LIBELLES.get(cle, cle)
    t = f"{mot} · {raison}" if raison else mot
    return f'<span class="pg-pt pg-ec-{cle}" title="{C.esc(t)}" role="img" aria-label="{C.esc(mot)}">{_svg(cle)}</span>'


def _grouper(ent: str) -> str:
    signe, ch = ("-", ent[1:]) if ent.startswith("-") else ("", ent)
    tr = []
    while len(ch) > 3:
        ch, tr = ch[:-3], [ch[-3:]] + tr
    return signe + C.FINE.join([ch] + tr)


def nombre(x: Optional[float], nd_max: int = 2) -> str:
    """Un nombre sans décimales inutiles (« 1 300 », « 0,79 », « 8,5 ») ; « — » s'il manque."""
    if x is None:
        return "—"
    s = f"{x:.{nd_max}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    ent, _, dec = s.partition(".")
    if ent in ("-0", ""):
        ent = "0"
    return _grouper(ent) + ("," + dec if dec else "")


def euros(x: Optional[float], nd: int = 0) -> str:
    """Un montant : « 14 455 € » (``nd`` décimales au plus, sans zéros inutiles) ; « — » s'il manque."""
    if x is None:
        return "—"
    v = round(x, nd) if nd else int((x + 0.5) // 1) if x >= 0 else -int((-x + 0.5) // 1)
    return f"{nombre(v, nd)}{C.NBSP}€"


def prix_unitaire(x: Optional[float]) -> str:
    """Un prix unitaire : deux décimales au plus, trois sous 1 € (« 0,792 € »)."""
    if x is None:
        return "—"
    return f"{nombre(x, 3 if abs(x) < 1 else 2)}{C.NBSP}€"


def onglets(page: str, badges: Optional[dict] = None) -> str:
    """Les onglets du module (Calcul · Fondations · Base des prix), un badge quand quelque chose attend."""
    b = badges or {}
    out = ""
    for k, t in ONGLETS:
        n = b.get(k)
        bd = f'<span class="pg-badge" title="{C.esc(n[1])}">{C.esc(n[0])}</span>' if n else ""
        out += (f'<button type="button" class="{"on" if k == page else ""}" data-chemin="page" data-sorte="choix" data-valeur="{k}"'
                f'{" aria-current=\"page\"" if k == page else ""}>{C.esc(t)}{bd}</button>')
    return f'<div class="pg-onglets seg full" role="tablist" aria-label="Pages du module">{out}</div>'


ID_IMPORT = "pg-import-base"


def import_base(libelle: str = "Importer la base de départ") -> str:
    """Le bouton qui lit le fichier de la base (JSON) — dans le cadre Données (l'établi y lit les fichiers)."""
    return (f'<label class="btn sm fichier pg-import" for="{ID_IMPORT}">{C.icon("upload")}{C.esc(libelle)}</label>'
            f'<input type="file" id="{ID_IMPORT}" data-chemin="base/importer" accept=".json,application/json" hidden>')


def ecran_vide() -> str:
    """Le centre d'un module sans base : une seule action, claire."""
    return (f'<div class="pg-vide"><div class="pg-vide-c">{C.icon("upload")}<b>Aucune base des prix sur cet appareil</b>'
            f'<label class="btn primary" for="{ID_IMPORT}">{C.icon("upload")}Importer la base de départ</label>'
            f'<small>Le fichier JSON de la base (une seule fois). Les prix restent sur cet appareil.</small></div></div>')
