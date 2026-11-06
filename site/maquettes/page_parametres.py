# -*- coding: utf-8 -*-
"""Maquettes du site « Stab Méthodes » : la page Paramètres — les valeurs du BUREAU, puis celles de chaque module. Une
note peut changer une valeur pour elle seule (le champ porte alors « modifié »), sans toucher à celle du bureau. Les valeurs
affichées sont les valeurs par défaut des moteurs (module poutre, moteur garde-corps), lues par extraire_donnees.py.
"""
from construire import D, FAMILLES, MOD, esc, icon, ligne, mnav, nb, num, rail, seg, sel


def _carte(titre: str, ic: str, corps: str) -> str:
    return f'<div class="pcard"><div class="h3">{ic}{titre}</div><div class="rows">{corps}</div></div>'


def parametres() -> tuple:
    dp, dg = D["poutre"]["defauts"], D["gc_defauts"]
    v = nb  # les valeurs du bureau telles quelles (γ_M2 = 1,25, pas « 1,2 »)
    onglets = (f'<a class="navitem on">{icon("building-2")}<span>Le bureau</span></a><div class="navlabel">Modules</div>'
               + "".join(f'<a class="navitem"><span class="modic {cle}">{MOD[k]}</span><span>{esc(n)}</span></a>'
                         for cle, _, ms in FAMILLES for k, n in ms))
    bureau = (ligne("Nom", f'<div class="field text"><input value="Stab Méthodes"></div>', full=True)
              + ligne("Unités", sel(dp["unites"]), full=True)
              + ligne("Format des notes", sel("A4 paysage"), full=True)
              + ligne("Indice old", sel("grisé + filigrane OLD"), full=True)
              + ligne("Impression", sel("boîte du navigateur"), full=True))
    materiaux = (ligne("Béton", sel(dp["beton"]), full=True) + ligne("Armatures", seg(["B400", "B500"], f"B{int(dp['fyk'])}"), full=True)
                 + ligne("Acier profilés", sel(dg["nuance"]), full=True) + ligne("module d'élasticité", num(f"{int(dg['E']):,}".replace(",", " "), "N/mm²"), "E"))
    poutre = (ligne("Coefficient acier", num(v(dp["gamma_s"]), ""), "γ<sub>s</sub>") + ligne("Enrobage", num(v(dp["enrobage"]), "cm"), "c")
              + ligne("Jeu premier lit", num(v(dp["jeu_1"]), "cm")) + ligne("Jeu entre lits", num(v(dp["jeu_lits"]), "cm"))
              + ligne("Montage", sel(dp["techno"]), full=True))
    gc = (ligne("Méthode", seg(["ELS", "Eurocode"], "ELS" if dg["mode"] == "els" else "Eurocode"), full=True)
          + ligne("σ<sub>adm</sub> = f<sub>y</sub> /", num(v(dg["gamma_adm"]), ""), "")
          + ligne("Flèche", sel(f"L/{int(dg['fleche_val'])}"), full=True)
          + ligne("Eurocode", num(v(dg["gamma_M0"]), ""), "γ<sub>M0</sub>") + ligne("Eurocode", num(v(dg["gamma_M2"]), ""), "γ<sub>M2</sub>"))
    ic_poutre = '<span class="modic beton">' + MOD["poutre"] + "</span>"
    ic_gc = '<span class="modic autres">' + MOD["garde_corps"] + "</span>"
    contenu = (f'<div class="phint">Ce sont les valeurs du bureau. Une note peut en changer une pour elle seule : le champ porte alors '
               f'<span class="src mod">modifié</span> et la valeur du bureau reste en dessous. Seuls les administrateurs changent les valeurs du bureau.</div>'
               f'<div class="pgrid">{_carte("Le bureau", icon("building-2"), bureau)}{_carte("Matériaux par défaut", icon("layers"), materiaux)}'
               f'{_carte("Poutres béton armé", ic_poutre, poutre)}{_carte("Garde-corps", ic_gc, gc)}</div>')
    top = (f'<header class="top"><div class="crumbs">{icon("settings")}<b>Paramètres</b></div><span class="spacer"></span>'
           f'<span class="muted" style="font-size:12px">Valeurs du bureau · modifiables par les administrateurs</span></header>')
    corps = (f'<div class="app rail">{rail("parametres")}<main class="main">{top}<div class="params"><nav class="ptabs">{onglets}</nav>'
             f'<section class="pcol">{contenu}</section></div>{mnav("parametres")}</main></div>')
    return "Paramètres", corps, ""
