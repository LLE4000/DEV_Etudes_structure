# -*- coding: utf-8 -*-
"""Le type « Prédalle » de l'écran des dalles (api_web/dalle.py) : ce qui n'appartient qu'à lui À L'AFFICHAGE — où la
règle pose chaque couche (dans la prédalle, sur la prédalle, depuis le dessus) et son pictogramme, la légende des
dessins, les avertissements de la prédalle, les hypothèses de la fenêtre de calcul et ce que le calcul vérifie.

L'ancien module « Prédalles » (une page à lui, api_web/predalle.py) a rejoint les dalles le 09/11/2026
(docs/site/DALLES_FUSION.md § 2.4) ; son écran, que plus aucune page n'ouvrait, est retiré le 13/11/2026 (lot 2,
rangement) : ne restent ici, recopiées à l'identique, que les pièces que l'écran des dalles emploie. La saisie de la
peau et de l'épaisseur totale est dans api_web/dalle.py, avec les autres saisies. Rien n'est calculé ici.
"""
from typing import Dict, List, Optional

from . import commun as C

POSES_TXT = {"predalle": "dans la prédalle", "sur_predalle": "sur la prédalle", "superieure": "depuis le dessus"}

POSES_AIDE = {"predalle": "Position automatique : la base est posée en fond de moule, DANS la prédalle — enrobage + ½ Ø (la "
                          "base secondaire, sur la principale) ; le calcul y ajoute le jeu de pose et arrondit.",
              "sur_predalle": "Position automatique : un renfort posé SUR la prédalle, au chantier — épaisseur de la prédalle "
                              "+ ½ Ø ; le calcul y ajoute le jeu de pose et arrondit.",
              "superieure": "Position automatique, depuis le dessus : enrobage + ½ Ø ; le calcul y ajoute le jeu de pose et "
                            "arrondit."}

# la légende des dessins (dernière ligne de l'en-tête de chaque vue : ce qu'on y lit)
LEGENDES = {"c": "prédalle hachurée · barres coupées en points, barres parallèles en traits",
            "p": "vue de dessus · barres X horizontales, Y verticales"}

# ------------------------------------------------------------------ les données : l'arborescence (colonne de gauche)
REGLES_TXT = {"predalle": "c + ½ Ø", "sur_predalle": "h<sub>p</sub> + ½ Ø", "superieure": "c + ½ Ø"}

# Le pictogramme de la POSE d'une couche : une coupe en miniature (la peau hachurée en bas, ou le dessus de la dalle),
# et la barre à sa place — dans la peau, posée sur la peau, ou sous le dessus. Dessiné à la main : aucun identifiant.
_PEAU = '<rect class="peau" x="0.5" y="{y}" width="17" height="{h}" rx="1"/><path class="hach" d="{d}"/>'

PICTOS_POSE = {
    "predalle": (_PEAU.format(y=4.5, h=7, d="M2 11.5 9 4.5M7 11.5 14 4.5M12 11.5 17 6.5") + '<circle class="barre" cx="9" cy="8" r="2.3"/>'),
    "sur_predalle": (_PEAU.format(y=7.5, h=4, d="M2 11.5 6 7.5M7 11.5 11 7.5M12 11.5 16 7.5") + '<circle class="barre" cx="9" cy="4.9" r="2.3"/>'),
    "superieure": '<path class="dessus" d="M0.5 1.2H17.5"/><circle class="barre" cx="9" cy="4.6" r="2.3"/>',
}

def _picto_pose(pose: str) -> str:
    return f'<svg class="pd-m pd-{pose}" viewBox="0 0 18 12" aria-hidden="true">{PICTOS_POSE[pose]}</svg>'

def sur_principale(x: Dict) -> str:
    """Sur quoi repose une couche (ou sa formule « e ») que le site a empilée : ce que ``dalle_positions.annoter`` en dit
    (``sur`` : « la base Y », « le renfort Y », « le renfort R2 Y » — les mots de la formule « e » et de la légende des
    coupes, une source), « la base principale » sans lui — S5 : un renfort principal plus haut que la base porte la nappe
    secondaire ; DAL5b : un renfort principal posé sur un treillis unique (``appui_sec``) repose sur « le treillis » ou
    sur « le renfort X »."""
    if x.get("appui_sec"):
        return x.get("sur") or "le treillis"
    return x.get("sur") or "la base principale"

def pose_couche(c: Dict, geo_c: Dict) -> str:
    """OÙ la règle de la prédalle pose une couche (dans la prédalle, sur la prédalle, depuis le dessus), et l'axe
    automatique qu'elle en tire — ou, l'axe saisi, ce que l'automatique serait : la ligne du détail de la couche. Une
    couche secondaire EMPILÉE par le site (module fusionné, § 4.1 : faces supérieures ; sa base et, depuis le 09/11/2026,
    ses renforts — Ø_s est alors celui du renfort) le dit, et sur quoi elle repose : la base principale, ou le renfort
    principal plus haut qu'elle (S5 : e_p, Ø_p sont alors ceux du renfort)."""
    pose = geo_c["pose"]
    if geo_c.get("empilee"):
        # (une couche supérieure secondaire que le site a posée sur la principale : « depuis le dessus (c + ½ Ø + j) » mentirait)
        txt = (f"position auto · posée sur {sur_principale(geo_c)} (e<sub>p</sub> + ½ Ø<sub>p</sub> + ½ Ø<sub>s</sub>)")
    elif c["dist"] is None:
        txt = f"position auto · {POSES_TXT[pose]} ({REGLES_TXT[pose]})"
    else:
        txt = f"position saisie · l'automatique serait {C.nb(geo_c['reel_auto'])} cm, {POSES_TXT[pose]}"
    return f'<div class="pd-pose" title="{C.esc(POSES_AIDE[pose])}">{_picto_pose(pose)}<span>{txt}</span></div>'

def avertissements(r: Dict, i: Optional[int] = None) -> str:
    """Les avertissements PROPRES à la prédalle (base hors de la peau, couches croisées sur la prédalle) : ceux de la
    section ``i`` (posés dans son bloc), ou, sans ``i``, ceux de la prédalle entière. Les bases X et Y confondues sont
    dites par la phrase des dalles (api_web/dalle._avertissements : seule une position saisie les y met, depuis la
    règle de position du 13/11/2026 en bas aussi)."""
    out = ""
    for a in r["avertissements"]:
        if a.get("section") != i or a["code"] == "nappes_confondues":
            continue
        if a["code"] == "enrobage_nul":
            out += C.message("Enrobage nul : comme l'écran Streamlit, le calcul pose les couches avec un enrobage de 3 cm.")
        elif a["code"] not in ("d_utile_negatif", "joint_d_utile_negatif"):   # (d ≤ 0 : la ligne le dit, en rouge)
            out += C.message(C.esc(a["message"]), grave=a["code"] in ("base_hors_predalle", "secondaire_hors_predalle"))
    return out

# Ce que le module VÉRIFIE, et ce qu'il ne vérifie pas : en entier dans chaque fenêtre de calcul, pour qu'un
# « conforme » ne se lise pas plus large qu'il n'est ; sous le résultat, la seule phrase « Non vérifiés » (NON_VERIFIES) —
# ce qui est vérifié, les lignes du résultat le montrent déjà (le propriétaire, 13/11/2026).
NON_VERIFIES = ("Non vérifiés : la phase provisoire (prédalle seule, étaiement, levage), le cisaillement à l'interface "
                "prédalle / béton coulé en place, la flèche et la fissuration.")

PERIMETRE = ("Vérifié : la section finale (prédalle et béton coulé en place ensemble), en flexion (hauteur, les quatre armatures), "
             "le joint de deux prédalles (la direction secondaire avec les barres de couture, leur longueur de recouvrement) "
             "et τ ≤ τ<sub>adm,I</sub>. " + NON_VERIFIES)

def hypotheses(p: Dict, m: Dict) -> List[str]:
    return [f"Béton <b>{C.esc(m['beton'])}</b> · f<sub>ck</sub> = {C.nb(m['fck'])} · f<sub>ck,cube</sub> = {C.nb(m['fck_cube'])} N/mm²",
            f"f<sub>ctm</sub> = 0,30·f<sub>ck</sub><sup>2/3</sup> = {C.fr(m['fctm'], 2)} N/mm²",
            f"Acier <b>B{int(m['fyk'])}</b> · f<sub>yd</sub> = {int(m['fyk'])} / {C.nb(m['gamma_s'])} = {C.fr(m['fyd'], 1)} N/mm²",
            f"α<sub>b</sub> = {C.nb(m['alpha_b'])} · μ = {C.nb(m['mu'], 4)}",
            f"Bande b × h = {C.nb(p['b'])} × {C.nb(p['h'])} cm, dont prédalle <b>{C.nb(p['h_pre'])} cm</b> et "
            f"{C.nb(p['h'] - p['h_pre'])} cm coulés en place",
            f"Enrobage {C.nb(p['enrobage'])} cm · jeu de pose {C.nb(p['jeu_premier_lit'])} cm · arrondi {C.txt_arrondi(p.get('arrondi'))}"]
