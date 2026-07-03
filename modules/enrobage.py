# =============================================================
#  enrobage.py — Calcul de l'enrobage du béton
#  VERSION 2.0 — refonte complète
#
#  Deux vérifications, on retient le maximum :
#   A. DURABILITÉ (EN 1992-1-1 §4.4.1) :
#        c_min = max( c_min,b ; c_min,dur + Δc_dur,γ − Δc_dur,st − Δc_dur,add ; 10 mm )
#        c_nom = c_min + Δc_dev
#      Ici Δc additionnels laissés à 0 (Annexe belge : valeurs par défaut).
#   B. FEU (EN 1992-1-2, méthode tabulée §5) :
#        distance à l'axe « a » lue dans les tableaux 5.x selon
#        l'élément, la durée R et la géométrie, puis
#        c_feu = a − Ø_barre/2 − Ø_étrier
#      (l'étrier est décompté car « a » est mesuré à l'axe de la barre
#       principale, à l'intérieur du cadre.)
#
#   Enrobage retenu = max( c_nom,durabilité ; c_nom,feu ).
#
#  Valeurs alignées Annexe nationale belge (NBN EN 1992). Les tableaux
#  feu reprennent les valeurs de a de l'EN 1992-1-2 (identiques dans
#  l'ANB). À valider par l'ingénieur pour tout projet.
# =============================================================

import streamlit as st


# =============================================================
#  STYLE (aligné poutre.py)
# =============================================================
C_COULEURS = {"ok": "#e6ffe6", "warn": "#fffbe6", "nok": "#ffe6e6", "info": "#eef2ff"}
C_ICONES = {"ok": "✅", "warn": "⚠️", "nok": "❌", "info": "ℹ️"}


def _bloc(left: str, right: str = "", etat: str = "ok"):
    right_html = f"<div style='font-weight:600;opacity:.9;white-space:nowrap;'>{right}</div>" if right else ""
    st.markdown(
        f"""
        <div style="background:{C_COULEURS.get(etat,'#f6f6f6')};padding:12px 14px;border-radius:10px;
             border:1px solid #d9d9d9;margin:8px 0 4px 0;display:flex;justify-content:space-between;
             align-items:center;gap:10px;">
          <div style="font-weight:700;">{left}</div>
          <div style="display:flex;align-items:center;gap:10px;">{right_html}
            <div style="font-size:20px;line-height:1;">{C_ICONES.get(etat,'')}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================
#  DONNÉES DURABILITÉ (EN 1992-1-1 / NBN)
# =============================================================
# Classes d'exposition : (titre court, description, exemples, quand la choisir)
EXPOSITION_INFO = {
    "X0": (
        "Aucun risque de corrosion ni d'attaque",
        "Béton non armé, ou béton armé en ambiance très sèche.",
        "Intérieur de bâtiments à très faible humidité (locaux chauffés secs).",
        "Élément jamais mouillé, armatures sans risque de corrosion. Rare pour du béton armé.",
    ),
    "XC1": (
        "Sec, ou humide en permanence",
        "Corrosion induite par carbonatation, ambiance sèche ou toujours humide.",
        "Intérieur de bâtiments (humidité faible) ; béton en permanence immergé dans l'eau.",
        "Élément intérieur au sec, OU totalement noyé sous l'eau sans alternance.",
    ),
    "XC2/XC3": (
        "Humidité modérée / rarement sec",
        "Carbonatation, contact durable avec l'eau ou humidité de l'air élevée.",
        "Fondations enterrées (XC2) ; béton extérieur abrité de la pluie (XC3).",
        "Fondations, vides sanitaires humides, éléments extérieurs sous auvent.",
    ),
    "XC4": (
        "Alternativement humide et sec",
        "Carbonatation, cycles d'humidification / séchage.",
        "Surfaces extérieures exposées à la pluie mais non aux sels.",
        "Façades, éléments extérieurs directement exposés à la pluie.",
    ),
    "XD1/XS1": (
        "Humidité modérée / air marin",
        "Corrosion par chlorures (sels de déverglaçage XD1, air marin XS1).",
        "Surfaces exposées aux embruns salins ou à des projections de sels.",
        "Ouvrages à moins de ~1 km de la côte, ou exposés aux sels de route sans contact direct.",
    ),
    "XD2/XS2": (
        "Humide, rarement sec / immersion marine",
        "Chlorures en milieu humide (XD2) ou immersion en eau de mer (XS2).",
        "Piscines, réservoirs d'eaux chlorées ; parties d'ouvrages marins immergées.",
        "Élément en contact durable avec une eau chlorée ou l'eau de mer.",
    ),
    "XD3/XS3": (
        "Alternativement humide et sec / zones de marnage et embruns",
        "Chlorures avec cycles humide/sec (XD3) ou zones de marnage/embruns (XS3).",
        "Tabliers de ponts, dalles de parking exposées aux sels ; ouvrages en zone de marée.",
        "Cas le plus sévère : sels + alternance humide/sec, embruns marins directs.",
    ),
}
EXPO_CLASSES = list(EXPOSITION_INFO.keys())

# Classe structurale recommandée par défaut (S4 = référence pour durée
# de vie 50 ans ; ajustée selon exposition — indicatif).
CLASSE_STRUCTURALE_DEFAULT = {
    "X0": "S1", "XC1": "S2", "XC2/XC3": "S4", "XC4": "S4",
    "XD1/XS1": "S5", "XD2/XS2": "S5", "XD3/XS3": "S6",
}

# c_min,dur (mm) selon classe structurale (lignes) × classe d'exposition
# (colonnes, dans l'ordre de EXPO_CLASSES). Tableau 4.4N (EN 1992-1-1).
C_MIN_DUR = {
    "S1": [10, 10, 10, 15, 20, 25, 30],
    "S2": [10, 10, 15, 20, 25, 30, 35],
    "S3": [10, 15, 20, 25, 30, 35, 40],
    "S4": [15, 20, 25, 30, 35, 40, 45],
    "S5": [20, 25, 30, 35, 40, 45, 50],
    "S6": [25, 30, 35, 40, 45, 50, 55],
}


# =============================================================
#  DONNÉES FEU (EN 1992-1-2, méthode tabulée §5)
#  Distance à l'axe « a » (mm) des armatures principales.
#  R = durée de résistance au feu (min).
# =============================================================
R_LEVELS = ["R30", "R60", "R90", "R120", "R180", "R240"]

# --- Dalles pleines portant sur une direction (tableau 5.8) ---
# a des armatures inférieures (mm)
FEU_DALLE = {
    "R30": 10, "R60": 20, "R90": 30, "R120": 40, "R180": 55, "R240": 65,
}

# --- Voiles porteurs (tableau 5.4), exposés sur une face ---
# a (mm) — dépend du taux d'exploitation ; on prend μ_fi ≈ 0,7 (courant).
FEU_VOILE = {
    "R30": 10, "R60": 10, "R90": 25, "R120": 35, "R180": 45, "R240": 55,
}

# --- Poteaux (tableau 5.2a), μ_fi = 0,7, exposés sur >1 face ---
# a (mm) donné pour un couple (b_min, a). On retient, par durée, une
# largeur mini courante et le a associé. Structure : R -> liste de
# variantes (b_min mm, a mm), l'utilisateur choisit selon sa largeur.
FEU_POTEAU = {
    "R30":  [(200, 25), (300, 25)],
    "R60":  [(200, 25), (300, 25)],
    "R90":  [(200, 31), (300, 25), (400, 25)],
    "R120": [(250, 40), (350, 35), (450, 35)],
    "R180": [(350, 45), (450, 40)],
    "R240": [(400, 50), (450, 40)],
}

# --- Poutres sur appuis simples (tableau 5.5) ---
# Par durée : liste de (b_min mm, a mm). a diminue quand b augmente.
FEU_POUTRE = {
    "R30":  [(80, 25), (120, 20), (160, 15), (200, 15)],
    "R60":  [(120, 40), (160, 35), (200, 30), (300, 25)],
    "R90":  [(150, 55), (200, 45), (300, 40), (400, 35)],
    "R120": [(200, 65), (240, 60), (300, 55), (500, 50)],
    "R180": [(240, 80), (300, 70), (400, 65), (600, 60)],
    "R240": [(280, 90), (350, 80), (500, 75), (700, 70)],
}


def a_feu(type_element: str, r_level: str, b_min_choice=None):
    """
    Retourne (a_mm, note) : distance à l'axe requise selon l'élément,
    la durée R et éventuellement la largeur mini choisie (poutre/poteau).
    """
    if type_element == "Dalle":
        return FEU_DALLE.get(r_level, 0), "Dalle pleine portant sur 1 direction (tab. 5.8)."
    if type_element == "Voile":
        return FEU_VOILE.get(r_level, 0), "Voile porteur exposé sur 1 face, μ_fi≈0,7 (tab. 5.4)."
    if type_element == "Poteau":
        variants = FEU_POTEAU.get(r_level, [])
        if not variants:
            return 0, ""
        if b_min_choice is None:
            b_min_choice = variants[0]
        b, a = b_min_choice
        return a, f"Poteau exposé sur >1 face, μ_fi≈0,7, b_min={b} mm (tab. 5.2a)."
    if type_element == "Poutre":
        variants = FEU_POUTRE.get(r_level, [])
        if not variants:
            return 0, ""
        if b_min_choice is None:
            b_min_choice = variants[0]
        b, a = b_min_choice
        return a, f"Poutre sur appuis simples, b_min={b} mm (tab. 5.5)."
    return 0, ""


# =============================================================
#  CALCULS (fonctions pures)
# =============================================================
def c_durabilite(classe_struct: str, classe_expo: str, diam_barre: float, delta_dev: float):
    """Retourne (c_min_dur, c_min_b, c_min, c_nom)."""
    idx = EXPO_CLASSES.index(classe_expo)
    c_min_dur = C_MIN_DUR[classe_struct][idx]
    c_min_b = diam_barre
    c_min = max(c_min_b, c_min_dur, 10)
    return c_min_dur, c_min_b, c_min, c_min + delta_dev


def c_feu(a_mm: float, diam_barre: float, diam_etrier: float, delta_dev: float):
    """
    c_feu,min = a − Ø_barre/2 − Ø_étrier  (enrobage au nu de l'étrier).
    Retourne (c_feu_min, c_feu_nom). Borné à 0.
    """
    c_min = max(a_mm - diam_barre / 2.0 - diam_etrier, 0.0)
    return c_min, c_min + delta_dev


# =============================================================
#  PAGE
# =============================================================
def show():
    st.markdown("## 🧱 Calcul de l'enrobage du béton")
    st.caption("Enrobage nominal retenu = max( durabilité EN 1992-1-1 ; feu EN 1992-1-2 méthode tabulée ). "
               "Valeurs Annexe nationale belge — à valider par l'ingénieur.")

    col_form, col_res = st.columns([1.15, 1])

    # ---------------------------------------------------------
    #  ENTRÉES
    # ---------------------------------------------------------
    with col_form:
        with st.expander("Paramètres généraux", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                type_element = st.selectbox("Type d'élément", ["Poutre", "Poteau", "Dalle", "Voile"])
            with c2:
                position = st.selectbox("Position dans l'ouvrage", ["Intérieur", "Extérieur"])

            c3, c4 = st.columns(2)
            with c3:
                diam_barre = st.number_input("Ø max armatures principales [mm]", value=20, step=2, min_value=6)
            with c4:
                diam_etrier = st.number_input("Ø étrier / cadre [mm]", value=8, step=2, min_value=0,
                                              help="Décompté dans la vérification au feu (a mesuré à l'axe de la barre principale).")

            delta_dev = st.radio("Tolérance d'exécution Δc_dev [mm]", [10, 5, 0], index=0, horizontal=True,
                                 help="10 mm par défaut ; peut être réduit avec contrôle qualité renforcé (ANB).")

        with st.expander("Durabilité — classe d'exposition", expanded=True):
            classe_expo = st.selectbox(
                "Classe d'exposition",
                EXPO_CLASSES,
                format_func=lambda x: f"{x} — {EXPOSITION_INFO[x][0]}",
            )
            titre, desc, exemple, quand = EXPOSITION_INFO[classe_expo]
            _bloc(f"{classe_expo} — {titre}", "", "info")
            st.markdown(f"**Mécanisme** : {desc}")
            st.markdown(f"**Quand la choisir** : {quand}")
            st.markdown(f"*Exemples* : {exemple}")

            classe_struct = st.selectbox(
                "Classe structurale",
                list(C_MIN_DUR.keys()),
                index=list(C_MIN_DUR.keys()).index(CLASSE_STRUCTURALE_DEFAULT[classe_expo]),
                help="S4 = référence (durée de vie 50 ans). Ajustée selon durée de vie, "
                     "classe de résistance, compacité (EN 1992-1-1 tab. 4.3N).",
            )

        with st.expander("Résistance au feu", expanded=False):
            check_feu = st.checkbox("Vérifier la résistance au feu (méthode tabulée EN 1992-1-2)")
            r_level = None
            b_choice = None
            if check_feu:
                r_level = st.select_slider("Durée de résistance requise", options=R_LEVELS, value="R60")
                if type_element in ("Poutre", "Poteau"):
                    variants = (FEU_POUTRE if type_element == "Poutre" else FEU_POTEAU).get(r_level, [])
                    if variants:
                        b_choice = st.selectbox(
                            "Largeur minimale de l'élément b [mm]",
                            variants,
                            format_func=lambda v: f"b ≥ {v[0]} mm  →  a = {v[1]} mm",
                            help="À largeur plus grande, la distance à l'axe requise diminue.",
                        )
                st.caption("μ_fi ≈ 0,7 (taux d'exploitation courant). Pour poteaux/voiles très chargés, "
                           "vérifier les tableaux complets.")

    # ---------------------------------------------------------
    #  RÉSULTATS
    # ---------------------------------------------------------
    with col_res:
        st.markdown("### Résultats")

        # --- A. Durabilité ---
        cmd, cmb, cmin_d, cnom_d = c_durabilite(classe_struct, classe_expo, diam_barre, delta_dev)
        _bloc("A — Durabilité", f"c_nom = {cnom_d:.0f} mm", "ok")
        st.markdown(
            f"- c_min,dur = **{cmd} mm** (classe {classe_struct} / {classe_expo})  \n"
            f"- c_min,b = **{cmb:.0f} mm** (= Ø barre)  \n"
            f"- c_min = max(c_min,b ; c_min,dur ; 10) = **{cmin_d:.0f} mm**  \n"
            f"- Δc_dev = **{delta_dev} mm**  \n"
            f"- **c_nom,durabilité = {cnom_d:.0f} mm**"
        )

        # --- B. Feu ---
        cnom_f = 0.0
        if check_feu and r_level:
            a_val, note = a_feu(type_element, r_level, b_choice)
            cmin_f, cnom_f = c_feu(a_val, diam_barre, diam_etrier, delta_dev)
            _bloc(f"B — Feu ({r_level})", f"c_nom = {cnom_f:.0f} mm", "ok")
            st.caption(note)
            st.markdown(
                f"- Distance à l'axe requise a = **{a_val:.0f} mm**  \n"
                f"- c_feu,min = a − Ø/2 − Ø_étrier = {a_val:.0f} − {diam_barre/2:.0f} − {diam_etrier:.0f} = **{cmin_f:.0f} mm**  \n"
                f"- Δc_dev = **{delta_dev} mm**  \n"
                f"- **c_nom,feu = {cnom_f:.0f} mm**"
            )
            if cmin_f <= 0:
                _bloc("Distance à l'axe couverte par la durabilité", "feu non dimensionnant", "info")
        else:
            _bloc("B — Feu", "non vérifié", "info")
            st.markdown("- Cochez « Vérifier la résistance au feu » pour l'inclure.")

        # --- C. Résumé ---
        c_final = max(cnom_d, cnom_f)
        gouv = "durabilité" if cnom_d >= cnom_f else "feu"
        _bloc("C — Enrobage nominal retenu", f"c_nom = {c_final:.0f} mm", "ok")
        st.markdown(f"Valeur gouvernée par la **{gouv}**.")
        if check_feu and abs(cnom_d - cnom_f) <= 2:
            _bloc("Durabilité et feu proches", "vérifier le cas réel", "warn")

    # ---------------------------------------------------------
    #  GUIDE CLASSES D'EXPOSITION
    # ---------------------------------------------------------
    st.divider()
    with st.expander("📖 Comment choisir la classe d'exposition ?", expanded=False):
        st.markdown(
            "La classe d'exposition décrit **l'agression environnementale** subie par le béton. "
            "On identifie d'abord le **mécanisme dominant**, puis le **degré d'humidité / d'exposition**."
        )
        st.markdown(
            "**1. Pas de risque** — X0 : élément jamais mouillé, béton non armé ou ambiance très sèche.\n\n"
            "**2. Carbonatation (XC)** — corrosion par CO₂, cas le plus fréquent en bâtiment :\n"
            "- XC1 : intérieur sec, ou béton en permanence immergé\n"
            "- XC2 : fondations, contact durable avec l'eau/sol humide\n"
            "- XC3 : extérieur abrité de la pluie, ou humidité de l'air élevée\n"
            "- XC4 : extérieur exposé à la pluie (cycles humide/sec)\n\n"
            "**3. Chlorures hors mer (XD)** — sels de déverglaçage, eaux industrielles :\n"
            "- XD1 : projections/embruns de sels · XD2 : immersion en eau chlorée · XD3 : cycles humide/sec avec sels (ponts, parkings)\n\n"
            "**4. Chlorures marins (XS)** — eau de mer :\n"
            "- XS1 : air marin (côte) · XS2 : immersion permanente · XS3 : zone de marnage et embruns\n\n"
            "**5. Autres agressions** (gel/dégel XF, chimique XA) : non couvertes ici, à traiter séparément."
        )
        _bloc("Règle pratique", "prendre la classe la plus sévère applicable", "info")
        st.markdown(
            "Un même élément peut relever de **plusieurs classes** (ex. une dalle de parking extérieure : "
            "XC4 + XD3 + XF). On retient alors, pour chaque mécanisme, la classe la plus contraignante, "
            "et l'enrobage résulte de la combinaison la plus sévère."
        )
