# TauxArmature.py
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Literal, Dict, Optional

import matplotlib.pyplot as plt
import streamlit as st


# =========================
# Types & constantes
# =========================

PositionType = Literal[
    "web_bottom", "web_top",
    "flange_left", "flange_right",
    "side_left", "side_right",
]

FamilyRole = Literal["main", "reinforcement", "lateral"]

StirrupTypeEnum = Literal["full", "inner", "flange_left", "flange_right"]

STEEL_DENSITY_KG_M3 = 7850.0


# =========================
# Géométrie de la poutre
# =========================

@dataclass
class BeamGeometry:
    """Géométrie de base de la section (âme + talons). Unités : cm."""
    b_web: float           # largeur âme (cm)
    h_total: float         # hauteur totale (cm)
    cover: float           # enrobage nominal (cm)

    flange_left_b: float = 0.0   # largeur talon gauche (cm)
    flange_left_h: float = 0.0   # hauteur talon gauche (cm)
    flange_right_b: float = 0.0  # largeur talon droit (cm)
    flange_right_h: float = 0.0  # hauteur talon droit (cm)

    def concrete_polygon(self) -> List[Tuple[float, float]]:
        """
        Polygon de béton (simplifié : âme rectangulaire + éventuels talons).
        Coordonnées en cm, origine au milieu de l’âme, fibre inf en y=0.
        """
        b = self.b_web
        h = self.h_total
        xL = -b / 2.0
        xR = b / 2.0

        pts: List[Tuple[float, float]] = []

        # bas âme
        pts.append((xL, 0.0))
        pts.append((xR, 0.0))

        # talon droit (option)
        if self.flange_right_b > 0 and self.flange_right_h > 0:
            pts.append((xR, self.flange_right_h))
            pts.append((xR + self.flange_right_b, self.flange_right_h))
            pts.append((xR + self.flange_right_b, h))

        # haut âme
        pts.append((xR, h))
        pts.append((xL, h))

        # talon gauche (option)
        if self.flange_left_b > 0 and self.flange_left_h > 0:
            pts.append((xL - self.flange_left_b, h))
            pts.append((xL - self.flange_left_b, self.flange_left_h))
            pts.append((xL, self.flange_left_h))

        return pts

    # ---------------------------------------------
    #  Bandes de placement des armatures
    # ---------------------------------------------
    def _horizontal_band(self, pos: PositionType) -> Tuple[Optional[float], float, float]:
        """
        Retourne (y, x_min, x_max) pour la ligne de centre des barres.
        Si y is None → bars verticales (barres latérales).
        """
        c = self.cover
        if pos == "web_bottom":
            y = c
            x_min = -self.b_web / 2.0 + c
            x_max = self.b_web / 2.0 - c

        elif pos == "web_top":
            y = self.h_total - c
            x_min = -self.b_web / 2.0 + c
            x_max = self.b_web / 2.0 - c

        elif pos == "flange_left":
            # talon gauche, bande horizontale
            if self.flange_left_b <= 0 or self.flange_left_h <= 0:
                raise ValueError("Talon gauche non défini pour placement des barres.")
            x_min = -self.b_web / 2.0 - self.flange_left_b + c
            x_max = -self.b_web / 2.0 - c
            y = self.flange_left_h - c

        elif pos == "flange_right":
            if self.flange_right_b <= 0 or self.flange_right_h <= 0:
                raise ValueError("Talon droit non défini pour placement des barres.")
            x_min = self.b_web / 2.0 + c
            x_max = self.b_web / 2.0 + self.flange_right_b - c
            y = self.flange_right_h - c

        elif pos == "side_left":
            # barre latérale gauche : on placera verticalement
            y = None
            x_min = -self.b_web / 2.0 + c
            x_max = x_min

        elif pos == "side_right":
            y = None
            x_max = self.b_web / 2.0 - c
            x_min = x_max

        else:
            raise ValueError(f"Position inconnue : {pos}")

        return y, x_min, x_max

    def bar_centers(
        self,
        pos: PositionType,
        n_bars: int,
        phi_mm: float,
    ) -> List[Tuple[float, float]]:
        """
        Calcule les centres des barres d’un lit.
        Retourne une liste de (x, y) en cm.
        """
        if n_bars <= 0:
            return []

        y, x_min, x_max = self._horizontal_band(pos)

        # lits horizontaux (Âme bas, Âme haut, Talons)
        if y is not None:
            if n_bars == 1:
                xs = [(x_min + x_max) / 2.0]
            else:
                xs = [
                    x_min + i * (x_max - x_min) / (n_bars - 1)
                    for i in range(n_bars)
                ]
            return [(x, y) for x in xs]

        # barres latérales verticales
        c = self.cover
        y_min = c
        y_max = self.h_total - c
        if n_bars == 1:
            ys = [(y_min + y_max) / 2.0]
        else:
            ys = [
                y_min + i * (y_max - y_min) / (n_bars - 1)
                for i in range(n_bars)
            ]
        x = x_min  # side_left ou side_right → même x
        return [(x, y) for y in ys]


# =========================
# Armatures longitudinales
# =========================

@dataclass
class RebarFamily:
    """
    Famille d’armatures longitudinales (lit principal, renfort local ou barre latérale).
    Longueur active définie en % de la portée.
    """
    id: str
    role: FamilyRole
    position: PositionType

    n_bars: int
    dia_mm: float

    with_hooks: bool = False
    hook_height_cm: float = 0.0

    with_lap: bool = False
    lap_length_mm: float = 0.0  # ℓ_rec (mm)

    x_start_pct: float = 0.0    # 0–100 %
    x_end_pct: float = 100.0    # 0–100 %

    def active_length_m(self, L_beam_m: float) -> float:
        """Longueur active en m, à partir des % de portée."""
        x1 = max(0.0, min(100.0, self.x_start_pct))
        x2 = max(0.0, min(100.0, self.x_end_pct))
        if x2 <= x1:
            return 0.0
        return L_beam_m * (x2 - x1) / 100.0

    def total_bar_length_m(self, L_beam_m: float, stock_length_m: float = 12.0) -> float:
        """
        Longueur totale d’acier pour cette famille, avec éventuels recouvrements.
        Formule :
            N = ceil( (L_active + ℓ_rec) / (L_stock - ℓ_rec) )
            L_tot = L_active + (N-1)*ℓ_rec
        puis × n_bars.
        """
        L_active = self.active_length_m(L_beam_m)
        if L_active <= 0.0 or self.n_bars <= 0:
            return 0.0

        if not self.with_lap or self.lap_length_mm <= 0.0:
            return L_active * self.n_bars

        lap_m = self.lap_length_mm / 1000.0
        usable = max(stock_length_m - lap_m, 1e-6)

        N = math.ceil((L_active + lap_m) / usable)
        L_tot_one = L_active + (N - 1) * lap_m
        return L_tot_one * self.n_bars

    def steel_area_mm2(self) -> float:
        """Section d’acier totale (mm²) de la famille."""
        return self.n_bars * math.pi * (self.dia_mm / 2.0) ** 2


def total_steel_length_m(
    families: List[RebarFamily],
    L_beam_m: float,
    stock_length_m: float = 12.0,
) -> float:
    """Somme des longueurs de toutes les familles."""
    return sum(f.total_bar_length_m(L_beam_m, stock_length_m) for f in families)


# =========================
# Étriers
# =========================

@dataclass
class StirrupType:
    """Type d’étrier (géométrie et diamètre)."""
    name: str
    phi_mm: float
    type: StirrupTypeEnum

    n_enclosed_bars: int = 0          # utile si "inner"
    n_vertical_legs: int = 2          # nombre de branches verticales
    bend_radius_mm: float = 0.0       # si 0 → 4ϕ par défaut


@dataclass
class StirrupZone:
    """Zone d’application en % de la portée."""
    name: str
    x_start_pct: float
    x_end_pct: float
    spacing_cm: float
    type_name: str                    # référence à StirrupType.name


@dataclass
class StirrupResult:
    zone: StirrupZone
    stirrup_type: StirrupType
    n_stirrups: int
    length_per_stirrup_m: float
    total_length_m: float


def stirrup_length_m(
    st_type: StirrupType,
    b_web_cm: float,
    h_total_cm: float,
    cover_cm: float,
) -> float:
    """
    Longueur d’un étrier (approx) en m.
    Modèle simple : périmètre rectangulaire de l’âme moins les enrobages.
    """
    phi = st_type.phi_mm
    r = st_type.bend_radius_mm or 4.0 * phi

    # passage en mm
    b_mm = b_web_cm * 10.0 - 2.0 * cover_cm * 10.0
    h_mm = h_total_cm * 10.0 - 2.0 * cover_cm * 10.0

    if st_type.type == "full":
        perim_mm = 2.0 * (b_mm + h_mm)
    else:
        # pour les autres types, on approxime à 3 côtés
        perim_mm = 1.5 * (b_mm + h_mm)

    # branches verticales supplémentaires → demi-cercle
    add_mm = max(st_type.n_vertical_legs - 2, 0) * (math.pi * r / 2.0)

    L_mm = perim_mm + add_mm
    return L_mm / 1000.0


def compute_stirrups_for_zones(
    zones: List[StirrupZone],
    types: List[StirrupType],
    L_beam_m: float,
    b_web_cm: float,
    h_total_cm: float,
    cover_cm: float,
) -> List[StirrupResult]:
    """Calcule le nombre et la longueur d’étriers pour chaque zone."""
    type_by_name = {t.name: t for t in types}
    results: List[StirrupResult] = []

    for zone in zones:
        st_type = type_by_name.get(zone.type_name)
        if st_type is None:
            continue

        x1 = max(0.0, min(100.0, zone.x_start_pct))
        x2 = max(0.0, min(100.0, zone.x_end_pct))
        if x2 <= x1 or zone.spacing_cm <= 0.0:
            continue

        L_zone_m = L_beam_m * (x2 - x1) / 100.0
        s_m = zone.spacing_cm / 100.0

        # nombre d’étriers (incluant celui du début de zone)
        n = int(L_zone_m / s_m) + 1

        L_one = stirrup_length_m(st_type, b_web_cm, h_total_cm, cover_cm)
        total = n * L_one

        results.append(
            StirrupResult(
                zone=zone,
                stirrup_type=st_type,
                n_stirrups=n,
                length_per_stirrup_m=L_one,
                total_length_m=total,
            )
        )

    return results


# =========================
# Section + quantités
# =========================

@dataclass
class RebarInstance:
    """Barre unique en section (centre + diamètre)."""
    x: float          # cm
    y: float          # cm
    phi_mm: float
    family_id: str


@dataclass
class BeamSection:
    """Section complète avec toutes les barres concrétisées."""
    geometry: BeamGeometry
    rebars: List[RebarInstance] = field(default_factory=list)

    def regenerate_rebars(self, families: List[RebarFamily]) -> None:
        """
        Reconstruit la liste de RebarInstance à partir des familles.
        """
        self.rebars.clear()
        for fam in families:
            centers = self.geometry.bar_centers(
                fam.position,
                fam.n_bars,
                fam.dia_mm,
            )
            for i, (x, y) in enumerate(centers):
                self.rebars.append(
                    RebarInstance(
                        x=x,
                        y=y,
                        phi_mm=fam.dia_mm,
                        family_id=fam.id,
                    )
                )


@dataclass
class QuantitiesResult:
    Vc_m3: float
    V_longitudinal_m3: float
    V_stirrups_m3: float
    mass_steel_kg: float
    rho_global_pct: float
    by_category_m3: Dict[str, float]


def rebar_volume_m3_from_length(length_m: float, dia_mm: float) -> float:
    """Volume (m³) d’acier = L × aire."""
    r_m = dia_mm / 1000.0 / 2.0
    area_m2 = math.pi * r_m**2
    return length_m * area_m2


def compute_quantities(
    b_web_cm: float,
    h_total_cm: float,
    L_beam_m: float,
    families: List[RebarFamily],
    stirrup_results: List[StirrupResult],
) -> QuantitiesResult:
    """Volumes béton / acier et taux d’armatures global + par catégorie."""
    Vc_m3 = (b_web_cm / 100.0) * (h_total_cm / 100.0) * L_beam_m

    V_long = 0.0
    cat_vol: Dict[str, float] = {
        "main": 0.0,
        "reinforcement": 0.0,
        "lateral": 0.0,
        "stirrups": 0.0,
    }

    for fam in families:
        L_m = fam.total_bar_length_m(L_beam_m)
        V = rebar_volume_m3_from_length(L_m, fam.dia_mm)
        V_long += V
        cat_vol[fam.role] += V

    V_st = 0.0
    for res in stirrup_results:
        V = rebar_volume_m3_from_length(
            res.total_length_m,
            res.stirrup_type.phi_mm,
        )
        V_st += V
        cat_vol["stirrups"] += V

    V_s_total = V_long + V_st
    mass_steel = V_s_total * STEEL_DENSITY_KG_M3
    rho_global = (V_s_total / Vc_m3) * 100.0 if Vc_m3 > 0 else 0.0

    return QuantitiesResult(
        Vc_m3=Vc_m3,
        V_longitudinal_m3=V_long,
        V_stirrups_m3=V_st,
        mass_steel_kg=mass_steel,
        rho_global_pct=rho_global,
        by_category_m3=cat_vol,
    )


# =========================
# Dessins matplotlib
# =========================

def draw_section(
    section: BeamSection,
    stirrup_type: Optional[StirrupType] = None,
    highlight_family_ids: Optional[List[str]] = None,
):
    """
    Dessin d’une section :
    - contour âme rectangulaire
    - toutes les barres longitudinales
    - étrier (type de zone)
    - éventuellement mise en évidence d’une famille (renfort)
    """
    geom = section.geometry
    b = geom.b_web
    h = geom.h_total

    fig, ax = plt.subplots(figsize=(4, 4))

    # Béton (âme rectangulaire)
    rect = plt.Rectangle(
        (-b / 2.0, 0.0),
        b,
        h,
        fill=False,
        linewidth=1.5,
    )
    ax.add_patch(rect)

    # Barres
    for bar in section.rebars:
        r_cm = bar.phi_mm / 20.0  # ϕ (mm) → rayon (cm)
        if highlight_family_ids and bar.family_id in highlight_family_ids:
            facecolor = "red"
        else:
            facecolor = "black"
        circ = plt.Circle((bar.x, bar.y), r_cm, color=facecolor)
        ax.add_patch(circ)

    # Étrier (contour simplifié à l’enrobage)
    if stirrup_type is not None:
        c = geom.cover
        str_rect = plt.Rectangle(
            (-b / 2.0 + c, c),
            b - 2 * c,
            h - 2 * c,
            fill=False,
            linestyle="--",
            linewidth=1.0,
            edgecolor="red",
        )
        ax.add_patch(str_rect)

    ax.set_aspect("equal", "box")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")

    margin = max(b, h) * 0.15
    ax.set_xlim(-b / 2.0 - margin, b / 2.0 + margin)
    ax.set_ylim(-margin, h + margin)
    ax.grid(True, linestyle=":", linewidth=0.5)

    fig.tight_layout()
    return fig


def draw_section_for_zone(
    section: BeamSection,
    stirrup_type: Optional[StirrupType] = None,
):
    """Alias pour compatibilité : une vue de section par zone d’étriers."""
    return draw_section(section, stirrup_type=stirrup_type)


# =========================
# Helpers Streamlit / état
# =========================

POSITION_LABELS = {
    "web_bottom": "Âme bas",
    "web_top": "Âme haut",
    "flange_left": "Talon gauche",
    "flange_right": "Talon droit",
    "side_left": "Barres latérales gauche",
    "side_right": "Barres latérales droite",
}

ROLE_LABELS = {
    "main": "Lit principal",
    "reinforcement": "Renfort local",
    "lateral": "Barres latérales",
}

POSITION_OPTIONS_MAIN = [
    ("Âme bas", "web_bottom"),
    ("Âme haut", "web_top"),
    ("Talon gauche", "flange_left"),
    ("Talon droit", "flange_right"),
]
POSITION_OPTIONS_LATERAL = [
    ("Latéral gauche", "side_left"),
    ("Latéral droit", "side_right"),
]


def _init_state():
    st.session_state.setdefault("L_beam_m", 6.0)

    st.session_state.setdefault("b_web_cm", 30.0)
    st.session_state.setdefault("h_total_cm", 60.0)
    st.session_state.setdefault("cover_cm", 3.0)

    st.session_state.setdefault("flange_left_b_cm", 0.0)
    st.session_state.setdefault("flange_left_h_cm", 0.0)
    st.session_state.setdefault("flange_right_b_cm", 0.0)
    st.session_state.setdefault("flange_right_h_cm", 0.0)

    st.session_state.setdefault("rebar_families", [])   # liste de dict
    st.session_state.setdefault("stirrup_types", [])    # liste de dict
    st.session_state.setdefault("stirrup_zones", [])    # liste de dict

    st.session_state.setdefault("family_counter", 0)
    st.session_state.setdefault("st_type_counter", 0)
    st.session_state.setdefault("st_zone_counter", 0)


def build_geometry() -> BeamGeometry:
    return BeamGeometry(
        b_web=st.session_state.b_web_cm,
        h_total=st.session_state.h_total_cm,
        cover=st.session_state.cover_cm,
        flange_left_b=st.session_state.flange_left_b_cm,
        flange_left_h=st.session_state.flange_left_h_cm,
        flange_right_b=st.session_state.flange_right_b_cm,
        flange_right_h=st.session_state.flange_right_h_cm,
    )


def build_rebar_families() -> List[RebarFamily]:
    families: List[RebarFamily] = []
    for d in st.session_state.rebar_families:
        families.append(
            RebarFamily(
                id=d["id"],
                role=d["role"],
                position=d["position"],
                n_bars=d["n_bars"],
                dia_mm=d["dia_mm"],
                with_hooks=d["with_hooks"],
                hook_height_cm=d["hook_height_cm"],
                with_lap=d["with_lap"],
                lap_length_mm=d["lap_length_mm"],
                x_start_pct=d["x_start_pct"],
                x_end_pct=d["x_end_pct"],
            )
        )
    return families


def build_stirrup_types() -> List[StirrupType]:
    types: List[StirrupType] = []
    for d in st.session_state.stirrup_types:
        types.append(
            StirrupType(
                name=d["name"],
                phi_mm=d["phi_mm"],
                type=d["type"],
                n_enclosed_bars=d["n_enclosed_bars"],
                n_vertical_legs=d["n_vertical_legs"],
                bend_radius_mm=d["bend_radius_mm"],
            )
        )
    return types


def build_stirrup_zones() -> List[StirrupZone]:
    zones: List[StirrupZone] = []
    for d in st.session_state.stirrup_zones:
        zones.append(
            StirrupZone(
                name=d["name"],
                x_start_pct=d["x_start_pct"],
                x_end_pct=d["x_end_pct"],
                spacing_cm=d["spacing_cm"],
                type_name=d["type_name"],
            )
        )
    return zones


# =========================
# UI Streamlit – édition des familles
# =========================

def add_family(role: FamilyRole):
    st.session_state.family_counter += 1
    fam_id = f"F{st.session_state.family_counter}"
    if role == "lateral":
        pos = "side_left"
    else:
        pos = "web_bottom" if role == "main" else "web_top"

    st.session_state.rebar_families.append(
        {
            "id": fam_id,
            "role": role,
            "position": pos,
            "n_bars": 2,
            "dia_mm": 16.0,
            "with_hooks": False,
            "hook_height_cm": 0.0,
            "with_lap": False,
            "lap_length_mm": 0.0,
            "x_start_pct": 0.0,
            "x_end_pct": 100.0,
        }
    )


def edit_family(idx: int, fam: Dict, role_label: str):
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(f"**{role_label} – {fam['id']}**")
    with c2:
        if st.button("🗑️ Supprimer", key=f"del_fam_{idx}"):
            st.session_state.rebar_families.pop(idx)
            st.rerun()

    cpos, c3, c4, c5 = st.columns([2, 1, 1, 1])

    with cpos:
        if fam["role"] == "lateral":
            opts = POSITION_OPTIONS_LATERAL
        else:
            opts = POSITION_OPTIONS_MAIN

        labels = [o[0] for o in opts]
        values = [o[1] for o in opts]
        i_sel = values.index(fam["position"]) if fam["position"] in values else 0
        pos_label = st.selectbox(
            "Position",
            labels,
            index=i_sel,
            key=f"pos_{idx}",
        )
        fam["position"] = values[labels.index(pos_label)]

    with c3:
        fam["n_bars"] = st.number_input(
            "n barres",
            min_value=1,
            max_value=20,
            value=int(fam["n_bars"]),
            step=1,
            key=f"nb_{idx}",
        )

    with c4:
        fam["dia_mm"] = st.number_input(
            "ϕ (mm)",
            min_value=6.0,
            max_value=40.0,
            value=float(fam["dia_mm"]),
            step=2.0,
            key=f"phi_{idx}",
        )

    with c5:
        st.markdown("Zone active (%L)")
        c5a, c5b = st.columns(2)
        with c5a:
            fam["x_start_pct"] = st.number_input(
                "De",
                min_value=0.0,
                max_value=100.0,
                value=float(fam["x_start_pct"]),
                step=5.0,
                key=f"x1_{idx}",
            )
        with c5b:
            fam["x_end_pct"] = st.number_input(
                "À",
                min_value=0.0,
                max_value=100.0,
                value=float(fam["x_end_pct"]),
                step=5.0,
                key=f"x2_{idx}",
            )

    c6, c7, c8 = st.columns(3)
    with c6:
        fam["with_hooks"] = st.checkbox(
            "Retours verticaux",
            value=bool(fam["with_hooks"]),
            key=f"hooks_{idx}",
        )
    with c7:
        fam["hook_height_cm"] = st.number_input(
            "Hauteur retour (cm)",
            min_value=0.0,
            max_value=1000.0,
            value=float(fam["hook_height_cm"]),
            step=1.0,
            key=f"h_hook_{idx}",
        )
    with c8:
        fam["with_lap"] = st.checkbox(
            "Recouvrement automatique",
            value=bool(fam["with_lap"]),
            key=f"lap_{idx}",
        )
        if fam["with_lap"]:
            fam["lap_length_mm"] = st.number_input(
                "ℓ_rec (mm)",
                min_value=0.0,
                max_value=5000.0,
                value=float(fam["lap_length_mm"] or 60.0 * fam["dia_mm"]),
                step=10.0,
                key=f"lap_len_{idx}",
            )
        else:
            fam["lap_length_mm"] = 0.0


# =========================
# UI Streamlit – étriers
# =========================

def add_stirrup_type():
    st.session_state.st_type_counter += 1
    name = f"T{st.session_state.st_type_counter}"
    st.session_state.stirrup_types.append(
        {
            "name": name,
            "phi_mm": 8.0,
            "type": "full",
            "n_enclosed_bars": 0,
            "n_vertical_legs": 2,
            "bend_radius_mm": 0.0,
        }
    )


def add_stirrup_zone():
    st.session_state.st_zone_counter += 1
    name = f"Z{st.session_state.st_zone_counter}"
    st.session_state.stirrup_zones.append(
        {
            "name": name,
            "x_start_pct": 0.0,
            "x_end_pct": 33.0,
            "spacing_cm": 15.0,
            "type_name": st.session_state.stirrup_types[0]["name"]
            if st.session_state.stirrup_types
            else "",
        }
    )


def edit_stirrup_type(idx: int, d: Dict):
    c1, c2 = st.columns([3, 1])
    with c1:
        d["name"] = st.text_input(
            "Nom du type",
            value=d["name"],
            key=f"st_name_{idx}",
        )
    with c2:
        if st.button("🗑️", key=f"del_st_type_{idx}"):
            st.session_state.stirrup_types.pop(idx)
            st.rerun()

    c3, c4, c5 = st.columns(3)
    with c3:
        d["phi_mm"] = st.number_input(
            "ϕ étrier (mm)",
            min_value=6.0,
            max_value=16.0,
            value=float(d["phi_mm"]),
            step=2.0,
            key=f"st_phi_{idx}",
        )
    with c4:
        type_label = st.selectbox(
            "Type",
            ["Complet", "Intérieur", "Talon gauche", "Talon droit"],
            index={"full": 0, "inner": 1, "flange_left": 2, "flange_right": 3}[
                d["type"]
            ],
            key=f"st_type_{idx}",
        )
        d["type"] = {
            "Complet": "full",
            "Intérieur": "inner",
            "Talon gauche": "flange_left",
            "Talon droit": "flange_right",
        }[type_label]
    with c5:
        d["n_vertical_legs"] = st.number_input(
            "Nb branches vert.",
            min_value=2,
            max_value=6,
            value=int(d["n_vertical_legs"]),
            step=1,
            key=f"st_legs_{idx}",
        )

    c6, c7 = st.columns(2)
    with c6:
        d["n_enclosed_bars"] = st.number_input(
            "Barres englobées (int.)",
            min_value=0,
            max_value=20,
            value=int(d["n_enclosed_bars"]),
            step=1,
            key=f"st_enc_{idx}",
        )
    with c7:
        d["bend_radius_mm"] = st.number_input(
            "Rayon courbures r (mm, 0 = 4ϕ)",
            min_value=0.0,
            max_value=500.0,
            value=float(d["bend_radius_mm"]),
            step=5.0,
            key=f"st_r_{idx}",
        )


def edit_stirrup_zone(idx: int, d: Dict):
    c1, c2 = st.columns([3, 1])
    with c1:
        d["name"] = st.text_input(
            "Nom de la zone",
            value=d["name"],
            key=f"zone_name_{idx}",
        )
    with c2:
        if st.button("🗑️", key=f"del_st_zone_{idx}"):
            st.session_state.stirrup_zones.pop(idx)
            st.rerun()

    c3, c4, c5 = st.columns(3)
    with c3:
        d["x_start_pct"] = st.number_input(
            "De (%L)",
            min_value=0.0,
            max_value=100.0,
            value=float(d["x_start_pct"]),
            step=5.0,
            key=f"zone_x1_{idx}",
        )
    with c4:
        d["x_end_pct"] = st.number_input(
            "À (%L)",
            min_value=0.0,
            max_value=100.0,
            value=float(d["x_end_pct"]),
            step=5.0,
            key=f"zone_x2_{idx}",
        )
    with c5:
        d["spacing_cm"] = st.number_input(
            "Espacement s (cm)",
            min_value=2.0,
            max_value=100.0,
            value=float(d["spacing_cm"]),
            step=1.0,
            key=f"zone_s_{idx}",
        )

    if st.session_state.stirrup_types:
        type_names = [t["name"] for t in st.session_state.stirrup_types]
        if d["type_name"] not in type_names:
            d["type_name"] = type_names[0]
        sel = st.selectbox(
            "Type d’étrier",
            type_names,
            index=type_names.index(d["type_name"]),
            key=f"zone_type_{idx}",
        )
        d["type_name"] = sel
    else:
        st.info("Crée au moins un type d’étrier pour l’assigner à la zone.")


# =========================
# Page principale
# =========================

def show():
    """Page Streamlit : Poutre BA – Modélisation et quantification du ferraillage."""
    _init_state()

    st.title("🧱 Poutre BA – Modélisation et quantification du ferraillage")

    st.sidebar.header("Géométrie & portée")
    st.sidebar.number_input(
        "Portée L (m)",
        min_value=0.5,
        max_value=60.0,
        value=float(st.session_state.L_beam_m),
        step=0.5,
        key="L_beam_m",
    )
    st.sidebar.number_input(
        "Âme b (cm)",
        min_value=5.0,
        max_value=200.0,
        value=float(st.session_state.b_web_cm),
        step=1.0,
        key="b_web_cm",
    )
    st.sidebar.number_input(
        "Hauteur h (cm)",
        min_value=10.0,
        max_value=300.0,
        value=float(st.session_state.h_total_cm),
        step=1.0,
        key="h_total_cm",
    )
    st.sidebar.number_input(
        "Enrobage nominal c (cm)",
        min_value=1.0,
        max_value=10.0,
        value=float(st.session_state.cover_cm),
        step=0.5,
        key="cover_cm",
    )

    with st.sidebar.expander("Talons (optionnels)", expanded=False):
        st.number_input(
            "Talon gauche – largeur (cm)",
            min_value=0.0,
            max_value=300.0,
            value=float(st.session_state.flange_left_b_cm),
            step=1.0,
            key="flange_left_b_cm",
        )
        st.number_input(
            "Talon gauche – hauteur (cm)",
            min_value=0.0,
            max_value=300.0,
            value=float(st.session_state.flange_left_h_cm),
            step=1.0,
            key="flange_left_h_cm",
        )
        st.number_input(
            "Talon droit – largeur (cm)",
            min_value=0.0,
            max_value=300.0,
            value=float(st.session_state.flange_right_b_cm),
            step=1.0,
            key="flange_right_b_cm",
        )
        st.number_input(
            "Talon droit – hauteur (cm)",
            min_value=0.0,
            max_value=300.0,
            value=float(st.session_state.flange_right_h_cm),
            step=1.0,
            key="flange_right_h_cm",
        )

    col_left, col_right = st.columns([2, 2])

    # ==============
    # COLONNE GAUCHE
    # ==============
    with col_left:
        st.subheader("1️⃣ Armatures longitudinales – Lits principaux")
        if st.button("➕ Ajouter un lit principal"):
            add_family("main")

        for idx, fam in enumerate(list(st.session_state.rebar_families)):
            if fam["role"] == "main":
                with st.container(border=True):
                    edit_family(idx, fam, "Lit principal")

        st.subheader("2️⃣ Renforts locaux")
        if st.button("➕ Ajouter un renfort local"):
            add_family("reinforcement")

        for idx, fam in enumerate(list(st.session_state.rebar_families)):
            if fam["role"] == "reinforcement":
                with st.container(border=True):
                    edit_family(idx, fam, "Renfort local")

        st.subheader("3️⃣ Barres latérales")
        if st.button("➕ Ajouter barres latérales"):
            add_family("lateral")

        for idx, fam in enumerate(list(st.session_state.rebar_families)):
            if fam["role"] == "lateral":
                with st.container(border=True):
                    edit_family(idx, fam, "Barres latérales")

    # ===============
    # COLONNE DROITE
    # ===============
    with col_right:
        st.subheader("4️⃣ Étriers")

        cst1, cst2 = st.columns(2)
        with cst1:
            if st.button("➕ Ajouter un type d’étrier"):
                add_stirrup_type()
        with cst2:
            if st.button("➕ Ajouter une zone d’étriers"):
                if not st.session_state.stirrup_types:
                    add_stirrup_type()
                add_stirrup_zone()

        if st.session_state.stirrup_types:
            st.markdown("**Types d’étriers**")
            for i, d in enumerate(list(st.session_state.stirrup_types)):
                with st.container(border=True):
                    edit_stirrup_type(i, d)

        if st.session_state.stirrup_zones:
            st.markdown("**Zones d’étriers (%L)**")
            for i, d in enumerate(list(st.session_state.stirrup_zones)):
                with st.container(border=True):
                    edit_stirrup_zone(i, d)

        st.markdown("---")
        st.subheader("5️⃣ Résultats & dessins")

        geom = build_geometry()
        families = build_rebar_families()
        stirrup_types = build_stirrup_types()
        stirrup_zones = build_stirrup_zones()

        section = BeamSection(geometry=geom)
        section.regenerate_rebars(families)

        stirrup_results = compute_stirrups_for_zones(
            stirrup_zones,
            stirrup_types,
            L_beam_m=st.session_state.L_beam_m,
            b_web_cm=st.session_state.b_web_cm,
            h_total_cm=st.session_state.h_total_cm,
            cover_cm=st.session_state.cover_cm,
        )

        qres = compute_quantities(
            b_web_cm=st.session_state.b_web_cm,
            h_total_cm=st.session_state.h_total_cm,
            L_beam_m=st.session_state.L_beam_m,
            families=families,
            stirrup_results=stirrup_results,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Volume béton Vc", f"{qres.Vc_m3:.3f} m³")
        c2.metric("Masse acier totale", f"{qres.mass_steel_kg:.1f} kg")
        c3.metric("Taux global ρ", f"{qres.rho_global_pct:.2f} %")

        with st.expander("Détail par catégorie"):
            st.write(
                {
                    "Lits (bas/haut + renforts)": f"{qres.by_category_m3['main'] + qres.by_category_m3['reinforcement']:.4f} m³",
                    "Renforts seuls": f"{qres.by_category_m3['reinforcement']:.4f} m³",
                    "Barres latérales": f"{qres.by_category_m3['lateral']:.4f} m³",
                    "Étriers": f"{qres.by_category_m3['stirrups']:.4f} m³",
                }
            )

        st.markdown("**Section générale (tous lits)**")
        fig_gen = draw_section(section)
        st.pyplot(fig_gen)

        # Sections par zone d’étriers
        if stirrup_results:
            st.markdown("**Sections par zone d’étriers**")
            for res in stirrup_results:
                st.markdown(
                    f"Zone **{res.zone.name}** : {res.zone.x_start_pct:.1f}–{res.zone.x_end_pct:.1f} %L  "
                    f"({res.n_stirrups} étriers, s = {res.zone.spacing_cm:.1f} cm)"
                )
                st.caption(
                    f"Type {res.stirrup_type.name} – ϕ {res.stirrup_type.phi_mm:.1f} mm"
                )
                fig_z = draw_section_for_zone(section, stirrup_type=res.stirrup_type)
                st.pyplot(fig_z)

        # Sections dédiées aux renforts – mise en évidence en rouge
        reinf_fams = [f for f in families if f.role == "reinforcement"]
        if reinf_fams:
            st.markdown("**Sections pour renforts locaux (barres en rouge)**")
            for f in reinf_fams:
                st.caption(
                    f"Renfort {f.id} – {POSITION_LABELS.get(f.position, f.position)} : "
                    f"{f.n_bars}Ø{f.dia_mm} de {f.x_start_pct:.0f} à {f.x_end_pct:.0f} %L"
                )
                fig_r = draw_section(section, highlight_family_ids=[f.id])
                st.pyplot(fig_r)


if __name__ == "__main__":
    st.set_page_config(page_title="Poutre BA – Ferraillage complet", page_icon="🧱", layout="wide")
    show()
