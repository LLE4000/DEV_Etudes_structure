# modules/taux_armature.py
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Literal, Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, PathPatch
from matplotlib.path import Path
import pandas as pd
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

STEEL_DENSITY_KG_M3 = 7860.0  # kg/m³

# Diamètres autorisés pour toutes les armatures
ALLOWED_DIA_MM = [6, 8, 10, 12, 16, 20, 25, 32, 40]


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
        Polygone de béton (âme rectangulaire + éventuels talons).
        Coordonnées en cm, origine au milieu de l’âme, fibre inf en y=0.

        Les talons sont pris comme des consoles qui DÉPASSENT vers le bas.
        """
        b = self.b_web
        h = self.h_total
        xL_web = -b / 2.0
        xR_web = b / 2.0

        has_left = self.flange_left_b > 0 and self.flange_left_h > 0
        has_right = self.flange_right_b > 0 and self.flange_right_h > 0

        x_extL = xL_web - self.flange_left_b if has_left else xL_web
        x_extR = xR_web + self.flange_right_b if has_right else xR_web

        pts: List[Tuple[float, float]] = []

        # On part du bas, côté gauche extérieur
        pts.append((x_extL, 0.0))

        # Bas : vers la droite
        pts.append((x_extR, 0.0))

        # Côté droit : remontée
        if has_right:
            # talon droit : monte de 0 à h_flange, puis âme jusqu'à h
            pts.append((x_extR, self.flange_right_h))
            pts.append((xR_web, self.flange_right_h))
            pts.append((xR_web, h))
        else:
            pts.append((xR_web, 0.0))
            pts.append((xR_web, h))

        # Haut : traverse de droite à gauche
        pts.append((xL_web, h))

        # Côté gauche : descente
        if has_left:
            pts.append((xL_web, self.flange_left_h))
            pts.append((x_extL, self.flange_left_h))
            pts.append((x_extL, 0.0))
        else:
            pts.append((xL_web, 0.0))
            # on est déjà revenu à (x_extL, 0) au début

        return pts

    def area_cm2(self) -> float:
        """Aire totale de la section (âme + talons) en cm²."""
        area = self.b_web * self.h_total
        if self.flange_left_b > 0 and self.flange_left_h > 0:
            area += self.flange_left_b * self.flange_left_h
        if self.flange_right_b > 0 and self.flange_right_h > 0:
            area += self.flange_right_b * self.flange_right_h
        return area

    # ---------------------------------------------
    #  Bandes de placement des armatures
    # ---------------------------------------------
    def _horizontal_band(self, pos: PositionType) -> Tuple[Optional[float], float, float]:
        """
        Retourne (y_face, x_min, x_max) pour la ligne de face la plus proche.
        y_face = coordonnée de la face béton (pas de l’axe).
        Si y_face is None → barres verticales (barres latérales).
        """
        c = self.cover
        if pos == "web_bottom":
            y_face = 0.0
            x_min = -self.b_web / 2.0 + c
            x_max = self.b_web / 2.0 - c

        elif pos == "web_top":
            y_face = self.h_total
            x_min = -self.b_web / 2.0 + c
            x_max = self.b_web / 2.0 - c

        elif pos == "flange_left":
            if self.flange_left_b <= 0 or self.flange_left_h <= 0:
                raise ValueError("Talon gauche non défini pour placement des barres.")
            x_min = -self.b_web / 2.0 - self.flange_left_b + c
            x_max = -self.b_web / 2.0 - c
            y_face = self.flange_left_h

        elif pos == "flange_right":
            if self.flange_right_b <= 0 or self.flange_right_h <= 0:
                raise ValueError("Talon droit non défini pour placement des barres.")
            x_min = self.b_web / 2.0 + c
            x_max = self.b_web / 2.0 + self.flange_right_b - c
            y_face = self.flange_right_h

        elif pos == "side_left":
            y_face = None
            x_min = -self.b_web / 2.0 + c
            x_max = x_min

        elif pos == "side_right":
            y_face = None
            x_max = self.b_web / 2.0 - c
            x_min = x_max

        else:
            raise ValueError(f"Position inconnue : {pos}")

        return y_face, x_min, x_max

    def bar_centers(
        self,
        pos: PositionType,
        n_bars: int,
        phi_mm: float,
        layer_index: int = 0,
    ) -> List[Tuple[float, float]]:
        """
        Calcule les centres des barres d’un lit.
        Retourne une liste de (x, y) en cm.
        layer_index : 0 pour le lit le plus proche de la face,
                      1 pour le lit juste au-dessus, etc.
        """
        if n_bars <= 0:
            return []

        y_face, x_min, x_max = self._horizontal_band(pos)

        # lits horizontaux (Âme bas, Âme haut, Talons)
        if y_face is not None:
            r_cm = phi_mm / 20.0
            base_cover = self.cover + r_cm
            pitch = 2.0 * r_cm + 2.0  # espacement vertical entre lits (2ϕ + 2 cm)

            if pos in ("web_bottom", "flange_left", "flange_right"):
                y = y_face + base_cover + layer_index * pitch
            else:  # web_top
                y = y_face - (base_cover + layer_index * pitch)

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
            # règle générale : on place n+2 barres, on retire les 2 extrêmes
            n_eff = n_bars + 2
            idxs = range(1, n_eff - 1)  # 1 .. n_eff-2 → n_bars positions
            ys = [
                y_min + i * (y_max - y_min) / (n_eff - 1)
                for i in idxs
            ]

        x = x_min
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
    hook_mode: str = "none"  # "none", "both", "left", "right"

    with_lap: bool = False
    lap_length_mm: float = 0.0  # ℓ_rec (mm)

    x_start_pct: float = 0.0    # 0–100 %
    x_end_pct: float = 100.0    # 0–100 %

    def active_length_m(self, L_beam_m: float) -> float:
        """Longueur active en m, à partir des % de portée (partie horizontale)."""
        x1 = max(0.0, min(100.0, self.x_start_pct))
        x2 = max(0.0, min(100.0, self.x_end_pct))
        if x2 <= x1:
            return 0.0
        return L_beam_m * (x2 - x1) / 100.0

    def _hooks_per_bar(self) -> int:
        if self.hook_mode == "both":
            return 2
        if self.hook_mode in ("left", "right"):
            return 1
        return 0

    def total_bar_length_m(self, L_beam_m: float, stock_length_m: float = 12.0) -> float:
        """
        Longueur totale d’acier pour cette famille, avec éventuels recouvrements et retours.
        """
        L_active = self.active_length_m(L_beam_m)
        if L_active <= 0.0 or self.n_bars <= 0:
            return 0.0

        hooks_per_bar = self._hooks_per_bar()
        hook_height_m = max(self.hook_height_cm, 0.0) / 100.0

        # Partie horizontale avec recouvrements
        if not self.with_lap or self.lap_length_mm <= 0.0:
            straight_total = L_active * self.n_bars
            n_stock_bars = 1
        else:
            lap_m = self.lap_length_mm / 1000.0
            usable = max(stock_length_m - lap_m, 1e-6)

            N = math.ceil((L_active + lap_m) / usable)
            straight_one = L_active + (N - 1) * lap_m
            straight_total = straight_one * self.n_bars
            n_stock_bars = N

        extra_hooks_total = hooks_per_bar * hook_height_m * self.n_bars * n_stock_bars

        return straight_total + extra_hooks_total

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

    n_enclosed_bars: int = 0
    n_vertical_legs: int = 2
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
    geom: BeamGeometry,
) -> float:
    """
    Longueur d’un étrier en m (approx) :
    on prend le contour de l’âme à l’enrobage, avec petits crochets.
    """
    c = geom.cover
    b_cm = geom.b_web - 2 * c
    h_cm = geom.h_total - 2 * c

    b_mm = b_cm * 10.0
    h_mm = h_cm * 10.0

    perim_mm = 2.0 * (b_mm + h_mm)

    phi = st_type.phi_mm
    hook_mm = 8.0 * phi * 2  # deux crochets
    L_mm = perim_mm + hook_mm

    return L_mm / 1000.0


def compute_stirrups_for_zones(
    zones: List[StirrupZone],
    types: List[StirrupType],
    L_beam_m: float,
    geom: BeamGeometry,
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

        n = int(L_zone_m / s_m) + 1

        L_one = stirrup_length_m(st_type, geom)
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
    color: str = "black"


@dataclass
class BeamSection:
    """Section complète avec toutes les barres concrétisées."""
    geometry: BeamGeometry
    rebars: List[RebarInstance] = field(default_factory=list)

    def regenerate_rebars(
        self,
        families: List[RebarFamily],
        family_colors: Optional[Dict[str, str]] = None,
        layer_by_id: Optional[Dict[str, int]] = None,
    ) -> None:
        self.rebars.clear()
        color_map = family_colors or {}
        layer_map = layer_by_id or {}
        for fam in families:
            layer_index = layer_map.get(fam.id, 0)
            centers = self.geometry.bar_centers(
                fam.position,
                fam.n_bars,
                fam.dia_mm,
                layer_index=layer_index,
            )
            fam_color = color_map.get(fam.id, "black")
            for (x, y) in centers:
                self.rebars.append(
                    RebarInstance(
                        x=x,
                        y=y,
                        phi_mm=fam.dia_mm,
                        family_id=fam.id,
                        color=fam_color,
                    )
                )


@dataclass
class QuantitiesResult:
    Vc_m3: float
    V_longitudinal_m3: float
    V_stirrups_m3: float
    mass_steel_kg: float
    rho_global_pct: float
    kg_per_m3: float
    by_category_m3: Dict[str, float]


def rebar_volume_m3_from_length(length_m: float, dia_mm: float) -> float:
    r_m = dia_mm / 1000.0 / 2.0
    area_m2 = math.pi * r_m**2
    return length_m * area_m2


def steel_linear_mass_kg_m(dia_mm: float) -> float:
    r_m = dia_mm / 1000.0 / 2.0
    area_m2 = math.pi * r_m**2
    return area_m2 * STEEL_DENSITY_KG_M3


def compute_quantities(
    geom: BeamGeometry,
    L_beam_m: float,
    families: List[RebarFamily],
    stirrup_results: List[StirrupResult],
) -> QuantitiesResult:
    area_cm2 = geom.area_cm2()
    Vc_m3 = area_cm2 / 10000.0 * L_beam_m

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
    kg_per_m3 = mass_steel / Vc_m3 if Vc_m3 > 0 else 0.0

    return QuantitiesResult(
        Vc_m3=Vc_m3,
        V_longitudinal_m3=V_long,
        V_stirrups_m3=V_st,
        mass_steel_kg=mass_steel,
        rho_global_pct=rho_global,
        kg_per_m3=kg_per_m3,
        by_category_m3=cat_vol,
    )


# =========================
# Outils couleurs & codage des lits
# =========================

def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _gradient(start: str, end: str, n: int) -> List[str]:
    if n <= 1:
        return [start]
    s = _hex_to_rgb(start)
    e = _hex_to_rgb(end)
    colors = []
    for i in range(n):
        t = i / (n - 1)
        rgb = tuple(int(s[k] + (e[k] - s[k]) * t) for k in range(3))
        colors.append(_rgb_to_hex(rgb))
    return colors


def _position_group(pos: str) -> str:
    if pos == "web_bottom":
        return "bottom"
    if pos in ("web_top", "flange_left", "flange_right"):
        return "top"
    return "other"


def compute_family_colors_for_families(
    families: List[RebarFamily],
) -> Dict[str, str]:
    bottom_ids = [f.id for f in families if _position_group(f.position) == "bottom"]
    top_ids = [f.id for f in families if _position_group(f.position) == "top"]
    other_ids = [f.id for f in families if _position_group(f.position) == "other"]

    colors: Dict[str, str] = {}

    reds = _gradient("#ffcccc", "#b30000", max(len(bottom_ids), 1))
    blues = _gradient("#cce0ff", "#0033cc", max(len(top_ids), 1))

    for i, fid in enumerate(bottom_ids):
        colors[fid] = reds[i]
    for i, fid in enumerate(top_ids):
        colors[fid] = blues[i]
    for fid in other_ids:
        colors[fid] = "#666666"

    return colors


def compute_family_colors_from_state() -> Dict[str, str]:
    raw = st.session_state.get("rebar_families", [])
    bottom_ids = [d["id"] for d in raw if _position_group(d["position"]) == "bottom"]
    top_ids = [d["id"] for d in raw if _position_group(d["position"]) == "top"]
    other_ids = [d["id"] for d in raw if _position_group(d["position"]) == "other"]

    colors: Dict[str, str] = {}

    reds = _gradient("#ffcccc", "#b30000", max(len(bottom_ids), 1))
    blues = _gradient("#cce0ff", "#0033cc", max(len(top_ids), 1))

    for i, fid in enumerate(bottom_ids):
        colors[fid] = reds[i]
    for i, fid in enumerate(top_ids):
        colors[fid] = blues[i]
    for fid in other_ids:
        colors[fid] = "#666666"

    return colors


def compute_layer_indices(families: List[RebarFamily]) -> Dict[str, int]:
    layer_by_id: Dict[str, int] = {}
    by_pos: Dict[str, List[RebarFamily]] = {}
    for f in families:
        by_pos.setdefault(f.position, []).append(f)
    for pos, flist in by_pos.items():
        for idx, f in enumerate(flist):
            layer_by_id[f.id] = idx
    return layer_by_id


def compute_family_codes(raw_families: List[Dict]) -> Dict[str, str]:
    bottom_ids = [d["id"] for d in raw_families if _position_group(d["position"]) == "bottom"]
    top_ids = [d["id"] for d in raw_families if _position_group(d["position"]) == "top"]
    other_ids = [d["id"] for d in raw_families if _position_group(d["position"]) == "other"]

    codes: Dict[str, str] = {}

    for i, fid in enumerate(bottom_ids, start=1):
        codes[fid] = f"INF{i}"
    for i, fid in enumerate(top_ids, start=1):
        codes[fid] = f"SUP{i}"
    for i, fid in enumerate(other_ids, start=1):
        codes[fid] = f"LAT{i}"

    return codes


# =========================
# Dessins matplotlib
# =========================

def _draw_concrete(ax, geom: BeamGeometry):
    pts = geom.concrete_polygon()
    poly = Polygon(pts, closed=True, fill=False, linewidth=1.8, edgecolor="black")
    ax.add_patch(poly)


def _draw_stirrup(ax, geom: BeamGeometry, st_type: Optional[StirrupType]):
    if st_type is None:
        return

    c = geom.cover
    b = geom.b_web
    h = geom.h_total
    phi = st_type.phi_mm
    line_w = max(phi / 10.0, 0.8)

    xL = -b / 2.0 + c
    xR = b / 2.0 - c
    yB = 0.0 + c
    yT = h - c

    hook_len = 8.0 * phi / 10.0  # cm

    verts = [
        (xL, yB),
        (xL, yT),
        (xL, yT + hook_len),
        (xL + hook_len, yT),
        (xR - hook_len, yT),
        (xR, yT + hook_len),
        (xR, yT),
        (xR, yB),
        (xL, yB),
    ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    path = Path(verts, codes)
    patch = PathPatch(path, fill=False, linewidth=line_w, edgecolor="red", linestyle="-")
    ax.add_patch(patch)


def draw_section(
    section: BeamSection,
    stirrup_type: Optional[StirrupType] = None,
    highlight_family_ids: Optional[List[str]] = None,
):
    geom = section.geometry
    b = geom.b_web
    h = geom.h_total

    fig, ax = plt.subplots(figsize=(2.5, 3.5))

    _draw_concrete(ax, geom)
    _draw_stirrup(ax, geom, stirrup_type)

    for bar in section.rebars:
        r_cm = bar.phi_mm / 20.0
        if highlight_family_ids and bar.family_id in highlight_family_ids:
            facecolor = "#ff6600"
        else:
            facecolor = bar.color
        circ = plt.Circle((bar.x, bar.y), r_cm, color=facecolor)
        ax.add_patch(circ)

    ax.set_aspect("equal", "box")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")

    margin = max(b, h) * 0.20
    ax.set_xlim(-b / 2.0 - margin, b / 2.0 + margin)
    ax.set_ylim(-margin, h + margin)
    ax.grid(True, linestyle=":", linewidth=0.5)

    fig.tight_layout()
    return fig


def draw_section_for_zone(
    section: BeamSection,
    stirrup_type: Optional[StirrupType] = None,
):
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

    st.session_state.setdefault("shape_section", "Rectangulaire")

    st.session_state.setdefault("steel_grade", "B500A")

    st.session_state.setdefault("rebar_families", [])
    st.session_state.setdefault("stirrup_types", [])
    st.session_state.setdefault("stirrup_zones", [])

    st.session_state.setdefault("family_counter", 0)
    st.session_state.setdefault("st_type_counter", 0)
    st.session_state.setdefault("st_zone_counter", 0)


def build_geometry() -> BeamGeometry:
    shape = st.session_state.shape_section

    if "gauche" in shape:
        flange_left_b = st.session_state.flange_left_b_cm
        flange_left_h = st.session_state.flange_left_h_cm
    else:
        flange_left_b = 0.0
        flange_left_h = 0.0

    if "droit" in shape:
        flange_right_b = st.session_state.flange_right_b_cm
        flange_right_h = st.session_state.flange_right_h_cm
    else:
        flange_right_b = 0.0
        flange_right_h = 0.0

    return BeamGeometry(
        b_web=st.session_state.b_web_cm,
        h_total=st.session_state.h_total_cm,
        cover=st.session_state.cover_cm,
        flange_left_b=flange_left_b,
        flange_left_h=flange_left_h,
        flange_right_b=flange_right_b,
        flange_right_h=flange_right_h,
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
                with_hooks=d.get("with_hooks", False),
                hook_height_cm=d.get("hook_height_cm", 0.0),
                hook_mode=d.get("hook_mode", "none"),
                with_lap=d.get("with_lap", False),
                lap_length_mm=d.get("lap_length_mm", 0.0),
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

def _new_family_id(role: FamilyRole) -> str:
    st.session_state.family_counter += 1
    return f"F{st.session_state.family_counter}"


def add_family(role: FamilyRole):
    fam_id = _new_family_id(role)
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
            "hook_mode": "none",
            "with_lap": False,
            "lap_length_mm": 0.0,
            "x_start_pct": 0.0,
            "x_end_pct": 100.0,
        }
    )


def edit_family(idx: int, fam: Dict, role_label: str, code: str, color: Optional[str] = None):
    c1, c2 = st.columns([3, 1])
    with c1:
        if color:
            badge = (
                f"<span style='display:inline-block;width:14px;height:14px;"
                f"border-radius:50%;background-color:{color};"
                f"margin-right:6px;vertical-align:middle;'></span>"
            )
        else:
            badge = ""
        st.markdown(
            f"{badge}<strong>{role_label} – {code}</strong>",
            unsafe_allow_html=True,
        )
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
        # Diamètres prédéfinis
        current_dia = int(fam["dia_mm"]) if int(fam["dia_mm"]) in ALLOWED_DIA_MM else 16
        fam["dia_mm"] = st.selectbox(
            "ϕ (mm)",
            ALLOWED_DIA_MM,
            index=ALLOWED_DIA_MM.index(current_dia),
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
        hook_mode_label_map = {
            "Aucun retour": "none",
            "Deux retours": "both",
            "Retour côté début": "left",
            "Retour côté fin": "right",
        }
        inv_map = {v: k for k, v in hook_mode_label_map.items()}
        current_mode = fam.get("hook_mode", "none")
        label_default = inv_map.get(current_mode, "Aucun retour")

        label_choice = st.selectbox(
            "Retours verticaux",
            list(hook_mode_label_map.keys()),
            index=list(hook_mode_label_map.keys()).index(label_default),
            key=f"hook_mode_{idx}",
        )
        fam["hook_mode"] = hook_mode_label_map[label_choice]
        fam["with_hooks"] = fam["hook_mode"] != "none"

    with c7:
        default_h = max(
            st.session_state.h_total_cm - 2 * st.session_state.cover_cm,
            0.0,
        )
        if fam["hook_mode"] != "none":
            if fam.get("hook_height_cm", 0.0) <= 0.0:
                fam["hook_height_cm"] = default_h
            fam["hook_height_cm"] = st.number_input(
                "Hauteur retour (cm)",
                min_value=0.0,
                max_value=1000.0,
                value=float(fam["hook_height_cm"]),
                step=1.0,
                key=f"h_hook_{idx}",
            )
        else:
            fam["hook_height_cm"] = 0.0
            st.write("Hauteur retour (cm) : —")

    with c8:
        fam["with_lap"] = st.checkbox(
            "Recouvrement automatique",
            value=bool(fam.get("with_lap", False)),
            key=f"lap_{idx}",
        )
        if fam["with_lap"]:
            fam["lap_length_mm"] = st.number_input(
                "ℓ_rec (mm)",
                min_value=0.0,
                max_value=5000.0,
                value=float(fam.get("lap_length_mm") or 60.0 * fam["dia_mm"]),
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
    name = f"E{st.session_state.st_type_counter}"
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
            "Nom du type (E1, E2, ...)",
            value=d["name"],
            key=f"st_name_{idx}",
        )
    with c2:
        if st.button("🗑️", key=f"del_st_type_{idx}"):
            st.session_state.stirrup_types.pop(idx)
            st.rerun()

    c3, c4, c5 = st.columns(3)
    with c3:
        current_dia = int(d["phi_mm"]) if int(d["phi_mm"]) in ALLOWED_DIA_MM else 8
        d["phi_mm"] = st.selectbox(
            "ϕ étrier (mm)",
            ALLOWED_DIA_MM,
            index=ALLOWED_DIA_MM.index(current_dia),
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
    """Page Streamlit : Poutre BA – Modélisation et taux d’armature."""
    _init_state()

    st.title("🧱 Poutre BA – Modélisation et taux d’armature")

    col_left, col_right = st.columns(2)

    # ==============
    # COLONNE GAUCHE
    # ==============
    with col_left:
        # ---- Données
        st.subheader("Données")

        # Ligne 1 : type de section + enrobage
        c_sec, c_cov = st.columns([2, 1])
        with c_sec:
            st.selectbox(
                "Section",
                [
                    "Rectangulaire",
                    "Rectangulaire + talon gauche",
                    "Rectangulaire + talon droit",
                    "Rectangulaire + deux talons",
                ],
                key="shape_section",
            )
        with c_cov:
            st.number_input(
                "Enrobage c (cm)",
                min_value=1.0,
                max_value=10.0,
                value=float(st.session_state.cover_cm),
                step=0.5,
                key="cover_cm",
            )

        # Ligne 2 : b, h, L
        c_dim1, c_dim2, c_dim3 = st.columns(3)
        with c_dim1:
            st.number_input(
                "Largeur b (cm)",
                min_value=5.0,
                max_value=200.0,
                value=float(st.session_state.b_web_cm),
                step=1.0,
                key="b_web_cm",
            )
        with c_dim2:
            st.number_input(
                "Hauteur h (cm)",
                min_value=10.0,
                max_value=300.0,
                value=float(st.session_state.h_total_cm),
                step=1.0,
                key="h_total_cm",
            )
        with c_dim3:
            st.number_input(
                "Portée L (m)",
                min_value=0.5,
                max_value=60.0,
                value=float(st.session_state.L_beam_m),
                step=0.5,
                key="L_beam_m",
            )

        # Talons selon type de section
        if "gauche" in st.session_state.shape_section or "deux talons" in st.session_state.shape_section:
            cg1, cg2 = st.columns(2)
            with cg1:
                st.number_input(
                    "Talon gauche – largeur (cm)",
                    min_value=0.0,
                    max_value=300.0,
                    value=float(st.session_state.flange_left_b_cm),
                    step=1.0,
                    key="flange_left_b_cm",
                )
            with cg2:
                st.number_input(
                    "Talon gauche – hauteur (cm)",
                    min_value=0.0,
                    max_value=300.0,
                    value=float(st.session_state.flange_left_h_cm),
                    step=1.0,
                    key="flange_left_h_cm",
                )

        if "droit" in st.session_state.shape_section or "deux talons" in st.session_state.shape_section:
            cd1, cd2 = st.columns(2)
            with cd1:
                st.number_input(
                    "Talon droit – largeur (cm)",
                    min_value=0.0,
                    max_value=300.0,
                    value=float(st.session_state.flange_right_b_cm),
                    step=1.0,
                    key="flange_right_b_cm",
                )
            with cd2:
                st.number_input(
                    "Talon droit – hauteur (cm)",
                    min_value=0.0,
                    max_value=300.0,
                    value=float(st.session_state.flange_right_h_cm),
                    step=1.0,
                    key="flange_right_h_cm",
                )

        st.markdown("---")

        # ---- Armatures
        st.subheader("Armatures")

        # Ligne de boutons d'ajout
        b_main, b_reinf, b_lat, b_st = st.columns(4)
        with b_main:
            if st.button("➕ Lit principal"):
                add_family("main")
        with b_reinf:
            if st.button("➕ Renfort local"):
                add_family("reinforcement")
        with b_lat:
            if st.button("➕ Latéral"):
                add_family("lateral")
        with b_st:
            if st.button("➕ Étrier"):
                # à chaque clic on crée un nouveau type + zone
                add_stirrup_type()
                add_stirrup_zone()

        family_colors_ui = compute_family_colors_from_state()
        family_codes_ui = compute_family_codes(st.session_state.rebar_families)

        # Lits principaux
        mains = [f for f in st.session_state.rebar_families if f["role"] == "main"]
        if mains:
            st.markdown("**Lits principaux**")
            for idx, fam in enumerate(list(st.session_state.rebar_families)):
                if fam["role"] == "main":
                    with st.container(border=True):
                        code = family_codes_ui.get(fam["id"], fam["id"])
                        edit_family(
                            idx,
                            fam,
                            "Lit principal",
                            code=code,
                            color=family_colors_ui.get(fam["id"]),
                        )

        # Renforts
        reinfs = [f for f in st.session_state.rebar_families if f["role"] == "reinforcement"]
        if reinfs:
            st.markdown("**Renforts locaux**")
            for idx, fam in enumerate(list(st.session_state.rebar_families)):
                if fam["role"] == "reinforcement":
                    with st.container(border=True):
                        code = family_codes_ui.get(fam["id"], fam["id"])
                        edit_family(
                            idx,
                            fam,
                            "Renfort local",
                            code=code,
                            color=family_colors_ui.get(fam["id"]),
                        )

        # Barres latérales
        lats = [f for f in st.session_state.rebar_families if f["role"] == "lateral"]
        if lats:
            st.markdown("**Barres latérales**")
            for idx, fam in enumerate(list(st.session_state.rebar_families)):
                if fam["role"] == "lateral":
                    with st.container(border=True):
                        code = family_codes_ui.get(fam["id"], fam["id"])
                        edit_family(
                            idx,
                            fam,
                            "Barres latérales",
                            code=code,
                            color=family_colors_ui.get(fam["id"]),
                        )

        st.markdown("---")

        # Étriers (édition)
        if st.session_state.stirrup_types or st.session_state.stirrup_zones:
            st.markdown("**Étriers (types et zones)**")

        if st.session_state.stirrup_types:
            st.markdown("Types d’étriers")
            for i, d in enumerate(list(st.session_state.stirrup_types)):
                with st.container(border=True):
                    edit_stirrup_type(i, d)

        if st.session_state.stirrup_zones:
            st.markdown("Zones d’étriers (%L)")
            for i, d in enumerate(list(st.session_state.stirrup_zones)):
                with st.container(border=True):
                    edit_stirrup_zone(i, d)

    # ===============
    # COLONNE DROITE
    # ===============
    with col_right:
        st.subheader("Résultats")

        geom = build_geometry()
        families = build_rebar_families()
        stirrup_types = build_stirrup_types()
        stirrup_zones = build_stirrup_zones()

        family_codes = compute_family_codes(st.session_state.rebar_families)
        family_colors_plot = compute_family_colors_for_families(families)
        layer_by_id = compute_layer_indices(families)

        section = BeamSection(geometry=geom)
        section.regenerate_rebars(families, family_colors_plot, layer_by_id)

        stirrup_results = compute_stirrups_for_zones(
            stirrup_zones,
            stirrup_types,
            L_beam_m=st.session_state.L_beam_m,
            geom=geom,
        )

        qres = compute_quantities(
            geom=geom,
            L_beam_m=st.session_state.L_beam_m,
            families=families,
            stirrup_results=stirrup_results,
        )

        # Majoration
        st.markdown("**Paramètres acier**")
        c_steel1, c_steel2 = st.columns(2)
        with c_steel1:
            st.session_state.steel_grade = st.selectbox(
                "Qualité d’acier",
                ["B500A", "B500B", "B500C"],
                index=["B500A", "B500B", "B500C"].index(st.session_state.steel_grade),
            )
        with c_steel2:
            maj_pct = st.number_input(
                "Majoration globale (%)",
                min_value=0.0,
                max_value=50.0,
                value=5.0,
                step=1.0,
            )

        mass_steel_majorated = qres.mass_steel_kg * (1.0 + maj_pct / 100.0)
        kg_m3_majorated = mass_steel_majorated / qres.Vc_m3 if qres.Vc_m3 > 0 else 0.0

        c1, c2, c3 = st.columns(3)
        c1.metric("Volume béton Vc", f"{qres.Vc_m3:.3f} m³")
        c2.metric("Masse acier (maj.)", f"{mass_steel_majorated:.1f} kg")
        c3.metric("Taux d’armature", f"{kg_m3_majorated:.1f} kg/m³")

        with st.expander("Détail par catégorie (volume d’acier)"):
            st.write(
                {
                    "Lits (bas/haut + renforts)": f"{qres.by_category_m3['main'] + qres.by_category_m3['reinforcement']:.4f} m³",
                    "Renforts seuls": f"{qres.by_category_m3['reinforcement']:.4f} m³",
                    "Barres latérales": f"{qres.by_category_m3['lateral']:.4f} m³",
                    "Étriers": f"{qres.by_category_m3['stirrups']:.4f} m³",
                }
            )

        # Tableau récapitulatif armatures
        st.markdown("### Tableau récapitulatif des armatures")

        rows = []
        L_beam = st.session_state.L_beam_m

        for fam in families:
            code = family_codes.get(fam.id, fam.id)
            L_tot = fam.total_bar_length_m(L_beam)
            m_lin = steel_linear_mass_kg_m(fam.dia_mm)
            m_tot = L_tot * m_lin * (1.0 + maj_pct / 100.0)

            rows.append(
                {
                    "Code": code,
                    "Position": POSITION_LABELS.get(fam.position, fam.position),
                    "ϕ (mm)": int(fam.dia_mm),
                    "n barres": int(fam.n_bars),
                    "L active (m)": round(fam.active_length_m(L_beam), 2),
                    "L tot (m)": round(L_tot, 2),
                    "kg/m": round(m_lin, 3),
                    "kg tot (maj.)": round(m_tot, 2),
                }
            )

        for res in stirrup_results:
            m_lin = steel_linear_mass_kg_m(res.stirrup_type.phi_mm)
            m_tot = res.total_length_m * m_lin * (1.0 + maj_pct / 100.0)
            rows.append(
                {
                    "Code": res.stirrup_type.name,
                    "Position": f"Zone {res.zone.name}",
                    "ϕ (mm)": int(res.stirrup_type.phi_mm),
                    "n barres": int(res.n_stirrups),
                    "L active (m)": round(res.length_per_stirrup_m, 2),
                    "L tot (m)": round(res.total_length_m, 2),
                    "kg/m": round(m_lin, 3),
                    "kg tot (maj.)": round(m_tot, 2),
                }
            )

        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, hide_index=True)
        else:
            st.info("Ajoute des armatures pour voir le tableau récapitulatif.")

        st.markdown("**Section générale (tous lits)**")
        fig_gen = draw_section(section)
        st.pyplot(fig_gen, use_container_width=False)

        if stirrup_results:
            st.markdown("**Sections par zone d’étriers**")
            for res in stirrup_results:
                st.markdown(
                    f"Zone **{res.zone.name}** : {res.zone.x_start_pct:.1f}–{res.zone.x_end_pct:.1f} %L  "
                    f"({res.n_stirrups} étriers, s = {res.zone.spacing_cm:.1f} cm, type {res.stirrup_type.name})"
                )
                fig_z = draw_section_for_zone(section, stirrup_type=res.stirrup_type)
                st.pyplot(fig_z, use_container_width=False)

        reinf_fams = [f for f in families if f.role == "reinforcement"]
        if reinf_fams:
            st.markdown("**Sections pour renforts locaux (barres mises en évidence)**")
            for f in reinf_fams:
                code = family_codes.get(f.id, f.id)
                st.caption(
                    f"Renfort {code} – {POSITION_LABELS.get(f.position, f.position)} : "
                    f"{f.n_bars}Ø{f.dia_mm} de {f.x_start_pct:.0f} à {f.x_end_pct:.0f} %L"
                )
                fig_r = draw_section(section, highlight_family_ids=[f.id])
                st.pyplot(fig_r, use_container_width=False)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Poutre BA – Ferraillage complet",
        page_icon="🧱",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    show()
