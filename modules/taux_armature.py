# modules/taux_armature.py
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Literal, Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, PathPatch
from matplotlib.path import Path
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

STEEL_DENSITY_KG_M3 = 7850.0  # kg/m³
CONCRETE_DENSITY_KG_M3 = 2500.0  # kg/m³ (≈ 25 kN/m³)

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
            # barre latérale gauche : on placera verticalement
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

        # lits horizontaux (bas, haut, talons)
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
    hook_mode: str = "none"  # "none", "both", "left", "right"

    with_lap: bool = False
    lap_length_mm: float = 0.0  # ℓ_rec (mm)

    x_start_pct: float = 0.0    # 0–100 %
    x_end_pct: float = 100.0    # 0–100 %

    both_sides: bool = False    # pour barres latérales : 2 côtés (gauche + droite)

    def active_length_m(self, L_beam_m: float) -> float:
        """Longueur active horizontale en m (avec correction enrobages pour 0–100 %)."""
        x1 = max(0.0, min(100.0, self.x_start_pct))
        x2 = max(0.0, min(100.0, self.x_end_pct))
        if x2 <= x1:
            return 0.0

        L = L_beam_m * (x2 - x1) / 100.0

        # Correction enrobage uniquement si la barre court sur toute la portée
        cover_cm = st.session_state.get("cover_cm", 0.0)
        if x1 <= 1e-3 and x2 >= 100.0 - 1e-3:
            L = max(L - 2.0 * cover_cm / 100.0, 0.0)

        return L

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

        # Partie horizontale avec ou sans recouvrement
        if not self.with_lap or self.lap_length_mm <= 0.0:
            straight_total = L_active * self.n_bars
            n_stock_bars = 1
        else:
            lap_m = self.lap_length_mm / 1000.0
            usable = max(stock_length_m - lap_m, 1e-6)

            N = math.ceil((L_active + lap_m) / usable)  # nb de barres stock par "ligne"
            straight_one = L_active + (N - 1) * lap_m
            straight_total = straight_one * self.n_bars
            n_stock_bars = N

        # Longueur supplémentaire due aux retours
        extra_hooks_total = hooks_per_bar * hook_height_m * self.n_bars * n_stock_bars

        # Barres latérales sur 2 côtés → longueur doublée
        factor_side = 2.0 if (self.role == "lateral" and self.both_sides) else 1.0

        return (straight_total + extra_hooks_total) * factor_side

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
    bend_radius_mm: float = 0.0


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
    Longueur d’un étrier en m :
    contour de l’âme à l’enrobage + petits crochets.
    """
    c = geom.cover
    b_cm = geom.b_web - 2 * c
    h_cm = geom.h_total - 2 * c

    b_mm = b_cm * 10.0
    h_mm = h_cm * 10.0

    perim_mm = 2.0 * (b_mm + h_mm)

    phi = st_type.phi_mm
    hook_mm = 8.0 * phi * 2  # deux crochets simplifiés

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

        # nombre d’étriers (incluant celui du début de zone)
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
        """
        Reconstruit la liste de RebarInstance à partir des familles.
        """
        self.rebars.clear()
        color_map = family_colors or {}
        layer_map = layer_by_id or {}
        for fam in families:
            layer_index = layer_map.get(fam.id, 0)

            # latérales sur deux côtés
            if fam.role == "lateral" and fam.both_sides:
                centers_left = self.geometry.bar_centers(
                    "side_left",
                    fam.n_bars,
                    fam.dia_mm,
                    layer_index=layer_index,
                )
                centers_right = self.geometry.bar_centers(
                    "side_right",
                    fam.n_bars,
                    fam.dia_mm,
                    layer_index=layer_index,
                )
                centers = centers_left + centers_right
            else:
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
    """Volume (m³) d’acier = L × aire."""
    r_m = dia_mm / 1000.0 / 2.0
    area_m2 = math.pi * r_m**2
    return length_m * area_m2


def steel_linear_mass_kg_m(dia_mm: float) -> float:
    """Masse linéique (kg/m) d’une barre de diamètre dia_mm."""
    r_m = dia_mm / 1000.0 / 2.0
    area_m2 = math.pi * r_m**2
    return area_m2 * STEEL_DENSITY_KG_M3


def compute_quantities(
    geom: BeamGeometry,
    L_beam_m: float,
    families: List[RebarFamily],
    stirrup_results: List[StirrupResult],
) -> QuantitiesResult:
    """Volumes béton / acier et taux d’armatures global + par catégorie."""
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
# Outils couleurs & codes
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
    """Classe grossière des lits pour les couleurs (bas / haut / autres)."""
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
        colors[fid] = "#666666"  # gris pour latérales / autres

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
    """
    Attribue un index de lit (0,1,2,...) par position.
    INF1 = layer 0, INF2 = layer 1, etc.
    """
    layer_by_id: Dict[str, int] = {}
    by_pos: Dict[str, List[RebarFamily]] = {}
    for f in families:
        by_pos.setdefault(f.position, []).append(f)
    for pos, flist in by_pos.items():
        for idx, f in enumerate(flist):
            layer_by_id[f.id] = idx
    return layer_by_id


def compute_family_codes(raw_families: List[Dict]) -> Dict[str, str]:
    """
    Génère des codes INF1, SUP1, LAT1, ... à partir des positions.
    """
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
    line_w = max(phi / 10.0, 0.8)  # épaisseur à l’écran

    # Rectangle intérieur à l’enrobage
    xL = -b / 2.0 + c
    xR = b / 2.0 - c
    yB = 0.0 + c
    yT = h - c

    hook_len = 8.0 * phi / 10.0  # en cm (≈ 8ϕ)

    # Path avec crochets simples vers l’intérieur
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
    """
    Dessin d’une section :
    - contour béton (avec talons)
    - toutes les barres longitudinales (couleurs par lit)
    - étrier (épaisseur liée à ϕ, crochets simplifiés)
    - éventuellement mise en évidence d’une famille (renfort) en orange
    """
    geom = section.geometry
    b = geom.b_web
    h = geom.h_total

    fig, ax = plt.subplots(figsize=(2.5, 3.5))

    # Béton
    _draw_concrete(ax, geom)

    # Étrier
    _draw_stirrup(ax, geom, stirrup_type)

    # Barres
    for bar in section.rebars:
        r_cm = bar.phi_mm / 20.0  # ϕ (mm) → rayon (cm)
        if highlight_family_ids and bar.family_id in highlight_family_ids:
            facecolor = "#ff6600"  # orange pour renfort sélectionné
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
    """Alias : une vue de section par zone d’étriers."""
    return draw_section(section, stirrup_type=stirrup_type)


# =========================
# Helpers Streamlit / état
# =========================

POSITION_LABELS = {
    "web_bottom": "Armatures inférieures",
    "web_top": "Armatures supérieures",
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

POSITION_OPTIONS_MAIN_BASE = [
    ("Armatures inférieures (âme)", "web_bottom"),
    ("Armatures supérieures (âme)", "web_top"),
    ("Talon gauche", "flange_left"),
    ("Talon droit", "flange_right"),
]

POSITION_OPTIONS_LATERAL = [
    ("Latéral gauche", "side_left"),
    ("Latéral droit", "side_right"),
    ("Deux côtés", "both_sides"),
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
    st.session_state.setdefault("length_mode", "%L")  # "%L" ou "m"

    st.session_state.setdefault("rebar_families", [])   # liste de dict
    st.session_state.setdefault("stirrup_types", [])    # liste de dict
    st.session_state.setdefault("stirrup_zones", [])    # liste de dict

    st.session_state.setdefault("family_counter", 0)
    st.session_state.setdefault("st_type_counter", 0)
    st.session_state.setdefault("st_zone_counter", 0)


def build_geometry() -> BeamGeometry:
    shape = st.session_state.shape_section

    # Talons selon le type choisi
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
                both_sides=d.get("both_sides", False),
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


def add_family(role: FamilyRole, default_position: Optional[PositionType] = None):
    fam_id = _new_family_id(role)
    if role == "lateral":
        pos = "side_left"
    else:
        pos = default_position or ("web_bottom" if role == "main" else "web_top")

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
            "both_sides": False,
        }
    )


def edit_family(idx: int, fam: Dict, role_label: str, code: str, color: Optional[str] = None):
    # Bandeau titre + poubelle
    c1, c2 = st.columns([8, 1])
    with c1:
        if color:
            badge = (
                f"<span style='display:inline-block;width:12px;height:12px;"
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
        if st.button("🗑️", key=f"del_fam_{idx}"):
            st.session_state.rebar_families.pop(idx)
            st.rerun()

    # Ligne unique : Position ; n barres ; Ø ; retour ; h_retour ; De ; À
    cpos, c_nb, c_dia, c_ret, c_h, c_de, c_a = st.columns([2.2, 0.8, 0.9, 0.9, 1.2, 0.8, 0.8])

    length_mode = st.session_state.get("length_mode", "%L")
    L_beam = st.session_state.L_beam_m

    # --- Position
    with cpos:
        if fam["role"] == "lateral":
            labels = [o[0] for o in POSITION_OPTIONS_LATERAL]
            values = [o[1] for o in POSITION_OPTIONS_LATERAL]
            current_value = "both_sides" if fam.get("both_sides") else fam["position"]
            if current_value not in values:
                current_value = "side_left"
            i_sel = values.index(current_value)
            pos_label = st.selectbox(
                "Position",
                labels,
                index=i_sel,
                key=f"pos_{idx}",
            )
            sel_val = values[labels.index(pos_label)]
            if sel_val == "both_sides":
                fam["both_sides"] = True
                fam["position"] = "side_left"
            else:
                fam["both_sides"] = False
                fam["position"] = sel_val
        else:
            # options talons uniquement si activés dans la géométrie
            shape = st.session_state.shape_section
            opts = []
            for label, val in POSITION_OPTIONS_MAIN_BASE:
                if "Talon gauche" in label and "gauche" not in shape:
                    continue
                if "Talon droit" in label and "droit" not in shape:
                    continue
                opts.append((label, val))
            if not opts:
                opts = POSITION_OPTIONS_MAIN_BASE[:2]

            labels = [o[0] for o in opts]
            values = [o[1] for o in opts]
            current_value = fam["position"]
            if current_value not in values:
                current_value = "web_bottom"
            i_sel = values.index(current_value)
            pos_label = st.selectbox(
                "Position",
                labels,
                index=i_sel,
                key=f"pos_{idx}",
            )
            fam["position"] = values[labels.index(pos_label)]

    # --- n barres
    with c_nb:
        fam["n_bars"] = st.number_input(
            "n barres",
            min_value=1,
            max_value=20,
            value=int(fam["n_bars"]),
            step=1,
            key=f"nb_{idx}",
        )

    # --- Ø
    with c_dia:
        dia_options = [6, 8, 10, 12, 16, 20, 25, 32, 40]
        if int(fam["dia_mm"]) not in dia_options:
            dia_options.append(int(fam["dia_mm"]))
            dia_options.sort()
        i_dia = dia_options.index(int(fam["dia_mm"]))
        sel = st.selectbox(
            "Ø (mm)",
            dia_options,
            index=i_dia,
            key=f"phi_{idx}",
        )
        fam["dia_mm"] = float(sel)

    # --- Retour (0 / 1 / 2)
    with c_ret:
        mode_map = {"none": 0, "left": 1, "right": 1, "both": 2}
        inv_map = {0: "none", 1: "left", 2: "both"}
        current = mode_map.get(fam.get("hook_mode", "none"), 0)
        nb_ret = st.selectbox(
            "Retour",
            [0, 1, 2],
            index=[0, 1, 2].index(current),
            key=f"ret_{idx}",
        )
        fam["hook_mode"] = inv_map[nb_ret]
        fam["with_hooks"] = nb_ret != 0

    # --- Hauteur de retour
    with c_h:
        if fam["hook_mode"] != "none":
            if fam["role"] == "lateral":
                default_h = max(
                    st.session_state.b_web_cm - 2 * st.session_state.cover_cm,
                    0.0,
                )
            else:
                default_h = max(
                    st.session_state.h_total_cm - 2 * st.session_state.cover_cm,
                    0.0,
                )
            if fam.get("hook_height_cm", 0.0) <= 0.0:
                fam["hook_height_cm"] = default_h
            fam["hook_height_cm"] = st.number_input(
                "h retour (cm)",
                min_value=0.0,
                max_value=1000.0,
                value=float(fam["hook_height_cm"]),
                step=1.0,
                key=f"h_hook_{idx}",
            )
        else:
            fam["hook_height_cm"] = 0.0
            st.write("h retour (cm) : —")

    # --- De / À
    with c_de:
        if length_mode == "%L" or L_beam <= 0:
            fam["x_start_pct"] = st.number_input(
                "De",
                min_value=0.0,
                max_value=100.0,
                value=float(fam["x_start_pct"]),
                step=5.0,
                key=f"x1_{idx}",
            )
        else:
            x_start_m = L_beam * fam["x_start_pct"] / 100.0
            x_start_m = st.number_input(
                "De (m)",
                min_value=0.0,
                max_value=float(L_beam),
                value=float(x_start_m),
                step=max(L_beam / 20.0, 0.1),
                key=f"x1m_{idx}",
            )
            fam["x_start_pct"] = 100.0 * x_start_m / L_beam if L_beam > 0 else 0.0

    with c_a:
        if length_mode == "%L" or L_beam <= 0:
            fam["x_end_pct"] = st.number_input(
                "À",
                min_value=0.0,
                max_value=100.0,
                value=float(fam["x_end_pct"]),
                step=5.0,
                key=f"x2_{idx}",
            )
        else:
            x_end_m = L_beam * fam["x_end_pct"] / 100.0
            x_end_m = st.number_input(
                "À (m)",
                min_value=0.0,
                max_value=float(L_beam),
                value=float(x_end_m),
                step=max(L_beam / 20.0, 0.1),
                key=f"x2m_{idx}",
            )
            fam["x_end_pct"] = 100.0 * x_end_m / L_beam if L_beam > 0 else 0.0


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

    c3, c4 = st.columns(2)
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

    # Choix de la saisie des zones (discret en haut)
    c_mode1, c_mode2 = st.columns([5, 2])
    with c_mode2:
        mode_label = st.radio(
            "Saisie des longueurs",
            ["% de L", "m le long de L"],
            index=0 if st.session_state.length_mode == "%L" else 1,
            horizontal=True,
        )
        st.session_state.length_mode = "%L" if mode_label == "% de L" else "m"

    # colonnes 50 / 50
    col_left, col_right = st.columns(2)

    # ==============
    # COLONNE GAUCHE
    # ==============
    with col_left:
        # ---------------- Données ----------------
        st.subheader("Données")

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

        c_dim4, c_dim5 = st.columns(2)
        with c_dim4:
            st.number_input(
                "Enrobage nominal c (cm)",
                min_value=1.0,
                max_value=10.0,
                value=float(st.session_state.cover_cm),
                step=0.5,
                key="cover_cm",
            )
        with c_dim5:
            st.session_state.shape_section = st.selectbox(
                "Section",
                [
                    "Rectangulaire",
                    "Rectangulaire + talon gauche",
                    "Rectangulaire + talon droit",
                    "Rectangulaire + deux talons",
                ],
                index=[
                    "Rectangulaire",
                    "Rectangulaire + talon gauche",
                    "Rectangulaire + talon droit",
                    "Rectangulaire + deux talons",
                ].index(st.session_state.shape_section),
            )

        # Talons visibles selon le type
        if "gauche" in st.session_state.shape_section:
            cg1, cg2 = st.columns(2)
            with cg1:
                st.number_input(
                    "Talon gauche – largeur (cm)",
                    min_value=0.0,
                    max_value=300.0,
                    value=float(st.session_state.flange_left_b_cm or 10.0),
                    step=5.0,
                    key="flange_left_b_cm",
                )
            with cg2:
                st.number_input(
                    "Talon gauche – hauteur (cm)",
                    min_value=0.0,
                    max_value=300.0,
                    value=float(st.session_state.flange_left_h_cm or 15.0),
                    step=5.0,
                    key="flange_left_h_cm",
                )

        if "droit" in st.session_state.shape_section:
            cd1, cd2 = st.columns(2)
            with cd1:
                st.number_input(
                    "Talon droit – largeur (cm)",
                    min_value=0.0,
                    max_value=300.0,
                    value=float(st.session_state.flange_right_b_cm or 10.0),
                    step=5.0,
                    key="flange_right_b_cm",
                )
            with cd2:
                st.number_input(
                    "Talon droit – hauteur (cm)",
                    min_value=0.0,
                    max_value=300.0,
                    value=float(st.session_state.flange_right_h_cm or 15.0),
                    step=5.0,
                    key="flange_right_h_cm",
                )

        # Couleurs & codes pour les familles (pour l'UI)
        family_colors_ui = compute_family_colors_from_state()
        family_codes_ui = compute_family_codes(st.session_state.rebar_families)

        # ---------------- Armatures ----------------
        st.subheader("Armatures")

        # Boutons d’ajout sur une seule ligne
        c_btn1, c_btn2, c_btn3, c_btn4, c_btn5 = st.columns(5)
        with c_btn1:
            if st.button("➕ Lit principal"):
                add_family("main")
        with c_btn2:
            if st.button("➕ Renfort local"):
                add_family("reinforcement")
        with c_btn3:
            if st.button("➕ Latéral"):
                add_family("lateral")
        with c_btn4:
            if st.button("➕ Étrier"):
                if not st.session_state.stirrup_types:
                    add_stirrup_type()
                add_stirrup_zone()
        with c_btn5:
            if "talon" in st.session_state.shape_section:
                if st.button("➕ Renfort talon"):
                    # renfort par défaut dans le premier talon actif
                    if "gauche" in st.session_state.shape_section:
                        add_family("reinforcement", default_position="flange_left")
                    else:
                        add_family("reinforcement", default_position="flange_right")

        # Lits principaux
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

        # Renforts locaux
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

        # Étriers (types + zones)
        if st.session_state.stirrup_types:
            st.markdown("**Types d’étriers**")
            for i, d in enumerate(list(st.session_state.stirrup_types)):
                with st.container(border=True):
                    edit_stirrup_type(i, d)

        if st.session_state.stirrup_zones:
            st.markdown("**Zones d’étriers (%L**
