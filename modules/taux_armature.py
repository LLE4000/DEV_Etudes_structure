# modules/taux_armature.py
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Literal, Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import streamlit as st


# =========================
# Types & constantes
# =========================

PositionType = Literal[
    "web_bottom", "web_top",
    "flange_left_bottom", "flange_left_top",
    "flange_right_bottom", "flange_right_top",
    "side_left", "side_right", "side_both",
]

FamilyRole = Literal["main", "reinforcement", "lateral"]

StirrupTypeEnum = Literal["full", "inner", "flange_left", "flange_right"]

STEEL_DENSITY_KG_M3 = 7850.0  # kg/m³
CONCRETE_DENSITY_KG_M3 = 2500.0  # masse volumique approx. du béton


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
        Polygone de béton (âme + talons inférieurs éventuels).
        Coordonnées en cm, origine au milieu de l’âme, fibre inf y = 0.
        Talons en bas, vers l’extérieur.
        """
        b = self.b_web
        h = self.h_total
        xL = -b / 2.0
        xR = b / 2.0

        has_left = self.flange_left_b > 0 and self.flange_left_h > 0
        has_right = self.flange_right_b > 0 and self.flange_right_h > 0

        XL_ext = xL - self.flange_left_b if has_left else xL
        XR_ext = xR + self.flange_right_b if has_right else xR

        pts: List[Tuple[float, float]] = []

        # départ bas gauche extérieur
        pts.append((XL_ext, 0.0))

        # talon gauche (marche)
        if has_left:
            pts.append((XL_ext, self.flange_left_h))
            pts.append((xL, self.flange_left_h))

        # monter âme gauche
        pts.append((xL, h))

        # haut âme droite
        pts.append((xR, h))

        # talon droit (marche)
        if has_right:
            pts.append((xR, self.flange_right_h))
            pts.append((XR_ext, self.flange_right_h))

        # bas droit extérieur
        pts.append((XR_ext, 0.0))

        # fermer sur bas gauche
        if XR_ext != XL_ext:
            pts.append((XL_ext, 0.0))

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
    def _horizontal_band(
        self, pos: PositionType
    ) -> Tuple[Optional[float], float, float]:
        """
        Retourne (y_face, x_min, x_max) pour la face béton la plus proche.
        y_face = coordonnée de la face béton (pas de l’axe).
        Si y_face is None → barres verticales (barres latérales).
        """
        c = self.cover
        b = self.b_web
        xL = -b / 2.0
        xR = b / 2.0

        if pos == "web_bottom":
            y_face = 0.0
            x_min = xL + c
            x_max = xR - c

        elif pos == "web_top":
            y_face = self.h_total
            x_min = xL + c
            x_max = xR - c

        elif pos == "flange_left_bottom":
            if self.flange_left_b <= 0 or self.flange_left_h <= 0:
                raise ValueError("Talon gauche non défini.")
            y_face = 0.0
            x_min = xL - self.flange_left_b + c
            x_max = xL - c

        elif pos == "flange_left_top":
            if self.flange_left_b <= 0 or self.flange_left_h <= 0:
                raise ValueError("Talon gauche non défini.")
            y_face = self.flange_left_h
            x_min = xL - self.flange_left_b + c
            x_max = xL - c

        elif pos == "flange_right_bottom":
            if self.flange_right_b <= 0 or self.flange_right_h <= 0:
                raise ValueError("Talon droit non défini.")
            y_face = 0.0
            x_min = xR + c
            x_max = xR + self.flange_right_b - c

        elif pos == "flange_right_top":
            if self.flange_right_b <= 0 or self.flange_right_h <= 0:
                raise ValueError("Talon droit non défini.")
            y_face = self.flange_right_h
            x_min = xR + c
            x_max = xR + self.flange_right_b - c

        elif pos in ("side_left", "side_right", "side_both"):
            # barres latérales verticales
            y_face = None
            if pos == "side_left":
                x_min = xL + c
                x_max = x_min
            elif pos == "side_right":
                x_max = xR - c
                x_min = x_max
            else:  # side_both (on utilisera les deux côtés plus tard)
                x_min = xL + c
                x_max = xR - c

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
        c = self.cover

        # Lits horizontaux
        if y_face is not None:
            r_cm = phi_mm / 20.0
            base_cover = c + r_cm
            pitch = 2.0 * r_cm + 2.0  # espacement vertical entre lits

            # bas / talon bas : on part du bas
            if pos in ("web_bottom", "flange_left_bottom", "flange_right_bottom"):
                y = y_face + base_cover + layer_index * pitch
            else:
                y = y_face - (base_cover + layer_index * pitch)

            if n_bars == 1:
                xs = [(x_min + x_max) / 2.0]
            else:
                xs = [
                    x_min + i * (x_max - x_min) / (n_bars - 1)
                    for i in range(n_bars)
                ]
            return [(x, y) for x in xs]

        # Barres latérales verticales
        y_min = c
        y_max = self.h_total - c

        if n_bars == 1:
            ys = [(y_min + y_max) / 2.0]
        else:
            # règle générale : on place n+2 barres, on retire les 2 extrêmes
            n_eff = n_bars + 2
            idxs = range(1, n_eff - 1)
            ys = [
                y_min + i * (y_max - y_min) / (n_eff - 1)
                for i in idxs
            ]

        xL = -self.b_web / 2.0 + c
        xR = self.b_web / 2.0 - c
        pts: List[Tuple[float, float]] = []

        if pos in ("side_left", "side_both"):
            pts.extend((xL, y) for y in ys)
        if pos in ("side_right", "side_both"):
            pts.extend((xR, y) for y in ys)

        return pts


# =========================
# Armatures longitudinales
# =========================

@dataclass
class RebarFamily:
    """
    Famille d’armatures longitudinales.
    Longueur active définie en % de la portée ou en m.
    """
    id: str
    role: FamilyRole
    position: PositionType

    n_bars: int
    dia_mm: float

    hook_count: int = 0          # 0, 1 ou 2 retours
    hook_height_cm: float = 0.0  # longueur de chaque retour (cm)

    x_start_pct: float = 0.0    # 0–100 %
    x_end_pct: float = 100.0    # 0–100 %

    def effective_n_bars(self) -> int:
        """Nombre réel de barres dans la famille (latérales 2 côtés)."""
        if self.position == "side_both":
            return self.n_bars * 2
        return self.n_bars

    def active_length_m(self, L_beam_m: float) -> float:
        """Longueur de la partie droite en m (sans retours)."""
        x1 = max(0.0, min(100.0, self.x_start_pct))
        x2 = max(0.0, min(100.0, self.x_end_pct))
        if x2 <= x1:
            return 0.0
        return L_beam_m * (x2 - x1) / 100.0

    def total_bar_length_m(self, L_beam_m: float) -> float:
        """
        Longueur totale d’acier pour cette famille, y compris retours.
        On ajoute hook_count × hook_height à chaque barre.
        """
        L_active = self.active_length_m(L_beam_m)
        if L_active <= 0.0 or self.n_bars <= 0:
            return 0.0

        n_eff = self.effective_n_bars()
        L_hooks = (self.hook_count * max(self.hook_height_cm, 0.0) / 100.0) * n_eff
        return L_active * n_eff + L_hooks

    def steel_area_mm2(self) -> float:
        """Section d’acier totale (mm²) de la famille."""
        return self.effective_n_bars() * math.pi * (self.dia_mm / 2.0) ** 2


def steel_linear_mass_kg_m(dia_mm: float) -> float:
    """Masse linéique (kg/m) d’un barreau Ø dia_mm."""
    r_m = dia_mm / 1000.0 / 2.0
    area_m2 = math.pi * r_m**2
    return area_m2 * STEEL_DENSITY_KG_M3


# =========================
# Étriers
# =========================

@dataclass
class StirrupType:
    """Type d’étrier (diamètre + type général)."""
    name: str
    phi_mm: float
    type: StirrupTypeEnum  # conservé pour futur, mais UI simplifiée


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


def stirrup_length_m(st_type: StirrupType, geom: BeamGeometry) -> float:
    """
    Longueur d’un étrier en m (approx) :
    contour rectangulaire de l’âme à l’enrobage + crochets simples.
    """
    c = geom.cover
    b_cm = geom.b_web - 2 * c
    h_cm = geom.h_total - 2 * c

    b_mm = b_cm * 10.0
    h_mm = h_cm * 10.0

    perim_mm = 2.0 * (b_mm + h_mm)

    phi = st_type.phi_mm
    hook_mm = 8.0 * phi * 2  # deux crochets approx.
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
        """Reconstruit les RebarInstance à partir des familles."""
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
    mass_concrete_kg: float
    V_steel_m3: float
    mass_steel_kg: float
    kg_per_m3: float
    by_category_m3: Dict[str, float]


def rebar_volume_m3_from_length(length_m: float, dia_mm: float) -> float:
    """Volume (m³) d’acier = L × aire."""
    r_m = dia_mm / 1000.0 / 2.0
    area_m2 = math.pi * r_m**2
    return length_m * area_m2


def compute_quantities(
    geom: BeamGeometry,
    L_beam_m: float,
    families: List[RebarFamily],
    stirrup_results: List[StirrupResult],
) -> QuantitiesResult:
    """Volumes béton / acier et taux d’armatures global + par catégorie."""
    area_cm2 = geom.area_cm2()
    Vc_m3 = area_cm2 / 10000.0 * L_beam_m
    mass_concrete = Vc_m3 * CONCRETE_DENSITY_KG_M3

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
    kg_per_m3 = mass_steel / Vc_m3 if Vc_m3 > 0 else 0.0

    return QuantitiesResult(
        Vc_m3=Vc_m3,
        mass_concrete_kg=mass_concrete,
        V_steel_m3=V_s_total,
        mass_steel_kg=mass_steel,
        kg_per_m3=kg_per_m3,
        by_category_m3=cat_vol,
    )


# =========================
# Outils couleurs
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
    if pos in ("web_bottom", "flange_left_bottom", "flange_right_bottom"):
        return "bottom"
    if pos in ("web_top", "flange_left_top", "flange_right_top"):
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
    """Indice de lit (0,1,2,...) par position."""
    layer_by_id: Dict[str, int] = {}
    by_pos: Dict[str, List[RebarFamily]] = {}
    for f in families:
        by_pos.setdefault(f.position, []).append(f)
    for pos, flist in by_pos.items():
        for idx, f in enumerate(flist):
            layer_by_id[f.id] = idx
    return layer_by_id


def compute_family_codes(raw_families: List[Dict]) -> Dict[str, str]:
    """Codes INF1, SUP1, LAT1, ... à partir des positions."""
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


def color_icon_for_position(pos: str) -> str:
    """Icône simple pour la couleur (lisible dans le tableau)."""
    grp = _position_group(pos)
    if grp == "bottom":
        return "🔴"
    if grp == "top":
        return "🔵"
    return "⚫️"


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

    rect = plt.Rectangle(
        (xL, yB),
        xR - xL,
        yT - yB,
        fill=False,
        linewidth=line_w,
        edgecolor="red",
        linestyle="-",
    )
    ax.add_patch(rect)


def draw_section(
    section: BeamSection,
    stirrup_type: Optional[StirrupType] = None,
    highlight_family_ids: Optional[List[str]] = None,
):
    geom = section.geometry
    b = geom.b_web
    h = geom.h_total

    fig, ax = plt.subplots(figsize=(2.6, 3.6))

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
    "web_bottom": "Armatures inférieures (âme)",
    "web_top": "Armatures supérieures (âme)",
    "flange_left_bottom": "Talon gauche – inférieure",
    "flange_left_top": "Talon gauche – supérieure",
    "flange_right_bottom": "Talon droit – inférieure",
    "flange_right_top": "Talon droit – supérieure",
    "side_left": "Latérales gauche",
    "side_right": "Latérales droite",
    "side_both": "Latérales 2 côtés",
}

ROLE_LABELS = {
    "main": "Lit principal",
    "reinforcement": "Renfort local",
    "lateral": "Barres latérales",
}


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

    st.session_state.setdefault("range_mode", "percent")  # "percent" ou "meter"


def build_geometry() -> BeamGeometry:
    shape = st.session_state.shape_section

    if "gauche" in shape:
        flange_left_b = st.session_state.flange_left_b_cm or 10.0
        flange_left_h = st.session_state.flange_left_h_cm or 15.0
    else:
        flange_left_b = 0.0
        flange_left_h = 0.0

    if "droit" in shape:
        flange_right_b = st.session_state.flange_right_b_cm or 10.0
        flange_right_h = st.session_state.flange_right_h_cm or 15.0
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
                hook_count=d.get("hook_count", 0),
                hook_height_cm=d.get("hook_height_cm", 0.0),
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

def _new_family_id() -> str:
    st.session_state.family_counter += 1
    return f"F{st.session_state.family_counter}"


def add_family(role: FamilyRole, position: Optional[PositionType] = None):
    fam_id = _new_family_id()
    geom = build_geometry()

    if role == "lateral":
        pos = position or "side_left"
        default_hook = geom.b_web - 2 * geom.cover
    else:
        pos = position or ("web_bottom" if role == "main" else "web_top")
        default_hook = geom.h_total - 2 * geom.cover

    st.session_state.rebar_families.append(
        {
            "id": fam_id,
            "role": role,
            "position": pos,
            "n_bars": 2,
            "dia_mm": 16.0,
            "hook_count": 0,
            "hook_height_cm": max(default_hook, 0.0),
            "x_start_pct": 0.0,
            "x_end_pct": 100.0,
        }
    )


def _position_options_for_family(role: FamilyRole) -> List[Tuple[str, PositionType]]:
    """Options de position en fonction du type de section."""
    shape = st.session_state.shape_section
    has_left = "gauche" in shape or "deux talons" in shape
    has_right = "droit" in shape or "deux talons" in shape

    opts: List[Tuple[str, PositionType]] = []

    if role == "lateral":
        opts.append(("Latérales gauche", "side_left"))
        opts.append(("Latérales droite", "side_right"))
        opts.append(("Latérales 2 côtés", "side_both"))
        return opts

    # sup / inf âme
    opts.append(("Armatures inférieures (âme)", "web_bottom"))
    opts.append(("Armatures supérieures (âme)", "web_top"))

    # talons uniquement si activés
    if has_left:
        opts.append(("Talon gauche – inférieure", "flange_left_bottom"))
        opts.append(("Talon gauche – supérieure", "flange_left_top"))
    if has_right:
        opts.append(("Talon droit – inférieure", "flange_right_bottom"))
        opts.append(("Talon droit – supérieure", "flange_right_top"))

    return opts


def edit_family(idx: int, fam: Dict, code: str, color: Optional[str] = None):
    # En-tête compact avec pastille couleur + code + bouton poubelle
    header_cols = st.columns([6, 1])
    with header_cols[0]:
        badge = ""
        if color:
            badge = (
                f"<span style='display:inline-block;width:12px;height:12px;"
                f"border-radius:50%;background-color:{color};"
                f"margin-right:6px;vertical-align:middle;'></span>"
            )
        st.markdown(
            f"{badge}<strong>{ROLE_LABELS[fam['role']]} – {code}</strong>",
            unsafe_allow_html=True,
        )
    with header_cols[1]:
        if st.button("🗑️", key=f"del_fam_{idx}"):
            st.session_state.rebar_families.pop(idx)
            st.rerun()

    # Ligne unique : Position ; n barres ; Ø ; retour ; h retour ; De ; À
    cpos, c_nb, c_phi, c_hook, c_h, c_de, c_a = st.columns(
        [2.3, 1.0, 1.0, 1.0, 1.3, 1.1, 1.1]
    )

    with cpos:
        opts = _position_options_for_family(fam["role"])
        labels = [o[0] for o in opts]
        values = [o[1] for o in opts]
        try:
            i_sel = values.index(fam["position"])
        except ValueError:
            i_sel = 0
            fam["position"] = values[0]
        pos_label = st.selectbox(
            "Position",
            labels,
            index=i_sel,
            key=f"pos_{idx}",
        )
        fam["position"] = values[labels.index(pos_label)]

    with c_nb:
        fam["n_bars"] = st.number_input(
            "n barres",
            min_value=1,
            max_value=20,
            value=int(fam["n_bars"]),
            step=1,
            key=f"nb_{idx}",
        )

    with c_phi:
        fam["dia_mm"] = st.selectbox(
            "Ø (mm)",
            [6, 8, 10, 12, 14, 16, 20, 25, 32, 40],
            index=[6, 8, 10, 12, 14, 16, 20, 25, 32, 40].index(int(fam["dia_mm"])),
            key=f"phi_{idx}",
        )

    with c_hook:
        fam["hook_count"] = st.selectbox(
            "Retours",
            [0, 1, 2],
            index=[0, 1, 2].index(int(fam.get("hook_count", 0))),
            key=f"hook_count_{idx}",
        )

    with c_h:
        if fam["hook_count"] == 0:
            st.text("Hauteur retour (cm)")
            st.text("—")
            fam["hook_height_cm"] = 0.0
        else:
            fam["hook_height_cm"] = st.number_input(
                "Hauteur retour (cm)",
                min_value=0.0,
                max_value=1000.0,
                value=float(fam.get("hook_height_cm", 0.0)),
                step=1.0,
                key=f"h_hook_{idx}",
            )

    range_mode = st.session_state.range_mode
    L_beam = st.session_state.L_beam_m

    with c_de:
        if range_mode == "percent":
            fam["x_start_pct"] = st.number_input(
                "De (%L)",
                min_value=0.0,
                max_value=100.0,
                value=float(fam["x_start_pct"]),
                step=5.0,
                format="%.0f",
                key=f"x1_{idx}",
            )
        else:
            val_m = L_beam * fam["x_start_pct"] / 100.0
            val_m = st.number_input(
                "De (m)",
                min_value=0.0,
                max_value=float(L_beam),
                value=float(val_m),
                step=max(L_beam / 20.0, 0.25),
                format="%.2f",
                key=f"x1m_{idx}",
            )
            fam["x_start_pct"] = 100.0 * val_m / L_beam if L_beam > 0 else 0.0

    with c_a:
        if range_mode == "percent":
            fam["x_end_pct"] = st.number_input(
                "À (%L)",
                min_value=0.0,
                max_value=100.0,
                value=float(fam["x_end_pct"]),
                step=5.0,
                format="%.0f",
                key=f"x2_{idx}",
            )
        else:
            val_m = L_beam * fam["x_end_pct"] / 100.0
            val_m = st.number_input(
                "À (m)",
                min_value=0.0,
                max_value=float(L_beam),
                value=float(val_m),
                step=max(L_beam / 20.0, 0.25),
                format="%.2f",
                key=f"x2m_{idx}",
            )
            fam["x_end_pct"] = 100.0 * val_m / L_beam if L_beam > 0 else 0.0


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

    d["phi_mm"] = st.number_input(
        "Ø (mm)",
        min_value=6.0,
        max_value=16.0,
        value=float(d["phi_mm"]),
        step=2.0,
        key=f"st_phi_{idx}",
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

    # ============== COLONNE GAUCHE ==============
    with col_left:
        # petite option pour la saisie des zones
        st.write("")
        st.radio(
            "Saisie des longueurs de lit",
            options=["En % de la portée", "En mètres (m)"],
            index=0 if st.session_state.range_mode == "percent" else 1,
            key="range_mode_radio",
            horizontal=True,
        )
        st.session_state.range_mode = (
            "percent" if st.session_state.range_mode_radio == "En % de la portée" else "meter"
        )

        # Données
        st.subheader("Données")

        c1, c2, c3 = st.columns(3)
        with c1:
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
        with c2:
            st.number_input(
                "Enrobage c (cm)",
                min_value=1.0,
                max_value=10.0,
                value=float(st.session_state.cover_cm),
                step=0.5,
                key="cover_cm",
            )
        with c3:
            st.number_input(
                "Portée L (m)",
                min_value=0.5,
                max_value=60.0,
                value=float(st.session_state.L_beam_m),
                step=0.5,
                key="L_beam_m",
            )

        c4, c5 = st.columns(2)
        with c4:
            st.number_input(
                "Largeur b (cm)",
                min_value=5.0,
                max_value=200.0,
                value=float(st.session_state.b_web_cm),
                step=1.0,
                key="b_web_cm",
            )
        with c5:
            st.number_input(
                "Hauteur h (cm)",
                min_value=10.0,
                max_value=300.0,
                value=float(st.session_state.h_total_cm),
                step=1.0,
                key="h_total_cm",
            )

        # Talons si activés
        shape = st.session_state.shape_section
        if "gauche" in shape or "deux talons" in shape:
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

        if "droit" in shape or "deux talons" in shape:
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

        # Armatures
        st.subheader("Armatures")

        btn_cols = st.columns(5)
        with btn_cols[0]:
            if st.button("+ Lit principal"):
                add_family("main")
        with btn_cols[1]:
            if st.button("+ Renfort local"):
                add_family("reinforcement")
        with btn_cols[2]:
            if st.button("+ Latéral"):
                add_family("lateral")
        with btn_cols[3]:
            # Renfort talon seulement si une section avec talon est choisie
            has_flange = "gauche" in shape or "droit" in shape or "deux talons" in shape
            if st.button("+ Renfort talon", disabled=not has_flange):
                # par défaut talon gauche inférieur si dispo, sinon talon droit
                pos: PositionType
                if "gauche" in shape or "deux talons" in shape:
                    pos = "flange_left_bottom"
                else:
                    pos = "flange_right_bottom"
                add_family("reinforcement", position=pos)
        with btn_cols[4]:
            if st.button("+ Étrier"):
                if not st.session_state.stirrup_types:
                    add_stirrup_type()
                add_stirrup_zone()

        # Couleurs & codes pour les familles
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
                        code=code,
                        color=family_colors_ui.get(fam["id"]),
                    )

        # Renforts
        reinfs = [f for f in st.session_state.rebar_families if f["role"] == "reinforcement"]
        if reinfs:
            st.markdown("**Renforts (y compris talons)**")
        for idx, fam in enumerate(list(st.session_state.rebar_families)):
            if fam["role"] == "reinforcement":
                with st.container(border=True):
                    code = family_codes_ui.get(fam["id"], fam["id"])
                    edit_family(
                        idx,
                        fam,
                        code=code,
                        color=family_colors_ui.get(fam["id"]),
                    )

        # Latérales
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
                        code=code,
                        color=family_colors_ui.get(fam["id"]),
                    )

        # Étriers
        if st.session_state.stirrup_types or st.session_state.stirrup_zones:
            st.subheader("Étriers")

        if st.session_state.stirrup_types:
            for i, d in enumerate(list(st.session_state.stirrup_types)):
                with st.container(border=True):
                    edit_stirrup_type(i, d)

        if st.session_state.stirrup_zones:
            st.markdown("**Zones d’étriers**")
            for i, d in enumerate(list(st.session_state.stirrup_zones)):
                with st.container(border=True):
                    edit_stirrup_zone(i, d)

    # =============== COLONNE DROITE ===============
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

        # Qualité acier + majoration
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
        kg_m3_majorated = (
            mass_steel_majorated / qres.Vc_m3 if qres.Vc_m3 > 0 else 0.0
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Volume béton Vc", f"{qres.Vc_m3:.3f} m³")
        m2.metric("Masse acier", f"{mass_steel_majorated:.1f} kg")
        m3.metric("Taux d’armature", f"{kg_m3_majorated:.1f} kg/m³")

        # Petit tableau récapitulatif masse / béton / taux
        st.markdown("**Synthèse**")
        synth_rows = [
            {
                "Élément": "Masse acier (majorée)",
                "Valeur": f"{mass_steel_majorated:.1f} kg",
            },
            {
                "Élément": "Volume béton",
                "Valeur": f"{qres.Vc_m3:.3f} m³",
            },
            {
                "Élément": "Poids béton (≈{CONCRETE_DENSITY_KG_M3:.0f} kg/m³)",
                "Valeur": f"{qres.mass_concrete_kg:.0f} kg",
            },
            {
                "Élément": "Taux d’armature acier",
                "Valeur": f"{kg_m3_majorated:.1f} kg/m³",
            },
        ]
        st.table(synth_rows)

        # Tableau récapitulatif des armatures
        st.markdown("### Tableau récapitulatif des armatures")

        rows = []
        L_beam = st.session_state.L_beam_m

        # Longitudinales
        for fam in families:
            code = family_codes.get(fam.id, fam.id)
            L_active = fam.active_length_m(L_beam)
            L_tot = fam.total_bar_length_m(L_beam)
            m_lin = steel_linear_mass_kg_m(fam.dia_mm)
            m_tot = L_tot * m_lin * (1.0 + maj_pct / 100.0)

            rows.append(
                {
                    "Couleur": color_icon_for_position(fam.position),
                    "Code": code,
                    "Position": POSITION_LABELS.get(fam.position, fam.position),
                    "n barres": fam.effective_n_bars(),
                    "Ø (mm)": int(fam.dia_mm),
                    "L active (m)": f"{L_active:.2f}",
                    "L tot (m)": f"{L_tot:.2f}",
                    "kg/m": f"{m_lin:.3f}",
                    "kg tot": f"{m_tot:.2f}",
                }
            )

        # Étriers
        for res in stirrup_results:
            m_lin = steel_linear_mass_kg_m(res.stirrup_type.phi_mm)
            m_tot = res.total_length_m * m_lin * (1.0 + maj_pct / 100.0)
            rows.append(
                {
                    "Couleur": "🟡",
                    "Code": res.stirrup_type.name,
                    "Position": f"Zone {res.zone.name}",
                    "n barres": res.n_stirrups,
                    "Ø (mm)": int(res.stirrup_type.phi_mm),
                    "L active (m)": f"{res.length_per_stirrup_m:.2f}",
                    "L tot (m)": f"{res.total_length_m:.2f}",
                    "kg/m": f"{m_lin:.3f}",
                    "kg tot": f"{m_tot:.2f}",
                }
            )

        if rows:
            import pandas as pd

            df = pd.DataFrame(rows)
            st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            st.info("Ajoute des armatures pour voir le tableau récapitulatif.")

        # Dessin de la section
        st.markdown("**Section générale (tous lits)**")
        fig_gen = draw_section(section)
        st.pyplot(fig_gen, use_container_width=False)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Poutre BA – Ferraillage complet",
        page_icon="🧱",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    show()
