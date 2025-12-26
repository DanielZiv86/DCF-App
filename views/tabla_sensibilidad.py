# views/tabla_sensibilidad.py
import pandas as pd
import numpy as np
import streamlit as st

from bonistas_hd_data import get_multi_table as get_hd_table
from bopreales_data import get_multi_table as get_bop_table
from datetime import datetime

import matplotlib.cm as cm
import matplotlib.colors as mcolors


# =========================
# Safe fetch (fallback ante 503)
# =========================
def _now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_get_hd_table(mercado: str = "MEP") -> pd.DataFrame | None:
    key_df = f"last_ok_hd_{mercado}"
    key_ts = f"last_ok_hd_{mercado}_ts"

    try:
        df = get_hd_table(mercado=mercado)
        st.session_state[key_df] = df
        st.session_state[key_ts] = _now_str()
        return df
    except Exception as e:
        if key_df in st.session_state and isinstance(st.session_state[key_df], pd.DataFrame):
            st.warning(
                f"⚠️ IOL no respondió (503). Mostrando último dato disponible "
                f"({st.session_state.get(key_ts, 'sin timestamp')})."
            )
            return st.session_state[key_df]

        st.error(
            "❌ No se pudo actualizar la data desde IOL (503 repetidos) y no hay cache previo. "
            "Probá nuevamente en unos minutos."
        )
        st.caption(f"Detalle técnico: {type(e).__name__}: {e}")
        return None


def safe_get_bop_table_usd() -> pd.DataFrame | None:
    key_df = "last_ok_bop_usd"
    key_ts = "last_ok_bop_usd_ts"

    try:
        df = get_bop_table()
        df = df[df["Mercado"] == "USD"].copy()
        st.session_state[key_df] = df
        st.session_state[key_ts] = _now_str()
        return df
    except Exception as e:
        if key_df in st.session_state and isinstance(st.session_state[key_df], pd.DataFrame):
            st.warning(
                f"⚠️ No se pudo actualizar BOPREAL. Mostrando último dato disponible "
                f"({st.session_state.get(key_ts, 'sin timestamp')})."
            )
            return st.session_state[key_df]

        st.error("❌ No se pudo cargar BOPREAL y no hay cache previo.")
        st.caption(f"Detalle técnico: {type(e).__name__}: {e}")
        return None


# =========================
# Targets
# =========================
TARGETS_HD = [0.06, 0.065, 0.07, 0.08]
TARGETS_BOP = [-0.05, -0.02, 0.00, 0.02]


# =========================
# Sensitivity calc
# =========================
def _fmt_target_col(t: float) -> str:
    # 0.065 -> "6,5%"
    s = f"{t*100:.1f}%"
    return s.replace(".0%", "%").replace(".", ",")


def build_sensitivity(df: pd.DataFrame, targets: list[float], id_col: str) -> pd.DataFrame:
    """
    Upside% ≈ -Duration * (TIR_obj - TIR_act) * 100
    """
    out = df[[id_col, "TIR", "Duration"]].dropna().copy()
    out["TIR"] = pd.to_numeric(out["TIR"], errors="coerce")
    out["Duration"] = pd.to_numeric(out["Duration"], errors="coerce")
    out = out.dropna(subset=["TIR", "Duration"])

    for t in targets:
        col = _fmt_target_col(t)
        out[col] = -out["Duration"] * (t - out["TIR"]) * 100.0

    return out.set_index(id_col)


# =========================
# Balanz-like HTML table
# =========================
def _rgba_to_hex(rgba):
    r, g, b, a = rgba
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def _cell_bg(val: float, neg_min: float, pos_max: float) -> str:
    """
    Paleta Balanz-like:
    - negativos: Reds (rojo)
    - >=0: YlGn (amarillo -> verde)
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "transparent"

    if val < 0:
        denom = (0 - neg_min) if (0 - neg_min) != 0 else 1.0
        t = (val - neg_min) / denom
        t = float(np.clip(t, 0, 1))
        rgba = cm.get_cmap("Reds")(0.25 + 0.65 * t)
        return _rgba_to_hex(rgba)

    denom = pos_max if pos_max != 0 else 1.0
    t = val / denom
    t = float(np.clip(t, 0, 1))
    rgba = cm.get_cmap("YlGn")(0.20 + 0.65 * t)
    return _rgba_to_hex(rgba)


def _render_balanz_like_table(df_sens: pd.DataFrame, targets_cols: list[str]):
    """
    Tabla HTML/CSS con:
    - header merged real: TIR OBJETIVO (colspan)
    - anchos iguales
    - colores Balanz (rojos solo negativos; amarillos->verdes positivos)
    """
    if df_sens is None or df_sens.empty:
        st.info("Sin datos para mostrar.")
        return

    # Rangos para color
    vals = df_sens[targets_cols].to_numpy(dtype=float) if targets_cols else np.array([])
    if vals.size:
        neg_min = float(np.nanmin(vals[vals < 0])) if np.any(vals < 0) else -1.0
        pos_vals = vals[(np.isfinite(vals)) & (vals >= 0)]
        pos_max = float(np.nanpercentile(pos_vals, 95)) if pos_vals.size else 1.0
        if pos_max <= 0:
            pos_max = 1.0
    else:
        neg_min, pos_max = -1.0, 1.0

    # formatos
    def fmt_pct(x):
        if pd.isna(x):
            return ""
        return f"{x:+.2f}%"

    def fmt_tir(x):
        if pd.isna(x):
            return ""
        return f"{x*100:.2f}%"

    def fmt_dur(x):
        if pd.isna(x):
            return ""
        return f"{x:.2f}"

    n_targets = len(targets_cols)

    # CSS (muy cercano a Balanz)
    html = f"""
    <style>
      .dcf-balanz-wrap {{
        width: 100%;
        overflow-x: auto;
        margin-top: 10px;
      }}
      table.dcf-balanz {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        table-layout: fixed;
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 12px;
        overflow: hidden;
        background: rgba(0,0,0,0.10);
        font-size: 15px;
      }}
      table.dcf-balanz th, table.dcf-balanz td {{
        padding: 10px 10px;
        text-align: center;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        border-right: 1px solid rgba(255,255,255,0.08);
        white-space: nowrap;
      }}
      table.dcf-balanz th {{
        background: rgba(255,255,255,0.06);
        font-weight: 800;
        color: rgba(255,255,255,0.95);
      }}
      table.dcf-balanz th.group {{
        background: rgba(255,255,255,0.08);
        font-size: 13px;
        letter-spacing: 0.6px;
      }}
      table.dcf-balanz td.cold {{
        background: transparent;
        font-weight: 700;
      }}
      table.dcf-balanz td.bono {{
        font-weight: 900;
      }}
      table.dcf-balanz tr:last-child td {{
        border-bottom: none;
      }}
      table.dcf-balanz th:last-child, table.dcf-balanz td:last-child {{
        border-right: none;
      }}

      /* Anchos iguales para TODAS las columnas como en el screenshot */
      table.dcf-balanz col {{ width: {100/(3+n_targets):.6f}% }}
    </style>

    <div class="dcf-balanz-wrap">
      <table class="dcf-balanz">
        <colgroup>
          <col><col><col>
          {''.join('<col>' for _ in targets_cols)}
        </colgroup>
        <thead>
          <tr>
            <th rowspan="2">BONO</th>
            <th rowspan="2">TIR ACTUAL</th>
            <th rowspan="2">DURATION</th>
            <th class="group" colspan="{n_targets}">TIR OBJETIVO</th>
          </tr>
          <tr>
            {''.join(f'<th>{c}</th>' for c in targets_cols)}
          </tr>
        </thead>
        <tbody>
    """

    for bono, row in df_sens.iterrows():
        html += "<tr>"
        html += f'<td class="bono">{bono}</td>'
        html += f'<td class="cold">{fmt_tir(row["TIR"])}</td>'
        html += f'<td class="cold">{fmt_dur(row["Duration"])}</td>'

        for c in targets_cols:
            v = row[c]
            bg = _cell_bg(v, neg_min=neg_min, pos_max=pos_max)

            text_color = "#0b0f14"
            if bg != "transparent":
                rgb = mcolors.to_rgb(bg)
                lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
                if lum < 0.55:
                    text_color = "white"

            html += (
                f'<td style="background:{bg}; color:{text_color}; font-weight:900;">'
                f'{fmt_pct(v)}'
                f"</td>"
            )

        html += "</tr>"

    html += """
        </tbody>
      </table>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)


def _pick_example(sens: pd.DataFrame, prefer: str, target: str):
    if prefer in sens.index and target in sens.columns:
        return prefer, float(sens.loc[prefer, "TIR"]), float(sens.loc[prefer, target])

    for idx in sens.index:
        if target in sens.columns and pd.notna(sens.loc[idx, target]):
            return str(idx), float(sens.loc[idx, "TIR"]), float(sens.loc[idx, target])
    return None


# =========================
# Render
# =========================
def render():
    st.subheader("Tabla de Sensibilidad de Bonos")
    
    st.markdown(
        "🧮 **Tabla de sensibilidad:** Potencial de Upside/Downside de los bonos soberanos y Bopreales, "
        "ante distintos escenarios de *exit yield* (baja del riesgo país)."
    )

    tab_glo, tab_bon, tab_bop = st.tabs(["🌍 Globales", "🇦🇷 Bonares", "🧾 BOPREAL"])

    # -------------------------
    # Globales (MEP)
    # -------------------------
    with tab_glo:
        df = safe_get_hd_table("MEP")
        if df is None:
            st.stop()

        df = df[df["Base"].astype(str).str.upper().str.startswith("GD")].copy()
        df["Bono"] = df["Base"].astype(str).str.upper()

        sens = build_sensitivity(df, TARGETS_HD, id_col="Bono")
        sens = sens[["TIR", "Duration"] + [c for c in sens.columns if c.endswith("%")]]
        sens = sens.sort_values("Duration", ascending=True)


        col_6 = _fmt_target_col(0.06)
        ex = _pick_example(sens, prefer="GD41", target=col_6)
        if ex:
            bono, tir_act, up = ex
            st.markdown(
                f"👉🏾 Se lee así: El **{bono}** hoy rinde **{tir_act*100:.2f}%** TIR, "
                f"pero si el rendimiento baja a **6%** (TIR Objetivo) su precio subirá aprox **{up:+.1f}%**."
            )

        targets_cols = [c for c in sens.columns if c.endswith("%")]
        _render_balanz_like_table(sens, targets_cols)

    # -------------------------
    # Bonares (MEP) - AL + AE + AN
    # -------------------------
    with tab_bon:
        df = safe_get_hd_table("MEP")
        if df is None:
            st.stop()

        bases = df["Base"].astype(str).str.upper()
        df = df[bases.str.startswith(("AL", "AE", "AN"))].copy()
        df["Bono"] = df["Base"].astype(str).str.upper()

        sens = build_sensitivity(df, TARGETS_HD, id_col="Bono")
        sens = sens[["TIR", "Duration"] + [c for c in sens.columns if c.endswith("%")]]
        sens = sens.sort_values("Duration", ascending=True)


        targets_cols = [c for c in sens.columns if c.endswith("%")]
        _render_balanz_like_table(sens, targets_cols)

    # -------------------------
    # BOPREAL (USD)
    # -------------------------
    with tab_bop:
        df = safe_get_bop_table_usd()
        if df is None:
            st.stop()

        df["Bono"] = df["Base"].astype(str).str.upper()

        sens = build_sensitivity(df, TARGETS_BOP, id_col="Bono")
        sens = sens[["TIR", "Duration"] + [c for c in sens.columns if c.endswith("%")]]
        sens = sens.sort_values("Duration", ascending=True)


        targets_cols = [c for c in sens.columns if c.endswith("%")]
        _render_balanz_like_table(sens, targets_cols)
