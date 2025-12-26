import os
import pandas as pd
import streamlit as st
import warnings

import market_cache
from app_theme import apply_global_styles, render_header
from views.bonos_hd import render_bonos_hd
from views.bopreales import render as render_bopreales
from views.letras_boncaps import render_letras_boncaps
from views.bonos_cer import render_bonos_cer
from views.ons_ytm import render as render_ons_ytm
from views.bonos_general import render as render_bonos_general
from views.tabla_sensibilidad import render as render_tabla_sensibilidad



warnings.filterwarnings(
    "ignore",
    message="The behavior of DatetimeProperties.to_pydatetime is deprecated*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="When grouping with a length-1 list-like*",
    category=FutureWarning,
)



# ============================================================
# CONFIG (DEBE IR PRIMERO)
# ============================================================
st.set_page_config(
    page_title="DCF | Herramientas de Análisis de Mercado",
    layout="wide",
)


# ============================================================
# HELPERS: secrets + user
# ============================================================
def _get_secrets_dict() -> dict:
    """
    En local sin secrets.toml, st.secrets levanta StreamlitSecretNotFoundError.
    Acá devolvemos {} para que no rompa.
    """
    try:
        return st.secrets.to_dict()
    except Exception:
        return {}


def _get_user_email() -> str | None:
    """
    Streamlit puede exponer email como atributo o como dict-like.
    """
    try:
        email = getattr(st.user, "email", None)
        if email:
            return str(email).strip().lower()
    except Exception:
        pass

    try:
        if hasattr(st, "user") and hasattr(st.user, "get"):
            email = st.user.get("email")
            if email:
                return str(email).strip().lower()
    except Exception:
        pass

    return None


# ============================================================
# LOGIN + WHITELIST (SOLO WEB CUANDO HAY [auth] EN SECRETS)
# ============================================================
def login_gate(secrets: dict):
    """
    Si no hay [auth] en secrets => LOCAL / sin auth => NO bloquea.
    Si hay [auth] => exige login con Google (Streamlit Cloud).
    """
    auth_cfg = secrets.get("auth")
    if not auth_cfg:
        return  # local / sin secrets => no login

    # Auth configurado -> exigir login
    if not hasattr(st, "login") or not hasattr(st, "user") or not hasattr(st.user, "is_logged_in"):
        st.error("Auth configurado pero esta versión/entorno no soporta login. Requiere streamlit>=1.42.0.")
        st.stop()

    if not st.user.is_logged_in:
        st.title("DCF Inversiones — Acceso a clientes")
        st.write("Ingresá con tu cuenta de Google para continuar.")
        st.button("Ingresar con Google", on_click=st.login, type="primary")
        st.stop()


@st.cache_data(ttl=60)
def load_whitelist_df(sheet_id: str, gid: str) -> pd.DataFrame:
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    return pd.read_csv(url)


def enforce_whitelist(secrets: dict):
    """
    Solo corre si existe [auth] (o sea, en Cloud).
    """
    auth_cfg = secrets.get("auth")
    if not auth_cfg:
        return  # local / sin auth => no whitelist

    email = _get_user_email()
    if not email:
        st.error("No se pudo obtener el email desde Google.")
        st.stop()

    # Requiere whitelist en secrets
    wl = secrets.get("whitelist", {})
    sheet_id = wl.get("sheet_id")
    gid = wl.get("gid")
    if not sheet_id or gid is None:
        st.error("Falta configurar whitelist en Secrets: whitelist.sheet_id y whitelist.gid")
        st.stop()

    df = load_whitelist_df(str(sheet_id), str(gid))

    if "email" not in df.columns or "active" not in df.columns:
        st.error(
            "El Google Sheet debe tener columnas 'email' y 'active'. "
            f"Columnas encontradas: {list(df.columns)}"
        )
        st.stop()

    # Normalización
    df["email"] = df["email"].astype(str).str.strip().str.lower()
    df["active"] = (
        df["active"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y", "si", "sí"])
    )

    row = df.loc[df["email"] == email]
    allowed = (not row.empty) and bool(row.iloc[0]["active"])

    if not allowed:
        st.title("Acceso restringido")
        st.write(f"Tu cuenta **{email}** no está habilitada para acceder.")
        st.write("Si sos cliente de DCF, pedinos acceso y lo habilitamos.")
        if hasattr(st, "logout"):
            st.button("Cerrar sesión", on_click=st.logout)
        st.stop()

    # Rol (opcional)
    role = "user"
    if "role" in df.columns and not row.empty:
        role = str(row.iloc[0].get("role", "user")).strip().lower() or "user"

    st.session_state["role"] = role


# Ejecutar gates antes del resto de la app
_secrets = _get_secrets_dict()
login_gate(_secrets)
enforce_whitelist(_secrets)


# ============================================================
# ESTILOS + HEADER
# ============================================================
apply_global_styles()
render_header()


# ============================================================
# AJUSTES UI (sidebar)
# ============================================================
st.markdown(
    """
    <style>
      section[data-testid="stSidebar"] [data-testid="stMetricLabel"] { font-size: 0.85rem; }
      section[data-testid="stSidebar"] [data-testid="stMetricValue"] { font-size: 1.35rem; }
      section[data-testid="stSidebar"] [data-testid="stMetricDelta"] { font-size: 0.75rem; }

      .dcf-badge {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-size: 0.80rem;
        font-weight: 600;
        line-height: 1.2;
        margin-top: 0.25rem;
      }
      .dcf-open  { background: rgba(22, 163, 74, 0.18); border: 1px solid rgba(22, 163, 74, 0.55); }
      .dcf-closed{ background: rgba(239, 68, 68, 0.16); border: 1px solid rgba(239, 68, 68, 0.55); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================
email = _get_user_email() or ""

if email and hasattr(st, "logout"):
    st.sidebar.success(f"Sesión iniciada: {email}")
    st.sidebar.button("Cerrar sesión", on_click=st.logout)

st.sidebar.title("Instrumentos")
st.sidebar.metric("Última actualización", market_cache.get_last_update_display())

is_open = market_cache.is_market_open()
badge_text = "Mercado abierto" if is_open else "Mercado cerrado"
badge_class = "dcf-open" if is_open else "dcf-closed"
st.sidebar.markdown(
    f'<span class="dcf-badge {badge_class}">{badge_text}</span>',
    unsafe_allow_html=True,
)


# ============================================================
# NAVEGACIÓN
# ============================================================
main_view = st.sidebar.radio(
    "Seleccioná el instrumento",
    options=[
        "Bonos",
        "Letras y Boncaps",
        "Bonos Ajustables CER",
        "ONs",
    ],
    index=0,
    key="main_view",
)

bonos_subview = st.session_state.get("bonos_subview", "Bonos Soberanos")

if main_view == "Bonos":
    st.sidebar.markdown("---")
    st.sidebar.markdown("<div style='margin-left:12px'>", unsafe_allow_html=True)

    bonos_subview = st.sidebar.radio(
        "Subtipo de bonos",
        options=[
            "Bonos Soberanos",
            "Bopreales",
            "Tabla de Sensibilidad",
            "Gráfico General",
        ],

        index=0,
        key="bonos_subview",
        label_visibility="collapsed",
    )

    st.sidebar.markdown("</div>", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption("Herramienta interna • DCF Inversiones")


# ============================================================
# LÓGICA PRINCIPAL
# ============================================================
if main_view == "Bonos":
    if bonos_subview == "Bonos Soberanos":
        render_bonos_hd()
    elif bonos_subview == "Bopreales":
        render_bopreales()
    elif bonos_subview == "Tabla de Sensibilidad":
        render_tabla_sensibilidad()
    elif bonos_subview == "Gráfico General":
        render_bonos_general()

elif main_view == "Letras y Boncaps":
    render_letras_boncaps()

elif main_view == "Bonos Ajustables CER":
    render_bonos_cer()

elif main_view == "ONs":
    render_ons_ytm()
