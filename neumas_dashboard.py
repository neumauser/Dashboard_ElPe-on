import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Neumas Dashboard", layout="wide")

# ---------- Estilo y colores ----------
PALETTES = {
    "Clásica (Azul + Dorado)": {"bar":"#FF7F0E", "line":"#1F77B4", "grid":"#E6E6E6", "bg":"#FFFFFF", "axes":"#333333", "text":"#111111"},
    "Plotly (Azul + Naranjo)": {"bar":"#1F77B4", "line":"#FF7F0E", "grid":"#E6E6E6", "bg":"#FFFFFF", "axes":"#333333", "text":"#111111"},
    "Escala Gris + Verde": {"bar":"#4F4F4F", "line":"#2CA02C", "grid":"#D9D9D9", "bg":"#FFFFFF", "axes":"#333333", "text":"#111111"}
}

palette_name = list(PALETTES.keys())[0]
show_labels = True
excel_path = "Planilla Bajas 2023-2024-2025 PEÑON (DEF).xlsx"

COLORS = PALETTES[palette_name]

# ---------- Cargar datos ----------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0)
    # Normalización de nombres
    df.columns = [str(c).strip().upper() for c in df.columns]
    
    # CORRECCIÓN MÁS CONSERVADORA: Solo limpiar columnas de texto sin convertir números
    for col in df.columns:
        if df[col].dtype == 'object':  # Solo para columnas de texto
            # Preservar valores NaN reales
            mask_notna = df[col].notna()
            if mask_notna.any():
                # Solo procesar valores no nulos
                df.loc[mask_notna, col] = df.loc[mask_notna, col].astype(str).str.strip()
                # Eliminar caracteres de espacio no visibles
                df.loc[mask_notna, col] = df.loc[mask_notna, col].str.replace('\xa0', ' ', regex=False)
                df.loc[mask_notna, col] = df.loc[mask_notna, col].str.replace('\u200B', '', regex=False)
                # Normalizar múltiples espacios a uno solo
                df.loc[mask_notna, col] = df.loc[mask_notna, col].str.replace(r'\s+', ' ', regex=True)
                # Volver a convertir strings vacíos o 'nan' a NaN
                df.loc[df[col].isin(['', 'nan', 'None']), col] = np.nan
    
    # Tipos de datos (sin cambios)
    for col in ["FECHA", "MES"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "AÑO" in df.columns:
        df["AÑO"] = pd.to_numeric(df["AÑO"], errors="coerce").astype("Int64")
    if "HORAS" in df.columns:
        df["HORAS"] = pd.to_numeric(df["HORAS"], errors="coerce")
    if "% GOMA REMANENTE" in df.columns:
        df["% GOMA REMANENTE"] = pd.to_numeric(df["% GOMA REMANENTE"], errors="coerce")
    return df

try:
    df = load_data(excel_path)
except Exception as e:
    st.error(f"No pude cargar el archivo: {e}")
    st.stop()

if df.empty:
    st.warning("El archivo se cargó pero no tiene filas.")
    st.stop()

# ---------- Helpers ----------
def safe_unique(series):
    """Función mejorada para obtener valores únicos sin duplicados"""
    if series is None or len(series) == 0:
        return []
    
    # Limpiar solo valores no nulos y mantener tipos originales
    valid_series = series.dropna()
    if len(valid_series) == 0:
        return []
    
    # Si es una serie de texto, limpiar espacios
    if valid_series.dtype == 'object':
        cleaned = valid_series.astype(str).str.strip()
        # Eliminar valores vacíos después del strip
        cleaned = cleaned[cleaned != '']
        cleaned = cleaned[cleaned.str.lower() != 'nan']
    else:
        cleaned = valid_series
    
    # Obtener únicos y ordenar
    unique_vals = sorted(cleaned.unique().tolist())
    
    # DEBUG: Solo mostrar cuando hay problemas potenciales
    if hasattr(series, 'name') and series.name == 'MOTIVO DE BAJA':
        if len(unique_vals) != len(set(unique_vals)):  # Solo si hay duplicados reales
            st.sidebar.write("⚠️ DEBUG - Posibles duplicados en MOTIVO DE BAJA:")
            for i, val in enumerate(unique_vals):
                st.sidebar.write(f"{i+1}. '{val}'")
    
    return unique_vals

def month_name(m):
    try:
        return pd.Timestamp(year=2000, month=int(m), day=1).strftime("%b")
    except Exception:
        return str(m)

# ---------- Barra superior de filtros (horizontal) ----------
st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {COLORS['bg']}; }}
    .filter-bar {{
        padding: 10px 12px;
        border-radius: 12px;
        background: #fafafa;
        border: 1px solid #eee;
        margin-bottom: 10px;
    }}
    .metric-box .stMetric-value {{ font-weight: 700; }}
    </style>
    """, unsafe_allow_html=True
)

st.markdown("### Filtros")
# --- Definición de caso ---
case_cols = []
with st.container():
    cdef = st.columns(4)
    case_def = cdef[0].selectbox(
        "Definición de 'caso'",
        [
            "Filas (cada registro)",
            "Neumáticos únicos (SERIE)",
            "Equipos únicos (Nº INTERNO)",
            "Único por SERIE+MOTIVO"
        ], index=0
    )
    only_with_motive = cdef[1].checkbox("Solo con MOTIVO DE BAJA", value=False)
    # Mostrar la paleta actual para referencia visual en el header
    cdef[2].markdown(f"**Barra:** <span style='color:{COLORS['bar']}'>■</span>", unsafe_allow_html=True)
    cdef[3].markdown(f"**Línea:** <span style='color:{COLORS['line']}'>●</span>", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
    cols = st.columns(7)  # todos horizontales
    # Definir columnas reales según disponibilidad
    has_mes = "MES" in df.columns or "FECHA" in df.columns
    # Año
    if "AÑO" in df.columns:
        ano_opts = safe_unique(df["AÑO"])
        ano_sel = cols[0].multiselect("Año", ano_opts, default=ano_opts)
    else:
        ano_sel = []
    # Mes
    if has_mes:
        if "FECHA" in df.columns and pd.api.types.is_datetime64_any_dtype(df["FECHA"]):
            months = sorted(df["FECHA"].dropna().dt.month.unique().tolist())
        elif "MES" in df.columns and pd.api.types.is_datetime64_any_dtype(df["MES"]):
            months = sorted(df["MES"].dropna().dt.month.unique().tolist())
        else:
            months = sorted(pd.to_datetime(df.get("MES", pd.Series([])), errors="coerce").dt.month.dropna().unique().tolist())
        nice_months = [f"{m:02d}-{month_name(m)}" for m in months]
        mes_choice = cols[1].multiselect("Mes", nice_months, default=nice_months)
        mes_sel = [int(x.split("-")[0]) for x in mes_choice]
    else:
        mes_sel = []
    # Equipo
    eq_sel = cols[2].multiselect("Flota (Equipo)", safe_unique(df["EQUIPO"]) if "EQUIPO" in df.columns else [],
                                 default=(safe_unique(df["EQUIPO"]) if "EQUIPO" in df.columns else []))
    # Medida
    med_sel = cols[3].multiselect("Medida", safe_unique(df["MEDIDA"]) if "MEDIDA" in df.columns else [],
                                  default=(safe_unique(df["MEDIDA"]) if "MEDIDA" in df.columns else []))
    # Marca
    mar_sel = cols[4].multiselect("Marca", safe_unique(df["MARCA"]) if "MARCA" in df.columns else [],
                                  default=(safe_unique(df["MARCA"]) if "MARCA" in df.columns else []))
    # Modelo
    mod_sel = cols[5].multiselect("Diseño (Modelo)", safe_unique(df["MODELO"]) if "MODELO" in df.columns else [],
                                  default=(safe_unique(df["MODELO"]) if "MODELO" in df.columns else []))
    # Motivo
    mot_sel = cols[6].multiselect("Motivo de Baja", safe_unique(df["MOTIVO DE BAJA"]) if "MOTIVO DE BAJA" in df.columns else [],
                                  default=(safe_unique(df["MOTIVO DE BAJA"]) if "MOTIVO DE BAJA" in df.columns else []))
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Aplicar filtros ----------
fdf = df.copy()
# Aplicar filtro de 'solo con motivo'
if only_with_motive and 'MOTIVO DE BAJA' in fdf.columns:
    fdf = fdf[fdf['MOTIVO DE BAJA'].notna()]

def get_case_group(df_in: pd.DataFrame) -> pd.Series:
    """Return a group identifier per row based on case_def"""
    if case_def == 'Filas (cada registro)':
        return pd.Series(range(len(df_in)), index=df_in.index)  # unique per row
    if case_def == 'Neumáticos únicos (SERIE)' and 'SERIE' in df_in.columns:
        return df_in['SERIE']
    if case_def == 'Equipos únicos (Nº INTERNO)':
        for col in ['Nº INTERNO','N° INTERNO','NO INTERNO','NUMERO INTERNO','Nº INTERNO']:
            if col in df_in.columns:
                return df_in[col]
        return pd.Series(range(len(df_in)), index=df_in.index)
    if case_def == 'Único por SERIE+MOTIVO' and 'SERIE' in df_in.columns and 'MOTIVO DE BAJA' in df_in.columns:
        return df_in['SERIE'].astype(str) + ' | ' + df_in['MOTIVO DE BAJA'].astype(str)
    return pd.Series(range(len(df_in)), index=df_in.index)

if len(ano_sel) > 0 and "AÑO" in fdf.columns:
    fdf = fdf[fdf["AÑO"].isin(ano_sel)]
if len(mes_sel) > 0 and has_mes:
    if "FECHA" in fdf.columns and pd.api.types.is_datetime64_any_dtype(fdf["FECHA"]):
        fdf = fdf[fdf["FECHA"].dt.month.isin(mes_sel)]
    elif "MES" in fdf.columns and pd.api.types.is_datetime64_any_dtype(fdf["MES"]):
        fdf = fdf[fdf["MES"].dt.month.isin(mes_sel)]
    else:
        tmp = pd.to_datetime(fdf.get("MES", pd.Series([])), errors="coerce")
        fdf = fdf[tmp.dt.month.isin(mes_sel)]
if len(eq_sel) > 0 and "EQUIPO" in fdf.columns:
    fdf = fdf[fdf["EQUIPO"].isin(eq_sel)]
if len(med_sel) > 0 and "MEDIDA" in fdf.columns:
    fdf = fdf[fdf["MEDIDA"].isin(med_sel)]
if len(mar_sel) > 0 and "MARCA" in fdf.columns:
    fdf = fdf[fdf["MARCA"].isin(mar_sel)]
if len(mod_sel) > 0 and "MODELO" in fdf.columns:
    fdf = fdf[fdf["MODELO"].isin(mod_sel)]
if len(mot_sel) > 0 and "MOTIVO DE BAJA" in fdf.columns:
    fdf = fdf[fdf["MOTIVO DE BAJA"].isin(mot_sel)]

# ---------- KPIs compactos ----------
k1, k2, k3 = st.columns(3)
total_bajas = len(fdf)
horas_prom = fdf["HORAS"].mean() if "HORAS" in fdf.columns else np.nan
goma_rem = fdf["% GOMA REMANENTE"].mean() 
if "% GOMA REMANENTE" in fdf.columns and len(fdf) > 0:
    vals = pd.to_numeric(
        fdf["% GOMA REMANENTE"].astype(str).str.replace("%", "").str.strip(),
        errors="coerce"
    )
    goma_rem = vals.sum() / len(fdf)
else:
    goma_rem = np.nan
with k1:
    st.metric("Cantidad Bajas", f"{total_bajas:,}")
with k2:
    st.metric("Promedio Horas", f"{horas_prom:,.0f}" if pd.notna(horas_prom) else "N/D")
with k3:
    st.metric(" % de Goma Remanente", f"{goma_rem*100:.2f}%")

st.markdown("---")

# ---------- Preparar datos de los gráficos ----------
# Grafico 1 por Mes
if "FECHA" in fdf.columns and pd.api.types.is_datetime64_any_dtype(fdf["FECHA"]):
    fdf["MESNUM"] = fdf["FECHA"].dt.month
elif "MES" in fdf.columns and pd.api.types.is_datetime64_any_dtype(fdf["MES"]):
    fdf["MESNUM"] = fdf["MES"].dt.month
else:
    if "MES" in fdf.columns:
        fdf["MESNUM"] = pd.to_datetime(fdf["MES"], errors="coerce").dt.month
    else:
        fdf["MESNUM"] = np.nan

month_order = list(range(1,13))
month_names = {i: pd.Timestamp(year=2000, month=i, day=1).strftime("%b") for i in month_order}

tmp = fdf.dropna(subset=["MESNUM"]).copy()
case_series = get_case_group(tmp)
tmp['__CASE__'] = case_series
g1 = tmp.groupby('MESNUM').agg(
    nro_casos=("__CASE__", pd.Series.nunique),
    prom_hrs=("HORAS", "mean") if "HORAS" in tmp.columns else ("__CASE__", pd.Series.nunique)
).reindex(month_order).fillna(0.0)

# Grafico 2 por Motivo
if "MOTIVO DE BAJA" in fdf.columns:
    if "HORAS" in fdf.columns:
        tmp2 = fdf.copy()
        tmp2['__CASE__'] = get_case_group(tmp2)
        g2 = tmp2.groupby('MOTIVO DE BAJA').agg(
            nro_casos=("__CASE__", pd.Series.nunique),
            prom_hrs=("HORAS", "mean")
        ).sort_values("nro_casos", ascending=False)
    else:
        g2 = fdf.groupby("MOTIVO DE BAJA").agg(
            nro_casos=("MOTIVO DE BAJA", "count")
        ).assign(prom_hrs=np.nan).sort_values("nro_casos", ascending=False)
else:
    g2 = pd.DataFrame(columns=["nro_casos","prom_hrs"])

# ---------- Helper: etiquetas de barras en la base ----------
def add_bar_base_labels(fig, x_vals, y_vals, labels, xaxis="x", yaxis="y", yshift=8):
    """
    Coloca etiquetas de barras en y=0 (parte inferior) con fuente gruesa y alto contraste.
    """
    for xv, yv, text in zip(x_vals, y_vals, labels):
        if text == "" or (not np.isfinite(yv)):
            continue
        fig.add_annotation(
            x=xv, y=0,
            xref=xaxis, yref=yaxis,
            text=text,
            showarrow=False,
            yshift=yshift,  # leve desplazamiento hacia arriba desde la base
            font=dict(family="Arial Black", size=11, color="black"),
            # bgcolor="rgba(255, 255, 255, 0.8)",
            # bordercolor="rgba(255, 255, 255, 0.8)",
            textangle=-35,
            opacity=0.95
        )

# ---------- Crear figuras ----------
def make_combined_bar_line(x, y_bar, y_line, x_labels=None, height=460):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    x_plot = x_labels if x_labels is not None else x

    # Barras (sin textposition; usaremos anotaciones en la base)
    fig.add_trace(
        go.Bar(
            x=x_plot,
            y=y_bar,
            name="Nro de Casos",
            marker_color=COLORS['bar'],
            marker_line=dict(width=0),
            hovertemplate="Nro de Casos: %{y}<extra></extra>",
        ),
        secondary_y=False
    )

    # Línea (con etiquetas en cada punto)
    fig.add_trace(
        go.Scatter(
            x=x_plot,
            y=y_line,
            name="Promedio Hrs acum.",
            mode="lines+markers+text" if show_labels else "lines+markers",
            text=[f"{v:,.0f}" if np.isfinite(v) else "" for v in y_line],
            textposition="top center",
            textfont=dict(family="Arial Black", size=12, color="black"),
            line=dict(color=COLORS['line'], width=3, shape="spline", smoothing=0.6),
            marker=dict(size=10, color=COLORS['line'], line=dict(width=3, color="white")),
            hovertemplate="Prom. Hrs: %{y:,.0f}<extra></extra>",
        ),
        secondary_y=True
    )

    # Etiquetas en la base de las barras
    if show_labels:
        bar_labels = [f"{int(v):,}" if np.isfinite(v) else "" for v in y_bar]
        add_bar_base_labels(fig, x_plot, y_bar, bar_labels, xaxis="x", yaxis="y", yshift=8)

    fig.update_layout(
        plot_bgcolor=COLORS['bg'],
        paper_bgcolor=COLORS['bg'],
        font=dict(color=COLORS['text'], size=13),
        bargap=0.25,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=20, b=20),
        height=height
    )
    fig.update_yaxes(title_text="Nro de Casos", secondary_y=False, showgrid=True, gridcolor=COLORS['grid'], color=COLORS['axes'])
    fig.update_yaxes(title_text="Promedio Hrs acum.", secondary_y=True, showgrid=False, color=COLORS['axes'])
    return fig

# ---------- Layout: gráficos lado a lado ----------
c1, c2 = st.columns(2, gap="large")

with c1:
    st.subheader("Rendimiento de Neumáticos dado de bajas (por Mes)")
    fig1 = make_combined_bar_line(
        x=g1.index,
        y_bar=g1["nro_casos"].values,
        y_line=g1["prom_hrs"].values,
        x_labels=[month_names.get(i, i) for i in g1.index]
    )
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    st.subheader("Pareto según motivo de baja")
    if len(g2) == 0:
        st.info("No existe columna 'MOTIVO DE BAJA' en los datos.")
    else:
        fig2 = make_combined_bar_line(
            x=g2.index,
            y_bar=g2["nro_casos"].values,
            y_line=g2["prom_hrs"].values,
            x_labels=g2.index,
            height=520
        )
        # Si los números siguen quedando muy juntos, separa más las barras:
        fig2.update_layout(bargap=0.45, bargroupgap=0.25, margin=dict(l=20, r=20, t=20, b=80))
        fig2.update_xaxes(tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True)