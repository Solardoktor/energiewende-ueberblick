"""
Energiewende Überblick
Streamlit-App für skalierte Tagesenergie-Übersicht basierend auf Energy-Charts-Daten.
Annahme: Intra-Day-Abweichungen durch Speicher und Flexibilitäten ausgleichbar.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta, date
import holidays
from dateutil.relativedelta import relativedelta
import locale
from pathlib import Path
import os

# Deutsche Formatierung versuchen
try:
    locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')
except:
    pass

st.set_page_config(
    page_title="Energiewende Überblick",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Header mit Logo ---
col_title, col_logo = st.columns([8, 1])
with col_title:
    st.title("⚡ Energiewende Überblick")
    st.markdown("""
    **Skalierte Tagesenergie-Übersicht für Deutschland**  
    Datenbasis: Energy-Charts.info (Fraunhofer ISE) – Gesamte Nettostromerzeugung.  
    Annahme: Intra-Day-Abweichungen zwischen Last und Erzeugung können vollständig durch Kurzzeitspeicher und Flexibilitäten ausgeglichen werden.  
    Weitere Infos, Erläuterungen und Ergebnisse sowie Platz für Ihren Kommentar finden Sie unter [solardoktor.de/energiewende-ueberblick](https://solardoktor.de/energiewende-ueberblick/).
    """)
with col_logo:
    logo_path = Path("solardoktor_logo.jpg")
    if logo_path.exists():
        import base64
        with open(logo_path, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()
        st.markdown(
            f'<div style="text-align:right;"><a href="https://www.solardoktor.de" target="_blank">'
            f'<img src="data:image/jpeg;base64,{logo_b64}" width="80" style="display:inline-block;"></a></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown('<div style="text-align:right;"><a href="https://www.solardoktor.de" target="_blank"><b>SOLAR DOKTOR</b></a></div>', unsafe_allow_html=True)

# --- Konstanten und Defaults ---
BASE_URL = "https://api.energy-charts.info/v2"
MUNICH_LAT, MUNICH_LON = 48.1351, 11.5820

# Annahmen Mobilität (Konstanten)
# Annahmen Mobilität (angepasst an plausible Jahreswerte)
KWH_PER_100KM_PKW = 18.0          # realistisch inkl. Ladeverluste
ANNUAL_KM_PKW = 12500
KWH_PER_EV_PER_YEAR = ANNUAL_KM_PKW * KWH_PER_100KM_PKW / 100   # ≈ 2250 kWh/a → 40 Mio PKW ≈ 90 TWh

KWH_PER_100KM_LKW = 50.0           # angepasste Annahme (statt 100)
ANNUAL_KM_LKW = 80000
KWH_PER_LKW_PER_YEAR = ANNUAL_KM_LKW * KWH_PER_100KM_LKW / 100  # ≈ 40 000 kWh/a → 4 Mio LKW ≈ 160 TWh

KWH_PER_PERSON_DHW_YEAR = 2000     # 40 Mio EW ≈ 80 TWh (plausibel)
BASE_HEATING_KWH_PER_PERSON_YEAR = 10000  # Direktelektrisch ≈ 400 TWh bei 40 Mio EW; mit WP-Mix deutlich weniger

import os
from pathlib import Path

# --- Lokaler Cache (spart API-Aufrufe) ---
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def cache_path(name):
    return CACHE_DIR / f"{name}.parquet"

def load_cached(name):
    p = cache_path(name)
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            return None
    return None

def save_cached(name, df):
    if df is not None and not df.empty:
        try:
            df.to_parquet(cache_path(name), index=False)
        except Exception:
            pass

@st.cache_data(ttl=3600)
def fetch_energy_charts(endpoint, params):
    """Holt Daten von der Energy-Charts API v2 (mit einfachem Fallback)."""
    url = f"{BASE_URL}/{endpoint}"
    try:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API-Fehler ({endpoint}): {e}")
        return None

@st.cache_data(ttl=3600*24)
def fetch_daily_temperatures(start_date, end_date, lat=MUNICH_LAT, lon=MUNICH_LON):
    """Holt Tagesmitteltemperaturen von Open-Meteo (historisch)."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": "temperature_2m_mean",
        "timezone": "Europe/Berlin"
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame({
            "date": pd.to_datetime(data["daily"]["time"]),
            "temp_mean": data["daily"]["temperature_2m_mean"]
        })
        return df
    except Exception as e:
        st.warning(f"Temperaturdaten konnten nicht geladen werden: {e}. Verwende Schätzwerte.")
        return None

def power_to_daily_energy(json_data):
    """Aggregiert 15-min Power-Daten zu Tagesenergie (GWh). Robust gegen fehlerhafte Timestamps."""
    if not json_data or not isinstance(json_data, dict) or "data" not in json_data or not json_data["data"]:
        return pd.DataFrame()

    records = []
    for rec in json_data["data"]:
        try:
            ts_raw = rec.get("timestamp")
            if not ts_raw:
                continue
            ts = pd.to_datetime(ts_raw, utc=True, errors="coerce")
            if pd.isna(ts):
                continue
            vals = rec.get("values", {}) or {}
            energy = {}
            for k, v in vals.items():
                try:
                    energy[k] = float(v) * 0.25 / 1000.0 if v is not None else 0.0  # MW × 0,25 h = MWh → GWh
                except (TypeError, ValueError):
                    energy[k] = 0.0
            energy["timestamp"] = ts
            records.append(energy)
        except Exception:
            continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Sicherstellen, dass timestamp wirklich datetime ist
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        return pd.DataFrame()

    df["date"] = df["timestamp"].dt.date
    daily = df.groupby("date", as_index=False).sum(numeric_only=True)
    return daily

def get_installed_power_monthly():
    """Holt monatliche installierte Leistungen."""
    data = fetch_energy_charts("installed_power", {"country": "de", "time_step": "monthly"})
    if not data or "data" not in data:
        return pd.DataFrame()
    records = []
    for rec in data["data"]:
        ts = pd.to_datetime(rec["timestamp"])
        vals = rec["values"]
        row = {"month": ts.to_period("M")}
        row.update(vals)
        records.append(row)
    return pd.DataFrame(records)

def german_number(n, decimals=1):
    """Deutsche Zahlenformatierung."""
    if pd.isna(n):
        return "-"
    return f"{n:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")

# --- Sidebar ---
st.sidebar.header("Einstellungen")

period_mode = st.sidebar.radio(
    "Zeitraum",
    ["Letzte 365 Tage", "Kalenderjahr"],
    index=0
)

# Zeitraum: bei "Letzte 365 Tage" bis Ende Vormonat (vermeidet unvollständige aktuelle Daten)
if period_mode == "Kalenderjahr":
    current_year = datetime.now().year
    # Nur vollständig abgeschlossene Jahre (aktuelles Jahr ausgeschlossen)
    available_years = list(range(current_year - 1, 2014, -1))
    year = st.sidebar.selectbox("Jahr", available_years, index=0)
    start_date = date(year, 1, 1)
    end_date = date(year, 12, 31)
    st.sidebar.caption(f"Letztes vollständiges Jahr: {year}")
else:
    today = date.today()
    # Ende = letzter Tag des Vormonats
    end_date = (today.replace(day=1) - timedelta(days=1))
    start_date = end_date - timedelta(days=364)
    year = None
    st.sidebar.caption(f"Zeitraum: {start_date.strftime('%d.%m.%Y')} – {end_date.strftime('%d.%m.%Y')}")

load_scale = st.sidebar.slider("Last-Skalierung (%)", 50, 200, 100, 5) / 100.0

st.sidebar.subheader("Ausbauszenarien")
scenario = st.sidebar.radio(
    "Szenario",
    ["Status Quo", "Ziele der Bundesregierung für 2030", "Agora Energiewende: Klimaneutrales Deutschland 2045"],
    index=0
)

# Ziele je Szenario (GWp / GW)
scenarios = {
    "Status Quo": {"solar": 130.0, "wind_onshore": 70.0, "wind_offshore": 12.0},
    "Ziele der Bundesregierung für 2030": {"solar": 215.0, "wind_onshore": 115.0, "wind_offshore": 30.0},
    "Agora Energiewende: Klimaneutrales Deutschland 2045": {"solar": 400.0, "wind_onshore": 180.0, "wind_offshore": 70.0},
}
sc = scenarios[scenario]
st.sidebar.caption(f"Solar: {sc['solar']} GWp | Wind Onshore: {sc['wind_onshore']} GW | Offshore: {sc['wind_offshore']} GW")

# Manuelle Überschreibung optional
use_manual = st.sidebar.checkbox("Manuelle Ziele verwenden", value=False)
disable_monthly_scaling = st.sidebar.checkbox("Monatliche Skalierung aus", value=False,
    help="Setzt die Skalierung der installierten Leistung für alle Monate auf 1 (Ziel = Ist). Zur Kontrolle gegenüber Energy-Charts.")
disable_pumped_storage = st.sidebar.checkbox("Pumpspeicher/-Ladung aus", value=True,
    help="Schließt Pumpspeicher und Pumpspeicher-Ladung aus der Berechnung aus.")
if use_manual:
    target_solar_gwp = st.sidebar.number_input("Ziel Solar (GWp)", 50.0, 600.0, sc["solar"], 5.0)
    target_wind_onshore_gw = st.sidebar.number_input("Ziel Wind Onshore (GW)", 30.0, 250.0, sc["wind_onshore"], 5.0)
    target_wind_offshore_gw = st.sidebar.number_input("Ziel Wind Offshore (GW)", 5.0, 100.0, sc["wind_offshore"], 1.0)
else:
    target_solar_gwp = sc["solar"]
    target_wind_onshore_gw = sc["wind_onshore"]
    target_wind_offshore_gw = sc["wind_offshore"]

st.sidebar.subheader("Sektorenkopplung – Zusatzlasten (über aktuellen Bestand hinaus)")
st.sidebar.caption("Defaults entsprechen ungefähr dem aktuellen Bestand. Die Eingabe ist der Gesamtzielwert; die Differenz zum Bestand wird als Zusatzlast modelliert.")

CURRENT_EV = 2
CURRENT_LKW = 0.0
CURRENT_DHW = 15
CURRENT_HEATING = 5

# --- Sektor Elektromobilität ---
st.sidebar.markdown("**Elektromobilität**")
mio_ev = st.sidebar.number_input("Millionen Elektro-PKW (2 … 40, gesamt)", CURRENT_EV, 40, CURRENT_EV, 1)
mio_lkw = st.sidebar.number_input("Millionen Elektro-LKW (0 … 4, gesamt)", CURRENT_LKW, 4.0, CURRENT_LKW, 0.1, format="%.1f")
weekend_share = st.sidebar.slider("Anteil Aufladung bevorzugt Wochenende/Feiertag (0 … 50 %)", 0, 50, 20) / 100.0

# --- Sektor Wärme ---
st.sidebar.markdown("**Wärme**")
mio_inhabitants_dhw = st.sidebar.number_input("Millionen Einwohner elektrische Warmwasserbereitung (15 … 80, gesamt)", CURRENT_DHW, 80, CURRENT_DHW, 1)
mio_inhabitants_heating = st.sidebar.number_input("Millionen Einwohner elektrische Heizung (5 … 80, gesamt)", CURRENT_HEATING, 80, CURRENT_HEATING, 1)

st.sidebar.markdown("**Heizung – Anteile (Summe ≤ 100 %)**")
hp_luft_share = st.sidebar.slider("Anteil Luft-Wasser-WP (%)", 0, 100, 60)
hp_wasser_share = st.sidebar.slider("Anteil Wasser-Wasser-/Sole-WP (%)", 0, 100, 10)
direct_share = max(0, 100 - hp_luft_share - hp_wasser_share)
st.sidebar.caption(f"Direktelektrisch: {direct_share} %")

# Zusatzlast = Differenz zum gerundeten Bestand
add_ev = max(0, int(mio_ev) - CURRENT_EV)
add_lkw = max(0.0, round(float(mio_lkw) - CURRENT_LKW, 1))
add_dhw = max(0, int(mio_inhabitants_dhw) - CURRENT_DHW)
add_heating = max(0, int(mio_inhabitants_heating) - CURRENT_HEATING)

st.sidebar.caption(f"Zusatzverbrauch (über Stand von 2026): +{add_ev:.1f} Mio PKW, +{add_lkw:.1f} Mio LKW, +{add_dhw:.0f} Mio EW WW, +{add_heating:.0f} Mio EW Heizung")

# Mobilität berechnen (nach den Eingaben)
# Mobilität nur als Zusatzlast (Toleranz gegen Rundung)
add_ev_safe = max(0.0, float(mio_ev) - CURRENT_EV)
add_lkw_safe = max(0.0, float(mio_lkw) - CURRENT_LKW)
if abs(add_ev_safe) < 0.05:
    add_ev_safe = 0.0
if abs(add_lkw_safe) < 0.05:
    add_lkw_safe = 0.0
annual_ev_gwh = add_ev_safe * 1e6 * KWH_PER_EV_PER_YEAR / 1e6
annual_lkw_gwh = add_lkw_safe * 1e6 * KWH_PER_LKW_PER_YEAR / 1e6
annual_mobility_gwh = annual_ev_gwh + annual_lkw_gwh

# (alte doppelte Inputs entfernt)

# --- Daten laden (mit lokalem Cache) ---
cache_key_power = f"power_{start_date}_{end_date}"
cache_key_inst = "installed_monthly"
cache_key_temp = f"temp_{start_date}_{end_date}"

with st.spinner("Lade Daten (Cache oder API)..."):
    # Power-Daten
    daily_energy = load_cached(cache_key_power)
    if daily_energy is None or daily_energy.empty:
        params = {"country": "de", "start": start_date.isoformat(), "end": end_date.isoformat()}
        power_json = fetch_energy_charts("total_power", params)
        daily_energy = power_to_daily_energy(power_json)
        if not daily_energy.empty:
            save_cached(cache_key_power, daily_energy)
            st.success("Neue Power-Daten geladen und lokal gecacht.")
    else:
        st.info("Power-Daten aus lokalem Cache geladen.")

    # Installierte Leistung
    installed = load_cached(cache_key_inst)
    if installed is None or installed.empty:
        installed = get_installed_power_monthly()
        if not installed.empty:
            save_cached(cache_key_inst, installed)

    # Temperaturen
    temps = load_cached(cache_key_temp)
    if temps is None or temps.empty:
        temps = fetch_daily_temperatures(start_date, end_date)
        if temps is not None and not temps.empty:
            save_cached(cache_key_temp, temps)

if daily_energy.empty:
    st.error("Keine Energiedaten erhalten. Bitte Zeitraum prüfen oder später erneut versuchen.")
    st.stop()

# --- Verarbeitung ---
daily_energy["date"] = pd.to_datetime(daily_energy["date"])
daily_energy = daily_energy.sort_values("date")

# Installierte Leistung zuordnen (vereinfacht: Summe relevanter EE)
if not installed.empty:
    # Beispiel: solar_ac + wind_onshore + wind_offshore + biomass + hydro...
    ee_cols = [c for c in installed.columns if c not in ["month"] and "solar" in c or "wind" in c or "biomass" in c or "hydro" in c]
    # Für Demo: verwende eine Näherung, falls Spalten vorhanden
    daily_energy["month"] = daily_energy["date"].dt.to_period("M")
    # Merge grob
    # (Vollständige Implementierung würde die genauen Serien-IDs mappen)

# Last skalieren
if "load_incl_self_consumption" in daily_energy.columns:
    daily_energy["base_load_gwh"] = daily_energy["load_incl_self_consumption"] * load_scale
else:
    # Fallback
    daily_energy["base_load_gwh"] = daily_energy.filter(like="load").sum(axis=1) * load_scale

# Sektorenlasten
de_holidays = holidays.Germany(years=range(start_date.year, end_date.year + 1))

def is_weekend_or_holiday(d):
    return d.weekday() >= 5 or d in de_holidays

# Mobilität – vektorisiert (PKW + LKW)
month_factors = {1: 0.85, 2: 0.85, 3: 0.95, 4: 1.0, 5: 1.05, 6: 1.15, 7: 1.2, 8: 1.15, 9: 1.05, 10: 0.95, 11: 0.85, 12: 0.9}
daily_energy["month_factor"] = daily_energy["date"].dt.month.map(month_factors).fillna(1.0)

daily_energy["is_weekend_or_holiday"] = daily_energy["date"].apply(
    lambda d: d.weekday() >= 5 or d in de_holidays
)
weekend_boost = np.where(daily_energy["is_weekend_or_holiday"] & (weekend_share > 0), 1.3, 1.0)

daily_energy["mobility_raw"] = (annual_mobility_gwh / 365) * daily_energy["month_factor"] * weekend_boost
total_raw = daily_energy["mobility_raw"].sum()
if total_raw > 0:
    daily_energy["mobility_gwh"] = daily_energy["mobility_raw"] * (annual_mobility_gwh / total_raw)
else:
    daily_energy["mobility_gwh"] = 0.0
daily_energy = daily_energy.drop(columns=["mobility_raw", "month_factor", "is_weekend_or_holiday"], errors="ignore")

# Warmwasser (nur Zusatz über aktuellen Bestand)
annual_dhw_gwh = add_dhw * 1e6 * KWH_PER_PERSON_DHW_YEAR / 1e6
is_winter = daily_energy["date"].dt.month.isin([11, 12, 1, 2, 3])
daily_energy["dhw_gwh"] = annual_dhw_gwh / 365 * np.where(is_winter, 1.1, 0.95)

# Heizung (nur Zusatz)
annual_heating_direct_gwh = add_heating * 1e6 * BASE_HEATING_KWH_PER_PERSON_YEAR / 1e6
# Temperaturabhängig
if temps is not None and not temps.empty:
    daily_energy = daily_energy.merge(temps, left_on="date", right_on="date", how="left")
    # Heizgradtag-ähnliche Skalierung: höher bei niedriger Temp
    # Referenz 15°C, linear
    daily_energy["heating_factor"] = np.clip((15 - daily_energy["temp_mean"]) / 15, 0, 2)
    daily_energy["heating_factor"] = daily_energy["heating_factor"].fillna(1.0)
else:
    # Fallback saisonal
    daily_energy["heating_factor"] = daily_energy["date"].dt.month.map(
        {1: 1.8, 2: 1.6, 3: 1.2, 4: 0.6, 5: 0.2, 6: 0.05, 7: 0.0, 8: 0.05, 9: 0.3, 10: 0.8, 11: 1.3, 12: 1.7}
    ).fillna(1.0)

daily_energy["heating_direct_gwh"] = (annual_heating_direct_gwh / 365) * daily_energy["heating_factor"]

# Wärmepumpen: zwei Typen gleichzeitig + Rest Direktelektrisch
def cop_luft(temp):
    if pd.isna(temp):
        temp = 5
    return max(1.8, min(4.5, 2.2 + 0.08 * temp))

def cop_wasser(temp):
    if pd.isna(temp):
        temp = 5
    return max(3.5, min(5.5, 4.0 + 0.03 * temp))

if "temp_mean" in daily_energy.columns:
    daily_energy["cop_luft"] = daily_energy["temp_mean"].apply(cop_luft)
    daily_energy["cop_wasser"] = daily_energy["temp_mean"].apply(cop_wasser)
else:
    daily_energy["cop_luft"] = 3.0
    daily_energy["cop_wasser"] = 4.2

# Anteile (0–1)
share_luft = hp_luft_share / 100.0
share_wasser = hp_wasser_share / 100.0
share_direct = max(0.0, 1.0 - share_luft - share_wasser)

daily_energy["heating_gwh"] = (
    daily_energy["heating_direct_gwh"] * share_direct +
    (daily_energy["heating_direct_gwh"] * share_luft) / daily_energy["cop_luft"] +
    (daily_energy["heating_direct_gwh"] * share_wasser) / daily_energy["cop_wasser"]
)

# Gesamtlast
daily_energy["total_load_gwh"] = (
    daily_energy["base_load_gwh"] +
    daily_energy["mobility_gwh"] +
    daily_energy["dhw_gwh"] +
    daily_energy["heating_gwh"]
)

# --- Verarbeitung ---
daily_energy["date"] = pd.to_datetime(daily_energy["date"])
daily_energy = daily_energy.sort_values("date")
daily_energy["month"] = daily_energy["date"].dt.to_period("M")

# Installierte Leistung mergen (solar_dc = Nennleistung DC / kWp)
if not installed.empty:
    installed = installed.rename(columns={
        "solar_dc": "installed_solar",
        "wind_onshore": "installed_wind_onshore",
        "wind_offshore": "installed_wind_offshore"
    })
    daily_energy = daily_energy.merge(
        installed[["month", "installed_solar", "installed_wind_onshore", "installed_wind_offshore"]],
        on="month", how="left"
    )
    for col in ["installed_solar", "installed_wind_onshore", "installed_wind_offshore"]:
        if col in daily_energy.columns:
            daily_energy[col] = daily_energy[col].ffill().bfill()
else:
    daily_energy["installed_solar"] = 130
    daily_energy["installed_wind_onshore"] = 70
    daily_energy["installed_wind_offshore"] = 11

# Skalierung pro Typ
scale_factor = 1.0 if disable_monthly_scaling else None
if "solar" in daily_energy.columns:
    if scale_factor == 1.0:
        daily_energy["solar_scaled"] = daily_energy["solar"]
    else:
        daily_energy["solar_scaled"] = daily_energy["solar"] * (
            target_solar_gwp / daily_energy["installed_solar"].replace(0, np.nan)
        )
        daily_energy["solar_scaled"] = daily_energy["solar_scaled"].fillna(0)
else:
    daily_energy["solar_scaled"] = 0

if "wind_onshore" in daily_energy.columns:
    if scale_factor == 1.0:
        daily_energy["wind_onshore_scaled"] = daily_energy["wind_onshore"]
    else:
        daily_energy["wind_onshore_scaled"] = daily_energy["wind_onshore"] * (
            target_wind_onshore_gw / daily_energy["installed_wind_onshore"].replace(0, np.nan)
        )
        daily_energy["wind_onshore_scaled"] = daily_energy["wind_onshore_scaled"].fillna(0)
else:
    daily_energy["wind_onshore_scaled"] = 0

if "wind_offshore" in daily_energy.columns:
    if scale_factor == 1.0:
        daily_energy["wind_offshore_scaled"] = daily_energy["wind_offshore"]
    else:
        daily_energy["wind_offshore_scaled"] = daily_energy["wind_offshore"] * (
            target_wind_offshore_gw / daily_energy["installed_wind_offshore"].replace(0, np.nan)
        )
        daily_energy["wind_offshore_scaled"] = daily_energy["wind_offshore_scaled"].fillna(0)
else:
    daily_energy["wind_offshore_scaled"] = 0

daily_energy["ee_scaled"] = (
    daily_energy["solar_scaled"] +
    daily_energy["wind_onshore_scaled"] +
    daily_energy["wind_offshore_scaled"]
)

# Sonstige EE vor der Grafik berechnen
# Sonstige EE (optional ohne Pumpspeicher)
other_ee_keywords = [
    "biomass", "hydro_run", "hydro_water", "hydro_storage", "geothermal",
    "waste", "other", "run_of_river", "hydro"
]
if disable_pumped_storage:
    other_ee_keywords = [k for k in other_ee_keywords if "pumped" not in k and "storage" not in k]

other_ee_cols = [c for c in daily_energy.columns if any(k in c.lower() for k in other_ee_keywords)
                 and "wind" not in c.lower() and "solar" not in c.lower()]
if other_ee_cols:
    daily_energy["other_ee"] = daily_energy[other_ee_cols].sum(axis=1)
else:
    daily_energy["other_ee"] = 0.0

# ee_day und Superpower/Residual vor der Grafik berechnen
daily_energy["ee_day"] = (
    daily_energy.get("solar_scaled", 0) +
    daily_energy.get("wind_onshore_scaled", 0) +
    daily_energy.get("wind_offshore_scaled", 0) +
    daily_energy.get("other_ee", 0)
)
daily_energy["ee_minus_load"] = daily_energy["ee_day"] - daily_energy["total_load_gwh"]
daily_energy["superpower_day"] = np.maximum(0, daily_energy["ee_minus_load"])
daily_energy["residual_day"] = np.maximum(0, -daily_energy["ee_minus_load"])

# --- Visualisierung: gestapelt durch Überdeckung (effizienter) ---
try:
    st.subheader("Tagesenergie: Skalierte Erzeugung und Last")

    # Sicherstellen, dass benötigte Spalten existieren
    for col in ["base_load_gwh", "mobility_gwh", "dhw_gwh", "heating_gwh",
                "solar_scaled", "wind_onshore_scaled", "wind_offshore_scaled"]:
        if col not in daily_energy.columns:
            daily_energy[col] = 0.0

    fig = go.Figure()

    colors = {
        "base": "#808080",          # Mittelgrau, deutlich heller
        "mobility": "#FF99FF",      # weniger grell
        "dhw": "#C2185B",           # Lila ins Rötliche
        "heating": "#d32f2f",
        "solar": "#FFCC00",
        "wind_onshore": "#7CB342",
        "wind_offshore": "#388E3C",
        "other_ee": "#81C784",
    }

    # Superpower ganz hinten: nur der Überschuss (positiv), vollflächig hellgelb
    fig.add_trace(go.Bar(
        x=daily_energy["date"], y=daily_energy["ee_day"],
        name="Superpower",
        marker_color="#FFF176",
        opacity=0.95,
        hovertemplate="EE − Last: %{customdata:.1f} GWh<extra></extra>",
        customdata=daily_energy["ee_minus_load"]
    ))

    # Last im Hintergrund: von hinten nach vorne (höchste Summe zuerst, Basis-Last zuletzt = vorne)
    load_total = (daily_energy["base_load_gwh"] + daily_energy["mobility_gwh"] +
                  daily_energy["dhw_gwh"] + daily_energy["heating_gwh"])
    fig.add_trace(go.Bar(
        x=daily_energy["date"], y=load_total,
        name="Heizung", marker_color=colors["heating"], opacity=0.85,
        hovertemplate="Heizung: %{customdata:.1f} GWh<extra></extra>",
        customdata=daily_energy["heating_gwh"]
    ))

    load_wo_heat = load_total - daily_energy["heating_gwh"]
    fig.add_trace(go.Bar(
        x=daily_energy["date"], y=load_wo_heat,
        name="Warmwasser", marker_color=colors["dhw"], opacity=0.85,
        hovertemplate="Warmwasser: %{customdata:.1f} GWh<extra></extra>",
        customdata=daily_energy["dhw_gwh"]
    ))

    load_wo_dhw = load_wo_heat - daily_energy["dhw_gwh"]
    fig.add_trace(go.Bar(
        x=daily_energy["date"], y=load_wo_dhw,
        name="Mobilität", marker_color=colors["mobility"], opacity=0.85,
        hovertemplate="Mobilität: %{customdata:.1f} GWh<extra></extra>",
        customdata=daily_energy["mobility_gwh"]
    ))

    fig.add_trace(go.Bar(
        x=daily_energy["date"], y=daily_energy["base_load_gwh"],
        name="Basis-Last", marker_color=colors["base"], opacity=0.9,
        hovertemplate="Basis-Last: %{y:.1f} GWh<extra></extra>"
    ))

    # Erzeugung: Sonstige ganz vorne, dann Offshore, Onshore, Solar
    ee_total = (daily_energy["solar_scaled"] + daily_energy["wind_onshore_scaled"] +
                daily_energy["wind_offshore_scaled"] + daily_energy.get("other_ee", 0))
    fig.add_trace(go.Bar(
        x=daily_energy["date"], y=ee_total,
        name="Solar",
        marker=dict(
            color="rgba(255,204,0,0.35)",
            pattern_shape="/",
            pattern_fgcolor="#FFCC00",
            pattern_size=5,
            pattern_solidity=0.4
        ),
        opacity=1.0,
        hovertemplate="Solar: %{customdata:.1f} GWh<extra></extra>",
        customdata=daily_energy["solar_scaled"]
    ))

    ee_wo_sol = ee_total - daily_energy["solar_scaled"]
    fig.add_trace(go.Bar(
        x=daily_energy["date"], y=ee_wo_sol,
        name="Wind Onshore",
        marker=dict(
            color="rgba(124,179,66,0.35)",
            pattern_shape="/",
            pattern_fgcolor="#7CB342",
            pattern_size=5,
            pattern_solidity=0.4
        ),
        opacity=1.0,
        hovertemplate="Wind Onshore: %{customdata:.1f} GWh<extra></extra>",
        customdata=daily_energy["wind_onshore_scaled"]
    ))

    ee_wo_on = ee_wo_sol - daily_energy["wind_onshore_scaled"]
    fig.add_trace(go.Bar(
        x=daily_energy["date"], y=ee_wo_on,
        name="Wind Offshore",
        marker=dict(
            color="rgba(56,142,60,0.35)",
            pattern_shape="/",
            pattern_fgcolor="#388E3C",
            pattern_size=5,
            pattern_solidity=0.4
        ),
        opacity=1.0,
        hovertemplate="Wind Offshore: %{customdata:.1f} GWh<extra></extra>",
        customdata=daily_energy["wind_offshore_scaled"]
    ))

    if "other_ee" in daily_energy.columns and daily_energy["other_ee"].sum() > 0:
        fig.add_trace(go.Bar(
            x=daily_energy["date"], y=daily_energy["other_ee"],
            name="Sonstige EE",
            marker=dict(
                color="rgba(25,118,210,0.35)",
                pattern_shape="/",
                pattern_fgcolor="#1976D2",
                pattern_size=5,
                pattern_solidity=0.4
            ),
            opacity=1.0,
            hovertemplate="Sonstige EE: %{y:.1f} GWh<extra></extra>"
        ))

    # Titel in einer Zeile + Legende unten
    period_str = f"{start_date.strftime('%d.%m.%Y')} – {end_date.strftime('%d.%m.%Y')}"
    wind_total = target_wind_onshore_gw + target_wind_offshore_gw
    title_main = f"Tagesbilanz von Stromerzeugung und Verbrauch in Deutschland  {period_str}"
    subtitle = (
        f"Gesamt-Nettostromerzeugung bei {target_solar_gwp:.0f} GWp Solar und {wind_total:.0f} GW Wind<br>"
        f"Gesamt-Last aus Basis-Last, e-PKW ({mio_ev:.1f} Mio.), e-LKW ({mio_lkw:.1f} Mio.), "
        f"e-Warmwasser ({mio_inhabitants_dhw:.0f} Mio. EW), e-Heizung ({mio_inhabitants_heating:.0f} Mio. EW)"
    )

    fig.update_layout(
        barmode="overlay",
        title=dict(text=title_main + "<br><span style='font-size:13px'>" + subtitle + "</span>",
                   font=dict(size=18), x=0.5, xanchor="center"),
        xaxis=dict(
            title="",
            showticklabels=True,
            tickformat="%d.%m.",
            range=[start_date, end_date],
            showgrid=True,
            gridcolor="rgba(200,200,200,0.25)"
        ),
        yaxis=dict(title="Energie (GWh)", showgrid=True, gridcolor="rgba(200,200,200,0.25)"),
        legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5, font=dict(size=10)),
        height=760,
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=30, t=100, b=140),
        bargap=0,
        bargroupgap=0,
        annotations=[
            dict(
                text="Hinweis: Legende nutzt kumulierte Überdeckung – nur zur Orientierung, keine Neuberechnung bei Abwahl.<br>" +
                     "(c) Dr. Andreas Horn, www.solardoktor.de  |  Daten: Energy-Charts.info, Open-Meteo  |  Stand: " + datetime.now().strftime("%d.%m.%Y"),
                xref="paper", yref="paper",
                x=0.5, y=-0.20,
                showarrow=False,
                font=dict(size=11, color="#333333"),
                xanchor="center"
            )
        ]
    )

    st.plotly_chart(fig, width="stretch")
except Exception as e:
    st.error(f"Fehler bei der Grafikerstellung: {e}")
    st.exception(e)
    st.stop()

# Sonstige EE früh berechnen (vor der Grafik)
# (zweite other_ee-Berechnung entfernt – die erste mit Schalter gilt)

solar_j = daily_energy["solar_scaled"].sum()
wind_on_j = daily_energy["wind_onshore_scaled"].sum()
wind_off_j = daily_energy["wind_offshore_scaled"].sum()
other_j = daily_energy["other_ee"].sum()
ee_sum = solar_j + wind_on_j + wind_off_j + other_j

load_j = daily_energy["base_load_gwh"].sum()
mob_j = daily_energy["mobility_gwh"].sum()
dhw_j = daily_energy["dhw_gwh"].sum()
heat_j = daily_energy["heating_gwh"].sum()
cons_sum = load_j + mob_j + dhw_j + heat_j

# EE-Anteil gewichtet
daily_energy["ee_share_day"] = np.minimum(1.0, daily_energy["ee_day"] / daily_energy["total_load_gwh"].replace(0, np.nan))
ee_share_weighted = (daily_energy["ee_share_day"] * daily_energy["total_load_gwh"]).sum() / cons_sum if cons_sum > 0 else 0
residual_energy = daily_energy["residual_day"].sum()
superpower = daily_energy["superpower_day"].sum()
residual_power_gw = (daily_energy["residual_day"].max() / 24)  # GW

co2_specific = st.sidebar.number_input("Spezifische CO₂-Intensität Residual (g/kWh)", 200, 1500, 1000, 50)
co2_mean = (residual_energy / cons_sum * co2_specific) if cons_sum > 0 else 0

# --- Zusammenfassungstabelle (TWh, 1 Nachkommastelle) ---
st.subheader("Jahresbilanz")

def to_twh(gwh):
    return gwh / 1000.0

solar_t = to_twh(solar_j)
wind_on_t = to_twh(wind_on_j)
wind_off_t = to_twh(wind_off_j)
other_t = to_twh(other_j)
ee_sum_t = to_twh(ee_sum)
residual_t = to_twh(residual_energy)

load_t = to_twh(load_j)
mob_t = to_twh(mob_j)
dhw_t = to_twh(dhw_j)
heat_t = to_twh(heat_j)
cons_sum_t = to_twh(cons_sum)
super_t = to_twh(superpower)

# Tagesweiser Residualanteil
daily_energy["residual_share"] = np.where(
    daily_energy["total_load_gwh"] > 0,
    daily_energy["residual_day"] / daily_energy["total_load_gwh"],
    0.0
)

# CO2 pro Verbrauchskategorie (tagesgewichtet nach Verbrauch der Kategorie)
def weighted_co2(col):
    if col not in daily_energy.columns or daily_energy[col].sum() == 0:
        return 0.0
    return (daily_energy[col] * daily_energy["residual_share"] * co2_specific).sum() / daily_energy[col].sum()

co2_base = weighted_co2("base_load_gwh")
co2_mob = weighted_co2("mobility_gwh")
co2_dhw = weighted_co2("dhw_gwh")
co2_heat = weighted_co2("heating_gwh")

# Dunkelflaute = Tage zwischen Superpower-Tagen (EE > Verbrauch), zyklisch
surplus_mask = (daily_energy["superpower_day"] > 0).values
n = len(surplus_mask)
total_deficit_days = int((daily_energy["residual_day"] > 0).sum())
total_super_days = int((daily_energy["superpower_day"] > 0).sum())

def find_longest_period(is_true_mask, energy_col):
    if not is_true_mask.any():
        return "0 Tage", 0.0, None, None
    extended = np.concatenate([is_true_mask, is_true_mask])
    groups = (extended != np.roll(extended, 1)).cumsum()
    lengths = pd.Series(extended.astype(int)).groupby(groups).sum()
    pos = lengths[lengths > 0]
    if pos.empty:
        return "0 Tage", 0.0, None, None
    max_d = int(pos.max())
    if max_d >= n:
        max_d = n
        s_idx, e_idx = 0, n - 1
    else:
        lg = pos.idxmax()
        idxs = np.where(groups == lg)[0]
        s_idx = idxs[0] % n
        e_idx = idxs[-1] % n
    s_date = daily_energy["date"].iloc[s_idx]
    e_date = daily_energy["date"].iloc[e_idx]
    label = f"{max_d} Tage ({s_date.strftime('%d.%m.')} bis {e_date.strftime('%d.%m.')})"
    if s_idx <= e_idx:
        twh = daily_energy[energy_col].iloc[s_idx:e_idx+1].sum() / 1000
    else:
        twh = (daily_energy[energy_col].iloc[s_idx:].sum() + daily_energy[energy_col].iloc[:e_idx+1].sum()) / 1000
    return label, twh, s_date, e_date

dunkelflaute_str, dunkelflaute_twh, _, _ = find_longest_period(~surplus_mask, "residual_day")
super_str, super_twh, _, _ = find_longest_period(surplus_mask, "superpower_day")

# Höchste Superleistung
if daily_energy["superpower_day"].max() > 0:
    max_super_idx = daily_energy["superpower_day"].idxmax()
    max_super_date = daily_energy.loc[max_super_idx, "date"]
    max_super_gw = daily_energy.loc[max_super_idx, "superpower_day"] / 24
    max_super_date_str = max_super_date.strftime("%d.%m.")
else:
    max_super_gw = 0.0
    max_super_date_str = "n/a"

# Top-10 Dunkelflauten
def get_top_periods(is_deficit_mask, energy_col, top_n=10):
    """Nur echte Dunkelflauten-Perioden (Defizit), keine Superpower-Perioden."""
    if not is_deficit_mask.any():
        return []
    # Nur zusammenhängende True-Abschnitte (Defizit)
    groups = (is_deficit_mask != np.roll(is_deficit_mask, 1)).cumsum()
    periods = []
    for g in np.unique(groups):
        idxs = np.where(groups == g)[0]
        if len(idxs) == 0 or not is_deficit_mask[idxs[0]]:
            continue  # überspringe Superpower-Abschnitte
        length = len(idxs)
        s_idx = idxs[0]
        e_idx = idxs[-1]
        deficit = daily_energy[energy_col].iloc[s_idx:e_idx+1].sum()
        periods.append((length, s_idx, deficit / 1000))
    periods = sorted(periods, key=lambda x: -x[0])[:top_n]
    result = []
    for i, (length, s_idx, twh) in enumerate(periods, 1):
        s_date = daily_energy["date"].iloc[s_idx]
        result.append({"Nr": i, "Starttag": s_date.strftime("%d.%m."), "Dauer [Tage]": length, "Gesamtdefizit [TWh]": german_number(twh, 1)})
    return result

top_df = pd.DataFrame(get_top_periods(~surplus_mask, "residual_day", 10))

# Heizgradtage (vereinfacht, Referenz 15 °C)
if "temp_mean" in daily_energy.columns:
    hgt = np.maximum(0, 15 - daily_energy["temp_mean"]).sum()
else:
    hgt = None

summary_data = {
    "Erzeugung": ["Solar", "Wind Onshore", "Wind Offshore", "Sonstige EE", "Summe EE", "Residualerzeugung"],
    "Wert [TWh]": [
        german_number(solar_t, 1), german_number(wind_on_t, 1), german_number(wind_off_t, 1),
        german_number(other_t, 1), german_number(ee_sum_t, 1), german_number(residual_t, 1)
    ],
    "(Mehr-) Verbrauch": ["Basis-Last", "Mobilität", "Warmwasser", "Heizung", "Summe Verbrauch", "Superpower"],
    "Wert [TWh] ": [
        german_number(load_t, 1), german_number(mob_t, 1), german_number(dhw_t, 1),
        german_number(heat_t, 1), german_number(cons_sum_t, 1), german_number(super_t, 1)
    ],
    "CO₂ [g/kWh]": [
        german_number(co2_base, 0), german_number(co2_mob, 0),
        german_number(co2_dhw, 0), german_number(co2_heat, 0),
        german_number(co2_mean, 0), "0"
    ]
}
summary = pd.DataFrame(summary_data)
st.dataframe(summary, width="stretch", hide_index=True)

# Datum der höchsten Residualleistung
if "residual_day" in daily_energy.columns and daily_energy["residual_day"].max() > 0:
    max_res_idx = daily_energy["residual_day"].idxmax()
    max_res_date = daily_energy.loc[max_res_idx, "date"]
    max_res_date_str = max_res_date.strftime("%d.%m.")
else:
    max_res_date_str = "n/a"

st.metric("Längste Superpower / Gesamttage mit Superpower / Höchste Superleistung (Tag)",
         f"{super_str} / {total_super_days} Tage / {german_number(max_super_gw, 1)} GW ({max_super_date_str})")

st.metric("Längste Dunkelflaute / Gesamttage mit EE-Defizit / Höchste Residualleistung (Tag)", 
         f"{dunkelflaute_str} / {total_deficit_days} Tage / {german_number(residual_power_gw, 1)} GW ({max_res_date_str})")

if not top_df.empty:
    st.subheader("Top-10 Dunkelflauten")
    st.dataframe(top_df, width="stretch", hide_index=True, column_config={
        "Nr": st.column_config.NumberColumn(width="small"),
        "Starttag": st.column_config.TextColumn(),
        "Dauer [Tage]": st.column_config.NumberColumn(),
        "Gesamtdefizit [TWh]": st.column_config.TextColumn()
    })

st.caption("""
**Dunkelflaute:** „Eine Dunkelflaute ist dadurch gekennzeichnet, dass die Tagesverbräuche über mehr als einen Tag nicht mehr vollständig aus Solar- und Windkrafterzeugung, sowie Ausspeicherung aus intra-diem-Speichern gedeckt werden kann und Residuallastkraftwerke mit gesicherter Leistung zur Deckung der Stromlücke verwendet werden müssen.“  
→ [Definition auf solardoktor.de](https://solardoktor.de/definition-dunkelflaute/)
""")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("EE-Anteil am Verbrauch", german_number(ee_share_weighted * 100, 1) + " %")
with col2:
    avg_temp = daily_energy["temp_mean"].mean() if "temp_mean" in daily_energy.columns else None
    st.metric("Ø Temperatur", german_number(avg_temp, 1) + " °C" if avg_temp else "n/a")
with col3:
    st.metric("Heizgradtage (15 °C)", german_number(hgt, 0) if hgt else "n/a")

# Erklärungstext
with st.expander("Erläuterung zur Darstellung und Annahmen", expanded=True):
    st.markdown("""
**Annahme zur Intra-Day-Ausgleichbarkeit**  
Die Tages-Energiewerte geben einen Überblick zur Energiewende unter der Annahme, dass Intra-Day-Abweichungen zwischen Last und Erzeugung vollständig durch Kurzzeitspeicher und Flexibilitäten (Intra-Day-Batteriespeicher, Lastverschiebung beim Laden von Elektrofahrzeugen, V2G, Wärmepufferspeicher etc.) ausgeglichen werden können.  

Dies ist Stand heute noch nicht in vollem Umfang möglich, aber mittelfristig aufgrund des massiven Ausbaus von Speichern und Flexibilitätsoptionen machbar.

**Datenbasis**  
- Gesamte Nettostromerzeugung (total_power) von Energy-Charts.info / Fraunhofer ISE  
- Skalierung der Erzeugung anhand monatlicher installierter Leistung  
- Zusatzlasten für Sektorenkopplung (Mobilität, Warmwasser, Heizung) mit saisonalen und temperaturabhängigen Mustern  
- Temperaturdaten: Open-Meteo (München als Default)

**Lizenz**  
Daten: CC BY 4.0, Attribution: energy-charts.info
    """)

# Debug: tägliche Differenz
with st.expander("Debug: tägliche Differenz EE – Verbrauch", expanded=False):
    st.markdown("""
**Spalten-Erklärung**
- `ee_day`: Summe der skalierten EE (Solar + Wind Onshore + Wind Offshore + Sonstige EE)
- `total_load_gwh`: Gesamtlast (Basis-Last + Zusatzlasten)
- `ee_minus_load` = `ee_day` − `total_load_gwh`
- `superpower_day` = max(0, ee_minus_load) → positiver Überschuss
- `residual_day` = max(0, −ee_minus_load) → fehlende Energie

Hinweis: Kleine Abweichungen zu Energy-Charts-Excel können durch unterschiedliche Aggregation (15-min → Tag), Rundung oder fehlende/zusätzliche Serien in der API entstehen. Die Sonstigen EE umfassen alle nicht-Wind/Solar-Serien, die das Keyword-Mapping trifft.
    """)
    debug_df = daily_energy[["date", "ee_day", "total_load_gwh", "ee_minus_load", "superpower_day", "residual_day"]].copy()
    debug_df["date"] = debug_df["date"].dt.strftime("%d.%m.%Y")
    st.dataframe(debug_df, width="stretch")

st.markdown("---")
st.markdown("Datenquelle: [Energy-Charts.info](https://www.energy-charts.info) | Temperaturen: Open-Meteo")

# --- Annahmen ---
with st.expander("Annahmen und Dokumentation der Berechnungen", expanded=False):
    st.markdown("""
### Annahmen des Modells (angepasst)

**Mobilität**
- Elektro-PKW: 12 500 km/Jahr × 18 kWh/100 km ≈ 2 250 kWh/Fahrzeug/Jahr → 40 Mio. PKW ≈ 90 TWh
- Elektro-LKW: 80 000 km/Jahr × 50 kWh/100 km ≈ 40 000 kWh/Fahrzeug/Jahr → 4 Mio. LKW ≈ 160 TWh
- Saisonale Verteilung (höher im Sommer)
- **Wochenend-/Feiertagsanteil**: Der eingestellte Anteil (0–50 %) erhöht die tägliche Mobilitätslast an Wochenenden und Feiertagen um den Faktor 1,3; die Jahresmenge bleibt erhalten (Normalisierung).

**Warmwasser**
- 2 000 kWh pro Einwohner und Jahr → 40 Mio. EW ≈ 80 TWh

**Heizung**
- Direktelektrisch: 10 000 kWh pro Einwohner und Jahr → 40 Mio. EW ≈ 400 TWh
- Mit Wärmepumpen-Mix (Luft-/Wasser-WP) sinkt der Bedarf temperaturabhängig auf ca. 160–200 TWh
- COP Luft-Wasser: ca. 2,2 + 0,08 × T (begrenzt 1,8–4,5)
- COP Wasser-Wasser/Sole: ca. 4,0 + 0,03 × T (begrenzt 3,5–5,5)

**Erzeugung**
- Tagesaggregation aus 15-min-Werten (MW × 0,25 h → GWh)
- Monatliche Skalierung auf Ziel-Leistung (abschaltbar)
- Sonstige EE unskaliert; Pumpspeicher optional ausschließbar

**CO₂**
- Residual-Intensität (Default 1 000 g/kWh) tagesweise auf die Verbraucher umgelegt
    """)

# --- Spenden ---
with st.expander("Unterstütze meine Arbeit", expanded=False):
    st.markdown("""
Wenn dir das Tool weiterhilft, freue ich mich über eine Unterstützung:

**Wero**  
[https://share.wero-wallet.eu/p/1/c/xJoLOYMLRK](https://share.wero-wallet.eu/p/1/c/xJoLOYMLRK)

**PayPal**  
[https://www.paypal.com/donate?hosted_button_id=33BEDTQ877SA4](https://www.paypal.com/donate?hosted_button_id=33BEDTQ877SA4)

Kontakt: [solardoktor.de](https://www.solardoktor.de)
    """)
    col_w, col_p = st.columns(2)
    with col_w:
        if Path("QR_wero.png").exists():
            st.image("QR_wero.png", width=180, caption="Wero")
    with col_p:
        if Path("QR_paypal.png").exists():
            st.image("QR_paypal.png", width=180, caption="PayPal")
    st.caption("Dezent, ohne Tracking – einfach als Hinweis.")
