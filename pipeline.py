"""
Nimbus Cement climate-risk v1 — data pipeline.

Reads the plant CSV, fetches real climate data for each plant's coordinates,
computes indicators, and writes data/plants.json for the web app.

Every number is wrapped as:
  {"value", "unit", "type": real|derived|assumption|missing, "source", "method"}
Nothing is invented: if a source fails or a manual file is absent, the value is
written as type "missing" and the app must show "No data".

Run:  pip install pandas numpy requests
      python pipeline.py                  # all steps
      python pipeline.py --skip-cyclones  # if the IBTrACS download is slow
"""

import argparse, json, math, time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).parent
PLANTS_CSV = ROOT / "nimbus_cement_plants.csv"
AQUEDUCT_CSV = ROOT / "manual" / "aqueduct_export.csv"   # manual step, see README
FLOOD_CSV = ROOT / "manual" / "flood_manual.csv"          # manual step, see README
CACHE = ROOT / "cache"
OUT = ROOT / "data"
TODAY = date.today().isoformat()

# ---------- Open-Meteo Climate API (CMIP6 HighResMIP, bias-corrected, ~RCP8.5) ----------
OM_URL = "https://climate-api.open-meteo.com/v1/climate"
# 3 of the 7 available models, to fit Open-Meteo's free limits (each request "costs"
# days/14 x variables/10 x models calls; limits are about 600/min, 5,000/hour, 10,000/day).
OM_MODELS = ["EC_Earth3P_HR", "MPI_ESM1_2_XR", "MRI_AGCM3_2_S"]
HOURLY_BUDGET = 4500      # stay under the 5,000/hour limit
OM_VARS = ["temperature_2m_max", "precipitation_sum", "relative_humidity_2m_min"]
WINDOWS = {  # 20-year windows. API ends 2050-01-01, so the 2040 window is 2030-2049.
    "baseline": ("1995-01-01", "2014-12-31"),   # IPCC AR6 reference period
    "y2040":    ("2030-01-01", "2049-12-31"),   # centred on ~2040
}
OM_SOURCE = "Open-Meteo Climate API (CMIP6 HighResMIP: EC-Earth3P-HR, MPI-ESM1-2-XR, MRI-AGCM3-2-S; bias-corrected; high-emissions pathway close to RCP8.5)"

# Methodology thresholds (choices, documented on the Methods page)
HEAT_T1, HEAT_T2 = 40.0, 45.0     # deg C, daily max
VERY_HEAVY_RAIN = 115.6           # mm/day, IMD "very heavy rainfall" lower bound
DRY_DAY = 1.0                     # mm/day
HEAT_INDEX_DANGER = 39.4          # deg C (103 F), US NWS "Danger" category lower bound

# ---------- IBTrACS (NOAA) North Indian basin ----------
IBTRACS_URL = ("https://www.ncei.noaa.gov/data/international-best-track-archive-for-"
               "climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NI.list.v04r01.csv")
CYCLONE_RADIUS_KM, CYCLONE_MIN_KT, CYCLONE_FROM = 100, 34, 1980


def wrap(value, unit, type_, source, method=""):
    if isinstance(value, np.integer):
        value = int(value)
    if value is not None and isinstance(value, (float, np.floating)):
        value = None if math.isnan(value) else round(float(value), 3)
    return {"value": value, "unit": unit, "type": type_, "source": source, "method": method}


def missing(unit, reason):
    return wrap(None, unit, "missing", "", reason)


# =====================================================================
# 1. Transition: emissions and intensity (real CSV data + derived maths)
# =====================================================================
def transition(df):
    src = "nimbus_cement_plants.csv (FY2025-26)"
    # CO2 per tonne of clinker at each kiln (includes P01 captive power, as the CSV does)
    kiln = df[df.clinker_produced_fy2025_26_mt > 0].set_index("plant_id")
    co2_per_t_clk = (kiln.scope1_co2_fy2025_26_kt / 1000) / kiln.clinker_produced_fy2025_26_mt

    # Clinker flows from the notes column (P01 -> P07, P06 -> P08 and P09)
    flows = [("P01", "P07", 1.12), ("P06", "P08", 0.68), ("P06", "P09", 0.60)]
    received = {d: (s, t) for s, d, t in flows}
    shipped_co2 = {p: 0.0 for p in df.plant_id}
    for s, d, t in flows:
        shipped_co2[s] += co2_per_t_clk[s] * t * 1000  # kt

    plants = {}
    for _, r in df.iterrows():
        p = r.plant_id
        own = r.scope1_co2_fy2025_26_kt
        incoming = received[p][1] * co2_per_t_clk[received[p][0]] * 1000 if p in received else 0.0
        footprint = own - shipped_co2[p] + incoming   # kt CO2 embodied in this plant's cement
        plants[p] = {
            "scope1_kt": wrap(own, "kt CO2", "real", src),
            "co2_per_t_clinker": (wrap(co2_per_t_clk[p], "tCO2/t clinker", "derived", src,
                                       "scope1 / clinker produced")
                                  if p in co2_per_t_clk else None),
            "clinker_supplier": received[p][0] if p in received else None,
            "embodied_co2_kt": wrap(footprint, "kt CO2", "derived", src,
                                    "own scope1 - CO2 in clinker shipped out + CO2 in clinker received; "
                                    "kiln CO2 allocated by clinker tonnes (assumption)"),
            "co2_per_t_cement": wrap(footprint / r.cement_output_fy2025_26_mt / 1000, "tCO2/t cement",
                                     "derived", src, "embodied CO2 / cement output"),
        }

    total = df.scope1_co2_fy2025_26_kt.sum()
    group = {
        "scope1_kt": wrap(total, "kt CO2", "real", src),
        "co2_per_t_cement": wrap(total / df.cement_output_fy2025_26_mt.sum() / 1000,
                                 "tCO2/t cement", "derived", src),
        "cost_per_100_inr_t_cr": wrap(total * 1000 * 100 / 1e7, "INR crore/yr", "derived", src,
                                      "scope1 x Rs100/t; a sensitivity, not a forecast"),
    }
    return plants, group


# =====================================================================
# 2. Physical: Open-Meteo climate indicators per plant
# =====================================================================
class DailyLimitReached(Exception):
    pass


_spent = []            # (timestamp, weight) of requests sent in this run
_daily_stop = False


def request_weight(start, end):
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    return days / 14 * len(OM_VARS) / 10 * len(OM_MODELS)


def pace(weight):
    """Wait so that the last hour's spending stays under HOURLY_BUDGET."""
    while True:
        now = time.time()
        recent = [(t, w) for t, w in _spent if now - t < 3600]
        if sum(w for _, w in recent) + weight <= HOURLY_BUDGET:
            return
        wait = int(3600 - (now - recent[0][0])) + 5
        print(f"    pacing to stay under the hourly limit: waiting {wait // 60} min")
        time.sleep(min(wait, 600))


def fetch_openmeteo(pid, lat, lon, window):
    global _daily_stop
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"om_v2_{pid}_{window}.json"   # v2 = includes humidity
    if f.exists():
        print(f"    {window}: using saved copy")
        return json.loads(f.read_text())
    if _daily_stop:
        raise DailyLimitReached("daily limit reached earlier in this run")
    start, end = WINDOWS[window]
    weight = request_weight(start, end)
    params = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
              "models": ",".join(OM_MODELS), "daily": ",".join(OM_VARS)}
    for attempt in range(8):
        pace(weight)
        r = requests.get(OM_URL, params=params, timeout=180)
        if r.status_code == 429:
            reason = r.text.lower()
            if "daily" in reason:
                _daily_stop = True
                raise DailyLimitReached("Open-Meteo daily limit reached")
            wait = 65 if "minut" in reason else 600
            print(f"    rate-limited ({'minute' if wait == 65 else 'hour'} limit), waiting {wait // 60 or 1} min")
            time.sleep(wait)
            continue
        r.raise_for_status()
        _spent.append((time.time(), weight))
        f.write_text(r.text)                   # cache raw response = audit trail
        print(f"    {window}: fetched")
        time.sleep(weight / 600 * 60 + 5)      # respect the per-minute limit
        return r.json()
    raise RuntimeError("Open-Meteo still rate-limited after retries; rerun later (saved copies keep progress)")


def series(daily, var, model):
    for key in (f"{var}_{model}", var if len(OM_MODELS) == 1 else None):
        if key and key in daily:
            return pd.Series(daily[key], index=pd.to_datetime(daily["time"]), dtype="float")
    return None


def heat_index_c(t_c, rh):
    """US NWS heat index (Rothfusz regression with Steadman for mild heat). Inputs deg C and %."""
    t = t_c * 9 / 5 + 32
    simple = 0.5 * (t + 61 + (t - 68) * 1.2 + rh * 0.094)
    full = (-42.379 + 2.04901523 * t + 10.14333127 * rh - 0.22475541 * t * rh
            - 6.83783e-3 * t * t - 5.481717e-2 * rh * rh + 1.22874e-3 * t * t * rh
            + 8.5282e-4 * t * rh * rh - 1.99e-6 * t * t * rh * rh)
    adj = np.where((rh < 13) & (t >= 80) & (t <= 112),
                   -((13 - rh) / 4) * np.sqrt(np.clip((17 - np.abs(t - 95)) / 17, 0, None)), 0)
    adj = adj + np.where((rh > 85) & (t >= 80) & (t <= 87), ((rh - 85) / 10) * ((87 - t) / 5), 0)
    hi = np.where((simple + t) / 2 >= 80, full + adj, simple)
    return (hi - 32) * 5 / 9


def longest_dry_spell(p):
    dry = (p < DRY_DAY).astype(int).values
    best = run = 0
    for d in dry:
        run = run + 1 if d else 0
        best = max(best, run)
    return best


def indicators_for(daily, model):
    t, p = series(daily, "temperature_2m_max", model), series(daily, "precipitation_sum", model)
    out = {}
    if t is not None and t.notna().mean() > 0.9:
        yr = t.groupby(t.index.year)
        out["days_over_40"] = (yr.apply(lambda s: (s > HEAT_T1).sum())).mean()
        out["days_over_45"] = (yr.apply(lambda s: (s > HEAT_T2).sum())).mean()
        out["mean_tmax"] = t.mean()
    rh = series(daily, "relative_humidity_2m_min", model)
    if t is not None and rh is not None and t.notna().mean() > 0.9 and rh.notna().mean() > 0.9:
        # daily max temperature paired with daily minimum humidity (they usually coincide mid-afternoon)
        hi = pd.Series(heat_index_c(t.values, rh.values), index=t.index)
        out["humid_heat_danger_days"] = hi.groupby(hi.index.year).apply(
            lambda s: (s >= HEAT_INDEX_DANGER).sum()).mean()
    if p is not None and p.notna().mean() > 0.9:
        yr = p.groupby(p.index.year)
        out["very_heavy_rain_days"] = (yr.apply(lambda s: (s >= VERY_HEAVY_RAIN).sum())).mean()
        out["max_1day_rain"] = yr.max().mean()
        out["longest_dry_spell"] = yr.apply(longest_dry_spell).mean()
    return out


UNITS = {"days_over_40": "days/yr", "days_over_45": "days/yr", "mean_tmax": "deg C",
         "humid_heat_danger_days": "days/yr",
         "very_heavy_rain_days": "days/yr", "max_1day_rain": "mm", "longest_dry_spell": "days"}


def physical(pid, lat, lon):
    try:
        data = {w: fetch_openmeteo(pid, lat, lon, w)["daily"] for w in WINDOWS}
    except Exception as e:
        print(f"    Open-Meteo failed for {pid}: {e}")
        return {k: missing(u, f"Open-Meteo fetch failed: {e}") for k, u in UNITS.items()}

    per_model = {m: {w: indicators_for(data[w], m) for w in WINDOWS} for m in OM_MODELS}
    result = {}
    for k, unit in UNITS.items():
        base = [v["baseline"][k] for v in per_model.values() if k in v["baseline"]]
        fut = [v["y2040"][k] for v in per_model.values() if k in v["y2040"]]
        delta = [v["y2040"][k] - v["baseline"][k] for v in per_model.values()
                 if k in v["baseline"] and k in v["y2040"]]
        if not delta:
            result[k] = missing(unit, "no model returned this variable")
            continue
        method = (f"mean annual value per model; median and min-max across {len(delta)} models; "
                  f"baseline 1995-2014, 2040 = 2030-2049")
        result[k] = {
            "baseline": wrap(np.median(base), unit, "real", OM_SOURCE, method),
            "y2040": wrap(np.median(fut), unit, "real", OM_SOURCE, method),
            "change": wrap(np.median(delta), unit, "derived", OM_SOURCE, "median of per-model change"),
            "change_range": [round(float(min(delta)), 2), round(float(max(delta)), 2)],
            "n_models": len(delta),
        }
    return result


# =====================================================================
# 3. Water stress: WRI Aqueduct 4.0 (manual export, see README)
# =====================================================================
def water(pid):
    src = "WRI Aqueduct 4.0 (baseline; 2030 and 2050 projections; no 2040 milestone)"
    if not AQUEDUCT_CSV.exists():
        return {"status": missing("", "manual/aqueduct_export.csv not provided yet")}
    aq = pd.read_csv(AQUEDUCT_CSV)
    row = aq[aq["plant_id"] == pid]
    if row.empty:
        return {"status": missing("", f"{pid} not found in Aqueduct export")}
    row = row.iloc[0]
    cols = [c for c in aq.columns if c != "plant_id"]
    return {c: wrap(row[c] if not pd.isna(row[c]) else None, "", "real", src,
                    "basin-level value at plant location; 2040 shown as 2030-2050 bounds, not interpolated")
            for c in cols}


# =====================================================================
# 3b. Flood depth: manually read from WRI Aqueduct Floods map (see README)
# =====================================================================
FLOOD_COLS = ["riverine_depth_m_rp100_today", "riverine_depth_m_rp100_2050",
              "coastal_depth_m_rp100_today", "coastal_depth_m_rp100_2050"]


def flood(pid):
    src = "Manually read from WRI Aqueduct Floods map (1-in-100-year flood)"
    if not FLOOD_CSV.exists():
        return {c: missing("m", "manual/flood_manual.csv not provided yet") for c in FLOOD_COLS}
    fl = pd.read_csv(FLOOD_CSV)
    row = fl[fl["plant_id"] == pid]
    if row.empty:
        return {c: missing("m", f"{pid} not in flood_manual.csv") for c in FLOOD_COLS}
    row = row.iloc[0]
    meta = f"scenario: {row.get('scenario_2050', '')}; read on {row.get('read_on', '')}; {row.get('notes', '')}"
    # Blank = not checked (missing). 0 = checked, no flooding shown.
    return {c: (missing("m", "not checked") if pd.isna(row.get(c)) else
                wrap(float(row[c]), "m", "real", src, meta)) for c in FLOOD_COLS}


# =====================================================================
# 4. Cyclones: IBTrACS historical tracks near each plant
# =====================================================================
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def load_ibtracs():
    CACHE.mkdir(exist_ok=True)
    f = CACHE / "ibtracs_NI.csv"
    if not f.exists():
        print("  downloading IBTrACS North Indian basin...")
        r = requests.get(IBTRACS_URL, timeout=300)
        r.raise_for_status()
        f.write_bytes(r.content)
    df = pd.read_csv(f, skiprows=[1], low_memory=False,       # row 2 holds units
                     usecols=["SID", "SEASON", "LAT", "LON", "WMO_WIND", "USA_WIND"])
    for c in ["SEASON", "LAT", "LON", "WMO_WIND", "USA_WIND"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["wind"] = df.WMO_WIND.fillna(df.USA_WIND)
    return df[(df.SEASON >= CYCLONE_FROM) & (df.wind >= CYCLONE_MIN_KT)]


def cyclones(tracks, lat, lon):
    src = "NOAA IBTrACS v04r01, North Indian basin"
    if tracks is None:
        return missing("storms", "IBTrACS skipped or unavailable")
    near = tracks[haversine_km(lat, lon, tracks.LAT.values, tracks.LON.values) <= CYCLONE_RADIUS_KM]
    n = near.SID.nunique()
    yrs = date.today().year - CYCLONE_FROM
    return wrap(n, f"storms since {CYCLONE_FROM}", "real", src,
                f"distinct storms >= {CYCLONE_MIN_KT} kt passing within {CYCLONE_RADIUS_KM} km "
                f"(about {n / yrs:.2f}/yr). LIMITATION: historical only. No reliable free projection "
                f"exists of how cyclones near each plant will change by 2040.")


# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cyclones", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(PLANTS_CSV)
    trans, group = transition(df)

    tracks = None
    if not args.skip_cyclones:
        try:
            tracks = load_ibtracs()
        except Exception as e:
            print(f"  IBTrACS failed ({e}); cyclone values will be 'missing'")

    plants = []
    for _, r in df.iterrows():
        print(f"{r.plant_id} {r.plant_name}")
        plants.append({
            "profile": {k: (None if pd.isna(v) else v.item() if hasattr(v, "item") else v)
                        for k, v in r.items()},
            "transition": trans[r.plant_id],
            "physical": physical(r.plant_id, r.latitude, r.longitude),
            "water": water(r.plant_id),
            "flood": flood(r.plant_id),
            "cyclones": cyclones(tracks, r.latitude, r.longitude),
        })

    OUT.mkdir(exist_ok=True)
    (OUT / "plants.json").write_text(json.dumps(
        {"generated_on": TODAY, "group": group, "plants": plants}, indent=2, default=str))
    missing_plants = [p["profile"]["plant_id"] for p in plants
                      if p["physical"]["days_over_40"].get("type") == "missing"]
    print(f"\nWrote {OUT / 'plants.json'}")
    if missing_plants:
        print(f"⚠️  Climate data still missing for: {', '.join(missing_plants)}")
        print("   Run this step again later. Saved copies are reused, so it continues where it stopped.")
    else:
        print("✅ Climate data complete for all 9 plants")


if __name__ == "__main__":
    main()
