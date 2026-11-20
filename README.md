# Nimbus Cement: climate risk exposure (v1 prototype)

A prototype for the Oren product manager assessment. It shows how Nimbus Cement's nine plants are exposed to climate hazards and to India's carbon rules, today and around 2040.

**Live prototype:** https://nimbus-climate-risk-mu.vercel.app

## Files
- `index.html`: the website. One self-contained file, hosted on Vercel.
- `pipeline.py`: the data collection script. For each plant's coordinates it fetches climate model data (Open-Meteo) and historical cyclone tracks (NOAA IBTrACS), calculates the risk measures, and writes the data the website uses.

## Data sources
- Plant data: supplied in the brief (fictional company)
- Water stress and flood risk: WRI Aqueduct 4.0
- Heat and rainfall: Open-Meteo Climate API (CMIP6 models)
- Cyclones: NOAA IBTrACS
- Map outline: Natural Earth

Every number on the site is labelled as real data, calculated, or an assumption.
