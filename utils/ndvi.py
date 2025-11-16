import ee
import pandas as pd
from datetime import timedelta

# Initialize EE
try:
    ee.Initialize(project='wildfire-prediction-477513')
except:
    ee.Authenticate()
    ee.Initialize(project='wildfire-prediction-477513')


def get_ndvi_data(today):
    """Fetch NDVI for today + past 7-day & 30-day rolling averages safely."""

    start_30 = today - timedelta(days=30)

    # MODIS MOD09GA contains Red & NIR
    collection = (
        ee.ImageCollection("MODIS/006/MOD09GA")
        .filterDate(start_30.strftime("%Y-%m-%d"),
                    (today + timedelta(days=1)).strftime("%Y-%m-%d"))
        .select(["sur_refl_b01", "sur_refl_b02"])  # Red, NIR
    )

    def add_ndvi(image):
        ndvi = image.normalizedDifference(["sur_refl_b02", "sur_refl_b01"]) \
                     .rename("NDVI")
        return image.addBands(ndvi)

    collection = collection.map(add_ndvi)

    # California bounds
    cali_bbox = ee.Geometry.Rectangle([-124.48, 32.53, -114.13, 42.01])

    ndvi_list = []

    # Loop last 30 days
    for i in range(31):
        day = start_30 + timedelta(days=i)

        day_start = day.strftime("%Y-%m-%d")
        day_end   = (day + timedelta(days=1)).strftime("%Y-%m-%d")

        day_img = collection.filterDate(day_start, day_end)

        # If no image exists for that day → append None
        if day_img.size().getInfo() == 0:
            ndvi_list.append(None)
            continue

        # MODIS gives many tiles → mean() to merge
        merged = day_img.mean()

        # Calculate region NDVI safely
        ndvi_dict = merged.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=cali_bbox,
            scale=500,
            maxPixels=1e13
        ).getInfo()

        # NDVI may still be missing due to clouds or blank tiles
        ndvi_val = ndvi_dict.get("NDVI", None)
        ndvi_list.append(ndvi_val)

    # Convert to series
    ndvi_series = pd.Series(ndvi_list)

    # Compute today's NDVI (last day)
    ndvi_today = ndvi_series.iloc[-1]

    # Compute rolling means only for days where NDVI exists
    ndvi_7day = ndvi_series.iloc[-7:].dropna().mean() if ndvi_series.iloc[-7:].notna().any() else None
    ndvi_30day = ndvi_series.dropna().mean() if ndvi_series.notna().any() else None

    return ndvi_today, ndvi_7day, ndvi_30day
