
from __future__ import annotations

import h3
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point
from loguru import logger

# ── H3 version shim ───────────────────────────────────────────────────────────
# H3 v4 renamed several core functions. We detect the version once at import
# time and bind the correct callables to consistent internal names.

_H3_MAJOR = int(h3.__version__.split(".")[0])
logger.debug(f"H3 version: {h3.__version__} (major={_H3_MAJOR})")

if _H3_MAJOR >= 4:
    def _polyfill(geojson_polygon: dict, resolution: int) -> set[str]:
        """Fill a GeoJSON polygon with H3 cells (v4 API)."""
        return h3.geo_to_cells(geojson_polygon, resolution)

    def _cell_to_polygon(cell_id: str) -> Polygon:
        """Convert an H3 cell ID to a Shapely Polygon (v4 API).
        cell_to_boundary returns [(lat, lng), ...] — note lat/lng order."""
        boundary = h3.cell_to_boundary(cell_id)   # [(lat, lng), ...]
        return Polygon([(lng, lat) for lat, lng in boundary])

    def _latlng_to_cell(lat: float, lng: float, resolution: int) -> str:
        """Convert lat/lng to the containing H3 cell ID (v4 API)."""
        return h3.latlng_to_cell(lat, lng, resolution)

else:
    def _polyfill(geojson_polygon: dict, resolution: int) -> set[str]:
        """Fill a GeoJSON polygon with H3 cells (v3 API)."""
        return h3.polyfill_geojson(geojson_polygon, resolution)

    def _cell_to_polygon(cell_id: str) -> Polygon:
        """Convert an H3 cell ID to a Shapely Polygon (v3 API).
        h3_to_geo_boundary returns [(lat, lng), ...] when geo_json=False."""
        boundary = h3.h3_to_geo_boundary(cell_id, geo_json=False)  # [(lat, lng)]
        return Polygon([(lng, lat) for lat, lng in boundary])

    def _latlng_to_cell(lat: float, lng: float, resolution: int) -> str:
        """Convert lat/lng to the containing H3 cell ID (v3 API)."""
        return h3.geo_to_h3(lat, lng, resolution)


# ── Core grid generation ───────────────────────────────────────────────────────

def city_hex_grid(
    city_gdf: gpd.GeoDataFrame,
    resolution: int = 9,
    buffer_m: float = 500.0,
) -> gpd.GeoDataFrame:
   
    # H3 operates in WGS-84 degrees — reproject bounding box
    city_wgs = city_gdf.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = city_wgs.total_bounds

    # Build a GeoJSON polygon covering the bounding box
    bbox_geojson = {
        "type": "Polygon",
        "coordinates": [[
            [minx, miny], [maxx, miny],
            [maxx, maxy], [minx, maxy],
            [minx, miny],
        ]],
    }

    logger.info(
        f"Filling bounding box with H3 resolution-{resolution} cells "
        f"({minx:.3f}, {miny:.3f}) → ({maxx:.3f}, {maxy:.3f}) …"
    )

    hex_ids: set[str] = _polyfill(bbox_geojson, resolution)
    logger.info(f"Generated {len(hex_ids):,} H3 cells before spatial filter")

    # Convert each H3 cell to a Shapely polygon
    records = []
    for hid in hex_ids:
        poly = _cell_to_polygon(hid)
        records.append({"h3_index": hid, "geometry": poly})

    hex_gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")

    # Clip to city boundary
    # city_union = city_wgs.union_all()
    # hex_gdf = hex_gdf[hex_gdf.geometry.intersects(city_union)].copy()
    hex_gdf = hex_gdf.reset_index(drop=True)

    # Reproject to match input CRS
    target_crs = city_gdf.crs
    hex_gdf = hex_gdf.to_crs(target_crs)

    # Add centroid columns (projected CRS — metres)
    hex_gdf["centroid_x"] = hex_gdf.geometry.centroid.x
    hex_gdf["centroid_y"] = hex_gdf.geometry.centroid.y

    # Add centroid columns in WGS-84 (for API calls)
    centroids_wgs = hex_gdf.geometry.centroid.to_crs("EPSG:4326")
    hex_gdf["centroid_lat"] = centroids_wgs.y
    hex_gdf["centroid_lng"] = centroids_wgs.x

    # logger.info(
    #     f"Final grid: {len(hex_gdf):,} cells after clipping to city boundary"
    # )
    logger.info(f"Grid ready: {len(hex_gdf):,} cells")

    return hex_gdf


def add_h3_index(
    gdf: gpd.GeoDataFrame,
    resolution: int = 9,
    lat_col: str = "centroid_lat",
    lng_col: str = "centroid_lng",
) -> gpd.GeoDataFrame:

    gdf = gdf.copy()
    gdf["h3_index"] = gdf.apply(
        lambda row: _latlng_to_cell(row[lat_col], row[lng_col], resolution),
        axis=1,
    )
    return gdf


def hex_centroids(hex_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
  
    pts = hex_gdf[["h3_index", "centroid_lat", "centroid_lng"]].copy()
    pts["geometry"] = pts.apply(
        lambda r: Point(r["centroid_lng"], r["centroid_lat"]), axis=1
    )
    return gpd.GeoDataFrame(pts, crs="EPSG:4326")