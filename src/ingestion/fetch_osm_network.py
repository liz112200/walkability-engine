from __future__ import annotations

import argparse
import sys
import time
import osmnx as ox

from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_OSM_DIR = PROJECT_ROOT / "data" / "raw" / "osm"


ox.settings.log_console = False        
ox.settings.use_cache = True          
ox.settings.cache_folder = str(PROJECT_ROOT / ".osmnx_cache")
ox.settings.timeout = 300              
ox.settings.max_query_area_size = 2_500_000_000  

DEFAULT_CITY = "Chicago, Illinois, USA"
CHICAGO_UTM = "EPSG:26916"

def _config_logger(dir_path: Path) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    logger.add(
        dir_path / "fetch_osm_network.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    )

def _parse_args() ->argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and save OSMnx pedestrian network for a given city.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--city",
        default=DEFAULT_CITY,
        help="Nominatim place string, e.g. 'Chicago, Illinois, USA'",
    )
    parser.add_argument(
        "--crs",
        default=CHICAGO_UTM,
        help="Target metric CRS for projection (EPSG code or WKT string)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(RAW_OSM_DIR),
        help="Output directory for raw files"
    )

    return parser.parse_args()

def _fetch_with_retry(
        city: str,
        network_type: str = "walk",
        max_retries: int = 3,
        backoff_seconds: float = 15.0
) -> ox.graph:
    attempt = 0
    wait = backoff_seconds

    while attempt < max_retries:
        attempt += 1
        try:
            logger.info(
                f"Fetching pedestrian network - attempt {attempt}/{max_retries} "
                f"for '{city}'"
            )
            G = ox.graph_from_place(
                city, 
                network_type=network_type,
                retain_all=False,
                truncate_by_edge=True,
                simplify=True
            )
            return G
        
        except Exception as e:
            logger.warning(f"Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                logger.info(f"Retrying in {wait:.0f}s ...")
                time.sleep(wait)
                wait *= 2
            else:
                logger.error("All retry attempts exhausted.")
                raise


def _validate_graph(G: ox.graph) -> dict[str, int | float]:
    
    stats: dict[str, int | float] = {
        "n_nodes": len(G.nodes),
        "n_edges": len(G.edges),
        "n_components": ox.stats.basic_stats(G).get("n_nodes", 0)
    }

    MIN_NODES = 10_000
    if stats["n_nodes"] < MIN_NODES:
        raise ValueError(
            f"Graph has only {stats['n_nodes']} nodes — expected ≥ {MIN_NODES}. "
            "Check that the place name resolves to the full city, not a suburb."
        )
    
    ratio = stats["n_edges"] / max(stats["n_nodes"], 1)
    if not (1.2 <= ratio <= 4.0):
        logger.warning(
            f"Edge/node ratio = {ratio:.2f} (expected 1.2–4.0). "
            "Double-check the network looks correct in the sanity-check notebook."
        )
    
    return stats


def fetch_and_save(
        city: str = DEFAULT_CITY,
        crs: str = CHICAGO_UTM,
        out_dir: Path = RAW_OSM_DIR
) -> tuple[Path, Path, Path]:
    
    """
    Full pipeline: download → validate → project → save.
    """ 
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Download ────────────────────────────────────────────────────────
    t_start = time.perf_counter()
    G_geo = _fetch_with_retry(city)
    elapsed = time.perf_counter() - t_start
    logger.info(f"Download complete in {elapsed:.1f}s")

    # ── 2. Validate ────────────────────────────────────────────────────────
    stats = _validate_graph(G_geo)
    logger.info(
        f"Graph validated - "
        f"nodes: {stats['n_nodes']:,}  |  "
        f"edges: {stats['n_edges']:,}  |  "
        f"edge/node ratio: {stats['n_edges']/stats['n_nodes']:.2f}"
    )

    # ── 3. Project to metric CRS ───────────────────────────────────────────
    logger.info(f"Projecting graph to {crs}...")
    G_proj = ox.projection.project_graph(G_geo, to_crs=crs)

    # ── 4. Save GraphML (full graph, preserves all OSM attributes) ─────────
    city_slug = city.split(",")[0].lower().replace(" ", "_")
    graphml_path = out_dir / f"{city_slug}_walk_graph.graphml"
    ox.save_graphml(G_proj, filepath=str(graphml_path))
    logger.info(f"Saved GraphML  →  {graphml_path.relative_to(PROJECT_ROOT)}")

    #── 5. Extract GeoDataFrames ───────────────────────────────────────────
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_proj)
    _list_cols = [c for c in edges_gdf.columns if edges_gdf[c].dtype == object
                    and edges_gdf[c].apply(lambda x: isinstance(x, list)).any()]
    if _list_cols:
        logger.debug(f"Converting list-type columns to strings: {_list_cols}")
        for col in _list_cols:
            edges_gdf[col] = edges_gdf[col].apply(
                lambda x: "|".join(map(str, x)) if isinstance(x, list) else x
            )

    # ── 6. Save GeoPackages ────────────────────────────────────────────────
    edges_path = out_dir / f"{city_slug}_walk_edges.gpkg"
    nodes_path = out_dir / f"{city_slug}_walk_nodes.gpkg"
    
    edges_gdf.reset_index().to_file(str(edges_path), driver="GPKG", layer="edges")
    logger.info(f"Saved edges     →  {edges_path.relative_to(PROJECT_ROOT)}")
 
    nodes_gdf.reset_index().to_file(str(nodes_path), driver="GPKG", layer="nodes")
    logger.info(f"Saved nodes     →  {nodes_path.relative_to(PROJECT_ROOT)}")

    # ── 7. Summary report ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("INGESTION SUMMARY")
    logger.info(f"  City            : {city}")
    logger.info(f"  CRS             : {crs}")
    logger.info(f"  Nodes           : {stats['n_nodes']:,}")
    logger.info(f"  Edges           : {stats['n_edges']:,}")
    logger.info(f"  GraphML size    : {graphml_path.stat().st_size / 1e6:.1f} MB")
    logger.info(f"  Edges GPKG size : {edges_path.stat().st_size / 1e6:.1f} MB")
    logger.info("=" * 60)
 
    return graphml_path, edges_path, nodes_path



if __name__ == "__main__":
    args = _parse_args()
    _config_logger(PROJECT_ROOT / "logs")

    try:
        graphml_path, edges_path, nodes_path = fetch_and_save(
            city=args.city,
            crs=args.crs,
            out_dir=Path(args.out_dir),
        )
        logger.success("Week 1 ingestion complete. Proceed to sanity-check notebook.")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Error during ingestion: {e}")
        sys.exit(1)

