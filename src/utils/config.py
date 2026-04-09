from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = PROJECT_ROOT / "configs" / "city_config.yml"

@dataclass(frozen=True)
class CityConfig:
    name: str
    slug: str
    crs: str
    crs_name: str

@dataclass(frozen=True)
class H3Config:
    resolution: int

@dataclass(frozen=True)
class NetworkConfig:
    network_type: str
    retain_all: bool
    truncate_by_edge: bool
    simplify: bool
    timeout: int

@dataclass(frozen=True)
class PathsConfig:
    raw_osm: Path
    raw_gtfs: Path
    raw_census: Path
    raw_elevation: Path
    raw_poi: Path
    raw_crash: Path
    processed: Path
    labels: Path
    splits: Path
    figures: Path
    models: Path
    reports: Path
    maps: Path

@dataclass(frozen=True)
class Config:
    city: CityConfig
    h3: H3Config
    network: NetworkConfig
    paths: PathsConfig
    project_root: Path
 

def _load() -> Config:
    with open(_CONFIG_PATH) as f:
        raw = yaml.safe_load(f)

    c = raw["city"]
    h = raw["h3"]
    n = raw["network"]
    p = raw["data_paths"]

    return Config(
        project_root=PROJECT_ROOT,
        city=CityConfig(
            name=c["name"],
            slug=c["slug"],
            crs=c["crs"],
            crs_name=c["crs_name"]
        ),
        h3=H3Config(resolution=h["resolution"]),
        network=NetworkConfig(
            network_type=n["network_type"],
            retain_all=n["retain_all"],
            truncate_by_edge=n["truncate_by_edge"],
            simplify=n["simplify"],
            timeout=n["timeout"],
        ),
        paths=PathsConfig(
            raw_osm=PROJECT_ROOT / p["raw_osm"],
            raw_gtfs=PROJECT_ROOT / p["raw_gtfs"],
            raw_census=PROJECT_ROOT / p["raw_census"],
            raw_elevation=PROJECT_ROOT / p["raw_elevation"],
            raw_poi=PROJECT_ROOT / p["raw_poi"],
            raw_crash=PROJECT_ROOT / p["raw_crash"],
            processed=PROJECT_ROOT / p["processed"],
            labels=PROJECT_ROOT / p["labels"],
            splits=PROJECT_ROOT / p["splits"],
            figures=PROJECT_ROOT / p["figures"],
            models=PROJECT_ROOT / p["models"],
            reports=PROJECT_ROOT / p["reports"],
            maps=PROJECT_ROOT / p["maps"],
        ),
    )

cfg: Config = _load()