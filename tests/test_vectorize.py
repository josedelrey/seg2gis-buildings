import numpy as np
import rasterio
from rasterio.transform import from_origin

from vectorize import (
    mask_to_contours,
    mask_to_geodataframe,
    save_vector_polygons,
    simplify_contours,
)


def test_polygon_extraction_finds_and_simplifies_building_region():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[4:14, 5:16] = 1

    contours = mask_to_contours(mask, min_area=10)
    polygons = simplify_contours(contours, epsilon_ratio=0.01)

    assert len(contours) == 1
    assert len(polygons) == 1
    assert len(polygons[0]) >= 4


def test_mask_to_geodataframe_uses_raster_transform_and_crs(tmp_path):
    raster_path = tmp_path / "source.tif"
    image = np.zeros((1, 10, 10), dtype=np.uint8)
    transform = from_origin(100.0, 200.0, 2.0, 2.0)

    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype=image.dtype,
        crs="EPSG:3857",
        transform=transform,
    ) as dst:
        dst.write(image)

    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 3:7] = 1

    gdf = mask_to_geodataframe(mask, str(raster_path), min_area=1)

    assert len(gdf) == 1
    assert gdf.crs.to_string() == "EPSG:3857"
    assert gdf.geometry.iloc[0].bounds == (106.0, 190.0, 114.0, 196.0)


def test_save_vector_polygons_writes_geojson_and_gpkg(tmp_path):
    raster_path = tmp_path / "source.tif"
    image = np.zeros((1, 8, 8), dtype=np.uint8)

    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=1,
        dtype=image.dtype,
        crs="EPSG:4326",
        transform=from_origin(0.0, 8.0, 1.0, 1.0),
    ) as dst:
        dst.write(image)

    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:4, 1:4] = 1

    geojson_path = tmp_path / "buildings.geojson"
    gpkg_path = tmp_path / "buildings.gpkg"
    gdf = save_vector_polygons(mask, str(raster_path), str(geojson_path), min_area=1)
    save_vector_polygons(mask, str(raster_path), str(gpkg_path), min_area=1)

    assert geojson_path.exists()
    assert gpkg_path.exists()
    assert len(gdf) == 1
