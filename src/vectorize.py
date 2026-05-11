import cv2
import geopandas as gpd
import numpy as np
import os
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape


def validate_area_filter_crs(crs, min_area, allow_geographic_area=False):
    if min_area is None or min_area <= 0:
        return

    if crs is None:
        raise ValueError(
            "Cannot apply vector min_area because the source raster has no CRS. "
            "Use a georeferenced raster with a projected CRS, set min_area=0, "
            "or pass allow_geographic_area=True if you understand the units."
        )

    if crs.is_geographic and not allow_geographic_area:
        raise ValueError(
            "Cannot apply vector min_area with a geographic CRS because polygon.area "
            "would be measured in square degrees, not square meters. Reproject the "
            "raster or vector output to a projected CRS, set min_area=0, or pass "
            "allow_geographic_area=True if degree-based area filtering is intentional."
        )


def mask_to_contours(mask, min_area=64):
    """
    Extract external contours from a binary mask.

    Args:
        mask: uint8 binary mask with values 0/1.
        min_area: minimum contour area in pixels.

    Returns:
        List of OpenCV contours.
    """
    mask_uint8 = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    filtered = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area >= min_area:
            filtered.append(contour)

    return filtered


def simplify_contours(contours, epsilon_ratio=0.01):
    """
    Simplify contours using Douglas-Peucker approximation.
    """
    simplified = []

    for contour in contours:
        perimeter = cv2.arcLength(contour, closed=True)
        epsilon = epsilon_ratio * perimeter

        polygon = cv2.approxPolyDP(
            contour,
            epsilon,
            closed=True,
        )

        if len(polygon) >= 3:
            simplified.append(polygon)

    return simplified


def draw_polygons_on_image(image_rgb, polygons, color=(255, 0, 0), thickness=2):
    """
    Draw polygon outlines on an RGB image.
    """
    overlay = image_rgb.copy()

    cv2.drawContours(
        overlay,
        polygons,
        contourIdx=-1,
        color=color,
        thickness=thickness,
    )

    return overlay


def mask_to_geodataframe(
    mask,
    raster_path,
    min_area=64,
    allow_geographic_area=False,
):
    """
    Convert a binary mask to CRS-aware vector geometries using a source raster.

    Args:
        mask: uint8 binary mask with values 0/1.
        raster_path: source georeferenced raster used for transform and CRS.
        min_area: minimum polygon area in raster CRS units.
        allow_geographic_area: allow min_area filtering when the raster CRS is
            geographic. This treats area as square degrees and is usually not
            appropriate for building footprints.

    Returns:
        GeoDataFrame with building polygons and source raster CRS.
    """
    mask = (mask > 0).astype(np.uint8)

    with rasterio.open(raster_path) as src:
        transform = src.transform
        crs = src.crs

    validate_area_filter_crs(
        crs=crs,
        min_area=min_area,
        allow_geographic_area=allow_geographic_area,
    )

    records = []

    for geom, value in shapes(mask, mask=mask.astype(bool), transform=transform):
        if value != 1:
            continue

        polygon = shape(geom)

        if polygon.is_empty or polygon.area < min_area:
            continue

        records.append(
            {
                "geometry": polygon,
                "area": polygon.area,
            }
        )

    if not records:
        return gpd.GeoDataFrame({"area": []}, geometry=[], crs=crs)

    return gpd.GeoDataFrame(records, geometry="geometry", crs=crs)


def save_vector_polygons(
    mask,
    raster_path,
    out_path,
    min_area=64,
    driver=None,
    allow_geographic_area=False,
):
    """
    Save mask-derived polygons to a GIS vector file.

    The output driver is inferred from the file extension unless explicitly
    provided. Common extensions: .geojson and .gpkg.
    """
    gdf = mask_to_geodataframe(
        mask=mask,
        raster_path=raster_path,
        min_area=min_area,
        allow_geographic_area=allow_geographic_area,
    )

    if driver is None:
        lower_path = out_path.lower()

        if lower_path.endswith(".geojson") or lower_path.endswith(".json"):
            driver = "GeoJSON"
        elif lower_path.endswith(".gpkg"):
            driver = "GPKG"

    if driver is None:
        raise ValueError(
            "Could not infer vector output driver. Use .geojson, .json, "
            "or .gpkg, or pass an explicit driver."
        )

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    gdf.to_file(out_path, driver=driver)

    return gdf
