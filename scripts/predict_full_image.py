import os
import argparse
import sys

import cv2
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("src"))

from config import DEFAULT_CONFIG_PATH, get_config_value, load_config
from gis_utils import (
    load_model,
    load_rgb_image,
    predict_full_image_tiled,
    threshold_probability_map,
    save_probability_map,
    save_probability_png,
    save_mask_png,
)

from postprocess import postprocess_mask
from vectorize import (
    mask_to_contours,
    simplify_contours,
    draw_polygons_on_image,
    save_vector_polygons,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)

    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)

    parser.add_argument("--architecture", type=str, default=None)
    parser.add_argument("--encoder", type=str, default=None)

    parser.add_argument("--threshold", type=float, default=None)

    parser.add_argument("--tile_size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)

    # Postprocessing
    parser.add_argument("--min_area", type=int, default=None)
    parser.add_argument("--open_kernel_size", type=int, default=None)

    # Polygon extraction
    parser.add_argument("--polygon_min_area", type=int, default=None)
    parser.add_argument("--epsilon_ratio", type=float, default=None)
    parser.add_argument("--export_vectors", action="store_true", default=None)
    parser.add_argument("--no_export_vectors", action="store_true")
    parser.add_argument("--vector_min_area", type=float, default=None)

    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--output_name", type=str, default=None)

    # Showcase crop
    parser.add_argument("--crop_x", type=int, default=None)
    parser.add_argument("--crop_y", type=int, default=None)
    parser.add_argument("--crop_size", type=int, default=None)

    return parser.parse_args()


def select_value(cli_value, config, *keys, default=None):
    if cli_value is not None:
        return cli_value
    return get_config_value(config, *keys, default=default)


def apply_config(args, config):
    args.image_path = select_value(args.image_path, config, "inference", "image_path")
    args.model_path = select_value(
        args.model_path,
        config,
        "inference",
        "model_path",
        default="models/unet_effb3_256_noaug_e10.pth",
    )
    args.architecture = select_value(
        args.architecture,
        config,
        "model",
        "architecture",
        default="unet",
    )
    args.encoder = select_value(
        args.encoder,
        config,
        "model",
        "encoder",
        default="efficientnet-b3",
    )
    args.threshold = select_value(args.threshold, config, "inference", "threshold", default=0.5)
    args.tile_size = select_value(args.tile_size, config, "inference", "tile_size", default=256)
    args.stride = select_value(args.stride, config, "inference", "stride", default=128)
    args.min_area = select_value(args.min_area, config, "inference", "min_area", default=500)
    args.open_kernel_size = select_value(
        args.open_kernel_size,
        config,
        "inference",
        "open_kernel_size",
        default=5,
    )
    args.polygon_min_area = select_value(
        args.polygon_min_area,
        config,
        "inference",
        "polygon_min_area",
        default=150,
    )
    args.vector_min_area = select_value(
        args.vector_min_area,
        config,
        "inference",
        "vector_min_area",
        default=args.polygon_min_area,
    )
    args.epsilon_ratio = select_value(
        args.epsilon_ratio,
        config,
        "inference",
        "epsilon_ratio",
        default=0.002,
    )
    args.out_dir = select_value(
        args.out_dir,
        config,
        "inference",
        "out_dir",
        default="outputs/full_predictions",
    )
    args.output_name = select_value(args.output_name, config, "inference", "output_name")
    args.crop_x = select_value(args.crop_x, config, "inference", "crop_x")
    args.crop_y = select_value(args.crop_y, config, "inference", "crop_y")
    args.crop_size = select_value(args.crop_size, config, "inference", "crop_size", default=1024)
    args.export_vectors = select_value(
        args.export_vectors,
        config,
        "inference",
        "export_vectors",
        default=True,
    )

    if args.no_export_vectors:
        args.export_vectors = False

    if args.image_path is None:
        raise ValueError(
            "No input image was provided. Set inference.image_path in the config "
            "or pass --image_path."
        )

    return args


def get_output_name(image_path, output_name):
    if output_name is not None:
        return output_name

    base = os.path.basename(image_path)
    name, _ = os.path.splitext(base)
    return name


def build_output_paths(out_dir, output_name):
    return {
        "prob_npy": os.path.join(out_dir, f"{output_name}_prob.npy"),
        "prob_png": os.path.join(out_dir, f"{output_name}_prob.png"),
        "raw_mask_png": os.path.join(out_dir, f"{output_name}_mask.png"),
        "clean_mask_png": os.path.join(out_dir, f"{output_name}_clean_mask.png"),
        "polygon_overlay_png": os.path.join(
            out_dir,
            f"{output_name}_polygons_overlay.png",
        ),
        "showcase_crop_png": os.path.join(
            out_dir,
            f"{output_name}_showcase_crop.png",
        ),
        "polygons_geojson": os.path.join(
            out_dir,
            f"{output_name}_buildings.geojson",
        ),
        "polygons_gpkg": os.path.join(
            out_dir,
            f"{output_name}_buildings.gpkg",
        ),
    }


def print_config(args):
    print("Device:", DEVICE)
    print("Image:", args.image_path)
    print("Model:", args.model_path)
    print("Architecture:", args.architecture)
    print("Encoder:", args.encoder)
    print("Threshold:", args.threshold)
    print("Tile size:", args.tile_size)
    print("Stride:", args.stride)
    print("Postprocess min area:", args.min_area)
    print("Postprocess open kernel size:", args.open_kernel_size)
    print("Polygon min area:", args.polygon_min_area)
    print("Vector min area:", args.vector_min_area)
    print("Polygon epsilon ratio:", args.epsilon_ratio)
    print("Export GIS vectors:", args.export_vectors)
    print("Crop x:", args.crop_x)
    print("Crop y:", args.crop_y)
    print("Crop size:", args.crop_size)


def run_full_image_inference(args):
    image_rgb = load_rgb_image(args.image_path)

    print("Input image shape:", image_rgb.shape)

    model = load_model(
        model_path=args.model_path,
        architecture=args.architecture,
        encoder=args.encoder,
        device=DEVICE,
    )

    prob_map = predict_full_image_tiled(
        model=model,
        image_rgb=image_rgb,
        tile_size=args.tile_size,
        stride=args.stride,
        device=DEVICE,
    )

    return image_rgb, prob_map


def generate_masks(args, prob_map):
    raw_mask = threshold_probability_map(
        prob_map=prob_map,
        threshold=args.threshold,
    )

    clean_mask = postprocess_mask(
        raw_mask,
        min_area=args.min_area,
        open_kernel_size=args.open_kernel_size,
    )

    return raw_mask, clean_mask


def generate_polygon_overlay(args, image_rgb, clean_mask):
    contours = mask_to_contours(
        clean_mask,
        min_area=args.polygon_min_area,
    )

    polygons = simplify_contours(
        contours,
        epsilon_ratio=args.epsilon_ratio,
    )

    overlay_rgb = draw_polygons_on_image(
        image_rgb=image_rgb,
        polygons=polygons,
        color=(255, 0, 0),
        thickness=2,
    )

    print("Extracted contours:", len(contours))
    print("Simplified polygons:", len(polygons))

    return overlay_rgb


def save_outputs(prob_map, raw_mask, clean_mask, polygon_overlay_rgb, output_paths):
    save_probability_map(prob_map, output_paths["prob_npy"])
    save_probability_png(prob_map, output_paths["prob_png"])
    save_mask_png(raw_mask, output_paths["raw_mask_png"])
    save_mask_png(clean_mask, output_paths["clean_mask_png"])

    polygon_overlay_bgr = cv2.cvtColor(
        polygon_overlay_rgb,
        cv2.COLOR_RGB2BGR,
    )

    cv2.imwrite(
        output_paths["polygon_overlay_png"],
        polygon_overlay_bgr,
    )


def save_vector_outputs(args, clean_mask, output_paths):
    if not args.export_vectors:
        return None

    gdf = save_vector_polygons(
        mask=clean_mask,
        raster_path=args.image_path,
        out_path=output_paths["polygons_geojson"],
        min_area=args.vector_min_area,
    )

    save_vector_polygons(
        mask=clean_mask,
        raster_path=args.image_path,
        out_path=output_paths["polygons_gpkg"],
        min_area=args.vector_min_area,
    )

    return gdf


def save_showcase_crop(
    image_rgb,
    prob_map,
    clean_mask,
    polygon_overlay_rgb,
    output_paths,
    crop_x,
    crop_y,
    crop_size,
):
    h, w = image_rgb.shape[:2]

    if crop_size < 1:
        raise ValueError("--crop_size must be at least 1.")

    if crop_x is None:
        crop_x = max(0, (w - crop_size) // 2)

    if crop_y is None:
        crop_y = max(0, (h - crop_size) // 2)

    x1 = max(0, crop_x)
    y1 = max(0, crop_y)
    x2 = min(w, x1 + crop_size)
    y2 = min(h, y1 + crop_size)

    image_crop = image_rgb[y1:y2, x1:x2]
    prob_crop = prob_map[y1:y2, x1:x2]
    mask_crop = clean_mask[y1:y2, x1:x2]
    overlay_crop = polygon_overlay_rgb[y1:y2, x1:x2]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    axes[0].imshow(image_crop)
    axes[0].set_title("Input crop")
    axes[0].axis("off")

    axes[1].imshow(prob_crop, cmap="gray")
    axes[1].set_title("Probability map")
    axes[1].axis("off")

    axes[2].imshow(mask_crop, cmap="gray", interpolation="nearest")
    axes[2].set_title("Clean mask")
    axes[2].axis("off")

    axes[3].imshow(overlay_crop)
    axes[3].set_title("Polygon overlay")
    axes[3].axis("off")

    plt.tight_layout(pad=0.6)
    plt.savefig(output_paths["showcase_crop_png"], dpi=150)
    plt.close()


def print_saved_outputs(output_paths):
    print()
    print("Saved probability map:     ", output_paths["prob_npy"])
    print("Saved probability preview: ", output_paths["prob_png"])
    print("Saved raw binary mask:     ", output_paths["raw_mask_png"])
    print("Saved cleaned binary mask: ", output_paths["clean_mask_png"])
    print("Saved polygon overlay:     ", output_paths["polygon_overlay_png"])
    print("Saved showcase crop:       ", output_paths["showcase_crop_png"])
    print()
    print("Done.")


def main():
    args = parse_args()
    config = load_config(args.config)
    args = apply_config(args, config)

    os.makedirs(args.out_dir, exist_ok=True)

    output_name = get_output_name(
        image_path=args.image_path,
        output_name=args.output_name,
    )

    output_paths = build_output_paths(
        out_dir=args.out_dir,
        output_name=output_name,
    )

    print_config(args)

    image_rgb, prob_map = run_full_image_inference(args)

    raw_mask, clean_mask = generate_masks(
        args=args,
        prob_map=prob_map,
    )

    polygon_overlay_rgb = generate_polygon_overlay(
        args=args,
        image_rgb=image_rgb,
        clean_mask=clean_mask,
    )

    save_outputs(
        prob_map=prob_map,
        raw_mask=raw_mask,
        clean_mask=clean_mask,
        polygon_overlay_rgb=polygon_overlay_rgb,
        output_paths=output_paths,
    )

    vector_gdf = save_vector_outputs(
        args=args,
        clean_mask=clean_mask,
        output_paths=output_paths,
    )

    save_showcase_crop(
        image_rgb=image_rgb,
        prob_map=prob_map,
        clean_mask=clean_mask,
        polygon_overlay_rgb=polygon_overlay_rgb,
        output_paths=output_paths,
        crop_x=args.crop_x,
        crop_y=args.crop_y,
        crop_size=args.crop_size,
    )

    print_saved_outputs(output_paths)

    if vector_gdf is not None:
        print("Saved GIS polygons:        ", output_paths["polygons_geojson"])
        print("Saved GIS polygons:        ", output_paths["polygons_gpkg"])
        print("Vector polygon count:      ", len(vector_gdf))


if __name__ == "__main__":
    main()
