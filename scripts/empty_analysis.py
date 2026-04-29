import cv2
from glob import glob
from tqdm import tqdm

mask_paths = sorted(glob("data/tiles_256/train/masks/*.png"))

empty = 0
non_empty = 0
building_pixels = 0
total_pixels = 0

for p in tqdm(mask_paths):
    mask = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    binary = mask > 127

    if binary.sum() == 0:
        empty += 1
    else:
        non_empty += 1

    building_pixels += binary.sum()
    total_pixels += binary.size

print("Total masks:", len(mask_paths))
print("Empty masks:", empty)
print("Non-empty masks:", non_empty)
print("Empty %:", empty / len(mask_paths) * 100)
print("Building pixel %:", building_pixels / total_pixels * 100)