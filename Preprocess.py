import rasterio
import numpy as np
import os
from scipy.ndimage import zoom


dates = ['20240624', '20240626', '20240629', '20240704', '20240707',
         '20240708', '20240709', '20240712', '20240713', '20240715',
         '20240719', '20240724', '20240725', '20240727', '20240801',
         '20240802', '20240804', '20240805', '20240806', '20240819',
         '20240820', '20240821', '20240822', '20240824', '20240826',
         '20240711', '20240717', '20240815', '20240827']

output_folder = [
    '2024_176_Jun24',  # 20240624
    '2024_178_Jun26',  # 20240626
    '2024_181_Jun29',  # 20240629
    '2024_186_Jul04',  # 20240704
    '2024_189_Jul07',  # 20240707
    '2024_190_Jul08',  # 20240708
    '2024_191_Jul09',  # 20240709
    '2024_194_Jul12',  # 20240712
    '2024_195_Jul13',  # 20240713
    '2024_197_Jul15',  # 20240715
    '2024_201_Jul19',  # 20240719
    '2024_206_Jul24',  # 20240724
    '2024_207_Jul25',  # 20240725
    '2024_209_Jul27',  # 20240727
    '2024_214_Aug01',  # 20240801
    '2024_215_Aug02',  # 20240802
    '2024_217_Aug04',  # 20240804
    '2024_218_Aug05',  # 20240805
    '2024_219_Aug06',  # 20240806
    '2024_232_Aug19',  # 20240819
    '2024_233_Aug20',  # 20240820
    '2024_234_Aug21',  # 20240821
    '2024_235_Aug22',  # 20240822
    '2024_237_Aug24',  # 20240824
    '2024_239_Aug26',  # 20240826
    '2024_193_Jul11',  # 20240711
    '2024_199_Jul17',  # 20240717
    '2024_228_Aug15',  # 20240815
    '2024_240_Aug27'   # 20240827
]

base_dir = "data/data_used"
output_root = "data/final_datasets_for_swinstfm"
output_root_tif = "data/final_datasets_for_swinstfm_tif"
os.makedirs(output_root, exist_ok=True)
os.makedirs(output_root_tif, exist_ok=True)

# Target size
TARGET_H, TARGET_W = 2710, 2637

UAV_BAND_ORDER = [2, 1, 0, 3, 4]  # R,G,B,RE,NIR
PS_BAND_ORDER = [5, 3, 1, 6, 7]    # R,G,B,RE,NIR

# Reference georeferencing from 20240624 UAV
ref_profile = None
ref_transform = None
ref_crs = None

ref_uav_path = os.path.join(base_dir, "20240624_Bottoms_UAV.tif")
if os.path.exists(ref_uav_path):
    with rasterio.open(ref_uav_path) as src:
        ref_profile = src.profile.copy()
        ref_transform = src.transform
        ref_crs = src.crs
    print("Loaded reference georeferencing from 20240624 UAV")
else:
    print("WARNING: Reference UAV image 20240624_Bottoms_UAV.tif not found! TIFFs will have default profile if saved.")

for date, outf in zip(dates, output_folder):
    uav_path = os.path.join(base_dir, f"{date}_Bottoms_UAV.tif")
    ps_path = os.path.join(base_dir, f"{date}_Bottoms_PS.tif")
    
    date_out = os.path.join(output_root, outf)
    date_out_tif = os.path.join(output_root_tif, outf)
    os.makedirs(date_out, exist_ok=True)
    os.makedirs(date_out_tif, exist_ok=True)
    
    has_uav = os.path.exists(uav_path)
    has_ps = os.path.exists(ps_path)
    
    if not has_uav and not has_ps:
        print(f"Skipping {date} — no UAV or PS image")
        continue
    
    # Prepare output profile for TIFFs (use reference if available)
    tiff_profile = ref_profile.copy() if ref_profile else {}
    tiff_profile.update(
        height=TARGET_H,
        width=TARGET_W,
        count=5,
        dtype=rasterio.float32
    )
    if ref_transform:
        # Scale transform to new dimensions using original reference size
        orig_ref_h, orig_ref_w = ref_profile['height'], ref_profile['width']
        scale_y = orig_ref_h / TARGET_H
        scale_x = orig_ref_w / TARGET_W
        tiff_profile['transform'] = ref_transform * rasterio.Affine.scale(scale_x, scale_y)
    if ref_crs:
        tiff_profile['crs'] = ref_crs
    
    if has_uav:
        # Load UAV
        with rasterio.open(uav_path) as src:
            uav = src.read().astype(np.float32)
            orig_h, orig_w = src.height, src.width
        
        # Resize UAV to target (bilinear)
        if (orig_h, orig_w) != (TARGET_H, TARGET_W):
            print(f"{date} UAV: {orig_h}×{orig_w} → {TARGET_H}×{TARGET_W}")
            uav_resized = np.zeros((uav.shape[0], TARGET_H, TARGET_W), dtype=np.float32)
            for b in range(uav.shape[0]):
                uav_resized[b] = zoom(uav[b], (TARGET_H / orig_h, TARGET_W / orig_w), order=1)
        else:
            uav_resized = uav
        
        # Clean invalid pixels
        invalid = (
            (np.all(uav_resized == 0, axis=0)) |
            (np.all(uav_resized == 65535, axis=0)) |
            (np.any(uav_resized < 0, axis=0))
        )
        if invalid.any():
            for b in range(uav_resized.shape[0]):
                band = uav_resized[b]
                median_val = np.median(band[~invalid]) if (~invalid).any() else 0
                band[invalid] = median_val
                uav_resized[b] = band
        
        uav_final = uav_resized[UAV_BAND_ORDER]
        
        # Save .npy
        np.save(os.path.join(date_out, f"UAV_{date}.npy"), uav_final)
        print(f"Saved UAV: UAV_{date}.npy")
        
        # Save georeferenced TIFF
        if ref_profile:
            with rasterio.open(os.path.join(date_out_tif, f"UAV_{date}.tif"), 'w', **tiff_profile) as dst:
                dst.write(uav_final)
            print(f"Saved georeferenced UAV: UAV_{date}_adjusted.tif")
        else:
            print("Skipped saving UAV TIFF (no reference profile)")
    
    if has_ps:
        # Load PS
        with rasterio.open(ps_path) as src:
            ps = src.read().astype(np.float32)
            ps_h, ps_w = src.height, src.width
        
        # Clean invalid pixels (before selecting bands)
        invalid_ps = (
            (np.all(ps == 0, axis=0)) |
            (np.all(ps == 65535, axis=0)) |
            (np.any(ps < 0, axis=0))
        )
        if invalid_ps.any():
            for b in range(ps.shape[0]):
                band = ps[b]
                median_val = np.median(band[~invalid_ps]) if (~invalid_ps).any() else 0
                band[invalid_ps] = median_val
                ps[b] = band
        
        ps_selected = ps[PS_BAND_ORDER]
        
        # Upsample PS to target (cubic)
        print(f"{date} PS: {ps_h}×{ps_w} → {TARGET_H}×{TARGET_W} (upsampled)")
        ps_upsampled = np.zeros((5, TARGET_H, TARGET_W), dtype=np.float32)
        for b in range(5):
            ps_upsampled[b] = zoom(ps_selected[b], (TARGET_H / ps_h, TARGET_W / ps_w), order=3)
        
        # Save .npy
        np.save(os.path.join(date_out, f"PS_{date}.npy"), ps_upsampled)
        print(f"Saved PS: PS_{date}.npy")
        
        # Save georeferenced TIFF
        if ref_profile:
            with rasterio.open(os.path.join(date_out_tif, f"PS_{date}_adjusted.tif"), 'w', **tiff_profile) as dst:
                dst.write(ps_upsampled)
            print(f"Saved georeferenced PS: PS_{date}.tif")
        else:
            print("Skipped saving PS TIFF (no reference profile)")
    
    # If no UAV but has PS (testing case), still process PS as above
    if not has_uav and has_ps:
        print(f"{date} has no UAV — processed PS for testing (upsampled + georeferenced)")

print("\nALL DONE! All processed images are exactly 2710×2637")
print("Saved .npy for model input and georeferenced .tif using 20240624 UAV reference.")
print("Ready for SwinSTFM!")