#!/bin/bash
# Install SAM2 camera predictor on cluster

SAM2_DIR="/home/barakaeli/.local/lib/python3.9/site-packages/sam2"

echo "Installing SAM2 camera predictor..."
echo "SAM2 directory: $SAM2_DIR"
echo ""

# 1. Copy camera predictor file
echo "1. Copying sam2_camera_predictor.py..."
cp sam2_camera_predictor.py "$SAM2_DIR/"
if [ $? -eq 0 ]; then
    echo "   ✓ Camera predictor copied successfully"
else
    echo "   ✗ Failed to copy camera predictor"
    exit 1
fi

# 2. Check if build_sam2_camera_predictor function already exists
echo ""
echo "2. Checking build_sam.py..."
if grep -q "def build_sam2_camera_predictor" "$SAM2_DIR/build_sam.py"; then
    echo "   ✓ build_sam2_camera_predictor already exists in build_sam.py"
else
    echo "   Adding build_sam2_camera_predictor function to build_sam.py..."

    # Create backup
    cp "$SAM2_DIR/build_sam.py" "$SAM2_DIR/build_sam.py.backup"

    # Add the function after build_sam2_video_predictor
    cat >> "$SAM2_DIR/build_sam.py" << 'CAMERA_PREDICTOR_EOF'


def build_sam2_camera_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    vos_optimized=False,
    **kwargs,
):
    hydra_overrides = [
        "++model._target_=sam2.sam2_camera_predictor.SAM2CameraPredictor",
    ]

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model
CAMERA_PREDICTOR_EOF

    echo "   ✓ build_sam2_camera_predictor function added"
    echo "   (Backup saved to build_sam.py.backup)"
fi

echo ""
echo "============================================"
echo "Installation complete!"
echo "============================================"
echo ""
echo "Verify installation:"
echo "  python -c 'from sam2.build_sam import build_sam2_camera_predictor; print(\"Camera predictor available!\")'"
