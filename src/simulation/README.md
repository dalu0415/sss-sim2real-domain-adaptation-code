# Final simulation source generation

This directory contains the code snapshot used to build the final semi-synthetic source domain described in the paper. It keeps the original stage boundaries and fixed random seeds. Early prototypes, rejected simulation variants, generated images, experiment outputs, and third-party raw datasets are not included.

This is a reference implementation, not a one-command package. Replace every `PATH_TO_*` placeholder with a valid local path before running a script. The required AI4Shipwrecks and Marine-PULSE inputs must be obtained separately under their own terms. The local silhouette inputs, curated `relook_strict.csv` frame manifest, and baseline/intermediate products named below are also not tracked, so this snapshot does not by itself rebuild the paper's 490-image source domain.

## Script order

1. Prepare the real seabed background blocks:

   ```text
   python bg_prepare.py
   ```

2. Extract and filter the baseline URM texture statistics:

   ```text
   python urm_process.py
   python filter_urm.py
   ```

3. Extract and filter the AI4Shipwrecks-derived highlight statistics. `ai4_process.py` also reads the filtered URM statistics for the shadow component and requires the locally prepared `relook_strict.csv` frame-selection manifest:

   ```text
   python ai4_process.py
   python filter_ai4.py
   ```

4. Build the paired background mean/standard-deviation pool:

   ```text
   python ai4_bg_brightness_pool_joint.py
   ```

5. Prepare the baseline object-mask set. `masks.py` contains the airplane/ship mask-generation implementation and the final Step-2 shadow functions, but its standalone `main()` is deliberately guarded to prevent accidental writes. Review its input/output placeholders and the guard before using the archived mask-generation body.

6. Bake the final 400 Route-B arm2 ship templates. `build_v5.py` is imported automatically from the same directory:

   ```text
   python bake_armtemplates_route_b.py
   ```

   Set `PATH_TO_STEP3_TEMPLATE_OUTPUT` to the same `step3_arm_templates` root that `full_generate_denorm_b2.py` reads.

   The baker evaluates the former arm3 superset only in memory to preserve the validated random-number stream. It writes no arm3 templates, metadata, images, or montage.

7. Generate the arm1 base set and the final arm2 ship set:

   ```text
   python full_generate_denorm_b2.py --mode full --shadow_mode new --step3 off
   python full_generate_denorm_b2.py --mode full --shadow_mode new --step3 on --step3_arm arm2
   ```

8. Point `ARM1` and `ARMS["arm2"]` in `_finalize_step5_airplane.py` to the generated datasets, then merge the 90 unchanged airplane samples into the 400-ship arm2 set:

   ```text
   python _finalize_step5_airplane.py
   ```

The expected final dataset contains 490 images: 90 airplane samples and 400 ship samples.

## File roles

- `bg_prepare.py`: real seabed background preparation.
- `urm_process.py`, `filter_urm.py`: baseline URM texture-statistics extraction and filtering.
- `ai4_process.py`, `filter_ai4.py`: AI4Shipwrecks-derived highlight-statistics extraction and filtering.
- `ai4_bg_brightness_pool_joint.py`: paired background brightness/contrast pool.
- `masks.py`: object-mask geometry and the final heuristic shadow implementation.
- `build_v5.py`, `bake_armtemplates_route_b.py`: final procedural ship geometry and arm2 template baking.
- `full_generate_denorm_b2.py`: final composition, intensity variation, shadowing, cropping, and saving.
- `_finalize_step5_airplane.py`: final 90-airplane/400-ship merge.
