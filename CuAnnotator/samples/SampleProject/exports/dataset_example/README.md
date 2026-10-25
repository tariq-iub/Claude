# Synthetic copper corrosion example — CuAnnotator dataset export

Exported 2026-10-25 16:44 UTC with CuAnnotator 1.0.0; 2 image(s).

## Layout

- `images/` source images (lossless PNG or byte-identical copies)
- `masks/` semantic masks: 8-bit single-channel PNG, same size as the image, pixel value = class ID (0 unlabeled, 255 ignore)
- `colorized_masks/` RGB visualisation only — do not train on these
- `annotations/` native per-image JSON (lossless, re-editable)
- `coco/` COCO instances (polygons, or compressed RLE when holes are present)
- `yolo/` YOLO segmentation labels (see warnings for holes / multipart regions)
- `metadata/` classes, dataset split and checksums, verification status

## Classes

| ID | Name | Formula |
|---|---|---|
| 0 | Unlabeled |  |
| 1 | Healthy Copper |  |
| 2 | Cu2O | Cu₂O |
| 3 | CuO | CuO |
| 4 | CuCl | CuCl |
| 5 | CuCl2 | CuCl₂ |
| 6 | Malachite | Cu₂CO₃(OH)₂ |
| 7 | Brochantite | Cu₄SO₄(OH)₆ |
| 8 | Basic Copper Chlorides | Cu₂(OH)₃Cl |
| 255 | Ignore |  |

## Scientific note

Class labels are annotations, not chemical measurements. Check `metadata/verification.json`: only annotations marked `InstrumentallyVerified` are backed by recorded analytical evidence.

## Export warnings

- YOLO: specimen_A_01.png annotation 8ff541bc-72fe-46fd-b38f-b7e87e7ceb61 has holes and was skipped (format cannot represent holes).
- YOLO: specimen_A_01.png annotation f5b7549d-76fd-46b1-9248-0ed0fab3c04e has 3 disconnected parts, written as separate instances.
- YOLO: specimen_B_01.png annotation 70fd1181-ac09-48b8-8758-29a10c769426 has holes and was skipped (format cannot represent holes).
- YOLO: specimen_B_01.png annotation 734901df-1f71-4d99-9d4c-0055b503d32b has 4 disconnected parts, written as separate instances.
