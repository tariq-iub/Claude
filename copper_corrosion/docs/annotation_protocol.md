# Annotation and adjudication protocol

## Roles
- **Annotators (at least 2):** trained on the class guide below. During the independent phase they are blinded to model output and to each other's work.
- **Adjudicator:** a senior materials scientist who did not take part in the independent labelling of the same image.

## Selecting the reference subset (set before any labelling)
1. Stratify by specimen and acquisition session. Draw at least 60 images (more if budget allows) with a fixed random seed.
   Use stratified sampling on Approach A's predicted class mix, so that rare classes appear, but **do not** use model error to pick images.
2. Split the subset at specimen level into tuning (≈ 40%) and **locked test** (≈ 60%). The test masks are never used for
   training, threshold selection or prompt design.
3. Images outside the subset make up the active-learning pool. Their reviewed masks are **model-assisted** and must never enter the test set.

## Phase 1: independent labelling (blinded)
- Use app mode "Independent (blinded)". The app shows no proposals, and the image order is randomised per annotator.
- Label every pixel as one of the 6 classes or **Unknown/Uncertain**. Use Uncertain for glare, out-of-focus areas, scale bars,
  and any material whose class you cannot decide. Never assign a corrosion class just to avoid choosing Uncertain.
- Saved revisions carry `blinded=true` and record the time spent.

## Phase 2: adjudication
- Predeclared policy: pixels where every annotator gave the same label keep that label. Every other pixel, and any pixel marked Uncertain by
  anyone, goes to the adjudicator (`consensus_proposal`).
- The adjudicator must give a written rationale. The store keeps the original masks, the disagreement fraction,
  the decision, and the hashes of all inputs.
- The final reference may still contain 255 (genuinely undecidable pixels). These are excluded from every metric.

## Phase 3: model-assisted review (pool images only)
Use app mode "Review". The reviewer accepts, rejects or corrects a proposal. Every save is a new revision, and the proposals are kept.
Correction time is recorded, which supports cost-of-annotation analysis.

## Class guide (visual. Calibrate it with the specimens the expert knows best)
| Class | Typical appearance | Common confusions |
|---|---|---|
| Healthy | metallic salmon/orange-brown copper, specular | Cu2O (thin films) |
| Cu2O | red-brown to pinkish or orange film | Healthy, CuO |
| CuO | dark brown to black | shadows, the dark CuCl2 reference |
| CuCl | pale, whitish to grey-green (nantokite) | Patina, CuCl2, glare |
| CuCl2 | blue-green to cyan | Patina |
| Patina | green, blue-green, matte | CuCl, CuCl2 |

## Training
Each annotator labels the same 5 non-test practice images and discusses disagreements with the adjudicator before
Phase 1. Practice images are excluded from all analyses.
