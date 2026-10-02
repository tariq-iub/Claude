# Paper-drafting material (no results are asserted; fill from runs)

## Title options (refine after the full literature check)
1. Physics-Constrained Virtual Polarimetric Imaging for Copper and Bronze Corrosion Assessment from Conventional RGB Images
2. VP-CorrosionNet: Software-Defined Polarimetric Imaging for Glare-Robust Metallic Corrosion Inspection
3. From Physical Polarizers to Virtual Polarimetric Cameras: Physics-Guided Deep Learning for Copper/Bronze Corrosion Assessment
4. Hardware-Supervised, Software-Defined Polarimetry for Reflective Metal Corrosion Inspection
5. How Much of Physical Polarization Imaging Can an RGB Camera Recover? A Hardware-to-Software Study on Cartridge Corrosion
6. Conductor-Aware Virtual Polarimetry: Latent Optical State Estimation for Metal Surface Inspection
7. Counterfactual Polarimetric Ensembles for Uncertainty-Aware Corrosion Assessment on Curved Metallic Surfaces
8. Polarimetric Surface Response Fields: Analyzer-Dependent Features for Corrosion Segmentation from Single Images
9. Distilling Polarization Hardware into an RGB Network for Copper Corrosion Inspection
10. Geometry-Aware Virtual Polarization for Curved Copper Surfaces: Fresnel Physics Meets Corrosion Segmentation
11. Physical, Virtual and Distilled Polarization for Corrosion Inspection of Cartridge Cases: A Controlled Comparison
12. Edge-Deployable Physics-Guided Polarimetric Corrosion Inspection from a Single RGB Frame

## Abstract template (placeholders in [brackets] must be filled from experiments)
Polarization imaging suppresses glare on curved reflective metals but needs dedicated optics at deployment. We ask how much of its corrosion-discriminative information can be transferred to a software-defined virtual polarimetric camera operating on ordinary RGB images. We model [conductor Fresnel reflection, microfacet roughness, geometry] in a differentiable physics layer driven by a latent optical state, expose the unavoidable single-image ambiguity through a counterfactual ensemble, and train a lightweight network (≈ [N] M parameters; [X] ms CPU) for corrosion segmentation, pitting and severity, optionally supervised by measured Stokes data from a hardware teacher. On [N] cartridges ([split protocol]) we find [result of H1], [result of H3 with CI], [virtual-vs-measured DoLP/AoLP agreement]. [Limitations sentence].

## Claims allowed only if supported (map to hypotheses)
C1 virtual analyzer/stack is Stokes-consistent [tests: yes]; C2 virtual features improve glare-stratified segmentation [H1: pending]; C3 distillation closes [x %] of the gap [H3: pending]; C4 geometry prior helps [H4: pending]; C5 PSRF more robust than single θ [H5: pending]; C6 CPE variance tracks error [RQ6: pending]; C7 runs at [x] FPS CPU [benchmark: measure]. Do **not** write "first" or "replaces the polarizer".

## Figure captions (draft)
Fig 1 Framework: deployment path (top) and training-only hardware-teacher branch (bottom); hatted quantities are estimates. Fig 2 Physical vs virtual DoLP/AoLP on [held-out cartridge]; AoLP error wrapped (period π). Fig 3 Fresnel/DoLP/retardance/diffuse DoLP models for copper (proxy constants) and a dielectric. Fig 4 Virtual analyzer sweep: markers at eight canonical angles, line = dense sweep. Fig 5 Stokes, DoLP, AoLP maps. Fig 6 RGB vs virtual parallel/crossed (convention unvalidated). Fig 7–8 segmentation/pitting vs baselines (held-out). Fig 9 Analyzer-angle objective terms and J(θ); line = periodic PCHIP, points = sampled angles. Fig 10 A/B/C comparison, seeds as points. Fig 11 ablation, seeds as points with mean ± 95 % CI. Fig 12 robustness (measured levels, mean ± SD). Fig 13 accuracy vs CPU latency. Fig 14 reliability, risk–coverage, entropy map. Fig 15 failure cases (selected by rule: lowest pixel accuracy).

## Paper series (merge if results are thin)
P1 Virtual polarimetric imaging from RGB (physics, validation against hardware, ambiguity/CPE). P2 Polarization-aware copper/bronze corrosion segmentation (features, colour spaces, fusion, pitting). P3 Hardware-to-software distillation (gap experiment). P4 Uncertainty-aware inspection with abstention (could merge with P2/P3). P5 Edge deployment (engineering; likely a section of P2/P4).
