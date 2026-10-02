# Model architecture (as implemented)

```
sRGB ─► linearize ─► OpticalBranch ─► LatentOpticsHead ─► Z ─► PhysicsStokesLayer ─► Ŝ ─► VirtualAnalyzer ─► I_θ stack ─► PSRF ─► a0,a1,b1,(a2,b2)
 │                     (enc/dec, 0.5x width)     (23 ch: Z + log-var)                 (B,3,3,H,W)     (B,A,H,W luminance)
 └─► colour features (RGB | Lab | HSV) + texture (+x,y) ───────────────────────────────┐
                                  PolFeatureBuilder: spec/diff prob, DoLP_hat, cos/sin 2AoLP_hat, analyzer images, normals+discontinuity, r, PSRF, glare, u
                                                                                          ▼
                                    CorrosionBackbone {early | mid | late}  → decoder (edge-aware, ECA) → heads:
                                    seg (7 classes) · PittingHead (cues: multiscale Laplacian, shadow/highlight dipole, normal discontinuity, local std)
                                    SeverityHead (ordinal + [0,1] score + soft corroded-area %) · log-variance head · distill projection
```
* Blocks (`models/blocks.py`): depthwise-separable conv, MobileNetV2 inverted-residual bottlenecks, ECA channel attention, Sobel edge gating on skips, bilinear multi-scale decoder.
* Latent head outputs bounded proxies: r ∈ [0.05,1], η-scale ∈ [0.5,2], k-scale ∈ [0.5,1.5], unit normals with n_z > 0, ρ ∈ [0,1], (cos2φ, sin2φ) unit vector, g ∈ [0,1].
* PhysicsStokesLayer: environment or directional illumination; per-channel conductor Fresnel; Atkinson–Hancock diffuse; optional bounded learned residual polarization (`residual_gain`, default 0); realizability clamp.
* Cartridge prior (`cartridge_prior`): cylinder-fit normals (from the silhouette) enter the optical branch and are added to the predicted normals through a learnable weight.
* Teacher (`models/teacher.py`): consumes measured analyzer stacks; Stokes by least squares (not learned) + same backbone → seg/pit/severity + decoder features (distillation targets). Student = VPCorrosionNet (+ `distill_outputs`).
* Baselines (`models/baseline.py`): CLAHE, gamma, MSR Retinex, highlight suppression, specular removal, pseudo-polarizer (strawman), physics-only virtual polarizer, Lab/HSV thresholding, UNetLite, RGB-only mobile arm (same backbone, physics off). DeepLabV3+ and YOLO: **not bundled** (adapter contract in `EXPERIMENTAL_PROTOCOL.md`).
* Size (measured in the development container, defaults, not an accuracy claim): ≈ 0.12 M parameters (mid fusion; early 0.10 M, late 0.17 M), 0.048 GMACs at 96×96, ≈ 28 ms/frame CPU at 96×96 (4 cores). `scripts/benchmark.py` reproduces these on your hardware.
