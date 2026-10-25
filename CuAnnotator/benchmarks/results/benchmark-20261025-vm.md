# CuAnnotator benchmark results

- Date (UTC): 2026-10-25 17:46
- Machine: vm; OS: Ubuntu 24.04.5 LTS; .NET 10.0.13
- CPU: Intel(R) Xeon(R) Processor @ 2.10GHz; logical processors visible: 4
- OpenCV: 4.13.0; build configuration: Release
- Peak working set of the benchmark process: 3708 MB
- Total CPU time / wall time: 448.8 s / 431.7 s
- Each value is the median of 3 runs after one warm-up run. Synthetic images (ellipses + Gaussian noise).

| Group | Case | Size | Median (ms) | Note |
|---|---|---|---:|---|
| I/O | Decode PNG + checksum + metadata | 3840×2160 | 255.3 | 14.4 MB file |
| Colour | sRGB→CIELAB (exact, full image) | 3840×2160 | 254.0 | 8.3 MP |
| Colour | CIEDE2000 distance map 1024×1024 | 3840×2160 | 110.9 | 1.05 MP, parallel |
| Segmentation | GrabCut 5 it. (Precise) | ROI 512² | 3834.2 | 3840×2160 image |
| Segmentation | GrabCut 5 it. (Balanced) | ROI 1024² | 4564.6 | 3840×2160 image |
| Segmentation | GrabCut 5 it. (Balanced) | ROI 2048² | 2091.4 | 3840×2160 image |
| Segmentation | GrabCut 5 it. (Fast) | ROI 2048² | 906.1 | 3840×2160 image |
| Segmentation | GrabCut 5 it. (Precise) | ROI 1024² | 64465.8 | 3840×2160 image |
| Segmentation | LabColor | ROI 800² | 97.3 | default parameters |
| Segmentation | Watershed | ROI 800² | 99.0 | default parameters |
| Segmentation | ActiveContour | ROI 800² | 1285.9 | default parameters |
| Segmentation | Hybrid | ROI 800² | 4837.7 | default parameters |
| Masks | Compose semantic label map (200 annotations) | 3840×2160 | 40.7 |  |
| Masks | Write 8-bit class-ID PNG | 3840×2160 | 78.6 |  |
| Export | COCO build + serialise (200 ann., auto mode) | 3840×2160 | 113.7 |  |
| Export | COCO build + serialise (200 ann., RLE) | 3840×2160 | 82.5 |  |
| Project | Save (JSON + SQLite index, 200 ann.) | 3840×2160 | 725.9 |  |
| Project | Load (200 ann.) | 3840×2160 | 235.1 | 11.60 MB JSON |
| I/O | Decode PNG + checksum + metadata | 7680×4320 | 917.3 | 57.4 MB file |
| Colour | sRGB→CIELAB (exact, full image) | 7680×4320 | 613.9 | 33.2 MP |
| Colour | CIEDE2000 distance map 1024×1024 | 7680×4320 | 58.3 | 1.05 MP, parallel |
| Segmentation | GrabCut 5 it. (Precise) | ROI 512² | 3264.1 | 7680×4320 image |
| Segmentation | GrabCut 5 it. (Balanced) | ROI 1024² | 4369.5 | 7680×4320 image |
| Segmentation | GrabCut 5 it. (Balanced) | ROI 2048² | 6931.3 | 7680×4320 image |
| Segmentation | GrabCut 5 it. (Fast) | ROI 2048² | 5693.5 | 7680×4320 image |
| Masks | Compose semantic label map (200 annotations) | 7680×4320 | 74.4 |  |
| Masks | Write 8-bit class-ID PNG | 7680×4320 | 313.3 |  |
| Export | COCO build + serialise (200 ann., auto mode) | 7680×4320 | 78.5 |  |
| Export | COCO build + serialise (200 ann., RLE) | 7680×4320 | 68.1 |  |
| Project | Save (JSON + SQLite index, 200 ann.) | 7680×4320 | 385.5 |  |
| Project | Load (200 ann.) | 7680×4320 | 186.0 | 11.65 MB JSON |

WPF rendering is not measured here (requires Windows); see documentation/11_Performance_Optimization.md.
