# 08 — Project Persistence

## Folder layout

```
MyProject/
├── project.cuann.json        authoritative project (native JSON, schema v1)
├── project.cuann.json.bak    previous valid version (kept by every save)
├── project.autosave.json     recovery data (only while there are unsaved changes)
├── index.db                  SQLite index (rebuilt on save)
├── audit.jsonl               append-only audit trail
├── images/                   imported originals (byte-identical copies; optional)
└── exports/                  default location for dataset exports
```

Paths to images are stored relative to the project folder (forward slashes) when the image is inside it,
absolute otherwise. The project folder can be moved or copied as a whole.

## Save algorithm (`ProjectStore.SaveAsync`)

1. Serialise the project.
2. **Self-check:** deserialise the produced JSON; if this fails, nothing is written.
3. Write to `.<name>.<guid>.tmp` in the same folder with `FileOptions.WriteThrough` and `Flush(true)`.
4. `File.Replace(tmp, project.cuann.json, project.cuann.json.bak)` — atomic replacement keeping the previous
   file as backup (first save: `File.Move`).
5. Rebuild the SQLite index in one transaction (failure is logged, not fatal).
6. Delete `project.autosave.json`.

The previous valid project is never truncated before the new file is fully on disk. Stale temp files from an
interrupted save are removed on the next load.

## Load algorithm

1. Remove stale temp files.
2. Read and validate `project.cuann.json` (format name, schema version ≤ supported; RLE pixel counts are checked
   against the stored `pixels` value to detect corruption).
3. On failure, load `project.cuann.json.bak` and report `LastLoadWarning` to the user.
4. If `project.autosave.json` is newer than the project file, the UI offers to restore it.

## Autosave and recovery

The autosave timer (project setting, default 120 s, minimum 15 s, 0 = off) writes `project.autosave.json`
(atomically) when there are unsaved changes. It never overwrites the main project file, so a crash during
autosave cannot damage the last manual save.

## Re-editing

Everything needed to continue editing is in the JSON: masks (exact), polygons (including edited vertices),
classes, verification and evidence, revision history (with masks), visibility, Z-order, annotation set,
algorithm parameters, efficiency data and project settings. Undo history is per session and not persisted
(revision history is).

## Revisions

See 04. Revisions are stored inline with their masks; the SQLite `revisions` table indexes them.

## Native JSON (excerpt)

```json
{
  "format": "CuAnnotator.Project",
  "schemaVersion": 1,
  "appVersion": "1.0.0",
  "savedAt": "2026-10-25T10:31:30.0000000+00:00",
  "coordinateConvention": "Origin top-left; X right; Y down; zero-based …",
  "rasterizationRule": "Components largest-first; …",
  "project": {
    "name": "Cu corrosion study",
    "taxonomy": { "name": "…", "version": 1, "classes": [ { "id": 2, "name": "Cu2O", "colorHex": "#D03030", … } ] },
    "images": [ {
      "id": "…", "fileName": "specimen_001.png", "relativePath": "images/specimen_001.png",
      "width": 3000, "height": 2000, "sha256": "…", "specimenId": "S-001",
      "annotations": [ {
        "id": "…", "classId": 2,
        "mask": { "bounds": [1201, 730, 421, 421], "pixels": 101233, "rle": "…compressed COCO RLE…" },
        "polygons": [ { "outer": [[1290, 730], [1291, 731], …], "holes": [] } ],
        "verificationStatus": "VisuallyEstimated",
        "segmentationAlgorithm": "GrabCut", "algorithmVersion": "1.0.0",
        "algorithmParameters": { "GrabCutIterations": "5", "RandomSeed": "12345", "efficiency.interactions": "3", … },
        "revision": 2, "revisionHistory": [ { "revision": 1, "reason": "Polygon edit", "mask": { … } } ]
      } ]
    } ]
  }
}
```

Masks use COCO compressed RLE (column-major) relative to `bounds` — the same codec as the COCO exporter,
tested against a pycocotools reference string.

## Schema evolution

`schemaVersion` is checked on load: newer files are refused with an explanation; older versions are migrated in
`NativeJsonSerializer.Migrate` (v1 is the first release, so migration currently only normalises references).
