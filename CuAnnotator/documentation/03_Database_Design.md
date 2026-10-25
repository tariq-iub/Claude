# 03 — Database Design

## Storage strategy

| Data | Store | Rationale |
|---|---|---|
| Project, taxonomy, images metadata, annotations (masks as RLE), revisions | `project.cuann.json` (native JSON, authoritative) | Single atomic file, human-inspectable, versioned schema, lossless |
| Previous valid project | `project.cuann.json.bak` | Recovery from a damaged main file |
| Unsaved work | `project.autosave.json` | Crash recovery without overwriting the saved state |
| Audit trail | `audit.jsonl` | Append-only, flushed per entry, independent of saves |
| Query index | `index.db` (SQLite) | Fast filtering, external analysis with standard SQL tools |
| Images | `images/` (byte-identical copies) or external absolute paths | Large binaries stay on the filesystem |

The SQLite database is an **index**, rebuilt transactionally on every save from the JSON project. If the
index cannot be written, the save still succeeds (the JSON is the source of truth) and a warning is logged.
Masks are not stored as BLOBs: they are compact COCO-RLE strings inside the JSON.

## SQLite schema (index schema version 1)

```sql
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);          -- indexSchemaVersion, projectId, projectName, projectSchemaVersion
CREATE TABLE classes(id INTEGER PRIMARY KEY, name TEXT NOT NULL, formula TEXT, color TEXT, parent TEXT,
                     enabled INTEGER, reserved INTEGER, required_verification TEXT);
CREATE TABLE images(id TEXT PRIMARY KEY, file_name TEXT, rel_path TEXT, width INTEGER, height INTEGER, sha256 TEXT,
                    specimen TEXT, workflow_state TEXT, annotation_count INTEGER, imported_at TEXT);
CREATE TABLE annotations(id TEXT PRIMARY KEY, image_id TEXT REFERENCES images(id), class_id INTEGER, verification TEXT,
                    verification_level INTEGER, area INTEGER, bbox_x INTEGER, bbox_y INTEGER, bbox_w INTEGER, bbox_h INTEGER,
                    revision INTEGER, created_at TEXT, modified_at TEXT, created_by TEXT, reviewed_by TEXT,
                    algorithm TEXT, annotation_set TEXT, notes TEXT);
CREATE INDEX ix_ann_image ON annotations(image_id);
CREATE INDEX ix_ann_class ON annotations(class_id);
CREATE INDEX ix_ann_verif ON annotations(verification_level);
CREATE TABLE revisions(annotation_id TEXT, revision INTEGER, timestamp TEXT, author TEXT, reason TEXT,
                    class_id INTEGER, verification TEXT, pixels INTEGER, PRIMARY KEY(annotation_id, revision));
CREATE TABLE audit_log(seq INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, actor TEXT, action TEXT,
                    image_id TEXT, annotation_id TEXT, details TEXT);
```

Timestamps are ISO 8601 with offset (UTC). Verification is stored both as text and as an ordinal level
(0 Unverified … 3 InstrumentallyVerified) so queries such as "everything below expert review" are simple.

Note: the bundled `e_sqlite3` enables foreign keys; tables are dropped child-first during a rebuild.

## Example queries

```sql
-- Review queue: chemically relevant labels not yet expert-reviewed
SELECT i.file_name, a.class_id, a.verification, a.area FROM annotations a JOIN images i ON i.id = a.image_id
WHERE a.verification_level < 2 AND a.class_id NOT IN (1, 255) ORDER BY i.file_name;

-- Pixels per class
SELECT k.name, SUM(a.area) FROM annotations a JOIN classes k ON k.id = a.class_id GROUP BY k.name;

-- Who changed what
SELECT timestamp, actor, action, details FROM audit_log WHERE action LIKE '%Accepted%' ORDER BY seq;
```

`SqliteProjectIndex.Query(AnnotationQuery)` implements the parameterised filters used by the application.
