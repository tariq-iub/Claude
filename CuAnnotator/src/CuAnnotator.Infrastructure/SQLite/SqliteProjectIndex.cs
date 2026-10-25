using CuAnnotator.Domain.Entities;
using CuAnnotator.Domain.Enums;
using Microsoft.Data.Sqlite;

namespace CuAnnotator.Infrastructure.SQLite;

public sealed class AnnotationQuery
{
    public Guid? ImageId { get; init; }
    public int? ClassId { get; init; }
    public VerificationStatus? Verification { get; init; }
    public VerificationStatus? MaxVerification { get; init; }
    public string? Reviewer { get; init; }
    public string? CreatedBy { get; init; }
    public DateTimeOffset? ModifiedAfter { get; init; }
    public DateTimeOffset? ModifiedBefore { get; init; }
    public ImageWorkflowState? WorkflowState { get; init; }
    public string? AnnotationSet { get; init; }
}

public sealed record AnnotationIndexRow(Guid AnnotationId, Guid ImageId, string ImageFile, int ClassId, string ClassName, VerificationStatus Verification,
    long Area, int Revision, DateTimeOffset ModifiedAt, string CreatedBy, string? ReviewedBy, ImageWorkflowState ImageState, string AnnotationSet);

/// <summary>
/// SQLite index of a project (images, annotations, revisions, classes, audit log). The native JSON file is
/// the source of truth; the index is rebuilt transactionally on every save and is used for fast filtering
/// (review queue) and external querying with standard SQLite tools. Masks and images stay on the
/// filesystem/JSON rather than as BLOBs.
/// </summary>
public sealed class SqliteProjectIndex(string databasePath)
{
    public const int IndexSchemaVersion = 1;

    private SqliteConnection Open()
    {
        var c = new SqliteConnection(new SqliteConnectionStringBuilder { DataSource = databasePath, Pooling = false }.ToString());
        c.Open();
        return c;
    }

    public void Rebuild(AnnotationProject project, IEnumerable<AuditEntry> audit)
    {
        using var c = Open();
        using var tx = c.BeginTransaction();
        Exec(c, tx, """
            DROP TABLE IF EXISTS revisions; DROP TABLE IF EXISTS annotations; DROP TABLE IF EXISTS audit_log;
            DROP TABLE IF EXISTS images; DROP TABLE IF EXISTS classes; DROP TABLE IF EXISTS meta;
            CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE classes(id INTEGER PRIMARY KEY, name TEXT NOT NULL, formula TEXT, color TEXT, parent TEXT, enabled INTEGER, reserved INTEGER, required_verification TEXT);
            CREATE TABLE images(id TEXT PRIMARY KEY, file_name TEXT, rel_path TEXT, width INTEGER, height INTEGER, sha256 TEXT,
                specimen TEXT, workflow_state TEXT, annotation_count INTEGER, imported_at TEXT);
            CREATE TABLE annotations(id TEXT PRIMARY KEY, image_id TEXT REFERENCES images(id), class_id INTEGER, verification TEXT, verification_level INTEGER,
                area INTEGER, bbox_x INTEGER, bbox_y INTEGER, bbox_w INTEGER, bbox_h INTEGER, revision INTEGER, created_at TEXT, modified_at TEXT,
                created_by TEXT, reviewed_by TEXT, algorithm TEXT, annotation_set TEXT, notes TEXT);
            CREATE INDEX ix_ann_image ON annotations(image_id); CREATE INDEX ix_ann_class ON annotations(class_id);
            CREATE INDEX ix_ann_verif ON annotations(verification_level);
            CREATE TABLE revisions(annotation_id TEXT, revision INTEGER, timestamp TEXT, author TEXT, reason TEXT, class_id INTEGER, verification TEXT, pixels INTEGER,
                PRIMARY KEY(annotation_id, revision));
            CREATE TABLE audit_log(seq INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, actor TEXT, action TEXT, image_id TEXT, annotation_id TEXT, details TEXT);
            """);
        void Ins(string sql, params (string, object?)[] ps)
        {
            using var cmd = c.CreateCommand();
            cmd.Transaction = tx; cmd.CommandText = sql;
            foreach (var (n, v) in ps) cmd.Parameters.AddWithValue(n, v ?? DBNull.Value);
            cmd.ExecuteNonQuery();
        }
        Ins("INSERT INTO meta VALUES('indexSchemaVersion', $v)", ("$v", IndexSchemaVersion.ToString()));
        Ins("INSERT INTO meta VALUES('projectId', $v)", ("$v", project.Id.ToString()));
        Ins("INSERT INTO meta VALUES('projectName', $v)", ("$v", project.Name));
        Ins("INSERT INTO meta VALUES('projectSchemaVersion', $v)", ("$v", project.SchemaVersion.ToString()));
        foreach (var k in project.Taxonomy.Classes)
            Ins("INSERT INTO classes VALUES($id,$n,$f,$c,$p,$e,$r,$rv)", ("$id", k.Id), ("$n", k.Name), ("$f", k.ChemicalFormula), ("$c", k.ColorHex),
                ("$p", k.ParentCategory), ("$e", k.IsEnabled ? 1 : 0), ("$r", k.IsReserved ? 1 : 0), ("$rv", k.RequiredVerification.ToString()));
        foreach (var i in project.Images)
        {
            Ins("INSERT INTO images VALUES($id,$f,$p,$w,$h,$s,$sp,$st,$n,$t)", ("$id", i.Id.ToString()), ("$f", i.FileName), ("$p", i.RelativePath),
                ("$w", i.Width), ("$h", i.Height), ("$s", i.Sha256), ("$sp", i.SpecimenId), ("$st", i.WorkflowState.ToString()), ("$n", i.Annotations.Count), ("$t", i.ImportedAt.ToString("O")));
            foreach (var a in i.Annotations)
            {
                Ins("INSERT INTO annotations VALUES($id,$img,$cls,$v,$vl,$area,$x,$y,$w,$h,$rev,$ca,$ma,$cb,$rb,$alg,$set,$notes)",
                    ("$id", a.Id.ToString()), ("$img", i.Id.ToString()), ("$cls", a.ClassId), ("$v", a.VerificationStatus.ToString()), ("$vl", (int)a.VerificationStatus),
                    ("$area", a.PixelArea), ("$x", a.BoundingBox.X), ("$y", a.BoundingBox.Y), ("$w", a.BoundingBox.Width), ("$h", a.BoundingBox.Height),
                    ("$rev", a.Revision), ("$ca", a.CreatedAt.ToString("O")), ("$ma", a.ModifiedAt.ToString("O")), ("$cb", a.CreatedBy), ("$rb", a.ReviewedBy),
                    ("$alg", a.SegmentationAlgorithm), ("$set", a.AnnotationSet), ("$notes", a.Notes));
                foreach (var r in a.RevisionHistory)
                    Ins("INSERT OR REPLACE INTO revisions VALUES($a,$r,$t,$au,$re,$c,$v,$p)", ("$a", a.Id.ToString()), ("$r", r.Revision), ("$t", r.Timestamp.ToString("O")),
                        ("$au", r.Author), ("$re", r.Reason), ("$c", r.ClassId), ("$v", r.VerificationStatus.ToString()), ("$p", r.Mask.PixelCount));
            }
        }
        foreach (var e in audit)
            Ins("INSERT INTO audit_log(timestamp,actor,action,image_id,annotation_id,details) VALUES($t,$a,$ac,$i,$an,$d)",
                ("$t", e.Timestamp.ToString("O")), ("$a", e.Actor), ("$ac", e.Action), ("$i", e.ImageId?.ToString()), ("$an", e.AnnotationId?.ToString()), ("$d", e.Details));
        tx.Commit();
    }

    public List<AnnotationIndexRow> Query(AnnotationQuery q)
    {
        using var c = Open();
        using var cmd = c.CreateCommand();
        var where = new List<string>();
        void P(string cond, string name, object v) { where.Add(cond); cmd.Parameters.AddWithValue(name, v); }
        if (q.ImageId is { } img) P("a.image_id = $img", "$img", img.ToString());
        if (q.ClassId is { } cls) P("a.class_id = $cls", "$cls", cls);
        if (q.Verification is { } v) P("a.verification_level = $v", "$v", (int)v);
        if (q.MaxVerification is { } mv) P("a.verification_level <= $mv", "$mv", (int)mv);
        if (q.Reviewer is { } r) P("a.reviewed_by = $r", "$r", r);
        if (q.CreatedBy is { } cb) P("a.created_by = $cb", "$cb", cb);
        if (q.ModifiedAfter is { } ma) P("a.modified_at >= $ma", "$ma", ma.ToString("O"));
        if (q.ModifiedBefore is { } mb) P("a.modified_at <= $mb", "$mb", mb.ToString("O"));
        if (q.WorkflowState is { } ws) P("i.workflow_state = $ws", "$ws", ws.ToString());
        if (q.AnnotationSet is { } set) P("a.annotation_set = $set", "$set", set);
        cmd.CommandText = """
            SELECT a.id, a.image_id, i.file_name, a.class_id, COALESCE(k.name, ''), a.verification, a.area, a.revision, a.modified_at,
                   a.created_by, a.reviewed_by, i.workflow_state, a.annotation_set
            FROM annotations a JOIN images i ON i.id = a.image_id LEFT JOIN classes k ON k.id = a.class_id
            """ + (where.Count > 0 ? " WHERE " + string.Join(" AND ", where) : "") + " ORDER BY i.file_name, a.created_at";
        var rows = new List<AnnotationIndexRow>();
        using var rd = cmd.ExecuteReader();
        while (rd.Read())
            rows.Add(new AnnotationIndexRow(Guid.Parse(rd.GetString(0)), Guid.Parse(rd.GetString(1)), rd.GetString(2), rd.GetInt32(3), rd.GetString(4),
                Enum.Parse<VerificationStatus>(rd.GetString(5)), rd.GetInt64(6), rd.GetInt32(7), DateTimeOffset.Parse(rd.GetString(8), System.Globalization.CultureInfo.InvariantCulture),
                rd.GetString(9), rd.IsDBNull(10) ? null : rd.GetString(10), Enum.Parse<ImageWorkflowState>(rd.GetString(11)), rd.GetString(12)));
        return rows;
    }

    public long Count(string table)
    {
        using var c = Open();
        using var cmd = c.CreateCommand();
        cmd.CommandText = $"SELECT COUNT(*) FROM {table switch { "images" or "annotations" or "revisions" or "audit_log" or "classes" => table, _ => throw new ArgumentException(table) }}";
        return (long)cmd.ExecuteScalar()!;
    }

    private static void Exec(SqliteConnection c, SqliteTransaction tx, string sql)
    {
        using var cmd = c.CreateCommand();
        cmd.Transaction = tx; cmd.CommandText = sql;
        cmd.ExecuteNonQuery();
    }
}
