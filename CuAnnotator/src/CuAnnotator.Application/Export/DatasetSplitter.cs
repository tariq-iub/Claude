using CuAnnotator.Domain.Entities;

namespace CuAnnotator.Application.Export;

public enum DatasetSplit { Train, Validation, Test }

/// <summary>
/// Reproducible train/validation/test split. By default images are grouped by specimen
/// (<see cref="ImageItem.SpecimenKey"/>) and whole groups are assigned, so patches of one physical
/// specimen never appear in two splits (no specimen-level leakage). Groups are shuffled with a seeded
/// Fisher–Yates permutation of their sorted keys, then assigned greedily to the split furthest below
/// its target image count.
/// </summary>
public static class DatasetSplitter
{
    public static Dictionary<Guid, DatasetSplit> Split(IReadOnlyList<ImageItem> images, double train, double validation, int seed, bool groupBySpecimen = true,
        IReadOnlyDictionary<Guid, DatasetSplit>? manual = null)
    {
        if (train < 0 || validation < 0 || train + validation > 1 + 1e-9) throw new ArgumentException("Invalid split fractions.");
        var result = new Dictionary<Guid, DatasetSplit>();
        var remaining = images.Where(i => manual is null || !manual.ContainsKey(i.Id)).ToList();
        if (manual is not null) foreach (var kv in manual) result[kv.Key] = kv.Value;

        var groups = remaining.GroupBy(i => groupBySpecimen ? i.SpecimenKey : $"image:{i.Id}")
                              .OrderBy(g => g.Key, StringComparer.Ordinal).Select(g => g.ToList()).ToList();
        var rnd = new Random(seed);
        for (int i = groups.Count - 1; i > 0; i--)
        {
            int j = rnd.Next(i + 1);
            (groups[i], groups[j]) = (groups[j], groups[i]);
        }
        double total = remaining.Count;
        var targets = new Dictionary<DatasetSplit, double>
        {
            [DatasetSplit.Train] = train * total,
            [DatasetSplit.Validation] = validation * total,
            [DatasetSplit.Test] = (1 - train - validation) * total,
        };
        var counts = targets.Keys.ToDictionary(k => k, _ => 0.0);
        foreach (var g in groups)
        {
            var split = targets.Keys.OrderByDescending(k => targets[k] - counts[k]).ThenBy(k => k).First();
            counts[split] += g.Count;
            foreach (var img in g) result[img.Id] = split;
        }
        return result;
    }

    /// <summary>Specimens appearing in more than one split (should be empty when grouping is enabled).</summary>
    public static IReadOnlyList<string> LeakingSpecimens(IReadOnlyList<ImageItem> images, IReadOnlyDictionary<Guid, DatasetSplit> split) =>
        images.Where(i => !string.IsNullOrWhiteSpace(i.SpecimenId)).GroupBy(i => i.SpecimenId!)
              .Where(g => g.Select(i => split[i.Id]).Distinct().Count() > 1).Select(g => g.Key).ToList();
}
