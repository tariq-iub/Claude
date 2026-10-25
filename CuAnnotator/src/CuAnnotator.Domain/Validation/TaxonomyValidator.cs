using System.Text.RegularExpressions;
using CuAnnotator.Domain.Entities;

namespace CuAnnotator.Domain.Validation;

public static partial class TaxonomyValidator
{
    [GeneratedRegex("^#[0-9A-Fa-f]{6}$")]
    private static partial Regex ColorRegex();

    /// <summary>Validates the taxonomy; <paramref name="usedIds"/> are IDs referenced by existing annotations.</summary>
    public static IReadOnlyList<string> Validate(Taxonomy taxonomy, IEnumerable<int>? usedIds = null)
    {
        var errors = new List<string>();
        foreach (var g in taxonomy.Classes.GroupBy(c => c.Id).Where(g => g.Count() > 1))
            errors.Add($"Class ID {g.Key} is used by more than one class ({string.Join(", ", g.Select(c => c.Name))}).");
        foreach (var c in taxonomy.Classes)
        {
            if (c.Id is < 0 or > 255) errors.Add($"Class '{c.Name}' has ID {c.Id}; 8-bit masks require 0–255.");
            if (string.IsNullOrWhiteSpace(c.Name)) errors.Add($"Class ID {c.Id} has no name.");
            if (!ColorRegex().IsMatch(c.ColorHex)) errors.Add($"Class '{c.Name}' colour '{c.ColorHex}' is not #RRGGBB.");
            if (!c.IsReserved && c.Id is Taxonomy.UnlabeledId or Taxonomy.IgnoreId)
                errors.Add($"ID {c.Id} is reserved and cannot be used by '{c.Name}'.");
        }
        if (taxonomy.Find(Taxonomy.UnlabeledId) is null) errors.Add("Reserved class 0 (Unlabeled) is missing.");
        if (taxonomy.Find(Taxonomy.IgnoreId) is null) errors.Add("Reserved class 255 (Ignore) is missing.");
        foreach (var g in taxonomy.Classes.Where(c => !string.IsNullOrEmpty(c.Shortcut))
                     .GroupBy(c => c.Shortcut!.ToUpperInvariant()).Where(g => g.Count() > 1))
            errors.Add($"Shortcut '{g.Key}' is assigned to several classes.");
        if (usedIds is not null)
            foreach (var id in usedIds.Distinct())
                if (taxonomy.Find(id) is null)
                    errors.Add($"Class ID {id} is used by existing annotations but is missing from the taxonomy.");
        return errors;
    }

    /// <summary>
    /// Checks that an edited taxonomy does not renumber or delete IDs already referenced by annotations.
    /// Renaming, recolouring and disabling are allowed; the numeric identity is permanent.
    /// </summary>
    public static IReadOnlyList<string> ValidateEdit(Taxonomy before, Taxonomy after, IEnumerable<int> usedIds)
    {
        var errors = Validate(after, usedIds).ToList();
        foreach (var r in before.Classes.Where(c => c.IsReserved))
        {
            var a = after.Find(r.Id);
            if (a is null || !a.IsReserved) errors.Add($"Reserved class {r.Id} ({r.Name}) cannot be removed or modified.");
        }
        return errors;
    }
}
