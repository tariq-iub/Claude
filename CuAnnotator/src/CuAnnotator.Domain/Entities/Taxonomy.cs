using CuAnnotator.Domain.Enums;

namespace CuAnnotator.Domain.Entities;

/// <summary>A labelled class. <see cref="Id"/> is the value written into 8-bit semantic masks.</summary>
public sealed class CorrosionClass
{
    public int Id { get; set; }
    public string Name { get; set; } = "";
    public string? ChemicalFormula { get; set; }
    public string Description { get; set; } = "";
    /// <summary>Display colour as #RRGGBB. Display only — never exported as a label.</summary>
    public string ColorHex { get; set; } = "#808080";
    /// <summary>Single keyboard key (e.g. "2", "I"), or null.</summary>
    public string? Shortcut { get; set; }
    public bool IsEnabled { get; set; } = true;
    /// <summary>Slash-separated hierarchy path, e.g. "Corroded/Oxides".</summary>
    public string? ParentCategory { get; set; }
    /// <summary>Minimum verification status the project expects before this class counts as verified.</summary>
    public VerificationStatus RequiredVerification { get; set; } = VerificationStatus.ExpertReviewed;
    /// <summary>Built-in IDs (0 Unlabeled, 255 Ignore) cannot be edited or reassigned.</summary>
    public bool IsReserved { get; set; }

    public CorrosionClass Clone() => (CorrosionClass)MemberwiseClone();
    public override string ToString() => $"{Id}: {Name}";
}

public sealed class Taxonomy
{
    public const int UnlabeledId = 0;
    public const int IgnoreId = 255;

    public string Name { get; set; } = "Default copper corrosion taxonomy";
    public int Version { get; set; } = 1;
    public List<CorrosionClass> Classes { get; set; } = new();

    public CorrosionClass? Find(int id) => Classes.FirstOrDefault(c => c.Id == id);
    public CorrosionClass Get(int id) => Find(id) ?? throw new KeyNotFoundException($"Class ID {id} is not defined in the taxonomy.");

    /// <summary>Classes that may be assigned to annotations (enabled and not the Unlabeled class).</summary>
    public IEnumerable<CorrosionClass> AssignableClasses => Classes.Where(c => c.IsEnabled && c.Id != UnlabeledId).OrderBy(c => c.Id);

    /// <summary>
    /// Default taxonomy. IDs 1–5 follow the original five-class copper-corrosion dataset ordering
    /// (Healthy, Cu2O, CuO, CuCl, CuCl2); 6–8 extend it without renumbering.
    /// </summary>
    public static Taxonomy CreateDefault() => new()
    {
        Classes =
        {
            new() { Id = 0, Name = "Unlabeled", Description = "Pixels not yet annotated", ColorHex = "#000000", IsReserved = true, RequiredVerification = VerificationStatus.Unverified },
            new() { Id = 1, Name = "Healthy Copper", Description = "Pristine or visually unaffected copper", ColorHex = "#E0A060", Shortcut = "1", ParentCategory = "Copper Surface/Healthy", RequiredVerification = VerificationStatus.VisuallyEstimated },
            new() { Id = 2, Name = "Cu2O", ChemicalFormula = "Cu₂O", Description = "Cuprous oxide (cuprite)", ColorHex = "#D03030", Shortcut = "2", ParentCategory = "Copper Surface/Corroded/Oxides" },
            new() { Id = 3, Name = "CuO", ChemicalFormula = "CuO", Description = "Cupric oxide (tenorite)", ColorHex = "#303030", Shortcut = "3", ParentCategory = "Copper Surface/Corroded/Oxides" },
            new() { Id = 4, Name = "CuCl", ChemicalFormula = "CuCl", Description = "Cuprous chloride (nantokite)", ColorHex = "#9AD0C0", Shortcut = "4", ParentCategory = "Copper Surface/Corroded/Chlorides", RequiredVerification = VerificationStatus.InstrumentallyVerified },
            new() { Id = 5, Name = "CuCl2", ChemicalFormula = "CuCl₂", Description = "Copper(II) chloride", ColorHex = "#2080D0", Shortcut = "5", ParentCategory = "Copper Surface/Corroded/Chlorides", RequiredVerification = VerificationStatus.InstrumentallyVerified },
            new() { Id = 6, Name = "Malachite", ChemicalFormula = "Cu₂CO₃(OH)₂", Description = "Basic copper carbonate", ColorHex = "#20A050", Shortcut = "6", ParentCategory = "Copper Surface/Corroded/Carbonates" },
            new() { Id = 7, Name = "Brochantite", ChemicalFormula = "Cu₄SO₄(OH)₆", Description = "Basic copper sulfate (not a carbonate)", ColorHex = "#40C0A0", Shortcut = "7", ParentCategory = "Copper Surface/Corroded/Sulfates", RequiredVerification = VerificationStatus.InstrumentallyVerified },
            new() { Id = 8, Name = "Basic Copper Chlorides", ChemicalFormula = "Cu₂(OH)₃Cl", Description = "Atacamite / paratacamite / clinoatacamite group (associated with bronze disease)", ColorHex = "#80E040", Shortcut = "8", ParentCategory = "Copper Surface/Corroded/Chlorides", RequiredVerification = VerificationStatus.InstrumentallyVerified },
            new() { Id = 255, Name = "Ignore", Description = "Explicitly excluded from annotation or evaluation", ColorHex = "#FF00FF", Shortcut = "I", IsReserved = true, RequiredVerification = VerificationStatus.Unverified },
        },
    };

    public Taxonomy Clone() => new() { Name = Name, Version = Version, Classes = Classes.Select(c => c.Clone()).ToList() };
}
