using CuAnnotator.Domain.Entities;
using CuAnnotator.Domain.Enums;

namespace CuAnnotator.Application.Commands;

public interface IUndoableCommand
{
    string Description { get; }
    void Execute();
    void Undo();
}

/// <summary>Linear undo/redo history. Executing a new command clears the redo branch.</summary>
public sealed class UndoRedoStack
{
    private readonly Stack<IUndoableCommand> _undo = new();
    private readonly Stack<IUndoableCommand> _redo = new();
    public int Capacity { get; init; } = 200;

    public event EventHandler? Changed;

    public bool CanUndo => _undo.Count > 0;
    public bool CanRedo => _redo.Count > 0;
    public string? UndoDescription => _undo.TryPeek(out var c) ? c.Description : null;
    public string? RedoDescription => _redo.TryPeek(out var c) ? c.Description : null;
    public int Count => _undo.Count;

    public void Execute(IUndoableCommand command)
    {
        command.Execute();
        _undo.Push(command);
        _redo.Clear();
        Trim();
        Changed?.Invoke(this, EventArgs.Empty);
    }

    public void Undo()
    {
        if (!_undo.TryPop(out var c)) return;
        c.Undo();
        _redo.Push(c);
        Changed?.Invoke(this, EventArgs.Empty);
    }

    public void Redo()
    {
        if (!_redo.TryPop(out var c)) return;
        c.Execute();
        _undo.Push(c);
        Changed?.Invoke(this, EventArgs.Empty);
    }

    public void Clear() { _undo.Clear(); _redo.Clear(); Changed?.Invoke(this, EventArgs.Empty); }

    private void Trim()
    {
        if (_undo.Count <= Capacity) return;
        var keep = _undo.Take(Capacity).Reverse().ToList();
        _undo.Clear();
        foreach (var c in keep) _undo.Push(c);
    }
}

/// <summary>
/// Snapshot command over the annotations of one image: captures the full annotation state before and
/// after an edit. Masks are immutable, so snapshots are cheap. Undo/redo restore state in place, keeping
/// the identity of existing annotation objects (UI bindings stay valid).
/// </summary>
public sealed class ImageSnapshotCommand : IUndoableCommand
{
    private readonly ImageItem _image;
    private readonly Action _mutation;
    private Snapshot? _before, _after;

    public string Description { get; }

    public ImageSnapshotCommand(ImageItem image, string description, Action mutation)
    {
        _image = image; Description = description; _mutation = mutation;
    }

    public void Execute()
    {
        if (_after is null)
        {
            _before = Snapshot.Take(_image);
            try { _mutation(); }
            catch
            {
                _before.Restore(_image); // never leave a half-applied edit behind
                throw;
            }
            _after = Snapshot.Take(_image);
        }
        else _after.Restore(_image);
    }

    public void Undo() => _before?.Restore(_image);

    /// <summary>Annotation state plus the original object references, so restoring never replaces an instance.</summary>
    private sealed record Snapshot(List<(Domain.Entities.Annotation Instance, Domain.Entities.Annotation State)> Annotations, ImageWorkflowState State)
    {
        public static Snapshot Take(ImageItem img) => new(img.Annotations.Select(a => (a, a.DeepClone())).ToList(), img.WorkflowState);

        public void Restore(ImageItem img)
        {
            var restored = new List<Domain.Entities.Annotation>(Annotations.Count);
            foreach (var (instance, state) in Annotations)
            {
                instance.RestoreFrom(state);
                restored.Add(instance);
            }
            img.Annotations.Clear();
            img.Annotations.AddRange(restored);
            img.WorkflowState = State;
        }
    }
}

/// <summary>Generic command from execute/undo delegates (for state outside the project model).</summary>
public sealed class DelegateUndoableCommand(string description, Action execute, Action undo) : IUndoableCommand
{
    public string Description { get; } = description;
    public void Execute() => execute();
    public void Undo() => undo();
}

/// <summary>Several commands executed and undone as one step.</summary>
public sealed class CompositeCommand(string description, IReadOnlyList<IUndoableCommand> commands) : IUndoableCommand
{
    public string Description { get; } = description;
    public void Execute() { foreach (var c in commands) c.Execute(); }
    public void Undo() { foreach (var c in commands.Reverse()) c.Undo(); }
}
