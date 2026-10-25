# Data layout

```
data/
  raw/settings.xlsx, raw/labels.xlsx    # supplied workbooks (committed; contain no images)
  images/                               # PRIVATE, git-ignored: original copper images
  annotations/
    splits.csv                          # image_id,group_id,split  (group = specimen)
    adjudicated/<image_id>.png          # locked reference masks (0-5, 255 ignored)
    store/                              # AnnotationStore (append-only revisions, audit log)
```
Never commit images or expert masks. Run `git check-ignore -v data/images/x.png` to confirm they are ignored.
