# Fixed external-holdout visualization validation

## Scope

This was a visualization-only rebuild from the saved native-space MRI volumes, expert masks, query-decoder predictions, and CSV metrics in `outputs/external_holdout/`. No model code or checkpoint was loaded; no training or inference was run; predictions and metrics were not modified. The earlier package remains untouched in `outputs/external_holdout/visualizations/`.

## Why the previous figures were unreadable

Programmatic inspection of the previous PNGs found a very low foreground-content fraction in several chart families: approximately 5.3% for the sparsest per-slice chart and 3.4% for the sparsest dataset chart. The custom raster chart implementation also lacked reliable axis objects, so titles, metric units, ticks, legends, and grid lines were incomplete or absent. Dice curves were visually compressed, large unused regions dominated some figures, and raster text placement could collide or clip.

The first manual review of the replacement package also caught two presentation defects before completion: adjacent montage headers touched, and two scatter-plot legends rendered unreliably. The montage was widened and its longest header split over two lines; scatter legends were rebuilt from explicit legend handles and moved to a safe lower-left position. The affected figures were regenerated and reopened.

## Validation performed

- Revalidated all 91 MRI/expert/prediction triplets for identical array shape, physical size, spacing, origin, direction, and affine-derived orientation.
- Recomputed subject Dice, IoU, precision, recall, FP, and FN from the saved native masks and matched the saved metrics within `1e-6`.
- Recomputed every per-slice Dice, IoU, FP, and FN value and matched the saved CSV rows within `1e-6`.
- Rendered every new chart with Matplotlib at 180 DPI using explicit figure sizes, titles, labels, ticks, legends, grids, and constrained layout.
- Reopened every saved PNG and checked dimensions, standard deviation, non-white pixel fraction, and border-ink fraction.
- Programmatically validated 159 regenerated figures. No figure was blank or mostly blank, and no automated border-clipping failure remained.
- Manually inspected two best overviews (sub-122 and sub-101), two worst overviews (sub-108 and sub-075), three per-slice charts (sub-108, sub-122, and sub-095), three hard-slice figures (sub-044 slice 3, sub-045 slice 54, and sub-048 slice 57), the hard-slice contact sheet, and all seven dataset-level charts.
- The manual sample confirmed visible titles and axis labels, readable text, aligned masks, correct legend colors, visible MRI anatomy, and nonblank content. Apparent truncation at some terminal 64-slice masks corresponds to anatomy reaching the native image boundary, not plotting crop loss.

Machine-readable image checks and the representative sample manifest are saved in [visualization_quality_checks.json](visualization_quality_checks.json).

## Output count

- 20 subject overview montages: best 5, median 5, and worst 10
- 91 four-panel per-slice metric charts
- 40 unique hard-slice figures selected from the lowest 20 Dice, largest 10 FN, and largest 10 FP lists after deduplication
- 1 twelve-case hard-slice contact sheet
- 7 fully labeled dataset-level charts
- Total: **159 figures**

No figures remained blank. No labels remained visibly clipped in the completed manual review.

## Required example paths

- Best subject overview: [sub-122 overview](subject_overviews/sub-122_overview.png)
- Worst subject overview: [sub-108 overview](subject_overviews/sub-108_overview.png)
- Worst hard-slice example: [sub-044 slice 3](hard_slices/sub-044_slice-003.png)
- Per-slice chart: [sub-108 per-slice metrics](per_slice_charts/sub-108_per_slice_metrics.png)
- Dice histogram: [Dice histogram](dataset_charts/dice_histogram.png)
- Sorted Dice chart: [sorted Dice by subject](dataset_charts/sorted_dice_by_subject.png)
- Hard-slice contact sheet: [top-12 contact sheet](hard_slices/hard_slice_contact_sheet_top12.png)

Exact plotted dataset-level values are saved at [dataset_metrics_plotted.csv](plotted_data/dataset_metrics_plotted.csv).
