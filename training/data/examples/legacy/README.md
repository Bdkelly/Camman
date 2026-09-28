# Preserved annotation data

These files were moved byte-for-byte from `infer/edited_json_output/` so unique
annotations are not discarded during repository cleanup. Original filenames
and directory relationships are retained. Embedded paths are historical and
must be mapped to your local images before use; these files are not loaded by
the application or packaged in the Python wheel.

Removed only:

- `combined_json_output/combined_video_sequence_final.json`: byte-identical to
  `combined_json_output/combined_video_sequence.json`, which is retained.
- `combined_json_output/new_dummy_video_data.json`: placeholder synthetic data.

Some combined files contain overlapping frame names across different videos.
Keep source videos separate when constructing a training dataset. New datasets
and generated annotations should be stored outside this historical directory.
