python build_mscoco_data.py \
  --train_image_dir="/dataset/mscoco/raw-data/train2014/" \
  --val_image_dir="/dataset/mscoco/raw-data/val2014/" \
  --train_captions_file="/dataset/mscoco/raw-data/annotations/captions_train2014.json" \
  --val_captions_file="/dataset/mscoco/raw-data/annotations/captions_val2014.json" \
  --train_instances_file="/dataset/mscoco/raw-data/annotations/instances_train2014.json" \
  --val_instances_file="/dataset/mscoco/raw-data/annotations/instances_val2014.json" \
  --output_dir="/dataset/mscoco/tfrecord-bbox/" \
  --word_counts_output_file="/dataset/mscoco/tfrecord-bbox/word_counts.txt" \
