import tensorflow as tf

tfrecord_file = '/home/eerodriguez3/elp/tfrecords_cherrypicked/train_cherrypick.tfrecord'

raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
for raw_record in raw_dataset.take(1):
    print(raw_record)

