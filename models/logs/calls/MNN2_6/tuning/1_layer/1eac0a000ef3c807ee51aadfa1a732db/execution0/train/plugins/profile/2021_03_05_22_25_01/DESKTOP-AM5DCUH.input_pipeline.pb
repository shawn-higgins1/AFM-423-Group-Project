	���$��1@���$��1@!���$��1@	/x��4�?/x��4�?!/x��4�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6���$��1@���х0@1��R�1��?A�yT���?I@��T��?Yh�K6l�?*	gffff�J@2U
Iterator::Model::ParallelMapV2��H�}�?!��k�:@)��H�}�?1��k�:@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�Q���?!�g��C@@)S�!�uq�?1ɡ.K5�8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeattF��_�?!Q��t6@)n���?1C"Lp272@:Preprocessing2F
Iterator::Model��_�L�?!ٌ�TC@)-C��6z?127^Ѵ�'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice	�^)�p?!��@��{@)	�^)�p?1��@��{@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipL7�A`�?!'s�M�N@)��H�}m?1��k�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!r��R:@)�J�4a?1r��R:@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 91.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9/x��4�?IsZ����X@Qw���^ �?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���х0@���х0@!���х0@      ��!       "	��R�1��?��R�1��?!��R�1��?*      ��!       2	�yT���?�yT���?!�yT���?:	@��T��?@��T��?!@��T��?B      ��!       J	h�K6l�?h�K6l�?!h�K6l�?R      ��!       Z	h�K6l�?h�K6l�?!h�K6l�?b      ��!       JGPUY/x��4�?b qsZ����X@yw���^ �?