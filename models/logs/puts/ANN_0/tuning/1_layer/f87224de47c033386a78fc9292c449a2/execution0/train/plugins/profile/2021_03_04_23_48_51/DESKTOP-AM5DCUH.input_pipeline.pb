	�a���2@�a���2@!�a���2@	r�7A��?r�7A��?!r�7A��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�a���2@~7ݲC1@1l|&��i�?AA��ǘ��?IV�pA6�?YY�+���n?*	33333sI@2U
Iterator::Model::ParallelMapV2�q����?!<(p���>@)�q����?1<(p���>@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��H�}�?!�8{�oJ<@)�g��s��?1d�H�<�4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatZd;�O��?!����6@)��~j�t�?1��}���2@:Preprocessing2F
Iterator::Model�I+��?!���j�E@)-C��6z?1�Nߔ�%)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceŏ1w-!o?!�-�0��@)ŏ1w-!o?1�-�0��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��j+���?!$`�cL@)�~j�t�h?1Z�]�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!�"�&o@)����Mb`?1�"�&o@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9r�7A��?I��_M��X@Q��"����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	~7ݲC1@~7ݲC1@!~7ݲC1@      ��!       "	l|&��i�?l|&��i�?!l|&��i�?*      ��!       2	A��ǘ��?A��ǘ��?!A��ǘ��?:	V�pA6�?V�pA6�?!V�pA6�?B      ��!       J	Y�+���n?Y�+���n?!Y�+���n?R      ��!       Z	Y�+���n?Y�+���n?!Y�+���n?b      ��!       JGPUYr�7A��?b q��_M��X@y��"����?