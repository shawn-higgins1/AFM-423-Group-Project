	�0�F7@�0�F7@!�0�F7@	���R0�?���R0�?!���R0�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�0�F7@�2o�u�4@1F��_��?A����K�?IB����@Y�a1�Z{o?*	�����YI@2U
Iterator::Model::ParallelMapV2�(��0�?!�k���B8@)�(��0�?1�k���B8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatlxz�,C�?!Y�K8;@)g��j+��?1;�;�7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate�(��0�?!�k���B8@)"��u���?1?
h�0@:Preprocessing2F
Iterator::Model�:pΈ�?!�d(��A@)�����w?1ؼ~�2�&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���{�?!����/P@)�g��s�u?1�0��D�$@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice���_vOn?!7Ir�0@)���_vOn?17Ir�0@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!z0��k�@)�J�4a?1z0��k�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���S㥋?!1��k��:@)a2U0*�S?1��WV�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9���R0�?I��*(��X@Q-�3��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�2o�u�4@�2o�u�4@!�2o�u�4@      ��!       "	F��_��?F��_��?!F��_��?*      ��!       2	����K�?����K�?!����K�?:	B����@B����@!B����@B      ��!       J	�a1�Z{o?�a1�Z{o?!�a1�Z{o?R      ��!       Z	�a1�Z{o?�a1�Z{o?!�a1�Z{o?b      ��!       JGPUY���R0�?b q��*(��X@y-�3��?