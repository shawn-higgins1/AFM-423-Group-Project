	l!�A	�2@l!�A	�2@!l!�A	�2@	��`�a�?��`�a�?!��`�a�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6l!�A	�2@+j0o1@1�y�Տ�?A0L�
F%�?I>�4a���?Y�+d���?*	�����?J@2U
Iterator::Model::ParallelMapV2F%u��?!J�$I�$9@)F%u��?1J�$I�$9@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap	�^)ː?!��<��<?@)�������?1�<��<�7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat �o_Ή?!      8@)M�O��?1��<��<3@:Preprocessing2F
Iterator::ModelM�O��?!��<��<C@)y�&1�|?1������*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�q����o?!n۶m۶@)�q����o?1n۶m۶@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipaTR'���?!1�0�N@)F%u�k?1J�$I�$@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{�G�zd?!�0�0@){�G�zd?1�0�0@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��`�a�?I8�d��X@Q9U�r��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	+j0o1@+j0o1@!+j0o1@      ��!       "	�y�Տ�?�y�Տ�?!�y�Տ�?*      ��!       2	0L�
F%�?0L�
F%�?!0L�
F%�?:	>�4a���?>�4a���?!>�4a���?B      ��!       J	�+d���?�+d���?!�+d���?R      ��!       Z	�+d���?�+d���?!�+d���?b      ��!       JGPUY��`�a�?b q8�d��X@y9U�r��?