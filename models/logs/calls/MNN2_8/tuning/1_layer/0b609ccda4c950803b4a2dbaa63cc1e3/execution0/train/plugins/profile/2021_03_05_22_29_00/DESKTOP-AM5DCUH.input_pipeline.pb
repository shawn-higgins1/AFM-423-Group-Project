	�2Q��U5@�2Q��U5@!�2Q��U5@	��'ܵ�?��'ܵ�?!��'ܵ�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�2Q��U5@S�h�3@1V*����?A��3���?I����G) @YC�O�}:n?*	fffff&L@2U
Iterator::Model::ParallelMapV2X�5�;N�?!�V��>@)X�5�;N�?1�V��>@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateK�=�U�?!�-;@)�+e�X�?1�>��?4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�]K�=�?!35I%��7@)46<�R�?1oɧ5N\3@:Preprocessing2F
Iterator::Model��0�*�?!�Q����D@)S�!�uq{?1�����'@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�q����o?!z<��m�@)�q����o?1z<��m�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�(���?!(�{Pu
M@)�����g?1ވᰙ�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*�c?!���/@)a2U0*�c?1���/@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapL7�A`�?!��u�N=@)a2U0*�S?1���/@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��'ܵ�?I8��ɢ�X@Q���v��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	S�h�3@S�h�3@!S�h�3@      ��!       "	V*����?V*����?!V*����?*      ��!       2	��3���?��3���?!��3���?:	����G) @����G) @!����G) @B      ��!       J	C�O�}:n?C�O�}:n?!C�O�}:n?R      ��!       Z	C�O�}:n?C�O�}:n?!C�O�}:n?b      ��!       JGPUY��'ܵ�?b q8��ɢ�X@y���v��?