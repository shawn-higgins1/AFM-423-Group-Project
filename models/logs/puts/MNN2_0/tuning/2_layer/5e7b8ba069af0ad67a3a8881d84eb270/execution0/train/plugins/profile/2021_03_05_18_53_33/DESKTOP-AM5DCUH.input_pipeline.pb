	>�D�7@>�D�7@!>�D�7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails->�D�7@2��Y�4@1^h��HK�?A��&��?I�߼8��@*	������J@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate
ףp=
�?!VUUUU�D@)��ZӼ�?19��8��B@:Preprocessing2U
Iterator::Model::ParallelMapV2?W[���?!h/����;@)?W[���?1h/����;@:Preprocessing2F
Iterator::ModelǺ���?!�Kh/��D@)�<,Ԛ�}?1��^B{	+@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��_�Lu?!&���^B#@)_�Q�k?1��Kh/@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipvq�-�?!%���^BM@)Ǻ���f?1�Kh/��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!������
@)��H�}]?1������
@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensora2U0*�S?!r�q�@)a2U0*�S?1r�q�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��0�*�?!�����E@)/n��R?1����K @:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice��H�}M?!�������?)��H�}M?1�������?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI3�GJC�X@Q;�n-��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	2��Y�4@2��Y�4@!2��Y�4@      ��!       "	^h��HK�?^h��HK�?!^h��HK�?*      ��!       2	��&��?��&��?!��&��?:	�߼8��@�߼8��@!�߼8��@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q3�GJC�X@y;�n-��?