	H5��Ķ5@H5��Ķ5@!H5��Ķ5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-H5��Ķ5@~͑�3@1�`U��N�?A$����ۧ?Ic}�E@*	43333sJ@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate+�����?!X��#VnB@)��y�):�?1�W���@@:Preprocessing2U
Iterator::Model::ParallelMapV2K�=�U�?!��x�(�<@)K�=�U�?1��x�(�<@:Preprocessing2F
Iterator::ModelA��ǘ��?!��R��D@)lxz�,C|?1!Y�B*@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�ZӼ�}?!#)�i��*@)��ZӼ�t?1��H#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip����o�?!e?��ZM@)�~j�t�h?1���.�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!'�TA�>@)����Mb`?1'�TA�>@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�0�*�?!���xC@)/n��R?1��㙢 @:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor/n��R?!��㙢 @)/n��R?1��㙢 @:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicea2U0*�C?!�2'�%�?)a2U0*�C?1�2'�%�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�W��X@Q+��p�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	~͑�3@~͑�3@!~͑�3@      ��!       "	�`U��N�?�`U��N�?!�`U��N�?*      ��!       2	$����ۧ?$����ۧ?!$����ۧ?:	c}�E@c}�E@!c}�E@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�W��X@y+��p�?