	�ϷK�1@�ϷK�1@!�ϷK�1@	����o�?����o�?!����o�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�ϷK�1@2����[0@1?�W��"�?A#K�Xޥ?I�]����?Yy�ՏM�?*	43333sH@2U
Iterator::Model::ParallelMapV2�Pk�w�?!)��@�l<@)�Pk�w�?1)��@�l<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��H�}�?!ɀz�r=@)g��j+��?1W��C'�7@:Preprocessing2F
Iterator::Model��_�L�?!�έ�DE@)lxz�,C|?1�f5�8,@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�0�*�?!�VJ�:5@)9��v��z?1�|Bٹ�*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceŏ1w-!o?!lb��v@)ŏ1w-!o?1lb��v@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�?Ɯ?!�51Rk�L@)_�Q�k?1�/]��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��_vOf?!Ɩ���@)��_vOf?1Ɩ���@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 91.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9����o�?I�]%�X@Q�^����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	2����[0@2����[0@!2����[0@      ��!       "	?�W��"�??�W��"�?!?�W��"�?*      ��!       2	#K�Xޥ?#K�Xޥ?!#K�Xޥ?:	�]����?�]����?!�]����?B      ��!       J	y�ՏM�?y�ՏM�?!y�ՏM�?R      ��!       Z	y�ՏM�?y�ՏM�?!y�ՏM�?b      ��!       JGPUY����o�?b q�]%�X@y�^����?