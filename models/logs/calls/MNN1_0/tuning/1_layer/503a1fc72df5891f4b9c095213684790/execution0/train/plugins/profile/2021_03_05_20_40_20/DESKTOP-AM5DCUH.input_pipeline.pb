	`2�C�1@`2�C�1@!`2�C�1@	t6mR��?t6mR��?!t6mR��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6`2�C�1@d=��j0@1��^~���?AZd;�O��?I�6U����?Y��0�*x?*	������K@2U
Iterator::Model::ParallelMapV2�:pΈ�?!�g��e@@)�:pΈ�?1�g��e@@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap2U0*��?!?���(q<@)��0�*�?1�ځ�v`5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatǺ����?!��(�3J4@)�&S��?1�(�3J|0@:Preprocessing2F
Iterator::Modela��+e�?!�ځ�vF@)S�!�uq{?1��%~F(@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�q����o?!d!Y�B@)�q����o?1d!Y�B@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipŏ1w-!�?!�%~F�K@)y�&1�l?1�s��\@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!�7��Mo@)�J�4a?1�7��Mo@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9u6mR��?I�D��Q�X@QԳ���F�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	d=��j0@d=��j0@!d=��j0@      ��!       "	��^~���?��^~���?!��^~���?*      ��!       2	Zd;�O��?Zd;�O��?!Zd;�O��?:	�6U����?�6U����?!�6U����?B      ��!       J	��0�*x?��0�*x?!��0�*x?R      ��!       Z	��0�*x?��0�*x?!��0�*x?b      ��!       JGPUYu6mR��?b q�D��Q�X@yԳ���F�?