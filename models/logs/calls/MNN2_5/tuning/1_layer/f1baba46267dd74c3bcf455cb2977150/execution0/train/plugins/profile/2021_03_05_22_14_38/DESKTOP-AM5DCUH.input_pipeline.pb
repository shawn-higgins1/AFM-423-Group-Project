	���d�2@���d�2@!���d�2@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-���d�2@G��>1@1ްmQf��?A䃞ͪϥ?I�����?*     �G@)       =2U
Iterator::Model::ParallelMapV2���QI�?!R�٨�l>@)���QI�?1R�٨�l>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�������?!r1���:@)��ׁsF�?1L� &W5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�I+��?!���F}g7@)�<,Ԛ�}?1L� &W/@:Preprocessing2F
Iterator::Model��ZӼ�?!G}g���E@)�HP�x?1w6�;�)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice���_vOn?!����F}@)���_vOn?1����F}@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�]K�=�?!���\ALL@)a��+ei?1���
b@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��_�Le?!�\AL� @)��_�Le?1�\AL� @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�5.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI
��#�X@Q���7�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	G��>1@G��>1@!G��>1@      ��!       "	ްmQf��?ްmQf��?!ްmQf��?*      ��!       2	䃞ͪϥ?䃞ͪϥ?!䃞ͪϥ?:	�����?�����?!�����?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q
��#�X@y���7�?