	4ڪ$�/5@4ڪ$�/5@!4ڪ$�/5@	��A�+3�?��A�+3�?!��A�+3�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails64ڪ$�/5@�ܘ���2@1g��I}Y�?A
ףp=
�?IQj/���?Ya2U0*�s?*	����̌G@2U
Iterator::Model::ParallelMapV2_�Qڋ?!�r���<@)_�Qڋ?1�r���<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��0�*�?!Eg��9@)��ׁsF�?1�ƾG�5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�I+��?!%�~��Z7@)���Q�~?1��	���/@:Preprocessing2F
Iterator::Model��ׁsF�?!�ƾG�E@)a��+ey?1���S*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey�&1�l?!u��W�@)y�&1�l?1u��W�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�#�����?!9A��L@)-C��6j?1"��-@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!d���"@)ŏ1w-!_?1d���"@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�(��0�?!��x�Y:@)��_�LU?1|+�g�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��A�+3�?Ir�Y]�X@Q��ѹ�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�ܘ���2@�ܘ���2@!�ܘ���2@      ��!       "	g��I}Y�?g��I}Y�?!g��I}Y�?*      ��!       2	
ףp=
�?
ףp=
�?!
ףp=
�?:	Qj/���?Qj/���?!Qj/���?B      ��!       J	a2U0*�s?a2U0*�s?!a2U0*�s?R      ��!       Z	a2U0*�s?a2U0*�s?!a2U0*�s?b      ��!       JGPUY��A�+3�?b qr�Y]�X@y��ѹ�?