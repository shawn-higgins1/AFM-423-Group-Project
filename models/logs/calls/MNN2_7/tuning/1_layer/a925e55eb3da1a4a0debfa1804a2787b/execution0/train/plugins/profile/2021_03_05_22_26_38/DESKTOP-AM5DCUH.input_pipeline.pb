	�dT��5@�dT��5@!�dT��5@	��!w��?��!w��?!��!w��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�dT��5@��0E��2@1��X���?A��ڊ�e�?I����=@Y?�̔�ߢ?*	�����YH@2U
Iterator::Model::ParallelMapV2�?�߾�?!=k���!<@)�?�߾�?1=k���!<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�������?!���ת9@)'�����?1�H��5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��@��ǈ?!U��/��8@)����Mb�?1�\0�Vm0@:Preprocessing2F
Iterator::Modeln���?!��!@�D@)��0�*x?1��-��:(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice	�^)�p?!�+K�x� @)	�^)�p?1�+K�x� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip\ A�c̝?!x[޿u�M@)-C��6j?1;ǳƊH@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!"@���@)��H�}]?1"@���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapS�!�uq�?!�4H�;@)��_�LU?1�r��Z@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 86.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��!w��?Ie���M�X@Q
�S�v�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��0E��2@��0E��2@!��0E��2@      ��!       "	��X���?��X���?!��X���?*      ��!       2	��ڊ�e�?��ڊ�e�?!��ڊ�e�?:	����=@����=@!����=@B      ��!       J	?�̔�ߢ??�̔�ߢ?!?�̔�ߢ?R      ��!       Z	?�̔�ߢ??�̔�ߢ?!?�̔�ߢ?b      ��!       JGPUY��!w��?b qe���M�X@y
�S�v�?