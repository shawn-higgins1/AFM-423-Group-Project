	�N^�2@�N^�2@!�N^�2@	��\�V|�?��\�V|�?!��\�V|�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�N^�2@�����j1@1�b����?A?�ܥ?I����?Y�	��.�p?*	�����G@2U
Iterator::Model::ParallelMapV2�ZӼ��?!��dw�>@)�ZӼ��?1��dw�>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatg��j+��?!�R.a9@)��ׁsF�?1�em'�y5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��_vO�?!�`�em7@)�ZӼ�}?1��dw�.@:Preprocessing2F
Iterator::Model��ZӼ�?!W�hO5 F@)a��+ey?1���t��*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice���_vOn?!�=�S @)���_vOn?1�=�S @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipp_�Q�?!�����K@)-C��6j?1����@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!\�*�<@)��H�}]?1\�*�<@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��\�V|�?IC���-�X@Q�xF	R5�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�����j1@�����j1@!�����j1@      ��!       "	�b����?�b����?!�b����?*      ��!       2	?�ܥ??�ܥ?!?�ܥ?:	����?����?!����?B      ��!       J	�	��.�p?�	��.�p?!�	��.�p?R      ��!       Z	�	��.�p?�	��.�p?!�	��.�p?b      ��!       JGPUY��\�V|�?b qC���-�X@y�xF	R5�?