	/��[<�3@/��[<�3@!/��[<�3@	��R:���?��R:���?!��R:���?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6/��[<�3@%�S;��1@1O[#�qp�?A���JY��?I�h㈵��?Y�lɪ7i?*	43333sH@2U
Iterator::Model::ParallelMapV2�J�4�?!��uǋ-A@)�J�4�?1��uǋ-A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat������?!*08͸7@)�j+��݃?1ݣ/]�3@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapǺ����?!xc�	e�6@)ŏ1w-!?1lb��v/@:Preprocessing2F
Iterator::Modelg��j+��?!W��C'�G@)F%u�{?1�bK�m�*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice��H�}m?!ɀz�r@)��H�}m?1ɀz�r@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�&1��?!�\w��J@)��_vOf?1Ɩ���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!lb��v@)ŏ1w-!_?1lb��v@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 90.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�7.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��R:���?I��/_�X@Q���
8h�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	%�S;��1@%�S;��1@!%�S;��1@      ��!       "	O[#�qp�?O[#�qp�?!O[#�qp�?*      ��!       2	���JY��?���JY��?!���JY��?:	�h㈵��?�h㈵��?!�h㈵��?B      ��!       J	�lɪ7i?�lɪ7i?!�lɪ7i?R      ��!       Z	�lɪ7i?�lɪ7i?!�lɪ7i?b      ��!       JGPUY��R:���?b q��/_�X@y���
8h�?