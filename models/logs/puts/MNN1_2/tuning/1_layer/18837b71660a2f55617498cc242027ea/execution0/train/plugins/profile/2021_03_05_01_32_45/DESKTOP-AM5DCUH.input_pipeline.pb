	��g�P@��g�P@!��g�P@	�I�5�?�I�5�?!�I�5�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6��g�P@]QJV�O@1�Q�GT�?A��ڊ�e�?I�z6��@YxԘsIu?*	������L@2U
Iterator::Model::ParallelMapV2��ǘ���?!�p��YR<@)��ǘ���?1�p��YR<@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate	�^)ː?!J9 2ܫ<@)��0�*�?1�5�n�4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat_�Qڋ?!=B����7@)g��j+��?1u�E]t4@:Preprocessing2F
Iterator::Model�+e�X�?!���C@)F%u�{?1;�;�'@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceHP�s�r?!)D�{ @)HP�s�r?1)D�{ @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip"��u���?!L[���N@)-C��6j?19 2ܫ`@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!D�{̒
@)ŏ1w-!_?1D�{̒
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap/n���?!O��N��>@)a2U0*�S?1+�%�� @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 95.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�3.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�I�5�?I����X@Q��=4��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	]QJV�O@]QJV�O@!]QJV�O@      ��!       "	�Q�GT�?�Q�GT�?!�Q�GT�?*      ��!       2	��ڊ�e�?��ڊ�e�?!��ڊ�e�?:	�z6��@�z6��@!�z6��@B      ��!       J	xԘsIu?xԘsIu?!xԘsIu?R      ��!       Z	xԘsIu?xԘsIu?!xԘsIu?b      ��!       JGPUY�I�5�?b q����X@y��=4��?