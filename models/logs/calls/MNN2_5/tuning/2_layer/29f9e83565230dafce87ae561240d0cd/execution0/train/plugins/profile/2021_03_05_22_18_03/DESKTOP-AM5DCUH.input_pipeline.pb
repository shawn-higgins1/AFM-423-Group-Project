	Hp#e��3@Hp#e��3@!Hp#e��3@	4K�5�?4K�5�?!4K�5�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6Hp#e��3@�ՔdI2@1�ܘ����?A�#H��Ѩ?I]P�2��?Yo���T��?*	33333�G@2U
Iterator::Model::ParallelMapV2���S㥋?!�D�#{<@)���S㥋?1�D�#{<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat_�Qڋ?!��O�%�<@)Zd;�O��?1�iVp�B8@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��_vO�?!�=Q��6@)�<,Ԛ�}?1F��h;�.@:Preprocessing2F
Iterator::Model{�G�z�?!�/��E@)9��v��z?1/��m+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlicey�&1�l?!�ur.�@)y�&1�l?1�ur.�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�?�߾�?!z�[�'�L@)�~j�t�h?1�ґ=Q@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!q�w��@)�J�4a?1q�w��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no95K�5�?I7�C÷�X@Q�l�q�D�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�ՔdI2@�ՔdI2@!�ՔdI2@      ��!       "	�ܘ����?�ܘ����?!�ܘ����?*      ��!       2	�#H��Ѩ?�#H��Ѩ?!�#H��Ѩ?:	]P�2��?]P�2��?!]P�2��?B      ��!       J	o���T��?o���T��?!o���T��?R      ��!       Z	o���T��?o���T��?!o���T��?b      ��!       JGPUY5K�5�?b q7�C÷�X@y�l�q�D�?