	
�_�q3@
�_�q3@!
�_�q3@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-
�_�q3@���*l�1@1��wF[��?A�!H��?I�67�'��?*	fffff&H@2U
Iterator::Model::ParallelMapV2�q����?!tW)&@@)�q����?1tW)&@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatM�St$�?!H]p�;e7@)U���N@�?1nk�1v3@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�~j�t��?! l<�?�8@)��ǘ���?1�b��*�0@:Preprocessing2F
Iterator::Model�I+��?!�bM:�F@)-C��6z?1���C�*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�q����o?!tW)& @)�q����o?1tW)& @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip	�c�?!8��9K@)�~j�t�h?1 l<�?�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!�f*�Px@)ŏ1w-!_?1�f*�Px@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 91.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�7.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIo2���X@Q'd3���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���*l�1@���*l�1@!���*l�1@      ��!       "	��wF[��?��wF[��?!��wF[��?*      ��!       2	�!H��?�!H��?!�!H��?:	�67�'��?�67�'��?!�67�'��?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qo2���X@y'd3���?