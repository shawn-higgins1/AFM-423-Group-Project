	6ɏ�{3@6ɏ�{3@!6ɏ�{3@	L�E�aX�?L�E�aX�?!L�E�aX�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails66ɏ�{3@�j,a�1@1�H�]��?A�e��a��?I���Bt��?Y?�{�&�?*	�����I@2U
Iterator::Model::ParallelMapV2?W[���?!���e>@)?W[���?1���e>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���S㥋?!�čOv�:@)M�St$�?1������6@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��0�*�?!-<?���7@)ŏ1w-!?1�M��dG.@:Preprocessing2F
Iterator::Model�e��a��?!Zx~LF@)y�&1�|?1dG6q�+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�J�4q?!�*�S�� @)�J�4q?1�*�S�� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�?Ɯ?!������K@)Ǻ���f?19�čO@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!�,<?��@)/n��b?1�,<?��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9L�E�aX�?I6�6�X@Qm?�0��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�j,a�1@�j,a�1@!�j,a�1@      ��!       "	�H�]��?�H�]��?!�H�]��?*      ��!       2	�e��a��?�e��a��?!�e��a��?:	���Bt��?���Bt��?!���Bt��?B      ��!       J	?�{�&�??�{�&�?!?�{�&�?R      ��!       Z	?�{�&�??�{�&�?!?�{�&�?b      ��!       JGPUYL�E�aX�?b q6�6�X@ym?�0��?