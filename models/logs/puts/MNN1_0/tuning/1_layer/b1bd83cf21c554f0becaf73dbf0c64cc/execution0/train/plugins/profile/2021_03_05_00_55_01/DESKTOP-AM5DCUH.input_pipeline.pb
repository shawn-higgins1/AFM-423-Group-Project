	P���6@P���6@!P���6@	�hr�-�?�hr�-�?!�hr�-�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6P���6@cC7��3@1ϺFˁ�?A䃞ͪϥ?I<f�2���?Y&p�n��`?*	333333L@2U
Iterator::Model::ParallelMapV2�q����?!��Q�٨;@)�q����?1��Q�٨;@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���Q��?!r1���:@)�+e�X�?1��l�w64@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�(��0�?!�����5@)�0�*�?1��c-C2@:Preprocessing2F
Iterator::Model�~j�t��?!�����FE@)�J�4�?1W�+��-@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��ǘ���?!br1�L@)��H�}m?1&W�+�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��H�}m?!&W�+�@)��H�}m?1&W�+�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!�'Ni^@)����Mb`?1�'Ni^@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapL7�A`�?!1���\A=@)�~j�t�X?1�����F@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�hr�-�?I���B��X@Qҿ:O���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	cC7��3@cC7��3@!cC7��3@      ��!       "	ϺFˁ�?ϺFˁ�?!ϺFˁ�?*      ��!       2	䃞ͪϥ?䃞ͪϥ?!䃞ͪϥ?:	<f�2���?<f�2���?!<f�2���?B      ��!       J	&p�n��`?&p�n��`?!&p�n��`?R      ��!       Z	&p�n��`?&p�n��`?!&p�n��`?b      ��!       JGPUY�hr�-�?b q���B��X@yҿ:O���?