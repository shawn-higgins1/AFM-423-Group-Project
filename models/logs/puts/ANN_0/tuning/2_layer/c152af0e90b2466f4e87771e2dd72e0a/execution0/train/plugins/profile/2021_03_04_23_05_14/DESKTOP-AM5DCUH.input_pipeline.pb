	�}"�3@�}"�3@!�}"�3@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�}"�3@ ��L 2@1��A{���?A�H�}�?I�r��o�?*	33333�G@2U
Iterator::Model::ParallelMapV2�?�߾�?!z�[�'�<@)�?�߾�?1z�[�'�<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�
F%u�?!���\�:@)'�����?1�[�'�6@:Preprocessing2F
Iterator::Model䃞ͪϕ?!�jq�wF@)ŏ1w-!?1g�/�0@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapM�O��?!I�1�N5@)F%u�{?1�g *�+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlicey�&1�l?!�ur.�@)y�&1�l?1�ur.�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip}гY���?!����K@)F%u�k?1�g *�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!l��Ӭ�@)����Mb`?1l��Ӭ�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�5.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��u��X@Q[՜b���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	 ��L 2@ ��L 2@! ��L 2@      ��!       "	��A{���?��A{���?!��A{���?*      ��!       2	�H�}�?�H�}�?!�H�}�?:	�r��o�?�r��o�?!�r��o�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��u��X@y[՜b���?