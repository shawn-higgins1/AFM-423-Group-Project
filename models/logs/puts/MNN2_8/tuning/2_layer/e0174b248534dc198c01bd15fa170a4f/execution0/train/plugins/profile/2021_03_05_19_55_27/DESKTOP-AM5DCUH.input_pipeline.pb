	�]ؚ��7@�]ؚ��7@!�]ؚ��7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�]ؚ��7@dZ���F4@1?��?4��?A�lV}��?I>�hɣ
@*	43333�I@2U
Iterator::Model::ParallelMapV2_�Qڋ?!F~ I4:@)_�Qڋ?1F~ I4:@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate2�%䃎?!g����<@)�I+��?1�H��15@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�������?!�x�	�8@)/�$��?1P2�0;4@:Preprocessing2F
Iterator::ModelM�O��?!��!d�uC@)F%u�{?1U����n)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�q����o?!��E5�@)�q����o?1��E5�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���~�:�?!euޛ�N@)a��+ei?1�N��`�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!H4:��@)����Mb`?1H4:��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?�ܵ�|�?!6^wm?@)a2U0*�S?1���B@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 84.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�13.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�4֗;�X@Qx�r
1�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	dZ���F4@dZ���F4@!dZ���F4@      ��!       "	?��?4��??��?4��?!?��?4��?*      ��!       2	�lV}��?�lV}��?!�lV}��?:	>�hɣ
@>�hɣ
@!>�hɣ
@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�4֗;�X@yx�r
1�?