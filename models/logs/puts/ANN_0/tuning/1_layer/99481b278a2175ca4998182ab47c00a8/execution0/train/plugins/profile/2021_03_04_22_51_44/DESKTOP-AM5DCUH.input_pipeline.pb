	��x�s3@��x�s3@!��x�s3@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��x�s3@�_��\1@1�x[���?A��ͪ�զ?I�b��	E�?*	33333sM@2U
Iterator::Model::ParallelMapV2jM�?!���6@@)jM�?1���6@@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap"��u���?!�I��3=@)F%u��?1�n��.i6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�{�Pk�?!��6���5@)�I+��?1]ܙ��2@:Preprocessing2F
Iterator::Model-C��6�?!��q�S�E@)9��v��z?1V��1A&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice����Mbp?!�W�(*@)����Mbp?1�W�(*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��镲�?!* ��DL@)-C��6j?1��q�S�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!�9�s�	@)ŏ1w-!_?1�9�s�	@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�k�;�X@Q4%���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�_��\1@�_��\1@!�_��\1@      ��!       "	�x[���?�x[���?!�x[���?*      ��!       2	��ͪ�զ?��ͪ�զ?!��ͪ�զ?:	�b��	E�?�b��	E�?!�b��	E�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�k�;�X@y4%���?