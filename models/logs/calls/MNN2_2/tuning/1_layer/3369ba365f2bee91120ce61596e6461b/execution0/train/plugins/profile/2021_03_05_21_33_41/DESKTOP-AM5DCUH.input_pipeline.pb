	)�7�M6@)�7�M6@!)�7�M6@	#��.�r?#��.�r?!#��.�r?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6)�7�M6@z�ަ?�3@1��i����?A����z�?IP�c*� @Y������P?*	    �H@2U
Iterator::Model::ParallelMapV22�%䃎?!ogH��>@)2�%䃎?1ogH��>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�������?!��~�@9@)�g��s��?1��+j5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate������?!ż�!1o7@)vq�-�?1A�_)P�/@:Preprocessing2F
Iterator::Model�0�*�?!z;Cb��D@)�+e�Xw?1|��'@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice���_vOn?!���3$�@)���_vOn?1���3$�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��j+���?!�ļ�!1M@)-C��6j?11ogH��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!
����@)ŏ1w-!_?1
����@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapF%u��?!������:@)-C��6Z?11ogH��	@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9#��.�r?I�%l��X@Q}�n����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	z�ަ?�3@z�ަ?�3@!z�ަ?�3@      ��!       "	��i����?��i����?!��i����?*      ��!       2	����z�?����z�?!����z�?:	P�c*� @P�c*� @!P�c*� @B      ��!       J	������P?������P?!������P?R      ��!       Z	������P?������P?!������P?b      ��!       JGPUY#��.�r?b q�%l��X@y}�n����?