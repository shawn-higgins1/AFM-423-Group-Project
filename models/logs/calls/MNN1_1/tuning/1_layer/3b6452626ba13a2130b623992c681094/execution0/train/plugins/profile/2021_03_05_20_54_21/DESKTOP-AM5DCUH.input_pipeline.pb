	z���5@z���5@!z���5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-z���5@�sa�U3@1�k����?AǺ���?IHk:!��?*	�����YK@2U
Iterator::Model::ParallelMapV2����Mb�?!~���@=@)����Mb�?1~���@=@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateL7�A`�?!
�s�*>@)F%u��?1���O�!8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatM�St$�?!'d΍{�4@)U���N@�?1��D/1@:Preprocessing2F
Iterator::ModelZd;�O��?!�A�E@)y�&1�|?1NSZ5�)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceF%u�k?!���O�!@)F%u�k?1���O�!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���~�:�?!n�x��L@)Ǻ���f?1r��y@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!�K
��@)ŏ1w-!_?1�K
��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap/n���?!l��v@@)/n��R?1l��v @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�"��q�X@Q�B�ƃ��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�sa�U3@�sa�U3@!�sa�U3@      ��!       "	�k����?�k����?!�k����?*      ��!       2	Ǻ���?Ǻ���?!Ǻ���?:	Hk:!��?Hk:!��?!Hk:!��?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�"��q�X@y�B�ƃ��?