	����p3@����p3@!����p3@	�Ґ+o.�?�Ґ+o.�?!�Ґ+o.�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6����p3@#���1@1&��)�?A�[���?I 	�v��?Y �!p$p?*	�����YK@2U
Iterator::Model::ParallelMapV2X9��v��?!�I6V<@)X9��v��?1�I6V<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatlxz�,C�?!��;�:9@)g��j+��?1�����c5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapX9��v��?!�I6V<@)46<�R�?1Q�G�3@:Preprocessing2F
Iterator::Model46<�R�?!Q�G�C@) �o_�y?1`�7c�'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceHP�s�r?!Bq�Ҫ� @)HP�s�r?1Bq�Ҫ� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip*��Dؠ?!��e�N@)��H�}m?1$�?(NS@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!+��م�@)�J�4a?1+��م�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�Ґ+o.�?IX�����X@Q`ͥ�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	#���1@#���1@!#���1@      ��!       "	&��)�?&��)�?!&��)�?*      ��!       2	�[���?�[���?!�[���?:	 	�v��? 	�v��?! 	�v��?B      ��!       J	 �!p$p? �!p$p?! �!p$p?R      ��!       Z	 �!p$p? �!p$p?! �!p$p?b      ��!       JGPUY�Ґ+o.�?b qX�����X@y`ͥ�?