	Z��88@Z��88@!Z��88@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-Z��88@���;�o5@1��Tl��?A`[?�gͧ?I?���2y@*	�����YI@2U
Iterator::Model::ParallelMapV2���_vO�?!7Ir�0=@)���_vO�?17Ir�0=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeata��+e�?!�_&�$u8@){�G�z�?1�v/FO�3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�HP��?!*w�d(8@)�� �rh�?1�$u���0@:Preprocessing2F
Iterator::Model46<�R�?!#l^�E@)y�&1�|?1 �u��+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���_vOn?!7Ir�0@)���_vOn?17Ir�0@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��j+���?!��@�L@)a��+ei?1�_&�$u@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*�c?!��WV�@)a2U0*�c?1��WV�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapS�!�uq�?!���/n:@)a2U0*�S?1��WV�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIFR콐X@Qzn�����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���;�o5@���;�o5@!���;�o5@      ��!       "	��Tl��?��Tl��?!��Tl��?*      ��!       2	`[?�gͧ?`[?�gͧ?!`[?�gͧ?:	?���2y@?���2y@!?���2y@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qFR콐X@yzn�����?