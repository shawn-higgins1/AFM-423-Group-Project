	�1��l�3@�1��l�3@!�1��l�3@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�1��l�3@�j���2@1�zi� ��?A�^)�Ǫ?I��ť*��?*	����̌M@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap^K�=��?!�岭��A@)�� �rh�?1�M�I��<@:Preprocessing2U
Iterator::Model::ParallelMapV2��H�}�?!�)`>�]8@)��H�}�?1�)`>�]8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��<,Ԋ?!���8s*6@)Ǻ����?1���0p�2@:Preprocessing2F
Iterator::Model/�$��?!�s{-9�A@)F%u�{?1t{-9�U&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice	�^)�p?!x�	G�@)	�^)�p?1x�	G�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip����ׁ�?!,FBi�P@)�q����o?1�גC\e@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!VH�A�	@)ŏ1w-!_?1VH�A�	@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 90.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�7.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIL�n���X@Q
m_d�^�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�j���2@�j���2@!�j���2@      ��!       "	�zi� ��?�zi� ��?!�zi� ��?*      ��!       2	�^)�Ǫ?�^)�Ǫ?!�^)�Ǫ?:	��ť*��?��ť*��?!��ť*��?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qL�n���X@y
m_d�^�?