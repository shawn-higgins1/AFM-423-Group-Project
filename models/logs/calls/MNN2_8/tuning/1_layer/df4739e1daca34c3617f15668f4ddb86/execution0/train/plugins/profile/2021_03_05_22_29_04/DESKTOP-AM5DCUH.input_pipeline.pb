	���j}5@���j}5@!���j}5@	��|;�?��|;�?!��|;�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6���j}5@������2@1��.Q�5�?A��&��?I{m���@YI�����r?*	33333�H@2U
Iterator::Model::ParallelMapV2�<,Ԛ�?!�,.B=@)�<,Ԛ�?1�,.B=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�
F%u�?!M3�Lns9@)�0�*�?1��+�4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��@��ǈ?!)U��?8@)����Mb�?1�<�Z50@:Preprocessing2F
Iterator::Model�0�*�?!��+�D@)�~j�t�x?1�Z5P(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice	�^)�p?!@1�I�n @)	�^)�p?1@1�I�n @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�X�� �?!CE���[M@)-C��6j?1�-}Ļ�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*�c?!?�]�=@)a2U0*�c?1?�]�=@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapS�!�uq�?!�����:@)��_�LU?1���x�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��|;�?IU�q1�X@Q�B�u���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	������2@������2@!������2@      ��!       "	��.Q�5�?��.Q�5�?!��.Q�5�?*      ��!       2	��&��?��&��?!��&��?:	{m���@{m���@!{m���@B      ��!       J	I�����r?I�����r?!I�����r?R      ��!       Z	I�����r?I�����r?!I�����r?b      ��!       JGPUY��|;�?b qU�q1�X@y�B�u���?