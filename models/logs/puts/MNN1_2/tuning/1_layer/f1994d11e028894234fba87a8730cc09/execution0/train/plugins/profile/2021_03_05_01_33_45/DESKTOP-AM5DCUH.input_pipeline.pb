	�r���4@�r���4@!�r���4@	{����~�?{����~�?!{����~�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�r���4@��ӹ�x2@1g{����?A,e�X�?I�ŏ1� @Yd��1�n?*	������I@2U
Iterator::Model::ParallelMapV2� �	��?!��=aO�=@)� �	��?1��=aO�=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��<,Ԋ?!�3�9c9@)Ǻ����?1mI[Җ�5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�HP��?!iKڒ��7@)�J�4�?1w�qG0@:Preprocessing2F
Iterator::Model46<�R�?!��E@)-C��6z?14�9c�(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceŏ1w-!o?!^QW�u@)ŏ1w-!o?1^QW�u@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip2�%䃞?!���?�L@)_�Q�k?1��%mI[@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!^QW�u@)ŏ1w-!_?1^QW�u@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�]K�=�?!2g�s�9@)/n��R?1Cސ7�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9{����~�?I������X@Q�6)���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��ӹ�x2@��ӹ�x2@!��ӹ�x2@      ��!       "	g{����?g{����?!g{����?*      ��!       2	,e�X�?,e�X�?!,e�X�?:	�ŏ1� @�ŏ1� @!�ŏ1� @B      ��!       J	d��1�n?d��1�n?!d��1�n?R      ��!       Z	d��1�n?d��1�n?!d��1�n?b      ��!       JGPUY{����~�?b q������X@y�6)���?