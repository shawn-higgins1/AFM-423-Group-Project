	l� [~7@l� [~7@!l� [~7@	RI`dH�?RI`dH�?!RI`dH�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6l� [~7@�VC�4@1X)�k{�?A
���ç?If���,p@Y��n�o��?*	������Q@2U
Iterator::Model::ParallelMapV2�z6�>�?!�;���?@)�z6�>�?1�;���?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat;�O��n�?!���:�9@)ŏ1w-!�?1����35@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�J�4�?!�gQ�Sn7@)�{�Pk�?1Z���1@:Preprocessing2F
Iterator::Model8gDio�?!�W2?�bF@)U���N@�?1�N�i8*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�q����o?!2N����@)�q����o?12N����@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��ׁsF�?!+���t�K@)���_vOn?1��Si=�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorǺ���f?!35��=@)Ǻ���f?135��=@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��d�`T�?!��\��8@)/n��R?1_ζ ���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9RI`dH�?I-�n�X@Q����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�VC�4@�VC�4@!�VC�4@      ��!       "	X)�k{�?X)�k{�?!X)�k{�?*      ��!       2	
���ç?
���ç?!
���ç?:	f���,p@f���,p@!f���,p@B      ��!       J	��n�o��?��n�o��?!��n�o��?R      ��!       Z	��n�o��?��n�o��?!��n�o��?b      ��!       JGPUYRI`dH�?b q-�n�X@y����?