	��}U.h5@��}U.h5@!��}U.h5@	�\�z��?�\�z��?!�\�z��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6��}U.h5@�L0�k83@1����Dh�?A��ͪ�զ?I���W_�?Y�}͑�?*	    �I@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::ConcatenateM�O��?!������C@)Έ����?1<<<<<<B@:Preprocessing2U
Iterator::Model::ParallelMapV2S�!�uq�?!GFFFFF:@)S�!�uq�?1GFFFFF:@:Preprocessing2F
Iterator::Model8��d�`�?!������C@)9��v��z?1}}}}}})@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatF%u�{?!������)@);�O��nr?1������!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�U���؟?!~}}}}}N@)a��+ei?1PPPPPP@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!xxxxxx@)�J�4a?1xxxxxx@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap'�����?!������D@)a2U0*�S?1������@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor����MbP?!______�?)����MbP?1______�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicea2U0*�C?!�������?)a2U0*�C?1�������?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�\�z��?Ip�ڢu�X@Q��G���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�L0�k83@�L0�k83@!�L0�k83@      ��!       "	����Dh�?����Dh�?!����Dh�?*      ��!       2	��ͪ�զ?��ͪ�զ?!��ͪ�զ?:	���W_�?���W_�?!���W_�?B      ��!       J	�}͑�?�}͑�?!�}͑�?R      ��!       Z	�}͑�?�}͑�?!�}͑�?b      ��!       JGPUY�\�z��?b qp�ڢu�X@y��G���?