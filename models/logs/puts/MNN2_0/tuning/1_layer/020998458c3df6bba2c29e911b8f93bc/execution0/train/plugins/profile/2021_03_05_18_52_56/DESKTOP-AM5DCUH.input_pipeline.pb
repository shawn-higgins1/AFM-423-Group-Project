	3m��Js6@3m��Js6@!3m��Js6@	�$�>!��?�$�>!��?!�$�>!��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails63m��Js6@(�$�%4@1vi���?A;m����?IbHN&n��?Y�d��)�?*	533333N@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate�������?!�d�D@)������?1:�i�4C@:Preprocessing2U
Iterator::Model::ParallelMapV2���H�?!KL%�S:@)���H�?1KL%�S:@:Preprocessing2F
Iterator::ModelM�St$�?!���jg�B@)S�!�uq{?1��^x/&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�ݓ��Z�?!b*1��JO@)��_�Lu?1���o�7!@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat-C��6z?!��Tb*1%@)ŏ1w-!o?1���Tb*@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��_�Le?!���o�7@)��_�Le?1���o�7@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��<,Ԛ?!E��`Q�E@)a2U0*�S?1�&����?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensora2U0*�S?!�&����?)a2U0*�S?1�&����?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicea2U0*�C?!�&����?)a2U0*�C?1�&����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�$�>!��?IU�7��X@Qu!h"�I�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	(�$�%4@(�$�%4@!(�$�%4@      ��!       "	vi���?vi���?!vi���?*      ��!       2	;m����?;m����?!;m����?:	bHN&n��?bHN&n��?!bHN&n��?B      ��!       J	�d��)�?�d��)�?!�d��)�?R      ��!       Z	�d��)�?�d��)�?!�d��)�?b      ��!       JGPUY�$�>!��?b qU�7��X@yu!h"�I�?