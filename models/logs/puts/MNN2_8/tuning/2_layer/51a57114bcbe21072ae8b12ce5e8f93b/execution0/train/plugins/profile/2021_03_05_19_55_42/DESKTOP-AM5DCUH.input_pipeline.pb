	I+���M7@I+���M7@!I+���M7@	��T���?��T���?!��T���?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6I+���M7@�M�a�_4@1p��1=a�?AM�St$�?Id��@Y"P��H�l?*	�����P@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate0*��D�?!H��fB@)�ݓ��Z�?1�,d!Y=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��ǘ���?!���e�'9@)���S㥋?1;�"�u�4@:Preprocessing2U
Iterator::Model::ParallelMapV2tF��_�?!7��Moz2@)tF��_�?17��Moz2@:Preprocessing2F
Iterator::Model���&�?!��U�	=@)_�Q�{?1��6%@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicea2U0*�s?!&�RL�@)a2U0*�s?1&�RL�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��ڊ�e�?!Ͼ�j��Q@)/n��r?1#�u�ET@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��_vOf?!�b��*�@)��_vOf?1�b��*�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapݵ�|г�?!i�`��|C@)Ǻ���V?1���,d@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��T���?I�I�ȷ�X@Q�鲦���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�M�a�_4@�M�a�_4@!�M�a�_4@      ��!       "	p��1=a�?p��1=a�?!p��1=a�?*      ��!       2	M�St$�?M�St$�?!M�St$�?:	d��@d��@!d��@B      ��!       J	"P��H�l?"P��H�l?!"P��H�l?R      ��!       Z	"P��H�l?"P��H�l?!"P��H�l?b      ��!       JGPUY��T���?b q�I�ȷ�X@y�鲦���?