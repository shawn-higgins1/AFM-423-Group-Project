	���u��3@���u��3@!���u��3@	|�h�"�?|�h�"�?!|�h�"�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6���u��3@9`W��$2@1"¿3�?A������?I�.�KR�?Y]�E�~�?*	����̌I@2U
Iterator::Model::ParallelMapV2� �	��?!1bĈ#>@)� �	��?11bĈ#>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat-C��6�?!Ȑ!C�9@)46<�R�?1J�*U�T5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�
F%u�?!�M�6m�8@)/n���?1�'N�81@:Preprocessing2F
Iterator::Model��JY�8�?!�s�Ν;E@) �o_�y?1�
*T�(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip%u��?!F�1b�L@)�q����o?1t�СC�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�q����o?!t�СC�@)�q����o?1t�СC�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!�۷o߾@)ŏ1w-!_?1�۷o߾@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 91.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9}�h�"�?I�-In�X@Q郃��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	9`W��$2@9`W��$2@!9`W��$2@      ��!       "	"¿3�?"¿3�?!"¿3�?*      ��!       2	������?������?!������?:	�.�KR�?�.�KR�?!�.�KR�?B      ��!       J	]�E�~�?]�E�~�?!]�E�~�?R      ��!       Z	]�E�~�?]�E�~�?!]�E�~�?b      ��!       JGPUY}�h�"�?b q�-In�X@y郃��?