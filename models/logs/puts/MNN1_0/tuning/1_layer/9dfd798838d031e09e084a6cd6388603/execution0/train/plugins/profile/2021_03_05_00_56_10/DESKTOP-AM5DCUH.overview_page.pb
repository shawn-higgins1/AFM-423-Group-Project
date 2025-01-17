�	�9?�q�6@�9?�q�6@!�9?�q�6@	M���_�?M���_�?!M���_�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�9?�q�6@f/�N[4@1������?AV����_�?I.�s`Y@Y܂����i?*	����̌I@2U
Iterator::Model::ParallelMapV2���H�?!رcǎ?@)���H�?1رcǎ?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatg��j+��?!W�\�r�6@)n���?1ٲe˖-3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�HP��?!������7@)"��u���?1F�5j�0@:Preprocessing2F
Iterator::ModelM�St$�?!ѡC�F@)S�!�uq{?1�#G�9*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��H�}m?!�.@)��H�}m?1�.@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�v��/�?!/^�x��K@)_�Q�k?1ԩS�N�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!�۷o߾@)ŏ1w-!_?1�۷o߾@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapS�!�uq�?!�#G�9:@)a2U0*�S?1�,Y�d�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9M���_�?I�mv[R�X@QVU@S��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	f/�N[4@f/�N[4@!f/�N[4@      ��!       "	������?������?!������?*      ��!       2	V����_�?V����_�?!V����_�?:	.�s`Y@.�s`Y@!.�s`Y@B      ��!       J	܂����i?܂����i?!܂����i?R      ��!       Z	܂����i?܂����i?!܂����i?b      ��!       JGPUYM���_�?b q�mv[R�X@yVU@S��?�"5
sequential/dense/MatMulMatMul�y���f�?!�y���f�?0"C
%gradient_tape/sequential/dense/MatMulMatMul���T��?!�=��4�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch\��S�?!�u��m�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad\��S�?!��V62S�?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul\��S�?!��r��o�?"7
sequential/dense_1/MatMulMatMul\��S�?!�����?0"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdamJ���?!�>)	�?")
sequential/CastCastJ���?!���K��?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdamo@��ݍ?!��2�$d�?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdamo@��ݍ?!tG;�� �?Q      Y@Y��/Ċ�0@a�	�N]�T@qQ>��@OW@ygR_U�?"�
both�Your program is POTENTIALLY input-bound because 89.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�93.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 