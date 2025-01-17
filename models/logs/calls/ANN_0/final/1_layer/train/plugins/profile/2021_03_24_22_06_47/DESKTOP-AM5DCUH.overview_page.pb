�	�Q+Lߛ@@�Q+Lߛ@@!�Q+Lߛ@@	�3L�~�?�3L�~�?!�3L�~�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�Q+Lߛ@@��:TS�?@12���J�?A0*��D�?I_���F��?Y�up�7�?*	43333�O@2U
Iterator::Model::ParallelMapV2/�$��?!}8��n@@)/�$��?1}8��n@@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapˡE����?!���j
@@)�St$���?1h(d�9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeattF��_�?!|���r�2@)n���?1��\�D�.@:Preprocessing2F
Iterator::ModelB>�٬��?!*h��$F@)�<,Ԛ�}?1��ܧ#�&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�q����o?!�b���i@)�q����o?1�b���i@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��y�):�?!ח�t$�K@)-C��6j?1�d{4@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!���4�J
@)�J�4a?1���4�J
@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 95.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�3.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�3L�~�?Is����X@Q��9�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��:TS�?@��:TS�?@!��:TS�?@      ��!       "	2���J�?2���J�?!2���J�?*      ��!       2	0*��D�?0*��D�?!0*��D�?:	_���F��?_���F��?!_���F��?B      ��!       J	�up�7�?�up�7�?!�up�7�?R      ��!       Z	�up�7�?�up�7�?!�up�7�?b      ��!       JGPUY�3L�~�?b qs����X@y��9�?�"5
sequential/dense/MatMulMatMulҧ�-^�?!ҧ�-^�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�~(87�?!���ʺ?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul���2��?!�-כ�U�?"7
sequential/dense_1/MatMulMatMul���2��?!��*b�F�?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�~(87�?!��:g~m�?"E
'gradient_tape/sequential/dense_1/MatMulMatMul�~(87�?!b�Jle��?0"!
Adam/PowPow���G[Ҙ?!�KFհ��?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdam�_j�?!����?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam�_j�?!0	o����?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam�_j�?!��TP��?Q      Y@Y�M�_{4@a��(�S@q֚l��:U@y7�?�=�?"�
both�Your program is POTENTIALLY input-bound because 95.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�3.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�84.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 