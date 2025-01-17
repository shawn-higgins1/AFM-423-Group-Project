�	g)YN�7@g)YN�7@!g)YN�7@	CT�_�?CT�_�?!CT�_�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6g)YN�7@
.V�`�4@1�2��A��?A,e�X�?I_}<��-@Yl=C8f�s?*	������I@2U
Iterator::Model::ParallelMapV2?W[���?!�7�yC=@)?W[���?1�7�yC=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���S㥋?!˚���):@)M�St$�?1:cΘ3�5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateǺ����?!mI[Җ�5@)�<,Ԛ�}?1���-iK,@:Preprocessing2F
Iterator::Modelw-!�l�?!�;�8E@)_�Q�{?1��%mI[*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�q����o?!��#�;@)�q����o?1��#�;@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���B�i�?!w�q�L@)_�Q�k?1��%mI[@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!Cސ7�@)/n��b?1Cސ7�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap-C��6�?!4�9c�8@)-C��6Z?14�9c�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9DT�_�?Iv]l�X@Qt�"�P�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	
.V�`�4@
.V�`�4@!
.V�`�4@      ��!       "	�2��A��?�2��A��?!�2��A��?*      ��!       2	,e�X�?,e�X�?!,e�X�?:	_}<��-@_}<��-@!_}<��-@B      ��!       J	l=C8f�s?l=C8f�s?!l=C8f�s?R      ��!       Z	l=C8f�s?l=C8f�s?!l=C8f�s?b      ��!       JGPUYDT�_�?b qv]l�X@yt�"�P�?�"5
sequential/dense/MatMulMatMul���@ī�?!���@ī�?0"C
%gradient_tape/sequential/dense/MatMulMatMuln�Q�<|�?!��v ��?0"E
'gradient_tape/sequential/dense_1/MatMulMatMuln�Q�<|�?!;Z��R�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMuln�Q�<|�?!y���?"7
sequential/dense_1/MatMulMatMuln�Q�<|�?!�m��-g�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch��%�Kە?!*E6�"�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul��%�Kە?!F�	� ��?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGradn�Q�<|�?!ttF��?"7
sequential/dense_2/MatMulMatMuln�Q�<|�?!Q#���?0"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam��n|c7�?!Y�%�1�?Q      Y@Y>����/@aX�i��U@qh{�l�uW@y<Z��R�?"�
both�Your program is POTENTIALLY input-bound because 88.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�93.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 