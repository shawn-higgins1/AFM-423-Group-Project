�	'�ei��5@'�ei��5@!'�ei��5@	v�p����?v�p����?!v�p����?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6'�ei��5@�<���3@1+4�f�?A������?IjN^d��?Y��h��?*�����G@)       =2U
Iterator::Model::ParallelMapV2a��+e�?!�[K���:@)a��+e�?1�[K���:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat9��v���?!ۍ��v#<@)'�����?1��)7@:Preprocessing2F
Iterator::Model�N@aÓ?!��W�/�D@)lxz�,C|?1{����-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate{�G�z�?!��v# �5@)�~j�t�x?1f�'�Y�)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����Mbp?!�Cł�P!@)����Mbp?1�Cł�P!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��q���?!As�X�M@)��_vOf?1���cj`@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP�s�b?!X�/���@)HP�s�b?1X�/���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapǺ����?!>���>8@)a2U0*�S?1�S{�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9v�p����?I�ع��{X@QddQ�>�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�<���3@�<���3@!�<���3@      ��!       "	+4�f�?+4�f�?!+4�f�?*      ��!       2	������?������?!������?:	jN^d��?jN^d��?!jN^d��?B      ��!       J	��h��?��h��?!��h��?R      ��!       Z	��h��?��h��?!��h��?b      ��!       JGPUYv�p����?b q�ع��{X@yddQ�>�?�"5
sequential/dense/MatMulMatMul�fA����?!�fA����?0"C
%gradient_tape/sequential/dense/MatMulMatMul����u��?!���5C�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul��M	�?!��i$���?"7
sequential/dense_1/MatMulMatMul��M	�?!�r�;]��?0"!
Adam/PowPowJc�
��?!8C]k�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchJc�
��?!�˞~���?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGradJc�
��?!
x��yp�?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam0x��t�?!GK ��?"O
5gradient_tape/mean_squared_error/weighted_loss/Tile_1Tile�[~�$�?!d�ЫC�?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam�L�q �?!�h�%�?Q      Y@Y��/Ċ�0@a�	�N]�T@q�s��/S@y$����a�?"�
both�Your program is POTENTIALLY input-bound because 88.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�76.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 