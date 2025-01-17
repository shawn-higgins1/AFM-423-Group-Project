�	I�H��7@I�H��7@!I�H��7@	t]�ߑ?t]�ߑ?!t]�ߑ?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6I�H��7@�|A�4@1���Y���?A��3���?I-$`t9@Y%=�N�p?*	gffff�F@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenatee�X��?!t�?;��B@)2U0*��?1qv��#A@:Preprocessing2U
Iterator::Model::ParallelMapV2Zd;�O��?!��̕�9@)Zd;�O��?1��̕�9@:Preprocessing2F
Iterator::ModelΈ����?!�6��@ND@)�ZӼ�}?1���./@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatǺ���v?!/Ct(@)���_vOn?1_OE�>( @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip_�Qڛ?!:�\)��M@)-C��6j?15��̕�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!��w��@)ŏ1w-!_?1��w��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapHP�sג?!�
�[D@)/n��R?1%s���6@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor/n��R?!%s���6@)/n��R?1%s���6@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice����Mb@?!!:ܟ�w�?)����Mb@?1!:ܟ�w�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9u]�ߑ?IǊ~���X@Q���P��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�|A�4@�|A�4@!�|A�4@      ��!       "	���Y���?���Y���?!���Y���?*      ��!       2	��3���?��3���?!��3���?:	-$`t9@-$`t9@!-$`t9@B      ��!       J	%=�N�p?%=�N�p?!%=�N�p?R      ��!       Z	%=�N�p?%=�N�p?!%=�N�p?b      ��!       JGPUYu]�ߑ?b qǊ~���X@y���P��?�"5
sequential/dense/MatMulMatMul��M�&�?!��M�&�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�ƶ$[��?!.3�@�?"E
'gradient_tape/sequential/dense_1/MatMulMatMul?'�e/��?!��~��}�?0"C
%gradient_tape/sequential/dense/MatMulMatMul��?���?!�YGH�v�?0"7
sequential/dense_1/MatMulMatMul��?���?!sUO8..�?0"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGradrx���=�?!��v���?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMulrx���=�?!�sȳ�}�?"7
sequential/dense_2/MatMulMatMulrx���=�?!���h%�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch!�+�r�?!������?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam�ƶ$[��?!-����i�?Q      Y@Y�C=�C=0@a��
��T@q�=�W@y��]in��?"�
both�Your program is POTENTIALLY input-bound because 87.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�11.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�94.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 