�	-AF@�4@-AF@�4@!-AF@�4@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails--AF@�4@���l�2@1�1�L�?AWya��?I��J̳R�?*	     @G@2U
Iterator::Model::ParallelMapV2��<,Ԋ?!,��,<@)��<,Ԋ?1,��,<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�{�Pk�?!���{�;@)'�����?1,��7@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap46<�R�?!q�7�p7@)lxz�,C|?1���Zk�-@:Preprocessing2F
Iterator::Model�N@aÓ?!�,��D@)a��+ey?1������*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice����Mbp?!5�DM4!@)����Mbp?15�DM4!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip_�Qڛ?!@���O?M@)_�Q�k?1@���O?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!���.��@)/n��b?1���.��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�.����X@Q�H���Y�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���l�2@���l�2@!���l�2@      ��!       "	�1�L�?�1�L�?!�1�L�?*      ��!       2	Wya��?Wya��?!Wya��?:	��J̳R�?��J̳R�?!��J̳R�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�.����X@y�H���Y�?�"5
sequential/dense/MatMulMatMul:L���?!:L���?0"C
%gradient_tape/sequential/dense/MatMulMatMul��
��i�?!�o�dW��?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul��
��i�?!�Ѱ95�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul��
��i�?!�[�	�?"7
sequential/dense_1/MatMulMatMul��
��i�?!���qx'�?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul+_9W[�?!���U�R�?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam��
��i�?!-O����?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad��
��i�?!�� �6�?"7
sequential/dense_2/MatMulMatMul��
��i�?!���D}�?0"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam�%���?!<��9r�?Q      Y@Y7��Moz2@a���,daT@q�p..f|X@y�Ѱ95��?"�
both�Your program is POTENTIALLY input-bound because 92.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�6.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 