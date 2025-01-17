�	��"��^3@��"��^3@!��"��^3@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��"��^3@�����1@1��"��J�?A��e�O7�?IQ��ڦ��?*	33333sL@2U
Iterator::Model::ParallelMapV2K�=�U�?! �5ѧ�:@)K�=�U�?1 �5ѧ�:@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap������?!��;S>@)�]K�=�?1����_7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���S㥋?!�=/Oй7@)46<�R�?1E�B�
(3@:Preprocessing2F
Iterator::ModelZd;�O��?!oZ�L�5D@)� �	�?1{UR��+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceŏ1w-!o?!ľ��@)ŏ1w-!o?1ľ��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipz6�>W[�?!����M@)F%u�k?1����2@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��_�Le?!x��G@)��_�Le?1x��G@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIr�i"�X@Q�{#�e��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�����1@�����1@!�����1@      ��!       "	��"��J�?��"��J�?!��"��J�?*      ��!       2	��e�O7�?��e�O7�?!��e�O7�?:	Q��ڦ��?Q��ڦ��?!Q��ڦ��?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qr�i"�X@y�{#�e��?�"5
sequential/dense/MatMulMatMul<#Hc���?!<#Hc���?0"C
%gradient_tape/sequential/dense/MatMulMatMuln�tOk�?!��k��?0"E
'gradient_tape/sequential/dense_1/MatMulMatMuln�tOk�?!�U&��?0"7
sequential/dense_1/MatMulMatMuln�tOk�?!�uW��?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul0O���ޡ?!nI}(L��?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�1s�֝?!����A�?"7
sequential/dense_2/MatMulMatMul
`�Q#��?!Խ��8�?0"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad�4"u��?!�E���?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam��
/���?!�7�ƪ��?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdam��
/���?!��6����?Q      Y@Y7��Moz2@a���,daT@qPp-�cX@y
z����?"�
both�Your program is POTENTIALLY input-bound because 92.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�6.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 