�	�`ʜ3@�`ʜ3@!�`ʜ3@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�`ʜ3@����	2@1�7�{�5�?A���JY��?I��Z�a�?*effff�H@)       =2U
Iterator::Model::ParallelMapV2vq�-�?!#NT��?@)vq�-�?1#NT��?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�HP��?!���f|8@)��_�L�?1�]K/�4@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapa��+e�?!��L�8@)"��u���?1^K/�D1@:Preprocessing2F
Iterator::Modelw-!�l�?!��h��E@)�HP�x?1���f|(@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceŏ1w-!o?!�����@)ŏ1w-!o?1�����@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip6�;Nё�?!~�!�L@)F%u�k?1���ˊ�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!���h�@)��H�}]?1���h�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�:��-�X@QbB1����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	����	2@����	2@!����	2@      ��!       "	�7�{�5�?�7�{�5�?!�7�{�5�?*      ��!       2	���JY��?���JY��?!���JY��?:	��Z�a�?��Z�a�?!��Z�a�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�:��-�X@ybB1����?�"5
sequential/dense/MatMulMatMulޯ,]�?!ޯ,]�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�fh)�?!=�+4�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul?)�{�?!Ln�G9�?0"7
sequential/dense_1/MatMulMatMul?)�{�?!��	�W�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul4D�<�?!�� 3���?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGradݎ�ȟ��?!�5?,R�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMulݎ�ȟ��?!�ӮӜ�?"7
sequential/dense_2/MatMulMatMulݎ�ȟ��?!�>}6�?0"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam?)�{�?!���?8~�?"!
Adam/PowPow�G��� �?!$� E��?Q      Y@Y7��Moz2@a���,daT@q��h��iX@y��o��?"�
both�Your program is POTENTIALLY input-bound because 92.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�6.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 