�	Y�&�&3@Y�&�&3@!Y�&�&3@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-Y�&�&3@dY0�1@1�G��[�?A_�BF��?I��f�b�?*	������N@2U
Iterator::Model::ParallelMapV2��_�L�?!P����@@)��_�L�?1P����@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���H�?!,?!���9@)S�!�uq�?1S{���5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap2U0*��?!���|9@)��@��ǈ?1I�$I��3@:Preprocessing2F
Iterator::Modelz6�>W�?!5�wL�E@)��0�*x?1��g�'#@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice��H�}m?!���cj`@)��H�}m?1���cj`@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��B�iޡ?!�O���SL@)F%u�k?1۶m۶m@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{�G�zd?!`��;@){�G�zd?1`��;@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 91.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�7.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��9/�X@Qї��<4�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	dY0�1@dY0�1@!dY0�1@      ��!       "	�G��[�?�G��[�?!�G��[�?*      ��!       2	_�BF��?_�BF��?!_�BF��?:	��f�b�?��f�b�?!��f�b�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��9/�X@yї��<4�?�"5
sequential/dense/MatMulMatMuls�$�!�?!s�$�!�?0"C
%gradient_tape/sequential/dense/MatMulMatMulI� �V^�?!^�"<��?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulI� �V^�?!`3sgӿ?"E
'gradient_tape/sequential/dense_1/MatMulMatMul����5�?!��I:��?0"7
sequential/dense_1/MatMulMatMul �ڋҡ?!+	�0�k�?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�)\�u�?!^,V�z��?"7
sequential/dense_2/MatMulMatMul�)\�u�?!�O�G8��?0"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGradTTT���?!-.�Wt�?")
sequential/CastCastI� �V^�?!#;0�<��?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam � �?!�<��~��?Q      Y@Y7��Moz2@a���,daT@q���XX@y`3sg��?"�
both�Your program is POTENTIALLY input-bound because 91.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�7.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 