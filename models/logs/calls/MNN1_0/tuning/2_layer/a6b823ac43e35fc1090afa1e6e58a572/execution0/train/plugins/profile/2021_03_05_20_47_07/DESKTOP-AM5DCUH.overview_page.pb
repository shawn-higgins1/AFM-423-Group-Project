�	��Z�4@��Z�4@!��Z�4@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��Z�4@�H�"i'2@1�bFx{�?AEGr��?I�<�r�]@*fffff�I@)       =2U
Iterator::Model::ParallelMapV2�q����?!�a�
�>@)�q����?1�a�
�>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�]K�=�?!Ϻ���9@)��_vO�?1���s�4@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapF%u��?!��W�{9@)��~j�t�?1p���V2@:Preprocessing2F
Iterator::Model��_vO�?!���s�D@)�~j�t�x?1�r�~�*'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?W[���?!h2�Z�&M@)���_vOn?1$I�$I�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice���_vOn?!$I�$I�@)���_vOn?1$I�$I�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{�G�zd?!�þN@){�G�zd?1�þN@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�+�6��X@Q|�^��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�H�"i'2@�H�"i'2@!�H�"i'2@      ��!       "	�bFx{�?�bFx{�?!�bFx{�?*      ��!       2	EGr��?EGr��?!EGr��?:	�<�r�]@�<�r�]@!�<�r�]@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�+�6��X@y|�^��?�"5
sequential/dense/MatMulMatMulg���#�?!g���#�?0"C
%gradient_tape/sequential/dense/MatMulMatMul������?!�}hfl�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul�ƁIl��?!�`)�"�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�ƁIl��?!"�Wl��?"7
sequential/dense_1/MatMulMatMul�ƁIl��?!��Uj���?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul98�[ǘ�?!���U���?"7
sequential/dense_2/MatMulMatMul98�[ǘ�?!�!NA���?0"!
Adam/PowPow�^F ���?!�v��2�?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad�ƁIl��?!;�C׽m�?"E
'gradient_tape/sequential/dense_2/MatMulMatMul��ȿ[7�?!�@�3��?0Q      Y@Y7��Moz2@a���,daT@q?�DG�jX@y�$�:�?"�
both�Your program is POTENTIALLY input-bound because 87.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�11.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 