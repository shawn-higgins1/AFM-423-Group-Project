�	;�%8��3@;�%8��3@!;�%8��3@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-;�%8��3@Lݕ]02@1�/L�
F�?A�a��4�?I�'�XQC�?*	     @J@2U
Iterator::Model::ParallelMapV2�St$���?!�y��y�?@)�St$���?1�y��y�?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap2�%䃎?!�a�a<@)'�����?1�a�a4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��0�*�?!��y��y6@)��~j�t�?1�a�a2@:Preprocessing2F
Iterator::Model�z6�>�?!�y��y�E@)�HP�x?1��<��<'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�J�4q?!       @)�J�4q?1       @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip2�%䃞?!�a�aL@)a��+ei?1�y��y�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP�s�b?!b�a�@)HP�s�b?1b�a�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��@ǬX@Q$Jy�/��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	Lݕ]02@Lݕ]02@!Lݕ]02@      ��!       "	�/L�
F�?�/L�
F�?!�/L�
F�?*      ��!       2	�a��4�?�a��4�?!�a��4�?:	�'�XQC�?�'�XQC�?!�'�XQC�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��@ǬX@y$Jy�/��?�"5
sequential/dense/MatMulMatMulOl��VV�?!Ol��VV�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul�K%.�?!���,B�?0"C
%gradient_tape/sequential/dense/MatMulMatMul��ۥjڣ?!��D&��?0"7
sequential/dense_1/MatMulMatMul��ۥjڣ?!�޻�K�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul<L Q_�?!	�$f�?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad�ROј?!�H��3��?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�ROј?!���wT��?"7
sequential/dense_2/MatMulMatMul�ROј?!'{�:Z�?0"!
Adam/PowPow��ۥjړ?!t7?;��?"E
'gradient_tape/sequential/dense_2/MatMulMatMul��ۥjړ?!�����?0Q      Y@Y7��Moz2@a���,daT@q�w�	mgX@y�c'�F�?"�
both�Your program is POTENTIALLY input-bound because 92.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�6.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 