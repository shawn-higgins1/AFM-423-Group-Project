�	���J?�7@���J?�7@!���J?�7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-���J?�7@=~oӟ�4@1~�.rO�?A
ףp=
�?I�&���+@*	������H@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate46<��?!���iB@)�J�4�?1�H���@@:Preprocessing2U
Iterator::Model::ParallelMapV2�
F%u�?!#��_��9@)�
F%u�?1#��_��9@:Preprocessing2F
Iterator::Model+�����?!F��!�C@)_�Q�{?1��C<;]+@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat_�Q�{?!��C<;]+@)a2U0*�s?1@���P#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?W[���?!��aN@)_�Q�k?1��C<;]@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!`��n�@)����Mb`?1`��n�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap{�G�z�?!�:n��D@)_�Q�[?1��C<;]@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor��H�}M?!�W�M��?)��H�}M?1�W�M��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicea2U0*�C?!@���P�?)a2U0*�C?1@���P�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI9�᝽X@Q��ٻ���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	=~oӟ�4@=~oӟ�4@!=~oӟ�4@      ��!       "	~�.rO�?~�.rO�?!~�.rO�?*      ��!       2	
ףp=
�?
ףp=
�?!
ףp=
�?:	�&���+@�&���+@!�&���+@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q9�᝽X@y��ٻ���?�"5
sequential/dense/MatMulMatMulF���DJ�?!F���DJ�?0"C
%gradient_tape/sequential/dense/MatMulMatMul[,C�%�?!P�^g�7�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul[,C�%�?!���ʺ?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul[,C�%�?!�oEվ��?"7
sequential/dense_1/MatMulMatMul[,C�%�?!�sf#��?0"7
sequential/dense_2/MatMulMatMul�E-	p�?!kܳ�$��?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul2���n�?!Ѿ2fT�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchBP��&�?!Ǽ��x�?"!
Adam/PowPow[,C�%�?!�H"B���?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad[,C�%�?!u�CE��?Q      Y@Y�C=�C=0@a��
��T@q`�hz|X@y�����?"�
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
Refer to the TF2 Profiler FAQb�97.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 