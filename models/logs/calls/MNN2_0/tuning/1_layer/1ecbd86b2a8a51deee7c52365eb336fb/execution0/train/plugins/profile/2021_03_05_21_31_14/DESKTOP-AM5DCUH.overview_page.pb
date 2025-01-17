�	H0�[�5@H0�[�5@!H0�[�5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-H0�[�5@����i>3@1}$%=��?A
pUj�?I��a�p�?*	333333J@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate+�����?!����[�B@)�:pΈ�?1�4w�_EA@:Preprocessing2U
Iterator::Model::ParallelMapV2�St$���?!}@u�?@)�St$���?1}@u�?@:Preprocessing2F
Iterator::ModelHP�s�?!'���E@) �o_�y?1�[��(@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatF%u�{?!T���0)@)�J�4q?1��g� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Ziph��|?5�?!���[&L@){�G�zd?1��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*�c?!�R@)a2U0*�c?1�R@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��ZӼ�?!{k�4wC@)��H�}M?1fDP{�?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor��H�}M?!fDP{�?)��H�}M?1fDP{�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice����Mb@?!ㄔ<ˈ�?)����Mb@?1ㄔ<ˈ�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI���l�X@QT�CФ�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	����i>3@����i>3@!����i>3@      ��!       "	}$%=��?}$%=��?!}$%=��?*      ��!       2	
pUj�?
pUj�?!
pUj�?:	��a�p�?��a�p�?!��a�p�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q���l�X@yT�CФ�?�"5
sequential/dense/MatMulMatMulr�y��d�?!r�y��d�?0"C
%gradient_tape/sequential/dense/MatMulMatMul��O���?!��d�&�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchb���s�?!R�ͩe^�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGradb���s�?!�xKa��?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulb���s�?!��O����?"7
sequential/dense_1/MatMulMatMulb���s�?!-v�7��?0"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam��O���?!u�/���?"!
Adam/PowPow��O���?!t�'���?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdamB���Wۍ?!A���W��?"[
AArithmeticOptimizer/ReorderCastLikeAndValuePreserving_double_CastCastB���Wۍ?!ur��?Q      Y@Y{	�%��1@a�����T@q��s&&�X@y�����?"�
both�Your program is POTENTIALLY input-bound because 88.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�99.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 