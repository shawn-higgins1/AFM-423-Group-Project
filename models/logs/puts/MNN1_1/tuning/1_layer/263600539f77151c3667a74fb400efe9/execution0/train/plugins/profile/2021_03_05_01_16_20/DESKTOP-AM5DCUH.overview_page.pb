�	N�w(
�5@N�w(
�5@!N�w(
�5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-N�w(
�5@.Y�&�2@1��Li��?A�X�� �?Ib�k_@o@*	    @M@2U
Iterator::Model::ParallelMapV22�%䃎?!�Wx�Wx9@)2�%䃎?1�Wx�Wx9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate/n���?!^�^�>@)_�Qڋ?1�s?�s?7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatŏ1w-!�?!������9@)F%u��?1i�i�6@:Preprocessing2F
Iterator::Model^K�=��?!����B@)a��+ey?1%S2%S2%@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����Mbp?!��Y��Y@)����Mbp?1��Y��Y@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���&�?!s?�s?�O@)-C��6j?1^�^�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!��Y��Y@)����Mb`?1��Y��Y@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��~j�t�?!�C=�C=@@)Ǻ���V?1T2%S2%@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�6����X@Q�P2W@��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	.Y�&�2@.Y�&�2@!.Y�&�2@      ��!       "	��Li��?��Li��?!��Li��?*      ��!       2	�X�� �?�X�� �?!�X�� �?:	b�k_@o@b�k_@o@!b�k_@o@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�6����X@y�P2W@��?�"5
sequential/dense/MatMulMatMul��C��\�?!��C��\�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�x�Y:�?!#f���?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchH�)�ؘ?!$U�%U�?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�}���?!KzҬ�?"7
sequential/dense_1/MatMulMatMul�}���?!ʐ���?0"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam8ӭT���?!k�&�$+�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad8ӭT���?!�>�f8��?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdamռ��Ѝ?!����F��?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdamռ��Ѝ?!l֜�Ua�?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdamռ��Ѝ?!9"�d>�?Q      Y@Y��/Ċ�0@a�	�N]�T@q����sX@y��;6K�?"�
both�Your program is POTENTIALLY input-bound because 87.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 