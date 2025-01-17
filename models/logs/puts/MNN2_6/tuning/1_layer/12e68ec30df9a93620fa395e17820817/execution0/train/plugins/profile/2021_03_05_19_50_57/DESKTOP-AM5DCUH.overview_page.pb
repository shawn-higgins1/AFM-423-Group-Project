�	�ފ��5@�ފ��5@!�ފ��5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�ފ��5@g���p3@1נ/����?A����z�?I)狽_@*	�����F@2U
Iterator::Model::ParallelMapV2��ZӼ�?!;�;�7@)��ZӼ�?1;�;�7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatg��j+��?!yxxxxx:@)��~j�t�?10C~/C~5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�g��s��?!�]��]�7@)�ZӼ�}?1�7�70@:Preprocessing2F
Iterator::ModelX�5�;N�?!C@)S�!�uq{?1QQ.@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�#�����?!������N@)���_vOn?1���� @:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey�&1�l?!瘬瘬@)y�&1�l?1瘬瘬@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!$��#��@)/n��b?1$��#��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaptF��_�?!�N��N�:@)��_�LU?1������@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI���ۆ�X@Q�	 %���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	g���p3@g���p3@!g���p3@      ��!       "	נ/����?נ/����?!נ/����?*      ��!       2	����z�?����z�?!����z�?:	)狽_@)狽_@!)狽_@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q���ۆ�X@y�	 %���?�"5
sequential/dense/MatMulMatMul�bWu�?!�bWu�?0"C
%gradient_tape/sequential/dense/MatMulMatMul��E�j�?!ܥ�8��?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�bWu�?!�~$z9�?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�bWu�?!�+�k���?"7
sequential/dense_1/MatMulMatMul�bWu�?!
h��?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�35R{�?!��N^���?"!
Adam/PowPow��E�j�?!�{׶UI�?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdamۡ�]%�?!��9o��?"[
AArithmeticOptimizer/ReorderCastLikeAndValuePreserving_double_CastCast�e��"$�?!1�u�C��?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam&�h&�!�?!:n�.9�?Q      Y@Y��/Ċ�0@a�	�N]�T@qʀJ�{X@y�;��c�?"�
both�Your program is POTENTIALLY input-bound because 87.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�11.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 