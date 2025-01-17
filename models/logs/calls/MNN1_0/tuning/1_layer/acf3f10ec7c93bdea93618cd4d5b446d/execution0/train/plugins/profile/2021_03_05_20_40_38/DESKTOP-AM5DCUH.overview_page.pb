�	f����1@f����1@!f����1@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-f����1@��-�k0@1,g*�?A�J�4�?I�I�UX�?*	������K@2U
Iterator::Model::ParallelMapV22U0*��?!e4��/<@)2U0*��?1e4��/<@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap� �	��?!����;@)Zd;�O��?1�"3u�4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��<,Ԋ?!I�kӄ7@)�I+��?12d~���3@:Preprocessing2F
Iterator::ModelǺ���?!}����D@)S�!�uq{?1����(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��镲�?!�Owl�M@)a2U0*�s?1"NnP5<!@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�q����o?!�>���@)�q����o?1�>���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!��L])@)�J�4a?1��L])@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�5.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�s�X@Q�:�>��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��-�k0@��-�k0@!��-�k0@      ��!       "	,g*�?,g*�?!,g*�?*      ��!       2	�J�4�?�J�4�?!�J�4�?:	�I�UX�?�I�UX�?!�I�UX�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�s�X@y�:�>��?�"5
sequential/dense/MatMulMatMulJ���⨬?!J���⨬?0"C
%gradient_tape/sequential/dense/MatMulMatMul߆6b�)�?!)�2i�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul��g؟?!�7}V�/�?"7
sequential/dense_1/MatMulMatMul��g؟?!:[wc�*�?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�{��y�?!��r�Y�?"
Abs_1Abs����A�?!P&oB=��?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam����A�?!�k}� �?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam����A�?!�4�B�?"!
Adam/PowPow����A�?!�L���s�?"#
Adam/addAddV2����A�?!`�0o��?Q      Y@Y�M�_{4@a��(�S@q�|<��kX@y��"���?"�
both�Your program is POTENTIALLY input-bound because 92.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�5.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 