�	�}"�3@�}"�3@!�}"�3@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�}"�3@ ��L 2@1��A{���?A�H�}�?I�r��o�?*	33333�G@2U
Iterator::Model::ParallelMapV2�?�߾�?!z�[�'�<@)�?�߾�?1z�[�'�<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�
F%u�?!���\�:@)'�����?1�[�'�6@:Preprocessing2F
Iterator::Model䃞ͪϕ?!�jq�wF@)ŏ1w-!?1g�/�0@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapM�O��?!I�1�N5@)F%u�{?1�g *�+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlicey�&1�l?!�ur.�@)y�&1�l?1�ur.�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip}гY���?!����K@)F%u�k?1�g *�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!l��Ӭ�@)����Mb`?1l��Ӭ�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�5.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��u��X@Q[՜b���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	 ��L 2@ ��L 2@! ��L 2@      ��!       "	��A{���?��A{���?!��A{���?*      ��!       2	�H�}�?�H�}�?!�H�}�?:	�r��o�?�r��o�?!�r��o�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��u��X@y[՜b���?�"5
sequential/dense/MatMulMatMula,��צ�?!a,��צ�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�V�by��?!��-�(�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul�V�by��?!r�K�r��?0"7
sequential/dense_1/MatMulMatMul�V�by��?!� ����?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulxDbn��?!����,K�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMula,��צ�?!I����?"7
sequential/dense_2/MatMulMatMula,��צ�?!��9@qZ�?0"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad�V�by��?!Vg�Ȣ�?"E
'gradient_tape/sequential/dense_2/MatMulMatMul�V�by��?!�6�l ��?0"K
$Adam/Adam/update_4/ResourceApplyAdamResourceApplyAdamY	H}c3�?!W�h�V.�?Q      Y@Y7��Moz2@a���,daT@qD��2(uX@y���?j�?"�
both�Your program is POTENTIALLY input-bound because 92.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�5.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 