�	P0�A�1@P0�A�1@!P0�A�1@	S�t�ww?S�t�ww?!S�t�ww?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6P0�A�1@�lu9%`0@1�};���?A��A�f�?IK�b��?Y������P?*������F@)       =2U
Iterator::Model::ParallelMapV2�Pk�w�?!���>�{>@)�Pk�w�?1���>�{>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�~j�t��?!�5��P:@)n���?1�Gp�}5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap46<�R�?!u�YL�7@)y�&1�|?1;��,��.@:Preprocessing2F
Iterator::Modeln���?!�Gp�}E@)�+e�Xw?1      )@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�q����o?!�����!@)�q����o?1�����!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip9��v���?!$���>�L@)a��+ei?1�YLg1@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!�t�YL@)/n��b?1�t�YL@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 91.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�5.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9S�t�ww?I����sX@Q>%k�{@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�lu9%`0@�lu9%`0@!�lu9%`0@      ��!       "	�};���?�};���?!�};���?*      ��!       2	��A�f�?��A�f�?!��A�f�?:	K�b��?K�b��?!K�b��?B      ��!       J	������P?������P?!������P?R      ��!       Z	������P?������P?!������P?b      ��!       JGPUYS�t�ww?b q����sX@y>%k�{@�"5
sequential/dense/MatMulMatMulɪ:�9�?!ɪ:�9�?0"C
%gradient_tape/sequential/dense/MatMulMatMul����-�?!���fN3�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulyfVD4"�?!yfVD4"�?"7
sequential/dense_1/MatMulMatMulQ렄:�?!c�s؄��?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrads� �9G�?!�l��?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam����-�?!%�Lp��?"S
2gradient_tape/sequential/dense/BiasAdd/BiasAddGradBiasAddGrad9"��w�?!lW}h�y�?"
Abs_1AbsyfVD4"�?!�x���?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdamyfVD4"�?!�xɼ�?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdamyfVD4"�?!��1#�?Q      Y@Y�M�_{4@a��(�S@q�"k�x'U@y������?"�
both�Your program is POTENTIALLY input-bound because 91.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�5.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�84.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 