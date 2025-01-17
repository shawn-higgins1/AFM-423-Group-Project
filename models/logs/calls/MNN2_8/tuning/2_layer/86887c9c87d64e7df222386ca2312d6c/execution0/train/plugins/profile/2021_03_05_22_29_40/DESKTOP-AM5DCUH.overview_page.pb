�	�cyW=�7@�cyW=�7@!�cyW=�7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�cyW=�7@g���4@1�&�|��?Ao�ŏ1�?I ���7@*	������L@2U
Iterator::Model::ParallelMapV2�Q���?!�7��Mo>@)�Q���?1�7��Mo>@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate;�O��n�?!��7��M?@)�!��u��?1�B���8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�I+��?!���,d!3@)�&S��?1{ӛ���/@:Preprocessing2F
Iterator::Model�b�=y�?!-d!Y�D@)-C��6z?1d!Y�B&@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�q����o?!���,d!@)�q����o?1���,d!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�J�4�?!ԛ���7M@)F%u�k?1ozӛ��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!�7��Mo
@)ŏ1w-!_?1�7��Mo
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�N@aÓ?!,d!Y�@@)��_�LU?1"Y�B@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIv)�)�X@Q����u��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	g���4@g���4@!g���4@      ��!       "	�&�|��?�&�|��?!�&�|��?*      ��!       2	o�ŏ1�?o�ŏ1�?!o�ŏ1�?:	 ���7@ ���7@! ���7@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qv)�)�X@y����u��?�"5
sequential/dense/MatMulMatMul�V�08q�?!�V�08q�?0"C
%gradient_tape/sequential/dense/MatMulMatMul����G�?!#��w\�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul����G�?!� !)S �?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul����G�?!�u�X��?"7
sequential/dense_1/MatMulMatMul��Yo6�?!D74/��?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul��tP���?!A�BY
��?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch��դ��?!�K��>��?"7
sequential/dense_2/MatMulMatMul��դ��?!�ă�s9�?0"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad�����T�?!8��&r�?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam����G�?!��ڗ}��?Q      Y@Y>����/@aX�i��U@q;P�U�`X@y��D
 �?"�
both�Your program is POTENTIALLY input-bound because 88.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 