�	�bԵ�6@�bԵ�6@!�bԵ�6@	w�b,��?w�b,��?!w�b,��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�bԵ�6@*� �hu3@1]~p>u�?A����K�?IMg'��@Y���8�~�?*	������L@2U
Iterator::Model::ParallelMapV2;�O��n�?!>����?@);�O��n�?1>����?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�Pk�w�?!�rO#,�7@)�~j�t��?1��=��4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatevq�-�?!����=;@)tF��_�?1�FX�i�4@:Preprocessing2F
Iterator::Model�b�=y�?!���{�D@)��0�*x?1j��FX$@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�q����o?!FX�i��@)�q����o?1FX�i��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�R�!�u�?!GX�i�eM@)�~j�t�h?1��=��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!sO#,�4
@)ŏ1w-!_?1sO#,�4
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap"��u���?!�{a��=@)Ǻ���V?1,�4�rO@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9w�b,��?I�f7��X@Q�����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	*� �hu3@*� �hu3@!*� �hu3@      ��!       "	]~p>u�?]~p>u�?!]~p>u�?*      ��!       2	����K�?����K�?!����K�?:	Mg'��@Mg'��@!Mg'��@B      ��!       J	���8�~�?���8�~�?!���8�~�?R      ��!       Z	���8�~�?���8�~�?!���8�~�?b      ��!       JGPUYw�b,��?b q�f7��X@y�����?�"5
sequential/dense/MatMulMatMul���M��?!���M��?0"C
%gradient_tape/sequential/dense/MatMulMatMul2���8��?!��c	�9�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch���M��?!E6bOV��?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul���M��?!�T��t��?"7
sequential/dense_1/MatMulMatMul���M��?!���m>��?0"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam@qaFב�?!��{VyS�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad@qaFב�?!��G?���?"E
'gradient_tape/sequential/dense_1/MatMulMatMul{�G�?�?!E�p<�m�?0"!
Adam/PowPowԢ��$ē?!�%�*��?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam�)���ڎ?!�_����?Q      Y@Y��/Ċ�0@a�	�N]�T@q�+��yX@yz �.��?"�
both�Your program is POTENTIALLY input-bound because 88.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�96.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 