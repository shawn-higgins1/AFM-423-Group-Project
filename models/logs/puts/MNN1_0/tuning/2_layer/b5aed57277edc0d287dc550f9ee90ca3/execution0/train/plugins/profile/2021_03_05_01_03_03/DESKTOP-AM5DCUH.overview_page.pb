�	���g?z7@���g?z7@!���g?z7@	����|�q?����|�q?!����|�q?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6���g?z7@*S�A��4@1��qQ-��?A&P6�
�?I��4�Ry�?Y������P?*	������H@2U
Iterator::Model::ParallelMapV2�!��u��?!�]�ڕ�<@)�!��u��?1�]�ڕ�<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�~j�t��?!dp>�c8@)��ׁsF�?1 ��184@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�~j�t��?!dp>�c8@)���Q�~?1}���|.@:Preprocessing2F
Iterator::Model�0�*��?!�JԮDmD@)�~j�t�x?1dp>�c(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice;�O��nr?!KԮD�J"@);�O��nr?1KԮD�J"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip\ A�c̝?!�+Q��M@)���_vOn?1M!�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!�+Q�@)�J�4a?1�+Q�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapS�!�uq�?!<�œ[<;@)Ǻ���V?1�F:l��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9����|�q?I���=�iX@Qʙ����@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	*S�A��4@*S�A��4@!*S�A��4@      ��!       "	��qQ-��?��qQ-��?!��qQ-��?*      ��!       2	&P6�
�?&P6�
�?!&P6�
�?:	��4�Ry�?��4�Ry�?!��4�Ry�?B      ��!       J	������P?������P?!������P?R      ��!       Z	������P?������P?!������P?b      ��!       JGPUY����|�q?b q���=�iX@yʙ����@�"5
sequential/dense/MatMulMatMul�@����?!�@����?0"E
'gradient_tape/sequential/dense_1/MatMulMatMulf��y���?!.��'@Ҳ?0"C
%gradient_tape/sequential/dense/MatMulMatMulj����ؠ?!c)��>�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul��/x{�?!�����N�?"7
sequential/dense_1/MatMulMatMul��/x{�?!�ʐ�3��?0"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad�����?!P�!t��?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�����?! ��
�A�?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdamj����ؐ?!���]�?"!
Adam/PowPowj����ؐ?!kg�,x�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchj����ؐ?!��ӣI�?Q      Y@Y>����/@aX�i��U@qFXtW@y愩��R�?"�
both�Your program is POTENTIALLY input-bound because 89.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�8.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�93.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 