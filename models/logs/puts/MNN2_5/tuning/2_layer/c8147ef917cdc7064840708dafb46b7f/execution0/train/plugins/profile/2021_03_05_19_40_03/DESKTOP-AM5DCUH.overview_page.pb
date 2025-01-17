�	�@IU7@�@IU7@!�@IU7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�@IU7@�L�T�4@1���G���?A�3��7�?I̘�5�@*	     �F@2U
Iterator::Model::ParallelMapV2��@��ǈ?!�~�:@)��@��ǈ?1�~�:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�]K�=�?!�;�;=@)�+e�X�?1��9@:Preprocessing2F
Iterator::Model�ݓ��Z�?!O��N��D@)_�Q�{?1>��=��-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatea2U0*��?!�Q�Q5@)�~j�t�x?1��_��_*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��H�}m?!_��_��@)��H�}m?1_��_��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�]K�=�?!�;�;M@)��_vOf?1�{��{�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!@�@�@)ŏ1w-!_?1@�@�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�g��s��?!��K��K7@)����MbP?1R�Q�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIO�H���X@QP���Q��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�L�T�4@�L�T�4@!�L�T�4@      ��!       "	���G���?���G���?!���G���?*      ��!       2	�3��7�?�3��7�?!�3��7�?:	̘�5�@̘�5�@!̘�5�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qO�H���X@yP���Q��?�"5
sequential/dense/MatMulMatMul�.��<�?!�.��<�?0"C
%gradient_tape/sequential/dense/MatMulMatMul��Ć�?!a �?�*�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul��!���?!���=Q��?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�) ���?!�� �[��?"7
sequential/dense_1/MatMulMatMulfe;y��?!LT(.�V�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch m*{=_�?!졍ݺ�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul m*{=_�?!��򌢮�?"7
sequential/dense_2/MatMulMatMul m*{=_�?!,=X<�Z�?0"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad=�9���?!�y�l���?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam��!���?!��!���?Q      Y@Y>����/@aX�i��U@q�u�9�{X@y@����?"�
both�Your program is POTENTIALLY input-bound because 87.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 