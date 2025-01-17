�	ҧU�J7@ҧU�J7@!ҧU�J7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-ҧU�J7@��C��4@1q��#�?A��'�8�?IYLl>�� @*	������J@2U
Iterator::Model::ParallelMapV2��<,Ԋ?!�o�e8@)��<,Ԋ?1�o�e8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���Q��?!��o���;@)�{�Pk�?1���}�8@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate_�Qڋ?!f�a+mS9@)a2U0*��?1���K��1@:Preprocessing2F
Iterator::Modeln���?!Q�`}�?B@)9��v��z?1-�d5(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�R�!�u�?!�9���O@)"��u��q?1�5�S� @:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����Mbp?!K�T~��@)����Mbp?1K�T~��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!B�D�H@)�J�4a?1B�D�H@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���_vO�?!�J�T~�;@)a2U0*�S?1���K��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�%�Gy�X@Q ��
�!�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��C��4@��C��4@!��C��4@      ��!       "	q��#�?q��#�?!q��#�?*      ��!       2	��'�8�?��'�8�?!��'�8�?:	YLl>�� @YLl>�� @!YLl>�� @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�%�Gy�X@y ��
�!�?�"5
sequential/dense/MatMulMatMul^S� ��?!^S� ��?0"C
%gradient_tape/sequential/dense/MatMulMatMul���Wğ�?!R����ɲ?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul��k٠?!(���6�?"E
'gradient_tape/sequential/dense_1/MatMulMatMul��D�ؠ?!�>z���?0"7
sequential/dense_1/MatMulMatMul��D�ؠ?!�6�.��?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul+����?!_j��'�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch?��U�?!�n'4���?"7
sequential/dense_2/MatMulMatMul?��U�?!�����k�?0"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam��D�ؐ?!#�3pC�?"!
Adam/PowPow��D�ؐ?!1�	x�P�?Q      Y@Y>����/@aX�i��U@q��Iz}jX@yϐe��R�?"�
both�Your program is POTENTIALLY input-bound because 89.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 