�	�W���~9@�W���~9@!�W���~9@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�W���~9@Z�wg�6@1v��y�]�?A6�;Nё�?I�ǚ�A.@*	�����G@2U
Iterator::Model::ParallelMapV2�{�Pk�?!��@���;@)�{�Pk�?1��@���;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�HP��?!�Y�v:@)��_�L�?1ШjE�6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate��_vO�?!�`�em7@)lxz�,C|?1�3/W�-@:Preprocessing2F
Iterator::Model��~j�t�?!���|�D@)�HP�x?1�Y�v*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice�q����o?!�~�t� @)�q����o?1�~�t� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��6��?!U��dM@)�~j�t�h?1�LN?�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!\�*�<@)��H�}]?1\�*�<@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaptF��_�?!fr�1>�9@)/n��R?1�Ӕ�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI'�lo�X@Q=��d?��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	Z�wg�6@Z�wg�6@!Z�wg�6@      ��!       "	v��y�]�?v��y�]�?!v��y�]�?*      ��!       2	6�;Nё�?6�;Nё�?!6�;Nё�?:	�ǚ�A.@�ǚ�A.@!�ǚ�A.@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q'�lo�X@y=��d?��?�"5
sequential/dense/MatMulMatMul&���u�?!&���u�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�����0�?!�hpOVS�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul�X �L�?!`������?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�4ˬE�?!��4	���?"7
sequential/dense_1/MatMulMatMul�4ˬE�?!jy��Q��?0"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad�A�l{��?!��-AB�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�A�l{��?!�I�}0��?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam̚B�bL�?!$�hռ�?"[
AArithmeticOptimizer/ReorderCastLikeAndValuePreserving_double_CastCast̚B�bL�?!~�-II�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch̚B�bL�?!�\�j��?Q      Y@Y>����/@aX�i��U@q�iA���X@yhʵ�E�?"�
both�Your program is POTENTIALLY input-bound because 89.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�8.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�99.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 