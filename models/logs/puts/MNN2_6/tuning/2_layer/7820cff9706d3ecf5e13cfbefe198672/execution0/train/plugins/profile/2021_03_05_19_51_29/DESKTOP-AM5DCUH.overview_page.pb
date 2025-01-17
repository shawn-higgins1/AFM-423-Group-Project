�	�;����6@�;����6@!�;����6@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�;����6@�<�v4@1�&�fe�?AHP�s�?I��0a4�@*	�����YM@2U
Iterator::Model::ParallelMapV2��y�):�?!����R>@)��y�):�?1����R>@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate"��u���?!���G1M=@)_�Qڋ?1x���,+7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatF%u��?!"9W��|6@)�I+��?1GZ�6F�2@:Preprocessing2F
Iterator::Model0*��D�?!�d��/D@)��0�*x?1�6F�*$@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��H�}m?!%��q�@)��H�}m?1%��q�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�Q���?!B�T��M@)a��+ei?1u\"�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!�����@)/n��b?1�����@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���&�?!F���]�?@)�~j�t�X?1���cq@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��+��X@Q��u��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�<�v4@�<�v4@!�<�v4@      ��!       "	�&�fe�?�&�fe�?!�&�fe�?*      ��!       2	HP�s�?HP�s�?!HP�s�?:	��0a4�@��0a4�@!��0a4�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��+��X@y��u��?�"5
sequential/dense/MatMulMatMulf�8z
�?!f�8z
�?0"C
%gradient_tape/sequential/dense/MatMulMatMul���?!*y`�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul���?!!�վў�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul���?!g���?"7
sequential/dense_1/MatMulMatMul���?!��=3\�?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul� ���_�?!%�D{2�?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad��{o`��?!�P4�^��?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch���?!y�� ���?"7
sequential/dense_2/MatMulMatMul���?!��b����?0"K
$Adam/Adam/update_4/ResourceApplyAdamResourceApplyAdamO��Ȑ�?!`��]���?Q      Y@Y>����/@aX�i��U@q�H�M�hX@y@�$����?"�
both�Your program is POTENTIALLY input-bound because 89.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 