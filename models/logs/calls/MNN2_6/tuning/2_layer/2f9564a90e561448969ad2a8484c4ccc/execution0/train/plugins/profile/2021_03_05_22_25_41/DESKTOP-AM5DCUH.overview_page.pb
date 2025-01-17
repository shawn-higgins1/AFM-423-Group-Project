�	|�y�3@|�y�3@!|�y�3@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-|�y�3@(ђ��2@1G!ɬ���?AB`��"۩?Ix^*6�5�?*	fffff�K@2U
Iterator::Model::ParallelMapV2r�����?!��n�?@)r�����?1��n�?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�o_��?!��/��=@)F%u��?1��~��7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatZd;�O��?!�ojT��4@)U���N@�?1.b3x��0@:Preprocessing2F
Iterator::Model=�U����?!CaھP�E@)-C��6z?1��[�f�&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���~�:�?!��%A�gL@)y�&1�l?1�[�f�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlicey�&1�l?!�[�f�@)y�&1�l?1�[�f�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!n��@)�J�4a?1n��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 91.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�7.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�/��X@Q�t�K�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	(ђ��2@(ђ��2@!(ђ��2@      ��!       "	G!ɬ���?G!ɬ���?!G!ɬ���?*      ��!       2	B`��"۩?B`��"۩?!B`��"۩?:	x^*6�5�?x^*6�5�?!x^*6�5�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�/��X@y�t�K�?�"5
sequential/dense/MatMulMatMul���Qx�?!���Qx�?0"C
%gradient_tape/sequential/dense/MatMulMatMul���w�?!
�L�x�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul��:���?!Kfǀ(:�?"7
sequential/dense_1/MatMulMatMul��:���?!NhP8�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMulzI���?!�zvy1�?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul��$����?!�%�/RP�?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam��:���?!����e��?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad��:���?!� �<'�?"E
'gradient_tape/sequential/dense_2/MatMulMatMul��:���?!�@���f�?0"7
sequential/dense_2/MatMulMatMul��:���?!�zp�P��?0Q      Y@Y7��Moz2@a���,daT@q�=r@kX@yr>�lYg�?"�
both�Your program is POTENTIALLY input-bound because 91.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�7.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 