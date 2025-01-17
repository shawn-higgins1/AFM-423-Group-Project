�	����	�1@����	�1@!����	�1@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-����	�1@|G�	1c0@1 C�*�?A�OU��X�?I31]��?*	gfffffG@2U
Iterator::Model::ParallelMapV2�
F%u�?!0#0#;@)�
F%u�?10#0#;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat_�Qڋ?!�P�P=@)�+e�X�?1��[��[8@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap46<�R�?!�dJ�dJ7@)���_vO~?1������/@:Preprocessing2F
Iterator::Modela2U0*��?!4H�4H�D@)9��v��z?1r�q�+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlicey�&1�l?!���@)y�&1�l?1���@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Ziplxz�,C�?!̷|˷|M@)a��+ei?1��~��~@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!۬�ڬ�@)/n��b?1۬�ڬ�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��ƾ�X@Q�"�� ��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	|G�	1c0@|G�	1c0@!|G�	1c0@      ��!       "	 C�*�? C�*�?! C�*�?*      ��!       2	�OU��X�?�OU��X�?!�OU��X�?:	31]��?31]��?!31]��?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��ƾ�X@y�"�� ��?�"5
sequential/dense/MatMulMatMulq�㲥��?!q�㲥��?0"C
%gradient_tape/sequential/dense/MatMulMatMul�k��]�?!���(��?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulh2
S/�?!��~�]�?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam�k��]�?!$���F��?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�k��]�?!����?"E
'gradient_tape/sequential/dense_1/MatMulMatMul�k��]�?! `|����?0"7
sequential/dense_1/MatMulMatMul�k��]�?!��\�}�?0"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdamzN���?!�{�Tߕ�?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam��Bwn�?!ƤS<F��?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdam��Bwn�?!���#���?Q      Y@Y�M�_{4@a��(�S@q�T٩w�X@y�X���?"�
both�Your program is POTENTIALLY input-bound because 92.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�6.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�99.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 