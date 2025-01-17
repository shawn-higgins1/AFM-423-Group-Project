�	y�z�L�7@y�z�L�7@!y�z�L�7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-y�z�L�7@�G�?�4@1;R}�%�?A
ףp=
�?I��"j�/@*	53333�F@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate"��u���?!��E�V�B@)���H�?16��тA@:Preprocessing2U
Iterator::Model::ParallelMapV2Zd;�O��?!Y&
ݔT9@)Zd;�O��?1Y&
ݔT9@:Preprocessing2F
Iterator::Model�&S��?!��gGD@)S�!�uq{?1���w�-@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�+e�Xw?!b�1)@)����Mbp?1TRb�!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip_�Qڛ?!'?���M@)-C��6j?1Sb�1@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor_�Q�[?!'?���@)_�Q�[?1'?���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap46<��?!�"k�x'D@)/n��R?1)tSRb@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensorǺ���F?!���"k��?)Ǻ���F?1���"k��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicea2U0*�C?!���B7%�?)a2U0*�C?1���B7%�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI �'$�X@Q��=�6�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�G�?�4@�G�?�4@!�G�?�4@      ��!       "	;R}�%�?;R}�%�?!;R}�%�?*      ��!       2	
ףp=
�?
ףp=
�?!
ףp=
�?:	��"j�/@��"j�/@!��"j�/@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q �'$�X@y��=�6�?�"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulU�����?!U�����?"5
sequential/dense/MatMulMatMul *!6��?!�����¸?0"C
%gradient_tape/sequential/dense/MatMulMatMulǉ�9���?!ǉ�9���?0"E
'gradient_tape/sequential/dense_1/MatMulMatMulǉ�9���?!9�gu��?0"7
sequential/dense_1/MatMulMatMulǉ�9���?!�������?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch8�gu��?!2��wAW�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul8�gu��?!������?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdamǉ�9���?!�J: ���?"!
Adam/PowPowǉ�9���?!���?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGradǉ�9���?!��Zg&�?Q      Y@Y�C=�C=0@a��
��T@q�����X@yu1�MX�?"�
both�Your program is POTENTIALLY input-bound because 87.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�11.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�99.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 