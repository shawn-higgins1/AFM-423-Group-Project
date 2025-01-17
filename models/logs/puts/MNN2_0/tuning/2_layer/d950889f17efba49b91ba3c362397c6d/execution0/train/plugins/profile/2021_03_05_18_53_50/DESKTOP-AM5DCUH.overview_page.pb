�	�ᱟ��6@�ᱟ��6@!�ᱟ��6@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�ᱟ��6@�}�֤s4@1�(�[Z�?A��&��?I@��߼x@*����̌H@)       =2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate�0�*��?!E�%�wD@)HP�sג?1ֲ�Ƚ�B@:Preprocessing2U
Iterator::Model::ParallelMapV2�HP��?!b�h��8@)�HP��?1b�h��8@:Preprocessing2F
Iterator::Modelr�����?!��ʱAB@)�I+�v?1Y���;g&@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatS�!�uq{?!�A܎J+@);�O��nr?1�2�vT"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip2U0*��?!h.5N��O@)_�Q�k?1���ղ@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!Gh��/�@)/n��b?1Gh��/�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap䃞ͪϕ?!��޿�E@)a2U0*�S?1d�ΙK�@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor����MbP?!)ӁUK @)����MbP?1)ӁUK @:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceǺ���F?!�'ނ��?)Ǻ���F?1�'ނ��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�k̨X@Q���<���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�}�֤s4@�}�֤s4@!�}�֤s4@      ��!       "	�(�[Z�?�(�[Z�?!�(�[Z�?*      ��!       2	��&��?��&��?!��&��?:	@��߼x@@��߼x@!@��߼x@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�k̨X@y���<���?�"5
sequential/dense/MatMulMatMul�%U�f�?!�%U�f�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�aH|w��?!��N8��?"7
sequential/dense_1/MatMulMatMul�aH|w��?!��r����?0"C
%gradient_tape/sequential/dense/MatMulMatMul���y̠?!�\��sv�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul?�~��?!�1��U.�?0"7
sequential/dense_2/MatMulMatMul�>�Re@�?!�y8�b��?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchQzZ[U?�?!ɣ9M~�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMulQzZ[U?�?!_�7&�?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam�aH|w��?!�$��&F�?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad�aH|w��?!k��
��?Q      Y@Y�C=�C=0@a��
��T@q�ʵX@y�1�*��?"�
both�Your program is POTENTIALLY input-bound because 88.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�98.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 