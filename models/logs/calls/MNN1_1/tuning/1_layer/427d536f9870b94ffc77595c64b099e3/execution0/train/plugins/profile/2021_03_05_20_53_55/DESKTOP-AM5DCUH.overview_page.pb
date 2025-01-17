�	h@�5�5@h@�5�5@!h@�5�5@	�_-��_�?�_-��_�?!�_-��_�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6h@�5�5@���~�3@1��a�1�?A�f��j+�?I�B;�Y`�?Y��^zo�?*	����̌I@2U
Iterator::Model::ParallelMapV22U0*��?!�+W�\�>@)2U0*��?1�+W�\�>@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��<,Ԋ?!-Z�hѢ9@)U���N@�?1S�L�2e2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatǺ����?!�^�z��5@)U���N@�?1S�L�2e2@:Preprocessing2F
Iterator::ModelA��ǘ��?!�7nܸE@)9��v��z?1.\�p)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���_vOn?!gϞ={�@)���_vOn?1gϞ={�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��j+���?!s�ȑ#GL@)_�Q�k?1ԩS�N�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!�.@)��H�}]?1�.@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���QI�?!�����;@)a2U0*�S?1�,Y�d�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�_-��_�?I�+)T�X@Q��^��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���~�3@���~�3@!���~�3@      ��!       "	��a�1�?��a�1�?!��a�1�?*      ��!       2	�f��j+�?�f��j+�?!�f��j+�?:	�B;�Y`�?�B;�Y`�?!�B;�Y`�?B      ��!       J	��^zo�?��^zo�?!��^zo�?R      ��!       Z	��^zo�?��^zo�?!��^zo�?b      ��!       JGPUY�_-��_�?b q�+)T�X@y��^��?�"5
sequential/dense/MatMulMatMulP��Q�5�?!P��Q�5�?0"C
%gradient_tape/sequential/dense/MatMulMatMul7J�����?!��%D�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul���஘?!.��^�C�?"7
sequential/dense_1/MatMulMatMul�<���?!��в7�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�6�S07�?!�<�ژ>�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�6�S07�?!l���~E�?"
Abs_1Absk��l,��?!3|ëQ�?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdamk��l,��?!�t�r$��?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdamk��l,��?!�mY9���?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdamk��l,��?!�f$ ʬ�?Q      Y@Y��/Ċ�0@a�	�N]�T@q����=T@yD2�F��?"�
both�Your program is POTENTIALLY input-bound because 89.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�81.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 