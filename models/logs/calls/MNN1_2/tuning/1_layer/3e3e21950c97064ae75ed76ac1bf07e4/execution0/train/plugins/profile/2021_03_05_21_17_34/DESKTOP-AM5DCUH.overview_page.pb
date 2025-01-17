�	(a��_�5@(a��_�5@!(a��_�5@	��W
ɾ�?��W
ɾ�?!��W
ɾ�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6(a��_�5@E�*k�J3@1Z��c!:�?A�,C��?I�J�(�@Y�_�n�?*	ffffffI@2U
Iterator::Model::ParallelMapV2K�=�U�?!���x<>@)K�=�U�?1���x<>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat9��v���?!�r�\.�9@)A��ǘ��?16��f��5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateA��ǘ��?!6��f��5@)ŏ1w-!?1|�^���-@:Preprocessing2F
Iterator::ModelA��ǘ��?!6��f��E@)lxz�,C|?1K�R�T*+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey�&1�l?!��x<�@)y�&1�l?1��x<�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���QI�?!�d2�L&L@)a��+ei?1�F��h@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!|�^���@)ŏ1w-!_?1|�^���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�������?!g��l6�8@)Ǻ���V?1��`0@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��W
ɾ�?I��,k�X@Q��s���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	E�*k�J3@E�*k�J3@!E�*k�J3@      ��!       "	Z��c!:�?Z��c!:�?!Z��c!:�?*      ��!       2	�,C��?�,C��?!�,C��?:	�J�(�@�J�(�@!�J�(�@B      ��!       J	�_�n�?�_�n�?!�_�n�?R      ��!       Z	�_�n�?�_�n�?!�_�n�?b      ��!       JGPUY��W
ɾ�?b q��,k�X@y��s���?�"5
sequential/dense/MatMulMatMulSޭ�c�?!Sޭ�c�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�~�Xx�?!��V޿�?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad����n��?!ez%�?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul����n��?!�f�E�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch|�JJ~m�?!n�K�:s�?"!
Adam/PowPow�~�Xx�?!�~{�E�?"7
sequential/dense_1/MatMulMatMul�~�Xx�?!*B��P��?0"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam�1����?!`Ȭy��?"
Abs_1Abso(=.���?!皐����?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdamo(=.���?!nmt8��?Q      Y@Y��/Ċ�0@a�	�N]�T@q��R��
V@yEI�'��?"�
both�Your program is POTENTIALLY input-bound because 88.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�88.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 