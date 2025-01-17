�	��\��5@��\��5@!��\��5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��\��5@�2o�u3@1 �����?AZd;�O��?IT��7��@*	�����G@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate�Q���?!��|�B@)��ǘ���?1?!��O�A@:Preprocessing2U
Iterator::Model::ParallelMapV2-C��6�?!��n��;@)-C��6�?1��n��;@:Preprocessing2F
Iterator::Model�ݓ��Z�?!u�E]tD@)�HP�x?1��9T,h*@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatǺ���v?!>���>(@)��H�}m?1���8+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�#�����?!�.�袋M@)�����g?1�n��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!�Cł�P@)����Mb`?1�Cł�P@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapU���N@�?!�bAs�XD@)��_�LU?1�Cł@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor-C��6J?!��n���?)-C��6J?1��n���?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice����Mb@?!�Cł�P�?)����Mb@?1�Cł�P�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��h��X@Q�8���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�2o�u3@�2o�u3@!�2o�u3@      ��!       "	 �����? �����?! �����?*      ��!       2	Zd;�O��?Zd;�O��?!Zd;�O��?:	T��7��@T��7��@!T��7��@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��h��X@y�8���?�"5
sequential/dense/MatMulMatMul.K��v�?!.K��v�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�UrYp��?!��ރ�7�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�ݨ����?!V�H�u�?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�ݨ����?!f�Y�?��?"7
sequential/dense_1/MatMulMatMul�ݨ����?!!������?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�q�<V�?!^�(R��?"E
'gradient_tape/sequential/dense_1/MatMulMatMulF~����?!'��e���?0"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdami=d��?!����`�?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdami=d��?!���*@�?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdami=d��?!�]f�?Q      Y@Y{	�%��1@a�����T@q}�wxX@yM�53�?"�
both�Your program is POTENTIALLY input-bound because 88.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 