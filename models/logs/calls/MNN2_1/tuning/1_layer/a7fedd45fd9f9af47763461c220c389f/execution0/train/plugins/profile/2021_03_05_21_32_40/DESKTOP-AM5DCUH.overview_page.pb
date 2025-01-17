�	fKVE�6@fKVE�6@!fKVE�6@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-fKVE�6@�t"��3@1�U��Ά�?AZd;�O��?I�����@*	23333�K@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate�I+��?!�TQ�C@)�0�*��?1+�$B@:Preprocessing2U
Iterator::Model::ParallelMapV2�������?!�4k\,�6@)�������?1�4k\,�6@:Preprocessing2F
Iterator::Model�N@aÓ?!�N��4kA@)_�Q�{?1��&�y�(@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat	�^)ˀ?!)G��V�-@)��0�*x?1$�L%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip]m���{�?!�X �eJP@)"��u��q?1eD^�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP�s�b?!`8wC�@)HP�s�b?1`8wC�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��0�*�?!$�LE@)-C��6Z?1��o+�@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor��_�LU?!��J���@)��_�LU?1��J���@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicea2U0*�C?!�ޓ�T�?)a2U0*�C?1�ޓ�T�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIR�=��X@Q;W a��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�t"��3@�t"��3@!�t"��3@      ��!       "	�U��Ά�?�U��Ά�?!�U��Ά�?*      ��!       2	Zd;�O��?Zd;�O��?!Zd;�O��?:	�����@�����@!�����@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qR�=��X@y;W a��?�"5
sequential/dense/MatMulMatMul.�z��Ħ?!.�z��Ħ?0"C
%gradient_tape/sequential/dense/MatMulMatMul��+-=�?!O�����?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�w�H�M�?!	m�oԻ?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMull��uxL�?!Ro����?"7
sequential/dense_1/MatMulMatMul%g1����?!7��73�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch��+-=�?!u�?�ܺ�?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam�S����?!� ���=�?"=
$mean_squared_error/weighted_loss/SumSum�IfK^�?!̲?�#�?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam躣��[�?!z� {<	�?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam躣��[�?!(*+����?Q      Y@Y{	�%��1@a�����T@qt�N�lwX@y8U����?"�
both�Your program is POTENTIALLY input-bound because 88.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 