�	Y�8��9@Y�8��9@!Y�8��9@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-Y�8��9@���h��5@1�M�#~��?A�Q�|�?I9�⪲�@*	33333�M@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::ConcatenateM�St$�?!xm��C@)��A�f�?1{	�%��A@:Preprocessing2U
Iterator::Model::ParallelMapV2��H�}�?!>���>8@)��H�}�?1>���>8@:Preprocessing2F
Iterator::Model��ͪ�Ֆ?!�j�c�B@)vq�-�?1��wm�*@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�� �rh�?!��wm��,@)�~j�t�x?1�ΐ��3$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��\m���?!���:O@)����Mbp?1(iv��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{�G�zd?!�Ω��@){�G�zd?1�Ω��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�(��0�?!
����D@)����Mb`?1(iv��
@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor����MbP?!(iv���?)����MbP?1(iv���?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceǺ���F?!iv����?)Ǻ���F?1iv����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIж1���X@Q>�$g���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���h��5@���h��5@!���h��5@      ��!       "	�M�#~��?�M�#~��?!�M�#~��?*      ��!       2	�Q�|�?�Q�|�?!�Q�|�?:	9�⪲�@9�⪲�@!9�⪲�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qж1���X@y>�$g���?�"C
%gradient_tape/sequential/dense/MatMulMatMulF�ԩ���?!F�ԩ���?0"5
sequential/dense/MatMulMatMulF�ԩ���?!F�ԩ���?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�I%�;�?!���Ӽ?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�I%�;�?!��3K��?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam�:�z��?!�:�z��?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�:�z��?!F�ԩ���?"7
sequential/dense_1/MatMulMatMul�:�z��?!�I%�;�?0"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam]X�7��?!)������?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdam]X�7��?!�����?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam]X�7��?!5z����?Q      Y@Y{	�%��1@a�����T@q�6i���W@y
�nϊU�?"�
both�Your program is POTENTIALLY input-bound because 87.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�11.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�94.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 