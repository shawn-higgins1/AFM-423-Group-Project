�	��E��5@��E��5@!��E��5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��E��5@:d�w73@1*�Z^���?A9��m4��?I�,AF@E @*	�����LL@2U
Iterator::Model::ParallelMapV2/n���?!��l?@)/n���?1��l?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate2U0*��?!��`�;@)tF��_�?1
/$U5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatg��j+��?!3�Co߫4@)n���?16��B�P1@:Preprocessing2F
Iterator::Model�HP��?!M�tm�E@)_�Q�{?10��<(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceŏ1w-!o?!�Ǚb��@)ŏ1w-!o?1�Ǚb��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?�ܵ�|�?!�N���qL@)_�Q�k?10��<@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!�Ǚb��
@)ŏ1w-!_?1�Ǚb��
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap������?!� �F6>@)Ǻ���V?1�q��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIq8f���X@Q����+<�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	:d�w73@:d�w73@!:d�w73@      ��!       "	*�Z^���?*�Z^���?!*�Z^���?*      ��!       2	9��m4��?9��m4��?!9��m4��?:	�,AF@E @�,AF@E @!�,AF@E @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qq8f���X@y����+<�?�"5
sequential/dense/MatMulMatMul�7�^Sz�?!�7�^Sz�?0"C
%gradient_tape/sequential/dense/MatMulMatMul������?!S�T�:�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch��r����?!z�1�y�?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul��r����?!�)����?"7
sequential/dense_1/MatMulMatMul��r����?!������?0"!
Adam/PowPow������?!�7�^Sz�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad������?!��r����?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam	�����?!0y��`��?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdamWJ�(o��?!�}z���?">
AssignAddVariableOp_9AssignAddVariableOpWJ�(o��?!z�	�n��?Q      Y@Y��/Ċ�0@a�	�N]�T@q�@Ȼ kX@yƢOu8�?"�
both�Your program is POTENTIALLY input-bound because 89.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 