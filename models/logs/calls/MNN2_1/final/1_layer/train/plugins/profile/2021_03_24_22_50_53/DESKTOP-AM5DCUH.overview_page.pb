�	��Wy�5@��Wy�5@!��Wy�5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��Wy�5@�4��A3@1��-II�?A����z�?Izލ�@*	53333�K@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate���<,�?!qxW��A@)�:pΈ�?1�CT8�0@@:Preprocessing2U
Iterator::Model::ParallelMapV2�� �rh�?!$�Ti>@)�� �rh�?1$�Ti>@:Preprocessing2F
Iterator::Model��@��ǘ?!�|�R�E@)��H�}}?1��9��)@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��H�}}?!��9��)@){�G�zt?1R�?��!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���~�:�?!p���ZL@)y�&1�l?1���r@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!��Fz|@)/n��b?1��Fz|@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap/�$��?!<
6��B@)��_�LU?1��)kʚ@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor��H�}M?!��9���?)��H�}M?1��9���?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceǺ���F?!Q�,�(	�?)Ǻ���F?1Q�,�(	�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI)��A��X@Q���/�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�4��A3@�4��A3@!�4��A3@      ��!       "	��-II�?��-II�?!��-II�?*      ��!       2	����z�?����z�?!����z�?:	zލ�@zލ�@!zލ�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q)��A��X@y���/�?�"5
sequential/dense/MatMulMatMul��Am�?!��Am�?0"C
%gradient_tape/sequential/dense/MatMulMatMull��R���?!8��ֵ?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�#��#��?!�����B�?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�#��#��?!�$�ՑW�?"7
sequential/dense_1/MatMulMatMulo���5`�?!�(����?0"!
Adam/PowPowl��R���?!]O�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchl��R���?!W�'��?"E
'gradient_tape/sequential/dense_1/MatMulMatMul�M���e�?!����3�?0"
Abs_1Abs!^W|�Ԏ?!�V۷!�?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdam!^W|�Ԏ?!Of�g*�?Q      Y@Y{	�%��1@a�����T@q�Q�>W@y�A$��?"�
both�Your program is POTENTIALLY input-bound because 88.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�93.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 