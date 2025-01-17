�	�ajKvB@�ajKvB@!�ajKvB@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�ajKvB@~8H���@@1���I'�?A���Q��?I�`6��@*	33333SR@2U
Iterator::Model::ParallelMapV2��_vO�?!/M���w=@)��_vO�?1/M���w=@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�e��a��?!�?��j&>@)������?1�ǔN+T7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat2U0*��?!���:8k5@)�?�߾�?1��`g��2@:Preprocessing2F
Iterator::Model%u��?!��Q�D@)�q����?1��pKH%@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice{�G�zt?!,��@�H@){�G�zt?1,��@�H@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip����z�?!@]���M@);�O��nr?1��m~�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!#L^���@)����Mb`?1#L^���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap0*��D�?!���Û*@@)-C��6Z?1pKHev@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 90.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noID�xhy�X@Q�]��K��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	~8H���@@~8H���@@!~8H���@@      ��!       "	���I'�?���I'�?!���I'�?*      ��!       2	���Q��?���Q��?!���Q��?:	�`6��@�`6��@!�`6��@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qD�xhy�X@y�]��K��?�"5
sequential/dense/MatMulMatMul�D;
ˢ?!�D;
ˢ?0"C
%gradient_tape/sequential/dense/MatMulMatMul�  ��?!xa�)���?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�  ��?!�a�5��?"7
sequential/dense_1/MatMulMatMul�  ��?!11��9�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul�!ȣa�?!�9�	R�?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul$(�?!�9��'��?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchg�d[�?!�MM�y�?"[
AArithmeticOptimizer/ReorderCastLikeAndValuePreserving_double_CastCast�  ��?!Q���?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad�  ��?!,US���?"E
'gradient_tape/sequential/dense_2/MatMulMatMul�  ��?!%�,�^�?0Q      Y@Y>����/@aX�i��U@q666?X@y�E����?"�
both�Your program is POTENTIALLY input-bound because 90.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�8.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 