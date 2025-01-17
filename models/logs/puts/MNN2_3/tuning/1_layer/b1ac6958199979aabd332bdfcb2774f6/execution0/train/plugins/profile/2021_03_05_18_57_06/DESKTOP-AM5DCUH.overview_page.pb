�	����Os6@����Os6@!����Os6@	6����r?6����r?!6����r?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6����Os6@2˞6�3@1ܝ��.�?Ao�ŏ1�?Iy�� @Y������P?*	����̌L@2U
Iterator::Model::ParallelMapV2�Q���?!�"�h�>@)�Q���?1�"�h�>@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���Q��?!�Z5E:@)A��ǘ��?1�RK�p3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��<,Ԋ?!�B�t4�6@)46<�R�?1�Rez�3@:Preprocessing2F
Iterator::Model��+e�?!h�,IGE@)_�Q�{?1��~_�'@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�q����o?!�G �R@)�q����o?1�G �R@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip	�^)ˠ?!�*�붸L@)a��+ei?1M�T�^�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!�~_ѷ�@)/n��b?1�~_ѷ�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�St$���?!N㿼a=@)-C��6Z?1�-.;�j@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no96����r?I�͙��X@Q
*����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	2˞6�3@2˞6�3@!2˞6�3@      ��!       "	ܝ��.�?ܝ��.�?!ܝ��.�?*      ��!       2	o�ŏ1�?o�ŏ1�?!o�ŏ1�?:	y�� @y�� @!y�� @B      ��!       J	������P?������P?!������P?R      ��!       Z	������P?������P?!������P?b      ��!       JGPUY6����r?b q�͙��X@y
*����?�"5
sequential/dense/MatMulMatMulP�
)��?!P�
)��?0"C
%gradient_tape/sequential/dense/MatMulMatMulx9iL8F�?!�}��n�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�۳Y	�?!���?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�۳Y	�?!�+��}�?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdampI�z�?!I�� �?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGradpI�z�?!A銜��?"E
'gradient_tape/sequential/dense_1/MatMulMatMulpI�z�?!o�Gz�?0"7
sequential/dense_1/MatMulMatMulpI�z�?!���i���?0"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdam'nq8�?!�-i�?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam'nq8�?!aҴ�_K�?Q      Y@Y��/Ċ�0@a�	�N]�T@q�J���V@y�� �K_�?"�
both�Your program is POTENTIALLY input-bound because 88.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�91.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 