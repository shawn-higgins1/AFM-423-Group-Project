�	A����6@A����6@!A����6@	 �$�w��? �$�w��?! �$�w��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6A����6@c��K�a4@12��n�?A�z6�>�?IZ)r��@Yj�drjgh?*	������H@2U
Iterator::Model::ParallelMapV2S�!�uq�?!<�œ[<;@)S�!�uq�?1<�œ[<;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeaty�&1��?!t�H��t<@)��@��ǈ?1��~Y�8@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate46<�R�?!(�xr�'6@)y�&1�|?1t�H��t,@:Preprocessing2F
Iterator::Modeln���?!����/�C@)a��+ey?14�@S4)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�q����o?!�+Q��@)�q����o?1�+Q��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���_vO�?!M!�N@)-C��6j?1�B�)@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!�On��@)ŏ1w-!_?1�On��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapa��+e�?!4�@S49@)�~j�t�X?1dp>�c@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9 �$�w��?I�D�:�X@Q۴�����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	c��K�a4@c��K�a4@!c��K�a4@      ��!       "	2��n�?2��n�?!2��n�?*      ��!       2	�z6�>�?�z6�>�?!�z6�>�?:	Z)r��@Z)r��@!Z)r��@B      ��!       J	j�drjgh?j�drjgh?!j�drjgh?R      ��!       Z	j�drjgh?j�drjgh?!j�drjgh?b      ��!       JGPUY �$�w��?b q�D�:�X@y۴�����?�"5
sequential/dense/MatMulMatMul;���eq�?!;���eq�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�J����?!�}��0F�?0"7
sequential/dense_1/MatMulMatMul��GcZH�?!�Z\�]�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul���8�?!9�>�5�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMuly��-�>�?!�����?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�%��ə?!�%��7�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch{��p��?!�Z\�]��?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdamw���nI�?!�z܋�?"!
Adam/PowPow��GcZH�?!��q�K�?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad��GcZH�?!��:�2�?Q      Y@Y>����/@aX�i��U@q���1�IX@y� ;�?"�
both�Your program is POTENTIALLY input-bound because 89.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 