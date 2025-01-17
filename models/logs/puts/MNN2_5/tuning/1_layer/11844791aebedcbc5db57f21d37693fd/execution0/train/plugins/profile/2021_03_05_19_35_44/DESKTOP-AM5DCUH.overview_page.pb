�	�&��H5@�&��H5@!�&��H5@	`3��?`3��?!`3��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�&��H5@0�^|Ѫ2@1�����?A9��m4��?I�(�A&�@Yh�
��?*	������G@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatV-��?!�<�"h�>@) �o_Ή?1�pR��:@:Preprocessing2U
Iterator::Model::ParallelMapV2tF��_�?!N�<69@)tF��_�?1N�<69@:Preprocessing2F
Iterator::Model�&S��?!A�IݗGC@) �o_�y?1�pR��*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��ׁsF�?!4��}�4@) �o_�y?1�pR��*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipV-��?!�<�"h�N@)��H�}m?1���c+�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��H�}m?!���c+�@)��H�}m?1���c+�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!�/��@)ŏ1w-!_?1�/��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap46<�R�?!A�I�7@)����MbP?1"h8��� @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9`3��?I�`��X@Qb��a\{�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	0�^|Ѫ2@0�^|Ѫ2@!0�^|Ѫ2@      ��!       "	�����?�����?!�����?*      ��!       2	9��m4��?9��m4��?!9��m4��?:	�(�A&�@�(�A&�@!�(�A&�@B      ��!       J	h�
��?h�
��?!h�
��?R      ��!       Z	h�
��?h�
��?!h�
��?b      ��!       JGPUY`3��?b q�`��X@yb��a\{�?�"5
sequential/dense/MatMulMatMul�+�^��?!�+�^��?0"C
%gradient_tape/sequential/dense/MatMulMatMul���)�?!R���Y��?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulN��T3c�?!&��&�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�K��8�?!��Ѧ�S�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch���)�?!ل��7��?"E
'gradient_tape/sequential/dense_1/MatMulMatMul���)�?!.�뼚�?0"7
sequential/dense_1/MatMulMatMul���)�?!�}B>�?0"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdam��+�=��?!�:���8�?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam��+�=��?!�A��3�?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam��+�=��?!�Z���?Q      Y@Y��/Ċ�0@a�	�N]�T@q�QG�U@y) ~�?"�
both�Your program is POTENTIALLY input-bound because 87.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�84.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 