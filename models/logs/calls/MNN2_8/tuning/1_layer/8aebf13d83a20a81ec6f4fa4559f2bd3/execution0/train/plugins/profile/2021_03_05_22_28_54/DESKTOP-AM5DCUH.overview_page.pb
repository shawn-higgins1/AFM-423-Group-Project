�	�Rb�f5@�Rb�f5@!�Rb�f5@	�CjvI�?�CjvI�?!�CjvI�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�Rb�f5@�3��2@1'k�C4��?AM�St$�?I1]���@YV��W9r?*	������H@2U
Iterator::Model::ParallelMapV2_�Qڋ?!��F:l�;@)_�Qڋ?1��F:l�;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��<,Ԋ?!�)�B�:@)�I+��?1[<�œ[6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateZd;�O��?!`��e�_7@)���_vO~?1M!�.@:Preprocessing2F
Iterator::Model��ZӼ�?!��+Q�D@)_�Q�{?1��F:l�+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice	�^)�p?!������ @)	�^)�p?1������ @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��H�}�?!D�JԮDM@)a��+ei?14�@S4@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!�+Q�@)�J�4a?1�+Q�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap �o_Ή?!�����9@)/n��R?1��-���@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�CjvI�?IHUR��X@Q4
�?��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�3��2@�3��2@!�3��2@      ��!       "	'k�C4��?'k�C4��?!'k�C4��?*      ��!       2	M�St$�?M�St$�?!M�St$�?:	1]���@1]���@!1]���@B      ��!       J	V��W9r?V��W9r?!V��W9r?R      ��!       Z	V��W9r?V��W9r?!V��W9r?b      ��!       JGPUY�CjvI�?b qHUR��X@y4
�?��?�"5
sequential/dense/MatMulMatMul�\�y�?!�\�y�?0"C
%gradient_tape/sequential/dense/MatMulMatMul[���!ޤ?!tX�,�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul2.�>��?!���wn��?"7
sequential/dense_1/MatMulMatMul�����D�?!g?��V��?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch[���!ޔ?!��?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad[���!ޔ?!=�vL߸�?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam�D�u�O�?!�V�3ܭ�?"
Sum_3Sum�D�u�O�?!��+٢�?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam	�m�2M�?!�BF���?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdam	�m�2M�?!��Yq��?Q      Y@Y��/Ċ�0@a�	�N]�T@q;���W@y��g�M�?"�
both�Your program is POTENTIALLY input-bound because 88.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�94.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 