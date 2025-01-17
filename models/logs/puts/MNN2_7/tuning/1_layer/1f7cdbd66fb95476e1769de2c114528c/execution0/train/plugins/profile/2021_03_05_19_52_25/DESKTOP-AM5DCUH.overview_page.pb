�	0���6@0���6@!0���6@	�K✕?�K✕?!�K✕?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails60���6@��M��3@1����`��?A9��m4��?I���>�@Y2��|�s?*	����̌E@2U
Iterator::Model::ParallelMapV2'�����?!����s�8@)'�����?1����s�8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat������?!]����:@)n���?1��(��6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate/�$��?!�{Я�\8@)S�!�uq{?1��+*/@:Preprocessing2F
Iterator::ModelX�5�;N�?!�Μ��C@)a��+ey?1���|2�,@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceŏ1w-!o?!4ج3'�!@)ŏ1w-!o?14ج3'�!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��<,Ԛ?!L1cw�dN@)�~j�t�h?1{�6��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!F�퐴@)��H�}]?1F�퐴@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap������?!]����:@)/n��R?1����j@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�K✕?I]S��Y�X@Q�g���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��M��3@��M��3@!��M��3@      ��!       "	����`��?����`��?!����`��?*      ��!       2	9��m4��?9��m4��?!9��m4��?:	���>�@���>�@!���>�@B      ��!       J	2��|�s?2��|�s?!2��|�s?R      ��!       Z	2��|�s?2��|�s?!2��|�s?b      ��!       JGPUY�K✕?b q]S��Y�X@y�g���?�"5
sequential/dense/MatMulMatMul����?/�?!����?/�?0"C
%gradient_tape/sequential/dense/MatMulMatMul6[�ƛ�?!� �(��?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�$l�?!k=�C1V�?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�$l�?!��e�oc�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad6[�ƛ�?!]����?"7
sequential/dense_1/MatMulMatMul6[�ƛ�?!�S�^a��?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul��p/�r�?!�d�D��?0"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdamш_��?!!]D�M�?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdamш_��?!�U�����?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdamш_��?!;N�)���?Q      Y@Y��/Ċ�0@a�	�N]�T@q⎸#�X@y�FT&��?"�
both�Your program is POTENTIALLY input-bound because 88.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�98.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 