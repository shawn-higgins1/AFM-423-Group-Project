�	�衶�5@�衶�5@!�衶�5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�衶�5@[�D�A&3@1���|y�?A$����ۧ?I�0�|@*	�����N@2U
Iterator::Model::ParallelMapV2� �	��?!fMYS֔9@)� �	��?1fMYS֔9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate;�O��n�?!z�9,|�=@)��H�}�?1��ǉ��7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatV-��?!�ϼ�8@)��@��ǈ?1�*�Ӄ4@:Preprocessing2F
Iterator::Model��JY�8�?!A_��B@) �o_�y?1<⎸#�$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��ʡE��?!��/��O@)HP�s�r?1�����@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice��H�}m?!��ǉ��@)��H�}m?1��ǉ��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*�c?!�'����@)a2U0*�c?1�'����@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�j+��ݓ?!��ϼ�@@)Ǻ���V?1��)kʚ@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�qf�r�X@Q��L���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	[�D�A&3@[�D�A&3@![�D�A&3@      ��!       "	���|y�?���|y�?!���|y�?*      ��!       2	$����ۧ?$����ۧ?!$����ۧ?:	�0�|@�0�|@!�0�|@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�qf�r�X@y��L���?�"5
sequential/dense/MatMulMatMul����9�?!����9�?0"C
%gradient_tape/sequential/dense/MatMulMatMul1�]>�?!wAק���?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch`dJ�-Ι?!����/�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad���{�?!EOQG4G�?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul=�B���?!mQ��+g�?"7
sequential/dense_1/MatMulMatMul�Y�?!������?0"S
2gradient_tape/sequential/dense/BiasAdd/BiasAddGradBiasAddGrad�r!w�{�?!l#��>��?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam���j��?!&5��z�?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam���j��?!��*,j�?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam���j��?!�ˢY�?Q      Y@Y��/Ċ�0@a�	�N]�T@q!�����X@y�~.�� �?"�
both�Your program is POTENTIALLY input-bound because 87.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�11.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�99.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 