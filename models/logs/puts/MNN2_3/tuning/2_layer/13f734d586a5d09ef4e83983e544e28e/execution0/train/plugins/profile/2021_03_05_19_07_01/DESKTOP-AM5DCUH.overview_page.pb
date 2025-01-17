�	�=AbW7@�=AbW7@!�=AbW7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�=AbW7@wd�6��4@1{��&3��?A$����ۧ?IT� Pō@*	23333�L@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��A�f�?!^�jEaB@)X�5�;N�?1:��10=@:Preprocessing2U
Iterator::Model::ParallelMapV2��0�*�?!�^�jEa4@)��0�*�?1�^�jEa4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat_�Qڋ?!��F }7@)Ǻ����?1�s~v�W3@:Preprocessing2F
Iterator::Model/n���?!w�}L�e>@)�����w?1�e���$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��3��?!��ଓfQ@)/n��r?1w�}L�e@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����Mbp?!��D�@)����Mbp?1��D�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*�c?!p��@��@)a2U0*�c?1p��@��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��ͪ�Ֆ?!�u��AC@)Ǻ���V?1�s~v�W@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIá���X@Q6��_އ�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	wd�6��4@wd�6��4@!wd�6��4@      ��!       "	{��&3��?{��&3��?!{��&3��?*      ��!       2	$����ۧ?$����ۧ?!$����ۧ?:	T� Pō@T� Pō@!T� Pō@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qá���X@y6��_އ�?�"5
sequential/dense/MatMulMatMul�
�	��?!�
�	��?0"C
%gradient_tape/sequential/dense/MatMulMatMul���hzd�?!vP{�z�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul���hzd�?!r��C�,�?"7
sequential/dense_1/MatMulMatMul���hzd�?!7�3<���?0"E
'gradient_tape/sequential/dense_1/MatMulMatMulo���S�?!�ԷcD�?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul��'���?!R�9���?"7
sequential/dense_2/MatMulMatMul��UК�?!n��p��?0"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam���hzd�?!m"' ��?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch���hzd�?!6��G�?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad���hzd�?!���`��?Q      Y@Y>����/@aX�i��U@q���khX@yr��C�,�?"�
both�Your program is POTENTIALLY input-bound because 88.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 