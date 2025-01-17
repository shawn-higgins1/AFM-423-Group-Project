�	aU��N�2@aU��N�2@!aU��N�2@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-aU��N�2@���h1@1�������?AK��F>��?I�?OI�?*	�����N@2U
Iterator::Model::ParallelMapV2���H�?!"8vi:@)���H�?1"8vi:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat2U0*��?!"8vi:@)F%u��?1{�L�I�5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap����Mb�?!�����:@)�~j�t��?1�8�r��3@:Preprocessing2F
Iterator::Modela��+e�?!&����D@)��y�):�?1d�Ojo�-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice����Mbp?!�����@)����Mbp?1�����@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipr�����?!� [	�fM@)���_vOn?1%����@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{�G�zd?!'Z��}�@){�G�zd?1'Z��}�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��Ś��X@Q5����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���h1@���h1@!���h1@      ��!       "	�������?�������?!�������?*      ��!       2	K��F>��?K��F>��?!K��F>��?:	�?OI�?�?OI�?!�?OI�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��Ś��X@y5����?�"5
sequential/dense/MatMulMatMulh��%�ԯ?!h��%�ԯ?0"C
%gradient_tape/sequential/dense/MatMulMatMul�5��.w�?!�|QU��?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulh��%�ԟ?!��eo�M�?"7
sequential/dense_1/MatMulMatMul��±@	�?!�4���.�?0"!
Adam/PowPow�5��.w�?!�{5V�]�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�5��.w�?!u��&���?"$
MaximumMaximum�sV��?!�@�q���?"
Abs_1Absr���b�?![g���?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdamr���b�?!�?�*��?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdamr���b�?!�w���?Q      Y@Y�M�_{4@a��(�S@q堁1YX@y!z�W��?"�
both�Your program is POTENTIALLY input-bound because 92.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�6.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 