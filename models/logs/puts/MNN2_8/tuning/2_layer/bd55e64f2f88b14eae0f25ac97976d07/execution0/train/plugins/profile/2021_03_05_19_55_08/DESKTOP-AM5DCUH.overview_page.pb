�	���2��6@���2��6@!���2��6@	
���~u�?
���~u�?!
���~u�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6���2��6@U�-��A4@1�j�ѯ�?A���JY��?IV���@YjM�St�?*	     �G@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�ZӼ��?!��l�w6>@)�~j�t��?1&W�+�9@:Preprocessing2U
Iterator::Model::ParallelMapV2Zd;�O��?!ڨ�l�w8@)Zd;�O��?1ڨ�l�w8@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate46<�R�?!�
br17@)��H�}}?1��F}g�.@:Preprocessing2F
Iterator::Model/n���?!br1�B@)�HP�x?1w6�;�)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���_vOn?!����F}@)���_vOn?1����F}@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip%u��?!�����FO@)�~j�t�h?1&W�+�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!br1�@)/n��b?1br1�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�HP��?!w6�;�9@)��_�LU?1�\AL� @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9
���~u�?Iҩ2���X@Q�$�de�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	U�-��A4@U�-��A4@!U�-��A4@      ��!       "	�j�ѯ�?�j�ѯ�?!�j�ѯ�?*      ��!       2	���JY��?���JY��?!���JY��?:	V���@V���@!V���@B      ��!       J	jM�St�?jM�St�?!jM�St�?R      ��!       Z	jM�St�?jM�St�?!jM�St�?b      ��!       JGPUY
���~u�?b qҩ2���X@y�$�de�?�"5
sequential/dense/MatMulMatMulrjꇙ�?!rjꇙ�?0"C
%gradient_tape/sequential/dense/MatMulMatMul-B	l�?!P�yǂ �?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul-B	l�?!fw��8y�?"7
sequential/dense_1/MatMulMatMul-B	l�?!>�Ag�x�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMulf&���?!ؕ�W���?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch����-�?!/9VN�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul����-�?!�z1��?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam-B	l�?!̢�<�?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad-B	l�?!��j0�?"E
'gradient_tape/sequential/dense_2/MatMulMatMul-B	l�?!�yj�K��?0Q      Y@Y>����/@aX�i��U@q���u�"X@yfw��8y�?"�
both�Your program is POTENTIALLY input-bound because 89.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�96.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 