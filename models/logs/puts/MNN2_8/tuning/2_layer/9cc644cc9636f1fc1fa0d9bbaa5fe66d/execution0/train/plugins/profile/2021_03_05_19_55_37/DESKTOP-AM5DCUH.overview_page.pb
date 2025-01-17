�	R`L�7@R`L�7@!R`L�7@	�E�H��?�E�H��?!�E�H��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6R`L�7@�ĭ�5@1�N\�W �?Ag��j+��?I�]���@Y��4s?*	433333H@2U
Iterator::Model::ParallelMapV2F%u��?!E]t�E;@)F%u��?1E]t�E;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���S㥋?!B�0�~�;@)g��j+��?1[=;n,8@:Preprocessing2F
Iterator::Model��ZӼ�?!pc�
E@)��H�}}?15_�g��-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�0�*�?!�
�G5@)a��+ey?1���|��)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice	�^)�p?!7�A�0� @)	�^)�p?17�A�0� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipy�&1��?!���k��L@)a��+ei?1���|��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!5_�g��@)��H�}]?15_�g��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�+e�X�?!_�g���7@)/n��R?1/�袋.@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�E�H��?I�:��<�X@Qق����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�ĭ�5@�ĭ�5@!�ĭ�5@      ��!       "	�N\�W �?�N\�W �?!�N\�W �?*      ��!       2	g��j+��?g��j+��?!g��j+��?:	�]���@�]���@!�]���@B      ��!       J	��4s?��4s?!��4s?R      ��!       Z	��4s?��4s?!��4s?b      ��!       JGPUY�E�H��?b q�:��<�X@yق����?�"5
sequential/dense/MatMulMatMul,��X�?!,��X�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�r�*ҡ?!��o�7�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul!�¥)g�?!Q`̢�?"E
'gradient_tape/sequential/dense_1/MatMulMatMul��ʦ/�?!�M�	[��?0"7
sequential/dense_1/MatMulMatMul��ʦ/�?!���O��?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�Ot��F�?!���Y#f�?"7
sequential/dense_2/MatMulMatMul�Ot��F�?!�";��.�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�r�*ґ?!����9i�?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad�r�*ґ?!Y��Z|��?"E
'gradient_tape/sequential/dense_2/MatMulMatMul�r�*ґ?!�6"����?0Q      Y@Y>����/@aX�i��U@qѓ>s/�W@y�
�1��?"�
both�Your program is POTENTIALLY input-bound because 88.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�94.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 