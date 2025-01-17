�	p��=8@p��=8@!p��=8@	Y~+��Mq?Y~+��Mq?!Y~+��Mq?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6p��=8@mV}���5@1���EB[�?A9��m4��?IS�r/0k @Y������P?*	43333�G@2U
Iterator::Model::ParallelMapV2 �o_Ή?!)�3�:@) �o_Ή?1)�3�:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�������?!i;�
_:@)/�$��?1M����&6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateM�St$�?!Y�>��7@)_�Q�{?1��O�%�,@:Preprocessing2F
Iterator::ModeljM�?!���O�%D@)9��v��z?1/��m+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice;�O��nr?!�]-n��"@);�O��nr?1�]-n��"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipB>�٬��?!`H�1�M@)_�Q�k?1��O�%�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!l��Ӭ�@)����Mb`?1l��Ӭ�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapa��+e�?!�Y��):@)/n��R?1w���@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9Y~+��Mq?I�'% �X@Q���/�&�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	mV}���5@mV}���5@!mV}���5@      ��!       "	���EB[�?���EB[�?!���EB[�?*      ��!       2	9��m4��?9��m4��?!9��m4��?:	S�r/0k @S�r/0k @!S�r/0k @B      ��!       J	������P?������P?!������P?R      ��!       Z	������P?������P?!������P?b      ��!       JGPUYY~+��Mq?b q�'% �X@y���/�&�?�"5
sequential/dense/MatMulMatMul<x�:��?!<x�:��?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�Q�߂�?!�� ��?"7
sequential/dense_1/MatMulMatMul��w0\q�?!6���S�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�4I��<�?!?#0q��?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul7�,[��?!&��<���?0"7
sequential/dense_2/MatMulMatMulL&iӖ�?!��&��?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul��z4�?!���Z�0�?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam�Q�߂�?!·�<a�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�Q�߂�?!	r~m��?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad�Q�߂�?!"�7����?Q      Y@Y>����/@aX�i��U@q3,�W@y�2f]��?"�
both�Your program is POTENTIALLY input-bound because 89.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�8.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�92.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 