�	7l[��87@7l[��87@!7l[��87@	��N��S�?��N��S�?!��N��S�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails67l[��87@;�i���4@1ŏ1w-!�?A|�Pk��?I�.��@Y⬈���q?*	333333J@2U
Iterator::Model::ParallelMapV2L7�A`�?!
)y�}?@)L7�A`�?1
)y�}?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�Pk�w�?!?�:�ֆ:@)��ׁsF�?1'��2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatg��j+��?!�g�T6@)�j+��݃?1�ֆi�2@:Preprocessing2F
Iterator::ModelM�St$�?!s�U��E@)�HP�x?1��*NH'@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����Mbp?!ㄔ<ˈ@)����Mbp?1ㄔ<ˈ@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip2�%䃞?!���coL@)a��+ei?1ɳ���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!ㄔ<ˈ@)����Mb`?1ㄔ<ˈ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���Q��?!�<ˈ>�<@)/n��R?1�℔<� @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��N��S�?I���"ĻX@Q[W�>���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	;�i���4@;�i���4@!;�i���4@      ��!       "	ŏ1w-!�?ŏ1w-!�?!ŏ1w-!�?*      ��!       2	|�Pk��?|�Pk��?!|�Pk��?:	�.��@�.��@!�.��@B      ��!       J	⬈���q?⬈���q?!⬈���q?R      ��!       Z	⬈���q?⬈���q?!⬈���q?b      ��!       JGPUY��N��S�?b q���"ĻX@y[W�>���?�"5
sequential/dense/MatMulMatMul	��^�f�?!	��^�f�?0"C
%gradient_tape/sequential/dense/MatMulMatMul$G4�?�?!��v�R�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul$G4�?�?!(�����?0"7
sequential/dense_1/MatMulMatMul$G4�?�?!]���?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul|wY.�?!�X�0Ҏ�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�X�0Ҏ�?!��v�@�?"7
sequential/dense_2/MatMulMatMul�X�0Ҏ�?!)������?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch$G4�?�?!8H�h�?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad$G4�?�?!���_JB�?"!
Adam/PowPow��p��?!b��6��?Q      Y@Y>����/@aX�i��U@qLR3��mW@y(������?"�
both�Your program is POTENTIALLY input-bound because 89.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�93.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 