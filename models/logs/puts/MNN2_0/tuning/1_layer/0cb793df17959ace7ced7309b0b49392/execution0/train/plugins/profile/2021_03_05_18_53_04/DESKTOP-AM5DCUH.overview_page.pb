�	'�y�s6@'�y�s6@!'�y�s6@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-'�y�s6@�|���A4@1;�� �>�?AɭI�%r�?Iծ	i���?*	     �M@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate �o_Ι?!'u_[E@)���Mb�?1��N�C@:Preprocessing2U
Iterator::Model::ParallelMapV2e�X��?!ylE�pR=@)e�X��?1ylE�pR=@:Preprocessing2F
Iterator::ModeltF��_�?!���c+D@)9��v��z?1��/��&@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�~j�t�x?!�Iݗ�V$@)�J�4q?14��}y@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip/n���?![4��M@)�����g?1=�"h8�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!�<�"h@)��H�}]?1�<�"h@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap	�c�?!�A�IF@)/n��R?1[4���?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor/n��R?![4���?)/n��R?1[4���?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicea2U0*�C?!��}ylE�?)a2U0*�C?1��}ylE�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 90.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�cs��X@Q��>#Q�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�|���A4@�|���A4@!�|���A4@      ��!       "	;�� �>�?;�� �>�?!;�� �>�?*      ��!       2	ɭI�%r�?ɭI�%r�?!ɭI�%r�?:	ծ	i���?ծ	i���?!ծ	i���?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�cs��X@y��>#Q�?�"5
sequential/dense/MatMulMatMul�q�7�̦?!�q�7�̦?0"C
%gradient_tape/sequential/dense/MatMulMatMul�ވ�?V�?!(._x��?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul;+h�k�?!��89l�?"7
sequential/dense_1/MatMulMatMul;+h�k�?!�١	�#�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch��jv�?!�����F�?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam�ވ�?V�?!������?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�ވ�?V�?!�/���\�?")
sequential/CastCast�ވ�?V�?!hKW�V��?"S
2gradient_tape/sequential/dense/BiasAdd/BiasAddGradBiasAddGrad^�a	:ܓ?!�����b�?"!
Adam/PowPow&t:�4b�?!�g>�g�?Q      Y@Y{	�%��1@a�����T@q��]��X@y��5B���?"�
both�Your program is POTENTIALLY input-bound because 90.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�8.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�99.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 