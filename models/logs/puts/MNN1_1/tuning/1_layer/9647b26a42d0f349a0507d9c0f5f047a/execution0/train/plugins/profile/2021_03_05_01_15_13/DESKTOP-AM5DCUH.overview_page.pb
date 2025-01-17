�	>�D���5@>�D���5@!>�D���5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails->�D���5@-?p�'03@1S��c${�?A�lV}��?IA�+�� @*	�����L@2U
Iterator::Model::ParallelMapV2�o_��?!�!�g��=@)�o_��?1�!�g��=@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���H�?!X@�n�W<@)tF��_�?16�d�M65@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�
F%u�?!���]�6@)/�$��?1��+�q�2@:Preprocessing2F
Iterator::Model$����ۗ?!��8�D@)F%u�{?1������'@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����Mbp?!���m?�@)����Mbp?1���m?�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip	�^)ˠ?!<Sj�;M@)a��+ei?1�Ԏw@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!______@)/n��b?1______@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�� �rh�?!N6�d�M>@)/n��R?1______�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noID`�,r�X@Q��ϟ���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	-?p�'03@-?p�'03@!-?p�'03@      ��!       "	S��c${�?S��c${�?!S��c${�?*      ��!       2	�lV}��?�lV}��?!�lV}��?:	A�+�� @A�+�� @!A�+�� @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qD`�,r�X@y��ϟ���?�"5
sequential/dense/MatMulMatMul�2`��Φ?!�2`��Φ?0"C
%gradient_tape/sequential/dense/MatMulMatMul���F�?!>�!y��?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulg��x�W�?!�!Yk`�?"7
sequential/dense_1/MatMulMatMulg��x�W�?!9&��#�?0"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam��\�[G�?!q�sc��?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch���F�?!�<�U�,�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad���F�?!S�lH���?"E
'gradient_tape/sequential/dense_1/MatMulMatMul�� �k�?!�5y
P��?0"/
sequential/dense/ReluRelu�� �k�?!���
��?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdamH��]#i�?!�c�i�?Q      Y@Y��/Ċ�0@a�	�N]�T@q�f��3jX@y�ׄ��?"�
both�Your program is POTENTIALLY input-bound because 89.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 