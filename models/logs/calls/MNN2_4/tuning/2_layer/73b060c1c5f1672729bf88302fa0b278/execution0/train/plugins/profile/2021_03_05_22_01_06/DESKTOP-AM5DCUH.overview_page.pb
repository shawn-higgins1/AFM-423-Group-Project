�	���>�7@���>�7@!���>�7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-���>�7@��Ӝ�5@1�ۂ����?A��3���?I�R#�3@*	33333�F@2U
Iterator::Model::ParallelMapV2tF��_�?!�^B{	�9@)tF��_�?1�^B{	�9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��<,Ԋ?!æ�mQ�<@)�g��s��?14��7@:Preprocessing2F
Iterator::ModelHP�sג?!�|`�'D@)9��v��z?16�n�R,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenaten���?!� ���Y5@)�HP�x?1ΰ�w[�*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���_vOn?!�t�; @)���_vOn?1�t�; @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip)\���(�?!��f��M@)Ǻ���f?1.J��f@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{�G�zd?!;t�W�@){�G�zd?1;t�W�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap������?!��~�E9@)��H�}]?1______@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI����X@Qqz�z�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��Ӝ�5@��Ӝ�5@!��Ӝ�5@      ��!       "	�ۂ����?�ۂ����?!�ۂ����?*      ��!       2	��3���?��3���?!��3���?:	�R#�3@�R#�3@!�R#�3@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q����X@yqz�z�?�"5
sequential/dense/MatMulMatMulL��,�?!L��,�?0"C
%gradient_tape/sequential/dense/MatMulMatMulD8�6�
�?!ȋd*k�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMulD8�6�
�?!�'��ʠ�?0"7
sequential/dense_1/MatMulMatMulD8�6�
�?!�0��?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulv"-��ҝ?!U��oM�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMulU��oM�?! �H���?"7
sequential/dense_2/MatMulMatMulU��oM�?!�'��ʠ�?0"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam������?!�<<�D��?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchD8�6�
�?!������?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGradD8�6�
�?!w��Gz��?Q      Y@Y>����/@aX�i��U@q�r�|X@y�]g���?"�
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
Refer to the TF2 Profiler FAQb�97.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 