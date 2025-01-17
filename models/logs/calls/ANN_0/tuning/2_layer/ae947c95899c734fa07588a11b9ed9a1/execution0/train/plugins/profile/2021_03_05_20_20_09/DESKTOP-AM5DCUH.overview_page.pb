�	u�Rz�_4@u�Rz�_4@!u�Rz�_4@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-u�Rz�_4@{�\��1@1&��:���?A�H�}�?I�O7P�}@*23333�H@)       =2U
Iterator::Model::ParallelMapV2�ZӼ��?!�W��<@)�ZӼ��?1�W��<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat �o_Ή?!X����9@)�g��s��?1R��u5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaptF��_�?!ma�6�8@)	�^)ˀ?1�Iψd�0@:Preprocessing2F
Iterator::Model��_vO�?!�ݦ���E@)���_vO~?1Z_|���-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice���_vOn?!Z_|���@)���_vOn?1Z_|���@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�Pk�w�?!4"Y&=#L@)F%u�k?1�+Z_|�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!���1@)����Mb`?1���1@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI}W���X@Q�`?*�R�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	{�\��1@{�\��1@!{�\��1@      ��!       "	&��:���?&��:���?!&��:���?*      ��!       2	�H�}�?�H�}�?!�H�}�?:	�O7P�}@�O7P�}@!�O7P�}@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q}W���X@y�`?*�R�?�"5
sequential/dense/MatMulMatMul
0�*Գ�?!
0�*Գ�?0"C
%gradient_tape/sequential/dense/MatMulMatMul��]�.�?!t%�Cq�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMulz��.�?!�h��L��?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulz��.�?!wۘت��?"7
sequential/dense_1/MatMulMatMulz��.�?!��S/��?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul��Oh�9�?!	�� b�?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdamz��.�?!���^$��?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGradz��.�?!�N��?"7
sequential/dense_2/MatMulMatMulz��.�?!�Z�l�I�?0"
Sum_4Sum	jX�Yݓ?!}�:���?Q      Y@Y7��Moz2@a���,daT@qKV�s�kX@y�c���?"�
both�Your program is POTENTIALLY input-bound because 87.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�11.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 