�	��x�s3@��x�s3@!��x�s3@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��x�s3@�_��\1@1�x[���?A��ͪ�զ?I�b��	E�?*	33333sM@2U
Iterator::Model::ParallelMapV2jM�?!���6@@)jM�?1���6@@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap"��u���?!�I��3=@)F%u��?1�n��.i6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�{�Pk�?!��6���5@)�I+��?1]ܙ��2@:Preprocessing2F
Iterator::Model-C��6�?!��q�S�E@)9��v��z?1V��1A&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice����Mbp?!�W�(*@)����Mbp?1�W�(*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��镲�?!* ��DL@)-C��6j?1��q�S�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!�9�s�	@)ŏ1w-!_?1�9�s�	@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�k�;�X@Q4%���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�_��\1@�_��\1@!�_��\1@      ��!       "	�x[���?�x[���?!�x[���?*      ��!       2	��ͪ�զ?��ͪ�զ?!��ͪ�զ?:	�b��	E�?�b��	E�?!�b��	E�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�k�;�X@y4%���?�"5
sequential/dense/MatMulMatMul���+ݫ?!���+ݫ?0"C
%gradient_tape/sequential/dense/MatMulMatMulܰJz�Ĩ?!�[��P�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul����_0�?!mdk����?"7
sequential/dense_1/MatMulMatMul]ݘ���?!��@��?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�ՙ/�?! q(4Y�?"E
'gradient_tape/sequential/dense_1/MatMulMatMulzVI��?!o4S��k�?0"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam��[r��?!5�A��?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam��[r��?!̚�F9��?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam��[r��?!\lp��?"!
Adam/PowPow��[r��?!`�����?Q      Y@Y�M�_{4@a��(�S@q�V�k�X@y,Z��Y�?"�
both�Your program is POTENTIALLY input-bound because 89.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�99.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 