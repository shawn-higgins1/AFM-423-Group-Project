�	��� �6@��� �6@!��� �6@	jm��/*�?jm��/*�?!jm��/*�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6��� �6@73��pb4@1i�'��?A������?I�����F@Yq���
x?*	�����LI@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate+�����?!P�Ak�DC@)��y�):�?1��LĖA@:Preprocessing2U
Iterator::Model::ParallelMapV29��v���?!�9�9@)9��v���?1�9�9@:Preprocessing2F
Iterator::ModelM�O��?!f����C@)��H�}}?1k#a `u,@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat �o_�y?!��T��(@)/n��r?1���,d!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipŏ1w-!�?!�3[
N@)�~j�t�h?1��Pp%�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!�3[
@)ŏ1w-!_?1�3[
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap^K�=��?!����D@)-C��6Z?1	��K	@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor/n��R?!���,d@)/n��R?1���,d@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicea2U0*�C?!F�@����?)a2U0*�C?1F�@����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9jm��/*�?I��gEC�X@Qi��F�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	73��pb4@73��pb4@!73��pb4@      ��!       "	i�'��?i�'��?!i�'��?*      ��!       2	������?������?!������?:	�����F@�����F@!�����F@B      ��!       J	q���
x?q���
x?!q���
x?R      ��!       Z	q���
x?q���
x?!q���
x?b      ��!       JGPUYjm��/*�?b q��gEC�X@yi��F�?�"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�ϴ�赯?!�ϴ�赯?"5
sequential/dense/MatMulMatMul����?!W���S^�?0"C
%gradient_tape/sequential/dense/MatMulMatMul:��?��?!:��?���?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul�*٠?!i����?0"7
sequential/dense_1/MatMulMatMul�ҹ/���?!ܣ�����?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch��͏�#�?!�_��lw�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul��͏�#�?!��	��?"7
sequential/dense_2/MatMulMatMul��͏�#�?!�k��4`�?0"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam:��?��?!H�a�n�?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad:��?��?!�h�5f}�?Q      Y@Y�C=�C=0@a��
��T@q��h��W@y�=�UX��?"�
both�Your program is POTENTIALLY input-bound because 88.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�94.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 