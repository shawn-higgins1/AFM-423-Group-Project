�	p�G6@p�G6@!p�G6@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-p�G6@�f�lt�3@1ȗP���?A�ׁsF��?I5��K @*	������P@2U
Iterator::Model::ParallelMapV2g��j+��?!�`�c1;A@)g��j+��?1�`�c1;A@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate2U0*��?!��)K7@)�~j�t��?1�i��L�1@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatF%u��?!$�!�p3@)'�����?1w�2�U�/@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipvOjM�?!�8���K@)lxz�,C|?1<���R$@:Preprocessing2F
Iterator::Model?W[���?!r�y�=F@)_�Q�{?14�A^�$@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���_vOn?!f��u��@)���_vOn?1f��u��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{�G�zd?!>[B0�t@){�G�zd?1>[B0�t@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap/n���?!����9@)ŏ1w-!_?1w����b@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIw��X@Q.�>b�<�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�f�lt�3@�f�lt�3@!�f�lt�3@      ��!       "	ȗP���?ȗP���?!ȗP���?*      ��!       2	�ׁsF��?�ׁsF��?!�ׁsF��?:	5��K @5��K @!5��K @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qw��X@y.�>b�<�?�"5
sequential/dense/MatMulMatMulȤx�L��?!Ȥx�L��?0"C
%gradient_tape/sequential/dense/MatMulMatMul$=kȵ�?!���$�;�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulm�:�'�?!t��yż?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchx_dc0ԙ?!�E6C��?"O
5gradient_tape/mean_squared_error/weighted_loss/Tile_1Tile$=kȵ�?!��C��:�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad$=kȵ�?!@Qxp��?"7
sequential/dense_1/MatMulMatMul$=kȵ�?!�|^1v�?0"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam�۠�c�?!��(<8l�?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam�۠�c�?!Z��Fib�?"!
Adam/PowPow�۠�c�?!��Q�X�?Q      Y@Y��/Ċ�0@a�	�N]�T@qZ��MX@y�Ǔ�X�?"�
both�Your program is POTENTIALLY input-bound because 89.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 