�	ND��~F7@ND��~F7@!ND��~F7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-ND��~F7@:����4@1�V���x�?A+ٱ�ץ?IYni5$.@*	333333J@2U
Iterator::Model::ParallelMapV2����Mb�?!ㄔ<ˈ>@)����Mb�?1ㄔ<ˈ>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeata��+e�?!ɳ���7@)��ZӼ�?1{k�4w3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�~j�t��?!�com��6@)	�^)ˀ?1ձ�6L/@:Preprocessing2F
Iterator::Model�b�=y�?!���*�F@)vq�-�?1��'.@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceŏ1w-!o?!��YF�@)ŏ1w-!o?1��YF�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�v��/�?!ZF��1K@)��_vOf?1Ls�U�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!�℔<�@)/n��b?1�℔<�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapS�!�uq�?!�U���9@)Ǻ���V?1k�4w�_@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI!L��:�X@Q���[��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	:����4@:����4@!:����4@      ��!       "	�V���x�?�V���x�?!�V���x�?*      ��!       2	+ٱ�ץ?+ٱ�ץ?!+ٱ�ץ?:	Yni5$.@Yni5$.@!Yni5$.@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q!L��:�X@y���[��?�"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulfhK��K�?!fhK��K�?"5
sequential/dense/MatMulMatMulr�zOU�?!�U�|v�?0"7
sequential/dense_1/MatMulMatMul�Z�<WL�?!@��T��?0"C
%gradient_tape/sequential/dense/MatMulMatMulfhK��K�?!ZZzTI��?0"E
'gradient_tape/sequential/dense_1/MatMulMatMulfhK��K�?!t4�>��?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchB��^�?!������?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul<��nkܓ?!�y�<���?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdamfhK��K�?!\�7��?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGradfhK��K�?!�:M�?"E
'gradient_tape/sequential/dense_2/MatMulMatMulfhK��K�?!h`�
�?0Q      Y@Y>����/@aX�i��U@q}t��lX@y�U�|v�?"�
both�Your program is POTENTIALLY input-bound because 88.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 