�	5S7@5S7@!5S7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-5S7@��d�``4@1�Ye����?A�e��a��?I��c"E@*	43333�L@2U
Iterator::Model::ParallelMapV2^K�=��?!���9aB@)^K�=��?1���9aB@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�~j�t��?!�;����4@){�G�z�?1#)�k1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�]K�=�?!,Q��+7@)�&S��?1�+Q��/@:Preprocessing2F
Iterator::Model��e�c]�?!�2Y� !H@)F%u�{?1$(ͦ�&@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�J�4q?!D�JԮD@)�J�4q?1D�JԮD@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���B�i�?!(ͦ��I@)a��+ei?1��[GP�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!��t��@)����Mb`?1��t��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�<,Ԛ�?!�fӍo9@)��_�LU?1C���S@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�c(ԼX@Q@�����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��d�``4@��d�``4@!��d�``4@      ��!       "	�Ye����?�Ye����?!�Ye����?*      ��!       2	�e��a��?�e��a��?!�e��a��?:	��c"E@��c"E@!��c"E@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�c(ԼX@y@�����?�"5
sequential/dense/MatMulMatMul?R�|�?!?R�|�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMulY{�ZI�?!3]�l�b�?0"C
%gradient_tape/sequential/dense/MatMulMatMulE���<R�?!V@k���?0"7
sequential/dense_1/MatMulMatMulE���<R�?!���Z�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul��*��O�?!���$�?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGradַ�̦�?!�}\��?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMulַ�̦�?!� 6��?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdamE���<R�?!ڋ�}��?"!
Adam/PowPowE���<R�?!�A9���?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchE���<R�?!���j��?Q      Y@Y>����/@aX�i��U@q��P�W@y��"�?"�
both�Your program is POTENTIALLY input-bound because 88.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�94.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 