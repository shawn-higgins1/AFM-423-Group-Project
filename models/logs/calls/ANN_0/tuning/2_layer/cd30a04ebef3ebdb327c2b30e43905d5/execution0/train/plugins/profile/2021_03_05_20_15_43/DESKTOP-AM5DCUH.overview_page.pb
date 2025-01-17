�	�����4@�����4@!�����4@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�����4@!O!Wb3@1�� ���?A+��Χ?I5z5@ih�?*	�����YI@2U
Iterator::Model::ParallelMapV2	�^)ː?!�G�o,@@)	�^)ː?1�G�o,@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat �o_Ή?!�HPS!�8@)/�$��?1�<pƵ4@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaptF��_�?!�=��x7@)�� �rh�?1�$u���0@:Preprocessing2F
Iterator::Model���Mb�?!l+�X�,G@)�ZӼ�}?1原,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice_�Q�k?!��!��@)_�Q�k?1��!��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip_�Qڛ?!��!��J@)��_vOf?1��"AM@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!z0��k�@)�J�4a?1z0��k�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�L��X@Q:9�����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	!O!Wb3@!O!Wb3@!!O!Wb3@      ��!       "	�� ���?�� ���?!�� ���?*      ��!       2	+��Χ?+��Χ?!+��Χ?:	5z5@ih�?5z5@ih�?!5z5@ih�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�L��X@y:9�����?�"5
sequential/dense/MatMulMatMul�w��*ا?!�w��*ا?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul,��0n�?!�D�-��?0"C
%gradient_tape/sequential/dense/MatMulMatMul�����1�?!;g�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�y P���?!��!����?"7
sequential/dense_1/MatMulMatMul�y P���?!�)��c�?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul��!H��?!��-*��?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGradY_�t��?!Pu��t�?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam�����1�?!�4u����?"7
sequential/dense_2/MatMulMatMul�����1�?!l�R���?0"
Sum_4Sum�@ ��?!{��Nk�?Q      Y@Y7��Moz2@a���,daT@q���pX@yk����?"�
both�Your program is POTENTIALLY input-bound because 92.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�6.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 