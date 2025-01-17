�	F�=��7@F�=��7@!F�=��7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-F�=��7@���'�4@1mT�YO�?A$����ۧ?IH����0@*	hfffffL@2U
Iterator::Model::ParallelMapV2���S㥋?!�h�́�7@)���S㥋?1�h�́�7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�St$���?!6��9=@)�{�Pk�?1��@��6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�]K�=�?!L=�]j7@)�5�;Nс?1H���@�.@:Preprocessing2F
Iterator::Model�g��s��?!/�d�v�B@)� �	�?1��V�+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceHP�s�r?!P�o�z2 @)HP�s�r?1P�o�z2 @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��y�):�?!�
��VO@)�J�4q?1z2~�ԓ@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor���_vOn?!́D+l@)���_vOn?1́D+l@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���_vO�?!́D+l:@)�~j�t�X?1$Zas @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIu#�&�X@Q̸"wH��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���'�4@���'�4@!���'�4@      ��!       "	mT�YO�?mT�YO�?!mT�YO�?*      ��!       2	$����ۧ?$����ۧ?!$����ۧ?:	H����0@H����0@!H����0@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qu#�&�X@y̸"wH��?�"5
sequential/dense/MatMulMatMul���TJ�?!���TJ�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMulxiZ�%�?!�7�?0"C
%gradient_tape/sequential/dense/MatMulMatMul��C�T�?!b�q���?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�JK���?!�ˁ��?"7
sequential/dense_1/MatMulMatMul�8���?!OμL�?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�1o�?!~o�ϝ��?"7
sequential/dense_2/MatMulMatMul�1o�?!����~��?0"!
Adam/PowPowxiZ�%�?!(��2��?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchxiZ�%�?!W*}����?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGradxiZ�%�?!�;�jM��?Q      Y@Y>����/@aX�i��U@q�W �eX@y��e��?"�
both�Your program is POTENTIALLY input-bound because 88.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 