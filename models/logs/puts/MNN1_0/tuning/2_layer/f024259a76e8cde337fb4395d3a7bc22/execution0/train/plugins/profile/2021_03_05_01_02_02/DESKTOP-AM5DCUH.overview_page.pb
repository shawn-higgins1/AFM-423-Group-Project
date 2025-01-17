�	(Hlw�7@(Hlw�7@!(Hlw�7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-(Hlw�7@UQ��چ4@1�6�^���?Au���?I� ��F!@*	������G@2U
Iterator::Model::ParallelMapV2 �o_Ή?!yxxxxx:@) �o_Ή?1yxxxxx:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat9��v���?!��O��O;@)�I+��?1e�e�7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate/�$��?!6@)_�Q�{?1$I�$I�,@:Preprocessing2F
Iterator::Modela2U0*��?!t+t+D@)F%u�{?1�(��(�+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���_vOn?!��@)���_vOn?1��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�ZӼ��?!��ԋ��M@)y�&1�l?1jiiiii@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!a��`��@)����Mb`?1a��`��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaptF��_�?!      9@)Ǻ���V?1������@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�a�>�qX@Q���"��@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	UQ��چ4@UQ��چ4@!UQ��چ4@      ��!       "	�6�^���?�6�^���?!�6�^���?*      ��!       2	u���?u���?!u���?:	� ��F!@� ��F!@!� ��F!@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�a�>�qX@y���"��@�"5
sequential/dense/MatMulMatMul��:H�?!��:H�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul"�>4@ؠ?!���7��?"C
%gradient_tape/sequential/dense/MatMulMatMulkx
�gǠ?!�އ1xI�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul  �rVb�?!m/x�Q=�?0"7
sequential/dense_1/MatMulMatMul|�m[pz�?!|������?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul֨Q�]�?!��y���?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�dNAP�?!4鹁U0�?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad�dNAP�?!ѵ���?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam"�>4@ؐ?!��k�'��?"7
sequential/dense_2/MatMulMatMul"�>4@ؐ?!ͱy���?0Q      Y@Y>����/@aX�i��U@q'�d�W�X@y%��Q�Q�?"�
both�Your program is POTENTIALLY input-bound because 87.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�98.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 