�	g���U7@g���U7@!g���U7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-g���U7@�#F�-L4@1sh��|?�?AM�St$�?IͰQ�o�@*	333333K@2U
Iterator::Model::ParallelMapV2�ZӼ��?![ZZZZ:@)�ZӼ��?1[ZZZZ:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat����Mb�?!jiiiii=@)�?�߾�?1/9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate������?!�����R5@)���_vO~?1�����4+@:Preprocessing2F
Iterator::ModelˡE����?!������B@) �o_�y?1jiiii)'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipz6�>W[�?!yxxxx(O@)��ZӼ�t?1     �"@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�J�4q?!������@)�J�4q?1������@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP�s�b?!jiiii�@)HP�s�b?1jiiii�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap-C��6�?!������7@)a2U0*�S?1������@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��k̱X@Q�J	����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�#F�-L4@�#F�-L4@!�#F�-L4@      ��!       "	sh��|?�?sh��|?�?!sh��|?�?*      ��!       2	M�St$�?M�St$�?!M�St$�?:	ͰQ�o�@ͰQ�o�@!ͰQ�o�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��k̱X@y�J	����?�"5
sequential/dense/MatMulMatMul%��yE�?!%��yE�?0"C
%gradient_tape/sequential/dense/MatMulMatMulطêB�?!"^�)D�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMulݏ��-�?!����ڼ?0"7
sequential/dense_1/MatMulMatMulݏ��-�?!���ָ�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�{�#��?!w�w��z�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�3�by�?!���&�)�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�3�by�?!m s��?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad���[�.�?!�i�>���?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdamݏ��-�?!��nWM�?"!
Adam/PowPowݏ��-�?!ˆ�'%�?Q      Y@Y>����/@aX�i��U@q��$irX@yɀI�B��?"�
both�Your program is POTENTIALLY input-bound because 87.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�11.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 