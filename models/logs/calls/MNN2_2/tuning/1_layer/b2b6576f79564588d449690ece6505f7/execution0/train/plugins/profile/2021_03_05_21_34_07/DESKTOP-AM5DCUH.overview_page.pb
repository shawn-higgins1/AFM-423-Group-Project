�	Fж�6@Fж�6@!Fж�6@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-Fж�6@��4���3@1U��C��?AǺ���?I}@�3iS�?*	�����?J@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�Q���?!������@@)%u��?1     <@:Preprocessing2U
Iterator::Model::ParallelMapV2��<,Ԋ?!=��<��8@)��<,Ԋ?1=��<��8@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::ConcatenateǺ����?!VUUUUU5@)�<,Ԛ�}?1�<��<�+@:Preprocessing2F
Iterator::Model���<,�?!1�0�B@)F%u�{?1J�$I�$)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice�q����o?!n۶m۶@)�q����o?1n۶m۶@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip	�^)ˠ?!��<��<O@)-C��6j?1�a�a@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorǺ���f?!VUUUUU@)Ǻ���f?1VUUUUU@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��@��ǈ?!�0�07@)��H�}M?1ܶm۶m�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 90.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIѻ|1��X@Q��A��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��4���3@��4���3@!��4���3@      ��!       "	U��C��?U��C��?!U��C��?*      ��!       2	Ǻ���?Ǻ���?!Ǻ���?:	}@�3iS�?}@�3iS�?!}@�3iS�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qѻ|1��X@y��A��?�"5
sequential/dense/MatMulMatMulN8�1�a�?!N8�1�a�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�+��X�?!�M�"�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�AJi�֝?!�B�k���?"7
sequential/dense_1/MatMulMatMul�D�r�ޘ?!�i$dh�?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�9k6��?!��J���?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�+��X�?!�Vi[a�?"]
?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nanDivNoNan�+��X�?!�(����?"!
Adam/PowPow�s�>���?!�*�xU�?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam?��ǔ�?!�A����?"S
2gradient_tape/sequential/dense/BiasAdd/BiasAddGradBiasAddGrad�ّŤ��?!���c�?Q      Y@Y��/Ċ�0@a�	�N]�T@qX�ʉtX@y�X���?"�
both�Your program is POTENTIALLY input-bound because 90.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�8.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 