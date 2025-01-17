�	7�h��C@7�h��C@!7�h��C@	ʲ1���?ʲ1���?!ʲ1���?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails67�h��C@��� �A@1U�����?A�>�-W?�?I���� @YV��Dׅo?*������I@)       =2U
Iterator::Model::ParallelMapV22U0*��?![Җ�%m>@)2U0*��?1[Җ�%m>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�(��0�?!6eMYS�7@)��ZӼ�?1qG�w�3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatea��+e�?!��8@)"��u���?1������0@:Preprocessing2F
Iterator::ModelA��ǘ��?!�/���E@)9��v��z?1�sƜ1)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceŏ1w-!o?!^QW�u@)ŏ1w-!o?1^QW�u@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip%u��?!_��}L@)a��+ei?1��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!w�qG@)�J�4a?1w�qG@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapy�&1��?!��!;@)-C��6Z?14�9c�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 93.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�5.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9ʲ1���?I����X@Q8:�O�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��� �A@��� �A@!��� �A@      ��!       "	U�����?U�����?!U�����?*      ��!       2	�>�-W?�?�>�-W?�?!�>�-W?�?:	���� @���� @!���� @B      ��!       J	V��Dׅo?V��Dׅo?!V��Dׅo?R      ��!       Z	V��Dׅo?V��Dׅo?!V��Dׅo?b      ��!       JGPUYʲ1���?b q����X@y8:�O�?�"5
sequential/dense/MatMulMatMul+R�U�?!+R�U�?0"C
%gradient_tape/sequential/dense/MatMulMatMulѺ\u�/�?!~��<�B�?0"7
sequential/dense_1/MatMulMatMulѺ\u�/�?!��`w�ں?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul+��ڠ?!��t] ��?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�Fb���?!�5!�f�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�鳒�{�?!Ͳw	��?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch9J�Д?!|;���?"!
Adam/PowPowѺ\u�/�?!n���?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGradѺ\u�/�?!Ȫ����?"7
sequential/dense_2/MatMulMatMulѺ\u�/�?!!� ��?0Q      Y@Y>����/@aX�i��U@q��'��W@y�e����?"�
both�Your program is POTENTIALLY input-bound because 93.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�5.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�94.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 