�	}���7@}���7@!}���7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-}���7@�'�X�4@1���#��?A�+e�X�?I,D���@@*	    �I@2U
Iterator::Model::ParallelMapV2� �	��?!-�H%�=@)� �	��?1-�H%�=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�(��0�?!��,�7@)�0�*�?1      4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�(��0�?!��,�7@)	�^)ˀ?1�6��;�/@:Preprocessing2F
Iterator::Model
ףp=
�?!�6��;�E@)�ZӼ�}?1^�	��+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice	�^)�p?!�6��;�@)	�^)�p?1�6��;�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipV-��?!R�yY�'L@)-C��6j?1C���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!H%�e@)����Mb`?1H%�e@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���S㥋?!�t��6:@)a2U0*�S?1r^�	�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI���<�X@Q�?�0�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�'�X�4@�'�X�4@!�'�X�4@      ��!       "	���#��?���#��?!���#��?*      ��!       2	�+e�X�?�+e�X�?!�+e�X�?:	,D���@@,D���@@!,D���@@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q���<�X@y�?�0�?�"E
'gradient_tape/sequential/dense_1/MatMulMatMul8����?!8����?0"5
sequential/dense/MatMulMatMul8����?!8����?0"C
%gradient_tape/sequential/dense/MatMulMatMul^�#0U�?!g�qo»?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul^�#0U�?!�N���6�?"7
sequential/dense_1/MatMulMatMul^�#0U�?!�Ч�ϋ�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchV�����?!�ɢ�AA�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMulv��,|��?!<{=T���?"7
sequential/dense_2/MatMulMatMulv��,|��?!�,����?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad^�#0U�?!��)oCk�?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad^�#0U�?!Q�gq���?Q      Y@Y>����/@aX�i��U@qkSd�pX@y�
8�?"�
both�Your program is POTENTIALLY input-bound because 89.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 