�	U���e7@U���e7@!U���e7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-U���e7@�(�'�4@1�)��?A/�H��?Iퟧ�d@*	�����J@2U
Iterator::Model::ParallelMapV2� �	��?!mYz���=@)� �	��?1mYz���=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�(��0�?!'0.Û7@)�g��s��?1l�qX4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��<,Ԋ?!��A*�$9@)��~j�t�?1�e�=�;2@:Preprocessing2F
Iterator::Model�e��a��?!Zz���5E@)S�!�uq{?1�6I�B�)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��H�}m?!��a���@)��H�}m?1��a���@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���Q��?!��pq�L@)a��+ei?1�0���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor_�Q�[?!m N��
@)_�Q�[?1m N��
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�<,Ԛ�?!�fw�<@)�~j�t�X?1R�&iZ@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��5M�X@Q7@���,�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�(�'�4@�(�'�4@!�(�'�4@      ��!       "	�)��?�)��?!�)��?*      ��!       2	/�H��?/�H��?!/�H��?:	ퟧ�d@ퟧ�d@!ퟧ�d@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��5M�X@y7@���,�?�"5
sequential/dense/MatMulMatMul���}cʢ?!���}cʢ?0"C
%gradient_tape/sequential/dense/MatMulMatMul���o��?!�Uٱ龱?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul���o��?!�E����?0"7
sequential/dense_1/MatMulMatMul���o��?!���,9�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�H�:�?!1$Fm��?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�М�'�?!L� A�?"7
sequential/dense_2/MatMulMatMul�X-�K��?!ci���?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�f���?!%C�w9!�?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam���o��?!7��t�7�?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad���o��?!�]��
��?Q      Y@Y>����/@aX�i��U@q!]8M�qX@y*�:!G��?"�
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
Refer to the TF2 Profiler FAQb�97.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 