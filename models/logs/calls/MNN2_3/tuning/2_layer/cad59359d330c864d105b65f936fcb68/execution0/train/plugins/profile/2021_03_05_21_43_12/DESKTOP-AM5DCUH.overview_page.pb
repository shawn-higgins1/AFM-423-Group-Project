�	&��	8@&��	8@!&��	8@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-&��	8@�Z(���5@16rݔ�Z�?A0*��D�?I����]� @*	    �K@2U
Iterator::Model::ParallelMapV2���S㥋?!�.�袋8@)���S㥋?1�.�袋8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��H�}�?!/�袋.:@)�(��0�?1]t�E]6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�!��u��?!��.��9@)'�����?1t�E]t3@:Preprocessing2F
Iterator::ModelM�O��?!]t�E]B@)S�!�uq{?1]t�E](@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�5�;Nѡ?!��.��O@)U���N@s?1]t�E!@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice_�Q�k?!�袋.�@)_�Q�k?1�袋.�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!�.�袋@)�J�4a?1�.�袋@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap2U0*��?!�.�袋<@)-C��6Z?1F]t�E@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 90.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI͋�ȾX@Q����M�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�Z(���5@�Z(���5@!�Z(���5@      ��!       "	6rݔ�Z�?6rݔ�Z�?!6rݔ�Z�?*      ��!       2	0*��D�?0*��D�?!0*��D�?:	����]� @����]� @!����]� @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q͋�ȾX@y����M�?�"5
sequential/dense/MatMulMatMul���1C�?!���1C�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�H�?!��/�<1�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul�H�?!s����?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�H�?!�O[[B��?"7
sequential/dense_1/MatMulMatMul�H�?!��a��?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�M��t�?!F���B�?"7
sequential/dense_2/MatMulMatMul��(!g�?!��>&��?0"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGradǇ꿢��?!��ZL�?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam�H�?!_� �!8�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�H�?!�h�NJ�?Q      Y@Y>����/@aX�i��U@q��$=�sX@ys����?"�
both�Your program is POTENTIALLY input-bound because 90.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�8.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 