�	8�q��?P@8�q��?P@!8�q��?P@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-8�q��?P@��� \O@1�`R||B�?A��MbX�?I�4f2 @*	ffffffL@2U
Iterator::Model::ParallelMapV2�:pΈ�?!�K=��?@)�:pΈ�?1�K=��?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�]K�=�?!L=�]j7@)M�St$�?1���.��3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateV-��?!�@�6�9@)M�O��?1�V��1@:Preprocessing2F
Iterator::Model�HP��?!SO�o�zE@) �o_�y?1ܥ���.&@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice/n��r?!��.�d�@)/n��r?1��.�d�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��ǘ���?!��9�h�L@)-C��6j?1!�
��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!i�́D+@)����Mb`?1i�́D+@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���H�?!�o�z2�;@)Ǻ���V?1�v�'�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 96.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�3.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��"���X@QgW�83�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��� \O@��� \O@!��� \O@      ��!       "	�`R||B�?�`R||B�?!�`R||B�?*      ��!       2	��MbX�?��MbX�?!��MbX�?:	�4f2 @�4f2 @!�4f2 @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��"���X@ygW�83�?�"5
sequential/dense/MatMulMatMul*��� �?!*��� �?0"C
%gradient_tape/sequential/dense/MatMulMatMulO�q�?!�SkD��?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulfFТJ��?!>#��?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad3ĹM���?!%ʺ��&�?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdamO�q�?!�j�=��?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchO�q�?!m~�C�?"E
'gradient_tape/sequential/dense_1/MatMulMatMulO�q�?!�_�V��?0"7
sequential/dense_1/MatMulMatMulO�q�?!�LA;�_�?0"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam�������?!0��:=J�?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam�������?!��:�4�?Q      Y@Y��/Ċ�0@a�	�N]�T@q
#U�׆X@y�W�K���?"�
both�Your program is POTENTIALLY input-bound because 96.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�3.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�98.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 