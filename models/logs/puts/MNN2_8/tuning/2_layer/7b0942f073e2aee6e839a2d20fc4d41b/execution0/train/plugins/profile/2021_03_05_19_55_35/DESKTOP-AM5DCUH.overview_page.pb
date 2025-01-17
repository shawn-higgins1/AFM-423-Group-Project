�	�#��8@�#��8@!�#��8@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�#��8@%t��Y�5@1!�'�>�?A�+e�X�?I�g��Һ@*	�����K@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�� �rh�?!� 2�]?@)%u��?1��f,;@:Preprocessing2U
Iterator::Model::ParallelMapV2��@��ǈ?!I!S6@)��@��ǈ?1I!S6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate-C��6�?!��3���7@)��~j�t�?1����1@:Preprocessing2F
Iterator::Model�l����?!ϋ�� A@)-C��6z?1��3���'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�4�8EG�?!��wP@)/n��r?15��"u< @:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceF%u�k?!O�m��Z@)F%u�k?1O�m��Z@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP�s�b?!�W-0c�@)HP�s�b?1�W-0c�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���QI�?!U�X>b:@)�~j�t�X?1_����#@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIUQ�U�X@QĪ+
�*�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	%t��Y�5@%t��Y�5@!%t��Y�5@      ��!       "	!�'�>�?!�'�>�?!!�'�>�?*      ��!       2	�+e�X�?�+e�X�?!�+e�X�?:	�g��Һ@�g��Һ@!�g��Һ@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qUQ�U�X@yĪ+
�*�?�"5
sequential/dense/MatMulMatMul�(�i�?!�(�i�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�Cdi�?!��5�i�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul�!��<A�?!�	�\G
�?0"7
sequential/dense_1/MatMulMatMul�!��<A�?!Tr�rU�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�s)H�/�?!Bj|�q��?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchQ�٥���?!��7�S�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMulQ�٥���?!�����?"7
sequential/dense_2/MatMulMatMul�lv�PB�?!n��.�?0"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam�!��<A�?!U�K#+�?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad�!��<A�?!s���6?�?Q      Y@Y>����/@aX�i��U@qH���jX@y�|�r8�?"�
both�Your program is POTENTIALLY input-bound because 89.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 