�	���'�^3@���'�^3@!���'�^3@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-���'�^3@YvQ��1@1��ᱟ��?A�I*S�A�?Ii�
���?*	�����LH@2U
Iterator::Model::ParallelMapV2�{�Pk�?!�n��:@)�{�Pk�?1�n��:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�������?!�3�\�9@)/�$��?1X�<ݚ5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�{�Pk�?!�n��:@)�&S��?1	(>gj�2@:Preprocessing2F
Iterator::Model+�����?!R�n�D@)F%u�{?1<ݚ)+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceŏ1w-!o?!�����F@)ŏ1w-!o?1�����F@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip\ A�c̝?!��a�2�M@)���_vOn?1ن��s@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!����u@)����Mb`?1����u@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�_��X@Q�K?��F�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	YvQ��1@YvQ��1@!YvQ��1@      ��!       "	��ᱟ��?��ᱟ��?!��ᱟ��?*      ��!       2	�I*S�A�?�I*S�A�?!�I*S�A�?:	i�
���?i�
���?!i�
���?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�_��X@y�K?��F�?�"5
sequential/dense/MatMulMatMul�1(n�?!�1(n�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul���P�?!r�噶��?0"C
%gradient_tape/sequential/dense/MatMulMatMul�' X��?!���bGK�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�' X��?!��y�N�?"7
sequential/dense_1/MatMulMatMul�' X��?!��
�R�?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�-I]�?!.�08kt�?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad�1(n�?!���bGK�?"7
sequential/dense_2/MatMulMatMul�1(n�?!�S�)Y��?0"E
'gradient_tape/sequential/dense_2/MatMulMatMul��o���?!ORTy��?0"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdam�;0���?!.�u�V�?Q      Y@Y7��Moz2@a���,daT@qEuxEuxX@yѮE�
��?"�
both�Your program is POTENTIALLY input-bound because 92.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�6.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 