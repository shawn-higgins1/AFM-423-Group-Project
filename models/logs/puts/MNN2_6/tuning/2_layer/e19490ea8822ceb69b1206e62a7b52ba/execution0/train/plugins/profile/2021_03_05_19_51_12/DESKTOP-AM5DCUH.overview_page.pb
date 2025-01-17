�	�GS=��8@�GS=��8@!�GS=��8@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�GS=��8@�R)v5@1ɓ�k&��?A��|?5^�?I0�Qd��	@*	�����F@2U
Iterator::Model::ParallelMapV2������?!S�>R�>:@)������?1S�>R�>:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat-C��6�?!����<@)/�$��?1�r��r�7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�j+��݃?!|�{�5@)a��+ey?1����,@:Preprocessing2F
Iterator::Model�Q���?!��ː��C@)��0�*x?1�c��c�*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey�&1�l?!瘬瘬@)y�&1�l?1瘬瘬@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipz6�>W�?!o 4o 4N@)a��+ei?1����@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP�s�b?!��м��@)HP�s�b?1��м��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��_vO�?! 4o 4o8@)/n��R?1$��#��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 85.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�13.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI6��"'�X@Q��nl�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�R)v5@�R)v5@!�R)v5@      ��!       "	ɓ�k&��?ɓ�k&��?!ɓ�k&��?*      ��!       2	��|?5^�?��|?5^�?!��|?5^�?:	0�Qd��	@0�Qd��	@!0�Qd��	@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q6��"'�X@y��nl�?�"5
sequential/dense/MatMulMatMul&��o��?!&��o��?0"C
%gradient_tape/sequential/dense/MatMulMatMul�H[�c�?!ܕ�p0z�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul�H[�c�?!&��),�?0"7
sequential/dense_1/MatMulMatMul�H[�c�?!8U"���?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul$�_�n�?!�����?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�����?!�\]h�t�?"7
sequential/dense_2/MatMulMatMul�H�D�?!��0l�?0"!
Adam/PowPow�H[�c�?!��W\�B�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�H[�c�?!0���ho�?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad�H[�c�?!�ƔY���?Q      Y@Y>����/@aX�i��U@q��1�X@yx��a�?"�
both�Your program is POTENTIALLY input-bound because 85.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�13.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�98.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 