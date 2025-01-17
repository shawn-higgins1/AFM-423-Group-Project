�	���-�v7@���-�v7@!���-�v7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-���-�v7@"3�<5@1D� ��?AEGr��?I�7�k�'@*	     �G@2U
Iterator::Model::ParallelMapV2S�!�uq�?!cr1��<@)S�!�uq�?1cr1��<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�+e�X�?!1���\A8@)�&S��?11���\3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateZd;�O��?!ڨ�l�w8@)� �	�?1���
b0@:Preprocessing2F
Iterator::Model�j+��ݓ?!��F}g�D@)�~j�t�x?1&W�+�)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceŏ1w-!o?!!&W�+ @)ŏ1w-!o?1!&W�+ @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Ziplxz�,C�?!1���\M@)y�&1�l?1W�+��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP�s�b?!� &W�@)HP�s�b?1� &W�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�
F%u�?!��
br;@)a2U0*�S?1Q�٨�l@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�din�X@QȚ�e��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	"3�<5@"3�<5@!"3�<5@      ��!       "	D� ��?D� ��?!D� ��?*      ��!       2	EGr��?EGr��?!EGr��?:	�7�k�'@�7�k�'@!�7�k�'@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�din�X@yȚ�e��?�"5
sequential/dense/MatMulMatMul��ۘh�?!��ۘh�?0"C
%gradient_tape/sequential/dense/MatMulMatMul/��Ѧ�?!�HDלٲ?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul�@5ݕ�?!@����P�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�@5ݕ�?!��<Z��?"7
sequential/dense_1/MatMulMatMul�@5ݕ�?!��Ѿ�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchǐ�T;*�?!	g<��?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGradǐ�T;*�?!"���Mj�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMulǐ�T;*�?!;;��?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam�@5ݕ�?!��p��?"!
Adam/PowPow�@5ݕ�?!�-DD�%�?Q      Y@Y>����/@aX�i��U@q���2�~X@y�4�)�t�?"�
both�Your program is POTENTIALLY input-bound because 89.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�98.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 