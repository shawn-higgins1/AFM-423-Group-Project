�	d*��G	@d*��G	@!d*��G	@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-d*��G	@ܜJ���?1��%P�?A��q���?I{�\�&� @*	     @K@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate
ףp=
�?!G*�Vg�D@)Q�|a2�?1�O����B@:Preprocessing2U
Iterator::Model::ParallelMapV2�?�߾�?!4R1�:#9@)�?�߾�?14R1�:#9@:Preprocessing2F
Iterator::ModelA��ǘ��?!߅���]D@)�� �rh�?1s�3R1/@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�����w?!�Ṱ�H%@)����Mbp?1����[@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipaTR'���?!!z|�M@)�����g?1�Ṱ�H@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!���d	l
@)��H�}]?1���d	l
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��0�*�?!l�O���E@)/n��R?1Z���% @:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor/n��R?!Z���% @)/n��R?1Z���% @:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceǺ���F?!�H����?)Ǻ���F?1�H����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 27.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�65.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI5���_W@Q�,g O@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	ܜJ���?ܜJ���?!ܜJ���?      ��!       "	��%P�?��%P�?!��%P�?*      ��!       2	��q���?��q���?!��q���?:	{�\�&� @{�\�&� @!{�\�&� @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q5���_W@y�,g O@�"5
sequential/dense/MatMulMatMul�� �%��?!�� �%��?0"C
%gradient_tape/sequential/dense/MatMulMatMulRR��=g�?!w�rǱ��?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�fc��?!1�9 ��?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�fc��?!u��<7�?"7
sequential/dense_1/MatMulMatMul�fc��?!RR��=g�?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�Ҵ�ߘ?!���%!��?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdamRR��=g�?!�v���?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdamRR��=g�?!9���?"E
'gradient_tape/sequential/dense_1/MatMulMatMul��mo>�?!oR���$�?0"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam{{V�ܚ�?!��-6��?Q      Y@Y{	�%��1@a�����T@q�61,X@y���P��?"�
both�Your program is POTENTIALLY input-bound because 27.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�65.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�96.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 