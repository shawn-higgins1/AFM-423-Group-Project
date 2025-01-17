�	[�����7@[�����7@![�����7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-[�����7@{fI��:5@1x�7N
��?A������?IӾ��z�@*	ffffffI@2U
Iterator::Model::ParallelMapV2�q����?!�V��j�>@)�q����?1�V��j�>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat_�Qڋ?!�X,��:@)46<�R�?1�N���t5@:Preprocessing2F
Iterator::ModelǺ���?!��`0F@)_�Q�{?1�X,��*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate/�$��?!k�Z�V�4@)S�!�uq{?1��`*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceŏ1w-!o?!|�^���@)ŏ1w-!o?1|�^���@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�ZӼ��?!}>����K@)Ǻ���f?1��`0@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��_vOf?!Q(
�B@)��_vOf?1Q(
�B@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�~j�t��?!���|>�7@)�~j�t�X?1���|>�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIÛ����X@Q&Y���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	{fI��:5@{fI��:5@!{fI��:5@      ��!       "	x�7N
��?x�7N
��?!x�7N
��?*      ��!       2	������?������?!������?:	Ӿ��z�@Ӿ��z�@!Ӿ��z�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qÛ����X@y&Y���?�"5
sequential/dense/MatMulMatMul��6K���?!��6K���?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul��+<�X�?!�e��܃�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul��+<�X�?! Z��>0�?"7
sequential/dense_1/MatMulMatMul��+<�X�?!<��Pn�?0"C
%gradient_tape/sequential/dense/MatMulMatMulȼ�wkG�?!n��]+��?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch��6K���?!�rQ
v�?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad��6K���?!&O���+�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul��6K���?!�+Z���?"7
sequential/dense_2/MatMulMatMul��+<�X�?!LT��o��?0"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam=�¢��?!`���z��?Q      Y@Y>����/@aX�i��U@q��C�qX@y
Y��e��?"�
both�Your program is POTENTIALLY input-bound because 88.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 