�	Eׅ��5@Eׅ��5@!Eׅ��5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-Eׅ��5@�
�7'3@1<FzQ��?A�P�[��?I`cD�0@*	     �K@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate46<�R�?!<zel��C@)Dio��ɔ?1��B�IB@:Preprocessing2U
Iterator::Model::ParallelMapV2���_vO�?!������:@)���_vO�?1������:@:Preprocessing2F
Iterator::Model��JY�8�?!��~G��C@)lxz�,C|?1���g�(@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatŏ1w-!?!2���+c+@)�����w?1jNq��$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipX�5�;N�?!5'��PsN@)��_vOf?1Z7�"�u@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!#�u�)�	@)��H�}]?1#�u�)�	@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���Mb�?!�2���+E@)_�Q�[?1�5'�@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor-C��6J?!q��$�?)-C��6J?1q��$�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceǺ���F?!�B�I .�?)Ǻ���F?1�B�I .�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�����X@Q�����C�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�
�7'3@�
�7'3@!�
�7'3@      ��!       "	<FzQ��?<FzQ��?!<FzQ��?*      ��!       2	�P�[��?�P�[��?!�P�[��?:	`cD�0@`cD�0@!`cD�0@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�����X@y�����C�?�"5
sequential/dense/MatMulMatMult��(�?!t��(�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�.�+]^�?!�q^�B��?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulH:�v�u�?!���?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam�.�+]^�?!�eu����?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�.�+]^�?!���Xw�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�.�+]^�?!�q^�B��?"E
'gradient_tape/sequential/dense_1/MatMulMatMul�.�+]^�?!^�ң0�?0"7
sequential/dense_1/MatMulMatMul�.�+]^�?!2}GIڻ�?0"]
?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nanDivNoNan����?!�uE�/.�?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdam�Ev����?!+�\��?Q      Y@Y{	�%��1@a�����T@q�����X@yڈ0�q��?"�
both�Your program is POTENTIALLY input-bound because 88.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�99.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 