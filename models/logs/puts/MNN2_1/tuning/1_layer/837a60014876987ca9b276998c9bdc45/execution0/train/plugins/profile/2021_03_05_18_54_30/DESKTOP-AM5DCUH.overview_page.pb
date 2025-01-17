�	ގpZ�b5@ގpZ�b5@!ގpZ�b5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-ގpZ�b5@����?�2@1D�+g��?AM�St$�?I���"8@*	hfffffI@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::ConcatenateU���N@�?! �B@)�5�;Nё?1�@ A@:Preprocessing2U
Iterator::Model::ParallelMapV2���_vO�?!H$�D"=@)���_vO�?1H$�D"=@:Preprocessing2F
Iterator::Model�I+��?!�t:�N�E@)��H�}}?1��b�X,@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��0�*x?!O���t:'@)�J�4q?1#�H$� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��H�}�?!��b�XL@)-C��6j?1L&��d2@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor_�Q�[?!�X,��
@)_�Q�[?1�X,��
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���<,�?!y<��cC@)��H�}M?1��b�X�?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor-C��6J?!L&��d2�?)-C��6J?1L&��d2�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicea2U0*�C?!�\.����?)a2U0*�C?1�\.����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��^E��X@Q��P]	�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	����?�2@����?�2@!����?�2@      ��!       "	D�+g��?D�+g��?!D�+g��?*      ��!       2	M�St$�?M�St$�?!M�St$�?:	���"8@���"8@!���"8@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��^E��X@y��P]	�?�"5
sequential/dense/MatMulMatMul�Ym����?!�Ym����?0"C
%gradient_tape/sequential/dense/MatMulMatMul��f�8�?!F�}	|�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul��@��F�?!i���ͻ?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch���ݤ�?!/i+�t��?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam��f�8�?!
��ل��?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad��f�8�?!����	�?"7
sequential/dense_1/MatMulMatMul��f�8�?!�1�3���?0"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdamC"��T�?!�CU�u�?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdamC"��T�?!�v=[�?"!
Adam/PowPowC"��T�?!,����@�?Q      Y@Y{	�%��1@a�����T@q󍒴��X@y��P0J��?"�
both�Your program is POTENTIALLY input-bound because 88.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�99.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 