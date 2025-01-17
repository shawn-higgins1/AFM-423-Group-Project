�	ea5@ea5@!ea5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-ea5@ݗ3��2@1��1�3/�?AHP�s�?I\Ǹ��(�?*	      J@2U
Iterator::Model::ParallelMapV2�������?!�؉��	8@)�������?1�؉��	8@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate2U0*��?!;�;1>@)�HP��?1b'vb'v7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat������?!��N��N6@)�j+��݃?1vb'vb�2@:Preprocessing2F
Iterator::Modeln���?!��؉��B@)�ZӼ�}?1��N��N+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��ǘ���?!vb'vb'O@)y�&1�l?1�N��N�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlicey�&1�l?!�N��N�@)y�&1�l?1�N��N�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!�;�;@)ŏ1w-!_?1�;�;@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMape�X��?!'vb'v�@@)-C��6Z?1ى�؉�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIt氺3�X@Q�b�Ss�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	ݗ3��2@ݗ3��2@!ݗ3��2@      ��!       "	��1�3/�?��1�3/�?!��1�3/�?*      ��!       2	HP�s�?HP�s�?!HP�s�?:	\Ǹ��(�?\Ǹ��(�?!\Ǹ��(�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qt氺3�X@y�b�Ss�?�"5
sequential/dense/MatMulMatMul�Q[#r�?!�Q[#r�?0"C
%gradient_tape/sequential/dense/MatMulMatMul��lN9��?!8��ķ?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch<.��O�?!�����?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul<.��O�?!��/W6�?"7
sequential/dense_1/MatMulMatMul<.��O�?!r�3M`�?0"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam��Ÿ�?�?!Ex'JE��?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad��Ÿ�?�?!0@a=p�?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam�(�_�?!���r7V�?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdam�(�_�?!�C�1<�?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam�(�_�?!����?Q      Y@Y��/Ċ�0@a�	�N]�T@q�����X@y��� ���?"�
both�Your program is POTENTIALLY input-bound because 88.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�99.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 