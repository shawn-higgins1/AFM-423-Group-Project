�	�'��Z7@�'��Z7@!�'��Z7@	�O��0B�?�O��0B�?!�O��0B�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�'��Z7@@��r�n4@1B�V�9��?A�����y�?I0�Б@Y���#ӡ�?*	    �H@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate��d�`T�?!�X�CB@)?�ܵ�|�?1۶m۶m@@:Preprocessing2U
Iterator::Model::ParallelMapV2�ZӼ��?!>4և��<@)�ZӼ��?1>4և��<@:Preprocessing2F
Iterator::Model��_vO�?!��S�r
F@)���_vO~?1����>4.@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat �o_�y?!m۶m۶)@)	�^)�p?1/���� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�?�߾�?!~h���K@)a2U0*�c?1�����@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!}h���@)/n��b?1}h���@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor/n��R?!}h���@)/n��R?1}h���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���&�?!/���C@)-C��6J?1����X�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceǺ���F?!�m۶m��?)Ǻ���F?1�m۶m��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�O��0B�?I�i$]}�X@Q���b8�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	@��r�n4@@��r�n4@!@��r�n4@      ��!       "	B�V�9��?B�V�9��?!B�V�9��?*      ��!       2	�����y�?�����y�?!�����y�?:	0�Б@0�Б@!0�Б@B      ��!       J	���#ӡ�?���#ӡ�?!���#ӡ�?R      ��!       Z	���#ӡ�?���#ӡ�?!���#ӡ�?b      ��!       JGPUY�O��0B�?b q�i$]}�X@y���b8�?�"5
sequential/dense/MatMulMatMulC��<`��?!C��<`��?0"C
%gradient_tape/sequential/dense/MatMulMatMulʣ�9�?!�9�L�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMulʣ�9�?!�o	8�_�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�^��#��?!ʣ�9��?"7
sequential/dense_1/MatMulMatMul�^��#��?!��:�}��?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch��`��?!6�;��5�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul��`��?!�r<o���?"7
sequential/dense_2/MatMulMatMul��`��?!fd=[|�?0"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGradtM�����?!����?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdamʣ�9�?!G����?Q      Y@Y�C=�C=0@a��
��T@q�-�^��U@y����/��?"�
both�Your program is POTENTIALLY input-bound because 88.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�87.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 