�	}�r���6@}�r���6@!}�r���6@	S����ϑ?S����ϑ?!S����ϑ?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6}�r���6@e�VAl4@1�T�	g��?AJ+��?I�E���@Ym��J�Rp?*	     �K@2U
Iterator::Model::ParallelMapV2�5�;Nё?!��.��?@)�5�;Nё?1��.��?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�������?!�袋.�6@)��_�L�?1颋.��2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate_�Qڋ?!�袋.�8@){�G�z�?1/�袋.2@:Preprocessing2F
Iterator::Model�~j�t��?!�E]t�E@)F%u�{?1      (@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��H�}m?!/�袋.@)��H�}m?1/�袋.@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipX9��v��?!.�袋.L@)�����g?1]t�E@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!�.�袋@)�J�4a?1�.�袋@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�q����?!]t�E]<@)����Mb`?1]t�E@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9S����ϑ?I��s[ٻX@QV_�i��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	e�VAl4@e�VAl4@!e�VAl4@      ��!       "	�T�	g��?�T�	g��?!�T�	g��?*      ��!       2	J+��?J+��?!J+��?:	�E���@�E���@!�E���@B      ��!       J	m��J�Rp?m��J�Rp?!m��J�Rp?R      ��!       Z	m��J�Rp?m��J�Rp?!m��J�Rp?b      ��!       JGPUYS����ϑ?b q��s[ٻX@yV_�i��?�"5
sequential/dense/MatMulMatMulL�L[L��?!L�L[L��?0"C
%gradient_tape/sequential/dense/MatMulMatMul���?rz�?!i��M_��?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul���?rz�?!,�hm�O�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul5�C���?!�}E1��?"7
sequential/dense_1/MatMulMatMul�w��G��?!�8���?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�0��ٕ?!�r.�z�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�0��ٕ?!��$�=5�?"7
sequential/dense_2/MatMulMatMul�0��ٕ?!�>�_��?0"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad���?rz�?!Pȉ���?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdamP��v�4�?!� �~#�?Q      Y@Y>����/@aX�i��U@qp?��-%W@y!���RO�?"�
both�Your program is POTENTIALLY input-bound because 89.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�92.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 