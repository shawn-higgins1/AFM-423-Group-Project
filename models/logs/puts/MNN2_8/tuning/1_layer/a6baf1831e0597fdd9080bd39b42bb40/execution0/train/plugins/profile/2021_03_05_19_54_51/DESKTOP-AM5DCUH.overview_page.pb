�	T;�Ԗ�5@T;�Ԗ�5@!T;�Ԗ�5@		2?N���?	2?N���?!	2?N���?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6T;�Ԗ�5@N�&�OB3@1�E�����?Ag��j+��?IK�� @YW��m�?*23333sG@)       =2U
Iterator::Model::ParallelMapV2�{�Pk�?!-��V]�;@)�{�Pk�?1-��V]�;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�{�Pk�?!-��V]�;@)A��ǘ��?1J��>٪7@:Preprocessing2F
Iterator::ModelM�O��?!Z�4��E@)�<,Ԛ�}?1�?�K!/@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateM�O��?!Z�4��5@) �o_�y?12����*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceŏ1w-!o?!�SK�f4 @)ŏ1w-!o?1�SK�f4 @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipz6�>W�?!��\�vL@)��_vOf?1O3�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!�"� �@)��H�}]?1�"� �@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapA��ǘ��?!J��>٪7@)����MbP?1}/�ܼ@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9	2?N���?I��@�I�X@Q��L�%�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	N�&�OB3@N�&�OB3@!N�&�OB3@      ��!       "	�E�����?�E�����?!�E�����?*      ��!       2	g��j+��?g��j+��?!g��j+��?:	K�� @K�� @!K�� @B      ��!       J	W��m�?W��m�?!W��m�?R      ��!       Z	W��m�?W��m�?!W��m�?b      ��!       JGPUY	2?N���?b q��@�I�X@y��L�%�?�"5
sequential/dense/MatMulMatMul��g�x�?!��g�x�?0"C
%gradient_tape/sequential/dense/MatMulMatMulqH��k��?!��O�9�?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad��N����?!�C�b�v�?"7
sequential/dense_1/MatMulMatMul�%$�0�?!�&G{��?0"!
Adam/PowPowqH��k��?!��jĨ@�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchqH��k��?!�x�Aֿ�?"E
'gradient_tape/sequential/dense_1/MatMulMatMulqH��k��?!�!�?�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulqH��k��?!��h<1��?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam����!��?!�IhZ���?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam����!��?!R�gx�|�?Q      Y@Y��/Ċ�0@a�	�N]�T@qD�F�V@y1A���5�?"�
both�Your program is POTENTIALLY input-bound because 89.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�88.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 