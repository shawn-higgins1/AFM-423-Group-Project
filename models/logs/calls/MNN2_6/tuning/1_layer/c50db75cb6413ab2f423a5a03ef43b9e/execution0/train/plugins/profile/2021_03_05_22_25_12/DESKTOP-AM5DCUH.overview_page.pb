�	,�F<ٝ1@,�F<ٝ1@!,�F<ٝ1@	�x�ؼ��?�x�ؼ��?!�x�ؼ��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6,�F<ٝ1@S"�^FM0@1e����?A?�ܥ?Iɪ7��?Y�B�=�p?*�����?H@)       =2U
Iterator::Model::ParallelMapV2���QI�?!r���
|=@)���QI�?1r���
|=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat9��v���?!�$2��:@)��ZӼ�?1���5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap46<�R�?!���fy6@)���_vO~?1�
|q��.@:Preprocessing2F
Iterator::Model�g��s��?!��$2�E@)lxz�,C|?1T���t,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlicey�&1�l?!,O"Ӱ�@)y�&1�l?1,O"Ӱ�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�#�����?!2���$L@)_�Q�k?1|q���
@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorǺ���f?!W?��@)Ǻ���f?1W?��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�x�ؼ��?I3�l�X@Q�R�Ӄ��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	S"�^FM0@S"�^FM0@!S"�^FM0@      ��!       "	e����?e����?!e����?*      ��!       2	?�ܥ??�ܥ?!?�ܥ?:	ɪ7��?ɪ7��?!ɪ7��?B      ��!       J	�B�=�p?�B�=�p?!�B�=�p?R      ��!       Z	�B�=�p?�B�=�p?!�B�=�p?b      ��!       JGPUY�x�ؼ��?b q3�l�X@y�R�Ӄ��?�"5
sequential/dense/MatMulMatMulH=�Gom�?!H=�Gom�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�4�ɠ!�?!����G�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul���	��?!Sz/@��?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdamϋ���D�?!�+e�?�?"[
AArithmeticOptimizer/ReorderCastLikeAndValuePreserving_double_CastCastϋ���D�?!Gݚl�g�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGradϋ���D�?!���U��?"7
sequential/dense_1/MatMulMatMulϋ���D�?!;@���?0"
Sum_5Sum�C��3��?!Y��	ˋ�?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam�(B���?!� ��?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdam�(B���?!u9T�>��?Q      Y@Y�M�_{4@a��(�S@qF�GBGW@y:��*ƽ�?"�
both�Your program is POTENTIALLY input-bound because 92.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�6.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�93.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 