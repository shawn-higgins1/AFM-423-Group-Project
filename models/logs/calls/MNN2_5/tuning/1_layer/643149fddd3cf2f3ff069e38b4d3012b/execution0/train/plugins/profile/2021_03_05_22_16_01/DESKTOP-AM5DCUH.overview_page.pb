�	F�v$2@F�v$2@!F�v$2@	��<P��?��<P��?!��<P��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6F�v$2@�m��0@1�ID�A�?AV����_�?Ib�� ���?Ya��>��d?*	�����G@2U
Iterator::Model::ParallelMapV2�ZӼ��?!��dw�>@)�ZӼ��?1��dw�>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatA��ǘ��?!|�[��8@)Έ����?11Hv�l,4@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapZd;�O��?!q	���8@)ŏ1w-!?1#���c|0@:Preprocessing2F
Iterator::Model�0�*�?!�μ\�WF@)-C��6z?1����+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�q����o?!�~�t� @)�q����o?1�~�t� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�&1��?!m1C�B�K@)�����g?1���	�)@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!\�*�<@)��H�}]?1\�*�<@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 91.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�7.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��<P��?I�^Z8��X@QB�[��I�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�m��0@�m��0@!�m��0@      ��!       "	�ID�A�?�ID�A�?!�ID�A�?*      ��!       2	V����_�?V����_�?!V����_�?:	b�� ���?b�� ���?!b�� ���?B      ��!       J	a��>��d?a��>��d?!a��>��d?R      ��!       Z	a��>��d?a��>��d?!a��>��d?b      ��!       JGPUY��<P��?b q�^Z8��X@yB�[��I�?�"5
sequential/dense/MatMulMatMul�_04��?!�_04��?0"C
%gradient_tape/sequential/dense/MatMulMatMul�8��M��?!��[X�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulY�Sr��?!�V�Ju��?"7
sequential/dense_1/MatMulMatMulY�Sr��?!�G$g��?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�8��M��?!�n�k��?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdamY��|�?!U�-g��?"
Abs_1Absk��":M�?!�K��po�?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdamk��":M�?!xT���l�?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdamk��":M�?!�_��?"!
Adam/PowPowk��":M�?!ƱI<3��?Q      Y@Y�M�_{4@a��(�S@q�b�)w�W@yd=�( F�?"�
both�Your program is POTENTIALLY input-bound because 91.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�7.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�94.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 