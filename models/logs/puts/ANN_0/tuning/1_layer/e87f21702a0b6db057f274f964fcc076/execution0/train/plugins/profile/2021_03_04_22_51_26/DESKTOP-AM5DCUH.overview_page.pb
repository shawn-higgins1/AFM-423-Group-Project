�	�s��2@�s��2@!�s��2@	L �G^ʂ?L �G^ʂ?!L �G^ʂ?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�s��2@����s1@1�Tm7�7�?A(���%V�?I�+��y��?Y>?�m\?*	������I@2U
Iterator::Model::ParallelMapV2V-��?!     R<@)V-��?1     R<@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�<,Ԛ�?!     �<@)�I+��?1     |5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatZd;�O��?!     v6@)��~j�t�?1     �2@:Preprocessing2F
Iterator::Model46<�R�?!     JE@)�<,Ԛ�}?1     �,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice��H�}m?!      @)��H�}m?1      @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip%u��?!     �L@)F%u�k?1     �@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!     @@)����Mb`?1     @@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9L �G^ʂ?It5ʉO�X@Q�mơ�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	����s1@����s1@!����s1@      ��!       "	�Tm7�7�?�Tm7�7�?!�Tm7�7�?*      ��!       2	(���%V�?(���%V�?!(���%V�?:	�+��y��?�+��y��?!�+��y��?B      ��!       J	>?�m\?>?�m\?!>?�m\?R      ��!       Z	>?�m\?>?�m\?!>?�m\?b      ��!       JGPUYL �G^ʂ?b qt5ʉO�X@y�mơ�?�"5
sequential/dense/MatMulMatMulF��HEK�?!F��HEK�?0"C
%gradient_tape/sequential/dense/MatMulMatMulv�Ԍ&�?!ޯ{鸺?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul!<~�4o�?!s�-[J�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGradk�8V'�?!@�t�Eo�?"7
sequential/dense_1/MatMulMatMul�����%�?!�4N���?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul��᎘?!��7pڥ�?0"
Abs_1Abs��~�Rܒ?!���d�?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam��~�Rܒ?!'�W�\�?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdam��~�Rܒ?!��s�<��?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam��~�Rܒ?!��;�
�?Q      Y@Y�M�_{4@a��(�S@q勉J
oW@y��ـ��?"�
both�Your program is POTENTIALLY input-bound because 92.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�6.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�93.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 