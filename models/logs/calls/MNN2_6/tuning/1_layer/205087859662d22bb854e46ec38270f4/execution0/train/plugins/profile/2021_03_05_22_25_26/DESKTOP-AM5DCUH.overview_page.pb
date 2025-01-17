�	 �ҥ52@ �ҥ52@! �ҥ52@	W�m�є?W�m�є?!W�m�є?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6 �ҥ52@f���0@1��HV�?A��JY�8�?I�q5�+��?Y'���Sn?*	����̌G@2U
Iterator::Model::ParallelMapV2���_vO�?!W��N)l?@)���_vO�?1W��N)l?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatZd;�O��?!���j8@)�j+��݃?1����/�4@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapA��ǘ��?!A���7@)lxz�,C|?1<׬�L-@:Preprocessing2F
Iterator::Model'�����?!�g���F@)F%u�{?1�^���,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�J�4q?!F����!@)�J�4q?1F����!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipp_�Q�?!0�X�IHK@)a��+ei?1���S@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!�O[h��@)��H�}]?1�O[h��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9W�m�є?I̴��$�X@Q��}#�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	f���0@f���0@!f���0@      ��!       "	��HV�?��HV�?!��HV�?*      ��!       2	��JY�8�?��JY�8�?!��JY�8�?:	�q5�+��?�q5�+��?!�q5�+��?B      ��!       J	'���Sn?'���Sn?!'���Sn?R      ��!       Z	'���Sn?'���Sn?!'���Sn?b      ��!       JGPUYW�m�є?b q̴��$�X@y��}#�?�"5
sequential/dense/MatMulMatMul�?�M��?!�?�M��?0"C
%gradient_tape/sequential/dense/MatMulMatMul�!ڨ?!�-��g�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulc*��?!M�)W��?"7
sequential/dense_1/MatMulMatMulc*��?!�om<���?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad6�ɾI�?!�j�2��?"!
Adam/PowPow�!ژ?!y�řs��?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam���ӯ�?!Q(�n��?"
Abs_1Abs ����?!*�q���?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam�D���?!{P���?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdam�D���?!̝�����?Q      Y@Y�M�_{4@a��(�S@q��z`�W@yg֟���?"�
both�Your program is POTENTIALLY input-bound because 92.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�6.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�94.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 