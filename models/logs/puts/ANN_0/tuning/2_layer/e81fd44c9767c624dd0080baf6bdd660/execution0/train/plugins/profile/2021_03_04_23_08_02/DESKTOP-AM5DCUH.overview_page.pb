�	��v3@��v3@!��v3@	�ǬM���?�ǬM���?!�ǬM���?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6��v3@w0b� �1@1�W�2�?A�[v��?I�bFx��?Y����>�?*	������H@2U
Iterator::Model::ParallelMapV2��H�}�?!D�JԮD=@)��H�}�?1D�JԮD=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeattF��_�?!0����/8@)��~j�t�?1On��O3@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap-C��6�?!�B�):@)�� �rh�?1G:l��F1@:Preprocessing2F
Iterator::Model䃞ͪϕ?!%jW�v�E@)lxz�,C|?1���|,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice"��u��q?!{����z!@)"��u��q?1{����z!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip6�;Nё�?!ڕ�]�ZL@)-C��6j?1�B�)@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*�c?!��18�@)a2U0*�c?1��18�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 91.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�7.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�ǬM���?IN��r��X@Qo��&��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	w0b� �1@w0b� �1@!w0b� �1@      ��!       "	�W�2�?�W�2�?!�W�2�?*      ��!       2	�[v��?�[v��?!�[v��?:	�bFx��?�bFx��?!�bFx��?B      ��!       J	����>�?����>�?!����>�?R      ��!       Z	����>�?����>�?!����>�?b      ��!       JGPUY�ǬM���?b qN��r��X@yo��&��?�"5
sequential/dense/MatMulMatMul��Uw��?!��Uw��?0"C
%gradient_tape/sequential/dense/MatMulMatMul�h��M1�?!�/�{bt�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul�h��M1�?!d�L	��?0"7
sequential/dense_1/MatMulMatMul�h��M1�?!>L����?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul��h�#��?!4�
�=�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul4�
�=�?!��C+Ue�?"7
sequential/dense_2/MatMulMatMul4�
�=�?! d�L	��?0"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam�h��M1�?!��[��	�?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad�h��M1�?!�uZ�L�?"S
2gradient_tape/sequential/dense/BiasAdd/BiasAddGradBiasAddGrad$0Y��?! R�
=��?Q      Y@Y7��Moz2@a���,daT@q���+V@y d�L	��?"�
both�Your program is POTENTIALLY input-bound because 91.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�7.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�88.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 