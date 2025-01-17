�	 �	F_5@ �	F_5@! �	F_5@	,�M�:�?,�M�:�?!,�M�:�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6 �	F_5@��8�2@1����[�?A�C��<��?I�j����@Y�]M��j?*	gffff&J@2U
Iterator::Model::ParallelMapV2�J�4�?!��٩@@)�J�4�?1��٩@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��@��ǈ?!|�I��"7@)U���N@�?1I!��%�1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateg��j+��?!2���^6@)	�^)ˀ?1�㐈[/@:Preprocessing2F
Iterator::ModeltF��_�?!�{�I��F@)y�&1�|?1�]���*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey�&1�l?!�]���@)y�&1�l?1�]���@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�v��/�?!*��$?K@)_�Q�k?1���� @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��_vOf?!� W�l�@)��_vOf?1� W�l�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap9��v���?!��{�I�8@)��_�LU?1�E����@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9,�M�:�?I���q�X@Q���- �?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��8�2@��8�2@!��8�2@      ��!       "	����[�?����[�?!����[�?*      ��!       2	�C��<��?�C��<��?!�C��<��?:	�j����@�j����@!�j����@B      ��!       J	�]M��j?�]M��j?!�]M��j?R      ��!       Z	�]M��j?�]M��j?!�]M��j?b      ��!       JGPUY,�M�:�?b q���q�X@y���- �?�"5
sequential/dense/MatMulMatMul��mc�?!��mc�?0"C
%gradient_tape/sequential/dense/MatMulMatMulk��̟�?!V!ڴ?0"7
sequential/dense_1/MatMulMatMul� @?�o�?!9&�6�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch��

���?!Wg�y�+�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad��

���?!�{�<�?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul��

���?!�W|�M�?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdamk��̟�?!-�����?"!
Adam/PowPowk��̟�?!�s���5�?"E
'gradient_tape/sequential/dense_1/MatMulMatMulk��̟�?!� 'Lש�?0"
Abs_1Abs� @?�o�?!}�@i@�?Q      Y@Y��/Ċ�0@a�	�N]�T@qE5��O�W@ytj	D��?"�
both�Your program is POTENTIALLY input-bound because 88.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�94.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 