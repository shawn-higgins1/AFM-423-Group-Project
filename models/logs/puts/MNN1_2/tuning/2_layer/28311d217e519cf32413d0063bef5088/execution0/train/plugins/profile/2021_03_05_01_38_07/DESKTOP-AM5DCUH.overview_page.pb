�	��r0��7@��r0��7@!��r0��7@	�HU�ц?�HU�ц?!�HU�ц?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6��r0��7@%!���5@1Ve����?A$����ۧ?I�7kp@Y���e?*	�����LK@2U
Iterator::Model::ParallelMapV2a2U0*��?!R�Q�A@)a2U0*��?1R�Q�A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��<,Ԋ?!����7@)46<�R�?1j��i��3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateǺ����?!4H�4H�4@)� �	�?1H�4H�4,@:Preprocessing2F
Iterator::ModelF%u��?!�-�-H@)��H�}}?1��_��_*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey�&1�l?!A�A�@)y�&1�l?1A�A�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�!��u��?!1��/��I@)��_�Le?1�0�0@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!��@)/n��b?1��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�������?!M��L��6@)��_�LU?1�0�0@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�HU�ц?Ii�2nu�X@Q.{"� 5�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	%!���5@%!���5@!%!���5@      ��!       "	Ve����?Ve����?!Ve����?*      ��!       2	$����ۧ?$����ۧ?!$����ۧ?:	�7kp@�7kp@!�7kp@B      ��!       J	���e?���e?!���e?R      ��!       Z	���e?���e?!���e?b      ��!       JGPUY�HU�ц?b qi�2nu�X@y.{"� 5�?�"5
sequential/dense/MatMulMatMul��^��{�?!��^��{�?0"C
%gradient_tape/sequential/dense/MatMulMatMul44��Q�?!w�S�f�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul44��Q�?!��uŔ�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul44��Q�?!�e�4��?"7
sequential/dense_1/MatMulMatMul44��Q�?!��՝0�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�f�%��?!��*����?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMulA�����?!���D��?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam44��Q�?!Fs�|y��?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad44��Q�?!�9Y���?"7
sequential/dense_2/MatMulMatMul44��Q�?!) ښq�?0Q      Y@Y>����/@aX�i��U@q��-�1�W@y��uŔ�?"�
both�Your program is POTENTIALLY input-bound because 88.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�94.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 