�	���2@���2@!���2@	�7��t�?�7��t�?!�7��t�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6���2@Z��լ�0@1Z�{,}�?A�e��a��?I'�O:� �?Y'���Sn?*	�����L@2U
Iterator::Model::ParallelMapV2r�����?!+P�)�~?@)r�����?1+P�)�~?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���Q��?!�)�~��:@)A��ǘ��?1�GYN:�3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��<,Ԋ?!�]�BO7@)46<�R�?1\���e3@:Preprocessing2F
Iterator::Modelݵ�|г�?!i��ٹTF@)���_vO~?1K�!�U*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�q����o?!��#��@)�q����o?1��#��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�U���؟?!�jC&F�K@)�~j�t�h?1�T21Z@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!�(�Q@)/n��b?1�(�Q@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�7��t�?I��^��X@Q�A���E�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	Z��լ�0@Z��լ�0@!Z��լ�0@      ��!       "	Z�{,}�?Z�{,}�?!Z�{,}�?*      ��!       2	�e��a��?�e��a��?!�e��a��?:	'�O:� �?'�O:� �?!'�O:� �?B      ��!       J	'���Sn?'���Sn?!'���Sn?R      ��!       Z	'���Sn?'���Sn?!'���Sn?b      ��!       JGPUY�7��t�?b q��^��X@y�A���E�?�"5
sequential/dense/MatMulMatMul���|�d�?!���|�d�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�XJ�>�?!�����@�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul���|�d�?!�oe!�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�XJ�>�?!к���0�?"E
'gradient_tape/sequential/dense_1/MatMulMatMul�XJ�>�?!���TT�?0"7
sequential/dense_1/MatMulMatMul�XJ�>�?!�PA��w�?0"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam�õL�?!y	�6]��?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam�·�Ւ?!��x�?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdam�·�Ւ?!�b~l��?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam�·�Ւ?!5y�o���?Q      Y@Y�M�_{4@a��(�S@q1�|�tW@y��En�?"�
both�Your program is POTENTIALLY input-bound because 92.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�6.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�93.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 